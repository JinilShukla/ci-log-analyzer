"""
retriever.py

Stage 4 (RAG): Given the current failure's signal lines, find the top-N
most similar past failures from history/failures.json using local embeddings.

Why local embeddings (sentence-transformers)?
  - Free, no API key, no quota limits
  - Runs offline — no data leaves your machine
  - 'all-MiniLM-L6-v2' is small (80MB), fast, and accurate enough for
    log similarity matching

Flow:
  1. Load history/failures.json
  2. Embed each past failure's signal_lines (cached — only recomputed when
     history changes)
  3. Embed the current signal lines
  4. Compute cosine similarity between current and all past failures
  5. Return top-N matches above a minimum similarity threshold

Usage:
  from retriever import retrieve_similar
  matches = retrieve_similar(signal_lines, top_n=3)
"""

import os
import json
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from schema import FailureRecord

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HISTORY_PATH    = os.path.join(os.path.dirname(__file__), "history", "failures.json")
CACHE_PATH      = os.path.join(os.path.dirname(__file__), "history", "embeddings_cache.npz")
MODEL_NAME      = "all-MiniLM-L6-v2"   # 80MB, fast, good semantic similarity
MIN_SIMILARITY  = 0.45                  # below this score we don't report a match

# Lazy-loaded — model only downloads on first use
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  Loading embedding model ({MODEL_NAME})...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------

def load_history() -> list[dict]:
    """
    Load RAG-eligible FailureRecords from history/failures.json.
    REJECTED records are excluded — their fixes were wrong and should not
    pollute future retrievals.
    """
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = [FailureRecord.from_dict(r) for r in raw]
    eligible = [r for r in records if r.is_rag_eligible()]
    return [r.to_dict() for r in eligible]


def save_to_history(entry: dict) -> None:
    """
    Upsert a FailureRecord into history/failures.json.
    Matches on record `id`. If found, replaces in-place; otherwise appends.
    Called by analyzer.py after every run, and by feedback.py on corrections.
    """
    from feedback import load_all_records, save_all_records, _invalidate_cache
    records = load_all_records()
    replaced = False
    new_record = FailureRecord.from_dict(entry)
    for i, r in enumerate(records):
        if r.id == new_record.id:
            records[i] = new_record
            replaced = True
            break
    if not replaced:
        records.append(new_record)
    save_all_records(records)
    _invalidate_cache()


# ---------------------------------------------------------------------------
# Embedding + caching
# ---------------------------------------------------------------------------

def _signal_lines_to_text(signal_lines: list[str]) -> str:
    """
    Join signal lines into a single string for embedding.
    We join with newlines so the model sees them as a coherent block.
    """
    return "\n".join(signal_lines)


def _history_hash(history: list[dict]) -> str:
    """Produce a hash of the history content to detect changes."""
    content = json.dumps([h["id"] for h in history], sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def _load_or_build_embeddings(history: list[dict]) -> np.ndarray:
    """
    Load cached embeddings if history hasn't changed, otherwise recompute.
    Embeddings are stored as a .npz file alongside the history JSON.

    Shape: (len(history), embedding_dim)
    """
    current_hash = _history_hash(history)

    # Try loading cache
    if os.path.exists(CACHE_PATH):
        cached = np.load(CACHE_PATH, allow_pickle=True)
        if str(cached.get("hash", "")) == current_hash:
            return cached["embeddings"]

    # Build embeddings
    model = _get_model()
    texts = [_signal_lines_to_text(h["signal_lines"]) for h in history]
    print(f"  Building embeddings for {len(texts)} historical failure(s)...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # Save cache
    np.savez(CACHE_PATH, embeddings=embeddings, hash=np.array(current_hash))
    return embeddings


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two normalized vectors.
    Since sentence-transformers normalizes embeddings, this is just a dot product.
    """
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve_similar(signal_lines: list[str], top_n: int = 3) -> list[dict]:
    """
    Find the top-N most similar past failures to the given signal lines.

    Returns a list of dicts (up to top_n), each with:
      - id, repo, run_id, job_name, branch
      - signal_lines
      - analysis  (the stored A–E result)
      - similarity  (cosine similarity score, 0.0–1.0)

    Only returns matches above MIN_SIMILARITY. May return fewer than top_n.
    """
    history = load_history()
    if not history:
        return []

    # Embed current signal lines
    model = _get_model()
    current_text = _signal_lines_to_text(signal_lines)
    current_embedding = model.encode(current_text, normalize_embeddings=True)

    # Load/build historical embeddings
    history_embeddings = _load_or_build_embeddings(history)

    # Score all past failures
    scores = [
        _cosine_similarity(current_embedding, history_embeddings[i])
        for i in range(len(history))
    ]

    # Sort by score descending, filter by threshold
    ranked = sorted(
        zip(scores, history),
        key=lambda x: x[0],
        reverse=True
    )

    results = []
    for score, record in ranked[:top_n]:
        if score < MIN_SIMILARITY:
            break
        results.append({**record, "similarity": round(score, 4)})

    return results


def format_for_prompt(matches: list[dict]) -> str:
    """
    Format retrieved similar failures as a readable block for the LLM prompt.
    Uses the effective (corrected-if-available) fix and root cause so that
    developer corrections automatically improve future suggestions.
    """
    if not matches:
        return ""

    lines = ["## SIMILAR PAST FAILURES (for context)\n"]
    for i, match in enumerate(matches, 1):
        sim_pct = round(match["similarity"] * 100, 1)
        record  = FailureRecord.from_dict(match)
        status_tag = (
            " ✅ developer-verified" if match.get("feedback_status") in ("approved", "corrected")
            else ""
        )
        lines.append(
            f"### Past Failure #{i} — {match['repo']} "
            f"(similarity: {sim_pct}%{status_tag})\n"
            f"Signal lines:\n" +
            "\n".join(f"  {l}" for l in match["signal_lines"]) +
            f"\n\nRoot cause: {record.effective_root_cause()}\n"
            f"Fix:\n" +
            "\n".join(f"  {j+1}. {s}" for j, s in enumerate(record.effective_fix())) +
            "\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test with signal lines from our real failing-job
    test_signal_lines = [
        "python -c \"import non_existent_module\"",
        "Traceback (most recent call last):",
        "  File \"<string>\", line 1, in <module>",
        "ModuleNotFoundError: No module named 'non_existent_module'",
        "##[error]Process completed with exit code 1.",
    ]

    print("\nSearching for similar past failures...\n")
    matches = retrieve_similar(test_signal_lines, top_n=3)

    if not matches:
        print("No similar failures found above similarity threshold.")
    else:
        print(f"Found {len(matches)} similar failure(s):\n")
        for m in matches:
            print(f"  [{m['similarity']*100:.1f}%] {m['repo']} / {m['job_name']} (run {m['run_id']})")
            print(f"    → {m['analysis']['root_cause_summary']}\n")

    print("\n--- FORMATTED FOR PROMPT ---")
    print(format_for_prompt(matches))
