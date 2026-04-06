"""
feedback.py

Stage 5: Developer feedback loop.

Responsibilities:
  1. Load a FailureRecord by ID from history/failures.json
  2. Accept developer feedback:  👍 approve  |  👎 reject  |  ✏️  correct
  3. Write the updated record back (status, corrected_fix, corrected_root_cause)
  4. Invalidate the RAG embedding cache so the next retrieval sees the correction
  5. Provide a summary of feedback stats (how many approved / rejected / corrected)

Storage contract:
  - All records live in history/failures.json as FailureRecord dicts
  - One record per (repo, run_id, job_name) — updated in-place on feedback
  - The schema is DB-migration-ready: see FailureRecord in schema.py for
    the SQL-equivalent column layout

Usage (CLI):
  # Interactive — prompts you for thumbs up/down and optional correction
  python3 feedback.py --id fail-abc12345

  # Non-interactive (for webhooks / automated pipelines)
  python3 feedback.py --id fail-abc12345 --status approved --by octocat
  python3 feedback.py --id fail-abc12345 --status corrected --by octocat \
      --fix "Pin numpy to 1.24.x" "Re-run pip install after pin"
  python3 feedback.py --id fail-abc12345 --status rejected  --by octocat

  # Show feedback stats
  python3 feedback.py --stats

Usage (API / programmatic):
  from feedback import record_feedback, load_all_records, get_stats
  record_feedback(
      record_id="fail-abc12345",
      status="corrected",
      by="octocat",
      corrected_fix=["Pin numpy to 1.24.x", "Re-run pip install after pin"],
      corrected_root_cause="numpy 2.0 is incompatible with the current code"
  )
"""

import os
import json
import argparse
from datetime import datetime, timezone
from typing import Optional

from schema import FailureRecord, FeedbackStatus

# ---------------------------------------------------------------------------
# Storage path
# ---------------------------------------------------------------------------

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "history", "failures.json")
CACHE_PATH   = os.path.join(os.path.dirname(__file__), "history", "embeddings_cache.npz")


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def load_all_records() -> list[FailureRecord]:
    """Load all FailureRecords from history/failures.json."""
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [FailureRecord.from_dict(r) for r in raw]


def save_all_records(records: list[FailureRecord]) -> None:
    """Overwrite history/failures.json with the given records."""
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)


def find_record(record_id: str) -> Optional[FailureRecord]:
    """Return the record with the given ID, or None."""
    for r in load_all_records():
        if r.id == record_id:
            return r
    return None


def upsert_record(updated: FailureRecord) -> None:
    """
    Write an updated FailureRecord back to history.
    If a record with the same ID already exists it is replaced in-place.
    If not, the record is appended.
    After saving, invalidate the embedding cache.
    """
    records = load_all_records()
    replaced = False
    for i, r in enumerate(records):
        if r.id == updated.id:
            records[i] = updated
            replaced = True
            break
    if not replaced:
        records.append(updated)
    save_all_records(records)
    _invalidate_cache()


def _invalidate_cache() -> None:
    """Remove the embeddings cache so the next retrieval rebuilds it."""
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
        print("  Embedding cache invalidated — will rebuild on next retrieval.")


# ---------------------------------------------------------------------------
# Core feedback function
# ---------------------------------------------------------------------------

def record_feedback(
    record_id: str,
    status: str,                          # "approved" | "rejected" | "corrected"
    by: str = "",                         # GitHub username or email
    corrected_fix: Optional[list[str]] = None,
    corrected_root_cause: str = "",
) -> FailureRecord:
    """
    Apply developer feedback to a FailureRecord and persist it.

    Args:
        record_id:            The `id` field of the FailureRecord to update.
        status:               "approved", "rejected", or "corrected".
        by:                   Who is giving feedback (GitHub username / email).
        corrected_fix:        Ordered list of corrected fix steps (status=corrected only).
        corrected_root_cause: Developer-written root cause override (optional).

    Returns:
        The updated FailureRecord.

    Raises:
        ValueError: if record_id is not found, or status is invalid.
    """
    # Validate status
    try:
        fb_status = FeedbackStatus(status)
    except ValueError:
        valid = [s.value for s in FeedbackStatus if s != FeedbackStatus.PENDING]
        raise ValueError(f"Invalid status '{status}'. Choose from: {valid}")

    if fb_status == FeedbackStatus.PENDING:
        raise ValueError("Cannot set status back to 'pending' via feedback.")

    # Load record
    record = find_record(record_id)
    if record is None:
        raise ValueError(
            f"No record found with id='{record_id}'.\n"
            f"Run: python3 feedback.py --stats  to list all record IDs."
        )

    # Apply feedback
    record.feedback_status      = fb_status
    record.feedback_by          = by
    record.feedback_at          = datetime.now(timezone.utc).isoformat()
    record.corrected_fix        = corrected_fix or []
    record.corrected_root_cause = corrected_root_cause

    # Persist
    upsert_record(record)

    return record


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_stats() -> dict:
    """
    Return feedback statistics across all records.

    Returns a dict with:
      total, pending, approved, rejected, corrected,
      approval_rate (%), correction_rate (%)
    """
    records = load_all_records()
    counts = {s.value: 0 for s in FeedbackStatus}
    for r in records:
        counts[r.feedback_status.value] += 1

    total     = len(records)
    responded = total - counts["pending"]
    return {
        "total":           total,
        "pending":         counts["pending"],
        "approved":        counts["approved"],
        "rejected":        counts["rejected"],
        "corrected":       counts["corrected"],
        "approval_rate":   round(counts["approved"] / responded * 100, 1) if responded else 0,
        "correction_rate": round(counts["corrected"] / responded * 100, 1) if responded else 0,
    }


def print_stats() -> None:
    stats = get_stats()
    print("\n" + "="*50)
    print("FEEDBACK STATS")
    print("="*50)
    print(f"  Total records  : {stats['total']}")
    print(f"  Pending        : {stats['pending']}")
    print(f"  Approved  (👍) : {stats['approved']}")
    print(f"  Rejected  (👎) : {stats['rejected']}")
    print(f"  Corrected (✏️ ) : {stats['corrected']}")
    if stats['total'] - stats['pending'] > 0:
        print(f"\n  Approval rate  : {stats['approval_rate']}%")
        print(f"  Correction rate: {stats['correction_rate']}%")
    print()


# ---------------------------------------------------------------------------
# Interactive CLI helper
# ---------------------------------------------------------------------------

def interactive_feedback(record: FailureRecord) -> FailureRecord:
    """Walk a developer through giving feedback interactively."""
    analysis = record.analysis
    print("\n" + "="*60)
    print(f"ANALYSIS: {record.repo} / {record.job_name}  (run {record.run_id})")
    if record.pr_number:
        print(f"PR: #{record.pr_number}  Branch: {record.branch}")
    print("="*60)
    print(f"\n[A] Root cause:\n    {analysis.get('root_cause_summary', '')}\n")
    fix = analysis.get("fix_recommendation", [])
    print("[D] Suggested fix:")
    for i, step in enumerate(fix, 1):
        print(f"    {i}. {step}")
    print(f"\n[E] Confidence: {analysis.get('confidence_score', {}).get('score', '?')}/100\n")

    # Thumbs up/down
    while True:
        choice = input("Was this analysis correct? [y]es / [n]o / [c]orrect it: ").strip().lower()
        if choice in ("y", "yes"):
            status = "approved"
            break
        elif choice in ("n", "no"):
            status = "rejected"
            break
        elif choice in ("c", "correct"):
            status = "corrected"
            break
        print("  Please enter y, n, or c.")

    by = input("Your GitHub username (leave blank to skip): ").strip()

    corrected_fix = []
    corrected_root_cause = ""
    if status == "corrected":
        print("\nEnter corrected fix steps (one per line, blank line to finish):")
        while True:
            line = input("  Step: ").strip()
            if not line:
                break
            corrected_fix.append(line)

        corrected_root_cause = input("Corrected root cause (leave blank to keep original): ").strip()

    return record_feedback(
        record_id=record.id,
        status=status,
        by=by,
        corrected_fix=corrected_fix if corrected_fix else None,
        corrected_root_cause=corrected_root_cause,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record developer feedback on a CI log analysis."
    )
    parser.add_argument("--id",      help="FailureRecord ID to update (e.g. fail-abc12345)")
    parser.add_argument("--status",  choices=["approved", "rejected", "corrected"],
                        help="Feedback: approved | rejected | corrected")
    parser.add_argument("--by",      default="", help="GitHub username giving feedback")
    parser.add_argument("--fix",     nargs="+",  help="Corrected fix steps (for --status corrected)")
    parser.add_argument("--cause",   default="", help="Corrected root cause text (for --status corrected)")
    parser.add_argument("--stats",   action="store_true", help="Show feedback statistics")
    parser.add_argument("--list",    action="store_true", help="List all record IDs and statuses")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        exit(0)

    if args.list:
        records = load_all_records()
        if not records:
            print("No records found.")
        else:
            print(f"\n{'ID':<20} {'Status':<12} {'Repo':<30} {'Job':<20} {'PR'}")
            print("-"*95)
            for r in records:
                print(f"{r.id:<20} {r.feedback_status.value:<12} {r.repo:<30} {r.job_name:<20} {r.pr_number or '-'}")
        exit(0)

    if not args.id:
        parser.print_help()
        exit(1)

    record = find_record(args.id)
    if record is None:
        print(f"ERROR: No record found with id='{args.id}'")
        exit(1)

    # Non-interactive mode if --status provided
    if args.status:
        updated = record_feedback(
            record_id=args.id,
            status=args.status,
            by=args.by,
            corrected_fix=args.fix,
            corrected_root_cause=args.cause,
        )
        emoji = {"approved": "👍", "rejected": "👎", "corrected": "✏️ "}.get(args.status, "")
        print(f"\n{emoji} Feedback recorded for {updated.id}")
        print(f"   Status    : {updated.feedback_status.value}")
        print(f"   By        : {updated.feedback_by or '(anonymous)'}")
        if updated.corrected_fix:
            print(f"   Fixed fix : {updated.corrected_fix}")
    else:
        # Interactive mode
        updated = interactive_feedback(record)
        emoji = {"approved": "👍", "rejected": "👎", "corrected": "✏️ "}.get(updated.feedback_status.value, "")
        print(f"\n{emoji} Feedback saved for {updated.id}")
