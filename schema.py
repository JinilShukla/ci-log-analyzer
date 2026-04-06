"""
schema.py

Defines the A–E output schema for CI Log Analyzer.

Every analysis — regardless of repo, language, or failure type — must produce
exactly this structure. This is the contract between the LLM (Stage 3) and
everything downstream (storage, UI, feedback loop).

Fields:
  A. root_cause_summary    — 1–2 sentence plain-English summary
  B. triggering_evidence   — quoted log lines grouped by category
  C. probable_cause        — reasoning behind the root cause
  D. fix_recommendation    — ordered list of actionable steps
  E. confidence_score      — how confident the analysis is (0–100 + label)

Optional:
  resembles_past_issue     — filled in by RAG when a similar historical failure exists

FailureRecord (canonical DB-ready format):
  The single format used by history/failures.json, the feedback loop,
  and any future database. Every record that enters the RAG store is a FailureRecord.
  Fields are chosen to be trivially mappable to a SQL or document DB row.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConfidenceLabel(str, Enum):
    LOW    = "Low"
    MEDIUM = "Medium"
    HIGH   = "High"

    @classmethod
    def from_score(cls, score: int) -> "ConfidenceLabel":
        """Derive a human label from a 0–100 numeric score."""
        if score >= 75:
            return cls.HIGH
        elif score >= 45:
            return cls.MEDIUM
        else:
            return cls.LOW


class EvidenceCategory(str, Enum):
    BUILD       = "build"
    TEST        = "test"
    DEPENDENCY  = "dependency"
    ENVIRONMENT = "environment"
    OTHER       = "other"


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------

@dataclass
class TriggeringEvidence:
    """
    B. Triggering Evidence — quoted log lines grouped by category.

    Each category holds a list of raw (cleaned) log lines that directly
    point to the failure. A category can be empty list if not applicable.
    """
    build:       list[str] = field(default_factory=list)
    test:        list[str] = field(default_factory=list)
    dependency:  list[str] = field(default_factory=list)
    environment: list[str] = field(default_factory=list)
    other:       list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([self.build, self.test, self.dependency, self.environment, self.other])


@dataclass
class ConfidenceScore:
    """
    E. Confidence Score — numeric 0–100 plus a derived Low/Medium/High label.
    """
    score: int                         # 0–100
    label: ConfidenceLabel = field(init=False)

    def __post_init__(self):
        if not (0 <= self.score <= 100):
            raise ValueError(f"Confidence score must be 0–100, got {self.score}")
        self.label = ConfidenceLabel.from_score(self.score)


@dataclass
class PastIssueReference:
    """
    Optional: populated by RAG in Stage 3 when a similar failure is found.
    'This resembles past issue #<issue_id> in repo <repo>.'
    """
    issue_id:    str
    repo:        str
    similarity:  float   # cosine similarity 0.0–1.0, for internal use
    summary:     str     # one-line description of the past issue


# ---------------------------------------------------------------------------
# Main schema
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """
    The complete A–E output for one CI log analysis.
    This is what the LLM must return, and what gets stored + displayed.
    """
    # Required fields (A–E)
    root_cause_summary:  str                  # A
    triggering_evidence: TriggeringEvidence   # B
    probable_cause:      str                  # C
    fix_recommendation:  list[str]            # D — ordered steps
    confidence_score:    ConfidenceScore      # E

    # Metadata (filled in by the pipeline, not the LLM)
    repo:       str = ""
    run_id:     str = ""
    job_name:   str = ""
    branch:     str = ""

    # Optional RAG output (Stage 3)
    resembles_past_issue: Optional[PastIssueReference] = None

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-safe)."""
        d = asdict(self)
        # Convert enums to their string values
        d["confidence_score"]["label"] = self.confidence_score.label.value
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisResult":
        """Deserialize from a plain dict (e.g. LLM JSON response)."""
        evidence_data = data.get("triggering_evidence", {})
        evidence = TriggeringEvidence(
            build=evidence_data.get("build", []),
            test=evidence_data.get("test", []),
            dependency=evidence_data.get("dependency", []),
            environment=evidence_data.get("environment", []),
            other=evidence_data.get("other", []),
        )

        cs_data = data.get("confidence_score", {})
        confidence = ConfidenceScore(score=int(cs_data.get("score", 0)))

        past_issue = None
        if data.get("resembles_past_issue"):
            pi = data["resembles_past_issue"]
            past_issue = PastIssueReference(
                issue_id=pi["issue_id"],
                repo=pi["repo"],
                similarity=float(pi.get("similarity", 0.0)),
                summary=pi.get("summary", ""),
            )

        return cls(
            root_cause_summary=data["root_cause_summary"],
            triggering_evidence=evidence,
            probable_cause=data["probable_cause"],
            fix_recommendation=data["fix_recommendation"],
            confidence_score=confidence,
            repo=data.get("repo", ""),
            run_id=data.get("run_id", ""),
            job_name=data.get("job_name", ""),
            branch=data.get("branch", ""),
            resembles_past_issue=past_issue,
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class SchemaValidationError(Exception):
    pass


def validate(data: dict) -> AnalysisResult:
    """
    Validate a raw dict (e.g. parsed from LLM JSON output) against the schema.

    Raises SchemaValidationError with a clear message on any violation.
    Returns a fully constructed AnalysisResult on success.
    """
    errors = []

    # A — root_cause_summary
    rcs = data.get("root_cause_summary", "")
    if not isinstance(rcs, str) or not rcs.strip():
        errors.append("A. root_cause_summary: must be a non-empty string")
    elif len(rcs.split()) > 60:
        errors.append("A. root_cause_summary: should be 1–2 sentences (got > 60 words)")

    # B — triggering_evidence
    te = data.get("triggering_evidence")
    if not isinstance(te, dict):
        errors.append("B. triggering_evidence: must be an object with category keys")
    else:
        valid_categories = {e.value for e in EvidenceCategory}
        for key, val in te.items():
            if key not in valid_categories:
                errors.append(f"B. triggering_evidence: unknown category '{key}'")
            if not isinstance(val, list):
                errors.append(f"B. triggering_evidence.{key}: must be a list of strings")
        evidence = TriggeringEvidence(**{k: te.get(k, []) for k in ["build", "test", "dependency", "environment", "other"]})
        if evidence.is_empty():
            errors.append("B. triggering_evidence: at least one category must have log lines")

    # C — probable_cause
    pc = data.get("probable_cause", "")
    if not isinstance(pc, str) or not pc.strip():
        errors.append("C. probable_cause: must be a non-empty string")

    # D — fix_recommendation
    fr = data.get("fix_recommendation", [])
    if not isinstance(fr, list) or len(fr) == 0:
        errors.append("D. fix_recommendation: must be a non-empty list of strings")
    elif not all(isinstance(s, str) and s.strip() for s in fr):
        errors.append("D. fix_recommendation: all items must be non-empty strings")

    # E — confidence_score
    cs = data.get("confidence_score", {})
    if not isinstance(cs, dict):
        errors.append("E. confidence_score: must be an object with a 'score' field")
    else:
        score = cs.get("score")
        if not isinstance(score, (int, float)) or not (0 <= score <= 100):
            errors.append("E. confidence_score.score: must be a number between 0 and 100")

    if errors:
        raise SchemaValidationError(
            f"Schema validation failed with {len(errors)} error(s):\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    return AnalysisResult.from_dict(data)


# ---------------------------------------------------------------------------
# Feedback status
# ---------------------------------------------------------------------------

class FeedbackStatus(str, Enum):
    PENDING   = "pending"    # analysis done, awaiting developer feedback
    APPROVED  = "approved"   # developer confirmed the fix was correct (👍)
    REJECTED  = "rejected"   # developer said the fix was wrong (👎)
    CORRECTED = "corrected"  # developer provided a corrected fix


# ---------------------------------------------------------------------------
# FailureRecord — canonical DB-ready format
# ---------------------------------------------------------------------------
# This is the SINGLE format stored in history/failures.json and any future DB.
# Every field maps directly to a SQL column or a document DB key.
# The schema is intentionally flat (no nested objects except analysis) so that
# migrating to PostgreSQL, SQLite, or a vector DB is a one-liner.
#
# DB table equivalent:
#   failures(
#     id TEXT PRIMARY KEY,
#     repo TEXT, run_id TEXT, job_name TEXT, branch TEXT, pr_number TEXT,
#     created_at TEXT,          -- ISO-8601 UTC
#     signal_lines TEXT,        -- JSON array
#     analysis TEXT,            -- JSON object (A–E fields)
#     feedback_status TEXT,     -- pending / approved / rejected / corrected
#     feedback_by TEXT,         -- GitHub username who gave feedback
#     feedback_at TEXT,         -- ISO-8601 UTC
#     corrected_fix TEXT,       -- JSON array of strings, if corrected
#     corrected_root_cause TEXT -- developer's override, if corrected
#   )

@dataclass
class FailureRecord:
    """
    One canonical failure record. Written to history on every analysis,
    updated in-place when developer feedback arrives.
    """
    # Identity
    id:         str = field(default_factory=lambda: f"fail-{uuid.uuid4().hex[:8]}")
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # CI context
    repo:      str = ""
    run_id:    str = ""
    job_name:  str = ""
    branch:    str = ""
    pr_number: str = ""          # empty for push-based runs; set for PR runs

    # Raw evidence (what the retriever embeds)
    signal_lines: list[str] = field(default_factory=list)

    # LLM output (A–E)
    analysis: dict = field(default_factory=dict)

    # Feedback fields (filled in after developer responds)
    feedback_status:       FeedbackStatus = FeedbackStatus.PENDING
    feedback_by:           str = ""   # GitHub username
    feedback_at:           str = ""   # ISO-8601 UTC
    corrected_fix:         list[str] = field(default_factory=list)
    corrected_root_cause:  str = ""

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["feedback_status"] = self.feedback_status.value
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "FailureRecord":
        return cls(
            id=d.get("id", f"fail-{uuid.uuid4().hex[:8]}"),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            repo=d.get("repo", ""),
            run_id=d.get("run_id", ""),
            job_name=d.get("job_name", ""),
            branch=d.get("branch", ""),
            pr_number=d.get("pr_number", ""),
            signal_lines=d.get("signal_lines", []),
            analysis=d.get("analysis", {}),
            feedback_status=FeedbackStatus(d.get("feedback_status", "pending")),
            feedback_by=d.get("feedback_by", ""),
            feedback_at=d.get("feedback_at", ""),
            corrected_fix=d.get("corrected_fix", []),
            corrected_root_cause=d.get("corrected_root_cause", ""),
        )

    def effective_fix(self) -> list[str]:
        """
        Return the best known fix for this failure.
        If a developer has corrected it, return that. Otherwise return the LLM's fix.
        This is what gets injected into the RAG prompt for future failures.
        """
        if self.corrected_fix:
            return self.corrected_fix
        return self.analysis.get("fix_recommendation", [])

    def effective_root_cause(self) -> str:
        """Return developer-corrected root cause, or the LLM's original."""
        if self.corrected_root_cause:
            return self.corrected_root_cause
        return self.analysis.get("root_cause_summary", "")

    def is_rag_eligible(self) -> bool:
        """
        Only APPROVED or CORRECTED records are used in RAG retrieval.
        PENDING records are included too (we don't gate on feedback for speed),
        but REJECTED records are excluded — they had wrong fixes.
        """
        return self.feedback_status != FeedbackStatus.REJECTED


# ---------------------------------------------------------------------------
# Example output (used in tests + LLM prompt engineering)
# ---------------------------------------------------------------------------

EXAMPLE_OUTPUT: dict = {
    "root_cause_summary": (
        "The workflow failed because Python attempted to import a module "
        "'non_existent_module' that is not installed in the runner environment."
    ),
    "triggering_evidence": {
        "build": [],
        "test": [],
        "dependency": [
            "ModuleNotFoundError: No module named 'non_existent_module'",
            "Traceback (most recent call last):",
            "  File \"<string>\", line 1, in <module>",
        ],
        "environment": [
            "##[error]Process completed with exit code 1.",
        ],
        "other": [],
    },
    "probable_cause": (
        "The step runs 'python -c \"import non_existent_module\"' which directly "
        "triggers a ModuleNotFoundError. The module was never installed via pip "
        "in the preceding 'Install dependencies' step, nor is it part of the "
        "Python standard library."
    ),
    "fix_recommendation": [
        "Add 'non_existent_module' to your requirements.txt (or pyproject.toml).",
        "Ensure the 'Install dependencies' step runs before any step that imports it.",
        "If this is an internal package, verify the package name is correct and the private registry is accessible.",
    ],
    "confidence_score": {
        "score": 95,
        "label": "High",
    },
    "repo":     "JinilShukla/ci-sample",
    "run_id":   "24013991316",
    "job_name": "failing-job",
    "branch":   "main",
    "resembles_past_issue": None,
}


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Validating example output against schema...\n")
    try:
        result = validate(EXAMPLE_OUTPUT)
        print("✅  Validation passed.\n")
        print(result.to_json())
    except SchemaValidationError as e:
        print(f"❌  {e}")
