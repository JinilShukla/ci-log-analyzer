"""
tests/test_schema.py

Tests for schema.py — covers:
  - Valid full output passes validation
  - AnalysisResult serializes to correct JSON shape
  - AnalysisResult round-trips through from_dict correctly
  - ConfidenceScore label derivation (Low / Medium / High)
  - Each A–E field fails validation independently when missing/invalid
  - EXAMPLE_OUTPUT from schema.py itself passes
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from schema import (
    validate,
    SchemaValidationError,
    ConfidenceScore,
    ConfidenceLabel,
    AnalysisResult,
    TriggeringEvidence,
    EXAMPLE_OUTPUT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def valid_payload(**overrides) -> dict:
    """Return a minimal valid A–E payload, with optional field overrides."""
    base = {
        "root_cause_summary": "The build failed due to a missing environment variable.",
        "triggering_evidence": {
            "build": ["make: command not found"],
            "test": [],
            "dependency": [],
            "environment": [],
            "other": [],
        },
        "probable_cause": "The CI runner does not have 'make' installed on this image.",
        "fix_recommendation": [
            "Add 'sudo apt-get install -y make' as a setup step in the workflow.",
        ],
        "confidence_score": {"score": 80},
        "repo": "org/repo",
        "run_id": "123",
        "job_name": "build",
        "branch": "main",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_payload_passes():
    result = validate(valid_payload())
    assert isinstance(result, AnalysisResult)


def test_example_output_passes():
    """The EXAMPLE_OUTPUT shipped in schema.py must always pass."""
    result = validate(EXAMPLE_OUTPUT)
    assert result.root_cause_summary != ""


def test_to_json_has_all_keys():
    result = validate(valid_payload())
    d = result.to_dict()
    assert "root_cause_summary" in d       # A
    assert "triggering_evidence" in d      # B
    assert "probable_cause" in d           # C
    assert "fix_recommendation" in d       # D
    assert "confidence_score" in d         # E


def test_round_trip_from_dict():
    """to_dict() → from_dict() must produce an equal object."""
    result = validate(valid_payload())
    d = result.to_dict()
    result2 = AnalysisResult.from_dict(d)
    assert result.root_cause_summary == result2.root_cause_summary
    assert result.probable_cause == result2.probable_cause
    assert result.fix_recommendation == result2.fix_recommendation
    assert result.confidence_score.score == result2.confidence_score.score


def test_resembles_past_issue_none_by_default():
    result = validate(valid_payload())
    assert result.resembles_past_issue is None


def test_resembles_past_issue_populated():
    payload = valid_payload()
    payload["resembles_past_issue"] = {
        "issue_id": "42",
        "repo": "org/other-repo",
        "similarity": 0.91,
        "summary": "Same ModuleNotFoundError on a different workflow.",
    }
    result = validate(payload)
    assert result.resembles_past_issue is not None
    assert result.resembles_past_issue.issue_id == "42"
    assert result.resembles_past_issue.similarity == 0.91


# ---------------------------------------------------------------------------
# ConfidenceScore label derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected_label", [
    (0,   ConfidenceLabel.LOW),
    (44,  ConfidenceLabel.LOW),
    (45,  ConfidenceLabel.MEDIUM),
    (74,  ConfidenceLabel.MEDIUM),
    (75,  ConfidenceLabel.HIGH),
    (100, ConfidenceLabel.HIGH),
])
def test_confidence_label_derivation(score, expected_label):
    cs = ConfidenceScore(score=score)
    assert cs.label == expected_label


def test_confidence_score_out_of_range():
    with pytest.raises(ValueError):
        ConfidenceScore(score=101)
    with pytest.raises(ValueError):
        ConfidenceScore(score=-1)


# ---------------------------------------------------------------------------
# A — root_cause_summary failures
# ---------------------------------------------------------------------------

def test_missing_root_cause_summary():
    payload = valid_payload()
    del payload["root_cause_summary"]
    with pytest.raises(SchemaValidationError, match="root_cause_summary"):
        validate(payload)


def test_empty_root_cause_summary():
    with pytest.raises(SchemaValidationError, match="root_cause_summary"):
        validate(valid_payload(root_cause_summary="   "))


def test_root_cause_summary_too_long():
    long_text = " ".join(["word"] * 61)
    with pytest.raises(SchemaValidationError, match="root_cause_summary"):
        validate(valid_payload(root_cause_summary=long_text))


# ---------------------------------------------------------------------------
# B — triggering_evidence failures
# ---------------------------------------------------------------------------

def test_missing_triggering_evidence():
    payload = valid_payload()
    del payload["triggering_evidence"]
    with pytest.raises(SchemaValidationError, match="triggering_evidence"):
        validate(payload)


def test_triggering_evidence_not_dict():
    with pytest.raises(SchemaValidationError, match="triggering_evidence"):
        validate(valid_payload(triggering_evidence="some string"))


def test_triggering_evidence_all_empty_fails():
    empty_evidence = {
        "build": [], "test": [], "dependency": [], "environment": [], "other": []
    }
    with pytest.raises(SchemaValidationError, match="triggering_evidence"):
        validate(valid_payload(triggering_evidence=empty_evidence))


def test_triggering_evidence_unknown_category():
    bad_evidence = {
        "build": ["something broke"],
        "unknown_category": ["bad key"],
    }
    with pytest.raises(SchemaValidationError, match="unknown category"):
        validate(valid_payload(triggering_evidence=bad_evidence))


# ---------------------------------------------------------------------------
# C — probable_cause failures
# ---------------------------------------------------------------------------

def test_missing_probable_cause():
    payload = valid_payload()
    del payload["probable_cause"]
    with pytest.raises(SchemaValidationError, match="probable_cause"):
        validate(payload)


def test_empty_probable_cause():
    with pytest.raises(SchemaValidationError, match="probable_cause"):
        validate(valid_payload(probable_cause=""))


# ---------------------------------------------------------------------------
# D — fix_recommendation failures
# ---------------------------------------------------------------------------

def test_missing_fix_recommendation():
    payload = valid_payload()
    del payload["fix_recommendation"]
    with pytest.raises(SchemaValidationError, match="fix_recommendation"):
        validate(payload)


def test_empty_fix_recommendation_list():
    with pytest.raises(SchemaValidationError, match="fix_recommendation"):
        validate(valid_payload(fix_recommendation=[]))


def test_fix_recommendation_contains_blank_string():
    with pytest.raises(SchemaValidationError, match="fix_recommendation"):
        validate(valid_payload(fix_recommendation=["valid step", "   "]))


# ---------------------------------------------------------------------------
# E — confidence_score failures
# ---------------------------------------------------------------------------

def test_missing_confidence_score():
    payload = valid_payload()
    del payload["confidence_score"]
    with pytest.raises(SchemaValidationError, match="confidence_score"):
        validate(payload)


def test_confidence_score_not_dict():
    with pytest.raises(SchemaValidationError, match="confidence_score"):
        validate(valid_payload(confidence_score=85))


def test_confidence_score_out_of_range_via_validate():
    with pytest.raises(SchemaValidationError, match="confidence_score"):
        validate(valid_payload(confidence_score={"score": 150}))


# ---------------------------------------------------------------------------
# Multiple errors reported at once
# ---------------------------------------------------------------------------

def test_multiple_errors_reported_together():
    payload = {
        "root_cause_summary": "",
        "triggering_evidence": {"build": [], "test": [], "dependency": [], "environment": [], "other": []},
        "probable_cause": "",
        "fix_recommendation": [],
        "confidence_score": {"score": 999},
    }
    with pytest.raises(SchemaValidationError) as exc_info:
        validate(payload)
    # Should report all 5 failures at once, not just the first
    assert exc_info.value.args[0].count("  -") >= 4
