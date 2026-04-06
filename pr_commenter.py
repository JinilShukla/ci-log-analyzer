"""
pr_commenter.py

Posts the CI log analysis as a comment on the GitHub PR that triggered the failure.

The comment contains:
  - Root cause summary
  - Triggering evidence (top lines)
  - Suggested fix (numbered steps)
  - Confidence score
  - The FailureRecord ID — developer uses this to submit feedback
  - Instructions: reply 👍 / 👎 / ✏️ with correction

Usage (standalone):
  python3 pr_commenter.py --repo owner/repo --pr 42 --record-id fail-abc12345

Usage (from analyzer.py after analysis):
  from pr_commenter import post_analysis_comment
  post_analysis_comment(repo="owner/repo", pr_number="42", record=record, analysis=analysis)

GitHub token needs `pull-requests: write` scope.
"""

import os
import json
import argparse
import requests
from dotenv import load_dotenv

from schema import AnalysisResult, FailureRecord
from feedback import find_record

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API   = "https://api.github.com"


# ---------------------------------------------------------------------------
# Comment builder
# ---------------------------------------------------------------------------

def _format_evidence(analysis: dict) -> str:
    """Pick the top 3 most informative evidence lines for the comment."""
    te = analysis.get("triggering_evidence", {})
    all_lines = []
    # Priority order: dependency > build > test > environment > other
    for cat in ("dependency", "build", "test", "environment", "other"):
        all_lines.extend(te.get(cat, []))
    top = all_lines[:4]  # cap at 4 lines to keep comment concise
    if not top:
        return "_No specific evidence lines extracted._"
    return "\n".join(f"```\n{line}\n```" for line in top)


def build_comment(record: FailureRecord, analysis: dict) -> str:
    """
    Build the GitHub PR comment markdown.

    Structure:
      🔴 CI Failure Analysis
      Root cause
      Evidence
      Fix steps
      Confidence
      --- feedback instructions ---
    """
    cs = analysis.get("confidence_score", {})
    score = cs.get("score", 0)
    label = cs.get("label", "Unknown")
    confidence_bar = "🟢" if score >= 75 else ("🟡" if score >= 45 else "🔴")

    fix_steps = record.effective_fix()
    fix_md = "\n".join(f"{i+1}. {step}" for i, step in enumerate(fix_steps))

    evidence_md = _format_evidence(analysis)

    comment = f"""## 🤖 CI Failure Analysis — `{record.job_name}`

> **Run:** [{record.run_id}](https://github.com/{record.repo}/actions/runs/{record.run_id}) · **Branch:** `{record.branch}`

---

### 🔍 Root Cause
{analysis.get('root_cause_summary', '_No summary available._')}

### 📋 Key Evidence
{evidence_md}

### 🛠️ Suggested Fix
{fix_md}

### 💡 Probable Cause
{analysis.get('probable_cause', '_Not available._')}

---

{confidence_bar} **Confidence:** {score}/100 ({label})
🔖 **Record ID:** `{record.id}`

---

### 📣 Was this helpful?
> Please reply with one of:
> - 👍 — Analysis was correct, I'll apply the fix
> - 👎 — Analysis was wrong
> - ✏️ `<your corrected fix here>` — The fix should be: ...

_Your feedback trains future analyses. Run locally:_
```bash
# Approve
python3 feedback.py --id {record.id} --status approved --by YOUR_GITHUB_USERNAME

# Reject
python3 feedback.py --id {record.id} --status rejected --by YOUR_GITHUB_USERNAME

# Correct
python3 feedback.py --id {record.id} --status corrected --by YOUR_GITHUB_USERNAME \\
  --fix "Your corrected step 1" "Your corrected step 2" \\
  --cause "Your corrected root cause"
```
"""
    return comment.strip()


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    if not GITHUB_TOKEN:
        raise EnvironmentError(
            "GITHUB_TOKEN is not set. Add it to your .env file.\n"
            "The token needs 'pull-requests: write' scope."
        )
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept":        "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def post_analysis_comment(
    repo: str,
    pr_number: str,
    record: FailureRecord,
    analysis: dict,
) -> str:
    """
    Post the analysis as a comment on the given PR.

    Args:
        repo:      owner/repo  (e.g. "JinilShukla/ci-sample")
        pr_number: PR number as string (e.g. "42")
        record:    The FailureRecord that was just persisted
        analysis:  The raw analysis dict (from AnalysisResult.to_dict())

    Returns:
        The URL of the posted comment.

    Raises:
        RuntimeError on API failure.
    """
    comment_body = build_comment(record, analysis)

    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    response = requests.post(
        url,
        headers=_headers(),
        json={"body": comment_body},
        timeout=15,
    )

    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"Failed to post PR comment: {response.status_code}\n{response.text}"
        )

    comment_url = response.json().get("html_url", "")
    print(f"  PR comment posted → {comment_url}")
    return comment_url


def delete_stale_bot_comments(repo: str, pr_number: str) -> int:
    """
    Delete previous bot analysis comments on this PR so we don't pile them up.
    Identifies bot comments by the '🤖 CI Failure Analysis' header.

    Returns the number of comments deleted.
    """
    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    response = requests.get(url, headers=_headers(), params={"per_page": 100}, timeout=15)
    if response.status_code != 200:
        return 0

    deleted = 0
    for comment in response.json():
        if "🤖 CI Failure Analysis" in comment.get("body", ""):
            del_url = f"{GITHUB_API}/repos/{repo}/issues/comments/{comment['id']}"
            requests.delete(del_url, headers=_headers(), timeout=15)
            deleted += 1

    return deleted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post a CI log analysis as a GitHub PR comment."
    )
    parser.add_argument("--repo",      required=True, help="GitHub repo (owner/repo)")
    parser.add_argument("--pr",        required=True, help="PR number")
    parser.add_argument("--record-id", required=True, help="FailureRecord ID to comment")
    parser.add_argument("--clean",     action="store_true",
                        help="Delete previous bot comments on this PR before posting")
    args = parser.parse_args()

    record = find_record(args.record_id)
    if record is None:
        print(f"ERROR: No record found with id='{args.record_id}'")
        exit(1)

    if args.clean:
        n = delete_stale_bot_comments(args.repo, args.pr)
        if n:
            print(f"  Deleted {n} stale bot comment(s).")

    try:
        url = post_analysis_comment(
            repo=args.repo,
            pr_number=args.pr,
            record=record,
            analysis=record.analysis,
        )
        print(f"\n✅ Comment posted: {url}")
    except RuntimeError as e:
        print(f"\n❌ {e}")
        exit(1)
