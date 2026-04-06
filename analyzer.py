"""
analyzer.py

Stage 3: Send preprocessed CI log signal lines to Gemini and get back
a validated A–E structured analysis.

Flow:
  1. Build a strict system prompt that defines the exact JSON schema
  2. Send signal lines as the user message (preprocessed, noise-free)
  3. Parse the JSON response
  4. Validate against schema.py
  5. Attach metadata (repo, run_id, job_name, branch)
  6. Return a validated AnalysisResult

Token efficiency:
  - We only send signal_lines (output of preprocessor.py), NOT the full raw log
  - The system prompt is reused across calls (Gemini caches it in context)
  - We ask for JSON only — no markdown, no explanation text

Usage:
  python3 analyzer.py --input logs/24013991316/failing-job_70030110584.txt
  python3 analyzer.py --input logs/24013991316/failing-job_70030110584.txt \
                      --repo JinilShukla/ci-sample \
                      --run_id 24013991316 \
                      --job_name failing-job \
                      --branch main
"""

import os
import json
import argparse
from dotenv import load_dotenv
from google import genai

from preprocessor import preprocess
from schema import validate, SchemaValidationError, AnalysisResult, EXAMPLE_OUTPUT

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# The system prompt is the most important part.
# Rules:
#   - Be extremely explicit about the JSON shape — LLMs drift without examples
#   - Include a real EXAMPLE so the model learns the format by demonstration
#   - Forbid markdown, prose, and code fences — we need raw JSON only
#   - Define each field's constraints directly in the prompt
SYSTEM_PROMPT = f"""
You are a senior DevOps engineer specializing in CI/CD failure analysis.
You will be given filtered log lines from a failed GitHub Actions workflow job.
Your job is to analyze the failure and return a structured JSON report.

## OUTPUT FORMAT
Return ONLY a valid JSON object. No markdown. No code fences. No explanation text.
The JSON must have exactly these fields:

{{
  "root_cause_summary": "<1-2 sentences: what failed and why, in plain English>",
  "triggering_evidence": {{
    "build":       ["<log line>", ...],
    "test":        ["<log line>", ...],
    "dependency":  ["<log line>", ...],
    "environment": ["<log line>", ...],
    "other":       ["<log line>", ...]
  }},
  "probable_cause": "<detailed reasoning: why did this happen, what triggered it>",
  "fix_recommendation": [
    "<step 1>",
    "<step 2>"
  ],
  "confidence_score": {{
    "score": <integer 0-100>
  }}
}}

## RULES
- root_cause_summary: max 2 sentences, no jargon
- triggering_evidence: quote EXACT lines from the log that prove the failure; put each line in the correct category; leave empty arrays for unused categories
- build: compilation errors, make failures, linker errors
- test: test failures, assertion errors, coverage failures
- dependency: missing packages, import errors, version conflicts
- environment: missing env vars, permission errors, runner setup issues, exit codes
- other: anything that doesn't fit above
- probable_cause: explain the chain of events that led to this failure
- fix_recommendation: ordered list of concrete actionable steps a developer can take RIGHT NOW
- confidence_score.score: 0=no idea, 100=certain; be honest, not optimistic

## EXAMPLE
Input log lines:
  ModuleNotFoundError: No module named 'non_existent_module'
  Traceback (most recent call last):
  ##[error]Process completed with exit code 1.

Expected output:
{json.dumps(EXAMPLE_OUTPUT, indent=2)}
""".strip()


# ---------------------------------------------------------------------------
# Core analyzer function
# ---------------------------------------------------------------------------

def build_user_message(signal_lines: list[str]) -> str:
    """
    Format the signal lines as a clean user message for the LLM.
    We explicitly label what we're sending so the model has context.
    """
    lines_text = "\n".join(f"  {line}" for line in signal_lines)
    return f"Analyze this failed GitHub Actions job. Here are the filtered signal lines:\n\n{lines_text}"


def call_gemini(user_message: str) -> str:
    """
    Send the prompt to Gemini and return the raw response text.
    Raises RuntimeError if the API call fails.
    """
    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=MODEL,
        contents=user_message,
        config={
            "system_instruction": SYSTEM_PROMPT,
            # Ask Gemini to return JSON directly — reduces hallucinated markdown wrapping
            "response_mime_type": "application/json",
            # Lower temperature = more deterministic, less creative hallucination
            "temperature": 0.1,
        },
    )
    return response.text


def parse_llm_response(raw: str) -> dict:
    """
    Parse the LLM's raw text response into a dict.
    Gemini with response_mime_type=application/json should return clean JSON,
    but we defensively strip any accidental markdown fences just in case.
    """
    text = raw.strip()

    # Strip accidental ```json ... ``` wrapping
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM response was not valid JSON.\n"
            f"Error: {e}\n"
            f"Raw response:\n{raw[:500]}"
        )


def analyze(
    input_path: str,
    repo: str = "",
    run_id: str = "",
    job_name: str = "",
    branch: str = "",
    context_lines: int = 2,
) -> AnalysisResult:
    """
    Full pipeline: preprocess log → call LLM → validate → return AnalysisResult.

    Args:
        input_path:    Path to the raw .txt log file
        repo:          GitHub repo in owner/repo format
        run_id:        Workflow run ID
        job_name:      Name of the job being analyzed
        branch:        Git branch the run was triggered on
        context_lines: Lines of context around each signal line (default 2)

    Returns:
        A validated AnalysisResult with all A–E fields populated.
    """
    # Step 1 — Preprocess: strip noise, extract signal lines
    print(f"  Preprocessing log: {input_path}")
    result = preprocess(input_path, context_lines=context_lines)
    signal_lines = result["signal_lines"]
    stats = result["stats"]

    print(f"  Signal lines extracted: {len(signal_lines)} / {stats['raw_line_count']} raw lines ({stats['reduction_pct']}% noise removed)")

    if not signal_lines:
        raise ValueError(
            "No signal lines found in this log. "
            "The job may have passed, or the log contains no recognized error patterns."
        )

    # Step 2 — Build prompt and call LLM
    user_message = build_user_message(signal_lines)
    print(f"  Calling Gemini ({MODEL})...")
    raw_response = call_gemini(user_message)

    # Step 3 — Parse JSON response
    data = parse_llm_response(raw_response)

    # Step 4 — Attach pipeline metadata (not filled in by LLM)
    data["repo"]     = repo
    data["run_id"]   = run_id
    data["job_name"] = job_name
    data["branch"]   = branch

    # Step 5 — Validate against schema
    analysis = validate(data)
    print(f"  Schema validation passed ✅")

    return analysis


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a GitHub Actions log file using Gemini LLM."
    )
    parser.add_argument("--input",    required=True,  help="Path to the raw log .txt file")
    parser.add_argument("--repo",     default="",     help="GitHub repo (owner/repo)")
    parser.add_argument("--run_id",   default="",     help="Workflow run ID")
    parser.add_argument("--job_name", default="",     help="Job name")
    parser.add_argument("--branch",   default="",     help="Git branch")
    parser.add_argument("--context",  type=int, default=2, help="Context lines around signal lines")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        exit(1)

    print(f"\nAnalyzing: {args.input}\n")

    try:
        result = analyze(
            input_path=args.input,
            repo=args.repo,
            run_id=args.run_id,
            job_name=args.job_name,
            branch=args.branch,
            context_lines=args.context,
        )

        print("\n" + "="*60)
        print("ANALYSIS RESULT")
        print("="*60)
        print(result.to_json())

        # Save result as JSON next to the log file
        output_path = os.path.splitext(args.input)[0] + ".analysis.json"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.to_json())
        print(f"\nSaved → {output_path}")

    except SchemaValidationError as e:
        print(f"\n❌ Schema validation failed:\n{e}")
        exit(1)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        exit(1)
