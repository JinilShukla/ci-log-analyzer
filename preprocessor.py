"""
preprocessor.py

Stage 2: Clean and filter a raw GitHub Actions log file.

What it does:
  1. Strip the ISO timestamp prefix on every line (e.g. "2026-04-06T00:32:42.817Z ")
  2. Strip ANSI color/formatting codes (GitHub Actions injects these)
  3. Strip GitHub runner boilerplate lines (##[group], ##[endgroup], [command], hint:, etc.)
  4. Strip the UTF-8 BOM character that GitHub prepends to log files
  5. Keep only lines that contain signal keywords (error, failed, exception, not found, warning)
     — plus a configurable number of context lines around each match

Output:
  - A cleaned full log  (all non-boilerplate lines, timestamps removed)
  - A filtered log      (only signal lines + context)

Usage:
  python3 preprocessor.py --input logs/24013991316/build-and-test_70030110583.txt
  python3 preprocessor.py --input logs/24013991316/build-and-test_70030110583.txt --context 3
"""

import re
import os
import argparse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keywords that indicate a meaningful signal line (case-insensitive)
SIGNAL_KEYWORDS = [
    "error",
    "failed",
    "exception",
    "not found",
    "warning",
    "traceback",
    "exit code",
]

# GitHub runner boilerplate prefixes/patterns to drop entirely
BOILERPLATE_PATTERNS = [
    r"^##\[group\]",
    r"^##\[endgroup\]",
    r"^##\[warning\]Node\.js 20 actions are deprecated",  # known non-actionable runner warning
    r"^\[command\]",
    r"^hint:",
    r"^Prepare workflow directory",
    r"^Prepare all required actions",
    r"^Getting action download info",
    r"^Download action repository",
    r"^Complete job name:",
    r"^Syncing repository:",
    r"^Temporarily overriding HOME=",
    r"^Adding repository directory",
    r"^Deleting the contents of",
    r"^Post job cleanup\.",
    r"^Cleaning up orphan processes",
    r"^Secret source:",
    r"^http\.https://github\.com/",   # git credential config lines
]

# Compiled once for performance
TIMESTAMP_RE = re.compile(r"^\ufeff?[\d]{4}-[\d]{2}-[\d]{2}T[\d:.]+Z ")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def strip_timestamp(line: str) -> str:
    """Remove the leading ISO8601 timestamp GitHub prepends to every log line."""
    return TIMESTAMP_RE.sub("", line)


def strip_ansi(line: str) -> str:
    """Remove ANSI escape codes (color/formatting) injected by GitHub Actions."""
    return ANSI_RE.sub("", line)


def is_boilerplate(line: str) -> bool:
    """Return True if this line is pure runner noise with no signal value."""
    return bool(BOILERPLATE_RE.match(line.strip()))


def clean_line(raw_line: str) -> str:
    """Apply all cleaning steps to a single raw line."""
    line = raw_line.rstrip("\n").rstrip("\r")
    line = line.lstrip("\ufeff")   # strip BOM if present on first line
    line = strip_timestamp(line)
    line = strip_ansi(line)
    return line


def is_signal_line(line: str) -> bool:
    """Return True if this line contains a keyword worth flagging."""
    lower = line.lower()
    return any(kw in lower for kw in SIGNAL_KEYWORDS)


def preprocess(input_path: str, context_lines: int = 2) -> dict:
    """
    Main preprocessing function.

    Returns a dict with:
      - "cleaned_lines"  : all non-boilerplate lines after cleaning
      - "signal_lines"   : only lines matching SIGNAL_KEYWORDS, with context
      - "stats"          : basic counts for visibility
    """
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    cleaned = []
    for raw in raw_lines:
        line = clean_line(raw)
        if not line.strip():
            continue  # drop blank lines
        if is_boilerplate(line):
            continue  # drop runner noise
        cleaned.append(line)

    # Find signal line indices within the cleaned list
    signal_indices = set()
    for i, line in enumerate(cleaned):
        if is_signal_line(line):
            # Add context window around the match
            for j in range(max(0, i - context_lines), min(len(cleaned), i + context_lines + 1)):
                signal_indices.add(j)

    signal_lines = [cleaned[i] for i in sorted(signal_indices)]

    stats = {
        "raw_line_count": len(raw_lines),
        "cleaned_line_count": len(cleaned),
        "signal_line_count": len(signal_lines),
        "reduction_pct": round((1 - len(cleaned) / max(len(raw_lines), 1)) * 100, 1),
    }

    return {
        "cleaned_lines": cleaned,
        "signal_lines": signal_lines,
        "stats": stats,
    }


def save_output(result: dict, input_path: str) -> tuple[str, str]:
    """
    Save cleaned and signal logs next to the input file.
    Returns (cleaned_path, signal_path).
    """
    base = os.path.splitext(input_path)[0]
    cleaned_path = f"{base}.cleaned.txt"
    signal_path = f"{base}.signal.txt"

    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result["cleaned_lines"]))

    with open(signal_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result["signal_lines"]))

    return cleaned_path, signal_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a raw GitHub Actions log file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw .txt log file downloaded by downloader.py",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=2,
        help="Number of lines to include above/below each signal line (default: 2)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        exit(1)

    print(f"\nPreprocessing: {args.input}")
    result = preprocess(args.input, context_lines=args.context)

    s = result["stats"]
    print(f"\n  Raw lines      : {s['raw_line_count']}")
    print(f"  Cleaned lines  : {s['cleaned_line_count']}")
    print(f"  Signal lines   : {s['signal_line_count']}")
    print(f"  Noise reduced  : {s['reduction_pct']}%")

    cleaned_path, signal_path = save_output(result, args.input)
    print(f"\n  Saved cleaned log → {cleaned_path}")
    print(f"  Saved signal log  → {signal_path}")

    print("\n--- SIGNAL LINES PREVIEW ---")
    for line in result["signal_lines"]:
        print(f"  {line}")
