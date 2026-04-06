"""
downloader.py

Fetches GitHub Actions job logs for a given repo + run_id.

Flow:
  1. List all jobs for the run via the GitHub REST API.
  2. For each job, download its raw log text (GitHub returns a redirect — requests follows it).
  3. Save each job log as a .txt file under logs/{run_id}/

Usage:
  python downloader.py --repo "owner/repo-name" --run_id 1234567890
"""

import os
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")


def get_headers() -> dict:
    """Build auth headers for GitHub API requests."""
    if not GITHUB_TOKEN:
        raise EnvironmentError(
            "GITHUB_TOKEN is not set. Add it to your .env file."
        )
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def list_jobs(owner: str, repo: str, run_id: str) -> list[dict]:
    """
    Returns a list of job objects for the given workflow run.
    Each job has: id, name, status, conclusion, steps.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
    response = requests.get(url, headers=get_headers())

    if response.status_code == 404:
        raise ValueError(f"Run ID {run_id} not found in {owner}/{repo}. Check the run ID and repo name.")
    if response.status_code == 401:
        raise PermissionError("GitHub token is invalid or lacks 'repo' / 'Actions: read' scope.")

    response.raise_for_status()

    data = response.json()
    jobs = data.get("jobs", [])
    print(f"Found {len(jobs)} job(s) for run {run_id}:")
    for job in jobs:
        print(f"  - [{job['conclusion'] or job['status']}] {job['name']} (job_id={job['id']})")
    return jobs


def download_job_log(owner: str, repo: str, job_id: int) -> str:
    """
    Downloads the raw log text for a single job.
    GitHub responds with a 302 redirect to a short-lived URL — requests follows it automatically.
    Returns the log content as a string.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"

    # allow_redirects=True is the default in requests, but being explicit here
    # because GitHub specifically uses a redirect pattern for log downloads.
    response = requests.get(url, headers=get_headers(), allow_redirects=True)

    if response.status_code == 410:
        # GitHub deletes logs after 90 days
        return "[LOG EXPIRED] GitHub has deleted these logs (older than 90 days)."

    response.raise_for_status()
    return response.text


def save_log(run_id: str, job_name: str, job_id: int, content: str) -> str:
    """
    Saves log content to: logs/{run_id}/{job_name}_{job_id}.txt
    Returns the path to the saved file.
    """
    run_dir = os.path.join(LOGS_DIR, str(run_id))
    os.makedirs(run_dir, exist_ok=True)

    # Sanitize job name for use as a filename
    safe_name = job_name.replace(" ", "_").replace("/", "-").replace("\\", "-")
    filename = f"{safe_name}_{job_id}.txt"
    filepath = os.path.join(run_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def download_run_logs(repo: str, run_id: str) -> None:
    """
    Main entry point.
    Given "owner/repo" and a run_id, downloads and saves all job logs.
    """
    if "/" not in repo:
        raise ValueError(f"Repo must be in 'owner/repo' format. Got: {repo!r}")

    owner, repo_name = repo.split("/", 1)

    print(f"\nFetching logs for: {owner}/{repo_name}, run_id={run_id}\n")

    jobs = list_jobs(owner, repo_name, run_id)

    if not jobs:
        print("No jobs found for this run.")
        return

    saved_files = []
    for job in jobs:
        job_id = job["id"]
        job_name = job["name"]
        print(f"\nDownloading log for job: {job_name} (id={job_id}) ...")

        try:
            log_content = download_job_log(owner, repo_name, job_id)
            path = save_log(run_id, job_name, job_id, log_content)
            print(f"  Saved → {path}")
            saved_files.append(path)
        except requests.HTTPError as e:
            print(f"  ERROR downloading log for job {job_id}: {e}")

    print(f"\nDone. {len(saved_files)} log file(s) saved under logs/{run_id}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download GitHub Actions job logs for a given run."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help='GitHub repo in owner/repo format. Example: "acme-org/backend-api"',
    )
    parser.add_argument(
        "--run_id",
        required=True,
        help="The workflow run ID from GitHub Actions.",
    )

    args = parser.parse_args()
    download_run_logs(repo=args.repo, run_id=args.run_id)
