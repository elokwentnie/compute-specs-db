"""
Minimal GitHub REST client for Compute Specs DB.

Used to keep the repository CSVs (the source of truth) in sync with admin
edits, and to store public proposals as GitHub issues.

Configuration (environment variables, read at call time):
  GITHUB_TOKEN   fine-grained PAT with Contents + Issues read/write on the repo
  GITHUB_REPO    owner/name (default: elokwentnie/compute-specs-db)
  GITHUB_BRANCH  branch the CSVs live on (default: main)
"""

from __future__ import annotations

import base64
import logging
import os

import requests

logger = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"
TIMEOUT_SECONDS = 20

_known_labels: set[str] = set()


class GitHubError(Exception):
    """A GitHub API call failed."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class GitHubConflict(GitHubError):
    """The file changed on GitHub since it was read (stale sha)."""


def is_configured() -> bool:
    return bool(os.environ.get("GITHUB_TOKEN"))


def repo() -> str:
    return os.environ.get("GITHUB_REPO", "elokwentnie/compute-specs-db")


def branch() -> str:
    return os.environ.get("GITHUB_BRANCH", "main")


def _request(method: str, path: str, **kwargs) -> requests.Response:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise GitHubError("GITHUB_TOKEN is not configured")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"{API_ROOT}{path}"
    try:
        response = requests.request(method, url, headers=headers, timeout=TIMEOUT_SECONDS, **kwargs)
    except requests.RequestException as exc:
        raise GitHubError(f"GitHub request failed: {exc}") from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get("message", response.text[:200])
        except ValueError:
            detail = response.text[:200]
        logger.warning("GitHub %s %s -> %s: %s", method, path, response.status_code, detail)
        error_cls = GitHubConflict if response.status_code == 409 else GitHubError
        raise error_cls(f"GitHub API error {response.status_code}: {detail}", response.status_code)
    return response


# ---------- Repository files ----------

def get_file(path: str) -> tuple[str, str]:
    """Return (text, sha) of a file on the configured branch."""
    response = _request("GET", f"/repos/{repo()}/contents/{path}", params={"ref": branch()})
    data = response.json()
    content = data.get("content")
    if content is None or data.get("encoding") != "base64":
        # Files over 1 MB come back without inline content; fetch the raw blob.
        raw = _request(
            "GET",
            f"/repos/{repo()}/git/blobs/{data['sha']}",
        ).json()
        content = raw["content"]
    text = base64.b64decode(content).decode("utf-8")
    return text, data["sha"]


def put_file(path: str, text: str, sha: str, message: str) -> dict:
    """Commit new file contents. Raises GitHubConflict if sha is stale."""
    payload = {
        "message": message,
        "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
        "sha": sha,
        "branch": branch(),
    }
    try:
        return _request("PUT", f"/repos/{repo()}/contents/{path}", json=payload).json()
    except GitHubError as exc:
        # A stale sha is reported as 409 (or occasionally 422 "does not match").
        if exc.status == 422 and "does not match" in str(exc):
            raise GitHubConflict(str(exc), exc.status) from exc
        raise


# ---------- Issues ----------

def ensure_labels(labels: list[str]) -> None:
    """Create any labels that do not exist yet (cached per process)."""
    for label in labels:
        if label in _known_labels:
            continue
        try:
            _request("GET", f"/repos/{repo()}/labels/{label}")
        except GitHubError as exc:
            if exc.status != 404:
                raise
            try:
                _request("POST", f"/repos/{repo()}/labels", json={"name": label})
            except GitHubError as create_exc:
                if create_exc.status != 422:  # 422 = created concurrently
                    raise
        _known_labels.add(label)


def create_issue(title: str, body: str, labels: list[str]) -> dict:
    ensure_labels(labels)
    return _request(
        "POST",
        f"/repos/{repo()}/issues",
        json={"title": title, "body": body, "labels": labels},
    ).json()


def get_issue(number: int) -> dict:
    return _request("GET", f"/repos/{repo()}/issues/{number}").json()


def list_issues(labels: list[str], state: str = "open") -> list[dict]:
    """List issues (not pull requests) carrying all of the given labels."""
    issues: list[dict] = []
    page = 1
    while True:
        batch = _request(
            "GET",
            f"/repos/{repo()}/issues",
            params={
                "labels": ",".join(labels),
                "state": state,
                "per_page": 100,
                "page": page,
                "sort": "created",
                "direction": "asc",
            },
        ).json()
        issues.extend(item for item in batch if "pull_request" not in item)
        if len(batch) < 100:
            return issues
        page += 1


def comment(number: int, body: str) -> dict:
    return _request(
        "POST",
        f"/repos/{repo()}/issues/{number}/comments",
        json={"body": body},
    ).json()


def close_issue(number: int, add_labels: list[str], reason: str = "completed") -> dict:
    """Close an issue with the given state_reason ("completed" or "not_planned")."""
    if add_labels:
        ensure_labels(add_labels)
        _request(
            "POST",
            f"/repos/{repo()}/issues/{number}/labels",
            json={"labels": add_labels},
        )
    return _request(
        "PATCH",
        f"/repos/{repo()}/issues/{number}",
        json={"state": "closed", "state_reason": reason},
    ).json()
