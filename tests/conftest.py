import importlib
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import github_client  # noqa: E402

APP_MODULES = ("app", "database", "import_data", "llm", "auth", "csv_store", "proposals")


class FakeGitHub:
    """In-memory stand-in for the GitHub API used by github_client."""

    def __init__(self):
        self.files = {}
        self.versions = {}
        self.commits = []
        self.issues = {}
        self.comments = {}
        self.fail = False
        self.next_issue = 1

    def _check(self):
        if self.fail:
            raise github_client.GitHubError("GitHub is down", 503)

    def get_file(self, path):
        self._check()
        return self.files[path], f"sha-{self.versions[path]}"

    def put_file(self, path, text, sha, message):
        self._check()
        if sha != f"sha-{self.versions[path]}":
            raise github_client.GitHubConflict("stale sha", 409)
        self.files[path] = text
        self.versions[path] += 1
        self.commits.append(message)
        return {"commit": {"html_url": f"https://github.com/test/repo/commit/{len(self.commits)}"}}

    def create_issue(self, title, body, labels):
        self._check()
        number = self.next_issue
        self.next_issue += 1
        self.issues[number] = {
            "number": number,
            "title": title,
            "body": body,
            "labels": [{"name": label} for label in labels],
            "state": "open",
            "state_reason": None,
            "html_url": f"https://github.com/test/repo/issues/{number}",
            "created_at": "2026-10-02T12:00:00Z",
        }
        return self.issues[number]

    def get_issue(self, number):
        self._check()
        if number not in self.issues:
            raise github_client.GitHubError("Not Found", 404)
        return self.issues[number]

    def list_issues(self, labels, state="open"):
        self._check()
        return [
            issue for issue in self.issues.values()
            if issue["state"] == state
            and set(labels) <= {label["name"] for label in issue["labels"]}
        ]

    def comment(self, number, body):
        self._check()
        self.comments.setdefault(number, []).append(body)
        return {}

    def close_issue(self, number, add_labels, reason="completed"):
        self._check()
        issue = self.issues[number]
        issue["labels"] += [{"name": label} for label in add_labels]
        issue["state"] = "closed"
        issue["state_reason"] = reason
        return issue


@pytest.fixture
def fake_github():
    fake = FakeGitHub()
    for name in ("cpu_spec_validated.csv", "gpu_spec_validated.csv"):
        with open(ROOT / name, encoding="utf-8", newline="") as f:
            fake.files[name] = f.read()
        fake.versions[name] = 1
    return fake


def _load_app(tmp_path, monkeypatch, fake, token):
    monkeypatch.chdir(tmp_path)
    for name in ("cpu_spec_validated.csv", "gpu_spec_validated.csv"):
        shutil.copy(ROOT / name, tmp_path / name)
    os.symlink(ROOT / "static", tmp_path / "static")

    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")
    if token:
        monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    else:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    for attr in ("get_file", "put_file", "create_issue", "get_issue", "list_issues", "comment", "close_issue"):
        monkeypatch.setattr(github_client, attr, getattr(fake, attr))

    for module in APP_MODULES:
        sys.modules.pop(module, None)
    app_module = importlib.import_module("app")

    from fastapi.testclient import TestClient
    client = TestClient(app_module.app)
    token_value = client.post("/api/auth/login", json={"password": "test-password"}).json()["access_token"]
    client.admin_headers = {"Authorization": f"Bearer {token_value}"}
    return client


@pytest.fixture
def client(tmp_path, monkeypatch, fake_github):
    """App wired to the fake GitHub, with a fresh database built from it."""
    return _load_app(tmp_path, monkeypatch, fake_github, token=True)


@pytest.fixture
def local_client(tmp_path, monkeypatch, fake_github):
    """App without GITHUB_TOKEN: database-only, proposals disabled."""
    return _load_app(tmp_path, monkeypatch, fake_github, token=False)
