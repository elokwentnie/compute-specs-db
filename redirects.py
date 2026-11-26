"""
Redirects for renamed models.

A model's page URL comes from its name (/cpu/<slug>), so renaming a model in
the admin panel moves its page. The old slug is recorded here and the old URL
answers with a 301 to the new one, so search engines and existing links keep
working.

Redirects are stored in redirects.csv next to the spec CSVs. Like the specs,
the file lives on GitHub when GITHUB_TOKEN is set (the database is rebuilt on
every deploy), and the database table is a copy loaded at startup.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import threading

from sqlalchemy.orm import Session

import github_client
from csv_store import COMMIT_SUFFIX
from database import SlugRedirect
from utils import model_slug

logger = logging.getLogger(__name__)

PATH = "redirects.csv"
HEADER = ["Kind", "Old Slug", "Model Name"]
_write_lock = threading.Lock()


# ---------- File format ----------

def parse(text: str) -> list[tuple[str, str, str]]:
    rows = []
    for row in csv.DictReader(io.StringIO(text.lstrip("\ufeff"))):
        kind = (row.get("Kind") or "").strip().lower()
        old_slug = (row.get("Old Slug") or "").strip()
        name = (row.get("Model Name") or "").strip()
        if kind in ("cpu", "gpu") and old_slug and name:
            rows.append((kind, old_slug, name))
    return rows


def serialize(rows: list[tuple[str, str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HEADER)
    writer.writerows(sorted(rows))
    return buffer.getvalue()


def apply_rename(rows: list[tuple[str, str, str]], kind: str, old_name: str, new_name: str) -> list[tuple[str, str, str]]:
    """Rows after renaming old_name to new_name: old links follow the model, with no chains or loops."""
    old_slug, new_slug = model_slug(old_name), model_slug(new_name)
    if old_slug == new_slug:
        return rows
    folded = old_name.strip().casefold()
    updated = []
    for row_kind, slug, name in rows:
        if row_kind == kind and name.strip().casefold() == folded:
            name = new_name  # earlier names of this model now point at the newest name
        if row_kind == kind and slug in (new_slug, old_slug):
            continue  # the new URL is live again; the old URL gets a fresh row below
        updated.append((row_kind, slug, name))
    updated.append((kind, old_slug, new_name))
    return updated


# ---------- Storage ----------

def load_text() -> str | None:
    """redirects.csv from GitHub when configured, else the bundled file."""
    if github_client.is_configured():
        try:
            text, _ = github_client.get_file(PATH)
            return text
        except github_client.GitHubError as exc:
            if exc.status != 404:
                logger.exception("Could not load %s from GitHub; using bundled file", PATH)
    if not os.path.exists(PATH):
        return None
    with open(PATH, encoding="utf-8", newline="") as file:
        return file.read()


def load_into_db(db: Session) -> int:
    """Fill the redirect table at startup if it is empty."""
    if db.query(SlugRedirect).count():
        return 0
    text = load_text()
    rows = parse(text) if text else []
    db.add_all(SlugRedirect(kind=k, old_slug=s, target_name=n) for k, s, n in rows)
    db.commit()
    return len(rows)


def _db_rows(db: Session) -> list[tuple[str, str, str]]:
    return [(r.kind, r.old_slug, r.target_name) for r in db.query(SlugRedirect).all()]


def _replace_db_rows(db: Session, rows: list[tuple[str, str, str]]) -> None:
    db.query(SlugRedirect).delete()
    db.add_all(SlugRedirect(kind=k, old_slug=s, target_name=n) for k, s, n in rows)
    db.commit()


def target_name(db: Session, kind: str, slug: str) -> str | None:
    row = db.query(SlugRedirect).filter(SlugRedirect.kind == kind, SlugRedirect.old_slug == slug).first()
    return row.target_name if row else None


def record_rename(db: Session, kind: str, old_name: str, new_name: str) -> None:
    """Remember the old URL of a renamed model, in the database and on GitHub.

    Called after the rename itself succeeded, so failures here are logged
    rather than raised: the rename stays, only the redirect may be missing.
    """
    if model_slug(old_name) == model_slug(new_name):
        return
    _replace_db_rows(db, apply_rename(_db_rows(db), kind, old_name, new_name))
    if not github_client.is_configured():
        return
    message = f"Redirect old {kind.upper()} URL: {old_name} -> {new_name}{COMMIT_SUFFIX}"
    with _write_lock:
        for attempt in range(2):
            try:
                try:
                    text, sha = github_client.get_file(PATH)
                except github_client.GitHubError as exc:
                    if exc.status != 404:
                        raise
                    text, sha = "", None
                rows = apply_rename(parse(text) if text else [], kind, old_name, new_name)
                github_client.put_file(PATH, serialize(rows), sha, message)
                return
            except github_client.GitHubConflict:
                if attempt == 1:
                    logger.exception("Could not save redirect for %s: %s changed on GitHub", old_name, PATH)
            except github_client.GitHubError:
                logger.exception("Could not save redirect for %s to GitHub", old_name)
                return
