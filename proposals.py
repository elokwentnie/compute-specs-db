"""
Public CPU/GPU proposals, stored as GitHub issues.

Each issue carries a human-readable summary plus a fenced JSON block holding
the exact submitted values, which the admin panel parses back out.
"""

from __future__ import annotations

import json
import re
from typing import Any

from csv_store import SCHEMAS

PROPOSAL_LABEL = "proposal"
ACCEPTED_LABEL = "accepted"
REJECTED_LABEL = "rejected"
DATA_MARKER = "<!-- proposal-data -->"

_DATA_BLOCK = re.compile(
    re.escape(DATA_MARKER) + r"\s*````json\s*\n(.*?)\n````",
    re.DOTALL,
)


def _inline(value: Any) -> str:
    """Render a value as an inline code span, so it can't trigger @mentions or markup."""
    text = " ".join(str(value).split()).replace("`", "'")
    return f"`{text}`"


def render_issue(kind: str, data: dict[str, Any], source_url: str, notes: str | None) -> tuple[str, str]:
    """Return (title, body) for a proposal issue."""
    schema = SCHEMAS[kind]
    name = data[schema["key_attr"]]
    title = f"[New {schema['label']}] {' '.join(str(name).split())[:120]}"

    lines = [
        f"A new **{schema['label']}** was proposed via the website.",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for attr, header in schema["fields"]:
        if attr == "validated":
            continue
        value = data.get(attr)
        lines.append(f"| {header} | {_inline(value) if value not in (None, '') else '—'} |")
    lines += ["", f"**Source:** {_inline(source_url)}", ""]
    if notes:
        safe_notes = notes.replace("```", "'''")
        lines += ["**Notes:**", "", "```text", safe_notes, "```", ""]
    lines += [
        "_Review and accept or reject this proposal in the admin panel._",
        "",
        DATA_MARKER,
    ]

    payload = {"kind": kind, "data": data, "source_url": source_url, "notes": notes}
    # Backticks only occur inside JSON strings; escaping them keeps the fence intact.
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False).replace("`", "\\u0060")
    body = "\n".join(lines) + f"\n````json\n{payload_json}\n````\n"
    return title, body


def parse_issue(issue: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the proposal payload from an issue, or None if it has none."""
    match = _DATA_BLOCK.search(issue.get("body") or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except ValueError:
        return None
    if payload.get("kind") not in SCHEMAS or not isinstance(payload.get("data"), dict):
        return None
    return payload


_TRADEMARKS = re.compile(r"\((r|tm)\)|[®™]", re.IGNORECASE)


def model_tokens(name: str) -> set[str]:
    """Tokens that identify a model number, e.g. {'6980p'} for 'Intel(R) Xeon(R) 6980P'."""
    text = _TRADEMARKS.sub(" ", name or "").lower()
    tokens = re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text)
    return {
        token for token in tokens
        if len(token) >= 3
        and any(ch.isdigit() for ch in token)
        and "." not in token  # clock speeds like 2.10 / 2.10ghz
        and not token.endswith(("core", "ghz"))
    }


def find_similar(name: str, candidates: list[tuple[int, str, bool]], limit: int = 3) -> list[dict[str, Any]]:
    """Existing entries whose name contains all of the proposal's model-number tokens."""
    wanted = model_tokens(name)
    if not wanted:
        return []
    folded = name.strip().casefold()
    matches = []
    for item_id, candidate, validated in candidates:
        if candidate.strip().casefold() == folded:
            continue  # exact duplicates are reported separately
        if wanted <= model_tokens(candidate):
            matches.append({"id": item_id, "name": candidate, "validated": bool(validated)})
            if len(matches) == limit:
                break
    return matches


def issue_kind(issue: dict[str, Any]) -> str | None:
    labels = {label["name"] for label in issue.get("labels", [])}
    for kind in SCHEMAS:
        if kind in labels:
            return kind
    return None


def is_open_proposal(issue: dict[str, Any]) -> bool:
    labels = {label["name"] for label in issue.get("labels", [])}
    return issue.get("state") == "open" and PROPOSAL_LABEL in labels
