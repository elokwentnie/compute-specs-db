"""
CSV store: the repository CSVs on GitHub are the source of truth.

Admin edits are written here first (as a one-line commit on GitHub) and only
then applied to the SQLite database, which is rebuilt from these CSVs on every
start (see auto_import_if_empty in app.py).

Parsing keeps each file's dialect (delimiter, BOM, line endings) and leaves
untouched rows byte-for-byte identical, so every commit is a minimal diff.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import github_client

logger = logging.getLogger(__name__)

# Appended to every data commit: the app already has the change in its live DB
# and re-reads the CSVs from GitHub on start, so neither a Render redeploy nor
# the push-triggered sync workflow is needed.
COMMIT_SUFFIX = os.environ.get("CSV_COMMIT_SUFFIX", " [skip render] [skip ci]")

SCHEMAS: dict[str, dict[str, Any]] = {
    "cpu": {
        "path": "cpu_spec_validated.csv",
        "label": "CPU",
        "key_attr": "cpu_model_name",
        "fields": [
            ("cpu_model_name", "CPU Model Name"),
            ("family", "Family"),
            ("cpu_model", "CPU Model"),
            ("codename", "Codename"),
            ("cores", "Cores"),
            ("threads", "Threads"),
            ("max_turbo_frequency_ghz", "Max Turbo Frequency (GHz)"),
            ("l3_cache_mb", "L3 Cache (MB)"),
            ("tdp_watts", "TDP (W)"),
            ("launch_year", "Launch Year"),
            ("max_memory_tb", "Max Memory (TB)"),
            ("validated", "Validated"),
        ],
    },
    "gpu": {
        "path": "gpu_spec_validated.csv",
        "label": "GPU",
        "key_attr": "gpu_model_name",
        "fields": [
            ("gpu_model_name", "GPU Model Name"),
            ("vendor", "Vendor"),
            ("gpu_model", "GPU Model"),
            ("form_factor", "Form Factor"),
            ("memory_gb", "Memory (GB)"),
            ("memory_type", "Memory Type"),
            ("tdp_watts", "TDP (W)"),
            ("validated", "Validated"),
        ],
    },
}

_write_lock = threading.Lock()


class CsvStoreError(Exception):
    """Base error for CSV store operations."""


class DuplicateRowError(CsvStoreError):
    """A row with the same model name already exists."""


@dataclass
class CsvDoc:
    header: list[str]
    rows: list[list[str]]
    delimiter: str = ","
    bom: bool = False
    newline: str = "\n"
    trailing_newline: bool = True
    # Original text of each row, kept so untouched rows serialize unchanged.
    raw_rows: list[str | None] = field(default_factory=list)


# ---------- Parsing / serializing ----------

def parse(text: str) -> CsvDoc:
    bom = text.startswith("﻿")
    if bom:
        text = text[1:]
    newline = "\r\n" if "\r\n" in text else "\n"
    trailing_newline = text.endswith(newline)
    lines = text.split(newline)
    if trailing_newline:
        lines = lines[:-1]
    if not lines:
        raise CsvStoreError("CSV is empty")

    delimiter = ";" if ";" in lines[0] else ","
    header = next(csv.reader([lines[0]], delimiter=delimiter))
    rows: list[list[str]] = []
    raw_rows: list[str | None] = []
    for line in lines[1:]:
        if line == "":
            continue
        rows.append(next(csv.reader([line], delimiter=delimiter)))
        raw_rows.append(line)
    return CsvDoc(header, rows, delimiter, bom, newline, trailing_newline, raw_rows)


def _format_line(cells: list[str], delimiter: str) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, delimiter=delimiter, lineterminator="").writerow(cells)
    return buffer.getvalue()


def serialize(doc: CsvDoc) -> str:
    header_line = _format_line(doc.header, doc.delimiter)
    lines = [header_line]
    for index, cells in enumerate(doc.rows):
        raw = doc.raw_rows[index] if index < len(doc.raw_rows) else None
        lines.append(raw if raw is not None else _format_line(cells, doc.delimiter))
    text = doc.newline.join(lines)
    if doc.trailing_newline:
        text += doc.newline
    return ("﻿" if doc.bom else "") + text


# ---------- Value conversion ----------

def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    # Keep every record on one line so the file stays line-diffable.
    return " ".join(str(value).split())


def _same_value(cell: str, value: Any) -> bool:
    """True if an existing cell already represents value (e.g. '64' vs 64.0)."""
    if cell == format_value(value):
        return True
    if value is None:
        return cell.strip() == ""
    if isinstance(value, bool):
        return cell.strip().lower() == ("true" if value else "false")
    if isinstance(value, (int, float)):
        try:
            return float(cell.replace(",", ".")) == float(value)
        except ValueError:
            return False
    return cell.strip() == str(value).strip()


def values_from_orm(kind: str, obj: Any) -> dict[str, Any]:
    return {attr: getattr(obj, attr) for attr, _ in SCHEMAS[kind]["fields"]}


def _key_index(doc: CsvDoc, kind: str) -> int:
    schema = SCHEMAS[kind]
    key_header = dict(schema["fields"])[schema["key_attr"]]
    try:
        return doc.header.index(key_header)
    except ValueError as exc:
        raise CsvStoreError(f"CSV is missing the '{key_header}' column") from exc


def _find_row(doc: CsvDoc, kind: str, name: str) -> int | None:
    key = _key_index(doc, kind)
    target = name.strip()
    for index, cells in enumerate(doc.rows):
        if key < len(cells) and cells[key].strip() == target:
            return index
    folded = target.casefold()
    for index, cells in enumerate(doc.rows):
        if key < len(cells) and cells[key].strip().casefold() == folded:
            return index
    return None


def _merge_cells(doc: CsvDoc, kind: str, old_cells: list[str] | None, values: dict[str, Any]) -> list[str]:
    header_for_attr = dict(SCHEMAS[kind]["fields"])
    attr_for_header = {header: attr for attr, header in header_for_attr.items()}
    cells = list(old_cells) if old_cells else [""] * len(doc.header)
    cells += [""] * (len(doc.header) - len(cells))
    for index, header in enumerate(doc.header):
        attr = attr_for_header.get(header)
        if attr is None or attr not in values:
            continue
        if not _same_value(cells[index], values[attr]):
            cells[index] = format_value(values[attr])
    return cells


# ---------- Pure row operations ----------

def upsert(doc: CsvDoc, kind: str, old_name: str | None, values: dict[str, Any]) -> CsvDoc:
    """Replace the row named old_name (or append when old_name is None/missing)."""
    key_attr = SCHEMAS[kind]["key_attr"]
    new_name = str(values[key_attr]).strip()

    index = _find_row(doc, kind, old_name) if old_name else None
    existing = _find_row(doc, kind, new_name)
    if existing is not None and existing != index:
        raise DuplicateRowError(f"{SCHEMAS[kind]['label']} '{new_name}' already exists in the CSV")

    if index is None:
        if old_name:
            logger.warning("%s '%s' not found in CSV; appending it", kind, old_name)
        doc.rows.append(_merge_cells(doc, kind, None, values))
        doc.raw_rows.append(None)
    else:
        merged = _merge_cells(doc, kind, doc.rows[index], values)
        if merged != doc.rows[index]:
            doc.rows[index] = merged
            doc.raw_rows[index] = None
    return doc


def remove(doc: CsvDoc, kind: str, name: str) -> CsvDoc:
    index = _find_row(doc, kind, name)
    if index is None:
        logger.warning("%s '%s' not found in CSV; nothing to remove", kind, name)
        return doc
    del doc.rows[index]
    del doc.raw_rows[index]
    return doc


# ---------- GitHub write-through ----------

def _commit(kind: str, mutate: Callable[[CsvDoc], CsvDoc], message: str) -> str | None:
    """Apply mutate to the CSV on GitHub.

    Returns the commit URL, or None if the CSV already matched (no commit).
    """
    path = SCHEMAS[kind]["path"]
    with _write_lock:
        for attempt in range(2):
            text, sha = github_client.get_file(path)
            new_text = serialize(mutate(parse(text)))
            if new_text == text:
                return None
            try:
                result = github_client.put_file(path, new_text, sha, message + COMMIT_SUFFIX)
                return (result.get("commit") or {}).get("html_url", "")
            except github_client.GitHubConflict:
                if attempt == 1:
                    raise
                logger.info("CSV %s changed on GitHub during edit; retrying", path)
    return None


def commit_upsert(kind: str, old_name: str | None, values: dict[str, Any], message: str) -> str | None:
    return _commit(kind, lambda doc: upsert(doc, kind, old_name, values), message)


def commit_remove(kind: str, name: str, message: str) -> str | None:
    return _commit(kind, lambda doc: remove(doc, kind, name), message)


def load_text(kind: str) -> str | None:
    """CSV text from GitHub when configured, else from the bundled file."""
    path = SCHEMAS[kind]["path"]
    if github_client.is_configured():
        try:
            text, _ = github_client.get_file(path)
            logger.info("Loaded %s from GitHub %s@%s", path, github_client.repo(), github_client.branch())
            return text
        except github_client.GitHubError:
            logger.exception("Could not load %s from GitHub; using bundled file", path)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", newline="") as file:
        return file.read()
