import difflib
from pathlib import Path

import pytest

import csv_store

ROOT = Path(__file__).resolve().parent.parent


def read_text(kind):
    with open(ROOT / csv_store.SCHEMAS[kind]["path"], encoding="utf-8", newline="") as f:
        return f.read()


def changed_lines(before, after):
    return [
        line for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]


def first_row_values(kind, text):
    doc = csv_store.parse(text)
    row = dict(zip(doc.header, doc.rows[0]))
    return doc, row


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
def test_round_trip_is_byte_identical(kind):
    text = read_text(kind)
    assert csv_store.serialize(csv_store.parse(text)) == text


def test_dialects_detected():
    cpu = csv_store.parse(read_text("cpu"))
    assert (cpu.delimiter, cpu.bom, cpu.newline) == (",", True, "\r\n")
    gpu = csv_store.parse(read_text("gpu"))
    assert (gpu.delimiter, gpu.bom, gpu.newline) == (",", False, "\n")


def test_toggle_validated_changes_one_line():
    text = read_text("cpu")
    doc, row = first_row_values("cpu", text)
    name = row["CPU Model Name"]
    flipped = row["Validated"] != "True"

    new = csv_store.serialize(csv_store.upsert(doc, "cpu", name, {"cpu_model_name": name, "validated": flipped}))

    diff = changed_lines(text, new)
    assert len(diff) == 2
    assert diff[1].endswith("," + ("True" if flipped else "False"))
    assert new.startswith("﻿") and "\r\n" in new


def test_numeric_equivalents_are_not_rewritten():
    text = read_text("cpu")
    doc, row = first_row_values("cpu", text)
    name = row["CPU Model Name"]
    values = {
        "cpu_model_name": name,
        "cores": int(row["Cores"]),
        "l3_cache_mb": float(row["L3 Cache (MB)"]),
        "max_memory_tb": float(row["Max Memory (TB)"]),
    }
    assert csv_store.serialize(csv_store.upsert(doc, "cpu", name, values)) == text


def test_rename_replaces_row_in_place():
    text = read_text("gpu")
    doc = csv_store.parse(text)
    old = doc.rows[3][0]

    new = csv_store.serialize(csv_store.upsert(doc, "gpu", old, {"gpu_model_name": old + " Renamed", "tdp_watts": 123}))

    diff = changed_lines(text, new)
    assert len(diff) == 2
    assert diff[0].startswith("-" + old + ",")
    assert diff[1].startswith("+" + old + " Renamed,") and ",123," in diff[1]
    assert len(csv_store.parse(new).rows) == len(doc.rows)


def test_append_new_row_uses_header_order():
    text = read_text("gpu")
    doc = csv_store.parse(text)
    values = {
        "gpu_model_name": "Test GPU X1", "vendor": "TestCo", "gpu_model": "X1",
        "form_factor": "PCIe card", "memory_gb": 48, "memory_type": "GDDR7",
        "tdp_watts": 350, "validated": True,
    }
    new = csv_store.serialize(csv_store.upsert(doc, "gpu", None, values))
    assert new == text + "Test GPU X1,TestCo,X1,PCIe card,48,GDDR7,350,True\n"


def test_values_with_commas_are_quoted_and_newlines_flattened():
    doc = csv_store.parse(read_text("gpu"))
    new = csv_store.serialize(csv_store.upsert(doc, "gpu", None, {
        "gpu_model_name": "Odd, Name", "form_factor": "line1\nline2", "validated": False,
    }))
    assert new.endswith('"Odd, Name",,,line1 line2,,,,False\n')
    assert csv_store.parse(new).rows[-1][0] == "Odd, Name"


def test_duplicate_name_rejected():
    doc = csv_store.parse(read_text("cpu"))
    existing = doc.rows[0][0]
    with pytest.raises(csv_store.DuplicateRowError):
        csv_store.upsert(doc, "cpu", None, {"cpu_model_name": existing.lower()})
    with pytest.raises(csv_store.DuplicateRowError):
        csv_store.upsert(doc, "cpu", doc.rows[1][0], {"cpu_model_name": existing})


def test_remove_deletes_one_line():
    text = read_text("cpu")
    doc = csv_store.parse(text)
    name = doc.rows[10][0]
    new = csv_store.serialize(csv_store.remove(doc, "cpu", name))
    diff = changed_lines(text, new)
    assert diff == ["-" + text.split("\r\n")[11]]
