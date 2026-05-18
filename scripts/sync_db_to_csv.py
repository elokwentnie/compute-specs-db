#!/usr/bin/env python3
"""
Pull the live Compute Specs DB export into the repo CSVs.

Used by .github/workflows/sync-data-csv.yml to open PRs when production
data differs from the repo copy. The relevant GET /api/export/* endpoints
are public; no auth.

Select which dataset to sync via the ``DATASET`` env var (``cpu`` or
``gpu``). ``CSV_PATH`` controls the destination file; ``API_BASE`` the
upstream host.
"""

from __future__ import annotations

import os
import sys
from io import BytesIO

import pandas as pd
import requests

API_BASE = os.environ.get("API_BASE", "https://computespecsdb.com").rstrip("/")

# Cloudflare in front of Render blocks the default `python-requests/X.Y.Z` UA
# (Bot Fight Mode flags it). Send a real-browser UA so the edge lets us through.
DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUEST_HEADERS = {
    "User-Agent": os.environ.get("SYNC_USER_AGENT", DEFAULT_UA),
    "Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.8",
}

DATASETS: dict[str, dict] = {
    "cpu": {
        "export_path": "/api/export/csv",
        "default_csv_path": "cpu_spec_validated.csv",
        "sort_column": "CPU Model Name",
        "columns": [
            "CPU Model Name",
            "Family",
            "CPU Model",
            "Codename",
            "Cores",
            "Threads",
            "Max Turbo Frequency (GHz)",
            "L3 Cache (MB)",
            "TDP (W)",
            "Launch Year",
            "Max Memory (TB)",
        ],
    },
    "gpu": {
        "export_path": "/api/export/gpus/csv",
        "default_csv_path": "gpu_spec_validated.csv",
        "sort_column": "GPU Model Name",
        "columns": [
            "GPU Model Name",
            "Vendor",
            "GPU Model",
            "Form Factor",
            "Memory (GB)",
            "Memory Type",
            "TDP (W)",
        ],
    },
}


def get_dataset_config() -> dict:
    name = os.environ.get("DATASET", "cpu").strip().lower()
    if name not in DATASETS:
        raise ValueError(
            f"Unknown DATASET={name!r}; expected one of {sorted(DATASETS)}"
        )
    return {"name": name, **DATASETS[name]}


def fetch_export(export_path: str) -> bytes:
    url = f"{API_BASE}{export_path}"
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=120)
    if not response.ok:
        # Cloudflare/edge errors return HTML — log a short preview so the
        # Actions log shows whether the block is Cloudflare or the origin.
        body_preview = response.text[:500].replace("\n", " ")
        cf_ray = response.headers.get("cf-ray", "n/a")
        server = response.headers.get("server", "n/a")
        print(
            f"Upstream returned {response.status_code} "
            f"(server={server}, cf-ray={cf_ray}): {body_preview}",
            file=sys.stderr,
        )
        response.raise_for_status()
    return response.content


def normalize_export(raw: bytes, columns: list[str], sort_column: str) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(raw), sep=";")
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Export missing columns: {', '.join(missing)}")

    df = df[columns]
    df = df.sort_values(sort_column, kind="mergesort").reset_index(drop=True)
    return df


def write_repo_csv(df: pd.DataFrame, path: str) -> None:
    # Match repo file: comma-separated with UTF-8 BOM (import_data accepts both , and ;)
    df.to_csv(path, index=False, sep=",", encoding="utf-8-sig")


def main() -> int:
    try:
        cfg = get_dataset_config()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    csv_path = os.environ.get("CSV_PATH", cfg["default_csv_path"])

    try:
        raw = fetch_export(cfg["export_path"])
        df = normalize_export(raw, cfg["columns"], cfg["sort_column"])
        write_repo_csv(df, csv_path)
    except requests.RequestException as exc:
        print(
            f"Failed to fetch {cfg['name']} export from {API_BASE}: {exc}",
            file=sys.stderr,
        )
        return 1
    except (ValueError, pd.errors.EmptyDataError) as exc:
        print(f"Invalid {cfg['name']} export data: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(df)} {cfg['name']} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
