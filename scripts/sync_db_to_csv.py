#!/usr/bin/env python3
"""
Pull the live Compute Specs DB export into cpu_spec_validated.csv.

Used by .github/workflows/sync-cpu-csv.yml to open PRs when production
data differs from the repo copy. GET /api/export/csv is public; no auth.
"""

from __future__ import annotations

import os
import sys
from io import BytesIO

import pandas as pd
import requests

API_BASE = os.environ.get("API_BASE", "https://computespecsdb.com").rstrip("/")
CSV_PATH = os.environ.get("CSV_PATH", "cpu_spec_validated.csv")

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

REPO_COLUMNS = [
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
]


def fetch_export() -> bytes:
    url = f"{API_BASE}/api/export/csv"
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


def normalize_export(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(raw), sep=";")
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    missing = [col for col in REPO_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Export missing columns: {', '.join(missing)}")

    df = df[REPO_COLUMNS]
    df = df.sort_values("CPU Model Name", kind="mergesort").reset_index(drop=True)
    return df


def write_repo_csv(df: pd.DataFrame, path: str) -> None:
    # Match repo file: comma-separated with UTF-8 BOM (import_data accepts both , and ;)
    df.to_csv(path, index=False, sep=",", encoding="utf-8-sig")


def main() -> int:
    try:
        raw = fetch_export()
        df = normalize_export(raw)
        write_repo_csv(df, CSV_PATH)
    except requests.RequestException as exc:
        print(f"Failed to fetch export from {API_BASE}: {exc}", file=sys.stderr)
        return 1
    except (ValueError, pd.errors.EmptyDataError) as exc:
        print(f"Invalid export data: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(df)} rows to {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
