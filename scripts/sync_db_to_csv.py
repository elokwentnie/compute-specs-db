#!/usr/bin/env python3
"""Sync the live site's export into the repository CSVs.

Run by .github/workflows/sync-data-csv.yml. Configured by the DATASET
(``cpu`` or ``gpu``), CSV_PATH and API_BASE environment variables.
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
            "Validated",
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
            "Validated",
        ],
    },
}


def get_dataset_config() -> dict:
    """Return the settings for the dataset named by DATASET.

    Raises:
        ValueError: For an unknown dataset.
    """
    name = os.environ.get("DATASET", "cpu").strip().lower()
    if name not in DATASETS:
        raise ValueError(
            f"Unknown DATASET={name!r}; expected one of {sorted(DATASETS)}"
        )
    return {"name": name, **DATASETS[name]}


def fetch_export(export_path: str) -> bytes:
    """Download a CSV export from the live API."""
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


def existing_validated(path: str, sort_column: str) -> dict[str, bool]:
    """Return the Validated flags in the current repo CSV, keyed by model name."""
    if not os.path.exists(path):
        return {}
    current = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    if "Validated" not in current.columns or sort_column not in current.columns:
        return {}
    flags = current["Validated"].astype(str).str.strip().str.lower() == "true"
    return dict(zip(current[sort_column].astype(str).str.strip(), flags))


def normalize_export(
    raw: bytes,
    columns: list[str],
    sort_column: str,
    previous_validated: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Convert an export into the repository CSV layout.

    Keeps the repo's Validated flags if the export has no Validated column.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(BytesIO(raw), sep=";")
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    if "Validated" not in df.columns:
        # Production predates the Validated export: keep the repo's flags
        # instead of dropping them, and never mark new rows as validated.
        previous_validated = previous_validated or {}
        df["Validated"] = [
            previous_validated.get(str(name).strip(), False) for name in df[sort_column]
        ]

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Export missing columns: {', '.join(missing)}")

    df = df[columns]
    df = df.sort_values(sort_column, kind="mergesort").reset_index(drop=True)
    return df


def write_repo_csv(df: pd.DataFrame, path: str) -> None:
    # Match repo file: comma-separated with UTF-8 BOM (import_data accepts both , and ;)
    """Write the CSV comma-separated with a UTF-8 BOM, like the repo file."""
    df.to_csv(path, index=False, sep=",", encoding="utf-8-sig")


def main() -> int:
    """Run the sync. Returns the process exit code."""
    try:
        cfg = get_dataset_config()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    csv_path = os.environ.get("CSV_PATH", cfg["default_csv_path"])

    try:
        raw = fetch_export(cfg["export_path"])
        previous = existing_validated(csv_path, cfg["sort_column"])
        df = normalize_export(raw, cfg["columns"], cfg["sort_column"], previous)
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
