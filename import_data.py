"""
CSV Import Script

Imports compute specifications from CSV file into the database.
Useful for initial data import or batch updates.
"""

import csv
import io
from database import SessionLocal, CPUSpec, GPUSpec, init_db
from utils import determine_cpu_generation


def clean_number(value, default=None):
    """Clean numeric values from CSV (handles European decimal format)"""
    if not value or value.strip() == "":
        return default

    value = str(value).strip().replace(",", ".")

    try:
        num = float(value)
        return int(num) if num.is_integer() else num
    except ValueError:
        return default


def parse_bool(value, default=False):
    """Parse a CSV boolean such as True/False, 1/0, yes/no."""
    if value is None:
        return default
    value = str(value).strip().lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    return default


def _dict_reader(text):
    """Build a DictReader for CSV text, detecting ; or , from the header line."""
    text = text.lstrip('﻿')
    header = text.split('\n', 1)[0]
    delimiter = ';' if ';' in header else ','
    return csv.DictReader(io.StringIO(text, newline=''), delimiter=delimiter)


def cpu_from_csv_row(row):
    """Build a CPUSpec from a CSV row dict, or None if the row has no model name."""
    cpu_model_name = (row.get('CPU Model Name') or '').strip()
    if not cpu_model_name:
        return None

    family = (row.get('Family') or '').strip() or None
    cpu_model = (row.get('CPU Model') or '').strip() or None
    launch_year = clean_number(row.get('Launch Year'), default=None)

    # Automatically determine codename if not provided
    codename = (row.get('Codename') or '').strip() or None
    if not codename and cpu_model and launch_year:
        codename = determine_cpu_generation(cpu_model, launch_year, family) or None

    return CPUSpec(
        cpu_model_name=cpu_model_name,
        family=family,
        cpu_model=cpu_model,
        codename=codename,
        cores=clean_number(row.get('Cores'), default=None),
        threads=clean_number(row.get('Threads'), default=None),
        max_turbo_frequency_ghz=clean_number(row.get('Max Turbo Frequency (GHz)'), default=None),
        l3_cache_mb=clean_number(row.get('L3 Cache (MB)'), default=None),
        tdp_watts=clean_number(row.get('TDP (W)'), default=None),
        launch_year=launch_year,
        max_memory_tb=clean_number(row.get('Max Memory (TB)'), default=None),
        validated=parse_bool(row.get('Validated')),
    )


def gpu_from_csv_row(row):
    """Build a GPUSpec from a CSV row dict, or None if the row has no model name."""
    gpu_model_name = (row.get('GPU Model Name') or '').strip()
    if not gpu_model_name:
        return None

    return GPUSpec(
        gpu_model_name=gpu_model_name,
        vendor=(row.get('Vendor') or '').strip() or None,
        gpu_model=(row.get('GPU Model') or '').strip() or None,
        form_factor=(row.get('Form Factor') or '').strip() or None,
        memory_gb=clean_number(row.get('Memory (GB)'), default=None),
        memory_type=(row.get('Memory Type') or '').strip() or None,
        tdp_watts=clean_number(row.get('TDP (W)'), default=None),
        validated=parse_bool(row.get('Validated')),
    )


def _import_rows(text, build_row, label):
    init_db()
    db = SessionLocal()

    try:
        imported_count = 0
        skipped_count = 0

        for row in _dict_reader(text):
            item = build_row(row)
            if item is None:
                skipped_count += 1
                continue
            db.add(item)
            imported_count += 1

        db.commit()

        print(f"Successfully imported {imported_count} {label}")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} rows with missing data")
        return imported_count
    except Exception as e:
        db.rollback()
        print(f"Error importing {label}: {e}")
        raise
    finally:
        db.close()


def import_cpu_text_to_db(text):
    """Import CPU rows from CSV text into the database."""
    return _import_rows(text, cpu_from_csv_row, "CPUs")


def import_gpu_text_to_db(text):
    """Import GPU rows from CSV text into the database."""
    return _import_rows(text, gpu_from_csv_row, "GPUs")


def _read_file(csv_file_path):
    try:
        with open(csv_file_path, 'r', encoding='utf-8-sig', newline='') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found!")
        print("Make sure the CSV file is in the same directory as this script.")
        return None


def import_csv_to_db(csv_file_path="cpu_spec_validated.csv"):
    """
    Import CPU data from CSV file to database

    Args:
        csv_file_path: Path to the CSV file to import
    """
    text = _read_file(csv_file_path)
    if text is not None:
        import_cpu_text_to_db(text)


def import_gpu_csv_to_db(csv_file_path="gpu_spec_validated.csv"):
    """
    Import GPU data from CSV file to database

    Args:
        csv_file_path: Path to the CSV file to import
    """
    text = _read_file(csv_file_path)
    if text is not None:
        import_gpu_text_to_db(text)


if __name__ == "__main__":
    print("Starting CSV import...")
    init_db()
    db = SessionLocal()

    existing_cpus = db.query(CPUSpec).count()
    if existing_cpus > 0:
        print(f"Clearing {existing_cpus} existing CPU records...")
        db.query(CPUSpec).delete()
        db.commit()

    existing_gpus = db.query(GPUSpec).count()
    if existing_gpus > 0:
        print(f"Clearing {existing_gpus} existing GPU records...")
        db.query(GPUSpec).delete()
        db.commit()

    db.close()

    import_csv_to_db()
    import_gpu_csv_to_db()
    print("Import complete!")
