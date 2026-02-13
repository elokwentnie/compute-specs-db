"""
CSV Import Script

Imports compute specifications from CSV file into the database.
Useful for initial data import or batch updates.
"""

import csv
from database import SessionLocal, CPUSpec, init_db
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


def import_csv_to_db(csv_file_path="cpu_spec_validated.csv"):
    """
    Import CPU data from CSV file to database
    
    Args:
        csv_file_path: Path to the CSV file to import
    """
    init_db()
    db = SessionLocal()

    try:
        with open(csv_file_path, 'r', encoding='utf-8-sig') as file:
            # Detect delimiter from first line
            sample = file.read(4096)
            file.seek(0)
            sniffer_delim = ';' if ';' in sample.split('\n')[0] else ','
            reader = csv.DictReader(file, delimiter=sniffer_delim)

            imported_count = 0
            skipped_count = 0

            for row in reader:
                cpu_model_name_key = '\ufeffCPU Model Name' if '\ufeffCPU Model Name' in row else 'CPU Model Name'

                cpu_model_name = row.get(cpu_model_name_key, '').strip()
                if not cpu_model_name:
                    skipped_count += 1
                    continue

                family = row.get('Family', '').strip() or None
                cpu_model = row.get('CPU Model', '').strip() or None
                launch_year = clean_number(row.get('Launch Year'), default=None)
                
                # Automatically determine codename if not provided
                codename = row.get('Codename', '').strip() or None
                if not codename and cpu_model and launch_year:
                    codename = determine_cpu_generation(cpu_model, launch_year, family) or None

                cpu = CPUSpec(
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
                )

                db.add(cpu)
                imported_count += 1

            db.commit()

            print(f"Successfully imported {imported_count} CPUs")
            if skipped_count > 0:
                print(f"Skipped {skipped_count} rows with missing data")

    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found!")
        print("Make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        db.rollback()
        print(f"Error importing data: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("Starting CSV import...")
    # Clear existing data before importing
    init_db()
    db = SessionLocal()
    existing = db.query(CPUSpec).count()
    if existing > 0:
        print(f"Clearing {existing} existing records...")
        db.query(CPUSpec).delete()
        db.commit()
    db.close()
    import_csv_to_db()
    print("Import complete!")
