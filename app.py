"""
Compute Specs DB API

A FastAPI web application for managing and accessing HPC and datacenter compute specifications.
Provides REST API endpoints and web interfaces for viewing and managing compute hardware data.
"""

from fastapi import FastAPI, Depends, Query, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Optional
from pydantic import BaseModel
import os
import pandas as pd
import io
from datetime import datetime

from database import get_db, CPUSpec, init_db, SessionLocal
from auth import (
    get_current_user,
    create_access_token
)
from utils import determine_cpu_generation

ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
ENABLE_ADMIN_UI = os.environ.get("ENABLE_ADMIN_UI", "false").lower() == "true"
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

if ENVIRONMENT == "production" and not ADMIN_PASSWORD:
    raise RuntimeError("ADMIN_PASSWORD must be set in production")

init_db()

# Auto-import CSV data on first run if database is empty
def auto_import_if_empty():
    """Automatically import CSV data if database is empty"""
    db = SessionLocal()
    try:
        count = db.query(CPUSpec).count()
        if count == 0:
            csv_file_path = "cpu_spec_validated.csv"
            if os.path.exists(csv_file_path):
                try:
                    from import_data import import_csv_to_db
                    import_csv_to_db(csv_file_path)
                    print(f"✅ Auto-imported data from {csv_file_path}")
                except Exception as e:
                    print(f"⚠️  Auto-import failed: {e}")
    finally:
        db.close()

auto_import_if_empty()

app = FastAPI(
    title="Compute Specs DB API",
    description="API for accessing HPC and datacenter compute specifications",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests to prevent 404 errors"""
    return Response(status_code=204)


@app.get("/robots.txt")
async def robots_txt():
    """Serve robots.txt for search engine crawlers"""
    content = """User-agent: *
Allow: /
Disallow: /admin
Disallow: /api/import/
Sitemap: https://computespecsdb.com/sitemap.xml
"""
    return Response(content=content, media_type="text/plain")


@app.get("/sitemap.xml")
async def sitemap_xml():
    """Serve sitemap.xml for search engine indexing"""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://computespecsdb.com/</loc>
        <changefreq>weekly</changefreq>
        <priority>1.0</priority>
    </url>
    <url>
        <loc>https://computespecsdb.com/visualizations</loc>
        <changefreq>weekly</changefreq>
        <priority>0.8</priority>
    </url>
    <url>
        <loc>https://computespecsdb.com/api</loc>
        <changefreq>monthly</changefreq>
        <priority>0.5</priority>
    </url>
    <url>
        <loc>https://computespecsdb.com/docs</loc>
        <changefreq>monthly</changefreq>
        <priority>0.5</priority>
    </url>
</urlset>
"""
    return Response(content=content, media_type="application/xml")


class CPUSpecResponse(BaseModel):
    """Response model for compute specifications"""
    id: int
    cpu_model_name: str
    family: Optional[str] = None
    cpu_model: Optional[str] = None
    codename: Optional[str] = None
    cores: Optional[int] = None
    threads: Optional[int] = None
    max_turbo_frequency_ghz: Optional[float] = None
    l3_cache_mb: Optional[float] = None
    tdp_watts: Optional[int] = None
    launch_year: Optional[int] = None
    max_memory_tb: Optional[float] = None

    class Config:
        from_attributes = True


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the public web interface"""
    return FileResponse("static/index.html")


@app.get("/visualizations", response_class=HTMLResponse)
async def visualizations():
    """Serve the visualizations page"""
    return FileResponse("static/visualizations.html")


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel():
    """
    Serve the admin panel interface.

    Disabled in production unless ENABLE_ADMIN_UI=true.
    """
    if ENVIRONMENT == "production" and not ENABLE_ADMIN_UI:
        raise HTTPException(status_code=404, detail="Not found")

    return FileResponse("static/admin.html")


@app.get("/api", response_class=JSONResponse)
async def api_info():
    """API information and available endpoints"""
    return {
        "message": "Compute Specs DB API",
        "version": "1.0.0",
        "endpoints": {
            "all_cpus": "/api/cpus",
            "search": "/api/cpus/search?q=EPYC",
            "by_id": "/api/cpus/{id}",
            "stats": "/api/stats",
            "docs": "/docs"
        }
    }


@app.get("/api/cpus", response_model=List[CPUSpecResponse])
async def get_all_cpus(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    db: Session = Depends(get_db)
):
    """Get all CPUs with pagination"""
    cpus = db.query(CPUSpec).offset(skip).limit(limit).all()
    return cpus


@app.get("/api/cpus/search", response_model=List[CPUSpecResponse])
async def search_cpus(
    q: str = Query(..., description="Search query (searches in model name, family, CPU model, and codename)"),
    db: Session = Depends(get_db)
):
    """Search CPUs by name, family, model, or codename"""
    search_filter = or_(
        CPUSpec.cpu_model_name.ilike(f"%{q}%"),
        CPUSpec.family.ilike(f"%{q}%"),
        CPUSpec.cpu_model.ilike(f"%{q}%"),
        CPUSpec.codename.ilike(f"%{q}%")
    )

    cpus = db.query(CPUSpec).filter(search_filter).all()
    return cpus


@app.get("/api/cpus/{cpu_id}", response_model=CPUSpecResponse)
async def get_cpu_by_id(cpu_id: int, db: Session = Depends(get_db)):
    """Get a specific CPU by ID"""
    cpu = db.query(CPUSpec).filter(CPUSpec.id == cpu_id).first()

    if cpu is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"CPU with ID {cpu_id} not found"}
        )

    return cpu


@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get statistics about the compute specs database"""
    total = db.query(CPUSpec).count()

    families = db.query(CPUSpec.family).distinct().all()
    unique_families = len([f[0] for f in families if f[0]])

    codenames = db.query(CPUSpec.codename).distinct().all()
    unique_codenames = len([c[0] for c in codenames if c[0]])

    avg_cores = db.query(CPUSpec.cores).filter(CPUSpec.cores.isnot(None)).all()
    avg_cores_value = sum([c[0] for c in avg_cores]) / len(avg_cores) if avg_cores else None

    max_cores_row = db.query(CPUSpec.cores).filter(CPUSpec.cores.isnot(None)).order_by(CPUSpec.cores.desc()).first()
    max_cores = max_cores_row[0] if max_cores_row else None

    years = db.query(CPUSpec.launch_year).filter(CPUSpec.launch_year.isnot(None)).all()
    year_values = [y[0] for y in years if y[0]]
    year_range = f"{min(year_values)}–{max(year_values)}" if year_values else None

    return {
        "total_cpus": total,
        "unique_families": unique_families,
        "unique_codenames": unique_codenames,
        "average_cores": round(avg_cores_value, 2) if avg_cores_value else None,
        "max_cores": max_cores,
        "year_range": year_range
    }


class LoginRequest(BaseModel):
    """Request model for login"""
    password: str


@limiter.limit("5/minute")
@app.post("/api/auth/login")
async def login(request: Request, payload: LoginRequest):
    """
    Login endpoint - Get authentication token
    
    Requires ADMIN_PASSWORD environment variable to be set.
    Returns a JWT token for authenticated requests.
    """
    admin_password = os.environ.get("ADMIN_PASSWORD")

    if not admin_password:
        raise HTTPException(
            status_code=500,
            detail="Admin password not configured. Set ADMIN_PASSWORD environment variable."
        )

    if payload.password != admin_password:
        raise HTTPException(
            status_code=401,
            detail="Invalid password"
        )

    access_token = create_access_token(data={"sub": "admin"})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "message": "Use this token in the Authorization header: Bearer <token>"
    }


@app.get("/api/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user information"""
    return {
        "authenticated": True,
        "message": "You are authenticated!"
    }


class CPUSpecCreate(BaseModel):
    """Request model for creating a new CPU"""
    cpu_model_name: str
    family: Optional[str] = None
    cpu_model: Optional[str] = None
    codename: Optional[str] = None
    cores: Optional[int] = None
    threads: Optional[int] = None
    max_turbo_frequency_ghz: Optional[float] = None
    l3_cache_mb: Optional[float] = None
    tdp_watts: Optional[int] = None
    launch_year: Optional[int] = None
    max_memory_tb: Optional[float] = None


class CPUSpecUpdate(BaseModel):
    """Request model for updating a CPU"""
    cpu_model_name: Optional[str] = None
    family: Optional[str] = None
    cpu_model: Optional[str] = None
    codename: Optional[str] = None
    cores: Optional[int] = None
    threads: Optional[int] = None
    max_turbo_frequency_ghz: Optional[float] = None
    l3_cache_mb: Optional[float] = None
    tdp_watts: Optional[int] = None
    launch_year: Optional[int] = None
    max_memory_tb: Optional[float] = None


@limiter.limit("30/minute")
@app.post("/api/cpus", response_model=CPUSpecResponse, status_code=201)
async def create_cpu(
    request: Request,
    cpu: CPUSpecCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new compute specification (requires authentication)"""
    # Automatically determine codename if not provided
    codename = cpu.codename
    if not codename and cpu.cpu_model and cpu.launch_year:
        codename = determine_cpu_generation(cpu.cpu_model, cpu.launch_year, cpu.family) or None
    
    db_cpu = CPUSpec(
        cpu_model_name=cpu.cpu_model_name,
        family=cpu.family,
        cpu_model=cpu.cpu_model,
        codename=codename,
        cores=cpu.cores,
        threads=cpu.threads,
        max_turbo_frequency_ghz=cpu.max_turbo_frequency_ghz,
        l3_cache_mb=cpu.l3_cache_mb,
        tdp_watts=cpu.tdp_watts,
        launch_year=cpu.launch_year,
        max_memory_tb=cpu.max_memory_tb
    )

    db.add(db_cpu)
    db.commit()
    db.refresh(db_cpu)

    return db_cpu


@limiter.limit("30/minute")
@app.put("/api/cpus/{cpu_id}", response_model=CPUSpecResponse)
async def update_cpu(
    request: Request,
    cpu_id: int,
    cpu: CPUSpecUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update an existing compute specification (requires authentication)"""
    db_cpu = db.query(CPUSpec).filter(CPUSpec.id == cpu_id).first()

    if db_cpu is None:
        raise HTTPException(status_code=404, detail=f"CPU with ID {cpu_id} not found")

    update_data = cpu.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_cpu, field, value)

    db.commit()
    db.refresh(db_cpu)

    return db_cpu


@limiter.limit("30/minute")
@app.delete("/api/cpus/{cpu_id}", status_code=204)
async def delete_cpu(
    request: Request,
    cpu_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a compute specification (requires authentication)"""
    db_cpu = db.query(CPUSpec).filter(CPUSpec.id == cpu_id).first()

    if db_cpu is None:
        raise HTTPException(status_code=404, detail=f"CPU with ID {cpu_id} not found")

    db.delete(db_cpu)
    db.commit()

    return None


@app.get("/api/export/csv")
async def export_csv(db: Session = Depends(get_db)):
    """Export all CPUs as CSV file"""
    cpus = db.query(CPUSpec).all()

    df = pd.DataFrame([{
        "ID": cpu.id,
        "CPU Model Name": cpu.cpu_model_name,
        "Family": cpu.family or "",
        "CPU Model": cpu.cpu_model or "",
        "Codename": cpu.codename or "",
        "Cores": cpu.cores or "",
        "Threads": cpu.threads or "",
        "Max Turbo Frequency (GHz)": cpu.max_turbo_frequency_ghz or "",
        "L3 Cache (MB)": cpu.l3_cache_mb or "",
        "TDP (W)": cpu.tdp_watts or "",
        "Launch Year": cpu.launch_year or "",
        "Max Memory (TB)": cpu.max_memory_tb or ""
    } for cpu in cpus])

    stream = io.StringIO()
    df.to_csv(stream, index=False, sep=';')
    csv_data = stream.getvalue()

    return StreamingResponse(
        io.BytesIO(csv_data.encode('utf-8')),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=cpu_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    )


@app.get("/api/export/excel")
async def export_excel(db: Session = Depends(get_db)):
    """Export all CPUs as Excel file"""
    cpus = db.query(CPUSpec).all()

    df = pd.DataFrame([{
        "ID": cpu.id,
        "CPU Model Name": cpu.cpu_model_name,
        "Family": cpu.family or "",
        "CPU Model": cpu.cpu_model or "",
        "Codename": cpu.codename or "",
        "Cores": cpu.cores or "",
        "Threads": cpu.threads or "",
        "Max Turbo Frequency (GHz)": cpu.max_turbo_frequency_ghz or "",
        "L3 Cache (MB)": cpu.l3_cache_mb or "",
        "TDP (W)": cpu.tdp_watts or "",
        "Launch Year": cpu.launch_year or "",
        "Max Memory (TB)": cpu.max_memory_tb or ""
    } for cpu in cpus])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Compute Specifications')

    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename=cpu_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        }
    )


def clean_number(value, default=None):
    """Clean numeric values from CSV (handles European decimal format)"""
    if pd.isna(value) or value == '' or value is None:
        return default
    value = str(value).strip().replace(',', '.')
    try:
        num = float(value)
        return int(num) if num.is_integer() else num
    except ValueError:
        return default


REQUIRED_CSV_COLUMNS = [
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
    "Max Memory (TB)"
]


def validate_csv_columns(df: pd.DataFrame) -> None:
    """Ensure required CSV columns exist after BOM cleanup."""
    missing = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {', '.join(missing)}"
        )


@limiter.limit("10/minute")
@app.post("/api/import/csv-file")
async def import_csv_file(
    request: Request,
    file: UploadFile = File(...),
    clear_existing: bool = Query(False, description="Clear existing data before import"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Import CPUs from uploaded CSV file (requires authentication)
    
    CSV should be semicolon-delimited matching cpu_spec_validated.csv format.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")

    if clear_existing:
        db.query(CPUSpec).delete()
        db.commit()

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV file too large. Max size is {MAX_UPLOAD_BYTES} bytes."
        )

    if contents.startswith(b'\xef\xbb\xbf'):
        contents = contents[3:]

    try:
        df = pd.read_csv(io.BytesIO(contents), delimiter=';', encoding='utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    imported = 0
    errors = []

    df.columns = df.columns.str.replace('\ufeff', '')
    validate_csv_columns(df)

    for idx, row in df.iterrows():
        try:
            cpu_model_name_key = 'CPU Model Name'
            if '\ufeffCPU Model Name' in df.columns:
                cpu_model_name_key = '\ufeffCPU Model Name'

            cpu_model_name = str(row.get(cpu_model_name_key, '')).strip()
            if not cpu_model_name:
                errors.append(f"Row {idx + 2}: Missing CPU Model Name")
                continue

            family = str(row.get('Family', '')).strip() or None
            cpu_model = str(row.get('CPU Model', '')).strip() or None
            launch_year = clean_number(row.get('Launch Year'))
            
            # Automatically determine codename if not provided
            codename = str(row.get('Codename', '')).strip() or None
            if not codename and cpu_model and launch_year:
                codename = determine_cpu_generation(cpu_model, launch_year, family) or None

            db_cpu = CPUSpec(
                cpu_model_name=cpu_model_name,
                family=family,
                cpu_model=cpu_model,
                codename=codename,
                cores=clean_number(row.get('Cores')),
                threads=clean_number(row.get('Threads')),
                max_turbo_frequency_ghz=clean_number(row.get('Max Turbo Frequency (GHz)')),
                l3_cache_mb=clean_number(row.get('L3 Cache (MB)')),
                tdp_watts=clean_number(row.get('TDP (W)')),
                launch_year=launch_year,
                max_memory_tb=clean_number(row.get('Max Memory (TB)'))
            )

            db.add(db_cpu)
            imported += 1

        except Exception as e:
            errors.append(f"Row {idx + 2}: {str(e)}")

    db.commit()

    return {
        "message": f"Imported {imported} CPUs successfully",
        "imported": imported,
        "errors": errors[:10],
        "total_errors": len(errors)
    }


@limiter.limit("10/minute")
@app.post("/api/import/csv-repo")
async def import_csv_from_repo(
    request: Request,
    clear_existing: bool = Query(False, description="Clear existing data before import"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Import CPUs from CSV file in repository (requires authentication)
    
    Reads from cpu_spec_validated.csv in the repository root.
    Useful for updating database when CSV is updated in GitHub.
    """
    csv_file_path = "cpu_spec_validated.csv"

    if not os.path.exists(csv_file_path):
        raise HTTPException(
            status_code=404,
            detail=f"CSV file '{csv_file_path}' not found in repository"
        )
    if os.path.getsize(csv_file_path) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV file too large. Max size is {MAX_UPLOAD_BYTES} bytes."
        )

    if clear_existing:
        db.query(CPUSpec).delete()
        db.commit()

    try:
        df = pd.read_csv(csv_file_path, delimiter=';', encoding='utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    imported = 0
    errors = []

    df.columns = df.columns.str.replace('\ufeff', '')
    validate_csv_columns(df)

    for idx, row in df.iterrows():
        try:
            cpu_model_name_key = 'CPU Model Name'
            if '\ufeffCPU Model Name' in df.columns:
                cpu_model_name_key = '\ufeffCPU Model Name'

            cpu_model_name = str(row.get(cpu_model_name_key, '')).strip()
            if not cpu_model_name:
                errors.append(f"Row {idx + 2}: Missing CPU Model Name")
                continue

            family = str(row.get('Family', '')).strip() or None
            cpu_model = str(row.get('CPU Model', '')).strip() or None
            launch_year = clean_number(row.get('Launch Year'))
            
            # Automatically determine codename if not provided
            codename = str(row.get('Codename', '')).strip() or None
            if not codename and cpu_model and launch_year:
                codename = determine_cpu_generation(cpu_model, launch_year, family) or None

            db_cpu = CPUSpec(
                cpu_model_name=cpu_model_name,
                family=family,
                cpu_model=cpu_model,
                codename=codename,
                cores=clean_number(row.get('Cores')),
                threads=clean_number(row.get('Threads')),
                max_turbo_frequency_ghz=clean_number(row.get('Max Turbo Frequency (GHz)')),
                l3_cache_mb=clean_number(row.get('L3 Cache (MB)')),
                tdp_watts=clean_number(row.get('TDP (W)')),
                launch_year=launch_year,
                max_memory_tb=clean_number(row.get('Max Memory (TB)'))
            )

            db.add(db_cpu)
            imported += 1

        except Exception as e:
            errors.append(f"Row {idx + 2}: {str(e)}")

    db.commit()

    return {
        "message": f"Imported {imported} CPUs successfully from repository CSV",
        "imported": imported,
        "errors": errors[:10],
        "total_errors": len(errors),
        "source": csv_file_path
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
