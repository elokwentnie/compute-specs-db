"""
Compute Specs DB API

A FastAPI web application for managing and accessing HPC and datacenter compute specifications.
Provides REST API endpoints and web interfaces for viewing and managing compute hardware data.
"""

from fastapi import FastAPI, Body, Depends, Query, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl, ValidationError
import os
import re
import time
import logging
import pandas as pd
import io
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from database import get_db, CPUSpec, GPUSpec, init_db, SessionLocal
from auth import (
    get_current_user,
    create_access_token
)
from utils import determine_cpu_generation
from llm import ask_question, LLMError, LLMRateLimitError, LLMTimeoutError
from import_data import import_cpu_text_to_db, import_gpu_text_to_db, parse_bool
import csv_store
import github_client
import proposals

ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
ENABLE_ADMIN_UI = os.environ.get("ENABLE_ADMIN_UI", "false").lower() == "true"
ENABLE_ASK_FEATURE = os.environ.get("ENABLE_ASK_FEATURE", "false").lower() == "true"
logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

if ENVIRONMENT == "production" and not ADMIN_PASSWORD:
    raise RuntimeError("ADMIN_PASSWORD must be set in production")

init_db()

# Auto-import CSV data on first run if database is empty.
# When GITHUB_TOKEN is set the CSVs are read from GitHub, so admin edits
# committed since this build was deployed are not lost on restart.
def auto_import_if_empty():
    """Automatically import CSV data if database tables are empty"""
    db = SessionLocal()
    try:
        for kind, model, importer in (
            ("cpu", CPUSpec, import_cpu_text_to_db),
            ("gpu", GPUSpec, import_gpu_text_to_db),
        ):
            if db.query(model).count() > 0:
                continue
            text = csv_store.load_text(kind)
            if text is None:
                continue
            try:
                importer(text)
                print(f"✅ Auto-imported {kind.upper()} data from {csv_store.SCHEMAS[kind]['path']}")
            except Exception as e:
                print(f"⚠️  {kind.upper()} auto-import failed: {e}")
    finally:
        db.close()

auto_import_if_empty()

app = FastAPI(
    title="Compute Specs DB API",
    description="API for accessing HPC and datacenter compute specifications",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Cloudflare caches /static/* for hours, so a deploy could pair new HTML with
# stale CSS/JS. Pages are served with a per-deploy version on asset URLs
# (Render sets RENDER_GIT_COMMIT) so every deploy fetches fresh assets.
ASSET_VERSION = (os.environ.get("RENDER_GIT_COMMIT") or str(int(time.time())))[:12]
_ASSET_URL = re.compile(r'((?:href|src)="/static/[^"?]+\.(?:css|js))"')


def serve_page(path: str) -> HTMLResponse:
    """Serve an HTML page with cache-busting versions on its CSS/JS links."""
    with open(path, encoding="utf-8") as file:
        html = file.read()
    return HTMLResponse(_ASSET_URL.sub(rf'\1?v={ASSET_VERSION}"', html))

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
        <loc>https://computespecsdb.com/propose</loc>
        <changefreq>monthly</changefreq>
        <priority>0.6</priority>
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
    validated: bool = False

    class Config:
        from_attributes = True


class GPUSpecResponse(BaseModel):
    """Response model for GPU specifications"""
    id: int
    gpu_model_name: str
    vendor: Optional[str] = None
    gpu_model: Optional[str] = None
    form_factor: Optional[str] = None
    memory_gb: Optional[int] = None
    memory_type: Optional[str] = None
    tdp_watts: Optional[int] = None
    validated: bool = False

    class Config:
        from_attributes = True


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the public web interface"""
    return serve_page("static/index.html")


@app.get("/visualizations", response_class=HTMLResponse)
async def visualizations():
    """Serve the visualizations page"""
    return serve_page("static/visualizations.html")


@app.get("/propose", response_class=HTMLResponse)
async def propose_page():
    """Serve the public form for proposing a new CPU or GPU"""
    return serve_page("static/propose.html")


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel():
    """
    Serve the admin panel interface.

    Disabled in production unless ENABLE_ADMIN_UI=true.
    """
    if ENVIRONMENT == "production" and not ENABLE_ADMIN_UI:
        raise HTTPException(status_code=404, detail="Not found")

    return serve_page("static/admin.html")


@app.get("/api", response_class=JSONResponse)
async def api_info():
    """API information and available endpoints"""
    return {
        "message": "Compute Specs DB API",
        "version": "1.0.0",
        "endpoints": {
            "all_cpus": "/api/cpus",
            "validated_cpus": "/api/cpus?validated=true",
            "search_cpus": "/api/cpus/search?q=EPYC",
            "cpu_by_id": "/api/cpus/{id}",
            "all_gpus": "/api/gpus",
            "validated_gpus": "/api/gpus?validated=true",
            "search_gpus": "/api/gpus/search?q=H100",
            "gpu_by_id": "/api/gpus/{id}",
            "stats": "/api/stats",
            "ask": "POST /api/ask",
            "propose_cpu": "POST /api/proposals/cpu",
            "propose_gpu": "POST /api/proposals/gpu",
            "docs": "/docs"
        },
        "notes": {
            "validated": (
                "Every CPU/GPU has a boolean 'validated' field: true means the specs "
                "were manually checked against official sources. Filter with ?validated=true|false."
            )
        }
    }


@app.get("/api/cpus", response_model=List[CPUSpecResponse])
async def get_all_cpus(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    validated: Optional[bool] = Query(None, description="Only return validated (true) or unvalidated (false) entries"),
    db: Session = Depends(get_db)
):
    """Get all CPUs with pagination"""
    query = db.query(CPUSpec)
    if validated is not None:
        query = query.filter(CPUSpec.validated == validated)
    cpus = query.offset(skip).limit(limit).all()
    return cpus


@app.get("/api/cpus/search", response_model=List[CPUSpecResponse])
async def search_cpus(
    q: str = Query(..., description="Search query (searches in model name, family, CPU model, and codename)"),
    validated: Optional[bool] = Query(None, description="Only return validated (true) or unvalidated (false) entries"),
    db: Session = Depends(get_db)
):
    """Search CPUs by name, family, model, or codename"""
    search_filter = or_(
        CPUSpec.cpu_model_name.ilike(f"%{q}%"),
        CPUSpec.family.ilike(f"%{q}%"),
        CPUSpec.cpu_model.ilike(f"%{q}%"),
        CPUSpec.codename.ilike(f"%{q}%")
    )

    query = db.query(CPUSpec).filter(search_filter)
    if validated is not None:
        query = query.filter(CPUSpec.validated == validated)
    return query.all()


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

    total_gpus = db.query(GPUSpec).count()
    gpu_vendors = db.query(GPUSpec.vendor).distinct().all()
    unique_gpu_vendors = len([v[0] for v in gpu_vendors if v[0]])

    gpu_memory_rows = db.query(GPUSpec.memory_gb).filter(GPUSpec.memory_gb.isnot(None)).all()
    max_gpu_memory = max([m[0] for m in gpu_memory_rows]) if gpu_memory_rows else None

    gpu_memory_types = db.query(GPUSpec.memory_type).distinct().all()
    unique_memory_types = len([m[0] for m in gpu_memory_types if m[0]])

    validated_cpus = db.query(CPUSpec).filter(CPUSpec.validated.is_(True)).count()
    validated_gpus = db.query(GPUSpec).filter(GPUSpec.validated.is_(True)).count()

    return {
        "total_cpus": total,
        "validated_cpus": validated_cpus,
        "unvalidated_cpus": total - validated_cpus,
        "unique_families": unique_families,
        "unique_codenames": unique_codenames,
        "average_cores": round(avg_cores_value, 2) if avg_cores_value else None,
        "max_cores": max_cores,
        "year_range": year_range,
        "total_gpus": total_gpus,
        "validated_gpus": validated_gpus,
        "unvalidated_gpus": total_gpus - validated_gpus,
        "unique_gpu_vendors": unique_gpu_vendors,
        "max_gpu_memory_gb": max_gpu_memory,
        "unique_memory_types": unique_memory_types
    }


class AskRequest(BaseModel):
    """Request model for the Ask feature"""
    question: str


class AskResponse(BaseModel):
    """Response model for the Ask feature"""
    answer: str
    sources: List[dict] = []


@app.post("/api/ask", response_model=AskResponse)
@limiter.limit("10/minute")
async def ask(
    request: Request,
    payload: AskRequest,
    db: Session = Depends(get_db),
):
    """
    Ask a natural-language question about CPU/GPU specs in the database.

    Answers are grounded in database queries via Groq tool calling.
    Requires GROQ_API_KEY and ENABLE_ASK_FEATURE=true.
    """
    if not ENABLE_ASK_FEATURE:
        raise HTTPException(status_code=503, detail="Ask feature is disabled")
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=503, detail="LLM not configured")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(question) > 500:
        raise HTTPException(status_code=400, detail="Question is too long (max 500 characters)")

    logger.info("Ask request (length=%d)", len(question))

    try:
        result = ask_question(question, db)
    except LLMRateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Too many requests to the language model. Please try again in a minute.",
        )
    except LLMTimeoutError:
        raise HTTPException(
            status_code=504,
            detail="The request timed out. Please try a simpler question.",
        )
    except LLMError as exc:
        logger.exception("Ask feature error")
        raise HTTPException(status_code=502, detail=str(exc))

    return AskResponse(answer=result["answer"], sources=result.get("sources", []))


@app.get("/api/gpus", response_model=List[GPUSpecResponse])
async def get_all_gpus(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    validated: Optional[bool] = Query(None, description="Only return validated (true) or unvalidated (false) entries"),
    db: Session = Depends(get_db)
):
    """Get all GPUs with pagination"""
    query = db.query(GPUSpec)
    if validated is not None:
        query = query.filter(GPUSpec.validated == validated)
    gpus = query.offset(skip).limit(limit).all()
    return gpus


@app.get("/api/gpus/search", response_model=List[GPUSpecResponse])
async def search_gpus(
    q: str = Query(..., description="Search query (searches in model name, vendor, GPU model, form factor, and memory type)"),
    validated: Optional[bool] = Query(None, description="Only return validated (true) or unvalidated (false) entries"),
    db: Session = Depends(get_db)
):
    """Search GPUs by name, vendor, model, form factor, or memory type"""
    search_filter = or_(
        GPUSpec.gpu_model_name.ilike(f"%{q}%"),
        GPUSpec.vendor.ilike(f"%{q}%"),
        GPUSpec.gpu_model.ilike(f"%{q}%"),
        GPUSpec.form_factor.ilike(f"%{q}%"),
        GPUSpec.memory_type.ilike(f"%{q}%")
    )
    query = db.query(GPUSpec).filter(search_filter)
    if validated is not None:
        query = query.filter(GPUSpec.validated == validated)
    return query.all()


@app.get("/api/gpus/{gpu_id}", response_model=GPUSpecResponse)
async def get_gpu_by_id(gpu_id: int, db: Session = Depends(get_db)):
    """Get a specific GPU by ID"""
    gpu = db.query(GPUSpec).filter(GPUSpec.id == gpu_id).first()
    if gpu is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"GPU with ID {gpu_id} not found"}
        )
    return gpu


class LoginRequest(BaseModel):
    """Request model for login"""
    password: str


@app.post("/api/auth/login")
@limiter.limit("5/minute")
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
        "message": "You are authenticated!",
        "github_sync": {
            "enabled": github_client.is_configured(),
            "repo": github_client.repo(),
            "branch": github_client.branch(),
        },
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
    validated: bool = False


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
    validated: Optional[bool] = None


class GPUSpecCreate(BaseModel):
    """Request model for creating a new GPU"""
    gpu_model_name: str
    vendor: Optional[str] = None
    gpu_model: Optional[str] = None
    form_factor: Optional[str] = None
    memory_gb: Optional[int] = None
    memory_type: Optional[str] = None
    tdp_watts: Optional[int] = None
    validated: bool = False


class GPUSpecUpdate(BaseModel):
    """Request model for updating a GPU"""
    gpu_model_name: Optional[str] = None
    vendor: Optional[str] = None
    gpu_model: Optional[str] = None
    form_factor: Optional[str] = None
    memory_gb: Optional[int] = None
    memory_type: Optional[str] = None
    tdp_watts: Optional[int] = None
    validated: Optional[bool] = None


class ValidatedUpdate(BaseModel):
    """Request model for toggling the validated flag"""
    validated: bool


# ---------- Write-through helpers ----------
# Every admin change is committed to the CSV on GitHub first and only then
# applied to SQLite, so the two never diverge. Without GITHUB_TOKEN (local
# development) changes only touch the database.

SPEC_MODELS = {"cpu": CPUSpec, "gpu": GPUSpec}
SPEC_CREATE_MODELS = {"cpu": CPUSpecCreate, "gpu": GPUSpecCreate}


def _spec_label(kind: str) -> str:
    return csv_store.SCHEMAS[kind]["label"]


def _find_by_name(db: Session, kind: str, name: str, exclude_id: Optional[int] = None):
    model = SPEC_MODELS[kind]
    column = getattr(model, csv_store.SCHEMAS[kind]["key_attr"])
    query = db.query(model).filter(func.lower(column) == name.strip().lower())
    if exclude_id is not None:
        query = query.filter(model.id != exclude_id)
    return query.first()


def _sync_csv(action):
    """Run a csv_store commit, translating failures into HTTP errors."""
    if not github_client.is_configured():
        return None
    try:
        return action()
    except csv_store.DuplicateRowError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except (github_client.GitHubError, csv_store.CsvStoreError) as exc:
        logger.exception("Could not write CSV change to GitHub")
        raise HTTPException(
            status_code=502,
            detail=f"Could not save the change to GitHub, so nothing was changed: {exc}",
        )


def _clean_name(kind: str, name: Optional[str]) -> str:
    name = " ".join((name or "").split())
    if not name:
        raise HTTPException(status_code=400, detail=f"{_spec_label(kind)} model name cannot be empty")
    return name


def _create_spec(db: Session, kind: str, values: dict, message: str):
    key = csv_store.SCHEMAS[kind]["key_attr"]
    values[key] = _clean_name(kind, values.get(key))
    if _find_by_name(db, kind, values[key]):
        raise HTTPException(status_code=409, detail=f"{_spec_label(kind)} '{values[key]}' already exists")

    commit_url = _sync_csv(lambda: csv_store.commit_upsert(kind, None, values, message))

    item = SPEC_MODELS[kind](**values)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item, commit_url


def _get_spec_or_404(db: Session, kind: str, item_id: int):
    item = db.query(SPEC_MODELS[kind]).filter(SPEC_MODELS[kind].id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail=f"{_spec_label(kind)} with ID {item_id} not found")
    return item


def _update_spec(db: Session, kind: str, item_id: int, changes: dict):
    item = _get_spec_or_404(db, kind, item_id)
    label = _spec_label(kind)
    key = csv_store.SCHEMAS[kind]["key_attr"]
    old_name = getattr(item, key)

    if key in changes:
        changes[key] = _clean_name(kind, changes[key])
        if _find_by_name(db, kind, changes[key], exclude_id=item_id):
            raise HTTPException(status_code=409, detail=f"{label} '{changes[key]}' already exists")
    if "validated" in changes and changes["validated"] is None:
        del changes["validated"]

    values = {**csv_store.values_from_orm(kind, item), **changes}
    new_name = values[key]
    if set(changes) == {"validated"}:
        message = f"Mark {label} {'validated' if values['validated'] else 'unvalidated'}: {new_name}"
    elif new_name != old_name:
        message = f"Rename {label}: {old_name} -> {new_name}"
    else:
        message = f"Update {label}: {new_name}"

    _sync_csv(lambda: csv_store.commit_upsert(kind, old_name, values, message))

    for field_name, value in changes.items():
        setattr(item, field_name, value)
    db.commit()
    db.refresh(item)
    return item


def _delete_spec(db: Session, kind: str, item_id: int) -> None:
    item = _get_spec_or_404(db, kind, item_id)
    name = getattr(item, csv_store.SCHEMAS[kind]["key_attr"])
    _sync_csv(lambda: csv_store.commit_remove(kind, name, f"Remove {_spec_label(kind)}: {name}"))
    db.delete(item)
    db.commit()


def _cpu_create_values(cpu: CPUSpecCreate) -> dict:
    values = cpu.model_dump()
    # Automatically determine codename if not provided
    if not values.get("codename") and cpu.cpu_model and cpu.launch_year:
        values["codename"] = determine_cpu_generation(cpu.cpu_model, cpu.launch_year, cpu.family) or None
    return values


@app.post("/api/cpus", response_model=CPUSpecResponse, status_code=201)
@limiter.limit("30/minute")
def create_cpu(
    request: Request,
    cpu: CPUSpecCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new compute specification (requires authentication)"""
    values = _cpu_create_values(cpu)
    item, _ = _create_spec(db, "cpu", values, f"Add CPU: {values['cpu_model_name']}")
    return item


@app.put("/api/cpus/{cpu_id}", response_model=CPUSpecResponse)
@limiter.limit("30/minute")
def update_cpu(
    request: Request,
    cpu_id: int,
    cpu: CPUSpecUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update an existing compute specification (requires authentication)"""
    return _update_spec(db, "cpu", cpu_id, cpu.model_dump(exclude_unset=True))


@app.patch("/api/cpus/{cpu_id}/validated", response_model=CPUSpecResponse)
@limiter.limit("60/minute")
def set_cpu_validated(
    request: Request,
    cpu_id: int,
    payload: ValidatedUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Mark a CPU as validated or unvalidated (requires authentication)"""
    return _update_spec(db, "cpu", cpu_id, {"validated": payload.validated})


@app.delete("/api/cpus/{cpu_id}", status_code=204)
@limiter.limit("30/minute")
def delete_cpu(
    request: Request,
    cpu_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a compute specification (requires authentication)"""
    _delete_spec(db, "cpu", cpu_id)
    return None


@app.post("/api/gpus", response_model=GPUSpecResponse, status_code=201)
@limiter.limit("30/minute")
def create_gpu(
    request: Request,
    gpu: GPUSpecCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new GPU specification (requires authentication)"""
    values = gpu.model_dump()
    item, _ = _create_spec(db, "gpu", values, f"Add GPU: {values['gpu_model_name']}")
    return item


@app.put("/api/gpus/{gpu_id}", response_model=GPUSpecResponse)
@limiter.limit("30/minute")
def update_gpu(
    request: Request,
    gpu_id: int,
    gpu: GPUSpecUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update an existing GPU specification (requires authentication)"""
    return _update_spec(db, "gpu", gpu_id, gpu.model_dump(exclude_unset=True))


@app.patch("/api/gpus/{gpu_id}/validated", response_model=GPUSpecResponse)
@limiter.limit("60/minute")
def set_gpu_validated(
    request: Request,
    gpu_id: int,
    payload: ValidatedUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Mark a GPU as validated or unvalidated (requires authentication)"""
    return _update_spec(db, "gpu", gpu_id, {"validated": payload.validated})


@app.delete("/api/gpus/{gpu_id}", status_code=204)
@limiter.limit("30/minute")
def delete_gpu(
    request: Request,
    gpu_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a GPU specification (requires authentication)"""
    _delete_spec(db, "gpu", gpu_id)
    return None


# ---------- Public proposals (stored as GitHub issues) ----------

PROPOSAL_MAX_TEXT = 200


class _ProposalExtras(BaseModel):
    source_url: HttpUrl
    notes: Optional[str] = Field(None, max_length=1000)
    # Honeypot: hidden in the form, so only bots fill it in.
    website: Optional[str] = None


class CPUProposal(_ProposalExtras, CPUSpecCreate):
    """Public request model for proposing a new CPU"""


class GPUProposal(_ProposalExtras, GPUSpecCreate):
    """Public request model for proposing a new GPU"""


class RejectRequest(BaseModel):
    """Request model for rejecting a proposal"""
    reason: Optional[str] = Field(None, max_length=2000)


def _require_github():
    if not github_client.is_configured():
        raise HTTPException(status_code=503, detail="Proposals are not configured on this server")


def _submit_proposal(kind: str, payload: _ProposalExtras, db: Session) -> dict:
    if payload.website:
        raise HTTPException(status_code=400, detail="Invalid submission")
    _require_github()

    data = payload.model_dump(exclude={"source_url", "notes", "website", "validated"})
    for field_name, value in list(data.items()):
        if isinstance(value, str):
            value = " ".join(value.split())
            if len(value) > PROPOSAL_MAX_TEXT:
                raise HTTPException(status_code=422, detail=f"'{field_name}' is too long")
            data[field_name] = value or None
        elif isinstance(value, (int, float)) and value < 0:
            raise HTTPException(status_code=422, detail=f"'{field_name}' cannot be negative")

    key = csv_store.SCHEMAS[kind]["key_attr"]
    data[key] = _clean_name(kind, data.get(key))
    if _find_by_name(db, kind, data[key]):
        raise HTTPException(status_code=409, detail=f"'{data[key]}' is already in the database")

    notes = (payload.notes or "").strip() or None
    title, body = proposals.render_issue(kind, data, str(payload.source_url), notes)
    try:
        issue = github_client.create_issue(title, body, [proposals.PROPOSAL_LABEL, kind])
    except github_client.GitHubError:
        logger.exception("Could not create proposal issue")
        raise HTTPException(status_code=502, detail="Could not submit the proposal right now. Please try again later.")

    return {"issue_number": issue["number"], "issue_url": issue["html_url"]}


@app.post("/api/proposals/cpu", status_code=201)
@limiter.limit("3/minute")
def propose_cpu(request: Request, payload: CPUProposal, db: Session = Depends(get_db)):
    """Propose a new CPU. Creates a GitHub issue for the maintainer to review."""
    return _submit_proposal("cpu", payload, db)


@app.post("/api/proposals/gpu", status_code=201)
@limiter.limit("3/minute")
def propose_gpu(request: Request, payload: GPUProposal, db: Session = Depends(get_db)):
    """Propose a new GPU. Creates a GitHub issue for the maintainer to review."""
    return _submit_proposal("gpu", payload, db)


def _load_open_proposal(number: int) -> tuple[dict, str, Optional[dict]]:
    try:
        issue = github_client.get_issue(number)
    except github_client.GitHubError as exc:
        status = 404 if exc.status == 404 else 502
        raise HTTPException(status_code=status, detail=f"Could not load issue #{number}: {exc}")
    if not proposals.is_open_proposal(issue):
        raise HTTPException(status_code=409, detail=f"Issue #{number} is not an open proposal")
    payload = proposals.parse_issue(issue)
    kind = payload["kind"] if payload else proposals.issue_kind(issue)
    if kind is None:
        raise HTTPException(status_code=409, detail=f"Issue #{number} is not labelled cpu or gpu")
    return issue, kind, payload


@app.get("/api/proposals")
def list_proposals(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """List open proposals (requires authentication)"""
    _require_github()
    try:
        issues = github_client.list_issues([proposals.PROPOSAL_LABEL])
    except github_client.GitHubError as exc:
        raise HTTPException(status_code=502, detail=f"Could not load proposals from GitHub: {exc}")

    candidates = {}
    for kind, model in SPEC_MODELS.items():
        name_column = getattr(model, csv_store.SCHEMAS[kind]["key_attr"])
        candidates[kind] = db.query(model.id, name_column, model.validated).all()

    results = []
    for issue in issues:
        payload = proposals.parse_issue(issue)
        kind = payload["kind"] if payload else proposals.issue_kind(issue)
        data = payload["data"] if payload else {}
        duplicate = None
        similar = []
        if kind and data:
            name = data.get(csv_store.SCHEMAS[kind]["key_attr"]) or ""
            existing = _find_by_name(db, kind, name) if name else None
            if existing:
                duplicate = {"id": existing.id, "name": getattr(existing, csv_store.SCHEMAS[kind]["key_attr"])}
            similar = proposals.find_similar(name, candidates[kind])
        results.append({
            "number": issue["number"],
            "title": issue["title"],
            "url": issue["html_url"],
            "created_at": issue["created_at"],
            "kind": kind,
            "data": data,
            "source_url": payload.get("source_url") if payload else None,
            "notes": payload.get("notes") if payload else None,
            "parsed": payload is not None,
            "duplicate_of": duplicate,
            "similar": similar,
        })
    return results


@app.post("/api/proposals/{number}/accept", status_code=201)
@limiter.limit("30/minute")
def accept_proposal(
    request: Request,
    number: int,
    payload: dict = Body(..., description="Final field values; defaults to validated=true"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Accept a proposal: add it to the CSV on GitHub and the database, then close the issue."""
    _require_github()
    _, kind, _ = _load_open_proposal(number)

    try:
        spec = SPEC_CREATE_MODELS[kind].model_validate({"validated": True, **payload})
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False))

    values = _cpu_create_values(spec) if kind == "cpu" else spec.model_dump()
    key = csv_store.SCHEMAS[kind]["key_attr"]
    label = _spec_label(kind)
    name = _clean_name(kind, values.get(key))
    item, commit_url = _create_spec(db, kind, values, f"Add {label}: {name} (closes #{number})")

    issue_closed = True
    try:
        added = f" in {commit_url}" if commit_url else ""
        github_client.comment(number, f"Thanks! This {label} was accepted and added to the database{added}.")
        github_client.close_issue(number, [proposals.ACCEPTED_LABEL], reason="completed")
    except github_client.GitHubError:
        logger.exception("Accepted proposal #%s but could not close the issue", number)
        issue_closed = False

    return {"kind": kind, "id": item.id, "name": name, "commit_url": commit_url, "issue_closed": issue_closed}


@app.post("/api/proposals/{number}/reject")
@limiter.limit("30/minute")
def reject_proposal(
    request: Request,
    number: int,
    payload: RejectRequest,
    current_user: dict = Depends(get_current_user)
):
    """Reject a proposal: comment with the reason and close the issue."""
    _require_github()
    _load_open_proposal(number)

    message = "Thanks for the proposal! It was not added to the database."
    reason = (payload.reason or "").strip()
    if reason:
        message += f"\n\n**Reason:** {reason}"
    try:
        github_client.comment(number, message)
        github_client.close_issue(number, [proposals.REJECTED_LABEL], reason="not_planned")
    except github_client.GitHubError as exc:
        raise HTTPException(status_code=502, detail=f"Could not update issue #{number}: {exc}")
    return {"number": number, "rejected": True}


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
        "Max Memory (TB)": cpu.max_memory_tb or "",
        "Validated": bool(cpu.validated)
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
        "Max Memory (TB)": cpu.max_memory_tb or "",
        "Validated": bool(cpu.validated)
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


@app.get("/api/export/gpus/csv")
async def export_gpus_csv(db: Session = Depends(get_db)):
    """Export all GPUs as CSV file"""
    gpus = db.query(GPUSpec).all()

    df = pd.DataFrame([{
        "ID": gpu.id,
        "GPU Model Name": gpu.gpu_model_name,
        "Vendor": gpu.vendor or "",
        "GPU Model": gpu.gpu_model or "",
        "Form Factor": gpu.form_factor or "",
        "Memory (GB)": gpu.memory_gb or "",
        "Memory Type": gpu.memory_type or "",
        "TDP (W)": gpu.tdp_watts or "",
        "Validated": bool(gpu.validated),
    } for gpu in gpus])

    stream = io.StringIO()
    df.to_csv(stream, index=False, sep=';')
    csv_data = stream.getvalue()

    return StreamingResponse(
        io.BytesIO(csv_data.encode('utf-8')),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=gpu_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    )


@app.get("/api/export/gpus/excel")
async def export_gpus_excel(db: Session = Depends(get_db)):
    """Export all GPUs as Excel file"""
    gpus = db.query(GPUSpec).all()

    df = pd.DataFrame([{
        "ID": gpu.id,
        "GPU Model Name": gpu.gpu_model_name,
        "Vendor": gpu.vendor or "",
        "GPU Model": gpu.gpu_model or "",
        "Form Factor": gpu.form_factor or "",
        "Memory (GB)": gpu.memory_gb or "",
        "Memory Type": gpu.memory_type or "",
        "TDP (W)": gpu.tdp_watts or "",
        "Validated": bool(gpu.validated),
    } for gpu in gpus])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='GPU Specifications')

    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename=gpu_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        }
    )


def read_import_csv(source) -> pd.DataFrame:
    """Read an uploaded or repository CSV, detecting ; or , from the header line."""
    if isinstance(source, bytes):
        text = source.decode('utf-8-sig')
    else:
        with open(source, 'r', encoding='utf-8-sig', newline='') as file:
            text = file.read()
    delimiter = ';' if ';' in text.split('\n', 1)[0] else ','
    return pd.read_csv(io.StringIO(text), delimiter=delimiter)


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


@app.post("/api/import/csv-file")
@limiter.limit("10/minute")
async def import_csv_file(
    request: Request,
    file: UploadFile = File(...),
    clear_existing: bool = Query(False, description="Clear existing data before import"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Import CPUs from uploaded CSV file (requires authentication)
    
    CSV may be comma- or semicolon-delimited, matching cpu_spec_validated.csv format.
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
        df = read_import_csv(contents)
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
                max_memory_tb=clean_number(row.get('Max Memory (TB)')),
                validated=parse_bool(row.get('Validated'))
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


@app.post("/api/import/csv-repo")
@limiter.limit("10/minute")
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
        df = read_import_csv(csv_file_path)
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
                max_memory_tb=clean_number(row.get('Max Memory (TB)')),
                validated=parse_bool(row.get('Validated'))
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


REQUIRED_GPU_CSV_COLUMNS = [
    "GPU Model Name",
    "Vendor",
    "GPU Model",
    "Form Factor",
    "Memory (GB)",
    "Memory Type",
    "TDP (W)"
]


def validate_gpu_csv_columns(df: pd.DataFrame) -> None:
    """Ensure required GPU CSV columns exist after BOM cleanup."""
    missing = [col for col in REQUIRED_GPU_CSV_COLUMNS if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required GPU columns: {', '.join(missing)}"
        )


@app.post("/api/import/gpu-csv-file")
@limiter.limit("10/minute")
async def import_gpu_csv_file(
    request: Request,
    file: UploadFile = File(...),
    clear_existing: bool = Query(False, description="Clear existing GPU data before import"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Import GPUs from uploaded CSV file (requires authentication)"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")

    if clear_existing:
        db.query(GPUSpec).delete()
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
        df = read_import_csv(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    imported = 0
    errors = []

    df.columns = df.columns.str.replace('\ufeff', '')
    validate_gpu_csv_columns(df)

    for idx, row in df.iterrows():
        try:
            gpu_model_name = str(row.get('GPU Model Name', '')).strip()
            if not gpu_model_name:
                errors.append(f"Row {idx + 2}: Missing GPU Model Name")
                continue

            db_gpu = GPUSpec(
                gpu_model_name=gpu_model_name,
                vendor=str(row.get('Vendor', '')).strip() or None,
                gpu_model=str(row.get('GPU Model', '')).strip() or None,
                form_factor=str(row.get('Form Factor', '')).strip() or None,
                memory_gb=clean_number(row.get('Memory (GB)')),
                memory_type=str(row.get('Memory Type', '')).strip() or None,
                tdp_watts=clean_number(row.get('TDP (W)')),
                validated=parse_bool(row.get('Validated')),
            )
            db.add(db_gpu)
            imported += 1

        except Exception as e:
            # Avoid exposing internal exception details to the client.
            errors.append(f"Row {idx + 2}: Unable to import row due to invalid or missing data")

    db.commit()

    return {
        "message": f"Imported {imported} GPUs successfully",
        "imported": imported,
        "errors": errors[:10],
        "total_errors": len(errors)
    }


@app.post("/api/import/gpu-csv-repo")
@limiter.limit("10/minute")
async def import_gpu_csv_from_repo(
    request: Request,
    clear_existing: bool = Query(False, description="Clear existing GPU data before import"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Import GPUs from CSV file in repository (requires authentication)"""
    csv_file_path = "gpu_spec_validated.csv"

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
        db.query(GPUSpec).delete()
        db.commit()

    try:
        df = read_import_csv(csv_file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    imported = 0
    errors = []

    df.columns = df.columns.str.replace('\ufeff', '')
    validate_gpu_csv_columns(df)

    for idx, row in df.iterrows():
        try:
            gpu_model_name = str(row.get('GPU Model Name', '')).strip()
            if not gpu_model_name:
                errors.append(f"Row {idx + 2}: Missing GPU Model Name")
                continue

            db_gpu = GPUSpec(
                gpu_model_name=gpu_model_name,
                vendor=str(row.get('Vendor', '')).strip() or None,
                gpu_model=str(row.get('GPU Model', '')).strip() or None,
                form_factor=str(row.get('Form Factor', '')).strip() or None,
                memory_gb=clean_number(row.get('Memory (GB)')),
                memory_type=str(row.get('Memory Type', '')).strip() or None,
                tdp_watts=clean_number(row.get('TDP (W)')),
                validated=parse_bool(row.get('Validated')),
            )
            db.add(db_gpu)
            imported += 1

        except Exception as e:
            logger.exception("GPU CSV import failed for row %s", idx + 2)
            errors.append(f"Row {idx + 2}: Invalid or unsupported row data")

    db.commit()

    return {
        "message": f"Imported {imported} GPUs successfully from repository CSV",
        "imported": imported,
        "errors": errors[:10],
        "total_errors": len(errors),
        "source": csv_file_path
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
