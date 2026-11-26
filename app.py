"""FastAPI app: REST API, admin endpoints and public pages."""

from fastapi import FastAPI, Body, Depends, Query, HTTPException, UploadFile, File, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
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
from urllib.parse import urlsplit

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
from utils import determine_cpu_generation, model_slug
from llm import ask_question, LLMError, LLMRateLimitError, LLMTimeoutError
from import_data import import_cpu_text_to_db, import_gpu_text_to_db, parse_bool, gpu_values
import csv_store
import github_client
import pages
import proposals
import redirects

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
    """Import the CSVs if the database tables are empty."""
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


def load_redirects():
    db = SessionLocal()
    try:
        redirects.load_into_db(db)
    except Exception:
        logging.getLogger(__name__).exception("Could not load redirects")
    finally:
        db.close()

load_redirects()

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


SITE_URL = "https://computespecsdb.com"

# Public pages are cached briefly by browsers; admin and error pages are not cached.
PUBLIC_PAGE_CACHE = "public, max-age=300"


pages.templates.env.globals["asset_version"] = ASSET_VERSION


def serve_page(path: str, status_code: int = 200, cache_control: str = PUBLIC_PAGE_CACHE,
               fragments: Optional[dict] = None) -> HTMLResponse:
    """Serve a static HTML page with versioned CSS/JS links.

    Args:
        path: HTML file to serve.
        status_code: Response status.
        cache_control: Cache-Control header value.
        fragments: Placeholder comment -> HTML to insert.
    """
    with open(path, encoding="utf-8") as file:
        html = file.read()
    for placeholder, fragment in (fragments or {}).items():
        html = html.replace(placeholder, fragment)
    return HTMLResponse(
        _ASSET_URL.sub(rf'\1?v={ASSET_VERSION}"', html),
        status_code=status_code,
        headers={"Cache-Control": cache_control},
    )


# Machine-readable endpoints stay crawlable (pages fetch them while rendering)
# but should never show up as search results themselves.
NOINDEX_PREFIXES = ("/api", "/docs", "/redoc", "/openapi.json", "/admin")


def _permanent_slash_redirect(scope, message):
    """Turn Starlette's 307 trailing-slash redirect into a relative 308.

    Args:
        scope: ASGI scope of the request.
        message: The ``http.response.start`` message with status 307.

    Returns:
        The rewritten message, or ``message`` unchanged if it is not a slash redirect.
    """
    headers = message.get("headers", [])
    location = next((value.decode("latin-1") for key, value in headers if key.lower() == b"location"), None)
    if location is None:
        return message
    target = urlsplit(location)
    path = scope["path"]
    if target.path not in (path.rstrip("/"), path + "/"):
        return message
    relative = target.path + (f"?{target.query}" if target.query else "")
    headers = [(k, v) for k, v in headers if k.lower() != b"location"]
    headers.append((b"location", relative.encode("latin-1")))
    return dict(message, status=308, headers=headers)


class SeoHeadersMiddleware:
    """Answer HEAD like GET, add noindex to non-page URLs and fix slash redirects."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        is_head = scope["method"] == "HEAD"
        if is_head:
            scope = dict(scope, method="GET")
        noindex = scope["path"].startswith(NOINDEX_PREFIXES)

        async def wrapped_send(message):
            if message["type"] == "http.response.start" and message["status"] == 307:
                message = _permanent_slash_redirect(scope, message)
            if message["type"] == "http.response.start" and noindex:
                message = dict(message, headers=[*message.get("headers", []), (b"x-robots-tag", b"noindex")])
            elif message["type"] == "http.response.body" and is_head:
                message = dict(message, body=b"")
            await send(message)

        await self.app(scope, receive, wrapped_send)


app.add_middleware(SeoHeadersMiddleware)
app.include_router(pages.router)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(StarletteHTTPException)
async def not_found_as_html(request: Request, exc: StarletteHTTPException):
    """Return an HTML 404 for pages and JSON for API paths."""
    if exc.status_code == 404 and not request.url.path.startswith(NOINDEX_PREFIXES):
        return serve_page("static/404.html", status_code=404, cache_control="no-store")
    return await http_exception_handler(request, exc)


@app.get("/favicon.ico")
async def favicon():
    """Serve the site icon."""
    return FileResponse("static/images/favicon.ico", media_type="image/x-icon",
                        headers={"Cache-Control": "public, max-age=604800"})


@app.get("/robots.txt")
async def robots_txt():
    """Serve robots.txt."""
    content = f"""User-agent: *
Allow: /
Disallow: /admin
Disallow: /api/import/
Sitemap: {SITE_URL}/sitemap.xml
"""
    return Response(content=content, media_type="text/plain")


# Static pages listed in the sitemap, followed by every spec and overview page.
# API endpoints and /docs are left out on purpose: they are sent with X-Robots-Tag: noindex.
SITEMAP_PAGES = ("/", "/visualizations", "/propose")


@app.get("/sitemap.xml")
async def sitemap_xml(db: Session = Depends(get_db)):
    """Serve sitemap.xml."""
    paths = [*SITEMAP_PAGES, *pages.sitemap_paths(db)]
    urls = "".join(f"  <url><loc>{SITE_URL}{path}</loc></url>\n" for path in paths)
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}</urlset>\n"
    )
    return Response(content=content, media_type="application/xml")


class CPUSpecResponse(BaseModel):
    """CPU in API responses."""
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
    slug: str = Field(description="URL slug of the spec page: /cpu/{slug}")
    tdp_per_thread_w: Optional[float] = Field(None, description="Derived: TDP (W) / threads, rounded to 2 decimals")

    class Config:
        from_attributes = True


class GPUSpecResponse(BaseModel):
    """GPU in API responses."""
    id: int
    gpu_model_name: str
    vendor: Optional[str] = None
    gpu_model: Optional[str] = None
    form_factor: Optional[str] = None
    memory_gb: Optional[int] = None
    memory_type: Optional[str] = None
    tdp_watts: Optional[int] = None
    architecture: Optional[str] = None
    launch_year: Optional[int] = None
    memory_bandwidth_tbs: Optional[float] = None
    fp64_tflops: Optional[float] = None
    fp32_tflops: Optional[float] = None
    fp16_bf16_tensor_tflops: Optional[float] = None
    fp8_tensor_tflops: Optional[float] = None
    fp4_tensor_tflops: Optional[float] = None
    interconnect: Optional[str] = None
    interconnect_bandwidth_gbs: Optional[float] = None
    pcie_interface: Optional[str] = None
    validated: bool = False
    slug: str = Field(description="URL slug of the spec page: /gpu/{slug}")

    class Config:
        from_attributes = True


@app.get("/", response_class=HTMLResponse)
async def root(db: Session = Depends(get_db)):
    """Serve the homepage."""
    return serve_page("static/index.html", fragments=pages.render_home_fragments(db))


@app.get("/visualizations", response_class=HTMLResponse)
async def visualizations():
    """Serve the visualizations page."""
    return serve_page("static/visualizations.html")


@app.get("/propose", response_class=HTMLResponse)
async def propose_page():
    """Serve the proposal form."""
    return serve_page("static/propose.html")


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel():
    """Serve the admin panel (disabled in production unless ENABLE_ADMIN_UI=true)."""
    if ENVIRONMENT == "production" and not ENABLE_ADMIN_UI:
        raise HTTPException(status_code=404, detail="Not found")

    return serve_page("static/admin.html", cache_control="no-store")


@app.get("/api", response_class=JSONResponse)
async def api_info():
    """List the API endpoints."""
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
            "report_cpu_error": "POST /api/reports/cpu/{id}",
            "report_gpu_error": "POST /api/reports/gpu/{id}",
            "docs": "/docs"
        },
        "notes": {
            "validated": (
                "Every CPU/GPU has a boolean 'validated' field: true means the specs "
                "were manually checked against official sources. Filter with ?validated=true|false."
            ),
            "tdp_per_thread_w": "Derived CPU field: TDP (W) divided by threads. Not stored in the CSV.",
        }
    }


@app.get("/api/cpus", response_model=List[CPUSpecResponse])
async def get_all_cpus(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    validated: Optional[bool] = Query(None, description="Only return validated (true) or unvalidated (false) entries"),
    db: Session = Depends(get_db)
):
    """List CPUs."""
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
    """Search CPUs by name, family, model or codename."""
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
    """Get a CPU by ID."""
    cpu = db.query(CPUSpec).filter(CPUSpec.id == cpu_id).first()

    if cpu is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"CPU with ID {cpu_id} not found"}
        )

    return cpu


@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get database statistics."""
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
    """Question for the Ask feature."""
    question: str


class AskResponse(BaseModel):
    """Answer from the Ask feature."""
    answer: str
    sources: List[dict] = []


@app.post("/api/ask", response_model=AskResponse)
@limiter.limit("10/minute")
async def ask(
    request: Request,
    payload: AskRequest,
    db: Session = Depends(get_db),
):
    """Answer a natural-language question about the specs (needs ENABLE_ASK_FEATURE)."""
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
    # 503 rather than 502/504: Cloudflare replaces 502 and 504 responses with its own
    # plain "error code" page, which would hide these messages from the chat window.
    except LLMTimeoutError:
        raise HTTPException(
            status_code=503,
            detail="The request timed out. Please try a simpler question.",
        )
    except LLMError as exc:
        logger.exception("Ask feature error")
        raise HTTPException(status_code=503, detail=str(exc))

    return AskResponse(answer=result["answer"], sources=result.get("sources", []))


@app.get("/api/gpus", response_model=List[GPUSpecResponse])
async def get_all_gpus(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    validated: Optional[bool] = Query(None, description="Only return validated (true) or unvalidated (false) entries"),
    db: Session = Depends(get_db)
):
    """List GPUs."""
    query = db.query(GPUSpec)
    if validated is not None:
        query = query.filter(GPUSpec.validated == validated)
    gpus = query.offset(skip).limit(limit).all()
    return gpus


@app.get("/api/gpus/search", response_model=List[GPUSpecResponse])
async def search_gpus(
    q: str = Query(..., description="Search query (searches in model name, vendor, GPU model, form factor, memory type, and architecture)"),
    validated: Optional[bool] = Query(None, description="Only return validated (true) or unvalidated (false) entries"),
    db: Session = Depends(get_db)
):
    """Search GPUs by name, vendor, model, form factor, memory type or architecture."""
    search_filter = or_(
        GPUSpec.gpu_model_name.ilike(f"%{q}%"),
        GPUSpec.vendor.ilike(f"%{q}%"),
        GPUSpec.gpu_model.ilike(f"%{q}%"),
        GPUSpec.form_factor.ilike(f"%{q}%"),
        GPUSpec.memory_type.ilike(f"%{q}%"),
        GPUSpec.architecture.ilike(f"%{q}%")
    )
    query = db.query(GPUSpec).filter(search_filter)
    if validated is not None:
        query = query.filter(GPUSpec.validated == validated)
    return query.all()


@app.get("/api/gpus/{gpu_id}", response_model=GPUSpecResponse)
async def get_gpu_by_id(gpu_id: int, db: Session = Depends(get_db)):
    """Get a GPU by ID."""
    gpu = db.query(GPUSpec).filter(GPUSpec.id == gpu_id).first()
    if gpu is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"GPU with ID {gpu_id} not found"}
        )
    return gpu


class LoginRequest(BaseModel):
    """Admin login."""
    password: str


@app.post("/api/auth/login")
@limiter.limit("5/minute")
async def login(request: Request, payload: LoginRequest):
    """Exchange the admin password for a JWT."""
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
    """Get the current user and GitHub sync status."""
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
    """New CPU."""
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
    """CPU fields to change."""
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
    """New GPU."""
    gpu_model_name: str
    vendor: Optional[str] = None
    gpu_model: Optional[str] = None
    form_factor: Optional[str] = None
    memory_gb: Optional[int] = None
    memory_type: Optional[str] = None
    tdp_watts: Optional[int] = None
    architecture: Optional[str] = None
    launch_year: Optional[int] = None
    memory_bandwidth_tbs: Optional[float] = None
    fp64_tflops: Optional[float] = None
    fp32_tflops: Optional[float] = None
    fp16_bf16_tensor_tflops: Optional[float] = None
    fp8_tensor_tflops: Optional[float] = None
    fp4_tensor_tflops: Optional[float] = None
    interconnect: Optional[str] = None
    interconnect_bandwidth_gbs: Optional[float] = None
    pcie_interface: Optional[str] = None
    validated: bool = False


class GPUSpecUpdate(BaseModel):
    """GPU fields to change."""
    gpu_model_name: Optional[str] = None
    vendor: Optional[str] = None
    gpu_model: Optional[str] = None
    form_factor: Optional[str] = None
    memory_gb: Optional[int] = None
    memory_type: Optional[str] = None
    tdp_watts: Optional[int] = None
    architecture: Optional[str] = None
    launch_year: Optional[int] = None
    memory_bandwidth_tbs: Optional[float] = None
    fp64_tflops: Optional[float] = None
    fp32_tflops: Optional[float] = None
    fp16_bf16_tensor_tflops: Optional[float] = None
    fp8_tensor_tflops: Optional[float] = None
    fp4_tensor_tflops: Optional[float] = None
    interconnect: Optional[str] = None
    interconnect_bandwidth_gbs: Optional[float] = None
    pcie_interface: Optional[str] = None
    validated: Optional[bool] = None


class ValidatedUpdate(BaseModel):
    """New value of the validated flag."""
    validated: bool


class CPUBatchItem(BaseModel):
    """One CPU change in a batch."""
    id: int
    changes: CPUSpecUpdate


class CPUBatchUpdate(BaseModel):
    """CPU changes saved as one commit."""
    items: List[CPUBatchItem] = Field(..., min_length=1)


class GPUBatchItem(BaseModel):
    """One GPU change in a batch."""
    id: int
    changes: GPUSpecUpdate


class GPUBatchUpdate(BaseModel):
    """GPU changes saved as one commit."""
    items: List[GPUBatchItem] = Field(..., min_length=1)


# ---------- Write-through helpers ----------
# Every admin change is committed to the CSV on GitHub first and only then
# applied to SQLite, so the two never diverge. Without GITHUB_TOKEN (local
# development) changes only touch the database.

SPEC_MODELS = {"cpu": CPUSpec, "gpu": GPUSpec}
SPEC_CREATE_MODELS = {"cpu": CPUSpecCreate, "gpu": GPUSpecCreate}


def _spec_label(kind: str) -> str:
    """Return ``'CPU'`` or ``'GPU'``."""
    return csv_store.SCHEMAS[kind]["label"]


def _find_by_name(db: Session, kind: str, name: str, exclude_id: Optional[int] = None):
    """Return an entry with the same name or slug as ``name``, or None."""
    model = SPEC_MODELS[kind]
    column = getattr(model, csv_store.SCHEMAS[kind]["key_attr"])
    query = db.query(model)
    if exclude_id is not None:
        query = query.filter(model.id != exclude_id)
    same_name = query.filter(func.lower(column) == name.strip().lower()).first()
    if same_name is not None:
        return same_name
    slug = model_slug(name)
    return next((item for item in query if item.slug == slug), None)


def _sync_csv(action):
    """Run a CSV commit, converting failures to HTTP errors.

    Does nothing if GitHub sync is off.

    Raises:
        HTTPException: 409 for a duplicate row, 502 if GitHub fails.
    """
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
    """Collapse whitespace in a model name.

    Raises:
        HTTPException: 400 if the name is empty.
    """
    name = " ".join((name or "").split())
    if not name:
        raise HTTPException(status_code=400, detail=f"{_spec_label(kind)} model name cannot be empty")
    return name


def _create_spec(db: Session, kind: str, values: dict, message: str):
    """Commit a new row to GitHub, then add it to the database.

    Returns:
        Tuple of (item, commit_url).

    Raises:
        HTTPException: 409 if the name exists, or an error from ``_sync_csv``.
    """
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
    """Return the row with ``item_id``, or raise a 404."""
    item = db.query(SPEC_MODELS[kind]).filter(SPEC_MODELS[kind].id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail=f"{_spec_label(kind)} with ID {item_id} not found")
    return item


def _prepare_update(db: Session, kind: str, item_id: int, changes: dict):
    """Validate one update without saving it.

    Returns:
        Tuple of (item, old_name, values, changes): the ORM row, its current name,
        the full row for the CSV, and the cleaned changes.

    Raises:
        HTTPException: 404 if the item is missing, 400/409 for a bad name.
    """
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
    return item, old_name, values, changes


def _update_message(kind: str, old_name: str, values: dict, changes: dict) -> str:
    """Commit message for a single update."""
    label = _spec_label(kind)
    new_name = values[csv_store.SCHEMAS[kind]["key_attr"]]
    if set(changes) == {"validated"}:
        return f"Mark {label} {'validated' if values['validated'] else 'unvalidated'}: {new_name}"
    if new_name != old_name:
        return f"Rename {label}: {old_name} -> {new_name}"
    return f"Update {label}: {new_name}"


def _update_spec(db: Session, kind: str, item_id: int, changes: dict):
    """Commit one update to GitHub, then apply it to the database."""
    item, old_name, values, changes = _prepare_update(db, kind, item_id, changes)
    message = _update_message(kind, old_name, values, changes)

    _sync_csv(lambda: csv_store.commit_upsert(kind, old_name, values, message))

    for field_name, value in changes.items():
        setattr(item, field_name, value)
    db.commit()
    db.refresh(item)
    new_name = values[csv_store.SCHEMAS[kind]["key_attr"]]
    if new_name != old_name:
        redirects.record_rename(db, kind, old_name, new_name)
    return item


BATCH_MAX_ITEMS = 500


def _batch_message(kind: str, prepared: list) -> str:
    """Commit message for a batch: a summary line, then one line per item.

    Args:
        kind: "cpu" or "gpu".
        prepared: Tuples returned by ``_prepare_update``.
    """
    label = _spec_label(kind)
    count = len(prepared)
    flags = {
        values["validated"] for _, _, values, changes in prepared if set(changes) == {"validated"}
    }
    validated_only = all(set(changes) == {"validated"} for _, _, _, changes in prepared)
    if validated_only and len(flags) == 1:
        subject = f"Mark {count} {label}s {'validated' if flags.pop() else 'unvalidated'}"
    elif validated_only:
        subject = f"Update validated flag on {count} {label}s"
    else:
        subject = f"Update {count} {label}s"
    lines = [f"- {_update_message(kind, old_name, values, changes)}" for _, old_name, values, changes in prepared]
    return subject + "\n\n" + "\n".join(lines)


def _update_specs_batch(db: Session, kind: str, updates: list) -> list:
    """Save several updates as one GitHub commit, then apply them to the database.

    Args:
        db: Database session.
        kind: "cpu" or "gpu".
        updates: Batch items with ``id`` and ``changes``. Repeated ids are merged.

    Returns:
        The updated ORM rows.

    Raises:
        HTTPException: 400 for an empty or oversized batch, 409 for name clashes,
            502 if GitHub rejects the commit (the database is then unchanged).
    """
    merged: dict[int, dict] = {}
    for update in updates:
        merged.setdefault(update.id, {}).update(update.changes.model_dump(exclude_unset=True))
    if not merged:
        raise HTTPException(status_code=400, detail="No changes to save")
    if len(merged) > BATCH_MAX_ITEMS:
        raise HTTPException(status_code=400, detail=f"At most {BATCH_MAX_ITEMS} items per batch")

    key = csv_store.SCHEMAS[kind]["key_attr"]
    prepared = [_prepare_update(db, kind, item_id, changes) for item_id, changes in merged.items()]
    prepared = [entry for entry in prepared if entry[3]]
    if not prepared:
        raise HTTPException(status_code=400, detail="No changes to save")

    new_names = [values[key].casefold() for _, _, values, _ in prepared]
    if len(set(new_names)) != len(new_names):
        raise HTTPException(status_code=409, detail="Two items in the batch would get the same name")

    message = _batch_message(kind, prepared)
    _sync_csv(lambda: csv_store.commit_upsert_many(
        kind, [(old_name, values) for _, old_name, values, _ in prepared], message,
    ))

    for item, _, _, changes in prepared:
        for field_name, value in changes.items():
            setattr(item, field_name, value)
    db.commit()
    items = [entry[0] for entry in prepared]
    for item in items:
        db.refresh(item)
    for _, old_name, values, _ in prepared:
        if values[key] != old_name:
            redirects.record_rename(db, kind, old_name, values[key])
    return items


def _delete_spec(db: Session, kind: str, item_id: int) -> None:
    """Remove a row from the CSV on GitHub, then from the database."""
    item = _get_spec_or_404(db, kind, item_id)
    name = getattr(item, csv_store.SCHEMAS[kind]["key_attr"])
    _sync_csv(lambda: csv_store.commit_remove(kind, name, f"Remove {_spec_label(kind)}: {name}"))
    db.delete(item)
    db.commit()


def _cpu_create_values(cpu: CPUSpecCreate) -> dict:
    """Return the values for a new CPU, filling in the codename if possible."""
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
    """Create a CPU (admin)."""
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
    """Update a CPU (admin)."""
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
    """Mark a CPU validated or unvalidated (admin)."""
    return _update_spec(db, "cpu", cpu_id, {"validated": payload.validated})


@app.post("/api/cpus/batch", response_model=List[CPUSpecResponse])
@limiter.limit("30/minute")
def update_cpus_batch(
    request: Request,
    payload: CPUBatchUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update several CPUs in one commit (admin)."""
    return _update_specs_batch(db, "cpu", payload.items)


@app.delete("/api/cpus/{cpu_id}", status_code=204)
@limiter.limit("30/minute")
def delete_cpu(
    request: Request,
    cpu_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a CPU (admin)."""
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
    """Create a GPU (admin)."""
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
    """Update a GPU (admin)."""
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
    """Mark a GPU validated or unvalidated (admin)."""
    return _update_spec(db, "gpu", gpu_id, {"validated": payload.validated})


@app.post("/api/gpus/batch", response_model=List[GPUSpecResponse])
@limiter.limit("30/minute")
def update_gpus_batch(
    request: Request,
    payload: GPUBatchUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update several GPUs in one commit (admin)."""
    return _update_specs_batch(db, "gpu", payload.items)


@app.delete("/api/gpus/{gpu_id}", status_code=204)
@limiter.limit("30/minute")
def delete_gpu(
    request: Request,
    gpu_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a GPU (admin)."""
    _delete_spec(db, "gpu", gpu_id)
    return None


# ---------- Public proposals (stored as GitHub issues) ----------

PROPOSAL_MAX_TEXT = 200


class _ProposalExtras(BaseModel):
    """Fields shared by public proposals."""
    source_url: HttpUrl
    notes: Optional[str] = Field(None, max_length=1000)
    # Honeypot: hidden in the form, so only bots fill it in.
    website: Optional[str] = None


class CPUProposal(_ProposalExtras, CPUSpecCreate):
    """Proposed CPU."""


class GPUProposal(_ProposalExtras, GPUSpecCreate):
    """Proposed GPU."""


class RejectRequest(BaseModel):
    """Reason for rejecting a proposal."""
    reason: Optional[str] = Field(None, max_length=2000)


def _require_github():
    """Raise a 503 if GitHub sync is not configured."""
    if not github_client.is_configured():
        raise HTTPException(status_code=503, detail="Proposals are not configured on this server")


def _submit_proposal(kind: str, payload: _ProposalExtras, db: Session) -> dict:
    """Validate a proposal and open a GitHub issue for it.

    Returns:
        Dict with ``issue_number`` and ``issue_url``.

    Raises:
        HTTPException: 400 for bots, 409 if the model exists, 422/502/503 otherwise.
    """
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
    """Propose a new CPU; opens a GitHub issue."""
    return _submit_proposal("cpu", payload, db)


@app.post("/api/proposals/gpu", status_code=201)
@limiter.limit("3/minute")
def propose_gpu(request: Request, payload: GPUProposal, db: Session = Depends(get_db)):
    """Propose a new GPU; opens a GitHub issue."""
    return _submit_proposal("gpu", payload, db)


# ---------- Public data-error reports (stored as GitHub issues) ----------

class DataErrorReport(BaseModel):
    """Report of a wrong value on a spec page."""
    field: str = Field(max_length=40, description="Attribute name of the wrong field, or 'other'")
    suggested_value: str = Field(min_length=1, max_length=PROPOSAL_MAX_TEXT, description="The correct value")
    source_url: Optional[HttpUrl] = None
    notes: Optional[str] = Field(None, max_length=1000)
    # Honeypot: hidden in the form, so only bots fill it in.
    website: Optional[str] = None


def _submit_report(kind: str, item_id: int, payload: DataErrorReport, db: Session) -> dict:
    """Validate a data-error report and open a GitHub issue for it.

    Returns:
        Dict with ``issue_number`` and ``issue_url``.
    """
    if payload.website:
        raise HTTPException(status_code=400, detail="Invalid submission")
    item = _get_spec_or_404(db, kind, item_id)
    labels = dict(proposals.report_fields(kind))
    if payload.field not in labels:
        raise HTTPException(status_code=422, detail="Please choose which field is wrong")
    suggested = " ".join(payload.suggested_value.split())
    if not suggested:
        raise HTTPException(status_code=422, detail="Please enter the correct value")
    _require_github()

    current = getattr(item, payload.field) if payload.field != proposals.REPORT_OTHER_FIELD else None
    title, body = proposals.render_report(
        kind,
        getattr(item, csv_store.SCHEMAS[kind]["key_attr"]),
        f"{SITE_URL}/{kind}/{item.slug}",
        payload.field,
        labels[payload.field],
        current,
        suggested,
        str(payload.source_url) if payload.source_url else None,
        (payload.notes or "").strip() or None,
    )
    try:
        issue = github_client.create_issue(title, body, [proposals.REPORT_LABEL, kind])
    except github_client.GitHubError:
        logger.exception("Could not create data-error issue")
        raise HTTPException(status_code=502, detail="Could not send the report right now. Please try again later.")

    return {"issue_number": issue["number"], "issue_url": issue["html_url"]}


@app.post("/api/reports/cpu/{cpu_id}", status_code=201)
@limiter.limit("3/minute;30/day")
def report_cpu_error(request: Request, cpu_id: int, payload: DataErrorReport, db: Session = Depends(get_db)):
    """Report a wrong CPU value; opens a GitHub issue."""
    return _submit_report("cpu", cpu_id, payload, db)


@app.post("/api/reports/gpu/{gpu_id}", status_code=201)
@limiter.limit("3/minute;30/day")
def report_gpu_error(request: Request, gpu_id: int, payload: DataErrorReport, db: Session = Depends(get_db)):
    """Report a wrong GPU value; opens a GitHub issue."""
    return _submit_report("gpu", gpu_id, payload, db)


# ---------- Reviewing data-error reports (admin) ----------

class ApplyReportRequest(BaseModel):
    """The value to store, as edited by the admin (defaults to the reported value)."""
    value: Optional[str] = Field(None, max_length=PROPOSAL_MAX_TEXT)


class CloseReportRequest(BaseModel):
    fixed: bool = False
    message: Optional[str] = Field(None, max_length=2000)


SPEC_UPDATE_MODELS = {"cpu": CPUSpecUpdate, "gpu": GPUSpecUpdate}


def _find_reported_item(db: Session, kind: str, name: str):
    model = SPEC_MODELS[kind]
    column = getattr(model, csv_store.SCHEMAS[kind]["key_attr"])
    return db.query(model).filter(func.lower(column) == name.strip().lower()).first()


def _load_open_report(number: int) -> tuple[dict, Optional[dict]]:
    try:
        issue = github_client.get_issue(number)
    except github_client.GitHubError as exc:
        status = 404 if exc.status == 404 else 502
        raise HTTPException(status_code=status, detail=f"Could not load issue #{number}: {exc}")
    if not proposals.is_open_report(issue):
        raise HTTPException(status_code=409, detail=f"Issue #{number} is not an open data-error report")
    return issue, proposals.parse_report(issue)


def _close_report(number: int, message: str, label: str, reason: str) -> None:
    try:
        github_client.comment(number, message)
        github_client.close_issue(number, [label], reason=reason)
    except github_client.GitHubError as exc:
        raise HTTPException(status_code=502, detail=f"Could not update issue #{number}: {exc}")


@app.get("/api/reports")
def list_reports(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """List open data-error reports, with each entry's current value (requires authentication)"""
    _require_github()
    try:
        issues = github_client.list_issues([proposals.REPORT_LABEL])
    except github_client.GitHubError as exc:
        raise HTTPException(status_code=502, detail=f"Could not load reports from GitHub: {exc}")

    result = []
    for issue in sorted(issues, key=lambda i: i["number"]):
        payload = proposals.parse_report(issue)
        entry = {
            "number": issue["number"], "title": issue["title"], "url": issue["html_url"],
            "created_at": issue.get("created_at"), "parsed": payload is not None,
        }
        if payload:
            item = _find_reported_item(db, payload["kind"], payload["name"])
            field = payload["field"]
            entry.update(
                kind=payload["kind"], name=payload["name"], field=field,
                field_label=payload.get("field_label") or field,
                reported_current=payload.get("current"), suggested=payload.get("suggested"),
                source_url=payload.get("source_url"), notes=payload.get("notes"),
                item_id=item.id if item else None,
                page_url=f"/{payload['kind']}/{item.slug}" if item else None,
                live_value=getattr(item, field, None) if item and field != proposals.REPORT_OTHER_FIELD else None,
                single_field=field != proposals.REPORT_OTHER_FIELD,
            )
        result.append(entry)
    return result


@app.post("/api/reports/{number}/apply")
@limiter.limit("30/minute")
def apply_report(
    request: Request,
    number: int,
    payload: ApplyReportRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Apply a reported fix to the entry (CSV on GitHub and database), then close the issue."""
    _require_github()
    _, report = _load_open_report(number)
    if report is None or report["field"] == proposals.REPORT_OTHER_FIELD:
        raise HTTPException(status_code=409, detail="This report has no single field to apply; edit the entry instead")
    kind, field = report["kind"], report["field"]
    item = _find_reported_item(db, kind, report["name"])
    if item is None:
        raise HTTPException(status_code=404, detail=f"'{report['name']}' is no longer in the database")

    raw = payload.value if payload.value is not None else report.get("suggested")
    raw = " ".join(str(raw or "").split())
    try:
        changes = SPEC_UPDATE_MODELS[kind](**{field: raw or None}).model_dump(exclude_unset=True)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"'{raw}' is not a valid value: {exc.errors()[0]['msg']}")

    old_value = getattr(item, field)
    item = _update_spec(db, kind, item.id, changes)
    new_value = getattr(item, field)
    label = report.get("field_label") or field
    _close_report(
        number,
        f"Thanks for the report! **{label}** was changed from `{old_value}` to `{new_value}`.",
        proposals.FIXED_LABEL, "completed",
    )
    return {"number": number, "field": field, "old_value": old_value, "new_value": new_value,
            "name": getattr(item, csv_store.SCHEMAS[kind]["key_attr"])}


@app.post("/api/reports/{number}/close")
@limiter.limit("30/minute")
def close_report(
    request: Request,
    number: int,
    payload: CloseReportRequest,
    current_user: dict = Depends(get_current_user)
):
    """Close a report without changing data: as already fixed, or dismissed with a reason."""
    _require_github()
    _load_open_report(number)
    note = (payload.message or "").strip()
    if payload.fixed:
        message = "Thanks for the report! The entry has been corrected."
    else:
        message = "Thanks for the report! After checking the sources, the entry was left unchanged."
    if note:
        message += f"\n\n{note}"
    _close_report(number, message,
                  proposals.FIXED_LABEL if payload.fixed else proposals.DISMISSED_LABEL,
                  "completed" if payload.fixed else "not_planned")
    return {"number": number, "closed": True, "fixed": payload.fixed}


def _load_open_proposal(number: int) -> tuple[dict, str, Optional[dict]]:
    """Fetch an open proposal issue.

    Returns:
        Tuple of (issue, kind, payload); payload is None if the issue has no data block.

    Raises:
        HTTPException: 404/502 if it cannot be loaded, 409 if it is not an open proposal.
    """
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
    """List open proposals (admin)."""
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
    """Accept a proposal: add it to the CSV and database, then close the issue (admin)."""
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
    """Reject a proposal and close the issue with a comment (admin)."""
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
    """Download all CPUs as CSV."""
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
    """Download all CPUs as Excel."""
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


def gpu_export_row(gpu: GPUSpec) -> dict:
    """Return a GPU's CSV columns (as in gpu_spec_validated.csv) for exports."""
    row = {}
    for attr, header in csv_store.SCHEMAS["gpu"]["fields"]:
        value = getattr(gpu, attr)
        row[header] = bool(value) if attr == "validated" else ("" if value is None else value)
    return row


@app.get("/api/export/gpus/csv")
async def export_gpus_csv(db: Session = Depends(get_db)):
    """Download all GPUs as CSV."""
    gpus = db.query(GPUSpec).all()

    df = pd.DataFrame([{
        "ID": gpu.id,
        **gpu_export_row(gpu),
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
    """Download all GPUs as Excel."""
    gpus = db.query(GPUSpec).all()

    df = pd.DataFrame([{
        "ID": gpu.id,
        **gpu_export_row(gpu),
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
    """Read a CSV into a DataFrame, detecting ``;`` or ``,``."""
    if isinstance(source, bytes):
        text = source.decode('utf-8-sig')
    else:
        with open(source, 'r', encoding='utf-8-sig', newline='') as file:
            text = file.read()
    delimiter = ';' if ';' in text.split('\n', 1)[0] else ','
    return pd.read_csv(io.StringIO(text), delimiter=delimiter)


def clean_number(value, default=None):
    """Parse a CSV number, accepting a comma as decimal separator."""
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
    """Check that the CPU CSV has the required columns.

    Raises:
        HTTPException: 400 if columns are missing.
    """
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
    """Import CPUs from an uploaded CSV; database only (admin)."""
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
    """Import CPUs from the bundled CSV; database only (admin)."""
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
    """Check that the GPU CSV has the required columns.

    Raises:
        HTTPException: 400 if columns are missing.
    """
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
    """Import GPUs from an uploaded CSV; database only (admin)."""
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

            values = gpu_values(
                lambda header: None if pd.isna(row.get(header)) else row.get(header),
                number=clean_number,
            )
            values["gpu_model_name"] = gpu_model_name
            db_gpu = GPUSpec(**values)
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
    """Import GPUs from the bundled CSV; database only (admin)."""
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

            values = gpu_values(
                lambda header: None if pd.isna(row.get(header)) else row.get(header),
                number=clean_number,
            )
            values["gpu_model_name"] = gpu_model_name
            db_gpu = GPUSpec(**values)
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
