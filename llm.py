"""Answer natural-language questions with Groq tool calls over the database."""

import json
import logging
import os
import re
import time
from typing import Any

from groq import Groq
from sqlalchemy import or_, text
from sqlalchemy.orm import Session

import content
from database import CPUSpec, GPUSpec
from utils import slugify

logger = logging.getLogger(__name__)

# Groq retires models regularly (llama-3.3-70b-versatile was shut down on 2026-08-16),
# so a retired or unknown model falls through to the next one in this list.
# GROQ_MODEL (env) is tried first when set.
DEFAULT_GROQ_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")
# gpt-oss models "think" before answering; spec lookups need little of that, and the
# default (medium) effort made answers take 15+ seconds.
REASONING_EFFORT = os.environ.get("GROQ_REASONING_EFFORT", "low")
MAX_COMPLETION_TOKENS = 1500
# Models in Groq's list that are not chat models (speech, safety classifiers).
NON_CHAT_MODEL_MARKERS = ("whisper", "orpheus", "guard", "safeguard", "tts", "distil")
GROQ_MODELS = tuple(dict.fromkeys(
    m for m in (os.environ.get("GROQ_MODEL", "").strip(), *DEFAULT_GROQ_MODELS) if m
))
GROQ_MODEL = GROQ_MODELS[0]
MAX_TOOL_ROUNDS = 3
MAX_SQL_ROWS = 50
MAX_SEARCH_RESULTS = 10

SCHEMA_PROMPT = """
Database schema (SQLite):

Table cpu_specs:
  id, cpu_model_name, family, cpu_model, codename, cores, threads, tdp_watts,
  launch_year, max_turbo_frequency_ghz, l3_cache_mb, max_memory_tb, validated

  family values are full product lines, NOT short vendor names. Examples:
    AMD: 'AMD EPYC', 'AMD Opteron', 'AMD Ryzen'
    Intel: 'Intel Xeon', 'Intel Xeon Platinum', 'Intel Xeon Gold', 'Intel Core', etc.
  To filter by vendor, use: family LIKE '%AMD%' or family LIKE '%Intel%'
  Do NOT use family = 'AMD' or family = 'Intel' — those will match zero rows.
  Date filtering uses launch_year (integer), e.g. launch_year BETWEEN 2020 AND 2025

Table gpu_specs:
  id, gpu_model_name, vendor, gpu_model, architecture, launch_year, form_factor, memory_gb,
  memory_type, memory_bandwidth_tbs, tdp_watts, fp64_tflops, fp32_tflops,
  fp16_bf16_tensor_tflops, fp8_tensor_tflops, fp4_tensor_tflops, interconnect,
  interconnect_bandwidth_gbs, pcie_interface, validated

  vendor examples: 'NVIDIA', 'AMD', 'Intel'
  architecture examples: 'Rubin', 'Blackwell Ultra', 'Blackwell', 'Hopper', 'Ada Lovelace', 'Ampere', 'Volta', 'CDNA 3'
  Throughput columns are TFLOPS per GPU (per superchip for GB200/GB300/GH200 rows):
  fp64/fp32 are vector (non-tensor); tensor columns are dense (without sparsity).
  fp16_bf16_tensor_tflops is FP16 for GPUs without BF16 (Volta, Turing).
  memory_bandwidth_tbs is TB/s; interconnect_bandwidth_gbs is GPU-to-GPU GB/s (NVLink, Infinity Fabric).

validated (both tables) is 1 when the specs were manually checked against official
sources, 0 when not yet verified. Filter with validated = 1 / validated = 0.

Derived metrics (use in run_sql when cores > 0):
  tdp_per_core = tdp_watts * 1.0 / cores
  threads_per_core = threads * 1.0 / cores
""".strip()

SYSTEM_PROMPT = f"""You are a helpful assistant for Compute Specs DB, a catalog of HPC and datacenter CPU and GPU specifications.

{SCHEMA_PROMPT}

About the website (computespecsdb.com), for questions about the site itself:
- Every model has a spec page: CPUs at /cpu/<slug>, GPUs at /gpu/<slug>. Tool results include each
  model's "url"; link to it when you mention a model.
- Overview pages: /cpus (all CPUs), /cpus/<family-slug> (e.g. /cpus/amd-epyc), /cpus/codename/<codename-slug>
  (e.g. /cpus/codename/genoa, with a written intro, platform facts and an FAQ), /gpus and /gpus/<vendor-slug>.
- /compare lists hand-picked side-by-side comparisons; the interactive tool on /visualizations
  (section "Comparison Tool") compares any two models and can export CSV, PNG and a citation.
- /about explains sources and validation. Specs come from manufacturers' official pages (Intel ARK, AMD,
  NVIDIA, Ampere); generation codenames were confirmed with TechPowerUp's CPU database.
- Wrong value? Use "Report an error" on the model's spec page (no account needed). Missing model? /propose.
- Data downloads: CSV/Excel exports and a JSON API (/api/cpus, /api/gpus, docs at /docs). The data is free.

Rules:
- Answer ONLY using data returned by your tools. Never invent specifications.
- For questions about a CPU generation/codename (socket, memory, launch, what it is), call
  get_generation_info as well as searching the CPUs.
- If data is missing, say you do not have that information in the database.
- Cite exact model names from the database when possible.
- Do not guess prices, availability, or benchmark scores — they are not in the database.
- In run_sql, always SELECT the id and the model name column too, so answers can link to the model pages.
- For ratios and rankings across many rows, use run_sql. For a single CPU/GPU after search, you may compute simple arithmetic.
- When filtering CPUs by AMD or Intel, always use family LIKE '%AMD%' or family LIKE '%Intel%'.
- If you cite a row whose validated value is false/0, mention that its specs are not yet verified.
- For off-topic questions, politely redirect to hardware specs in the database.
- This is a specs catalog, not buying advice. Keep answers concise and factual.
- Format: short sentences or a simple "-" bullet list. No tables or headings. When you mention a model or
  page, write its link as a markdown link with the site path, e.g. [AMD EPYC 9654](/cpu/amd-epyc-9654).
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_cpus",
            "description": "Search CPUs by model name, family, CPU model, or codename.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "Search query, e.g. EPYC, Milan, Xeon Platinum",
                    }
                },
                "required": ["q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_gpus",
            "description": "Search GPUs by model name, vendor, GPU model, form factor, or memory type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "Search query, e.g. H100, NVIDIA, HBM3",
                    }
                },
                "required": ["q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_generation_info",
            "description": (
                "Get the written overview of a CPU generation by codename (e.g. Genoa, Sapphire Rapids, Turin): "
                "launch, platform facts such as socket and memory, and its page URL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "codename": {"type": "string", "description": "Codename, e.g. Genoa or Granite Rapids"}
                },
                "required": ["codename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stats",
            "description": "Get summary statistics: total and validated CPUs/GPUs, families, codenames, averages, max cores, year range, etc.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Run a read-only SELECT query on cpu_specs and/or gpu_specs. "
                "Use for rankings, aggregations, and calculated fields like tdp_per_core."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A single SELECT SQL statement.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)

# Some Groq/Llama models emit tool calls as text instead of structured tool_calls.
_TEXT_TOOL_CALL = re.compile(
    r"<function/(\w+)\s*(\{.*?\})\s*</function>",
    re.DOTALL | re.IGNORECASE,
)
_TEXT_TOOL_CALL_ALT = re.compile(
    r"<function=(\w+)>\s*(\{.*?\})\s*</function>",
    re.DOTALL | re.IGNORECASE,
)


def _get_live_schema_context(db: Session) -> str:
    """Describe the actual column values, so the model does not guess filters."""
    families = [
        row[0]
        for row in db.query(CPUSpec.family).distinct().order_by(CPUSpec.family).all()
        if row[0]
    ]
    vendors = [
        row[0]
        for row in db.query(GPUSpec.vendor).distinct().order_by(GPUSpec.vendor).all()
        if row[0]
    ]
    codenames = [
        row[0]
        for row in db.query(CPUSpec.codename).distinct().order_by(CPUSpec.codename).all()
        if row[0]
    ]
    comparisons = [f"/compare/{a}-vs-{b}" for _, a, b in content.COMPARISONS]
    return (
        f"\nLive data hints (from the database right now):\n"
        f"  cpu_specs.family values: {', '.join(repr(f) for f in families)}\n"
        f"  cpu_specs.codename values: {', '.join(repr(c) for c in codenames)}\n"
        f"  gpu_specs.vendor values: {', '.join(repr(v) for v in vendors)}\n"
        f"  comparison pages: {', '.join(comparisons)}\n"
    )


def _build_system_prompt(db: Session) -> str:
    """Return the system prompt, including the live schema context."""
    return SYSTEM_PROMPT + _get_live_schema_context(db)


def _cpu_to_dict(cpu: CPUSpec) -> dict[str, Any]:
    """Convert a CPU row to a tool-result dict."""
    return {
        "type": "cpu",
        "id": cpu.id,
        "name": cpu.name,
        "url": f"/cpu/{cpu.slug}",
        "cpu_model_name": cpu.cpu_model_name,
        "family": cpu.family,
        "cpu_model": cpu.cpu_model,
        "codename": cpu.codename,
        "cores": cpu.cores,
        "threads": cpu.threads,
        "tdp_watts": cpu.tdp_watts,
        "launch_year": cpu.launch_year,
        "max_turbo_frequency_ghz": cpu.max_turbo_frequency_ghz,
        "l3_cache_mb": cpu.l3_cache_mb,
        "max_memory_tb": cpu.max_memory_tb,
        "validated": bool(cpu.validated),
    }


def _gpu_to_dict(gpu: GPUSpec) -> dict[str, Any]:
    """Convert a GPU row to a tool-result dict."""
    return {
        "type": "gpu",
        "id": gpu.id,
        "name": gpu.name,
        "url": f"/gpu/{gpu.slug}",
        "gpu_model_name": gpu.gpu_model_name,
        "vendor": gpu.vendor,
        "gpu_model": gpu.gpu_model,
        "form_factor": gpu.form_factor,
        "memory_gb": gpu.memory_gb,
        "memory_type": gpu.memory_type,
        "tdp_watts": gpu.tdp_watts,
        "architecture": gpu.architecture,
        "launch_year": gpu.launch_year,
        "memory_bandwidth_tbs": gpu.memory_bandwidth_tbs,
        "fp64_tflops": gpu.fp64_tflops,
        "fp32_tflops": gpu.fp32_tflops,
        "fp16_bf16_tensor_tflops": gpu.fp16_bf16_tensor_tflops,
        "fp8_tensor_tflops": gpu.fp8_tensor_tflops,
        "fp4_tensor_tflops": gpu.fp4_tensor_tflops,
        "interconnect": gpu.interconnect,
        "interconnect_bandwidth_gbs": gpu.interconnect_bandwidth_gbs,
        "pcie_interface": gpu.pcie_interface,
        "validated": bool(gpu.validated),
    }


def _normalize_sql(query: str) -> str:
    """Fix common LLM mistakes in a SQL query."""
    normalized = query.strip()
    # family = 'AMD' never matches; families are 'AMD EPYC', 'AMD Opteron', etc.
    normalized = re.sub(
        r"\bfamily\s*=\s*'AMD'",
        "family LIKE '%AMD%'",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\bfamily\s*=\s*\"AMD\"",
        "family LIKE '%AMD%'",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\bfamily\s*=\s*'Intel'",
        "family LIKE '%Intel%'",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\bfamily\s*=\s*\"Intel\"",
        "family LIKE '%Intel%'",
        normalized,
        flags=re.IGNORECASE,
    )
    # cpu_specs has no vendor column — models often confuse it with gpu_specs
    if "cpu_specs" in normalized.lower() or "gpu_specs" not in normalized.lower():
        normalized = re.sub(
            r"\bvendor\s*=\s*'AMD'",
            "family LIKE '%AMD%'",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\bvendor\s*=\s*'Intel'",
            "family LIKE '%Intel%'",
            normalized,
            flags=re.IGNORECASE,
        )
    return normalized


def _sql_result_is_empty(rows: list[dict[str, Any]]) -> bool:
    """Return True for no rows, or a single row of NULLs."""
    if not rows:
        return True
    if len(rows) == 1:
        return all(value is None for value in rows[0].values())
    return False


def _vendor_fallback_sql(query: str) -> str | None:
    """Build a broader query when the model's filter matched nothing, or None."""
    lower = query.lower()
    year_match = re.search(
        r"launch_year\s+between\s+(\d{4})\s+and\s+(\d{4})",
        query,
        re.IGNORECASE,
    )
    year_clause = (
        f" AND launch_year BETWEEN {year_match.group(1)} AND {year_match.group(2)}"
        if year_match
        else ""
    )

    if "amd" not in lower:
        return None

    if "avg" in lower and "core" in lower:
        return (
            "SELECT ROUND(AVG(cores), 2) AS avg_cores, COUNT(*) AS cpu_count "
            f"FROM cpu_specs WHERE family LIKE '%AMD%' AND cores > 0{year_clause}"
        )

    return (
        "SELECT cpu_model_name, family, launch_year, cores "
        f"FROM cpu_specs WHERE family LIKE '%AMD%'{year_clause} "
        "ORDER BY launch_year DESC LIMIT 10"
    )


def _validate_sql(query: str) -> str:
    """Normalize a query and check that it is a single read-only SELECT.

    Raises:
        ValueError: If the query is not allowed.
    """
    normalized = _normalize_sql(query).rstrip(";").strip()
    if not normalized:
        raise ValueError("Empty SQL query")

    if ";" in normalized:
        raise ValueError("Multiple SQL statements are not allowed")

    if not re.match(r"^SELECT\b", normalized, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")

    if _FORBIDDEN_SQL.search(normalized):
        raise ValueError("Query contains forbidden SQL keywords")

    return normalized


def _search_cpus(db: Session, q: str) -> list[dict[str, Any]]:
    """Search CPUs by name, family, model or codename."""
    search_filter = or_(
        CPUSpec.cpu_model_name.ilike(f"%{q}%"),
        CPUSpec.family.ilike(f"%{q}%"),
        CPUSpec.cpu_model.ilike(f"%{q}%"),
        CPUSpec.codename.ilike(f"%{q}%"),
    )
    cpus = db.query(CPUSpec).filter(search_filter).limit(MAX_SEARCH_RESULTS).all()
    return [_cpu_to_dict(cpu) for cpu in cpus]


def _search_gpus(db: Session, q: str) -> list[dict[str, Any]]:
    """Search GPUs by name, vendor, model, form factor or memory type."""
    search_filter = or_(
        GPUSpec.gpu_model_name.ilike(f"%{q}%"),
        GPUSpec.vendor.ilike(f"%{q}%"),
        GPUSpec.gpu_model.ilike(f"%{q}%"),
        GPUSpec.form_factor.ilike(f"%{q}%"),
        GPUSpec.memory_type.ilike(f"%{q}%"),
        GPUSpec.architecture.ilike(f"%{q}%"),
    )
    gpus = db.query(GPUSpec).filter(search_filter).limit(MAX_SEARCH_RESULTS).all()
    return [_gpu_to_dict(gpu) for gpu in gpus]


def _get_generation_info(db: Session, codename: str) -> dict[str, Any]:
    """Written intro and platform facts for a CPU generation, from content.py."""
    wanted = slugify(codename or "")
    match = next((name for name in content.GENERATIONS if slugify(name) == wanted), None)
    if match is None:
        match = next((name for name in content.GENERATIONS if wanted and wanted in slugify(name)), None)
    if match is None:
        return {"error": f"No written overview for '{codename}'. Known: {', '.join(content.GENERATIONS)}"}
    written = content.GENERATIONS[match]
    count = db.query(CPUSpec).filter(CPUSpec.codename == match).count()
    return {
        "codename": match,
        "url": f"/cpus/codename/{slugify(match)}",
        "models_in_database": count,
        "intro": written.get("intro", []),
        "facts": dict(written.get("facts", [])),
    }


def _get_stats(db: Session) -> dict[str, Any]:
    """Return summary statistics for CPUs and GPUs."""
    total = db.query(CPUSpec).count()

    families = db.query(CPUSpec.family).distinct().all()
    unique_families = len([f[0] for f in families if f[0]])

    codenames = db.query(CPUSpec.codename).distinct().all()
    unique_codenames = len([c[0] for c in codenames if c[0]])

    avg_cores = db.query(CPUSpec.cores).filter(CPUSpec.cores.isnot(None)).all()
    avg_cores_value = sum(c[0] for c in avg_cores) / len(avg_cores) if avg_cores else None

    max_cores_row = (
        db.query(CPUSpec.cores)
        .filter(CPUSpec.cores.isnot(None))
        .order_by(CPUSpec.cores.desc())
        .first()
    )
    max_cores = max_cores_row[0] if max_cores_row else None

    years = db.query(CPUSpec.launch_year).filter(CPUSpec.launch_year.isnot(None)).all()
    year_values = [y[0] for y in years if y[0]]
    year_range = f"{min(year_values)}–{max(year_values)}" if year_values else None

    total_gpus = db.query(GPUSpec).count()
    gpu_vendors = db.query(GPUSpec.vendor).distinct().all()
    unique_gpu_vendors = len([v[0] for v in gpu_vendors if v[0]])

    gpu_memory_rows = db.query(GPUSpec.memory_gb).filter(GPUSpec.memory_gb.isnot(None)).all()
    max_gpu_memory = max(m[0] for m in gpu_memory_rows) if gpu_memory_rows else None

    gpu_memory_types = db.query(GPUSpec.memory_type).distinct().all()
    unique_memory_types = len([m[0] for m in gpu_memory_types if m[0]])

    validated_cpus = db.query(CPUSpec).filter(CPUSpec.validated.is_(True)).count()
    validated_gpus = db.query(GPUSpec).filter(GPUSpec.validated.is_(True)).count()

    return {
        "total_cpus": total,
        "validated_cpus": validated_cpus,
        "unique_families": unique_families,
        "unique_codenames": unique_codenames,
        "average_cores": round(avg_cores_value, 2) if avg_cores_value else None,
        "max_cores": max_cores,
        "year_range": year_range,
        "total_gpus": total_gpus,
        "validated_gpus": validated_gpus,
        "unique_gpu_vendors": unique_gpu_vendors,
        "max_gpu_memory_gb": max_gpu_memory,
        "unique_memory_types": unique_memory_types,
    }


def _run_sql(db: Session, query: str) -> list[dict[str, Any]]:
    """Run a validated SELECT, with a broader fallback if it returns nothing.

    Raises:
        ValueError: If the query is invalid or returns more than MAX_SQL_ROWS rows.
    """
    safe_query = _validate_sql(query)
    logger.info("Running SQL: %s", safe_query)
    result = db.execute(text(safe_query))
    rows = result.mappings().fetchmany(MAX_SQL_ROWS + 1)
    if len(rows) > MAX_SQL_ROWS:
        raise ValueError(f"Query returned more than {MAX_SQL_ROWS} rows; refine your query")
    parsed = [dict(row) for row in rows]

    if _sql_result_is_empty(parsed):
        fallback = _vendor_fallback_sql(query)
        if fallback:
            logger.info("SQL returned no rows; trying fallback: %s", fallback)
            fallback_result = db.execute(text(fallback))
            fallback_rows = [dict(row) for row in fallback_result.mappings().fetchall()]
            if fallback_rows and not _sql_result_is_empty(fallback_rows):
                return fallback_rows

    return parsed


def execute_tool(name: str, args: dict[str, Any], db: Session) -> Any:
    """Run a tool requested by the model.

    Raises:
        ValueError: For an unknown tool name.
    """
    if name == "search_cpus":
        return _search_cpus(db, args.get("q", ""))
    if name == "search_gpus":
        return _search_gpus(db, args.get("q", ""))
    if name == "get_stats":
        return _get_stats(db)
    if name == "get_generation_info":
        return _get_generation_info(db, args.get("codename", ""))
    if name == "run_sql":
        return _run_sql(db, args.get("query", ""))
    raise ValueError(f"Unknown tool: {name}")


def _extract_sources(tool_results: list[Any]) -> list[dict[str, Any]]:
    """Collect up to 10 unique CPU/GPU rows from tool results."""
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for result in tool_results:
        if not isinstance(result, list):
            continue
        for item in result:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            item_id = item.get("id")
            if item_type in ("cpu", "gpu") and item_id is not None:
                key = (item_type, item_id)
                if key not in seen:
                    seen.add(key)
                    sources.append(item)
            elif item.get("id") is None and (item.get("cpu_model_name") or item.get("gpu_model_name")):
                kind = "cpu" if item.get("cpu_model_name") else "gpu"
                key = (kind, item.get(f"{kind}_model_name"))
                if key not in seen:
                    seen.add(key)
                    sources.append({"type": kind, **item})
            elif "cpu_model_name" in item and item.get("id") is not None:
                key = ("cpu", item["id"])
                if key not in seen:
                    seen.add(key)
                    sources.append({"type": "cpu", **item})
            elif "gpu_model_name" in item and item.get("id") is not None:
                key = ("gpu", item["id"])
                if key not in seen:
                    seen.add(key)
                    sources.append({"type": "gpu", **item})

    return sources[:10]


def _with_page_urls(db: Session, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill in ``name`` and ``url`` for sources that came from raw SQL rows."""
    for source in sources:
        if source.get("url"):
            continue
        model = CPUSpec if source.get("type") == "cpu" else GPUSpec
        if source.get("id") is not None:
            row = db.get(model, source["id"])
        else:
            name_column = model.cpu_model_name if model is CPUSpec else model.gpu_model_name
            row = db.query(model).filter(name_column == source.get(f"{source['type']}_model_name")).first()
        if row is not None:
            source["name"] = row.name
            source["url"] = f"/{source['type']}/{row.slug}"
    return [s for s in sources if s.get("url")]


class LLMError(Exception):
    """Raised when the LLM provider returns an error."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider rate-limits requests."""


class LLMTimeoutError(LLMError):
    """Raised when the LLM provider times out."""


def _looks_like_tool_call_text(text: str) -> bool:
    """Return True if the text contains a tool call written as text."""
    if not text:
        return False
    return bool(
        _TEXT_TOOL_CALL.search(text)
        or _TEXT_TOOL_CALL_ALT.search(text)
        or "<function" in text.lower()
    )


def _parse_text_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse tool calls written in the message text (Llama fallback format)."""
    parsed: list[dict[str, Any]] = []
    for pattern in (_TEXT_TOOL_CALL, _TEXT_TOOL_CALL_ALT):
        for index, (name, args_json) in enumerate(pattern.findall(content)):
            try:
                args = json.loads(args_json)
            except json.JSONDecodeError:
                # Try fixing missing closing brace before </function>
                try:
                    args = json.loads(args_json + "}")
                except json.JSONDecodeError:
                    logger.warning("Could not parse tool args for %s: %s", name, args_json[:80])
                    continue
            parsed.append(
                {
                    "id": f"parsed-{len(parsed)}",
                    "name": name,
                    "arguments": args,
                }
            )
    return parsed


def _compact(result: Any) -> Any:
    """Drop empty fields from tool results: fewer tokens for the model to read."""
    if isinstance(result, list):
        return [_compact(item) for item in result]
    if isinstance(result, dict):
        return {k: v for k, v in result.items() if v not in (None, "", [])}
    return result


def _run_tool_calls(
    tool_calls: list[Any],
    db: Session,
    messages: list[dict[str, Any]],
    all_tool_results: list[Any],
    tools_used: list[str],
) -> None:
    """Execute tool calls and append their results to ``messages``."""
    for tool_call in tool_calls:
        if hasattr(tool_call, "function"):
            tool_id = tool_call.id
            tool_name = tool_call.function.name
            raw_args = tool_call.function.arguments or "{}"
        else:
            tool_id = tool_call["id"]
            tool_name = tool_call["name"]
            raw_args = json.dumps(tool_call.get("arguments", {}))

        try:
            tool_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            tool_args = {}

        tools_used.append(tool_name)
        logger.info("LLM tool call: %s", tool_name)

        try:
            result = _compact(execute_tool(tool_name, tool_args, db))
        except Exception as exc:
            result = {"error": str(exc)}

        all_tool_results.append(result)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": json.dumps(result, default=str),
            }
        )


def _model_unavailable(exc: Exception) -> bool:
    """True when the model is retired, unknown or blocked for this key (worth trying another one)."""
    text_ = str(exc).lower()
    return getattr(exc, "status_code", None) in (400, 403, 404) and any(
        marker in text_
        for marker in ("decommissioned", "model_not_found", "does not exist", "not supported",
                       "do not have access", "blocked", "permission")
    )


def _models_for_this_key(client: Groq) -> list[str]:
    """Chat models this API key can use, according to Groq's models endpoint."""
    try:
        listed = [m.id for m in client.models.list().data if getattr(m, "active", True)]
    except Exception:
        logger.exception("Could not list Groq models")
        return []
    return [m for m in listed if not any(marker in m.lower() for marker in NON_CHAT_MODEL_MARKERS)]


def _complete(client: Groq, **kwargs: Any) -> Any:
    """chat.completions.create with the first usable model.

    Tries GROQ_MODEL (or the defaults) first; if all of them are retired or blocked for
    this key, asks Groq which models the key can use and tries those.
    """
    global GROQ_MODEL
    last_exc: Exception | None = None
    candidates = [GROQ_MODEL, *[m for m in GROQ_MODELS if m != GROQ_MODEL]]
    tried: set[str] = set()
    discovered = False
    while candidates:
        model = candidates.pop(0)
        if model in tried:
            continue
        tried.add(model)
        extra = {"max_completion_tokens": MAX_COMPLETION_TOKENS}
        if model.startswith("openai/gpt-oss"):
            extra["reasoning_effort"] = REASONING_EFFORT
        started = time.monotonic()
        try:
            response = client.chat.completions.create(model=model, **extra, **kwargs)
        except Exception as exc:
            if _model_unavailable(exc):
                logger.warning("Groq model %s is unavailable (%s); trying the next one", model, exc)
                last_exc = exc
                if not candidates and not discovered:
                    discovered = True
                    available = _models_for_this_key(client)
                    logger.warning("Models available to this Groq key: %s", ", ".join(available) or "none")
                    preferred = [m for m in DEFAULT_GROQ_MODELS if m in available]
                    candidates = preferred + [m for m in available if m not in preferred]
                continue
            raise
        usage = getattr(response, "usage", None)
        logger.info("Groq %s answered in %.1fs (prompt %s, completion %s tokens)", model,
                    time.monotonic() - started, getattr(usage, "prompt_tokens", "?"),
                    getattr(usage, "completion_tokens", "?"))
        if model != GROQ_MODEL:
            logger.warning("Switched the Ask model to %s; update GROQ_MODEL", model)
            GROQ_MODEL = model
        return response
    raise last_exc or RuntimeError("No Groq model configured")


def _synthesize_answer(client: Groq, messages: list[dict[str, Any]]) -> str:
    """Ask the model for a final answer from the tool results, without tools."""
    synthesis_messages = messages + [
        {
            "role": "user",
            "content": (
                "Using only the tool results above, write a clear natural-language answer "
                "to the user's original question. Do not call tools or output function syntax."
            ),
        }
    ]
    response = _complete(client, messages=synthesis_messages, temperature=0.2)
    answer = (response.choices[0].message.content or "").strip()
    if _looks_like_tool_call_text(answer):
        return "I found relevant data but could not format the answer. Please try rephrasing."
    return answer or "I could not generate an answer. Please try rephrasing your question."


def _answer_claims_no_data(text: str) -> bool:
    """Return True if the answer says the data is not available."""
    lower = text.lower()
    return any(
        phrase in lower
        for phrase in (
            "do not have",
            "don't have",
            "do not know",
            "no information",
            "not in the database",
            "not enough information",
            "could not find",
            "i cannot find",
        )
    )


def ask_question(question: str, db: Session) -> dict[str, Any]:
    """Answer a question about the specs in the database.

    Returns:
        Dict with ``answer``, ``sources`` and ``tools_used``.

    Raises:
        LLMError: If GROQ_API_KEY is missing or the provider fails.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not configured")

    client = Groq(api_key=api_key, timeout=30.0)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt(db)},
        {"role": "user", "content": question},
    ]

    all_tool_results: list[Any] = []
    tools_used: list[str] = []

    for round_num in range(MAX_TOOL_ROUNDS):
        try:
            response = _complete(
                client,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as exc:
            exc_name = type(exc).__name__
            if "RateLimit" in exc_name or getattr(exc, "status_code", None) == 429:
                raise LLMRateLimitError("Groq rate limit exceeded") from exc
            if "Timeout" in exc_name:
                raise LLMTimeoutError("Groq request timed out") from exc
            logger.exception("Groq API error (model %s): %s", GROQ_MODEL, exc)
            raise LLMError("The assistant is unavailable right now. Please try again later.") from exc

        choice = response.choices[0]
        message = choice.message
        content = (message.content or "").strip()

        structured_tool_calls = message.tool_calls or []
        text_tool_calls = (
            _parse_text_tool_calls(content) if not structured_tool_calls and content else []
        )
        has_tool_calls = bool(structured_tool_calls or text_tool_calls)

        if has_tool_calls:
            assistant_message: dict[str, Any] = {"role": "assistant", "content": content or None}

            if structured_tool_calls:
                tool_calls_to_run = structured_tool_calls
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in structured_tool_calls
                ]
            else:
                tool_calls_to_run = text_tool_calls
                assistant_message["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in text_tool_calls
                ]

            messages.append(assistant_message)
            _run_tool_calls(
                tool_calls_to_run, db, messages, all_tool_results, tools_used
            )
            continue

        if content and not _looks_like_tool_call_text(content):
            if tools_used and _answer_claims_no_data(content):
                logger.info("Model claimed no data after tools ran; forcing synthesis")
                break
            return {
                "answer": content,
                "sources": _with_page_urls(db, _extract_sources(all_tool_results)),
                "tools_used": tools_used,
            }

    if tools_used:
        try:
            answer = _synthesize_answer(client, messages)
        except Exception:
            logger.exception("Answer synthesis failed")
            answer = "I found data but could not summarize it. Please try a simpler question."
        return {
            "answer": answer,
            "sources": _with_page_urls(db, _extract_sources(all_tool_results)),
            "tools_used": tools_used,
        }

    return {
        "answer": "I need more steps to answer that question. Please try a simpler query.",
        "sources": _with_page_urls(db, _extract_sources(all_tool_results)),
        "tools_used": tools_used,
    }
