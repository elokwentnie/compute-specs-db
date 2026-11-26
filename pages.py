"""Server-rendered HTML pages for CPUs, GPUs and their groups.

One page per CPU and GPU, overview pages per CPU family, CPU codename
(generation) and GPU vendor, hand-picked comparisons, and the About page.
Specs come from the database on each request, so pages always match the API
and the CSVs; hand-written text lives in content.py.
"""

import csv
import io
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

import content
import proposals
import redirects
from database import CPUSpec, GPUSpec, get_db
from utils import slugify

SITE_URL = "https://computespecsdb.com"
PAGE_CACHE = "public, max-age=300"

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
router = APIRouter()


# ---------- Formatting ----------

def fmt(value, unit: str = "") -> str:
    """Format a number without a trailing ``.0``, plus ``unit``; ``""`` if missing."""
    if value is None or value == "":
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value}{unit}"


def span(values: Iterable, unit: str = "") -> str:
    """Format a range like ``8–60 W``, a single value, or ``""`` if none."""
    present = sorted(v for v in values if v is not None)
    if not present:
        return ""
    low, high = present[0], present[-1]
    return fmt(low, unit) if low == high else f"{fmt(low)}–{fmt(high, unit)}"


templates.env.filters["fmt"] = fmt
templates.env.globals["site_url"] = SITE_URL


def json_ld(data) -> str:
    """Serialize data as JSON-LD that is safe inside a ``<script>`` tag."""
    text = json.dumps(data, ensure_ascii=False)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def breadcrumb_ld(crumbs) -> dict:
    """Build a schema.org BreadcrumbList from ``(label, url)`` pairs."""
    return {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {"@type": "ListItem", "position": i, "name": label, "item": f"{SITE_URL}{url}"}
            for i, (label, url) in enumerate(crumbs, start=1)
        ],
    }


TITLE_SUFFIX = " — Compute Specs DB"
TITLE_MAX = 60  # Google truncates longer titles


def fit_title(*candidates: str) -> str:
    """Pick the first title that fits in TITLE_MAX.

    Each candidate is tried with TITLE_SUFFIX first, then without it.

    Args:
        *candidates: Titles from most to least detailed.

    Returns:
        The chosen title, or the last candidate if none fits.
    """
    for text in candidates:
        if len(text) + len(TITLE_SUFFIX) <= TITLE_MAX:
            return text + TITLE_SUFFIX
        if len(text) <= TITLE_MAX:
            return text
    return candidates[-1]


def render(request: Request, template: str, crumbs, title_fallbacks=(), **context):
    """Render a page template.

    Args:
        request: Incoming request.
        template: Template file name.
        crumbs: ``(label, url)`` breadcrumb pairs.
        title_fallbacks: Shorter titles for ``fit_title`` if ``context["title"]`` is too long.
        **context: Template variables; must include ``title``.
    """
    context.update(request=request, crumbs=crumbs, breadcrumb_ld=json_ld(breadcrumb_ld(crumbs)),
                   page_title=fit_title(context["title"], *title_fallbacks))
    return templates.TemplateResponse(template, context, headers={"Cache-Control": PAGE_CACHE})


# ---------- Data helpers ----------

CPU_SPECS = (
    ("Cores", "cores", ""),
    ("Threads", "threads", ""),
    ("Max turbo frequency", "max_turbo_frequency_ghz", " GHz"),
    ("L3 cache", "l3_cache_mb", " MB"),
    ("TDP", "tdp_watts", " W"),
    ("Launch year", "launch_year", ""),
    ("Max memory", "max_memory_tb", " TB"),
)

GPU_SPECS = (
    ("Vendor", "vendor", ""),
    ("Architecture", "architecture", ""),
    ("Launch year", "launch_year", ""),
    ("Form factor", "form_factor", ""),
    ("Memory", "memory_gb", " GB"),
    ("Memory type", "memory_type", ""),
    ("Memory bandwidth", "memory_bandwidth_tbs", " TB/s"),
    ("TDP", "tdp_watts", " W"),
    ("FP64 (vector)", "fp64_tflops", " TFLOPS"),
    ("FP32 (vector)", "fp32_tflops", " TFLOPS"),
    ("FP16/BF16 Tensor (dense)", "fp16_bf16_tensor_tflops", " TFLOPS"),
    ("FP8 Tensor (dense)", "fp8_tensor_tflops", " TFLOPS"),
    ("FP4 Tensor (dense)", "fp4_tensor_tflops", " TFLOPS"),
    ("Interconnect", "interconnect", ""),
    ("Interconnect bandwidth", "interconnect_bandwidth_gbs", " GB/s"),
    ("Host interface", "pcie_interface", ""),
)


def cpu_vendor(cpu: CPUSpec) -> str:
    """Return the vendor (first word of the name)."""
    return cpu.name.split(" ", 1)[0]


def all_cpus(db: Session) -> List[CPUSpec]:
    """Return all CPUs ordered by id."""
    return db.query(CPUSpec).order_by(CPUSpec.id).all()


def all_gpus(db: Session) -> List[GPUSpec]:
    """Return all GPUs ordered by id."""
    return db.query(GPUSpec).order_by(GPUSpec.id).all()


def newest_first(items):
    """Sort by launch year, then core count, both descending."""
    return sorted(items, key=lambda c: (-(c.launch_year or 0), -(c.cores or 0), c.name))


def group_by(items, attr):
    """Group items by an attribute, skipping empty values."""
    groups = defaultdict(list)
    for item in items:
        value = getattr(item, attr)
        if value:
            groups[value].append(item)
    return groups


def cpu_group_summary(cpus: List[CPUSpec]) -> dict:
    """Summarize a group of CPUs: count, ranges and largest L3."""
    return {
        "count": len(cpus),
        "validated": sum(c.validated for c in cpus),
        "years": span(c.launch_year for c in cpus),
        "cores": span(c.cores for c in cpus),
        "tdp": span((c.tdp_watts for c in cpus), " W"),
        "max_l3": fmt(max((c.l3_cache_mb for c in cpus if c.l3_cache_mb), default=None), " MB"),
        "latest_year": max((c.launch_year or 0 for c in cpus), default=0),
    }


def cpu_sentences(cpu: CPUSpec) -> List[str]:
    """Describe a CPU in plain sentences, using only known values."""
    parts = []
    shape = ", ".join(p for p in (
        f"{fmt(cpu.cores)}-core" if cpu.cores else "",
        f"{fmt(cpu.threads)}-thread" if cpu.threads else "",
    ) if p)
    first = f"The {cpu.name} is a {shape + ' ' if shape else ''}{cpu.family or 'server'} processor"
    if cpu.codename:
        first += f" from the {cpu.codename} generation"
    if cpu.launch_year:
        first += f", launched in {cpu.launch_year}"
    parts.append(first + ".")

    details = [d for d in (
        f"reaches up to {fmt(cpu.max_turbo_frequency_ghz)} GHz" if cpu.max_turbo_frequency_ghz else "",
        f"has {fmt(cpu.l3_cache_mb)} MB of L3 cache" if cpu.l3_cache_mb else "",
        f"is rated at {fmt(cpu.tdp_watts)} W TDP" if cpu.tdp_watts else "",
        f"supports up to {memory_text(cpu.max_memory_tb)} of memory" if cpu.max_memory_tb else "",
    ) if d]
    if details:
        joined = details[0] if len(details) == 1 else ", ".join(details[:-1]) + " and " + details[-1]
        parts.append(f"It {joined}.")
    return parts


def gpu_sentences(gpu: GPUSpec) -> List[str]:
    """Describe a GPU in plain sentences, using only known values."""
    first = f"The {gpu.name} is a datacenter accelerator"
    if gpu.vendor:
        first += f" from {gpu.vendor}"
    memory = " ".join(p for p in (fmt(gpu.memory_gb, " GB"), "of " + gpu.memory_type if gpu.memory_type else "") if p)
    if gpu.memory_gb:
        first += f" with {memory} memory"
    if gpu.tdp_watts:
        first += f" and a {fmt(gpu.tdp_watts)} W TDP" if gpu.memory_gb else f" with a {fmt(gpu.tdp_watts)} W TDP"
    parts = [first + "."]
    origin = " ".join(p for p in (
        f"on the {gpu.architecture} architecture" if gpu.architecture else "",
        f"in {gpu.launch_year}" if gpu.launch_year else "",
    ) if p)
    if origin:
        parts.append(f"It was introduced {origin}.")
    perf = [f"{fmt(value, ' TFLOPS')} {label}" for label, value in (
        ("FP64", gpu.fp64_tflops),
        ("FP32", gpu.fp32_tflops),
        ("dense FP16/BF16 tensor", gpu.fp16_bf16_tensor_tflops),
        ("dense FP8 tensor", gpu.fp8_tensor_tflops),
        ("dense FP4 tensor", gpu.fp4_tensor_tflops),
    ) if value]
    if gpu.memory_bandwidth_tbs:
        perf.insert(0, f"{fmt(gpu.memory_bandwidth_tbs, ' TB/s')} memory bandwidth")
    if perf:
        joined = perf[0] if len(perf) == 1 else ", ".join(perf[:-1]) + " and " + perf[-1]
        parts.append(f"Peak figures: {joined}.")
    if gpu.form_factor:
        parts.append(f"Form factor: {gpu.form_factor}.")
    return parts


def cpu_titles(cpu: CPUSpec) -> List[str]:
    """Return candidate page titles for ``fit_title``, most detailed first."""
    cores = fmt(cpu.cores, " Cores") if cpu.cores else ""
    l3 = fmt(cpu.l3_cache_mb, " MB L3") if cpu.l3_cache_mb else ""
    tdp = fmt(cpu.tdp_watts, " W") if cpu.tdp_watts else ""
    titles = []
    for parts in ((cores, l3, tdp), (cores, l3), (cores,), ()):
        facts = ", ".join(p for p in parts if p)
        title = f"{cpu.name} Specs{': ' + facts if facts else ''}"
        if title not in titles:
            titles.append(title)
    return titles


VENDOR_PREFIXES = ("AMD ", "Intel ", "NVIDIA ", "Ampere ")


def without_vendor(name: str) -> str:
    """Drop the vendor from a model name for short titles, e.g. "Xeon Platinum 8480+"."""
    return next((name[len(p):] for p in VENDOR_PREFIXES if name.startswith(p)), name)


def plural(count: int, word: str) -> str:
    """Format a count with a noun, e.g. "1 model" or "3 models"."""
    return f"{count} {word}{'' if count == 1 else 's'}"


def group_title(group: str, kind: str, members) -> str:
    """Title for a group page; names the model when the group has only one."""
    if len(members) == 1:
        return f"{group} {kind}: {members[0].name} Specs"
    return f"{group} {kind} List: Specs of All {len(members)} Models"


def group_title_fallbacks(group: str, kind: str, members) -> tuple:
    """Shorter versions of ``group_title`` for ``fit_title``."""
    return (f"{group} {kind} List: {plural(len(members), 'Model')}", f"{group} {kind} Specs")


def duplicate_codenames(cpus: List[CPUSpec]) -> dict:
    """Find codenames whose CPUs are exactly the CPUs of one family.

    Their pages duplicate the family page, so they are canonicalized to it
    and left out of the sitemap.

    Returns:
        Mapping of codename to family slug.
    """
    families = {frozenset(c.id for c in members): slugify(family)
                for family, members in group_by(cpus, "family").items()}
    return {codename: families[ids]
            for codename, members in group_by(cpus, "codename").items()
            if (ids := frozenset(c.id for c in members)) in families}


DESCRIPTION_MAX = 160  # search engines cut longer snippets; Bing flags them


def fit_description(*candidates: str) -> str:
    """Return the first candidate within DESCRIPTION_MAX, else the last one truncated."""
    for text in candidates:
        if len(text) <= DESCRIPTION_MAX:
            return text
    return candidates[-1][:DESCRIPTION_MAX - 1].rsplit(" ", 1)[0] + "…"


def memory_text(tb) -> str:
    """Format memory in TB, or in GB below 1 TB."""
    if tb < 1:
        return fmt(round(tb * 1000), " GB")
    return fmt(tb, " TB")


def cpu_description(cpu: CPUSpec) -> str:
    """Build the meta description for a CPU page."""
    cores = (f"{fmt(cpu.cores)} cores/{fmt(cpu.threads)} threads" if cpu.cores and cpu.threads
             else fmt(cpu.cores, " cores") if cpu.cores else "")
    turbo = fmt(cpu.max_turbo_frequency_ghz, " GHz turbo") if cpu.max_turbo_frequency_ghz else ""
    cache = fmt(cpu.l3_cache_mb, " MB L3") if cpu.l3_cache_mb else ""
    tdp = fmt(cpu.tdp_watts, " W TDP") if cpu.tdp_watts else ""
    memory = f"up to {memory_text(cpu.max_memory_tb)} memory" if cpu.max_memory_tb else ""
    context = ", ".join(str(p) for p in (cpu.codename, cpu.launch_year) if p)
    status = "Validated." if cpu.validated else "Not yet validated."

    def build(spec_parts, extras):
        """Join the name, spec parts and extra sentences."""
        specs = ", ".join(p for p in spec_parts if p)
        text = f"{cpu.name} specs" + (f": {specs}." if specs else ".")
        return " ".join([text, *(e for e in extras if e)])

    return fit_description(
        build((cores, turbo, cache, tdp, memory), (context + "." if context else "", status)),
        build((cores, turbo, cache, tdp, memory), (context + "." if context else "",)),
        build((cores, turbo, cache, tdp, memory), ()),
        build((cores, cache, tdp), ()),
    )


def gpu_description(gpu: GPUSpec) -> str:
    """Build the meta description for a GPU page."""
    specs = ", ".join(s for s in (
        gpu.architecture or "",
        " ".join(p for p in (fmt(gpu.memory_gb, " GB"), gpu.memory_type or "") if p),
        fmt(gpu.memory_bandwidth_tbs, " TB/s") if gpu.memory_bandwidth_tbs else "",
        fmt(gpu.fp8_tensor_tflops, " TFLOPS FP8") if gpu.fp8_tensor_tflops else "",
        fmt(gpu.tdp_watts, " W TDP") if gpu.tdp_watts else "",
        gpu.form_factor or "",
    ) if s)
    status = "Validated against manufacturer sources." if gpu.validated else "Not yet validated."
    lead = f"{gpu.name} datacenter GPU specifications" + (f": {specs}." if specs else ".")
    return fit_description(f"{lead} {status}", lead)


def similar_cpus(cpu: CPUSpec, cpus: List[CPUSpec], limit: int = 8) -> List[CPUSpec]:
    """Return CPUs of the same generation, then the same family, closest first."""
    others = [c for c in cpus if c.id != cpu.id]
    same_gen = sorted((c for c in others if cpu.codename and c.codename == cpu.codename),
                      key=lambda c: (abs((c.cores or 0) - (cpu.cores or 0)), c.name))
    picked = same_gen[:limit]
    if len(picked) < limit:
        chosen = {c.id for c in picked}
        same_family = sorted((c for c in others if c.family == cpu.family and c.id not in chosen),
                             key=lambda c: (abs((c.launch_year or 0) - (cpu.launch_year or 0)),
                                            abs((c.cores or 0) - (cpu.cores or 0)), c.name))
        picked += same_family[:limit - len(picked)]
    return picked


def similar_gpus(gpu: GPUSpec, gpus: List[GPUSpec], limit: int = 6) -> List[GPUSpec]:
    """Return GPUs of the same vendor first, closest in memory size."""
    others = [g for g in gpus if g.id != gpu.id]
    return sorted(others, key=lambda g: (g.vendor != gpu.vendor,
                                         abs((g.memory_gb or 0) - (gpu.memory_gb or 0)), g.name))[:limit]


def find_by_slug(items, slug: str):
    """Return the item with ``slug``, or raise a 404."""
    for item in items:
        if item.slug == slug:
            return item
    raise HTTPException(status_code=404, detail="Not found")


def renamed_target(db: Session, kind: str, items, slug: str) -> Optional[str]:
    """Current URL of a model that used to live at /<kind>/<slug>, if it was renamed."""
    name = redirects.target_name(db, kind, slug)
    if not name:
        return None
    key = proposals.SCHEMAS[kind]["key_attr"]
    folded = name.strip().casefold()
    item = next((i for i in items if (getattr(i, key) or "").strip().casefold() == folded), None)
    return f"/{kind}/{item.slug}" if item and item.slug != slug else None


def find_model_or_redirect(db: Session, kind: str, items, slug: str):
    """The model at this slug, or a 301 to its new URL when it was renamed."""
    try:
        return find_by_slug(items, slug)
    except HTTPException:
        target = renamed_target(db, kind, items, slug)
        if target:
            return RedirectResponse(target, status_code=301)
        raise


def find_group(groups: dict, slug: str):
    """Return ``(key, members)`` for the group with ``slug``, or raise a 404."""
    for key, members in groups.items():
        if slugify(key) == slug:
            return key, members
    raise HTTPException(status_code=404, detail="Not found")


def report_url(name: str) -> str:
    """Return the GitHub URL for a prefilled data-error issue."""
    title = quote(f"Data error: {name}")
    return f"https://github.com/elokwentnie/compute-specs-db/issues/new?template=report-data-error.yml&title={title}"


def report_form(kind: str, item) -> dict:
    """Return template context for the 'Report an error' form."""
    fields = []
    for attr, label in proposals.report_fields(kind):
        current = getattr(item, attr, None) if attr != proposals.REPORT_OTHER_FIELD else None
        fields.append({"attr": attr, "label": label, "current": fmt(current)})
    return {
        "endpoint": f"/api/reports/{kind}/{item.id}",
        "fields": fields,
        "fallback_url": report_url(item.name),
    }


# ---------- Shared pieces for the homepage and sitemap ----------

def browse_links(db: Session) -> dict:
    """Return family, codename and vendor links for the homepage and /cpus."""
    cpus = all_cpus(db)
    gpus = all_gpus(db)
    families = group_by(cpus, "family")
    codenames = group_by(cpus, "codename")

    family_links = sorted(
        ({"label": fam, "url": f"/cpus/{slugify(fam)}", **cpu_group_summary(members)}
         for fam, members in families.items()),
        key=lambda f: (-f["count"], f["label"]))

    codename_by_vendor = defaultdict(list)
    for codename, members in codenames.items():
        codename_by_vendor[cpu_vendor(members[0])].append(
            {"label": codename, "url": f"/cpus/codename/{slugify(codename)}", **cpu_group_summary(members)})
    for links in codename_by_vendor.values():
        links.sort(key=lambda c: (-c["latest_year"], c["label"]))
    vendor_order = sorted(codename_by_vendor, key=lambda v: -sum(c["count"] for c in codename_by_vendor[v]))

    vendors = group_by(gpus, "vendor")
    vendor_links = sorted(
        ({"label": vendor, "url": f"/gpus/{slugify(vendor)}", "count": len(members)}
         for vendor, members in vendors.items()),
        key=lambda v: (-v["count"], v["label"]))

    return {
        "families": family_links,
        "codenames": [(vendor, codename_by_vendor[vendor]) for vendor in vendor_order],
        "gpu_vendors": vendor_links,
        "cpu_count": len(cpus),
        "gpu_count": len(gpus),
    }


def render_home_fragments(db: Session, rows: int = 25) -> dict:
    """Render homepage HTML fragments for crawlers without JavaScript."""
    first_rows = all_cpus(db)[:rows]
    return {
        "<!--SSR:CPU_ROWS-->": templates.get_template("partials/home_rows.html").render(cpus=first_rows),
        "<!--SSR:BROWSE-->": templates.get_template("partials/browse.html").render(**browse_links(db)),
    }


def sitemap_paths(db: Session) -> List[str]:
    """Return all spec and group page paths for the sitemap."""
    cpus = all_cpus(db)
    gpus = all_gpus(db)
    paths = ["/cpus", "/gpus", "/about", "/compare"]
    paths += sorted(f"/cpus/{slugify(f)}" for f in group_by(cpus, "family"))
    duplicates = duplicate_codenames(cpus)
    paths += sorted(f"/cpus/codename/{slugify(c)}" for c in group_by(cpus, "codename") if c not in duplicates)
    paths += sorted(f"/gpus/{slugify(v)}" for v in group_by(gpus, "vendor"))
    paths += sorted(f"/cpu/{c.slug}" for c in cpus)
    paths += sorted(f"/gpu/{g.slug}" for g in gpus)
    paths += [pair["url"] for pair in comparison_pairs(db)]
    return paths


# ---------- CPU pages ----------

@router.get("/cpus", include_in_schema=False)
async def cpus_index(request: Request, db: Session = Depends(get_db)):
    """CPU overview page."""
    links = browse_links(db)
    cpus = all_cpus(db)
    return render(
        request, "cpus_index.html",
        crumbs=[("Home", "/"), ("CPUs", "/cpus")],
        title="Server CPU Specifications by Family and Generation",
        description=(f"Browse specifications for {len(cpus)} server-class CPUs, including Intel Xeon and AMD EPYC, "
                     "grouped by product family and generation (codename)."),
        canonical="/cpus",
        summary=cpu_group_summary(cpus),
        guide=content.CPUS_INDEX_GUIDE,
        cpu_comparisons=[pair for pair in comparison_pairs(db) if pair["kind"] == "cpu"],
        **links,
    )


@router.get("/cpus/codename/{slug}", include_in_schema=False)
async def cpu_codename_page(slug: str, request: Request, db: Session = Depends(get_db)):
    """Page listing the CPUs of one generation."""
    cpus = all_cpus(db)
    codename, members = find_group(group_by(cpus, "codename"), slug)
    summary = cpu_group_summary(members)
    families = sorted(group_by(members, "family").items(), key=lambda kv: -len(kv[1]))
    vendor = cpu_vendor(members[0])
    same_as_family = duplicate_codenames(cpus).get(codename)
    written = content.GENERATIONS.get(codename, {})
    return render(
        request, "cpu_group.html",
        crumbs=[("Home", "/"), ("CPUs", "/cpus"), (codename, f"/cpus/codename/{slug}")],
        title=group_title(codename, "CPU", members),
        title_fallbacks=group_title_fallbacks(codename, "CPU", members),
        description=(f"Specifications of {plural(summary['count'], vendor + ' ' + codename + ' processor')}"
                     f"{' (' + summary['years'] + ')' if summary['years'] else ''}: cores, threads, turbo frequency, "
                     "L3 cache, TDP and memory."),
        canonical=f"/cpus/{same_as_family}" if same_as_family else f"/cpus/codename/{slug}",
        heading=f"{codename} processors",
        group_label=f"{vendor} {codename}",
        summary=summary,
        related_title="Product families",
        related=[{"label": fam, "url": f"/cpus/{slugify(fam)}", "count": len(cpus)} for fam, cpus in families],
        cpus=newest_first(members),
        show_column="family",
        intro=written.get("intro", []),
        facts=written.get("facts", []),
        faq=generation_faq(codename, members, dict(written.get("facts", []))),
    )


@router.get("/cpus/{slug}", include_in_schema=False)
async def cpu_family_page(slug: str, request: Request, db: Session = Depends(get_db)):
    """Page listing the CPUs of one family."""
    family, members = find_group(group_by(all_cpus(db), "family"), slug)
    summary = cpu_group_summary(members)
    codenames = sorted(group_by(members, "codename").items(),
                       key=lambda kv: -max(c.launch_year or 0 for c in kv[1]))
    return render(
        request, "cpu_group.html",
        crumbs=[("Home", "/"), ("CPUs", "/cpus"), (family, f"/cpus/{slug}")],
        title=group_title(family, "CPU", members),
        title_fallbacks=group_title_fallbacks(family, "CPU", members),
        description=(f"Specifications of {plural(summary['count'], family + ' processor')}"
                     f"{' (' + summary['years'] + ')' if summary['years'] else ''}: cores, threads, turbo frequency, "
                     "L3 cache, TDP and memory."),
        canonical=f"/cpus/{slug}",
        heading=f"{family} processors",
        group_label=family,
        summary=summary,
        related_title="Generations",
        related=[{"label": name, "url": f"/cpus/codename/{slugify(name)}", "count": len(cpus)}
                 for name, cpus in codenames],
        cpus=newest_first(members),
        show_column="codename",
        intro=[content.FAMILIES[family]] if family in content.FAMILIES else [],
    )


@router.get("/cpu/{slug}", include_in_schema=False)
async def cpu_page(slug: str, request: Request, db: Session = Depends(get_db)):
    """Spec page for one CPU."""
    cpus = all_cpus(db)
    cpu = find_model_or_redirect(db, "cpu", cpus, slug)
    if isinstance(cpu, RedirectResponse):
        return cpu
    crumbs = [("Home", "/"), ("CPUs", "/cpus")]
    if cpu.family:
        crumbs.append((cpu.family, f"/cpus/{slugify(cpu.family)}"))
    if cpu.codename:
        crumbs.append((cpu.codename, f"/cpus/codename/{slugify(cpu.codename)}"))
    crumbs.append((cpu.name, f"/cpu/{cpu.slug}"))
    specs = [(label, memory_text(cpu.max_memory_tb) if attr == "max_memory_tb" and cpu.max_memory_tb
              else fmt(getattr(cpu, attr), unit)) for label, attr, unit in CPU_SPECS]
    return render(
        request, "cpu.html",
        crumbs=crumbs,
        title=cpu_titles(cpu)[0],
        title_fallbacks=cpu_titles(cpu)[1:],
        description=cpu_description(cpu),
        canonical=f"/cpu/{cpu.slug}",
        cpu=cpu,
        sentences=cpu_sentences(cpu),
        specs=[(label, value) for label, value in specs if value],
        similar=similar_cpus(cpu, cpus),
        report=report_form("cpu", cpu),
        comparisons=comparisons_for(db, "cpu", cpu.slug),
        family_url=f"/cpus/{slugify(cpu.family)}" if cpu.family else None,
        codename_url=f"/cpus/codename/{slugify(cpu.codename)}" if cpu.codename else None,
    )


# ---------- GPU pages ----------

def _gpu_group(request: Request, gpus, crumbs, canonical: str, heading: str, title: str, description: str,
               vendors: Optional[list] = None, title_fallbacks=(), intro: Optional[list] = None,
               guide: Optional[list] = None, comparisons: Optional[list] = None):
    """Render a GPU list page."""
    return render(
        request, "gpu_group.html",
        crumbs=crumbs, title=title, title_fallbacks=title_fallbacks, description=description,
        canonical=canonical, heading=heading,
        intro=intro or [], guide=guide or [], gpu_comparisons=comparisons or [],
        gpus=sorted(gpus, key=lambda g: (g.vendor or "", -(g.memory_gb or 0), g.name)),
        validated=sum(g.validated for g in gpus),
        memory=span((g.memory_gb for g in gpus), " GB"),
        tdp=span((g.tdp_watts for g in gpus), " W"),
        vendors=vendors or [],
    )


@router.get("/gpus", include_in_schema=False)
async def gpus_index(request: Request, db: Session = Depends(get_db)):
    """GPU overview page."""
    gpus = all_gpus(db)
    return _gpu_group(
        request, gpus,
        crumbs=[("Home", "/"), ("GPUs", "/gpus")],
        canonical="/gpus",
        heading="Datacenter GPU specifications",
        title="Datacenter GPU Specs: NVIDIA, AMD Instinct, Intel",
        description=(f"Specifications for {len(gpus)} datacenter GPUs and accelerators: memory size and type, "
                     "TDP and form factor for NVIDIA, AMD Instinct and Intel."),
        vendors=browse_links(db)["gpu_vendors"],
        guide=content.GPUS_INDEX_GUIDE,
        comparisons=[pair for pair in comparison_pairs(db) if pair["kind"] == "gpu"],
    )


@router.get("/gpus/{slug}", include_in_schema=False)
async def gpu_vendor_page(slug: str, request: Request, db: Session = Depends(get_db)):
    """Page listing one vendor's GPUs."""
    vendor, members = find_group(group_by(all_gpus(db), "vendor"), slug)
    return _gpu_group(
        request, members,
        crumbs=[("Home", "/"), ("GPUs", "/gpus"), (vendor, f"/gpus/{slug}")],
        canonical=f"/gpus/{slug}",
        heading=f"{vendor} datacenter GPUs",
        title=group_title(f"{vendor} Datacenter", "GPU", members),
        title_fallbacks=group_title_fallbacks(f"{vendor} Datacenter", "GPU", members),
        description=(f"Specifications of {plural(len(members), vendor + ' datacenter GPU')}"
                     f"{'' if len(members) == 1 else ' and accelerators'}: memory size and type, TDP and form factor."),
        intro=[content.GPU_VENDORS[vendor]] if vendor in content.GPU_VENDORS else [],
    )


@router.get("/gpu/{slug}", include_in_schema=False)
async def gpu_page(slug: str, request: Request, db: Session = Depends(get_db)):
    """Spec page for one GPU."""
    gpus = all_gpus(db)
    gpu = find_model_or_redirect(db, "gpu", gpus, slug)
    if isinstance(gpu, RedirectResponse):
        return gpu
    crumbs = [("Home", "/"), ("GPUs", "/gpus")]
    if gpu.vendor:
        crumbs.append((gpu.vendor, f"/gpus/{slugify(gpu.vendor)}"))
    crumbs.append((gpu.name, f"/gpu/{gpu.slug}"))
    specs = [(label, fmt(getattr(gpu, attr), unit)) for label, attr, unit in GPU_SPECS]
    memory = " ".join(p for p in (fmt(gpu.memory_gb, " GB"), gpu.memory_type or "") if p)
    tdp = fmt(gpu.tdp_watts, " W") if gpu.tdp_watts else ""
    titles = [f"{gpu.name} Specs{': ' + facts if facts else ''}"
              for facts in (", ".join(f for f in (memory, tdp) if f), memory)]
    return render(
        request, "gpu.html",
        crumbs=crumbs,
        title=titles[0],
        title_fallbacks=(*titles[1:], f"{gpu.name} Specs"),
        description=gpu_description(gpu),
        canonical=f"/gpu/{gpu.slug}",
        gpu=gpu,
        sentences=gpu_sentences(gpu),
        specs=[(label, value) for label, value in specs if value],
        similar=similar_gpus(gpu, gpus),
        report=report_form("gpu", gpu),
        comparisons=comparisons_for(db, "gpu", gpu.slug),
        vendor_url=f"/gpus/{slugify(gpu.vendor)}" if gpu.vendor else None,
    )


# ---------- Generation FAQ ----------

def generation_faq(codename: str, cpus: List[CPUSpec], facts: dict) -> List[dict]:
    """Questions people search for, answered from the data and the generation's facts."""
    faq = []
    n = len(cpus)
    cores = span(c.cores for c in cpus)
    if cores:
        answer = (f"The {n} {codename} model{'s' if n != 1 else ''} listed here "
                  f"{'have' if n != 1 else 'has'} {cores} cores.")
        if facts.get("Max cores"):
            answer += f" The platform supports up to {facts['Max cores']} cores per socket."
        faq.append({"q": f"How many cores do {codename} processors have?", "a": answer})
    with_cores = [c for c in cpus if c.cores]
    if n > 1 and with_cores:
        most = max(c.cores for c in with_cores)
        top = sorted((c for c in with_cores if c.cores == most), key=lambda c: c.name)
        faq.append({"q": f"Which {codename} processor has the most cores?",
                    "a": f"With {fmt(most)} cores:", "models": [("cpu", c) for c in top]})
    with_turbo = [c for c in cpus if c.max_turbo_frequency_ghz]
    if n > 1 and with_turbo:
        best = max(c.max_turbo_frequency_ghz for c in with_turbo)
        top = sorted((c for c in with_turbo if c.max_turbo_frequency_ghz == best), key=lambda c: c.name)
        faq.append({"q": f"Which {codename} processor has the highest turbo frequency?",
                    "a": f"Up to {fmt(best)} GHz:", "models": [("cpu", c) for c in top]})
    if facts.get("Socket"):
        faq.append({"q": f"What socket do {codename} processors use?", "a": f"{facts['Socket']}."})
    if facts.get("Memory"):
        faq.append({"q": f"What memory do {codename} processors support?", "a": f"{facts['Memory']} per socket."})
    tdp = span((c.tdp_watts for c in cpus), " W")
    if tdp:
        faq.append({"q": f"What is the TDP of {codename} processors?",
                    "a": f"The models listed here are rated at {tdp}."})
    return faq


# ---------- Comparisons ----------

def comparison_pairs(db: Session) -> List[dict]:
    """The hand-picked comparisons whose two models both exist."""
    by_kind = {"cpu": {c.slug: c for c in all_cpus(db)}, "gpu": {g.slug: g for g in all_gpus(db)}}
    pairs = []
    for kind, slug_a, slug_b in content.COMPARISONS:
        a, b = by_kind[kind].get(slug_a), by_kind[kind].get(slug_b)
        if a and b:
            pairs.append({"kind": kind, "a": a, "b": b, "url": f"/compare/{slug_a}-vs-{slug_b}",
                          "label": f"{a.name} vs {b.name}"})
    return pairs


def comparisons_for(db: Session, kind: str, slug: str) -> List[dict]:
    links = []
    for pair in comparison_pairs(db):
        if pair["kind"] != kind or slug not in (pair["a"].slug, pair["b"].slug):
            continue
        other = pair["b"] if pair["a"].slug == slug else pair["a"]
        links.append({"label": f"vs {other.name}", "url": pair["url"]})
    return links


# (label, attribute, unit, phrase used when one model has more)
CPU_COMPARE_ROWS = (
    ("Generation", "codename", "", None),
    ("Launch year", "launch_year", "", None),
    ("Cores", "cores", "", "more cores"),
    ("Threads", "threads", "", "more threads"),
    ("Max turbo frequency", "max_turbo_frequency_ghz", " GHz", "a higher max turbo frequency"),
    ("L3 cache", "l3_cache_mb", " MB", "more L3 cache"),
    ("TDP", "tdp_watts", " W", "a higher TDP"),
    ("Max memory", "max_memory_tb", " TB", "supports more memory"),
)
GPU_COMPARE_ROWS = (
    ("Vendor", "vendor", "", None),
    ("Architecture", "architecture", "", None),
    ("Launch year", "launch_year", "", None),
    ("Form factor", "form_factor", "", None),
    ("Memory", "memory_gb", " GB", "more memory"),
    ("Memory type", "memory_type", "", None),
    ("Memory bandwidth", "memory_bandwidth_tbs", " TB/s", "more memory bandwidth"),
    ("TDP", "tdp_watts", " W", "a higher TDP"),
    ("FP64 (vector)", "fp64_tflops", " TFLOPS", "more FP64 throughput"),
    ("FP32 (vector)", "fp32_tflops", " TFLOPS", "more FP32 throughput"),
    ("FP16/BF16 Tensor (dense)", "fp16_bf16_tensor_tflops", " TFLOPS", "more FP16/BF16 tensor throughput"),
    ("FP8 Tensor (dense)", "fp8_tensor_tflops", " TFLOPS", "more FP8 tensor throughput"),
    ("FP4 Tensor (dense)", "fp4_tensor_tflops", " TFLOPS", "more FP4 tensor throughput"),
    ("Interconnect", "interconnect", "", None),
    ("Interconnect bandwidth", "interconnect_bandwidth_gbs", " GB/s", "more interconnect bandwidth"),
    ("Host interface", "pcie_interface", "", None),
)


def compare_value(attr: str, value, unit: str) -> str:
    if attr == "max_memory_tb" and value:
        return memory_text(value)
    return fmt(value, unit)


def compare_rows(kind: str, a, b) -> tuple[list, list]:
    """Table rows and plain-language differences, from the data only."""
    rows, differences = [], []
    for label, attr, unit, phrase in (CPU_COMPARE_ROWS if kind == "cpu" else GPU_COMPARE_ROWS):
        va, vb = getattr(a, attr), getattr(b, attr)
        if va in (None, "") and vb in (None, ""):
            continue
        rows.append((label, compare_value(attr, va, unit) or "—", compare_value(attr, vb, unit) or "—"))
        if phrase and isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != vb:
            more, less = (a, b) if va > vb else (b, a)
            hi, lo = (va, vb) if va > vb else (vb, va)
            verb = "" if phrase.startswith("supports") else "has "
            differences.append(f"The {more.name} {verb}{phrase} "
                               f"({compare_value(attr, hi, unit)} vs {compare_value(attr, lo, unit)}).")
    status = [f"The {item.name} is {'validated against manufacturer sources' if item.validated else 'not yet validated'}."
              for item in (a, b)]
    return rows, differences + status


@router.get("/compare", include_in_schema=False)
async def compare_index(request: Request, db: Session = Depends(get_db)):
    pairs = comparison_pairs(db)
    return render(
        request, "compare_index.html",
        crumbs=[("Home", "/"), ("Comparisons", "/compare")],
        title="Server CPU and GPU Comparisons",
        description=("Side-by-side spec comparisons of popular server processors and datacenter GPUs: "
                     "AMD EPYC vs Intel Xeon, NVIDIA H100 vs H200, MI300X vs H100 and more."),
        canonical="/compare",
        heading="Comparisons",
        cpu_pairs=[p for p in pairs if p["kind"] == "cpu"],
        gpu_pairs=[p for p in pairs if p["kind"] == "gpu"],
    )


def find_comparison(db: Session, pair: str, suffix: str = ""):
    """The hand-picked comparison for ``a-vs-b``, a 301 for ``b-vs-a``, else 404."""
    if "-vs-" not in pair:
        raise HTTPException(status_code=404, detail="Not found")
    left, right = pair.split("-vs-", 1)
    pairs = comparison_pairs(db)
    match = next((p for p in pairs if (p["a"].slug, p["b"].slug) == (left, right)), None)
    if match is None:
        reverse = next((p for p in pairs if (p["a"].slug, p["b"].slug) == (right, left)), None)
        if reverse is not None:
            return RedirectResponse(reverse["url"] + suffix, status_code=301), pairs
        raise HTTPException(status_code=404, detail="Not found")
    return match, pairs


def comparison_csv(match: dict) -> str:
    """The comparison as CSV: raw values with units in the labels, then where it came from."""
    kind, a, b = match["kind"], match["a"], match["b"]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Specification", a.name, b.name])
    for label, attr, unit, _ in (CPU_COMPARE_ROWS if kind == "cpu" else GPU_COMPARE_ROWS):
        va, vb = getattr(a, attr), getattr(b, attr)
        if va in (None, "") and vb in (None, ""):
            continue
        writer.writerow([f"{label} ({unit.strip()})" if unit else label,
                         "" if va is None else va, "" if vb is None else vb])
    writer.writerow(["Validated", "yes" if a.validated else "no", "yes" if b.validated else "no"])
    writer.writerow([])
    writer.writerow(["Source", f"Compute Specs DB, {SITE_URL}{match['url']}"])
    writer.writerow(["Licence", f"MIT, please credit Compute Specs DB ({SITE_URL}/about)"])
    writer.writerow(["Downloaded", date.today().isoformat()])
    return buffer.getvalue()


@router.get("/api/compare/{pair}.csv", include_in_schema=False)
async def compare_csv(pair: str, db: Session = Depends(get_db)):
    """Download one comparison as CSV (under /api so it is never indexed)."""
    match, _ = find_comparison(db, pair, suffix="")
    if isinstance(match, RedirectResponse):
        return RedirectResponse(match.headers["location"].replace("/compare/", "/api/compare/", 1) + ".csv",
                                status_code=301)
    slug = match["url"].rsplit("/", 1)[-1]
    return Response(comparison_csv(match), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{slug}.csv"'})


@router.get("/compare/{pair}", include_in_schema=False)
async def compare_page(pair: str, request: Request, db: Session = Depends(get_db)):
    match, pairs = find_comparison(db, pair)
    if isinstance(match, RedirectResponse):
        return match

    kind, a, b = match["kind"], match["a"], match["b"]
    rows, differences = compare_rows(kind, a, b)
    noun = "Specs Comparison"
    facts = ("cores, threads, turbo frequency, L3 cache, TDP and memory" if kind == "cpu"
             else "memory, bandwidth, TDP and tensor TFLOPS")
    return render(
        request, "compare.html",
        crumbs=[("Home", "/"), ("Comparisons", "/compare"), (match["label"], match["url"])],
        title=f"{a.name} vs {b.name}: {noun}",
        title_fallbacks=(f"{a.name} vs {b.name} Specs", f"{a.name} vs {b.name}",
                         f"{without_vendor(a.name)} vs {without_vendor(b.name)} Specs",
                         f"{without_vendor(a.name)} vs {without_vendor(b.name)}"),
        description=fit_description(f"{a.name} vs {b.name}: {facts} side by side.",
                                    f"{a.name} vs {b.name} specs side by side."),
        canonical=match["url"],
        heading=f"{a.name} vs {b.name}",
        kind=kind, a=a, b=b, rows=rows, differences=differences,
        csv_url=f"/api/compare/{match['url'].rsplit('/', 1)[-1]}.csv",
        tool_url=f"/visualizations?kind={kind}&a={quote(a.slug)}&b={quote(b.slug)}#compare-tool",
        source_url=f"{SITE_URL}{match['url']}",
        others=[p for p in pairs if p is not match and p["kind"] == kind],
    )


# ---------- About and llms.txt ----------

@router.get("/about", include_in_schema=False)
async def about_page(request: Request, db: Session = Depends(get_db)):
    cpus, gpus = all_cpus(db), all_gpus(db)
    return render(
        request, "about.html",
        crumbs=[("Home", "/"), ("About", "/about")],
        title="About the Data: Sources and Validation",
        description=("Where Compute Specs DB gets its CPU and GPU specifications, how entries are validated, "
                     "how to report errors and how to use the data."),
        canonical="/about",
        heading="About Compute Specs DB",
        cpu_count=len(cpus), gpu_count=len(gpus),
        validated=sum(c.validated for c in cpus) + sum(g.validated for g in gpus),
        cpu_families=len(group_by(cpus, "family")), codenames=len(group_by(cpus, "codename")),
    )


@router.get("/llms.txt", include_in_schema=False)
async def llms_txt(db: Session = Depends(get_db)):
    cpus, gpus = all_cpus(db), all_gpus(db)
    lines = [
        "# Compute Specs DB",
        "",
        f"> A free, open catalog of specifications for {len(cpus)} server-class CPUs (Intel Xeon, AMD EPYC and "
        f"others) and {len(gpus)} datacenter GPUs (NVIDIA, AMD Instinct, Intel). Each entry is marked as "
        "validated against manufacturer sources or not yet validated. The data is MIT-licensed.",
        "",
        "Every model has its own page at /cpu/<slug> or /gpu/<slug>. When citing a spec, link to that page.",
        "",
        "## Pages",
        "",
        f"- [Server CPUs by family and generation]({SITE_URL}/cpus): overview of all processors",
        f"- [Datacenter GPUs]({SITE_URL}/gpus): overview of all accelerators",
        f"- [Comparisons]({SITE_URL}/compare): side-by-side specs of popular models",
        f"- [About the data]({SITE_URL}/about): sources, validation and how to report errors",
        "",
        "## Data",
        "",
        f"- [All CPUs as CSV]({SITE_URL}/api/export/csv)",
        f"- [All GPUs as CSV]({SITE_URL}/api/export/gpus/csv)",
        f"- [CPUs as JSON]({SITE_URL}/api/cpus?limit=1000)",
        f"- [GPUs as JSON]({SITE_URL}/api/gpus?limit=1000)",
        f"- [API documentation]({SITE_URL}/docs)",
        "- [Source data and history on GitHub](https://github.com/elokwentnie/compute-specs-db)",
        "",
    ]
    return PlainTextResponse("\n".join(lines), headers={"Cache-Control": PAGE_CACHE})
