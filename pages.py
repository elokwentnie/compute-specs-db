"""Server-rendered HTML pages for CPUs, GPUs and their groups.

Pages are built from the database on each request.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

import proposals
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
    ("Form factor", "form_factor", ""),
    ("Memory", "memory_gb", " GB"),
    ("Memory type", "memory_type", ""),
    ("TDP", "tdp_watts", " W"),
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
        f"supports up to {fmt(cpu.max_memory_tb)} TB of memory" if cpu.max_memory_tb else "",
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
        " ".join(p for p in (fmt(gpu.memory_gb, " GB"), gpu.memory_type or "") if p),
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
    paths = ["/cpus", "/gpus"]
    paths += sorted(f"/cpus/{slugify(f)}" for f in group_by(cpus, "family"))
    duplicates = duplicate_codenames(cpus)
    paths += sorted(f"/cpus/codename/{slugify(c)}" for c in group_by(cpus, "codename") if c not in duplicates)
    paths += sorted(f"/gpus/{slugify(v)}" for v in group_by(gpus, "vendor"))
    paths += sorted(f"/cpu/{c.slug}" for c in cpus)
    paths += sorted(f"/gpu/{g.slug}" for g in gpus)
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
    )


@router.get("/cpu/{slug}", include_in_schema=False)
async def cpu_page(slug: str, request: Request, db: Session = Depends(get_db)):
    """Spec page for one CPU."""
    cpus = all_cpus(db)
    cpu = find_by_slug(cpus, slug)
    crumbs = [("Home", "/"), ("CPUs", "/cpus")]
    if cpu.family:
        crumbs.append((cpu.family, f"/cpus/{slugify(cpu.family)}"))
    if cpu.codename:
        crumbs.append((cpu.codename, f"/cpus/codename/{slugify(cpu.codename)}"))
    crumbs.append((cpu.name, f"/cpu/{cpu.slug}"))
    specs = [(label, fmt(getattr(cpu, attr), unit)) for label, attr, unit in CPU_SPECS]
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
        family_url=f"/cpus/{slugify(cpu.family)}" if cpu.family else None,
        codename_url=f"/cpus/codename/{slugify(cpu.codename)}" if cpu.codename else None,
    )


# ---------- GPU pages ----------

def _gpu_group(request: Request, gpus, crumbs, canonical: str, heading: str, title: str, description: str,
               vendors: Optional[list] = None, title_fallbacks=()):
    """Render a GPU list page."""
    return render(
        request, "gpu_group.html",
        crumbs=crumbs, title=title, title_fallbacks=title_fallbacks, description=description,
        canonical=canonical, heading=heading,
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
    )


@router.get("/gpu/{slug}", include_in_schema=False)
async def gpu_page(slug: str, request: Request, db: Session = Depends(get_db)):
    """Spec page for one GPU."""
    gpus = all_gpus(db)
    gpu = find_by_slug(gpus, slug)
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
        vendor_url=f"/gpus/{slugify(gpu.vendor)}" if gpu.vendor else None,
    )
