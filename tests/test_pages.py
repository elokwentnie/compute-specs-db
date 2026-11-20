import json
import re

import pytest

from utils import display_name, model_slug


def json_ld(html):
    return [json.loads(block) for block in re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)]


@pytest.mark.parametrize("raw, name, slug", [
    ("Intel(R) Xeon(R) Gold 6330 CPU @ 2.00", "Intel Xeon Gold 6330", "intel-xeon-gold-6330"),
    ("AMD EPYC 9654 96-Core Processor", "AMD EPYC 9654", "amd-epyc-9654"),
    ("Intel(R) Xeon(R) Platinum 8480+", "Intel Xeon Platinum 8480+", "intel-xeon-platinum-8480-plus"),
    ("Intel (R) Xeon (R) CPU Max 9462", "Intel Xeon Max 9462", "intel-xeon-max-9462"),
    ("Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20", "Intel Xeon E5-2630 v4", "intel-xeon-e5-2630-v4"),
    ("AMD Opteron(tm) Processor 6376", "AMD Opteron 6376", "amd-opteron-6376"),
    ("NVIDIA H100 SXM", "NVIDIA H100 SXM", "nvidia-h100-sxm"),
])
def test_display_name_and_slug(raw, name, slug):
    assert display_name(raw) == name
    assert model_slug(raw) == slug


def test_every_model_has_a_unique_slug(client):
    for kind in ("cpus", "gpus"):
        slugs = [item["slug"] for item in client.get(f"/api/{kind}", params={"limit": 1000}).json()]
        assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", s) for s in slugs)
        assert len(slugs) == len(set(slugs)), kind


def test_cpu_page(client):
    html = client.get("/cpu/intel-xeon-platinum-8480-plus").text
    assert "<title>Intel Xeon Platinum 8480+ Specs: 56 Cores, 105 MB L3, 350 W</title>" in html
    assert "<h1>Intel Xeon Platinum 8480+ specifications</h1>" in html
    assert '<a href="/cpus/codename/sapphire-rapids">Sapphire Rapids</a>' in html
    assert '<a href="/cpus/intel-xeon-platinum">Intel Xeon Platinum</a>' in html
    assert "Similar processors" in html and html.count('href="/cpu/') >= 5
    assert ('<meta name="description" content="Intel Xeon Platinum 8480+ specs: 56 cores/112 threads, 3.8 GHz turbo, '
            '105 MB L3, 350 W TDP, up to 4 TB memory. Sapphire Rapids, 2023. Validated.">') in html

    breadcrumbs = json_ld(html)[0]
    assert breadcrumbs["@type"] == "BreadcrumbList"
    assert [item["name"] for item in breadcrumbs["itemListElement"]] == [
        "Home", "CPUs", "Intel Xeon Platinum", "Sapphire Rapids", "Intel Xeon Platinum 8480+"]


def test_gpu_page(client):
    html = client.get("/gpu/nvidia-h100-sxm").text
    assert "<h1>NVIDIA H100 SXM specifications</h1>" in html
    assert "80 GB" in html and "HBM3" in html and "700 W" in html
    assert '<a href="/gpus/nvidia">NVIDIA</a>' in html


def test_overview_pages_link_to_every_member(client):
    cpus = client.get("/api/cpus", params={"limit": 1000}).json()
    epyc = [c for c in cpus if c["family"] == "AMD EPYC"]
    html = client.get("/cpus/amd-epyc").text
    assert all(f'href="/cpu/{c["slug"]}"' in html for c in epyc)
    assert f"Specs of All {len(epyc)} Models" in html

    genoa = [c for c in cpus if c["codename"] == "Genoa"]
    html = client.get("/cpus/codename/genoa").text
    assert all(f'href="/cpu/{c["slug"]}"' in html for c in genoa)

    html = client.get("/cpus").text
    assert 'href="/cpus/codename/turin"' in html and 'href="/cpus/intel-xeon-gold"' in html


def test_unknown_spec_pages_are_html_404s(client):
    for path in ("/cpu/no-such-cpu", "/gpu/no-such-gpu", "/cpus/no-such-family", "/cpus/codename/nope", "/gpus/nope"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert "Page not found" in response.text, path


def test_homepage_is_crawlable_without_javascript(client):
    html = client.get("/").text
    assert "<!--SSR:" not in html
    assert len(re.findall(r'<tbody id="tableBody">.*?</tbody>', html, re.S)[0].split("<tr>")) - 1 == 25
    first = client.get("/api/cpus", params={"limit": 1}).json()[0]
    assert f'href="/cpu/{first["slug"]}"' in html
    assert 'href="/cpus/amd-epyc"' in html and 'href="/cpus/codename/sapphire-rapids"' in html
    assert 'href="/gpus/nvidia"' in html


def test_model_names_are_escaped(client, fake_github):
    response = client.post(
        "/api/cpus",
        json={"cpu_model_name": "Test <script>alert(1)</script> CPU", "family": "AMD EPYC", "codename": "Genoa"},
        headers=client.admin_headers,
    )
    assert response.status_code == 201
    slug = response.json()["slug"]
    for path in (f"/cpu/{slug}", "/cpus/codename/genoa", "/cpus/amd-epyc"):
        html = client.get(path).text
        assert "<script>alert(1)</script>" not in html, path
        assert "&lt;script&gt;" in html, path


def test_names_that_share_a_page_url_are_rejected(client, fake_github):
    # "AMD EPYC 9654 96-Core Processor" already exists and lives at /cpu/amd-epyc-9654.
    response = client.post("/api/cpus", json={"cpu_model_name": "AMD EPYC 9654"}, headers=client.admin_headers)
    assert response.status_code == 409
