"""Phase 3 pages: hand-written intros, FAQ, comparisons, About and llms.txt."""
import re

import content
from utils import model_slug, slugify


def all_items(client, kind):
    return client.get(f"/api/{kind}s", params={"limit": 1000}).json()


def test_content_keys_match_the_data(client):
    """A typo in content.py would silently drop an intro, so every key must exist in the data."""
    cpus, gpus = all_items(client, "cpu"), all_items(client, "gpu")
    assert set(content.GENERATIONS) <= {c["codename"] for c in cpus}
    assert set(content.FAMILIES) <= {c["family"] for c in cpus}
    assert set(content.GPU_VENDORS) <= {g["vendor"] for g in gpus}
    slugs = {"cpu": {c["slug"] for c in cpus}, "gpu": {g["slug"] for g in gpus}}
    for kind, a, b in content.COMPARISONS:
        assert a in slugs[kind] and b in slugs[kind], (kind, a, b)
    pairs = [frozenset((a, b)) for _, a, b in content.COMPARISONS]
    assert len(pairs) == len(set(pairs)), "duplicate comparison"


def test_generation_page_has_intro_facts_and_faq(client):
    html = client.get("/cpus/codename/genoa").text
    assert "EPYC 9004 series from November 2022" in html
    assert "<th scope=\"row\">Socket</th><td>SP5</td>" in html
    assert "How many cores do Genoa processors have?" in html
    assert "What socket do Genoa processors use?" in html
    faq_links = re.findall(r'Which Genoa processor has the most cores\?</h3>\s*</summary>\s*<div class="faq-answer"><p>[^<]*(<a href="/cpu/[^"]+">)', html)
    assert faq_links, "most-cores answer links to the model"


def test_generation_without_written_text_still_works(client):
    html = client.get("/cpus/codename/grace").text
    assert "Frequently asked questions" in html  # data-only FAQ
    assert "platform facts" not in html


def test_family_and_vendor_intros(client):
    assert "Italian city" in client.get("/cpus/amd-epyc").text
    assert "line of datacenter accelerators for HPC and AI" in client.get("/gpus/amd").text
    assert "How to read CPU specs" in client.get("/cpus").text
    assert "How to read GPU specs" in client.get("/gpus").text


def test_epyc_4004_is_listed_as_raphael(client):
    cpus = all_items(client, "cpu")
    raphael = [c for c in cpus if c["codename"] == "Raphael"]
    assert {c["cpu_model"] for c in raphael} >= {"EPYC 4564P", "EPYC 4584PX", "EPYC 4124P"}
    assert not [c for c in cpus if c["cpu_model"] == "EPYC 4564P" and c["codename"] == "Genoa"]
    html = client.get("/cpus/codename/raphael").text
    assert "AM5" in html and all(f'href="/cpu/{c["slug"]}"' in html for c in raphael)


def test_comparison_page(client):
    html = client.get("/compare/amd-epyc-9654-vs-intel-xeon-platinum-8480-plus").text
    assert "<title>AMD EPYC 9654 vs Intel Xeon Platinum 8480+: Specs Comparison" in html
    assert "The AMD EPYC 9654 has more cores (96 vs 56)." in html
    assert 'href="/cpu/amd-epyc-9654"' in html and 'href="/cpu/intel-xeon-platinum-8480-plus"' in html

    html = client.get("/compare/nvidia-h200-sxm-vs-nvidia-h100-sxm").text
    assert "The NVIDIA H200 SXM has more memory (141 GB vs 80 GB)." in html


def test_comparison_urls(client):
    response = client.get("/compare/intel-xeon-platinum-8480-plus-vs-amd-epyc-9654", follow_redirects=False)
    assert response.status_code == 301
    assert response.headers["location"] == "/compare/amd-epyc-9654-vs-intel-xeon-platinum-8480-plus"
    # Only hand-picked pairs exist; arbitrary pairs are not generated.
    assert client.get("/compare/amd-epyc-9654-vs-amd-epyc-7251").status_code == 404
    assert client.get("/compare/nonsense").status_code == 404

    index = client.get("/compare").text
    assert all(f'href="/compare/{a}-vs-{b}"' in index for _, a, b in content.COMPARISONS)
    assert 'href="/compare/amd-epyc-9654-vs-intel-xeon-platinum-8480-plus"' in client.get("/cpu/amd-epyc-9654").text


def test_about_page_and_footer_links(client):
    html = client.get("/about").text
    assert "How validation works" in html and "MIT licence" in html
    for path in ("/", "/about", "/cpus", "/propose", "/visualizations", "/no-such-page"):
        page = client.get(path).text
        assert 'href="/about"' in page and 'href="/compare"' in page, path


def test_llms_txt_links_work(client):
    response = client.get("/llms.txt")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert text.startswith("# Compute Specs DB\n")
    for url in re.findall(r"\((https://computespecsdb\.com[^)]*)\)", text):
        path = url.removeprefix("https://computespecsdb.com")
        assert client.get(path).status_code == 200, path


def test_memory_under_a_terabyte_shows_in_gb(client):
    cpu = next(c for c in all_items(client, "cpu") if c["cpu_model"] == "EPYC 4124P")
    html = client.get(f"/cpu/{cpu['slug']}").text
    assert "<th scope=\"row\">Max memory</th><td>192 GB</td>" in html


def test_slugify_is_stable_for_content_keys():
    assert slugify("Sapphire Rapids HBM") == "sapphire-rapids-hbm"
    assert model_slug("Intel Gaudi 3") == "intel-gaudi-3"


def test_comparison_downloads_carry_the_source(client):
    html = client.get("/compare/nvidia-h200-sxm-vs-nvidia-h100-sxm").text
    assert 'href="/api/compare/nvidia-h200-sxm-vs-nvidia-h100-sxm.csv"' in html
    assert "Source: <a href=\"https://computespecsdb.com/compare/nvidia-h200-sxm-vs-nvidia-h100-sxm\"" in html
    assert "/static/js/compare-export.js" in html
    assert "Memory bandwidth" in html  # GPU comparisons include the newer columns

    response = client.get("/api/compare/nvidia-h200-sxm-vs-nvidia-h100-sxm.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["x-robots-tag"] == "noindex"
    lines = response.text.splitlines()
    assert lines[0] == "Specification,NVIDIA H200 SXM,NVIDIA H100 SXM"
    assert "Memory (GB),141,80" in lines
    assert "Source,\"Compute Specs DB, https://computespecsdb.com/compare/nvidia-h200-sxm-vs-nvidia-h100-sxm\"" in lines

    reverse = client.get("/api/compare/nvidia-h100-sxm-vs-nvidia-h200-sxm.csv", follow_redirects=False)
    assert reverse.status_code == 301
    assert reverse.headers["location"] == "/api/compare/nvidia-h200-sxm-vs-nvidia-h100-sxm.csv"
    assert client.get("/api/compare/amd-epyc-9654-vs-amd-epyc-7251.csv").status_code == 404


def test_homepage_has_no_template_links():
    """Google crawled the literal "/gpu/${...}" from the homepage script once."""
    with open("static/index.html", encoding="utf-8") as file:
        page = file.read()
    assert not re.search(r'href="/(cpu|gpu)/\$\{', page)


def test_comparisons_link_to_the_interactive_tool(client):
    html = client.get("/compare/nvidia-h200-sxm-vs-nvidia-h100-sxm").text
    assert 'href="/visualizations?kind=gpu&amp;a=nvidia-h200-sxm&amp;b=nvidia-h100-sxm#compare-tool"' in html
    assert 'href="/visualizations#compare-tool"' in client.get("/compare").text
    assert 'id="compare-tool"' in client.get("/visualizations").text
