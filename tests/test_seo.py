import html
import json
import re
import xml.etree.ElementTree as ET

PAGES = ("/", "/visualizations", "/propose")


def json_ld(html):
    return [json.loads(block) for block in re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)]


def test_head_requests_mirror_get(client):
    for path in (*PAGES, "/sitemap.xml", "/robots.txt", "/favicon.ico", "/api/cpus"):
        get = client.get(path)
        head = client.head(path)
        assert head.status_code == get.status_code == 200, path
        assert head.content == b"", path
        assert head.headers["content-type"] == get.headers["content-type"], path


def test_api_and_docs_are_noindex_but_pages_are_not(client):
    for path in ("/api", "/api/cpus", "/api/stats", "/docs", "/openapi.json", "/api/cpus/999999"):
        assert client.get(path).headers.get("x-robots-tag") == "noindex", path
    for path in PAGES:
        response = client.get(path)
        assert "x-robots-tag" not in response.headers, path
        assert response.headers["cache-control"] == "public, max-age=300", path


def test_unknown_page_returns_html_404(client):
    response = client.get("/no-such-page")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert '<meta name="robots" content="noindex">' in response.text
    assert response.headers["cache-control"] == "no-store"


def test_unknown_api_path_keeps_json_404(client):
    for path in ("/api/no-such-endpoint", "/api/cpus/999999"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert response.headers["content-type"] == "application/json", path


def sitemap_paths(client):
    root = ET.fromstring(client.get("/sitemap.xml").content)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [loc.text for loc in root.findall("s:url/s:loc", ns)]
    assert all(loc.startswith("https://computespecsdb.com/") for loc in locs)
    return [loc.removeprefix("https://computespecsdb.com") for loc in locs]


def test_sitemap_lists_every_page_and_only_pages(client):
    paths = sitemap_paths(client)
    assert len(paths) == len(set(paths))
    assert paths[:3] == list(PAGES)
    assert not any(p.startswith(("/api", "/docs", "/admin")) for p in paths)

    cpus = client.get("/api/cpus", params={"limit": 1000}).json()
    gpus = client.get("/api/gpus", params={"limit": 1000}).json()
    assert {f"/cpu/{c['slug']}" for c in cpus} <= set(paths)
    assert {f"/gpu/{g['slug']}" for g in gpus} <= set(paths)
    assert {"/cpus", "/gpus", "/cpus/amd-epyc", "/cpus/codename/sapphire-rapids", "/gpus/nvidia"} <= set(paths)

    for path in paths:
        response = client.get(path)
        assert response.status_code == 200, path
        assert f'<link rel="canonical" href="https://computespecsdb.com{path}">' in response.text, path
        description = html.unescape(re.search(r'<meta name="description" content="([^"]*)"', response.text).group(1))
        assert 50 <= len(description) <= 160, (path, len(description))


def test_favicon_is_a_real_icon(client):
    response = client.get("/favicon.ico")
    assert response.headers["content-type"] == "image/x-icon"
    assert response.content[:4] == b"\x00\x00\x01\x00"


def test_pages_have_social_preview_tags(client):
    for path in PAGES:
        html = client.get(path).text
        assert 'property="og:image" content="https://computespecsdb.com/static/images/og-image.png"' in html, path
        assert 'name="twitter:card" content="summary_large_image"' in html, path
        assert f'<link rel="canonical" href="https://computespecsdb.com{path}">' in html, path
    for asset in ("/static/images/og-image.png", "/static/images/apple-touch-icon.png"):
        assert client.get(asset).status_code == 200, asset


def test_structured_data_is_valid_and_downloads_exist(client):
    blocks = {block["@type"]: block for block in json_ld(client.get("/").text)}
    assert "potentialAction" not in blocks["WebSite"]  # no search box Google can't use
    dataset = blocks["Dataset"]
    assert len(dataset["description"]) >= 50
    for download in dataset["distribution"]:
        url = download["contentUrl"].replace("https://computespecsdb.com", "")
        assert client.get(url).status_code == 200, url


def page_title(html_text):
    return html.unescape(re.search(r"<title>(.*?)</title>", html_text, re.S).group(1))


def test_titles_are_unique_and_fit_in_search_results(client):
    titles = {}
    for path in sitemap_paths(client)[3:]:
        title = page_title(client.get(path).text)
        assert len(title) <= 60, (path, title)
        assert title not in titles, (path, titles.get(title))
        titles[title] = path
    # Short titles keep the site name.
    assert page_title(client.get("/gpus/amd").text) == "AMD Datacenter GPU List: Specs of All 9 Models"
    assert page_title(client.get("/cpus/codename/genoa").text).endswith(" — Compute Specs DB")


def test_codename_page_identical_to_a_family_page_points_to_it(client):
    paths = sitemap_paths(client)
    assert "/cpus/codename/grace" not in paths and "/cpus/nvidia-grace" in paths
    duplicate = client.get("/cpus/codename/grace")
    assert duplicate.status_code == 200
    assert '<link rel="canonical" href="https://computespecsdb.com/cpus/nvidia-grace">' in duplicate.text
    # A codename spanning several families keeps its own page.
    assert "/cpus/codename/sapphire-rapids" in paths


def test_one_model_groups_use_singular_wording(client):
    for path in ("/cpus/nvidia-grace", "/gpus/intel"):
        text = client.get(path).text
        assert "1 models" not in text and "All 1 " not in text and "1 NVIDIA Grace processors" not in text, path
    assert "Specifications of 1 NVIDIA Grace processor (" in client.get("/cpus/nvidia-grace").text


def test_trailing_slash_redirect_is_permanent_and_relative(client):
    response = client.get("/cpus/", follow_redirects=False)
    assert response.status_code == 308
    assert response.headers["location"] == "/cpus"
    assert client.get("/cpus/").url.path == "/cpus"
