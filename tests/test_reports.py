"""Data-error reports sent from the CPU and GPU spec pages."""


def first(client, kind):
    return client.get(f"/api/{kind}s", params={"limit": 1}).json()[0]


def test_report_creates_a_github_issue(client, fake_github):
    cpu = first(client, "cpu")
    response = client.post(f"/api/reports/cpu/{cpu['id']}", json={
        "field": "tdp_watts",
        "suggested_value": "  999   W ",
        "source_url": "https://www.amd.com/en/products/specifications.html",
        "notes": "Checked the spec sheet.",
    })
    assert response.status_code == 201
    number = response.json()["issue_number"]
    assert response.json()["issue_url"].endswith(f"/issues/{number}")

    issue = fake_github.issues[number]
    assert {label["name"] for label in issue["labels"]} == {"data-error", "cpu"}
    assert issue["title"].startswith("[Data Error] ") and issue["title"].endswith(": TDP (W)")
    body = issue["body"]
    assert f"| Current value | `{cpu['tdp_watts']}` |" in body
    assert "| Suggested value | `999 W` |" in body
    assert f"https://computespecsdb.com/cpu/{cpu['slug']}" in body
    assert "Checked the spec sheet." in body


def test_report_without_source_or_notes(client, fake_github):
    gpu = first(client, "gpu")
    response = client.post(f"/api/reports/gpu/{gpu['id']}", json={"field": "other", "suggested_value": "Wrong memory and TDP"})
    assert response.status_code == 201
    issue = fake_github.issues[response.json()["issue_number"]]
    assert {label["name"] for label in issue["labels"]} == {"data-error", "gpu"}
    assert "| Current value | — |" in issue["body"]
    assert "| Source | — |" in issue["body"]


def test_report_text_cannot_break_out_of_the_issue_markup(client, fake_github):
    cpu = first(client, "cpu")
    response = client.post(f"/api/reports/cpu/{cpu['id']}", json={
        "field": "cores",
        "suggested_value": "`@someone` | **bold**",
        "notes": "```\n@everyone\n```",
    })
    body = fake_github.issues[response.json()["issue_number"]]["body"]
    assert "| Suggested value | `'@someone' | **bold**` |" in body
    assert "```\n@everyone\n```" not in body and "'''" in body


def test_invalid_reports_are_rejected(client, fake_github):
    cpu = first(client, "cpu")
    url = f"/api/reports/cpu/{cpu['id']}"
    assert client.post(url, json={"field": "price", "suggested_value": "1"}).status_code == 422
    assert client.post(url, json={"field": "cores", "suggested_value": "   "}).status_code == 422
    assert client.post(url, json={"field": "cores", "suggested_value": "8", "website": "spam"}).status_code == 400
    assert client.post("/api/reports/cpu/999999", json={"field": "cores", "suggested_value": "8"}).status_code == 404
    assert fake_github.issues == {}


def test_reports_are_rate_limited(client, fake_github):
    cpu = first(client, "cpu")
    url = f"/api/reports/cpu/{cpu['id']}"
    codes = [client.post(url, json={"field": "cores", "suggested_value": str(n)}).status_code for n in range(4)]
    assert codes == [201, 201, 201, 429]


def test_report_needs_github_configured(local_client):
    cpu = first(local_client, "cpu")
    response = local_client.post(f"/api/reports/cpu/{cpu['id']}", json={"field": "cores", "suggested_value": "8"})
    assert response.status_code == 503


def test_report_when_github_is_down(client, fake_github):
    fake_github.fail = True
    cpu = first(client, "cpu")
    response = client.post(f"/api/reports/cpu/{cpu['id']}", json={"field": "cores", "suggested_value": "8"})
    assert response.status_code == 502


def test_spec_pages_have_the_report_form(client):
    for kind in ("cpu", "gpu"):
        item = first(client, kind)
        html = client.get(f"/{kind}/{item['slug']}").text
        assert f'data-endpoint="/api/reports/{kind}/{item["id"]}"' in html, kind
        assert 'href="#report"' in html, kind
        assert '<option value="tdp_watts"' in html, kind
        assert "/static/js/report.js" in html, kind
        assert "Report it on GitHub" not in html, kind
