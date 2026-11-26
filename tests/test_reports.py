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


# ---------- Reviewing reports in the admin panel ----------

def report(client, kind, item, field, value):
    response = client.post(f"/api/reports/{kind}/{item['id']}", json={"field": field, "suggested_value": value})
    assert response.status_code == 201
    return response.json()["issue_number"]


def test_admin_lists_reports_with_live_values(client, fake_github):
    cpu = first(client, "cpu")
    number = report(client, "cpu", cpu, "tdp_watts", "999")
    fake_github.create_issue("[Data Error] typed by hand", "No data block here", ["data-error"])

    assert client.get("/api/reports").status_code in (401, 403)
    reports = client.get("/api/reports", headers=client.admin_headers).json()
    parsed = next(r for r in reports if r["number"] == number)
    assert parsed["parsed"] and parsed["single_field"]
    assert parsed["name"] == cpu["cpu_model_name"] and parsed["item_id"] == cpu["id"]
    assert parsed["field"] == "tdp_watts" and parsed["suggested"] == "999"
    assert parsed["live_value"] == parsed["reported_current"] == cpu["tdp_watts"]
    assert parsed["page_url"] == f"/cpu/{cpu['slug']}"
    assert any(not r["parsed"] for r in reports)


def test_apply_report_updates_the_csv_and_closes_the_issue(client, fake_github):
    cpu = first(client, "cpu")
    number = report(client, "cpu", cpu, "tdp_watts", "999")

    response = client.post(f"/api/reports/{number}/apply", json={"value": "998"}, headers=client.admin_headers)
    assert response.status_code == 200, response.text
    assert response.json()["new_value"] == 998
    assert client.get(f"/api/cpus/{cpu['id']}").json()["tdp_watts"] == 998
    assert any(line.startswith(cpu["cpu_model_name"]) and ",998," in line
               for line in fake_github.files["cpu_spec_validated.csv"].splitlines())

    issue = fake_github.issues[number]
    assert issue["state"] == "closed" and "fixed" in {label["name"] for label in issue["labels"]}
    assert "changed from" in fake_github.comments[number][-1]
    # A closed report can't be applied twice.
    assert client.post(f"/api/reports/{number}/apply", json={}, headers=client.admin_headers).status_code == 409


def test_apply_report_rejects_bad_values_and_other_field(client, fake_github):
    gpu = first(client, "gpu")
    number = report(client, "gpu", gpu, "memory_gb", "lots")
    response = client.post(f"/api/reports/{number}/apply", json={}, headers=client.admin_headers)
    assert response.status_code == 422
    assert fake_github.issues[number]["state"] == "open"

    other = report(client, "gpu", gpu, "other", "Everything is wrong")
    assert client.post(f"/api/reports/{other}/apply", json={}, headers=client.admin_headers).status_code == 409


def test_close_report_as_dismissed_or_fixed(client, fake_github):
    cpu = first(client, "cpu")
    dismissed = report(client, "cpu", cpu, "cores", "1")
    fixed = report(client, "cpu", cpu, "threads", "2")

    response = client.post(f"/api/reports/{dismissed}/close", json={"fixed": False, "message": "Spec sheet says otherwise."},
                           headers=client.admin_headers)
    assert response.status_code == 200
    issue = fake_github.issues[dismissed]
    assert issue["state_reason"] == "not_planned" and "dismissed" in {l["name"] for l in issue["labels"]}
    assert "Spec sheet says otherwise." in fake_github.comments[dismissed][-1]

    client.post(f"/api/reports/{fixed}/close", json={"fixed": True}, headers=client.admin_headers)
    assert fake_github.issues[fixed]["state_reason"] == "completed"
    assert client.get(f"/api/cpus/{cpu['id']}").json()["cores"] == cpu["cores"]  # closing never changes data
