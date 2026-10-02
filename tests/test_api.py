import csv
import re
import difflib
import io

CPU_PATH = "cpu_spec_validated.csv"
GPU_PATH = "gpu_spec_validated.csv"


def changed_lines(before, after):
    return [
        line for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]


def all_cpus(client, **params):
    return client.get("/api/cpus", params={"limit": 1000, **params}).json()


def proposal_payload(**overrides):
    payload = {
        "cpu_model_name": "Intel Xeon 6980P Test",
        "family": "Intel Xeon",
        "cpu_model": "6980P",
        "cores": 128,
        "threads": 256,
        "tdp_watts": 500,
        "launch_year": 2024,
        "source_url": "https://www.intel.com/content/www/us/en/products/sku/240777.html",
        "notes": "Launched with Granite Rapids. Ping @someone",
        "website": "",
    }
    payload.update(overrides)
    return payload


# ---------- Validated read path ----------

def test_validated_exposed_in_api_and_stats(client):
    cpus = all_cpus(client)
    assert len(cpus) == 428
    assert all(isinstance(cpu["validated"], bool) for cpu in cpus)
    assert len(all_cpus(client, validated="true")) == 212
    assert len(all_cpus(client, validated="false")) == 216

    gpus = client.get("/api/gpus", params={"limit": 1000}).json()
    assert len(gpus) == 35 and sum(g["validated"] for g in gpus) == 17
    assert len(client.get("/api/gpus/search", params={"q": "NVIDIA", "validated": "true"}).json()) > 0

    stats = client.get("/api/stats").json()
    assert (stats["validated_cpus"], stats["unvalidated_cpus"]) == (212, 216)
    assert (stats["validated_gpus"], stats["unvalidated_gpus"]) == (17, 18)


def test_exports_include_validated(client):
    text = client.get("/api/export/csv").text
    rows = list(csv.DictReader(io.StringIO(text), delimiter=";"))
    assert sum(row["Validated"] == "True" for row in rows) == 212
    gpu_text = client.get("/api/export/gpus/csv").text
    assert gpu_text.splitlines()[0].endswith(";Validated")


def test_startup_imports_csv_from_github(tmp_path, monkeypatch, fake_github):
    fake_github.files[GPU_PATH] += "Only On GitHub,TestCo,Z9,PCIe card,8,GDDR6,75,True\n"
    from conftest import _load_app
    client = _load_app(tmp_path, monkeypatch, fake_github, token=True)
    names = [g["gpu_model_name"] for g in client.get("/api/gpus", params={"limit": 1000}).json()]
    assert "Only On GitHub" in names


# ---------- Admin write-through ----------

def test_toggle_validated_commits_one_line(client, fake_github):
    cpu = all_cpus(client, validated="true")[0]
    before = fake_github.files[CPU_PATH]

    response = client.patch(f"/api/cpus/{cpu['id']}/validated", json={"validated": False}, headers=client.admin_headers)

    assert response.status_code == 200 and response.json()["validated"] is False
    assert fake_github.commits[-1].startswith(f"Mark CPU unvalidated: {cpu['cpu_model_name']}")
    assert "[skip render]" in fake_github.commits[-1]
    diff = changed_lines(before, fake_github.files[CPU_PATH])
    assert len(diff) == 2 and diff[1].endswith(",False")
    assert client.get(f"/api/cpus/{cpu['id']}").json()["validated"] is False


def test_edit_and_rename_gpu(client, fake_github):
    gpu = client.get("/api/gpus", params={"limit": 1000}).json()[2]
    before = fake_github.files[GPU_PATH]

    response = client.put(
        f"/api/gpus/{gpu['id']}",
        json={"gpu_model_name": gpu["gpu_model_name"] + " Rev2", "tdp_watts": 999, "validated": True},
        headers=client.admin_headers,
    )

    assert response.status_code == 200
    assert fake_github.commits[-1].startswith("Rename GPU:")
    diff = changed_lines(before, fake_github.files[GPU_PATH])
    assert len(diff) == 2
    assert diff[1].startswith("+" + gpu["gpu_model_name"] + " Rev2,") and diff[1].endswith(",999,True")


def test_create_and_delete_cpu(client, fake_github):
    response = client.post(
        "/api/cpus",
        json={"cpu_model_name": "AMD EPYC 9999 Test", "family": "AMD EPYC", "cores": 8, "validated": True},
        headers=client.admin_headers,
    )
    assert response.status_code == 201
    assert fake_github.files[CPU_PATH].endswith("AMD EPYC 9999 Test,AMD EPYC,,,8,,,,,,,True\r\n")

    response = client.delete(f"/api/cpus/{response.json()['id']}", headers=client.admin_headers)
    assert response.status_code == 204
    assert fake_github.commits[-1].startswith("Remove CPU: AMD EPYC 9999 Test")
    assert "AMD EPYC 9999 Test" not in fake_github.files[CPU_PATH]


def test_duplicate_name_is_rejected(client, fake_github):
    existing = all_cpus(client)[0]["cpu_model_name"]
    commits = len(fake_github.commits)
    response = client.post("/api/cpus", json={"cpu_model_name": existing.upper()}, headers=client.admin_headers)
    assert response.status_code == 409
    assert len(fake_github.commits) == commits


def test_github_failure_leaves_database_unchanged(client, fake_github):
    cpu = all_cpus(client, validated="true")[0]
    fake_github.fail = True

    response = client.patch(f"/api/cpus/{cpu['id']}/validated", json={"validated": False}, headers=client.admin_headers)

    assert response.status_code == 502
    assert client.get(f"/api/cpus/{cpu['id']}").json()["validated"] is True


def test_admin_endpoints_require_auth(client):
    assert client.patch("/api/cpus/1/validated", json={"validated": True}).status_code in (401, 403)
    assert client.get("/api/proposals").status_code in (401, 403)


def test_without_token_changes_only_touch_database(local_client, fake_github):
    cpu = all_cpus(local_client, validated="true")[0]
    response = local_client.patch(f"/api/cpus/{cpu['id']}/validated", json={"validated": False}, headers=local_client.admin_headers)
    assert response.status_code == 200 and response.json()["validated"] is False
    assert fake_github.commits == []
    assert local_client.post("/api/proposals/cpu", json=proposal_payload()).status_code == 503


# ---------- Proposals ----------

def test_proposal_accept_flow(client, fake_github):
    response = client.post("/api/proposals/cpu", json=proposal_payload())
    assert response.status_code == 201
    number = response.json()["issue_number"]
    issue = fake_github.issues[number]
    assert {label["name"] for label in issue["labels"]} == {"proposal", "cpu"}
    assert "`Launched" not in issue["body"] and "@someone" in issue["body"]

    listed = client.get("/api/proposals", headers=client.admin_headers).json()
    assert listed[0]["number"] == number and listed[0]["parsed"]
    assert listed[0]["data"]["cpu_model_name"] == "Intel Xeon 6980P Test"
    assert listed[0]["source_url"].startswith("https://www.intel.com/")
    assert listed[0]["duplicate_of"] is None
    assert [s["name"] for s in listed[0]["similar"]] == ["Intel(R) Xeon(R) 6980P"]

    edited = {**listed[0]["data"], "codename": "Granite Rapids", "max_turbo_frequency_ghz": 3.9}
    response = client.post(f"/api/proposals/{number}/accept", json=edited, headers=client.admin_headers)

    assert response.status_code == 201, response.text
    assert fake_github.commits[-1].startswith(f"Add CPU: Intel Xeon 6980P Test (closes #{number})")
    assert fake_github.files[CPU_PATH].endswith(
        "Intel Xeon 6980P Test,Intel Xeon,6980P,Granite Rapids,128,256,3.9,,500,2024,,True\r\n"
    )
    assert issue["state"] == "closed" and {"name": "accepted"} in issue["labels"]
    added = client.get("/api/cpus/search", params={"q": "6980P Test"}).json()
    assert len(added) == 1 and added[0]["validated"] is True
    assert client.get("/api/proposals", headers=client.admin_headers).json() == []


def test_proposal_reject_flow(client, fake_github):
    number = client.post("/api/proposals/gpu", json={
        "gpu_model_name": "Fake GPU 1", "vendor": "Nobody",
        "source_url": "https://example.com/spec",
    }).json()["issue_number"]

    response = client.post(f"/api/proposals/{number}/reject", json={"reason": "Not a datacenter part"}, headers=client.admin_headers)

    assert response.status_code == 200
    issue = fake_github.issues[number]
    assert issue["state"] == "closed" and issue["state_reason"] == "not_planned"
    assert {"name": "rejected"} in issue["labels"]
    assert "Not a datacenter part" in fake_github.comments[number][-1]
    assert client.post(f"/api/proposals/{number}/accept", json={"gpu_model_name": "Fake GPU 1"}, headers=client.admin_headers).status_code == 409


def test_proposal_validation(client, fake_github):
    assert client.post("/api/proposals/cpu", json=proposal_payload(website="http://spam")).status_code == 400
    assert client.post("/api/proposals/cpu", json=proposal_payload(source_url="not a url")).status_code == 422
    existing = all_cpus(client)[0]["cpu_model_name"]
    assert client.post("/api/proposals/cpu", json=proposal_payload(cpu_model_name=existing)).status_code == 409
    assert fake_github.issues == {}


def test_proposal_rate_limit(client):
    statuses = [
        client.post("/api/proposals/cpu", json=proposal_payload(cpu_model_name=f"Rate Test {i}")).status_code
        for i in range(4)
    ]
    assert statuses == [201, 201, 201, 429]


def test_login_is_rate_limited(client):
    # The fixture already logged in once; the limit is 5/minute.
    statuses = [client.post("/api/auth/login", json={"password": "wrong"}).status_code for _ in range(5)]
    assert statuses == [401, 401, 401, 401, 429]


def test_pages_version_static_assets(client):
    for page in ("/", "/visualizations", "/propose", "/admin"):
        html = client.get(page).text
        assert re.search(r'href="/static/css/common\.css\?v=\w+"', html), page
        assert re.search(r'src="/static/js/theme\.js\?v=\w+"', html), page
        assert 'href="/static/images/server-logo.png"' in html  # images untouched
