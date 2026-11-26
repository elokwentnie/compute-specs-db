"""Old spec page URLs keep working after a model is renamed."""
import redirects
from conftest import _load_app


def cpu_by_model(client, model):
    return next(c for c in client.get("/api/cpus", params={"limit": 1000}).json() if c["cpu_model"] == model)


def rename(client, item_id, name):
    response = client.put(f"/api/cpus/{item_id}", json={"cpu_model_name": name}, headers=client.admin_headers)
    assert response.status_code == 200, response.text
    return response.json()


def test_apply_rename_follows_the_model():
    rows = redirects.apply_rename([], "cpu", "Old Name", "Middle Name")
    assert rows == [("cpu", "old-name", "Middle Name")]
    rows = redirects.apply_rename(rows, "cpu", "Middle Name", "New Name")
    assert sorted(rows) == [("cpu", "middle-name", "New Name"), ("cpu", "old-name", "New Name")]
    # Renaming back makes the old URL live again and drops its redirect.
    rows = redirects.apply_rename(rows, "cpu", "New Name", "Old Name")
    assert sorted(rows) == [("cpu", "middle-name", "Old Name"), ("cpu", "new-name", "Old Name")]
    # A change that keeps the same URL (e.g. trademark symbols) needs no redirect.
    assert redirects.apply_rename([], "cpu", "Intel(R) Xeon(R) Gold 6148", "Intel Xeon Gold 6148") == []


def test_renamed_model_redirects(client, fake_github):
    cpu = cpu_by_model(client, "EPYC 9654")
    old_url = f"/cpu/{cpu['slug']}"
    renamed = rename(client, cpu["id"], "AMD EPYC 9654 Renamed Test")

    response = client.get(old_url, follow_redirects=False)
    assert response.status_code == 301
    assert response.headers["location"] == f"/cpu/{renamed['slug']}"
    assert client.get(response.headers["location"]).status_code == 200
    assert "amd-epyc-9654,AMD EPYC 9654 Renamed Test" in fake_github.files["redirects.csv"]

    # A second rename keeps the first old URL working, without a redirect chain.
    final = rename(client, cpu["id"], "AMD EPYC 9654 Final Test")
    for url in (old_url, f"/cpu/{renamed['slug']}"):
        response = client.get(url, follow_redirects=False)
        assert response.headers["location"] == f"/cpu/{final['slug']}", url


def test_redirects_survive_a_restart(tmp_path, monkeypatch, fake_github):
    fake_github.files["redirects.csv"] = "Kind,Old Slug,Model Name\ncpu,amd-epyc-9654-old,AMD EPYC 9654 96-Core Processor\n"
    fake_github.versions["redirects.csv"] = 1
    client = _load_app(tmp_path, monkeypatch, fake_github, token=True)
    response = client.get("/cpu/amd-epyc-9654-old", follow_redirects=False)
    assert response.status_code == 301
    assert response.headers["location"] == "/cpu/amd-epyc-9654"


def test_unknown_and_stale_redirects_are_404(client, fake_github):
    assert client.get("/cpu/never-existed").status_code == 404
    cpu = cpu_by_model(client, "EPYC 9554")
    rename(client, cpu["id"], "AMD EPYC 9554 Gone Test")
    client.delete(f"/api/cpus/{cpu['id']}", headers=client.admin_headers)
    assert client.get(f"/cpu/{cpu['slug']}", follow_redirects=False).status_code == 404
