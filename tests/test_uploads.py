"""Admin CSV uploads go through python-multipart; keep them working and keep the parser safe."""
import time

import pytest


def test_admin_can_upload_a_cpu_csv(local_client):
    csv_text = (
        "CPU Model Name,Family,CPU Model,Codename,Cores,Threads,Max Turbo Frequency (GHz),"
        "L3 Cache (MB),TDP (W),Launch Year,Max Memory (TB),Validated\n"
        "Upload Test CPU 1234,Test Family,Test 1234,Testlake,8,16,3.5,32,120,2026,1,False\n"
    )
    response = local_client.post(
        "/api/import/csv-file",
        files={"file": ("cpus.csv", csv_text.encode(), "text/csv")},
        headers=local_client.admin_headers,
    )
    assert response.status_code == 200, response.text
    names = [c["cpu_model_name"] for c in local_client.get("/api/cpus", params={"limit": 1000}).json()]
    assert "Upload Test CPU 1234" in names


def test_semicolon_form_body_is_not_slow(local_client):
    """python-multipart < 0.0.30 needed ~7 s of CPU for this 1 MB body, even without a login."""
    start = time.monotonic()
    response = local_client.post(
        "/api/import/csv-file",
        content=b"a;" * 500_000,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code in (401, 403, 422)
    assert time.monotonic() - start < 2


BOUNDARY = b"BOUNDARY"
PART = (b"--" + BOUNDARY + b'\r\nContent-Disposition: form-data; name="file"; filename="x.csv"\r\n'
        b"Content-Type: text/csv\r\n\r\na,b\r\n")
CLOSE = b"--" + BOUNDARY + b"--\r\n"

# Malformed multipart bodies from python-multipart advisories (fixed in 0.0.26/0.0.27).
# On 0.0.22 the header cases took 5-7 s and the preamble ~2 s, all before the login check.
SLOW_MULTIPART_BODIES = {
    "huge header value": b"--" + BOUNDARY + b'\r\nContent-Disposition: form-data; name="file"\r\nX-Junk: '
                         + b"a" * 20_000_000,
    "many header lines": b"--" + BOUNDARY + b"\r\n" + b"X-A: 1\r\n" * 1_000_000,
    "long CR/LF preamble": b"\r\n" * 10_000_000 + PART + CLOSE,
    "long epilogue": PART + CLOSE + b"x\r\n" * 7_000_000,
}


@pytest.mark.parametrize("name", SLOW_MULTIPART_BODIES)
def test_malformed_multipart_bodies_are_not_slow(local_client, name):
    body = SLOW_MULTIPART_BODIES[name]
    start = time.monotonic()
    response = local_client.post(
        "/api/import/csv-file",
        content=body,
        headers={"Content-Type": "multipart/form-data; boundary=BOUNDARY"},
    )
    assert response.status_code in (400, 401, 403, 422), name
    assert time.monotonic() - start < 1.5, name
