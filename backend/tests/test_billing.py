from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from app.main import app, billing_service, store

client = TestClient(app)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def test_plan_catalog_is_public():
    plans = client.get("/api/billing/plans").json()
    names = {p["plan"] for p in plans}
    assert {"free", "pro", "enterprise"} == names
    pro = next(p for p in plans if p["plan"] == "pro")
    assert "crawler" in pro["features"] and "api" in pro["features"]


def test_new_account_is_free_with_usage(make_user):
    headers, _, _ = make_user()
    me = client.get("/api/billing/me", headers=headers).json()
    assert me["plan"] == "free"
    assert me["usage"]["analyses_limit"] == 100
    assert me["usage"]["analyses_used"] == 0


def test_analysis_metering_counts_usage(make_user, make_photo, to_bytes):
    headers, _, _ = make_user()
    client.post(
        "/api/registrations",
        json={"media_type": "image", "filename": "m.png", "content_b64": _b64(to_bytes(make_photo(1)))},
        headers=headers,
    )
    client.post(
        "/api/analyze",
        json={"media_type": "image", "filename": "p.png", "content_b64": _b64(to_bytes(make_photo(2)))},
        headers=headers,
    )
    me = client.get("/api/billing/me", headers=headers).json()
    assert me["usage"]["analyses_used"] == 1


def test_free_analysis_quota_is_enforced(make_user, make_photo, to_bytes):
    headers, user, _ = make_user()
    # Push usage to the free limit directly, then the next call must be blocked.
    billing_service._usage(user["user_id"])["analyses"] = 100
    store.persist()
    blocked = client.post(
        "/api/analyze",
        json={"media_type": "image", "filename": "p.png", "content_b64": _b64(to_bytes(make_photo(3)))},
        headers=headers,
    )
    assert blocked.status_code == 402


def test_crawler_is_gated_to_paid_plans(make_user):
    headers, _, _ = make_user()
    assert client.post("/api/crawler/jobs", json={"seed_urls": ["https://example.com"]}, headers=headers).status_code == 403
    client.post("/api/billing/dev/set-plan", json={"plan": "pro"}, headers=headers)
    # Now allowed through the feature gate (400 = bad seed, i.e. gate passed).
    assert client.post("/api/crawler/jobs", json={"seed_urls": ["ftp://x"]}, headers=headers).status_code == 400


def test_api_keys_require_paid_plan_then_work(make_user, make_photo, to_bytes):
    headers, _, _ = make_user()
    assert client.post("/api/apikeys", json={"name": "k"}, headers=headers).status_code == 403

    client.post("/api/billing/dev/set-plan", json={"plan": "pro"}, headers=headers)
    created = client.post("/api/apikeys", json={"name": "ci"}, headers=headers)
    assert created.status_code == 201
    token = created.json()["api_key"]
    assert token.startswith("hik_live_")

    # The API key authenticates a request with no bearer token.
    analysis = client.post(
        "/api/analyze",
        json={"media_type": "image", "filename": "p.png", "content_b64": _b64(to_bytes(make_photo(5)))},
        headers={"X-API-Key": token},
    )
    assert analysis.status_code == 200

    keys = client.get("/api/apikeys", headers=headers).json()
    assert len(keys) == 1 and "user_id" not in keys[0]
    revoked = client.delete(f"/api/apikeys/{keys[0]['key_id']}", headers=headers)
    assert revoked.status_code == 200
    # Revoked key no longer authenticates.
    assert client.get("/api/billing/me", headers={"X-API-Key": token}).status_code == 401


def test_api_key_limit_enforced(make_user):
    headers, _, _ = make_user()
    client.post("/api/billing/dev/set-plan", json={"plan": "pro"}, headers=headers)
    for _ in range(3):  # pro allows 3
        assert client.post("/api/apikeys", json={"name": "k"}, headers=headers).status_code == 201
    assert client.post("/api/apikeys", json={"name": "k"}, headers=headers).status_code == 402


def test_enterprise_license_activation(make_user):
    headers, user, _ = make_user()
    # Issue a license bound to this account's email (admin path in prod; here
    # we sign directly via the service).
    license_token = billing_service.issue_license(email=user["email"], seats=10, days=365)
    activated = client.post("/api/billing/license/activate", json={"license": license_token}, headers=headers)
    assert activated.status_code == 200
    assert activated.json()["plan"] == "enterprise"

    me = client.get("/api/billing/me", headers=headers).json()
    assert me["plan"] == "enterprise"
    assert me["usage"]["analyses_limit"] >= 1_000_000

    # A license for a different email is rejected.
    other = billing_service.issue_license(email="someone-else@example.com", seats=1, days=30)
    headers2, _, _ = make_user()
    assert client.post("/api/billing/license/activate", json={"license": other}, headers=headers2).status_code == 403

    # A tampered license fails signature verification.
    import json

    blob = json.loads(base64.urlsafe_b64decode(license_token))
    blob["plan"] = "enterprise"
    blob["seats"] = 9999
    tampered = base64.urlsafe_b64encode(json.dumps(blob).encode()).decode()
    assert client.post("/api/billing/license/activate", json={"license": tampered}, headers=headers).status_code == 400


def test_checkout_without_stripe_returns_setup_guidance(make_user):
    headers, _, _ = make_user()
    resp = client.post("/api/billing/checkout", json={"plan": "pro"}, headers=headers)
    assert resp.status_code == 501
    assert "HIKMAON_STRIPE_KEY" in resp.json()["detail"]


def test_stripe_webhook_requires_signature(make_user):
    # Without the webhook secret configured, the endpoint is disabled.
    resp = client.post("/api/billing/webhook", json={"type": "checkout.session.completed"})
    assert resp.status_code in (403, 501)
