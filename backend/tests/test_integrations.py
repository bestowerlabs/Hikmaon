from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from app.main import app, store, webhook_service

client = TestClient(app)


def test_integrations_status_lists_all_providers():
    status = client.get("/api/integrations/status").json()
    providers = {entry["provider"] for entry in status}
    assert {"instagram", "facebook", "snapchat", "google_drive", "dropbox", "x", "reddit"} <= providers
    for entry in status:
        assert "required_env" in entry and len(entry["required_env"]) == 2


def test_oauth_start_unconfigured_returns_setup_instructions():
    response = client.get(
        "/api/connectors/oauth/instagram/start",
        params={"owner_id": "owner-x", "owner_public_key": "pubkey_x_123456"},
    )
    assert response.status_code == 501
    assert "HIKMAON_INSTAGRAM_CLIENT_ID" in response.json()["detail"]


def test_oauth_start_configured_builds_pkce_authorization_url(monkeypatch):
    monkeypatch.setenv("HIKMAON_DROPBOX_CLIENT_ID", "client-abc")
    monkeypatch.setenv("HIKMAON_DROPBOX_CLIENT_SECRET", "secret-xyz")
    response = client.get(
        "/api/connectors/oauth/dropbox/start",
        params={"owner_id": "owner-x", "owner_public_key": "pubkey_x_123456"},
    )
    assert response.status_code == 200
    data = response.json()
    url = data["authorization_url"]
    assert url.startswith("https://www.dropbox.com/oauth2/authorize?")
    assert "client_id=client-abc" in url
    assert "code_challenge=" in url and "code_challenge_method=S256" in url
    assert data["state"]


def test_oauth_unknown_provider_404():
    response = client.get(
        "/api/connectors/oauth/myspace/start",
        params={"owner_id": "owner-x", "owner_public_key": "pubkey_x_123456"},
    )
    assert response.status_code == 404


def test_webhook_get_verification(monkeypatch):
    monkeypatch.setenv("HIKMAON_WEBHOOK_VERIFY_TOKEN", "verify-me")
    response = client.get(
        "/api/webhooks/instagram",
        params={"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "challenge-42"},
    )
    assert response.status_code == 200
    assert response.text == "challenge-42"

    bad = client.get(
        "/api/webhooks/instagram",
        params={"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "x"},
    )
    assert bad.status_code == 403


def test_webhook_post_requires_authentication():
    response = client.post("/api/webhooks/instagram", json={"media_url": "https://x/y.jpg"})
    assert response.status_code == 403


def test_webhook_event_registers_media(monkeypatch, make_photo, to_bytes):
    monkeypatch.setenv("HIKMAON_WEBHOOK_SHARED_SECRET", "hook-secret")

    connected = client.post(
        "/api/connectors",
        json={
            "owner_id": "owner-hook",
            "owner_public_key": "pubkey_hook_123456",
            "provider": "instagram",
            "account_handle": "@hooked",
        },
    ).json()

    media = to_bytes(make_photo(91))
    monkeypatch.setattr(webhook_service, "_download", lambda url: media)

    response = client.post(
        "/api/webhooks/instagram",
        json={
            "connector_id": connected["connector_id"],
            "media_type": "image",
            "filename": "hooked.png",
            "media_url": "https://cdn.instagram.com/hooked.png",
            "source_url": "https://instagram.com/p/hooked",
        },
        headers={"X-Hikmaon-Webhook-Secret": "hook-secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["processed"] == 1
    assert body["results"][0]["status"] == "registered"
    media_id = body["results"][0]["media_id"]
    assert media_id in store.registrations
    assert store.registrations[media_id].certificate_id


def test_model_status_reports_not_deployed_by_default():
    status = client.get("/api/model/status").json()
    assert status["neural_detector"] in ("not_deployed", "loaded")
    assert "note" in status
