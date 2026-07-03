from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _email() -> str:
    return f"auth_{uuid.uuid4().hex[:10]}@test.hikmaon"


def test_register_login_me_flow():
    email = _email()
    registered = client.post(
        "/api/auth/register",
        json={"email": email, "password": "CorrectHorse7!", "display_name": "Ayan"},
    )
    assert registered.status_code == 201
    tokens = registered.json()
    assert tokens["access_token"] and tokens["refresh_token"]
    assert tokens["user"]["owner_public_key"]  # Ed25519 identity issued at signup

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == email
    assert "password_hash" not in body and "signing_key_ciphertext" not in body

    login = client.post("/api/auth/login", json={"email": email, "password": "CorrectHorse7!"})
    assert login.status_code == 200


def test_password_policy_enforced():
    weak_cases = ["short1A", "alllettersonly", "1234567890123", "password123"]
    for password in weak_cases:
        response = client.post(
            "/api/auth/register",
            json={"email": _email(), "password": password, "display_name": "X"},
        )
        assert response.status_code == 400, password


def test_duplicate_email_rejected():
    email = _email()
    client.post("/api/auth/register", json={"email": email, "password": "GoodPass123!", "display_name": "A"})
    duplicate = client.post(
        "/api/auth/register", json={"email": email, "password": "GoodPass123!", "display_name": "B"}
    )
    assert duplicate.status_code == 409


def test_wrong_password_and_lockout():
    email = _email()
    client.post("/api/auth/register", json={"email": email, "password": "GoodPass123!", "display_name": "A"})
    for _ in range(5):
        bad = client.post("/api/auth/login", json={"email": email, "password": "WrongPass123!"})
        assert bad.status_code == 401
    locked = client.post("/api/auth/login", json={"email": email, "password": "GoodPass123!"})
    assert locked.status_code == 429


def test_refresh_rotation_and_reuse_detection():
    email = _email()
    tokens = client.post(
        "/api/auth/register", json={"email": email, "password": "GoodPass123!", "display_name": "A"}
    ).json()

    first_refresh = tokens["refresh_token"]
    rotated = client.post("/api/auth/refresh", json={"refresh_token": first_refresh})
    assert rotated.status_code == 200
    second_refresh = rotated.json()["refresh_token"]
    assert second_refresh != first_refresh

    # Re-using the rotated token is a theft signal: whole family revoked.
    reuse = client.post("/api/auth/refresh", json={"refresh_token": first_refresh})
    assert reuse.status_code == 401
    killed = client.post("/api/auth/refresh", json={"refresh_token": second_refresh})
    assert killed.status_code == 401


def test_protected_routes_require_token(make_photo, to_bytes):
    import base64

    assert client.get("/api/registrations").status_code == 401
    assert client.get("/api/incidents").status_code == 401
    assert client.post("/api/crawler/jobs", json={"seed_urls": ["https://example.com"]}).status_code == 401
    payload = {
        "media_type": "image",
        "filename": "x.png",
        "content_b64": base64.b64encode(to_bytes(make_photo(1))).decode(),
    }
    assert client.post("/api/registrations", json=payload).status_code == 401

    garbage = client.get("/api/registrations", headers={"Authorization": "Bearer not.a.token"})
    assert garbage.status_code == 401


def test_user_scoping_between_accounts(make_user, make_photo, to_bytes):
    import base64

    headers_a, user_a, _ = make_user()
    headers_b, user_b, _ = make_user()

    created = client.post(
        "/api/registrations",
        json={
            "media_type": "image",
            "filename": "mine.png",
            "content_b64": base64.b64encode(to_bytes(make_photo(71))).decode(),
        },
        headers=headers_a,
    )
    assert created.status_code == 200
    record = created.json()
    assert record["owner_id"] == user_a["user_id"]
    assert record["ownership_proven"] is True  # auto-signed with account key

    mine = client.get("/api/registrations", headers=headers_a).json()
    theirs = client.get("/api/registrations", headers=headers_b).json()
    assert any(r["media_id"] == record["media_id"] for r in mine)
    assert not any(r["media_id"] == record["media_id"] for r in theirs)
