from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

# Isolate persistence and thresholds before app.main is imported by any test.
os.environ["HIKMAON_DATA_DIR"] = tempfile.mkdtemp(prefix="hikmaon_test_")


def photo_like(seed: int, size: int = 256, noise: float = 0.008) -> Image.Image:
    """Synthetic image with natural-photo (1/f) spectral statistics."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size, 3))
    fy = np.fft.fftfreq(size).reshape(-1, 1)
    fx = np.fft.fftfreq(size).reshape(1, -1)
    freq = np.hypot(fy, fx)
    freq[0, 0] = 1
    for channel in range(3):
        phase = rng.uniform(0, 2 * np.pi, (size, size))
        img = np.real(np.fft.ifft2((1.0 / freq**1.2) * np.exp(1j * phase)))
        img = (img - img.min()) / (img.max() - img.min())
        out[..., channel] = img
    out += rng.normal(0, noise, out.shape)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype("uint8"))


def image_bytes(image: Image.Image, fmt: str = "PNG", **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **save_kwargs)
    return buffer.getvalue()


@pytest.fixture
def make_photo():
    return photo_like


@pytest.fixture
def to_bytes():
    return image_bytes


@pytest.fixture
def make_user():
    """Register a fresh account and return (auth headers, user dict)."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    def _create(role_admin_first: bool = False):
        import uuid as _uuid

        email = f"user_{_uuid.uuid4().hex[:10]}@test.hikmaon"
        response = client.post(
            "/api/auth/register",
            json={"email": email, "password": "Str0ngPassw0rd!", "display_name": "Test User"},
        )
        assert response.status_code == 201, response.text
        tokens = response.json()
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        return headers, tokens["user"], tokens

    return _create


@pytest.fixture
def auth_headers(make_user):
    headers, _user, _tokens = make_user()
    return headers
