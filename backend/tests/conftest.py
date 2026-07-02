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
