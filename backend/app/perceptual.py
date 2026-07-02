"""Perceptual fingerprinting and visual similarity primitives.

Implements real perceptual matching so that *edited* copies of registered
media (re-encoded, resized, brightness-shifted, lightly cropped) still match:

- 64-bit DCT perceptual hash (pHash) for images
- 64-bit difference hash (dHash) as a secondary signal
- A handcrafted visual feature embedding (DCT energy + color distribution +
  edge orientation) used for cosine similarity
- Chunk-based rolling fingerprints for media we cannot decode in-process
  (video/audio) so exact and partially-trimmed copies still match

The embedding is deliberately model-free so it runs anywhere; the interface
matches what a learned encoder (CLIP/ViT/ArcFace) would provide, so a model
server can replace `embed_image` without touching callers.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

PHASH_BITS = 64
EMBEDDING_VERSION = "visual-features-v1"
PHASH_VERSION = "phash-dct-v1"
CHUNK_SIZE = 64 * 1024
CHUNK_STRIDE = 32 * 1024


def decode_image(raw_bytes: bytes) -> Image.Image | None:
    """Decode bytes into an RGB image, or None if not decodable."""
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
        return image.convert("RGB")
    except Exception:
        return None


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    mat = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2.0 * n))
    mat[0, :] = np.sqrt(1.0 / n)
    return mat


_DCT32 = _dct_matrix(32)
_DCT64 = _dct_matrix(64)


def _dct2(block: np.ndarray, mat: np.ndarray) -> np.ndarray:
    return mat @ block @ mat.T


def _grayscale(image: Image.Image, size: int) -> np.ndarray:
    gray = image.convert("L").resize((size, size), Image.LANCZOS)
    return np.asarray(gray, dtype=np.float64)


def phash(image: Image.Image) -> int:
    """64-bit DCT perceptual hash (classic pHash)."""
    pixels = _grayscale(image, 32)
    coeffs = _dct2(pixels, _DCT32)[:8, :8].flatten()
    coeffs = coeffs[1:]  # drop DC term
    median = np.median(coeffs)
    bits = coeffs > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def dhash(image: Image.Image) -> int:
    """64-bit difference hash (row-wise gradient sign)."""
    gray = image.convert("L").resize((9, 8), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def hash_similarity(a: int, b: int, bits: int = PHASH_BITS) -> float:
    """1.0 for identical hashes, ~0.5 for unrelated images."""
    return 1.0 - (hamming(a, b) / bits)


def embed_image(image: Image.Image) -> np.ndarray:
    """Handcrafted visual embedding: DCT energy + color + edge orientation.

    Each block is L2-normalized before weighting so no component dominates.
    """
    # 1) Low-frequency DCT coefficients capture global structure.
    pixels = _grayscale(image, 64)
    pixels = (pixels - pixels.mean()) / (pixels.std() + 1e-8)
    coeffs = _dct2(pixels, _DCT64)[:12, :12].flatten()[1:]
    dct_feat = np.sign(coeffs) * np.log1p(np.abs(coeffs))
    dct_feat /= np.linalg.norm(dct_feat) + 1e-8

    small = np.asarray(image.resize((64, 64), Image.LANCZOS), dtype=np.float64)

    # 2) Color distribution (Hellinger-scaled RGB histogram, 4x4x4 bins).
    quant = np.clip((small // 64).astype(int), 0, 3)
    idx = quant[..., 0] * 16 + quant[..., 1] * 4 + quant[..., 2]
    hist = np.bincount(idx.flatten(), minlength=64).astype(np.float64)
    hist = np.sqrt(hist / (hist.sum() + 1e-8))
    hist /= np.linalg.norm(hist) + 1e-8

    # 3) Edge orientation histogram over a 3x3 spatial grid (8 bins each).
    gray = small.mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude = np.hypot(gx, gy)
    orientation = np.arctan2(gy, gx)  # -pi..pi
    bins = np.clip(((orientation + np.pi) / (2 * np.pi) * 8).astype(int), 0, 7)
    edge_feat = np.zeros(9 * 8)
    for row in range(3):
        for col in range(3):
            cell_bins = bins[row * 21 : (row + 1) * 21 + 1, col * 21 : (col + 1) * 21 + 1]
            cell_mag = magnitude[row * 21 : (row + 1) * 21 + 1, col * 21 : (col + 1) * 21 + 1]
            cell = np.bincount(cell_bins.flatten(), weights=cell_mag.flatten(), minlength=8)
            edge_feat[(row * 3 + col) * 8 : (row * 3 + col + 1) * 8] = cell
    edge_feat = np.sqrt(edge_feat / (edge_feat.sum() + 1e-8))
    edge_feat /= np.linalg.norm(edge_feat) + 1e-8

    vector = np.concatenate([1.2 * dct_feat, 1.0 * hist, 1.0 * edge_feat])
    return vector / (np.linalg.norm(vector) + 1e-8)


def cosine(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def chunk_fingerprints(raw_bytes: bytes) -> list[str]:
    """Rolling chunk hashes for undecodable media (video/audio placeholder).

    Tolerates prepend/append/trim; a production system replaces this with
    frame-level perceptual hashing (vPDQ/TMK) and audio fingerprinting.
    """
    if len(raw_bytes) <= CHUNK_SIZE:
        return [hashlib.sha256(raw_bytes).hexdigest()[:16]]
    prints = []
    for start in range(0, max(len(raw_bytes) - CHUNK_SIZE + 1, 1), CHUNK_STRIDE):
        prints.append(hashlib.sha256(raw_bytes[start : start + CHUNK_SIZE]).hexdigest()[:16])
    return prints


def chunk_similarity(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    a, b = set(left), set(right)
    return len(a & b) / len(a | b)


@dataclass
class MediaFingerprint:
    """Complete perceptual identity of one media item."""

    media_kind: str  # "image" | "binary"
    phash_hex: str | None = None
    dhash_hex: str | None = None
    embedding: list[float] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)

    @property
    def commitment(self) -> str:
        """Hash commitment of the fingerprint, suitable for on-chain anchoring."""
        material = f"{self.media_kind}|{self.phash_hex}|{self.dhash_hex}|{','.join(self.chunks)}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


def fingerprint_media(raw_bytes: bytes) -> MediaFingerprint:
    image = decode_image(raw_bytes)
    if image is not None:
        return MediaFingerprint(
            media_kind="image",
            phash_hex=format(phash(image), "016x"),
            dhash_hex=format(dhash(image), "016x"),
            embedding=embed_image(image).tolist(),
            chunks=chunk_fingerprints(raw_bytes),
        )
    return MediaFingerprint(media_kind="binary", chunks=chunk_fingerprints(raw_bytes))


def match_percentage(probe: MediaFingerprint, registered: MediaFingerprint) -> dict:
    """Compare two fingerprints; returns component scores and a 0-100%."""
    if probe.media_kind == "image" and registered.media_kind == "image":
        p_sim = hash_similarity(int(probe.phash_hex, 16), int(registered.phash_hex, 16))
        d_sim = hash_similarity(int(probe.dhash_hex, 16), int(registered.dhash_hex, 16))
        emb_sim = cosine(probe.embedding, registered.embedding)
        # Rescale: unrelated images sit near 0.5 hash similarity and ~0.4
        # embedding cosine; identical media sits at 1.0 on all components.
        p_score = max(0.0, min(1.0, (p_sim - 0.5) / 0.5))
        d_score = max(0.0, min(1.0, (d_sim - 0.5) / 0.5))
        e_score = max(0.0, min(1.0, (emb_sim - 0.4) / 0.6))
        combined = 0.45 * p_score + 0.20 * d_score + 0.35 * e_score
        return {
            "match_percentage": round(100.0 * combined, 1),
            "phash_similarity": round(p_sim, 4),
            "dhash_similarity": round(d_sim, 4),
            "embedding_similarity": round(emb_sim, 4),
            "method": "perceptual-image-v1",
        }
    overlap = chunk_similarity(probe.chunks, registered.chunks)
    return {
        "match_percentage": round(100.0 * overlap, 1),
        "chunk_overlap": round(overlap, 4),
        "method": "chunk-binary-v1",
    }
