"""High-level video and audio perceptual fingerprinting.

Video
-----
Frames are extracted with ffmpeg at a fixed rate and each frame receives a
64-bit DCT perceptual hash. A video's identity is the *sequence* of frame
hashes; matching aligns two sequences over every temporal offset and scores
the best-aligned overlap. This survives re-encoding, resizing, bitrate
changes, and trimming — the transformations stolen video actually undergoes.

Audio
-----
Haitsma–Kalker spectral fingerprinting (the classic robust-audio-hash design
used by industrial audio matchers): mono 8 kHz PCM → overlapping FFT frames →
17 log-spaced energy bands (300–3000 Hz) → 16 bits per frame from the sign of
the band-energy derivative in time and frequency. Matching slides one bit
sequence over the other and measures bit error rate (BER); genuinely matching
audio lands far below the 0.35 BER decision line even after MP3/AAC
re-encoding and volume changes.

Video files also carry an audio fingerprint of their soundtrack, so a stolen
clip matches on either channel.

ffmpeg is located from (in order): $HIKMAON_FFMPEG, the system PATH, or the
static binary bundled with the ``imageio-ffmpeg`` pip package.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from app.perceptual import hash_similarity, phash

VIDEO_FPS = 2.0
MAX_FRAMES = 240  # 2 minutes of video at 2 fps
FRAME_SIZE = 256
AUDIO_SR = 8000
AUDIO_MAX_SECONDS = 300
FFT_WINDOW = 4096  # 512 ms — long windows make band energies vary slowly,
FFT_HOP = 256      # 32 ms hops give fine alignment granularity (Haitsma-Kalker)
BANDS = 17  # -> 16 bits per frame
BAND_LOW_HZ, BAND_HIGH_HZ = 300.0, 3000.0

MIN_FRAME_OVERLAP = 5
MIN_AUDIO_OVERLAP = 94  # ~3 seconds at 32 ms per frame
AUDIO_BER_MATCH = 0.35  # below this bit-error-rate two clips are related

AV_VERSION = "av-fingerprint-v1"


def ffmpeg_exe() -> str | None:
    override = os.environ.get("HIKMAON_FFMPEG")
    if override and Path(override).exists():
        return override
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _run_ffmpeg(args: list[str], input_path: str) -> bytes | None:
    exe = ffmpeg_exe()
    if exe is None:
        return None
    try:
        result = subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-i", input_path, *args],
            capture_output=True,
            timeout=180,
        )
        return result.stdout if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_frame_hashes(media_path: str, fps: float = VIDEO_FPS, max_frames: int = MAX_FRAMES) -> list[str]:
    """Decode video frames and return one perceptual hash per sampled frame."""
    with tempfile.TemporaryDirectory(prefix="hikmaon_frames_") as frame_dir:
        exe = ffmpeg_exe()
        if exe is None:
            return []
        try:
            result = subprocess.run(
                [
                    exe, "-hide_banner", "-loglevel", "error",
                    "-i", media_path,
                    "-vf", f"fps={fps},scale={FRAME_SIZE}:{FRAME_SIZE}",
                    "-frames:v", str(max_frames),
                    os.path.join(frame_dir, "f_%05d.png"),
                ],
                capture_output=True,
                timeout=180,
            )
        except (subprocess.TimeoutExpired, OSError):
            return []
        if result.returncode != 0:
            return []
        hashes = []
        for frame_file in sorted(Path(frame_dir).glob("f_*.png")):
            with Image.open(frame_file) as frame:
                hashes.append(format(phash(frame.convert("RGB")), "016x"))
        return hashes


def sample_video_frames(raw_bytes: bytes, fps: float = 1.0, max_frames: int = 16) -> list[Image.Image]:
    """Decode up to ``max_frames`` RGB frames from video bytes (for neural
    inference). Returns [] if the bytes are not decodable video."""
    if ffmpeg_exe() is None:
        return []
    with tempfile.NamedTemporaryFile(prefix="hikmaon_nn_", delete=False) as handle:
        handle.write(raw_bytes)
        media_path = handle.name
    try:
        with tempfile.TemporaryDirectory(prefix="hikmaon_nnframes_") as frame_dir:
            try:
                result = subprocess.run(
                    [
                        ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                        "-i", media_path,
                        "-vf", f"fps={fps}",
                        "-frames:v", str(max_frames),
                        os.path.join(frame_dir, "f_%05d.png"),
                    ],
                    capture_output=True,
                    timeout=180,
                )
            except (subprocess.TimeoutExpired, OSError):
                return []
            if result.returncode != 0:
                return []
            frames = []
            for frame_file in sorted(Path(frame_dir).glob("f_*.png")):
                with Image.open(frame_file) as frame:
                    frames.append(frame.convert("RGB").copy())
            return frames
    finally:
        os.unlink(media_path)


def extract_audio_pcm(media_path: str, sample_rate: int = AUDIO_SR) -> np.ndarray | None:
    """Decode any audio stream to mono PCM float32 in [-1, 1]."""
    raw = _run_ffmpeg(
        ["-t", str(AUDIO_MAX_SECONDS), "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"],
        media_path,
    )
    if not raw or len(raw) < FFT_WINDOW * 2:
        return None
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def audio_fingerprint_bits(pcm: np.ndarray, sample_rate: int = AUDIO_SR) -> list[int]:
    """Haitsma–Kalker subfingerprints: one 16-bit int per ~64 ms of audio."""
    if pcm is None or len(pcm) < FFT_WINDOW:
        return []
    n_frames = 1 + (len(pcm) - FFT_WINDOW) // FFT_HOP
    if n_frames < 2:
        return []
    window = np.hanning(FFT_WINDOW)
    # Log-spaced band edges over 300-3000 Hz.
    edges_hz = np.geomspace(BAND_LOW_HZ, BAND_HIGH_HZ, BANDS + 1)
    edges_bin = np.clip((edges_hz / sample_rate * FFT_WINDOW).astype(int), 1, FFT_WINDOW // 2 - 1)

    strides = np.lib.stride_tricks.sliding_window_view(pcm, FFT_WINDOW)[::FFT_HOP][:n_frames]
    spectra = np.abs(np.fft.rfft(strides * window, axis=1)) ** 2

    energies = np.empty((n_frames, BANDS))
    for band in range(BANDS):
        lo, hi = edges_bin[band], max(edges_bin[band + 1], edges_bin[band] + 1)
        energies[:, band] = spectra[:, lo:hi].sum(axis=1)

    # bit(n, m) = sign of the time-and-frequency energy derivative.
    diff = (energies[1:, :-1] - energies[1:, 1:]) - (energies[:-1, :-1] - energies[:-1, 1:])
    bits = (diff > 0).astype(np.uint16)
    weights = (1 << np.arange(BANDS - 1, dtype=np.uint16))[::-1]
    return (bits @ weights).astype(int).tolist()


@dataclass
class AVFingerprint:
    media_kind: str  # "video" | "audio" | "none"
    frame_phashes: list[str] = field(default_factory=list)
    audio_bits: list[int] = field(default_factory=list)


def fingerprint_av_bytes(raw_bytes: bytes) -> AVFingerprint:
    """Fingerprint arbitrary media bytes as video and/or audio."""
    if ffmpeg_exe() is None:
        return AVFingerprint(media_kind="none")
    with tempfile.NamedTemporaryFile(prefix="hikmaon_media_", delete=False) as handle:
        handle.write(raw_bytes)
        media_path = handle.name
    try:
        frame_hashes = extract_frame_hashes(media_path)
        pcm = extract_audio_pcm(media_path)
        audio_bits = audio_fingerprint_bits(pcm) if pcm is not None else []
    finally:
        os.unlink(media_path)

    if frame_hashes:
        return AVFingerprint(media_kind="video", frame_phashes=frame_hashes, audio_bits=audio_bits)
    if audio_bits:
        return AVFingerprint(media_kind="audio", audio_bits=audio_bits)
    return AVFingerprint(media_kind="none")


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
_POPCOUNT16 = np.array([bin(v).count("1") for v in range(1 << 16)], dtype=np.uint8)


def match_frame_sequences(probe: list[str], registered: list[str]) -> dict:
    """Best temporally-aligned mean frame-hash similarity, as a percentage."""
    if not probe or not registered:
        return {"match_percentage": 0.0, "aligned_frames": 0, "offset": 0}
    a = [int(h, 16) for h in probe]
    b = [int(h, 16) for h in registered]

    best_score, best_offset, best_overlap = 0.0, 0, 0
    for offset in range(-(len(a) - MIN_FRAME_OVERLAP), len(b) - MIN_FRAME_OVERLAP + 1):
        overlap = min(len(a), len(b) - offset) if offset >= 0 else min(len(a) + offset, len(b))
        if overlap < MIN_FRAME_OVERLAP:
            continue
        a_start = max(0, -offset)
        b_start = max(0, offset)
        sims = [
            hash_similarity(a[a_start + i], b[b_start + i])
            for i in range(overlap)
        ]
        score = float(np.mean(sims))
        if score > best_score:
            best_score, best_offset, best_overlap = score, offset, overlap

    percentage = max(0.0, min(1.0, (best_score - 0.5) / 0.5)) * 100.0
    return {
        "match_percentage": round(percentage, 1),
        "mean_frame_similarity": round(best_score, 4),
        "aligned_frames": best_overlap,
        "offset": best_offset,
    }


def match_audio_bits(probe: list[int], registered: list[int]) -> dict:
    """Best-aligned bit error rate between two audio fingerprints, as a %."""
    if len(probe) < MIN_AUDIO_OVERLAP or len(registered) < MIN_AUDIO_OVERLAP:
        return {"match_percentage": 0.0, "ber": 1.0, "aligned_seconds": 0.0}
    a = np.asarray(probe, dtype=np.uint16)
    b = np.asarray(registered, dtype=np.uint16)

    def ber_at(offset: int) -> tuple[float, int]:
        a_start = max(0, -offset)
        b_start = max(0, offset)
        overlap = min(len(a) - a_start, len(b) - b_start)
        if overlap < MIN_AUDIO_OVERLAP:
            return 1.0, 0
        xor = np.bitwise_xor(a[a_start : a_start + overlap], b[b_start : b_start + overlap])
        return float(_POPCOUNT16[xor].sum()) / (overlap * (BANDS - 1)), overlap

    # Coarse-to-fine offset search keeps long clips tractable.
    best_ber, best_offset, best_overlap = 1.0, 0, 0
    coarse = range(-(len(a) - MIN_AUDIO_OVERLAP), len(b) - MIN_AUDIO_OVERLAP + 1, 4)
    for offset in coarse:
        ber, overlap = ber_at(offset)
        if ber < best_ber:
            best_ber, best_offset, best_overlap = ber, offset, overlap
    for offset in range(best_offset - 3, best_offset + 4):
        ber, overlap = ber_at(offset)
        if ber < best_ber:
            best_ber, best_overlap = ber, overlap

    # BER 0.5 = unrelated (random bits), 0.0 = identical. Map the decision
    # band so BER at the 0.35 threshold ~ 30% and BER 0 = 100%.
    score = max(0.0, min(1.0, (0.5 - best_ber) / 0.5))
    return {
        "match_percentage": round(100.0 * score, 1),
        "ber": round(best_ber, 4),
        "aligned_seconds": round(best_overlap * FFT_HOP / AUDIO_SR, 1),
    }


def match_av(probe: AVFingerprint, registered: AVFingerprint) -> dict:
    """Combined verdict across visual and audio channels (best evidence wins,
    with a bonus when both channels agree)."""
    video_result = match_frame_sequences(probe.frame_phashes, registered.frame_phashes)
    audio_result = match_audio_bits(probe.audio_bits, registered.audio_bits)

    video_pct = video_result["match_percentage"]
    audio_pct = audio_result["match_percentage"]
    if video_pct > 0 and audio_pct > 0:
        combined = max(video_pct, audio_pct) * 0.8 + min(video_pct, audio_pct) * 0.2
    else:
        combined = max(video_pct, audio_pct)

    return {
        "match_percentage": round(combined, 1),
        "video": video_result,
        "audio": audio_result,
        "method": AV_VERSION,
    }
