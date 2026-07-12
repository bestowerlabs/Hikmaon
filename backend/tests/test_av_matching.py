from __future__ import annotations

import subprocess

import pytest

from app.av_fingerprint import ffmpeg_exe
from app.perceptual import fingerprint_media, match_percentage

pytestmark = pytest.mark.skipif(ffmpeg_exe() is None, reason="ffmpeg not available")


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", *args, "-y"],
        check=True,
        capture_output=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def media_files(tmp_path_factory):
    """Original video/audio plus stolen (re-encoded/trimmed) and unrelated variants."""
    root = tmp_path_factory.mktemp("av_media")
    original_video = root / "orig.mp4"
    stolen_video = root / "stolen.mp4"
    other_video = root / "other.mp4"
    original_audio = root / "orig.wav"
    stolen_audio = root / "stolen.mp3"
    other_audio = root / "other.wav"

    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=10",
        "-f", "lavfi", "-i", "anoisesrc=color=pink:duration=10:amplitude=0.5,lowpass=f=2500",
        "-c:v", "libx264", "-crf", "20", "-c:a", "aac", "-shortest", str(original_video),
    )
    # Stolen: 2 seconds trimmed, downscaled, heavily recompressed.
    _ffmpeg(
        "-ss", "2", "-i", str(original_video),
        "-vf", "scale=320:180", "-c:v", "libx264", "-crf", "35", "-c:a", "aac", "-b:a", "32k",
        str(stolen_video),
    )
    _ffmpeg(
        "-f", "lavfi", "-i", "mandelbrot=size=640x360:rate=24",
        "-f", "lavfi", "-i", "anoisesrc=color=pink:duration=10:amplitude=0.5:seed=99,lowpass=f=2500",
        "-t", "10", "-c:v", "libx264", "-crf", "20", "-c:a", "aac", str(other_video),
    )

    _ffmpeg("-f", "lavfi", "-i", "anoisesrc=color=pink:duration=15:amplitude=0.5,lowpass=f=2500", str(original_audio))
    _ffmpeg("-ss", "3", "-i", str(original_audio), "-b:a", "32k", str(stolen_audio))
    _ffmpeg(
        "-f", "lavfi", "-i", "anoisesrc=color=pink:duration=15:amplitude=0.5:seed=42,lowpass=f=2500",
        str(other_audio),
    )
    return {name: path.read_bytes() for name, path in {
        "orig_video": original_video, "stolen_video": stolen_video, "other_video": other_video,
        "orig_audio": original_audio, "stolen_audio": stolen_audio, "other_audio": other_audio,
    }.items()}


def test_video_is_fingerprinted_with_frames_and_audio(media_files):
    fingerprint = fingerprint_media(media_files["orig_video"])
    assert fingerprint.media_kind == "video"
    assert len(fingerprint.frame_phashes) >= 10
    assert len(fingerprint.audio_bits) > 50


def test_reencoded_trimmed_video_matches(media_files):
    original = fingerprint_media(media_files["orig_video"])
    stolen = fingerprint_media(media_files["stolen_video"])
    scores = match_percentage(stolen, original)
    assert scores["match_percentage"] >= 55.0
    # The temporal alignment must discover the 2 s trim (offset ~4 at 2 fps).
    assert scores["video"]["offset"] >= 2


def test_unrelated_video_does_not_match(media_files):
    original = fingerprint_media(media_files["orig_video"])
    other = fingerprint_media(media_files["other_video"])
    assert match_percentage(other, original)["match_percentage"] < 35.0


def test_mp3_reencoded_trimmed_audio_matches(media_files):
    original = fingerprint_media(media_files["orig_audio"])
    stolen = fingerprint_media(media_files["stolen_audio"])
    assert original.media_kind == "audio" and stolen.media_kind == "audio"
    scores = match_percentage(stolen, original)
    assert scores["match_percentage"] >= 55.0
    assert scores["audio"]["ber"] < 0.35  # canonical Haitsma-Kalker decision line


def test_unrelated_audio_does_not_match(media_files):
    original = fingerprint_media(media_files["orig_audio"])
    other = fingerprint_media(media_files["other_audio"])
    scores = match_percentage(other, original)
    assert scores["match_percentage"] < 35.0
    assert scores["audio"]["ber"] > 0.4


def test_audio_does_not_match_unrelated_video_soundtrack(media_files):
    video = fingerprint_media(media_files["orig_video"])
    audio = fingerprint_media(media_files["other_audio"])
    assert match_percentage(audio, video)["match_percentage"] < 35.0
