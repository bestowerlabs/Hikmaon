"""Turn a folder of videos into training frames — the first step of training.

Extracts frames from every video in a folder into a per-video subfolder, which
is exactly the layout `ml.make_manifest` expects (it groups by subfolder so one
video's frames never leak across train/val/test).

No machine-learning dependencies — just ffmpeg (installed automatically with
`imageio-ffmpeg`). Safe to re-run: existing frame folders are skipped.

Usage:
    python -m ml.prepare_dataset --videos /data/ffpp/original --out /data/frames/real
    python -m ml.prepare_dataset --videos /data/ffpp/Deepfakes --out /data/frames/deepfakes

Then build the manifest pointing --real/--fake at these --out folders.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def extract(videos_dir: str, out_dir: str, fps: float, max_frames: int, size: int) -> None:
    exe = _ffmpeg()
    if exe is None:
        raise SystemExit("ffmpeg not found. Run:  pip install imageio-ffmpeg")

    videos = sorted(p for p in Path(videos_dir).rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise SystemExit(f"No videos found under {videos_dir}")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    for index, video in enumerate(videos, 1):
        # Group key = per-video folder (parent + stem keeps it unique across
        # nested compression subfolders like c23/ and c40/).
        group = f"{video.parent.name}_{video.stem}"
        destination = out_root / group
        if destination.exists() and any(destination.glob("frame_*.png")):
            continue  # already extracted — safe to re-run
        destination.mkdir(parents=True, exist_ok=True)

        vf = f"fps={fps}"
        if size:
            vf += f",scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}"
        subprocess.run(
            [
                exe, "-hide_banner", "-loglevel", "error",
                "-i", str(video),
                "-vf", vf,
                "-frames:v", str(max_frames),
                str(destination / "frame_%04d.png"),
            ],
            check=False,
        )
        count = len(list(destination.glob("frame_*.png")))
        total_frames += count
        print(f"[{index}/{len(videos)}] {video.name}: {count} frames")

    print(f"\nDone: {total_frames} frames from {len(videos)} videos -> {out_dir}")
    print(f"Next: point --real or --fake at {out_dir} in `python -m ml.make_manifest`")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract training frames from a folder of videos")
    parser.add_argument("--videos", required=True, help="Folder containing videos (searched recursively)")
    parser.add_argument("--out", required=True, help="Output folder for extracted frames")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to sample (default 2)")
    parser.add_argument("--max-frames", type=int, default=40, help="Max frames per video (default 40)")
    parser.add_argument("--size", type=int, default=256, help="Square crop size in px, 0 to keep original (default 256)")
    args = parser.parse_args()
    extract(args.videos, args.out, args.fps, args.max_frames, args.size)


if __name__ == "__main__":
    main()
