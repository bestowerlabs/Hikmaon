"""Build the training manifest CSV for HikmaonNet from dataset folders.

The manifest is the index of your training data: one row per image (or per
extracted video frame) with its label, the generator that produced it, and
the train/val/test split. `ml/train.py`, `ml/evaluate.py` consume it.

Usage
-----
    python -m ml.make_manifest \
        --real /data/ffpp/real --real /data/celebdf/real \
        --fake deepfakes=/data/ffpp/deepfakes \
        --fake face2face=/data/ffpp/face2face \
        --fake neuraltextures=/data/ffpp/neuraltextures \
        --fake celebdf=/data/celebdf/fake \
        --holdout celebdf \
        --val 0.1 --test 0.1 \
        --out /data/manifest.csv

- ``--real DIR`` (repeatable): folders of authentic images/frames.
- ``--fake NAME=DIR`` (repeatable): folders of fake images/frames, tagged
  with the generator that produced them (used for per-generator evaluation).
- ``--holdout NAME`` (repeatable): generators excluded from training and
  placed only in val/test — their AUC is your cross-generator
  generalization number, the release-gating metric.

Two correctness guarantees that are easy to get wrong by hand:

1. **No frame leakage.** Frames from the same source video must never be
   split between train and val/test (the model would memorize the video and
   the validation score would lie). Frames are grouped by their parent
   directory (the per-video folder in FaceForensics++/DFDC-style layouts)
   and each *group* is assigned to exactly one split.
2. **Deterministic splits.** Assignment is a hash of the group key, so
   re-running on a grown dataset keeps existing files in their old splits.

Expected dataset layout (any depth works; the direct parent dir is the group):

    /data/ffpp/real/012/frame_0001.png
    /data/ffpp/real/012/frame_0002.png     <- same group "012"
    /data/ffpp/deepfakes/012_026/frame_0001.png
    ...
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _collect(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"error: {directory} is not a directory")
    files = [
        p for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        raise SystemExit(f"error: no images found under {directory}")
    return files


def _group_key(root: Path, file: Path) -> str:
    """Frames grouped by parent dir relative to the source root (video id)."""
    relative_parent = file.parent.relative_to(root)
    return str(relative_parent) if str(relative_parent) != "." else file.stem


def _assign_split(group: str, val_fraction: float, test_fraction: float, holdout: bool) -> str:
    bucket = int(hashlib.sha1(group.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if holdout:
        # Held-out generators never train: split val/test proportionally.
        boundary = val_fraction / (val_fraction + test_fraction) if (val_fraction + test_fraction) else 0.5
        return "val" if bucket < boundary else "test"
    if bucket < val_fraction:
        return "val"
    if bucket < val_fraction + test_fraction:
        return "test"
    return "train"


def build_manifest(
    real_dirs: list[Path],
    fake_sources: list[tuple[str, Path]],
    val_fraction: float,
    test_fraction: float,
    holdout_generators: set[str],
) -> list[dict]:
    rows: list[dict] = []

    for directory in real_dirs:
        for file in _collect(directory):
            group = f"real|{directory}|{_group_key(directory, file)}"
            rows.append(
                {
                    "path": str(file.resolve()),
                    "label": 0,
                    "generator": "real",
                    "split": _assign_split(group, val_fraction, test_fraction, holdout=False),
                }
            )

    for generator, directory in fake_sources:
        held_out = generator in holdout_generators
        for file in _collect(directory):
            group = f"{generator}|{directory}|{_group_key(directory, file)}"
            rows.append(
                {
                    "path": str(file.resolve()),
                    "label": 1,
                    "generator": generator,
                    "split": _assign_split(group, val_fraction, test_fraction, holdout=held_out),
                }
            )
    return rows


def _summarize(rows: list[dict]) -> str:
    lines = []
    by_generator = Counter((r["generator"], r["split"]) for r in rows)
    generators = sorted({r["generator"] for r in rows})
    lines.append(f"{'generator':<18}{'train':>8}{'val':>8}{'test':>8}{'total':>9}")
    for generator in generators:
        train = by_generator.get((generator, "train"), 0)
        val = by_generator.get((generator, "val"), 0)
        test = by_generator.get((generator, "test"), 0)
        lines.append(f"{generator:<18}{train:>8}{val:>8}{test:>8}{train + val + test:>9}")
    total = len(rows)
    fakes = sum(r["label"] for r in rows)
    lines.append(f"\ntotal {total} rows: {total - fakes} real / {fakes} fake")

    # Class balance per split. A split with only one class makes training or
    # AUC evaluation impossible (AUC needs BOTH real and fake), so surface it
    # loudly here instead of after hours of training.
    lines.append("")
    lines.append(f"{'split':<10}{'real':>8}{'fake':>8}")
    problems = []
    for split in ("train", "val", "test"):
        real = sum(1 for r in rows if r["split"] == split and r["label"] == 0)
        fake = sum(1 for r in rows if r["split"] == split and r["label"] == 1)
        lines.append(f"{split:<10}{real:>8}{fake:>8}")
        if real == 0 or fake == 0:
            problems.append((split, real, fake))
    if problems:
        lines.append("")
        lines.append("=" * 62)
        for split, real, fake in problems:
            missing = "REAL" if real == 0 else "FAKE"
            lines.append(f"  WARNING: '{split}' split has {real} real / {fake} fake — no {missing} samples.")
        lines.append("  Training/AUC needs BOTH classes. Add the missing data and rebuild:")
        lines.append("  - real media goes under --real DIR (e.g. FaceForensics++ original_sequences,")
        lines.append("    Celeb-DF Celeb-real / YouTube-real). Extract frames with ml.prepare_dataset first.")
        lines.append("=" * 62)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the HikmaonNet training manifest")
    parser.add_argument("--real", action="append", default=[], metavar="DIR",
                        help="Directory of real images/frames (repeatable)")
    parser.add_argument("--fake", action="append", default=[], metavar="NAME=DIR",
                        help="Directory of fake images tagged with its generator name (repeatable)")
    parser.add_argument("--holdout", action="append", default=[], metavar="NAME",
                        help="Generator name to exclude from training (val/test only, repeatable)")
    parser.add_argument("--val", type=float, default=0.1, help="Validation fraction (default 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="Test fraction (default 0.1)")
    parser.add_argument("--out", default="manifest.csv", help="Output CSV path")
    args = parser.parse_args()

    if not args.real or not args.fake:
        parser.error("need at least one --real DIR and one --fake NAME=DIR")
    if args.val + args.test >= 0.9:
        parser.error("--val + --test must leave a sensible training fraction")

    fake_sources: list[tuple[str, Path]] = []
    for spec in args.fake:
        if "=" not in spec:
            parser.error(f"--fake expects NAME=DIR, got {spec!r}")
        name, _, directory = spec.partition("=")
        fake_sources.append((name.strip(), Path(directory).expanduser()))

    holdout = set(args.holdout)
    known = {name for name, _ in fake_sources}
    unknown = holdout - known
    if unknown:
        parser.error(f"--holdout names not among --fake generators: {sorted(unknown)}")

    rows = build_manifest(
        real_dirs=[Path(d).expanduser() for d in args.real],
        fake_sources=fake_sources,
        val_fraction=args.val,
        test_fraction=args.test,
        holdout_generators=holdout,
    )

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "generator", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(_summarize(rows))
    print(f"\nwrote {out_path}")
    print(f"train: python -m ml.train --manifest {out_path} --out runs/v1 --epochs 30")


if __name__ == "__main__":
    main()
