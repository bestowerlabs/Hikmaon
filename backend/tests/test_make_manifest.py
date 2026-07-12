from __future__ import annotations

from pathlib import Path

from ml.make_manifest import build_manifest


def _make_dataset(root: Path, layout: dict[str, int]) -> None:
    """layout: {subdir: n_frames} — creates tiny stand-in image files."""
    for subdir, count in layout.items():
        directory = root / subdir
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (directory / f"frame_{index:03d}.png").write_bytes(b"\x89PNG-stub")


def test_manifest_rows_labels_and_generators(tmp_path):
    real = tmp_path / "real"
    fakes = tmp_path / "deepfakes"
    _make_dataset(real, {"vid_a": 3, "vid_b": 2})
    _make_dataset(fakes, {"vid_x": 4})

    rows = build_manifest(
        real_dirs=[real],
        fake_sources=[("deepfakes", fakes)],
        val_fraction=0.2,
        test_fraction=0.2,
        holdout_generators=set(),
    )
    assert len(rows) == 9
    assert {r["label"] for r in rows if r["generator"] == "real"} == {0}
    assert {r["label"] for r in rows if r["generator"] == "deepfakes"} == {1}
    assert all(Path(r["path"]).exists() for r in rows)


def test_frames_of_same_video_stay_in_one_split(tmp_path):
    real = tmp_path / "real"
    fakes = tmp_path / "fk"
    _make_dataset(real, {f"vid_{i}": 5 for i in range(40)})
    _make_dataset(fakes, {f"vid_{i}": 5 for i in range(40)})

    rows = build_manifest(
        real_dirs=[real],
        fake_sources=[("fk", fakes)],
        val_fraction=0.2,
        test_fraction=0.2,
        holdout_generators=set(),
    )
    split_by_group: dict[str, set[str]] = {}
    for row in rows:
        group = str(Path(row["path"]).parent) + row["generator"]
        split_by_group.setdefault(group, set()).add(row["split"])
    # No video's frames may span two splits (frame leakage).
    assert all(len(splits) == 1 for splits in split_by_group.values())
    # With 80 groups all three splits should be populated.
    assert {r["split"] for r in rows} == {"train", "val", "test"}


def test_holdout_generator_never_trains(tmp_path):
    real = tmp_path / "real"
    seen = tmp_path / "seen"
    held = tmp_path / "held"
    _make_dataset(real, {f"v{i}": 2 for i in range(20)})
    _make_dataset(seen, {f"v{i}": 2 for i in range(20)})
    _make_dataset(held, {f"v{i}": 2 for i in range(20)})

    rows = build_manifest(
        real_dirs=[real],
        fake_sources=[("seen", seen), ("held", held)],
        val_fraction=0.1,
        test_fraction=0.1,
        holdout_generators={"held"},
    )
    held_splits = {r["split"] for r in rows if r["generator"] == "held"}
    assert "train" not in held_splits
    assert held_splits <= {"val", "test"}
    assert "train" in {r["split"] for r in rows if r["generator"] == "seen"}


def test_splits_are_deterministic(tmp_path):
    real = tmp_path / "real"
    fakes = tmp_path / "fk"
    _make_dataset(real, {"a": 2, "b": 2})
    _make_dataset(fakes, {"c": 2})

    first = build_manifest([real], [("fk", fakes)], 0.2, 0.2, set())
    second = build_manifest([real], [("fk", fakes)], 0.2, 0.2, set())
    assert first == second
