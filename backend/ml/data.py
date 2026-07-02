"""Dataset and augmentation pipeline for HikmaonNet training.

Manifest format (CSV with header), one row per image or per extracted frame:

    path,label,generator,split
    /data/ffpp/real/000/frame_001.png,0,real,train
    /data/ffpp/deepfakes/000/frame_001.png,1,deepfakes,train
    /data/dfdc/fake/xyz/frame_010.png,1,dfdc_unknown,val
    ...

- ``label``: 0 = real, 1 = fake/manipulated
- ``generator``: which method produced the fake ("real" for real media).
  Used for cross-generator evaluation — keep at least one generator
  entirely OUT of train and only in val/test to measure generalization.
- ``split``: train / val / test

Recommended sources: FaceForensics++ (c23 AND c40 compressions), DFDC,
Celeb-DF v2, DeeperForensics, plus current diffusion-generated sets. Extract
frames at 1-3 fps, crop faces with a detector (e.g. RetinaFace) with 30%
margin, save crops. Keep raw frames too for whole-frame training later.

The **degradation augmentations are the most important part of this file**:
real-world deepfakes arrive re-encoded, resized, and re-shared. A detector
trained on pristine frames dies in production. Do not remove them.
"""
from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from ml.model import INPUT_SIZE

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class ManifestRow:
    path: str
    label: int
    generator: str
    split: str


def read_manifest(manifest_path: str | Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with open(manifest_path, newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                ManifestRow(
                    path=record["path"],
                    label=int(record["label"]),
                    generator=record.get("generator", "unknown"),
                    split=record.get("split", "train"),
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# Augmentations (PIL-based, dependency-free)
# --------------------------------------------------------------------------- #
def _jpeg_cycle(image: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.randint(30, 90)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _resize_cycle(image: Image.Image, rng: random.Random) -> Image.Image:
    scale = rng.uniform(0.5, 1.0)
    small = image.resize(
        (max(int(image.width * scale), 32), max(int(image.height * scale), 32)),
        Image.BILINEAR,
    )
    return small.resize(image.size, rng.choice([Image.BILINEAR, Image.LANCZOS]))


def _color_jitter(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    return ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))


def _blur(image: Image.Image, rng: random.Random) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))


def _noise(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    array += np.random.default_rng(rng.randrange(2**31)).normal(0, rng.uniform(1, 6), array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


TRAIN_DEGRADATIONS = (_jpeg_cycle, _resize_cycle, _color_jitter, _blur, _noise)


def augment_train(image: Image.Image, rng: random.Random) -> Image.Image:
    # Random resized crop (0.7-1.0 area) + horizontal flip.
    area = rng.uniform(0.7, 1.0)
    w = int(image.width * area**0.5)
    h = int(image.height * area**0.5)
    left = rng.randint(0, image.width - w)
    top = rng.randint(0, image.height - h)
    image = image.crop((left, top, left + w, top + h))
    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # Apply 1-3 degradations with 90% probability overall.
    if rng.random() < 0.9:
        for op in rng.sample(TRAIN_DEGRADATIONS, k=rng.randint(1, 3)):
            image = op(image, rng)
    return image


def to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array.transpose(2, 0, 1))


class DeepfakeDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str, train_augment: bool | None = None) -> None:
        self.rows = [r for r in read_manifest(manifest_path) if r.split == split]
        if not self.rows:
            raise ValueError(f"No rows with split={split!r} in {manifest_path}")
        self.train_augment = split == "train" if train_augment is None else train_augment
        self.rng = random.Random(0xB157)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image = Image.open(row.path).convert("RGB")
        if self.train_augment:
            image = augment_train(image, self.rng)
        return {
            "image": to_tensor(image),
            "label": torch.tensor(float(row.label)),
            "generator": row.generator,
        }

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency sample weights for a WeightedRandomSampler."""
        positives = sum(r.label for r in self.rows)
        negatives = len(self.rows) - positives
        weight_pos = len(self.rows) / max(2 * positives, 1)
        weight_neg = len(self.rows) / max(2 * negatives, 1)
        return torch.tensor([weight_pos if r.label else weight_neg for r in self.rows])
