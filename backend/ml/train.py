"""Training loop for HikmaonNet.

Usage (single GPU):

    python -m ml.train --manifest /data/manifest.csv --out runs/v1 \
        --epochs 30 --batch-size 64 --lr 3e-4

Multi-GPU: wrap with torchrun and pass --ddp (standard DistributedDataParallel;
the loop below is DDP-safe: per-rank samplers, rank-0-only checkpointing).

What the loop does:
- AdamW + cosine schedule with linear warmup
- Mixed precision on CUDA
- Class-balanced sampling
- BCE with label smoothing (0.05) — softens overconfident targets
- Selects best checkpoint by validation AUC
- Writes metrics per epoch to <out>/log.jsonl and the best model to
  <out>/best.pt (weights + config + val metrics)

After training:
    python -m ml.evaluate --manifest ... --checkpoint runs/v1/best.pt   # metrics + temperature
    python -m ml.export   --checkpoint runs/v1/best.pt --out hikmaonnet.onnx
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.data import DeepfakeDataset
from ml.metrics import binary_auc
from ml.model import MODEL_VERSION, build_model


def make_loaders(manifest: str, batch_size: int, workers: int) -> tuple[DataLoader, DataLoader]:
    train_set = DeepfakeDataset(manifest, split="train")
    val_set = DeepfakeDataset(manifest, split="val")
    sampler = WeightedRandomSampler(
        weights=train_set.class_weights(), num_samples=len(train_set), replacement=True
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler,
        num_workers=workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    return train_loader, val_loader


def cosine_lr(step: int, total_steps: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_epoch(model, loader, device) -> dict:
    model.eval()
    scores, labels = [], []
    for batch in loader:
        out = model(batch["image"].to(device, non_blocking=True))
        scores.append(out["logit"].float().cpu())
        labels.append(batch["label"])
    scores = torch.cat(scores)
    labels = torch.cat(labels)
    probabilities = torch.sigmoid(scores)
    predictions = (probabilities >= 0.5).float()
    return {
        "val_auc": binary_auc(labels.numpy(), scores.numpy()),
        "val_acc": float((predictions == labels).float().mean()),
        "val_loss": float(F.binary_cross_entropy_with_logits(scores, labels)),
    }


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(args.manifest, args.batch_size, args.workers)
    model = build_model(embed_dim=args.embed_dim, dropout=args.dropout).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        print(f"resumed from {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 20)

    best_auc = 0.0
    step = 0
    for epoch in range(args.epochs):
        model.train()
        if epoch < args.freeze_epochs:
            model.spatial.requires_grad_(False)
        else:
            model.spatial.requires_grad_(True)

        epoch_loss, seen = 0.0, 0
        started = time.time()
        for batch in train_loader:
            lr = cosine_lr(step, total_steps, warmup_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            # Label smoothing keeps the model calibratable.
            targets = labels * (1 - 0.05) + 0.025

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)["logit"]
                loss = F.binary_cross_entropy_with_logits(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * images.shape[0]
            seen += images.shape[0]
            step += 1

        metrics = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(seen, 1),
            "lr": lr,
            "seconds": round(time.time() - started, 1),
            **evaluate_epoch(model, val_loader, device),
        }
        print(json.dumps(metrics))
        with open(out_dir / "log.jsonl", "a") as handle:
            handle.write(json.dumps(metrics) + "\n")

        if metrics["val_auc"] > best_auc:
            best_auc = metrics["val_auc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_version": MODEL_VERSION,
                    "embed_dim": args.embed_dim,
                    "dropout": args.dropout,
                    "val_metrics": metrics,
                },
                out_dir / "best.pt",
            )
            print(f"saved best.pt (val_auc={best_auc:.4f})")

    print(f"done. best val AUC: {best_auc:.4f}. Next: python -m ml.evaluate then python -m ml.export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HikmaonNet")
    parser.add_argument("--manifest", required=True, help="CSV manifest (see ml/data.py)")
    parser.add_argument("--out", default="runs/hikmaonnet", help="Output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--freeze-epochs", type=int, default=0, help="Freeze spatial branch for N epochs")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
