"""Evaluation + calibration for trained HikmaonNet checkpoints.

    python -m ml.evaluate --manifest /data/manifest.csv --checkpoint runs/v1/best.pt \
        --split test --fit-temperature

Reports:
- Overall AUC, EER, accuracy at the EER threshold
- **Per-generator breakdown** — the number that matters. A detector that
  scores 0.99 AUC on generators it trained on and 0.65 on a held-out one is
  not ready; expect to iterate on data diversity until held-out generators
  stay above your bar.
- Expected Calibration Error before/after temperature scaling.
  --fit-temperature fits T on the given split and writes it INTO the
  checkpoint so export/serving produce calibrated probabilities.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.data import DeepfakeDataset
from ml.metrics import binary_auc, equal_error_rate, expected_calibration_error
from ml.model import build_model


@torch.no_grad()
def collect_scores(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    logits, labels, generators = [], [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        logits.append(out["logit"].float().cpu().numpy())
        labels.append(batch["label"].numpy())
        generators.extend(batch["generator"])
    return np.concatenate(logits), np.concatenate(labels), generators


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """1-D NLL minimization over temperature via golden-section search."""
    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(labels).float()

    def nll(temperature: float) -> float:
        return float(
            torch.nn.functional.binary_cross_entropy_with_logits(logits_t / temperature, labels_t)
        )

    low, high = 0.05, 10.0
    golden = (5**0.5 - 1) / 2
    for _ in range(60):
        mid1 = high - golden * (high - low)
        mid2 = low + golden * (high - low)
        if nll(mid1) < nll(mid2):
            high = mid2
        else:
            low = mid1
    return (low + high) / 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fit-temperature", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    model = build_model(embed_dim=state.get("embed_dim", 256), backbone=state.get("backbone")).to(device)
    model.load_state_dict(state["model"])

    dataset = DeepfakeDataset(args.manifest, split=args.split, train_augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    logits, labels, generators = collect_scores(model, loader, device)

    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())
    if n_real == 0 or n_fake == 0:
        missing = "real" if n_real == 0 else "fake"
        print(json.dumps({
            "split": args.split,
            "n": len(labels),
            "n_real": n_real,
            "n_fake": n_fake,
            "auc": None,
            "error": (
                f"Cannot evaluate: the '{args.split}' split has no {missing} samples "
                f"({n_real} real / {n_fake} fake). AUC compares real vs fake, so it needs "
                "BOTH classes. Rebuild the manifest with real AND fake data present in this split "
                "(see ml.make_manifest's per-split balance report)."
            ),
        }, indent=2))
        return

    eer, threshold = equal_error_rate(labels, logits)
    report: dict = {
        "split": args.split,
        "n": len(labels),
        "n_real": n_real,
        "n_fake": n_fake,
        "auc": round(binary_auc(labels, logits), 4),
        "eer": round(eer, 4),
        "eer_threshold_logit": round(threshold, 4),
    }

    per_generator: dict[str, dict] = {}
    indices_by_generator = defaultdict(list)
    for index, generator in enumerate(generators):
        indices_by_generator[generator].append(index)
    real_indices = indices_by_generator.get("real", [])
    for generator, indices in sorted(indices_by_generator.items()):
        if generator == "real":
            continue
        subset = np.array(indices + real_indices)
        per_generator[generator] = {
            "n_fake": len(indices),
            "auc_vs_real": round(binary_auc(labels[subset], logits[subset]), 4),
        }
    report["per_generator"] = per_generator

    temperature = float(model.temperature.item())
    probabilities = 1 / (1 + np.exp(-logits / temperature))
    report["ece_current"] = round(expected_calibration_error(labels, probabilities), 4)

    if args.fit_temperature:
        temperature = fit_temperature(logits, labels)
        calibrated = 1 / (1 + np.exp(-logits / temperature))
        report["temperature_fitted"] = round(temperature, 4)
        report["ece_calibrated"] = round(expected_calibration_error(labels, calibrated), 4)
        state["model"]["temperature"] = torch.tensor([temperature])
        torch.save(state, args.checkpoint)
        report["checkpoint_updated"] = True

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
