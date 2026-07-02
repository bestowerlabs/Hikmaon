from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed (training environment only)")

from ml.metrics import binary_auc, equal_error_rate, expected_calibration_error  # noqa: E402
from ml.model import build_model  # noqa: E402


def test_model_forward_and_backward():
    model = build_model(embed_dim=64)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out["logit"].shape == (2,)
    assert ((out["probability"] >= 0) & (out["probability"] <= 1)).all()
    out["logit"].sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients flowed"


def test_model_learns_trivial_separation():
    model = build_model(embed_dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    first_loss = last_loss = None
    for _ in range(6):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x)["logit"], labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        first_loss = first_loss if first_loss is not None else loss.item()
        last_loss = loss.item()
    assert last_loss < first_loss


def test_metrics_sanity():
    labels = np.array([0, 0, 1, 1])
    perfect = np.array([-2.0, -1.0, 1.0, 2.0])
    assert binary_auc(labels, perfect) == 1.0
    assert binary_auc(labels, -perfect) == 0.0
    eer, _ = equal_error_rate(labels, perfect)
    assert eer == 0.0
    ece = expected_calibration_error(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.1, 0.2]))
    assert 0 <= ece < 0.3
