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


def test_neural_serving_scores_image_and_video(tmp_path):
    """The deployed detector must score BOTH still images and video frames
    (video is the primary deepfake medium)."""
    ort = pytest.importorskip("onnxruntime")
    import io
    import subprocess

    from PIL import Image

    from app.av_fingerprint import ffmpeg_exe
    from app.services.model_serving import DeepfakeModelServer
    from ml.model import build_model

    model = build_model(embed_dim=32).eval()

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)["probability"]

    onnx_path = tmp_path / "m.onnx"
    torch.onnx.export(
        _Wrap(model), (torch.randn(1, 3, 224, 224),), str(onnx_path),
        input_names=["image"], output_names=["probability"],
        dynamic_axes={"image": {0: "batch"}, "probability": {0: "batch"}}, opset_version=17,
    )

    server = DeepfakeModelServer(model_path=str(onnx_path))
    assert server.available

    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(0).uniform(0, 1, (256, 256, 3)) * 255).astype("uint8")).save(buf, "PNG")
    p_img = server.predict_probability(buf.getvalue())
    assert p_img is not None and 0.0 <= p_img <= 1.0

    if ffmpeg_exe() is not None:
        vid = tmp_path / "clip.mp4"
        subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "testsrc2=size=320x240:rate=10:duration=2", "-c:v", "libx264", "-y", str(vid)],
            check=True, capture_output=True, timeout=120,
        )
        p_vid = server.predict_probability(vid.read_bytes())
        assert p_vid is not None and 0.0 <= p_vid <= 1.0


def test_pretrained_backbone_path_builds_and_learns():
    """The timm-backbone spatial branch builds offline (pretrained=False) and
    can drive its loss down — the recommended real-training configuration."""
    pytest.importorskip("timm")
    from ml.model import build_model

    model = build_model(embed_dim=64, backbone="tf_efficientnet_b0_ns", pretrained=False)
    assert model.spatial.backbone is not None

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    first = last = None
    for _ in range(6):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x)["logit"], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first
