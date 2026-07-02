"""Export a trained HikmaonNet checkpoint to ONNX for production serving.

    python -m ml.export --checkpoint runs/v1/best.pt --out hikmaonnet.onnx

The API server loads the ONNX file via `HIKMAON_MODEL_PATH` (see
app/services/model_serving.py) — no torch required at serving time.
"""
from __future__ import annotations

import argparse

import torch

from ml.model import INPUT_SIZE, MODEL_VERSION, build_model


class _ExportWrapper(torch.nn.Module):
    """Exports a single calibrated-probability output."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["probability"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="hikmaonnet.onnx")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(embed_dim=state.get("embed_dim", 256))
    model.load_state_dict(state["model"])
    model.eval()

    wrapper = _ExportWrapper(model)
    example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        wrapper,
        (example,),
        args.out,
        input_names=["image"],
        output_names=["probability"],
        dynamic_axes={"image": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {MODEL_VERSION} -> {args.out}")
    print(f"deploy with: HIKMAON_MODEL_PATH={args.out} uvicorn app.main:app")

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        result = session.run(None, {"image": example.numpy()})[0]
        reference = wrapper(example).detach().numpy()
        max_delta = float(abs(result - reference).max())
        assert max_delta < 1e-4, f"ONNX output mismatch: {max_delta}"
        print(f"verified against torch (max delta {max_delta:.2e})")
    except ImportError:
        print("onnxruntime not installed; skipped numerical verification")


if __name__ == "__main__":
    main()
