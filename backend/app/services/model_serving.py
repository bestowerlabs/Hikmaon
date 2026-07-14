"""Neural deepfake detector serving.

Loads an exported HikmaonNet ONNX model (``HIKMAON_MODEL_PATH``) and exposes
it to the analysis pipeline. When a trained model is deployed, its calibrated
probability becomes the dominant manipulation signal, layered on top of the
forensic heuristics; when absent, the system reports heuristics only and
says so — it never fakes a neural score.

Serving is torch-free: only onnxruntime + numpy + PIL are required.
"""
from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image

INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DeepfakeModelServer:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or os.environ.get("HIKMAON_MODEL_PATH")
        self.session = None
        self.load_error: str | None = None
        if self.model_path:
            self._load()

    def _load(self) -> None:
        try:
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            available = ort.get_available_providers()
            self.session = ort.InferenceSession(
                self.model_path,
                providers=[p for p in providers if p in available] or ["CPUExecutionProvider"],
            )
        except Exception as exc:  # missing file, missing onnxruntime, bad model
            self.session = None
            self.load_error = str(exc)

    @property
    def available(self) -> bool:
        return self.session is not None

    def status(self) -> dict:
        return {
            "neural_detector": "loaded" if self.available else "not_deployed",
            "model_path": self.model_path,
            "load_error": self.load_error,
            "note": (
                "HikmaonNet is serving calibrated probabilities"
                if self.available
                else "Train with ml/train.py, export with ml/export.py, deploy via HIKMAON_MODEL_PATH"
            ),
        }

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return ((array - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)

    def _infer_one(self, chw: np.ndarray) -> float:
        # HikmaonNet's FFT branch exports with a fixed batch dim, so the model
        # is served one frame at a time.
        batch = np.ascontiguousarray(chw[None], dtype=np.float32)
        return float(self.session.run(None, {"image": batch})[0].reshape(-1)[0])

    def _score_image(self, image: Image.Image) -> float:
        """Score one image with horizontal-flip test-time augmentation."""
        chw = self._preprocess(image)
        return (self._infer_one(chw) + self._infer_one(np.ascontiguousarray(chw[:, :, ::-1]))) / 2.0

    def predict_probability(self, raw_bytes: bytes) -> float | None:
        """Calibrated fake-probability for an image or video, or None if the
        model is not deployed / the media is not decodable.

        Video: frames are sampled and scored; the clip probability is the 80th
        percentile of frame scores — deepfakes typically manipulate only part
        of a clip, so a robust high quantile beats the mean without chasing a
        single outlier frame.
        """
        if not self.available:
            return None
        try:
            image = Image.open(io.BytesIO(raw_bytes))
            image.load()
            return self._score_image(image)
        except Exception:
            pass  # not a still image — try video

        from app.av_fingerprint import sample_video_frames

        frames = sample_video_frames(raw_bytes)
        if not frames:
            return None
        scores = [self._score_image(frame) for frame in frames]
        return float(np.quantile(scores, 0.8))
