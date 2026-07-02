"""HikmaonNet: trainable deepfake detection model and training pipeline.

This package is self-contained and GPU-ready. It is imported by the API
server only through `app/services/model_serving.py`, which loads an exported
ONNX model — so the server never needs torch installed in production.
"""
