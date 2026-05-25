"""ONNX Runtime helpers constrained to local CPU execution."""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort

CPU_PROVIDERS = ["CPUExecutionProvider"]


def create_cpu_session(model_path: str | Path) -> ort.InferenceSession:
    """Create an ONNX Runtime session after validating the model path."""

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    return ort.InferenceSession(str(path), providers=CPU_PROVIDERS)
