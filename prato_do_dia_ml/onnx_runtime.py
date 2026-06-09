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

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    return ort.InferenceSession(str(path), sess_options=opts, providers=CPU_PROVIDERS)
