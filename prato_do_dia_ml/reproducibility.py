"""Reproducibility helpers for experiment runs."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from prato_do_dia_ml.config import PipelineConfig

TRACKED_PACKAGES = (
    "numpy",
    "opencv-python",
    "onnxruntime",
    "pandas",
    "scikit-image",
    "scipy",
)


def set_random_seed(seed: int) -> None:
    """Seed Python and NumPy for deterministic surrounding logic."""

    random.seed(seed)
    np.random.seed(seed)


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    """Convert the typed config dataclasses to plain serializable values."""

    return {
        "paths": {
            "input_dir": str(config.paths.input_dir),
            "ground_truth_dir": str(config.paths.ground_truth_dir),
            "output_dir": str(config.paths.output_dir),
            "mask_dir": str(config.paths.mask_dir),
            "overlay_dir": str(config.paths.overlay_dir),
            "report_dir": str(config.paths.report_dir),
        },
        "image": {
            "canonical_format": config.image.canonical_format,
            "background_rgb": list(config.image.background_rgb),
            "allow_alpha_input": config.image.allow_alpha_input,
        },
        "yolo": {
            "model_path": str(config.yolo.model_path),
            "imgsz": config.yolo.imgsz,
            "confidence_threshold": config.yolo.confidence_threshold,
            "nms_iou_threshold": config.yolo.nms_iou_threshold,
            "max_detections": config.yolo.max_detections,
        },
        "sam2": {
            "encoder_path": str(config.sam2.encoder_path),
            "decoder_path": str(config.sam2.decoder_path),
            "imgsz": config.sam2.imgsz,
            "mask_threshold": config.sam2.mask_threshold,
        },
        "postprocess": {
            "min_mask_area_ratio": config.postprocess.min_mask_area_ratio,
            "fill_holes_px": config.postprocess.fill_holes_px,
            "remove_components_px": config.postprocess.remove_components_px,
            "overlap_policy": config.postprocess.overlap_policy,
        },
        "evaluation": {"metrics": list(config.evaluation.metrics)},
    }


def write_yaml(data: dict[str, Any], output_path: str | Path) -> Path:
    """Write a small YAML file without adding a YAML dependency."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_to_yaml(data), encoding="utf-8")
    return path


def write_environment(output_path: str | Path) -> Path:
    """Write Python, package, git, and hardware metadata."""

    info = collect_environment()
    lines = [f"{key}: {value}" for key, value in info.items() if key != "packages"]
    lines.append("packages:")
    for package, version in info["packages"].items():
        lines.append(f"  {package}: {version}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def collect_environment() -> dict[str, Any]:
    """Collect deterministic environment metadata for a run."""

    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_commit": git_commit_hash(),
        "cuda_available": _cuda_available(),
        "gpu_name": _gpu_name(),
        "onnxruntime_providers": _onnxruntime_providers(),
        "packages": {package: _package_version(package) for package in TRACKED_PACKAGES},
    }


def model_versions(config: PipelineConfig) -> dict[str, Any]:
    """Return model path and source metadata for the current config."""

    return {
        "yolo": {
            "name": config.yolo.model_path.name,
            "path": str(config.yolo.model_path),
            "source": "local ONNX file",
            "input_size": config.yolo.imgsz,
        },
        "sam2": {
            "encoder_name": config.sam2.encoder_path.name,
            "encoder_path": str(config.sam2.encoder_path),
            "decoder_name": config.sam2.decoder_path.name,
            "decoder_path": str(config.sam2.decoder_path),
            "source": "local ONNX files",
            "input_size": config.sam2.imgsz,
        },
    }


def write_model_versions(config: PipelineConfig, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model_versions(config), indent=2), encoding="utf-8")
    return path


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _onnxruntime_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    return list(ort.get_available_providers())


def _cuda_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _gpu_name() -> str:
    if shutil.which("nvidia-smi") is None:
        return "none"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ", ".join(names) if names else "unknown"


def _to_yaml(value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(_to_yaml(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{prefix}{key}:")
                for entry in item:
                    lines.append(f"{prefix}  - {_format_scalar(entry)}")
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{_format_scalar(value)}"


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int | float):
        return str(value)
    text = str(value)
    if not text or any(char in text for char in ":#[]{}"):
        return json.dumps(text)
    return text
