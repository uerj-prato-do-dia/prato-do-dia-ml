"""Typed TOML configuration for the segmentation pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    input_dir: Path
    ground_truth_dir: Path
    output_dir: Path
    mask_dir: Path
    overlay_dir: Path
    report_dir: Path


@dataclass(frozen=True)
class ImageConfig:
    canonical_format: str
    background_rgb: tuple[int, int, int]
    allow_alpha_input: bool


@dataclass(frozen=True)
class YoloConfig:
    model_path: Path
    imgsz: int
    confidence_threshold: float
    nms_iou_threshold: float
    max_detections: int


@dataclass(frozen=True)
class Sam2Config:
    encoder_path: Path
    decoder_path: Path
    imgsz: int
    mask_threshold: float


@dataclass(frozen=True)
class PostprocessConfig:
    min_mask_area_ratio: float
    fill_holes_px: int
    remove_components_px: int
    overlap_policy: str


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class PipelineConfig:
    paths: PathConfig
    image: ImageConfig
    yolo: YoloConfig
    sam2: Sam2Config
    postprocess: PostprocessConfig
    evaluation: EvaluationConfig


def load_config(path: str | Path = "configs/default.toml") -> PipelineConfig:
    """Load pipeline configuration from a TOML file."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    return PipelineConfig(
        paths=PathConfig(
            input_dir=Path(data["paths"]["input_dir"]),
            ground_truth_dir=Path(data["paths"]["ground_truth_dir"]),
            output_dir=Path(data["paths"]["output_dir"]),
            mask_dir=Path(data["paths"]["mask_dir"]),
            overlay_dir=Path(data["paths"]["overlay_dir"]),
            report_dir=Path(data["paths"]["report_dir"]),
        ),
        image=ImageConfig(
            canonical_format=str(data["image"]["canonical_format"]),
            background_rgb=_rgb_tuple(data["image"]["background_rgb"]),
            allow_alpha_input=bool(data["image"]["allow_alpha_input"]),
        ),
        yolo=YoloConfig(
            model_path=Path(data["yolo"]["model_path"]),
            imgsz=int(data["yolo"]["imgsz"]),
            confidence_threshold=float(data["yolo"]["confidence_threshold"]),
            nms_iou_threshold=float(data["yolo"]["nms_iou_threshold"]),
            max_detections=int(data["yolo"]["max_detections"]),
        ),
        sam2=Sam2Config(
            encoder_path=Path(data["sam2"]["encoder_path"]),
            decoder_path=Path(data["sam2"]["decoder_path"]),
            imgsz=int(data["sam2"]["imgsz"]),
            mask_threshold=float(data["sam2"]["mask_threshold"]),
        ),
        postprocess=PostprocessConfig(
            min_mask_area_ratio=float(data["postprocess"]["min_mask_area_ratio"]),
            fill_holes_px=int(data["postprocess"]["fill_holes_px"]),
            remove_components_px=int(data["postprocess"]["remove_components_px"]),
            overlap_policy=str(data["postprocess"]["overlap_policy"]),
        ),
        evaluation=EvaluationConfig(metrics=tuple(str(item) for item in data["evaluation"]["metrics"])),
    )


def _rgb_tuple(value: object) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("background_rgb must contain exactly three values")
    rgb = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError("background_rgb values must be in [0, 255]")
    return rgb
