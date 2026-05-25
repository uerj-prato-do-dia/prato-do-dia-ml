"""Shared data structures for detection and segmentation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


BBoxXYXY = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class Detection:
    """Single detector result in original image pixel coordinates."""

    class_id: int
    confidence: float
    bbox_xyxy: BBoxXYXY
    prompt_xyxy: BBoxXYXY | None = None


@dataclass(frozen=True)
class SegmentationMask:
    """Single segmentation result with a normalized YOLO polygon."""

    class_id: int
    confidence: float
    polygon: tuple[Point, ...]
    mask: np.ndarray | None = None
    instance_id: int = 0
    bbox_xyxy: BBoxXYXY | None = None
    yolo_confidence: float = 0.0
    sam_iou_prediction: float = 0.0
    area_px: int = 0

    def with_updates(self, **changes: object) -> "SegmentationMask":
        """Return a copy with selected fields replaced."""

        return replace(self, **changes)


@dataclass(frozen=True)
class PipelineResult:
    """Output paths and predictions produced for one input image."""

    image_path: Path
    annotation_path: Path
    instance_mask_path: Path | None
    class_mask_path: Path | None
    metadata_path: Path | None
    overlay_path: Path | None
    detections: tuple[Detection, ...]
    segmentations: tuple[SegmentationMask, ...]
