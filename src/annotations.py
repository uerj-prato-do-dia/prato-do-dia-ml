"""Segmentation serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.schema import SegmentationMask


def write_yolo_segmentation_txt(
    masks: tuple[SegmentationMask, ...],
    output_path: str | Path,
) -> Path:
    """Write normalized YOLO segmentation polygons to a TXT file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [_format_mask(mask) for mask in masks]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _format_mask(mask: SegmentationMask) -> str:
    if mask.class_id < 0:
        raise ValueError("class_id must be non-negative")
    if len(mask.polygon) < 3:
        raise ValueError("YOLO segmentation polygons require at least 3 points")

    values: list[str] = [str(mask.class_id)]
    for x, y in mask.polygon:
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("polygon coordinates must be normalized to [0, 1]")
        values.extend((f"{x:.6f}", f"{y:.6f}"))

    return " ".join(values)


def mask_to_polygon(
    mask: np.ndarray,
    original_width: int,
    original_height: int,
) -> list[tuple[float, float]]:
    """Convert a binary mask to a normalized exterior polygon."""

    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return []

    epsilon = 0.002 * cv2.arcLength(contour, True)
    approximated = cv2.approxPolyDP(contour, epsilon, True)
    points = approximated.reshape(-1, 2)
    if len(points) < 3:
        return []

    return [
        (
            float(np.clip(x / max(original_width - 1, 1), 0.0, 1.0)),
            float(np.clip(y / max(original_height - 1, 1), 0.0, 1.0)),
        )
        for x, y in points
    ]


def write_mask_png(mask: np.ndarray, output_path: str | Path) -> Path:
    """Write a single-channel integer mask as PNG."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask.ndim != 2:
        raise ValueError("mask PNG output must be single-channel")
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"failed to write mask PNG: {path}")
    return path


def write_metadata_json(
    *,
    image_path: str | Path,
    width: int,
    height: int,
    model_versions: dict[str, str],
    segmentations: tuple[SegmentationMask, ...],
    output_path: str | Path,
) -> Path:
    """Write per-image segmentation metadata."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": str(image_path),
        "width": width,
        "height": height,
        "model_versions": model_versions,
        "instances": [_metadata_instance(segmentation) for segmentation in segmentations],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _metadata_instance(segmentation: SegmentationMask) -> dict[str, object]:
    return {
        "instance_id": segmentation.instance_id,
        "proposal_class_id": segmentation.class_id,
        "box_xyxy": list(segmentation.bbox_xyxy) if segmentation.bbox_xyxy is not None else None,
        "yolo_confidence": segmentation.yolo_confidence,
        "sam_iou_prediction": segmentation.sam_iou_prediction,
        "area_px": segmentation.area_px,
        "polygon": [[x, y] for x, y in segmentation.polygon],
    }
