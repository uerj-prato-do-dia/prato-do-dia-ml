"""Smoke tests for Phase 3 mask serialization and instance metrics."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import write_yolo_segmentation_txt
from src.io_utils import validate_instance_mask_png
from src.metrics import dice_score, evaluate_instance_masks, intersection_over_union, rasterize_yolo_polygons
from src.postprocessing import resolve_overlaps
from src.schema import SegmentationMask


def main() -> None:
    _test_yolo_rasterization()
    _test_png_ground_truth_validation()
    _test_iou_and_dice()
    _test_overlap_resolution()
    print("phase 3 output smoke tests passed")


def _test_yolo_rasterization() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mask.txt"
        write_yolo_segmentation_txt(
            (
                SegmentationMask(
                    class_id=0,
                    confidence=1.0,
                    polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                ),
            ),
            path,
        )
        mask = rasterize_yolo_polygons(path, (10, 10))
        if mask.sum() != 100:
            raise AssertionError("YOLO polygon rasterization should fill the full image")


def _test_png_ground_truth_validation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample_instances.png"
        source = np.zeros((8, 8), dtype=np.uint8)
        source[1:4, 1:4] = 1
        cv2.imwrite(str(path), source)
        loaded = validate_instance_mask_png(path)
        if loaded.shape != source.shape or int(loaded.max()) != 1:
            raise AssertionError("PNG instance mask validation returned unexpected values")

        jpg_path = Path(directory) / "sample_instances.jpg"
        cv2.imwrite(str(jpg_path), source)
        try:
            validate_instance_mask_png(jpg_path)
        except ValueError:
            return
        raise AssertionError("JPEG ground truth should be rejected")


def _test_iou_and_dice() -> None:
    predicted = np.zeros((10, 10), dtype=bool)
    ground_truth = np.zeros((10, 10), dtype=bool)
    predicted[:5, :5] = True
    ground_truth[:5, :5] = True
    if intersection_over_union(predicted, ground_truth) != 1.0:
        raise AssertionError("identical masks should have IoU 1")
    if dice_score(predicted, ground_truth) != 1.0:
        raise AssertionError("identical masks should have Dice 1")

    metrics = evaluate_instance_masks(predicted.astype(np.uint8), ground_truth.astype(np.uint8))
    if metrics["miou"] != 1.0 or metrics["false_positives"] != 0 or metrics["missed_instances"] != 0:
        raise AssertionError("identical instance masks should match perfectly")


def _test_overlap_resolution() -> None:
    low = np.zeros((12, 12), dtype=bool)
    high = np.zeros((12, 12), dtype=bool)
    low[2:8, 2:8] = True
    high[5:10, 5:10] = True
    resolved = resolve_overlaps(
        (
            SegmentationMask(0, 0.6, ((0.1, 0.1), (0.7, 0.1), (0.7, 0.7)), low, yolo_confidence=0.6, sam_iou_prediction=0.2, area_px=int(low.sum())),
            SegmentationMask(0, 0.8, ((0.4, 0.4), (0.9, 0.4), (0.9, 0.9)), high, yolo_confidence=0.8, sam_iou_prediction=0.9, area_px=int(high.sum())),
        ),
        (12, 12),
    )
    if len(resolved) != 2:
        raise AssertionError("overlap resolution should keep both non-empty masks")
    if not resolved[1].mask[5, 5]:
        raise AssertionError("higher-confidence mask should own overlap pixels")


if __name__ == "__main__":
    main()
