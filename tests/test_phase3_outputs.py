from __future__ import annotations

import cv2
import numpy as np

from prato_do_dia_ml.annotations import write_yolo_segmentation_txt
from prato_do_dia_ml.io_utils import validate_instance_mask_png
from prato_do_dia_ml.metrics import (
    dice_score,
    evaluate_instance_masks,
    intersection_over_union,
    rasterize_yolo_polygons,
)
from prato_do_dia_ml.postprocessing import resolve_overlaps
from prato_do_dia_ml.schema import SegmentationMask


def test_yolo_rasterization_fills_polygon(tmp_path) -> None:
    path = tmp_path / "mask.txt"
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

    assert int(mask.sum()) == 100


def test_instance_mask_validation_rejects_jpeg(tmp_path) -> None:
    png_path = tmp_path / "sample_instances.png"
    source = np.zeros((8, 8), dtype=np.uint8)
    source[1:4, 1:4] = 1
    assert cv2.imwrite(str(png_path), source)

    loaded = validate_instance_mask_png(png_path)

    assert loaded.shape == source.shape
    assert int(loaded.max()) == 1

    jpg_path = tmp_path / "sample_instances.jpg"
    assert cv2.imwrite(str(jpg_path), source)

    try:
        validate_instance_mask_png(jpg_path)
    except ValueError:
        return
    raise AssertionError("JPEG ground truth should be rejected")


def test_iou_dice_and_instance_metrics_match_identical_masks() -> None:
    predicted = np.zeros((10, 10), dtype=bool)
    ground_truth = np.zeros((10, 10), dtype=bool)
    predicted[:5, :5] = True
    ground_truth[:5, :5] = True

    assert intersection_over_union(predicted, ground_truth) == 1.0
    assert dice_score(predicted, ground_truth) == 1.0

    metrics = evaluate_instance_masks(predicted.astype(np.uint8), ground_truth.astype(np.uint8))

    assert metrics["miou"] == 1.0
    assert metrics["false_positives"] == 0
    assert metrics["missed_instances"] == 0


def test_overlap_resolution_keeps_high_confidence_overlap() -> None:
    low = np.zeros((12, 12), dtype=bool)
    high = np.zeros((12, 12), dtype=bool)
    low[2:8, 2:8] = True
    high[5:10, 5:10] = True

    resolved = resolve_overlaps(
        (
            SegmentationMask(
                0,
                0.6,
                ((0.1, 0.1), (0.7, 0.1), (0.7, 0.7)),
                low,
                yolo_confidence=0.6,
                sam_iou_prediction=0.2,
                area_px=int(low.sum()),
            ),
            SegmentationMask(
                0,
                0.8,
                ((0.4, 0.4), (0.9, 0.4), (0.9, 0.9)),
                high,
                yolo_confidence=0.8,
                sam_iou_prediction=0.9,
                area_px=int(high.sum()),
            ),
        ),
        (12, 12),
    )

    assert len(resolved) == 2
    assert resolved[1].mask is not None
    assert resolved[1].mask[5, 5]
