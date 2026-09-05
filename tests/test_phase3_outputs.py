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
from prato_do_dia_ml.schema import SegmentationItem


def test_yolo_rasterization_fills_polygon(tmp_path) -> None:
    path = tmp_path / "mask.txt"
    item = SegmentationItem(
        class_id=0,
        class_name="tomate",
        confidence=1.0,
        box=[0.0, 0.0, 1.0, 1.0],
        polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        relative_area_percentage=100.0,
    )
    write_yolo_segmentation_txt([item], path)

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
