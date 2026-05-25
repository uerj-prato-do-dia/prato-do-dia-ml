"""Deterministic mask metrics for visual validation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import find_boundaries

Color = tuple[int, int, int]

BACKGROUND_CLASS_ID = 0

GROUND_TRUTH_COLOR_TO_CLASS: dict[Color, int] = {
    (0, 0, 0): BACKGROUND_CLASS_ID,
    (255, 255, 255): 1,
    (255, 238, 0): 2,
    (142, 16, 142): 3,
    (50, 255, 0): 4,
    (102, 14, 13): 5,
    (0, 45, 255): 6,
    (254, 0, 0): 7,
    (19, 104, 0): 8,
    (1, 13, 255): 9,
    (254, 230, 0): 10,
}


def ground_truth_rgb_to_class_mask(
    image_rgb: np.ndarray,
    color_to_class: dict[Color, int] = GROUND_TRUTH_COLOR_TO_CLASS,
) -> np.ndarray:
    """Convert an exact-color ground-truth image to a 2D class mask.

    Unknown colors raise immediately. This intentionally rejects JPEG artifacts
    and other corrupted masks instead of silently clustering or thresholding.
    """

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("ground-truth image must have shape HxWx3")

    class_mask = np.full(image_rgb.shape[:2], -1, dtype=np.int32)
    for color, class_id in color_to_class.items():
        color_array = np.array(color, dtype=image_rgb.dtype)
        matches = np.all(image_rgb == color_array, axis=2)
        class_mask[matches] = class_id

    unknown = class_mask < 0
    if np.any(unknown):
        unknown_colors = np.unique(image_rgb[unknown].reshape(-1, 3), axis=0)
        sample = [tuple(int(value) for value in color) for color in unknown_colors[:10]]
        raise ValueError(
            "ground-truth image contains colors outside the exact mapping. "
            f"Unknown color count={len(unknown_colors)} sample={sample}. "
            "Use lossless PNG masks or update GROUND_TRUTH_COLOR_TO_CLASS."
        )

    return class_mask


def load_ground_truth_class_mask(path: str | Path) -> np.ndarray:
    """Load a ground-truth image and convert it through the exact RGB mapping."""

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"could not load ground-truth image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return ground_truth_rgb_to_class_mask(image_rgb)


def rasterize_yolo_polygons(
    txt_path: str | Path,
    image_shape: tuple[int, int],
    class_ids: set[int] | None = None,
) -> np.ndarray:
    """Rasterize normalized YOLO segmentation TXT polygons into a boolean mask."""

    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive height and width")

    mask = np.zeros((height, width), dtype=np.uint8)
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"YOLO annotation not found: {path}")

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        class_id, points = parse_yolo_polygon_line(line, line_number)
        if class_ids is not None and class_id not in class_ids:
            continue

        pixel_points = np.array(
            [
                [
                    int(round(np.clip(x, 0.0, 1.0) * (width - 1))),
                    int(round(np.clip(y, 0.0, 1.0) * (height - 1))),
                ]
                for x, y in points
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pixel_points], 1)

    return mask.astype(bool)


def intersection_over_union(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate IoU as area of overlap divided by area of union."""

    if predicted.shape != ground_truth.shape:
        raise ValueError(f"mask shapes differ: predicted={predicted.shape} gt={ground_truth.shape}")

    pred = predicted.astype(bool)
    gt = ground_truth.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    return float(intersection / union)


def dice_score(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate Dice/F1 overlap for two binary masks."""

    if predicted.shape != ground_truth.shape:
        raise ValueError(f"mask shapes differ: predicted={predicted.shape} gt={ground_truth.shape}")
    pred = predicted.astype(bool)
    gt = ground_truth.astype(bool)
    denominator = pred.sum() + gt.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denominator)


def boundary_f_score(predicted: np.ndarray, ground_truth: np.ndarray, tolerance_px: int = 2) -> float:
    """Calculate a simple boundary F-score with pixel tolerance."""

    if predicted.shape != ground_truth.shape:
        raise ValueError(f"mask shapes differ: predicted={predicted.shape} gt={ground_truth.shape}")
    pred_boundary = find_boundaries(predicted.astype(bool), mode="outer")
    gt_boundary = find_boundaries(ground_truth.astype(bool), mode="outer")
    if not np.any(pred_boundary) and not np.any(gt_boundary):
        return 1.0
    if not np.any(pred_boundary) or not np.any(gt_boundary):
        return 0.0

    kernel = np.ones((tolerance_px * 2 + 1, tolerance_px * 2 + 1), dtype=np.uint8)
    pred_dilated = cv2.dilate(pred_boundary.astype(np.uint8), kernel).astype(bool)
    gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), kernel).astype(bool)
    precision = np.logical_and(pred_boundary, gt_dilated).sum() / max(pred_boundary.sum(), 1)
    recall = np.logical_and(gt_boundary, pred_dilated).sum() / max(gt_boundary.sum(), 1)
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def evaluate_instance_masks(predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, object]:
    """Evaluate predicted and ground-truth instance-ID masks."""

    if predicted.shape != ground_truth.shape:
        raise ValueError(f"mask shapes differ: predicted={predicted.shape} gt={ground_truth.shape}")

    pred_ids = [int(value) for value in np.unique(predicted) if value != 0]
    gt_ids = [int(value) for value in np.unique(ground_truth) if value != 0]
    if not pred_ids and not gt_ids:
        return {
            "matches": [],
            "miou": 1.0,
            "mean_dice": 1.0,
            "mean_boundary_f": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "false_positives": 0,
            "missed_instances": 0,
            "mean_area_error": 0.0,
        }

    iou_matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    for pred_index, pred_id in enumerate(pred_ids):
        pred_mask = predicted == pred_id
        for gt_index, gt_id in enumerate(gt_ids):
            iou_matrix[pred_index, gt_index] = intersection_over_union(pred_mask, ground_truth == gt_id)

    matches = []
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    if iou_matrix.size:
        pred_indices, gt_indices = linear_sum_assignment(1.0 - iou_matrix)
        for pred_index, gt_index in zip(pred_indices, gt_indices, strict=True):
            iou = float(iou_matrix[pred_index, gt_index])
            if iou <= 0.0:
                continue
            pred_id = pred_ids[pred_index]
            gt_id = gt_ids[gt_index]
            pred_mask = predicted == pred_id
            gt_mask = ground_truth == gt_id
            matched_pred.add(pred_id)
            matched_gt.add(gt_id)
            gt_area = int(gt_mask.sum())
            pred_area = int(pred_mask.sum())
            matches.append(
                {
                    "pred_instance_id": pred_id,
                    "gt_instance_id": gt_id,
                    "iou": iou,
                    "dice": dice_score(pred_mask, gt_mask),
                    "boundary_f": boundary_f_score(pred_mask, gt_mask),
                    "area_error": float((pred_area - gt_area) / max(gt_area, 1)),
                }
            )

    false_positives = len([pred_id for pred_id in pred_ids if pred_id not in matched_pred])
    missed_instances = len([gt_id for gt_id in gt_ids if gt_id not in matched_gt])
    precision = len(matches) / max(len(pred_ids), 1)
    recall = len(matches) / max(len(gt_ids), 1)
    return {
        "matches": matches,
        "miou": _mean([float(item["iou"]) for item in matches]),
        "mean_dice": _mean([float(item["dice"]) for item in matches]),
        "mean_boundary_f": _mean([float(item["boundary_f"]) for item in matches]),
        "precision": float(precision),
        "recall": float(recall),
        "false_positives": false_positives,
        "missed_instances": missed_instances,
        "mean_area_error": _mean([abs(float(item["area_error"])) for item in matches]),
    }


def foreground_mask_from_class_mask(class_mask: np.ndarray) -> np.ndarray:
    """Return all non-background pixels as a boolean foreground mask."""

    return class_mask != BACKGROUND_CLASS_ID


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def parse_yolo_polygon_line(
    line: str,
    line_number: int,
) -> tuple[int, list[tuple[float, float]]]:
    parts = line.split()
    if len(parts) < 7 or len(parts[1:]) % 2 != 0:
        raise ValueError(f"invalid YOLO polygon at line {line_number}: {line!r}")

    class_id = int(parts[0])
    values = [float(value) for value in parts[1:]]
    points = list(zip(values[0::2], values[1::2], strict=True))
    return class_id, points
