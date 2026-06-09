"""Visual debugging images for segmentation experiments."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from prato_do_dia_ml.schema import Detection

GT_COLOR = (0, 255, 0)
PRED_COLOR = (0, 0, 255)
FALSE_POSITIVE_COLOR = (0, 0, 255)
MISSED_COLOR = (0, 255, 255)
TRUE_POSITIVE_COLOR = (0, 180, 0)
BOX_COLOR = (255, 180, 0)


def render_debug_overlay(
    image_bgr: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    detections: tuple[Detection, ...],
    output_path: str | Path,
) -> Path:
    """Render original, GT contour, prediction contour, and error panels."""

    if image_bgr.shape[:2] != ground_truth.shape or ground_truth.shape != prediction.shape:
        raise ValueError("image, ground-truth, and prediction shapes must match")

    original = image_bgr.copy()
    _draw_detections(original, detections)

    gt_panel = image_bgr.copy()
    _draw_mask_contours(gt_panel, ground_truth != 0, GT_COLOR, thickness=2)
    _put_label(gt_panel, "ground truth", GT_COLOR)

    pred_panel = image_bgr.copy()
    _draw_mask_contours(pred_panel, prediction != 0, PRED_COLOR, thickness=2)
    _draw_detections(pred_panel, detections)
    _put_label(pred_panel, "prediction", PRED_COLOR)

    error_panel = _error_panel(image_bgr, ground_truth != 0, prediction != 0)
    _put_label(error_panel, "green=hit red=fp yellow=missed", (255, 255, 255))
    _put_label(original, "original + YOLO boxes", BOX_COLOR)

    canvas = np.concatenate((original, gt_panel, pred_panel, error_panel), axis=1)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write debug overlay: {path}")
    return path


def _draw_detections(image_bgr: np.ndarray, detections: tuple[Detection, ...]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{detection.class_id}:{detection.confidence:.2f}"
        cv2.putText(
            image_bgr,
            label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            BOX_COLOR,
            1,
            cv2.LINE_AA,
        )


def _draw_mask_contours(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_bgr, contours, contourIdx=-1, color=color, thickness=thickness)


def _error_panel(image_bgr: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    panel = image_bgr.copy()
    layer = panel.copy()
    true_positive = np.logical_and(gt, pred)
    false_positive = np.logical_and(~gt, pred)
    missed = np.logical_and(gt, ~pred)
    layer[true_positive] = TRUE_POSITIVE_COLOR
    layer[false_positive] = FALSE_POSITIVE_COLOR
    layer[missed] = MISSED_COLOR
    panel = cv2.addWeighted(layer, 0.55, panel, 0.45, 0.0)
    _draw_mask_contours(panel, false_positive, FALSE_POSITIVE_COLOR, thickness=2)
    _draw_mask_contours(panel, missed, MISSED_COLOR, thickness=2)
    return panel


def _put_label(image_bgr: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(image_bgr, (0, 0), (max(220, len(text) * 9), 24), (0, 0, 0), -1)
    cv2.putText(image_bgr, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
