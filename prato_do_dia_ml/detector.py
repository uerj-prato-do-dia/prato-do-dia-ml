"""YOLOv11 ONNX detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from prato_do_dia_ml.onnx_runtime import create_cpu_session
from prato_do_dia_ml.preprocessing import LetterboxResult, letterbox_image, normalize_bgr_to_rgb
from prato_do_dia_ml.schema import Detection


class YoloOnnxDetector:
    """CPU-only YOLOv11 ONNX wrapper."""

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 640,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 20,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.session = create_cpu_session(self.model_path)

    def prepare_input(self, image_bgr: np.ndarray) -> tuple[np.ndarray, LetterboxResult]:
        """Letterbox and normalize an OpenCV BGR image to NCHW float input."""

        letterboxed = letterbox_image(image_bgr, size=self.input_size)
        image_rgb = normalize_bgr_to_rgb(letterboxed.image)
        tensor = np.transpose(image_rgb, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(tensor, dtype=np.float32), letterboxed

    def detect(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        """Run YOLOv11 detection and return boxes in original image coordinates."""

        input_tensor, letterbox = self.prepare_input(image_bgr)
        input_name = _first_input_name(self.session)
        outputs = self.session.run(None, {input_name: input_tensor})
        if not outputs:
            return ()

        return _decode_yolo_output(
            outputs[0],
            letterbox=letterbox,
            original_shape=image_bgr.shape[:2],
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
        )


def _first_input_name(session: ort.InferenceSession) -> str:
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("ONNX detector model has no inputs")
    return inputs[0].name


def _decode_yolo_output(
    output: np.ndarray,
    letterbox: LetterboxResult,
    original_shape: tuple[int, int],
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> tuple[Detection, ...]:
    predictions = np.asarray(output, dtype=np.float32)
    if predictions.ndim != 3:
        raise ValueError(f"expected YOLO output rank 3, got {predictions.shape}")
    if predictions.shape[0] != 1:
        raise ValueError(f"expected batch size 1, got {predictions.shape}")

    candidates = predictions[0]
    if candidates.shape[0] < candidates.shape[1]:
        candidates = candidates.T
    if candidates.shape[1] < 5:
        raise ValueError(f"expected YOLO rows with box + class scores, got {candidates.shape}")

    boxes_xywh = candidates[:, :4]
    class_scores = candidates[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = confidences >= confidence_threshold
    if not np.any(keep):
        return ()

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh[keep])
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    selected_indices = _nms(boxes_xyxy, confidences, iou_threshold)
    detections: list[Detection] = []
    original_height, original_width = original_shape

    for index in selected_indices[:max_detections]:
        prompt_box = tuple(float(value) for value in boxes_xyxy[index])
        original_box = _unletterbox_box(
            boxes_xyxy[index],
            letterbox=letterbox,
            original_width=original_width,
            original_height=original_height,
        )
        detections.append(
            Detection(
                class_id=int(class_ids[index]),
                confidence=float(confidences[index]),
                bbox_xyxy=original_box,
                prompt_xyxy=prompt_box,
            )
        )

    return tuple(detections)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = np.empty_like(boxes)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return result


def _unletterbox_box(
    box: np.ndarray,
    letterbox: LetterboxResult,
    original_width: int,
    original_height: int,
) -> tuple[float, float, float, float]:
    x1 = (box[0] - letterbox.pad_left) / letterbox.scale
    y1 = (box[1] - letterbox.pad_top) / letterbox.scale
    x2 = (box[2] - letterbox.pad_left) / letterbox.scale
    y2 = (box[3] - letterbox.pad_top) / letterbox.scale
    return (
        float(np.clip(x1, 0, original_width - 1)),
        float(np.clip(y1, 0, original_height - 1)),
        float(np.clip(x2, 0, original_width - 1)),
        float(np.clip(y2, 0, original_height - 1)),
    )


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = np.argsort(scores)[::-1]
    selected: list[int] = []

    while order.size > 0:
        current = int(order[0])
        selected.append(current)
        if order.size == 1:
            break

        ious = _box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]

    return selected


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-6)
