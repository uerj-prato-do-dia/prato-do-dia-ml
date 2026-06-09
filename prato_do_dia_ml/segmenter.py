"""SAM 2 ONNX segmenter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from prato_do_dia_ml.annotations import mask_to_polygon
from prato_do_dia_ml.onnx_runtime import create_cpu_session
from prato_do_dia_ml.preprocessing import LetterboxResult, letterbox_image, normalize_bgr_to_rgb
from prato_do_dia_ml.schema import Detection, SegmentationMask

SAM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SAM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SamOnnxSegmenter:
    """CPU-only SAM 2 ONNX wrapper for box-prompted masks."""

    def __init__(
        self,
        encoder_path: str | Path,
        decoder_path: str | Path,
        input_size: int = 1024,
        mask_threshold: float = 0.0,
    ) -> None:
        self.encoder_path = Path(encoder_path)
        self.decoder_path = Path(decoder_path)
        self.input_size = input_size
        self.mask_threshold = mask_threshold
        self.encoder_session = create_cpu_session(self.encoder_path)
        self.decoder_session = create_cpu_session(self.decoder_path)

    def segment(
        self,
        image_bgr: np.ndarray,
        detections: tuple[Detection, ...],
    ) -> tuple[SegmentationMask, ...]:
        """Generate masks once the SAM 2 ONNX input/output contract is fixed."""

        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must have shape HxWx3")
        if not detections:
            return ()

        image_tensor, letterbox = self._prepare_encoder_input(image_bgr)
        encoder_outputs = self.encoder_session.run(None, {"image": image_tensor})
        if len(encoder_outputs) != 3:
            raise RuntimeError("SAM encoder must return high_res_feats_0, high_res_feats_1, image_embed")

        segmentations: list[SegmentationMask] = []
        for detection in detections:
            point_coords, point_labels = _box_to_prompt_tensors(detection, letterbox)
            decoder_inputs = {
                "image_embed": encoder_outputs[2],
                "high_res_feats_0": encoder_outputs[0],
                "high_res_feats_1": encoder_outputs[1],
                "point_coords": point_coords,
                "point_labels": point_labels,
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.zeros((1,), dtype=np.float32),
            }
            masks, iou_predictions = self.decoder_session.run(None, decoder_inputs)
            segmentations.extend(
                _masks_to_segmentations(
                    masks=np.asarray(masks),
                    iou_predictions=np.asarray(iou_predictions),
                    detections=(detection,),
                    letterbox=letterbox,
                    original_shape=image_bgr.shape[:2],
                    threshold=self.mask_threshold,
                    input_size=self.input_size,
                )
            )

        return tuple(segmentations)

    def _prepare_encoder_input(self, image_bgr: np.ndarray) -> tuple[np.ndarray, LetterboxResult]:
        letterboxed = letterbox_image(image_bgr, size=self.input_size)
        image_rgb = normalize_bgr_to_rgb(letterboxed.image)
        image_rgb = (image_rgb - SAM_MEAN) / SAM_STD
        tensor = np.transpose(image_rgb, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(tensor, dtype=np.float32), letterboxed


def _box_to_prompt_tensors(
    detection: Detection,
    letterbox: LetterboxResult,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.zeros((1, 2, 2), dtype=np.float32)
    labels = np.array([[2.0, 3.0]], dtype=np.float32)

    x1, y1, x2, y2 = detection.bbox_xyxy
    coords[0, 0] = (
        x1 * letterbox.scale + letterbox.pad_left,
        y1 * letterbox.scale + letterbox.pad_top,
    )
    coords[0, 1] = (
        x2 * letterbox.scale + letterbox.pad_left,
        y2 * letterbox.scale + letterbox.pad_top,
    )

    return coords, labels


def _masks_to_segmentations(
    masks: np.ndarray,
    iou_predictions: np.ndarray,
    detections: tuple[Detection, ...],
    letterbox: LetterboxResult,
    original_shape: tuple[int, int],
    threshold: float,
    input_size: int,
) -> tuple[SegmentationMask, ...]:
    if masks.ndim == 3:
        masks = masks[:, None, :, :]
    if masks.ndim != 4:
        raise ValueError(f"expected SAM masks with rank 4, got {masks.shape}")

    original_height, original_width = original_shape
    segmentations: list[SegmentationMask] = []

    for index, detection in enumerate(detections):
        mask_set = masks[index]
        scores = iou_predictions[index] if iou_predictions.ndim == 2 else np.zeros((mask_set.shape[0],))
        best_mask_index = int(np.argmax(scores)) if scores.size else 0
        best_mask = mask_set[best_mask_index]
        confidence = float(scores[best_mask_index]) if scores.size else detection.confidence

        original_mask = _restore_mask_to_original_image(
            best_mask,
            letterbox=letterbox,
            original_width=original_width,
            original_height=original_height,
            threshold=threshold,
            input_size=input_size,
        )
        polygon = _mask_to_polygon(original_mask, original_width, original_height)
        if not polygon:
            continue

        segmentations.append(
            SegmentationMask(
                class_id=detection.class_id,
                confidence=confidence,
                polygon=tuple(polygon),
                mask=original_mask,
                bbox_xyxy=detection.bbox_xyxy,
                yolo_confidence=detection.confidence,
                sam_iou_prediction=confidence,
                area_px=int(original_mask.sum()),
            )
        )

    return tuple(segmentations)


def _restore_mask_to_original_image(
    mask: np.ndarray,
    letterbox: LetterboxResult,
    original_width: int,
    original_height: int,
    threshold: float,
    input_size: int,
) -> np.ndarray:
    mask_1024 = cv2.resize(
        mask.astype(np.float32),
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR,
    )
    crop = mask_1024[
        letterbox.pad_top : letterbox.pad_top + letterbox.resized_height,
        letterbox.pad_left : letterbox.pad_left + letterbox.resized_width,
    ]
    restored = cv2.resize(crop, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    return restored > threshold


def _mask_to_polygon(
    mask: np.ndarray,
    original_width: int,
    original_height: int,
) -> list[tuple[float, float]]:
    return mask_to_polygon(mask, original_width, original_height)
