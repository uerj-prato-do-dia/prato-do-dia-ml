"""Public inference interface for YOLOv11-seg ONNX model execution."""

from __future__ import annotations

import threading
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from prato_do_dia_ml.onnx_runtime import create_cpu_session_options
from prato_do_dia_ml.postprocessing import fill_small_holes, remove_small_components
from prato_do_dia_ml.preprocessing import letterbox_image, normalize_bgr_to_rgb
from prato_do_dia_ml.schema import (
    CLASS_ID_TO_NAME,
    PlatePredictionResponse,
    SegmentationItem,
)

PredictionResponse = PlatePredictionResponse


class MLError(Exception):
    """Base class for public ML inference errors."""


class MLInvalidImageError(MLError):
    """Raised when image bytes cannot be decoded."""


class MLModelUnavailableError(MLError):
    """Raised when required model files are unavailable."""


class ModelIntegrityError(MLModelUnavailableError):
    """Raised when an ONNX model file fails SHA-256 integrity validation against models_manifest.json."""


class MLInferenceError(MLError):
    """Raised when inference fails for an otherwise valid request."""


def _mask_to_normalized_polygon(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    epsilon_ratio: float = 0.005,
) -> list[list[float]]:
    """Convert binary mask (uint8 0/255) into a single normalized simplified polygon."""
    if mask is None or np.count_nonzero(mask) == 0:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    largest_cnt = max(contours, key=cv2.contourArea)
    if len(largest_cnt) < 3:
        return []

    perimeter = cv2.arcLength(largest_cnt, closed=True)
    epsilon = epsilon_ratio * perimeter
    approx = cv2.approxPolyDP(largest_cnt, epsilon, closed=True)

    pts = approx.reshape((-1, 2))
    if len(pts) < 3:
        pts = largest_cnt.reshape((-1, 2))

    normalized: list[list[float]] = []
    for pt in pts:
        nx = float(max(0.0, min(1.0, pt[0] / img_w)))
        ny = float(max(0.0, min(1.0, pt[1] / img_h)))
        normalized.append([round(nx, 6), round(ny, 6)])

    return normalized


def _calculate_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model_checksum(model_path: Path, models_dir: Path | None = None) -> None:
    import json

    parent_dir = models_dir or model_path.parent
    manifest_path = parent_dir / "models_manifest.json"
    if not manifest_path.exists():
        manifest_path = parent_dir / "model_manifest.json"

    if not manifest_path.exists():
        return

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        models = manifest_data.get("models", [])
        expected_sha = None
        for m in models:
            if isinstance(m, dict) and m.get("filename") == model_path.name:
                expected_sha = m.get("sha256")
                break

        if expected_sha:
            actual_sha = _calculate_sha256(model_path)
            if actual_sha.lower() != str(expected_sha).lower():
                raise ModelIntegrityError(
                    f"Model integrity validation failed for '{model_path.name}'. "
                    f"Expected SHA-256: {expected_sha}, Actual SHA-256: {actual_sha}"
                )
    except ModelIntegrityError:
        raise
    except Exception as exc:
        raise ModelIntegrityError(f"Error checking model integrity: {exc}") from exc


class FoodPredictor:
    """Stable API-facing ONNX predictor for YOLOv11-seg single-stage food segmentation."""

    def __init__(
        self,
        model_path: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        verify_checksum: bool = True,
    ) -> None:
        self.model_path = model_path.resolve()
        if not self.model_path.exists():
            raise MLModelUnavailableError(f"Model file not found: {self.model_path}")

        if verify_checksum:
            _verify_model_checksum(self.model_path)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._lock = threading.Lock()

        sess_options = create_cpu_session_options()
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise MLModelUnavailableError(f"Failed to load ONNX session: {exc}") from exc

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    @classmethod
    def from_models_dir(
        cls,
        models_dir: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> FoodPredictor:
        candidate_names = ["best.onnx", "yolo11n_food_seg.onnx", "yolov11_food.onnx"]
        selected = None
        for name in candidate_names:
            p = models_dir / name
            if p.exists():
                selected = p
                break

        if selected is None:
            selected = models_dir / "best.onnx"

        return cls(selected, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

    def predict_bytes(self, image_bytes: bytes) -> PlatePredictionResponse:
        image_bgr = _decode_image(image_bytes)
        return self.predict_image(image_bgr)

    def predict_image(self, image_bgr: np.ndarray) -> PlatePredictionResponse:
        start_time = time.perf_counter()
        orig_h, orig_w = image_bgr.shape[:2]

        # 1. Letterbox & Normalize
        letterboxed = letterbox_image(image_bgr, size=(640, 640))
        rgb_norm = normalize_bgr_to_rgb(letterboxed.image)
        if rgb_norm.ndim == 3:
            chw = np.transpose(rgb_norm, (2, 0, 1))
            input_tensor = np.expand_dims(chw, axis=0)
        elif rgb_norm.ndim == 4:
            input_tensor = rgb_norm
        else:
            raise ValueError(f"Formato de imagem inválido para ONNX: {rgb_norm.shape}")

        # 2. Run ONNX Session (locked for thread safety on CPU)
        with self._lock:
            try:
                outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            except Exception as exc:
                raise MLInferenceError(f"ONNX execution failed: {exc}") from exc

        # Parse outputs
        output0 = outputs[0]  # Shape [1, 4 + 16 + 32, 8400] -> [1, 52, 8400]
        output1 = outputs[1] if len(outputs) > 1 else None  # Shape [1, 32, 160, 160]

        predictions = np.squeeze(output0, axis=0)  # Shape [52, 8400]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T  # Transpose to [8400, 52]

        # Split predictions into boxes (4), scores (16), mask_coeffs (32)
        boxes_cxcywh = predictions[:, :4]
        scores = predictions[:, 4:20]
        mask_coeffs = predictions[:, 20:52] if predictions.shape[1] >= 52 else None

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence threshold
        mask_filter = confidences >= self.conf_threshold
        if not np.any(mask_filter):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return PlatePredictionResponse.create(elapsed_ms, [])

        boxes_cxcywh = boxes_cxcywh[mask_filter]
        confidences = confidences[mask_filter]
        class_ids = class_ids[mask_filter]
        if mask_coeffs is not None:
            mask_coeffs = mask_coeffs[mask_filter]

        # Map bounding boxes from letterbox 640x640 back to original image space
        boxes_xywh_pixel = []
        boxes_xyxy_pixel = []
        pad_x, pad_y = letterboxed.pad_x, letterboxed.pad_y
        scale = letterboxed.scale

        for cx, cy, w, h in boxes_cxcywh:
            x1 = (cx - w / 2.0 - pad_x) / scale
            y1 = (cy - h / 2.0 - pad_y) / scale
            x2 = (cx + w / 2.0 - pad_x) / scale
            y2 = (cy + h / 2.0 - pad_y) / scale

            x1 = max(0.0, min(float(orig_w), x1))
            y1 = max(0.0, min(float(orig_h), y1))
            x2 = max(0.0, min(float(orig_w), x2))
            y2 = max(0.0, min(float(orig_h), y2))

            bw = x2 - x1
            bh = y2 - y1
            boxes_xywh_pixel.append([int(x1), int(y1), int(bw), int(bh)])
            boxes_xyxy_pixel.append([x1, y1, x2, y2])

        # 3. Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh_pixel,
            [float(c) for c in confidences],
            self.conf_threshold,
            self.iou_threshold,
        )

        if len(indices) == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return PlatePredictionResponse.create(elapsed_ms, [])

        indices = np.array(indices).flatten()
        items: list[SegmentationItem] = []
        total_img_area = orig_w * orig_h

        # 4. Decode Proto Masks (if output1 is available)
        proto_masks = None
        if output1 is not None and mask_coeffs is not None:
            # output1 shape [1, 32, 160, 160] -> [32, 160*160]
            num_protos = output1.shape[1]
            proto_h, proto_w = output1.shape[2], output1.shape[3]
            protos = output1.reshape((num_protos, proto_h * proto_w))

        for idx in indices:
            cid = int(class_ids[idx])
            conf = float(confidences[idx])
            cname = CLASS_ID_TO_NAME.get(cid, f"class_{cid}")
            x1, y1, x2, y2 = boxes_xyxy_pixel[idx]

            norm_box = [
                round(x1 / orig_w, 6),
                round(y1 / orig_h, 6),
                round(x2 / orig_w, 6),
                round(y2 / orig_h, 6),
            ]

            bin_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

            if proto_masks is not None and mask_coeffs is not None:
                # Dot product coeffs x protos -> sigmoid -> threshold
                coeff = mask_coeffs[idx]  # shape [32]
                raw_mask = (coeff @ protos).reshape((proto_h, proto_w))
                sig_mask = 1.0 / (1.0 + np.exp(-raw_mask))

                # Resize to 640x640 letterbox space
                mask_640 = cv2.resize(sig_mask, (640, 640), interpolation=cv2.INTER_LINEAR)
                # Unletterbox mask back to original image size
                crop_mask = mask_640[pad_y : 640 - pad_y, pad_x : 640 - pad_x]
                if crop_mask.size > 0:
                    mask_orig = cv2.resize(crop_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    bin_mask = (mask_orig >= 0.5).astype(np.uint8) * 255
            else:
                # Fallback: fill bounding box if proto masks not available
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                bin_mask[iy1:iy2, ix1:ix2] = 255

            # Morphological Cleanup
            bin_mask_bool = bin_mask > 0
            bin_mask_clean = remove_small_components(bin_mask_bool)
            bin_mask_clean = fill_small_holes(bin_mask_clean)
            bin_mask = (bin_mask_clean.astype(np.uint8)) * 255

            # Polygon Vectorization
            polygon = _mask_to_normalized_polygon(bin_mask, orig_w, orig_h, epsilon_ratio=0.005)
            if not polygon:
                # Fallback polygon from bounding box if contour extraction is empty
                polygon = [
                    [norm_box[0], norm_box[1]],
                    [norm_box[2], norm_box[1]],
                    [norm_box[2], norm_box[3]],
                    [norm_box[0], norm_box[3]],
                ]

            area_px = int(np.count_nonzero(bin_mask))
            rel_area_pct = (area_px / total_img_area * 100.0) if total_img_area > 0 else 0.0

            items.append(
                SegmentationItem(
                    class_id=cid,
                    class_name=cname,
                    confidence=conf,
                    box=norm_box,
                    polygon=polygon,
                    relative_area_percentage=rel_area_pct,
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return PlatePredictionResponse.create(elapsed_ms, items)


def predict(image_bytes: bytes, models_dir: Path | None = None) -> PlatePredictionResponse:
    root = Path(__file__).resolve().parent.parent
    predictor = get_predictor(models_dir or root / "models")
    return predictor.predict_bytes(image_bytes)


_PREDICTOR_CACHE: dict[Path, FoodPredictor] = {}
_PREDICTOR_CACHE_LOCK = threading.Lock()


def get_predictor(models_dir: Path) -> FoodPredictor:
    key = models_dir.resolve()
    with _PREDICTOR_CACHE_LOCK:
        predictor = _PREDICTOR_CACHE.get(key)
        if predictor is None:
            predictor = FoodPredictor.from_models_dir(models_dir)
            _PREDICTOR_CACHE[key] = predictor
        return predictor


def _decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise MLInvalidImageError("empty image bytes")
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            transposed = ImageOps.exif_transpose(img)
            rgb = transposed.convert("RGB")
            return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise MLInvalidImageError("invalid image bytes") from exc
