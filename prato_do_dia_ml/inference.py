"""Public inference interface for API consumers."""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from prato_do_dia_ml.detector import YoloOnnxDetector
from prato_do_dia_ml.pipeline import FoodSegmentationPipeline
from prato_do_dia_ml.segmenter import SamOnnxSegmenter


class MLError(Exception):
    """Base class for public ML inference errors."""


class MLInvalidImageError(MLError):
    """Raised when image bytes cannot be decoded."""


class MLModelUnavailableError(MLError):
    """Raised when required model files are unavailable."""


class MLInferenceError(MLError):
    """Raised when inference fails for an otherwise valid request."""


@dataclass(frozen=True)
class PredictionModelInfo:
    pipeline: str = "yolo11_sam2_onnx"
    version: str = "baseline-2026-06-15"


@dataclass(frozen=True)
class PredictionImageInfo:
    width: int
    height: int


@dataclass(frozen=True)
class PredictionInstance:
    instance_id: int
    proposal_class_id: int
    label: str | None
    bbox: tuple[float, float, float, float]
    confidence: float
    area_px: int
    polygon: tuple[tuple[float, float], ...] | None = None
    sam_iou_prediction: float | None = None


@dataclass(frozen=True)
class PredictionResponse:
    schema_version: str
    image_width: int
    image_height: int
    instances: list[PredictionInstance] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    model_info: PredictionModelInfo = field(default_factory=PredictionModelInfo)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FoodPredictor:
    """Stable API-facing wrapper around the experimental segmentation pipeline."""

    def __init__(self, pipeline: FoodSegmentationPipeline) -> None:
        self._pipeline = pipeline

    @classmethod
    def from_models_dir(cls, models_dir: Path, config_path: Path | None = None) -> FoodPredictor:
        if config_path is not None:
            from prato_do_dia_ml.config import load_config

            return cls(FoodSegmentationPipeline.from_config(load_config(config_path)))

        yolo_path = models_dir / "yolov11_food.onnx"
        sam_encoder_path = models_dir / "sam2.1_hiera_tiny.encoder.onnx"
        sam_decoder_path = models_dir / "sam2.1_hiera_tiny.decoder.onnx"
        missing = [path for path in (yolo_path, sam_encoder_path, sam_decoder_path) if not path.exists()]
        if missing:
            names = ", ".join(path.name for path in missing)
            raise MLModelUnavailableError(f"missing model files: {names}")

        output_root = models_dir.parent / "data"
        pipeline = FoodSegmentationPipeline(
            YoloOnnxDetector(yolo_path, confidence_threshold=0.15, max_detections=10),
            SamOnnxSegmenter(sam_encoder_path, sam_decoder_path),
            output_dir=output_root / "raw_segmentations",
            mask_dir=output_root / "masks",
            overlay_dir=output_root / "overlays",
            report_dir=output_root / "reports",
        )
        return cls(pipeline)

    def predict_bytes(self, image_bytes: bytes) -> PredictionResponse:
        image = _decode_image(image_bytes)
        height, width = image.shape[:2]

        with tempfile.NamedTemporaryFile(suffix=".jpg") as handle:
            handle.write(image_bytes)
            handle.flush()
            try:
                result = self._pipeline.run_image(Path(handle.name))
            except MLError:
                raise
            except Exception as exc:
                raise MLInferenceError("inference failed") from exc

        instances: list[PredictionInstance] = []
        for index, segmentation in enumerate(result.segmentations, start=1):
            bbox = segmentation.bbox_xyxy
            if bbox is None:
                matching_detection = next(
                    (det for det in result.detections if det.class_id == segmentation.class_id),
                    None,
                )
                bbox = matching_detection.bbox_xyxy if matching_detection is not None else (0.0, 0.0, 0.0, 0.0)

            instances.append(
                PredictionInstance(
                    instance_id=segmentation.instance_id or index,
                    proposal_class_id=segmentation.class_id,
                    label=None,
                    bbox=tuple(float(value) for value in bbox),
                    confidence=float(segmentation.confidence or segmentation.yolo_confidence),
                    area_px=int(segmentation.area_px),
                    polygon=tuple((float(x), float(y)) for x, y in segmentation.polygon),
                    sam_iou_prediction=float(segmentation.sam_iou_prediction)
                    if segmentation.sam_iou_prediction is not None
                    else None,
                )
            )

        artifacts = {
            name: str(path)
            for name, path in {
                "annotation": result.annotation_path,
                "instance_mask": result.instance_mask_path,
                "class_mask": result.class_mask_path,
                "metadata": result.metadata_path,
                "overlay": result.overlay_path,
            }.items()
            if path is not None
        }
        warnings = ["no_detections"] if not instances else []
        return PredictionResponse(
            schema_version="1.0",
            image_width=width,
            image_height=height,
            instances=instances,
            warnings=warnings,
            artifacts=artifacts,
        )


def predict(image_bytes: bytes, models_dir: Path | None = None) -> PredictionResponse:
    root = Path(__file__).resolve().parent.parent
    predictor = FoodPredictor.from_models_dir(models_dir or root / "models")
    return predictor.predict_bytes(image_bytes)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise MLInvalidImageError("empty image bytes")
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise MLInvalidImageError("invalid image bytes")
    if image.ndim not in (2, 3):
        raise MLInvalidImageError("unsupported image shape")
    return image
