from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import prato_do_dia_ml.inference as inference_module
from prato_do_dia_ml.inference import (
    DEFAULT_MAX_DETECTIONS,
    FoodPredictor,
    MLInvalidImageError,
    MLModelUnavailableError,
    PredictionInstance,
    PredictionResponse,
)


def test_prediction_response_serializes() -> None:
    response = PredictionResponse(
        schema_version="1.0",
        image_width=640,
        image_height=640,
        instances=[
            PredictionInstance(
                instance_id=1,
                proposal_class_id=4,
                label="rice",
                bbox=(1.0, 2.0, 3.0, 4.0),
                confidence=0.8,
                area_px=123,
            )
        ],
    )

    data = response.to_dict()

    assert data["schema_version"] == "1.0"
    assert data["instances"][0]["proposal_class_id"] == 4


def test_predict_bytes_invalid_image() -> None:
    predictor = FoodPredictor.__new__(FoodPredictor)

    with pytest.raises(MLInvalidImageError):
        predictor.predict_bytes(b"not an image")


def test_from_models_dir_reports_missing_models(tmp_path: Path) -> None:
    with pytest.raises(MLModelUnavailableError):
        FoodPredictor.from_models_dir(tmp_path)


def test_from_models_dir_uses_default_detection_limit_and_isolated_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ["yolov11_food.onnx", "sam2.1_hiera_tiny.encoder.onnx", "sam2.1_hiera_tiny.decoder.onnx"]:
        (tmp_path / name).write_bytes(b"model")

    class DummyDetector:
        def __init__(self, model_path: Path, *, confidence_threshold: float, max_detections: int) -> None:
            self.model_path = model_path
            self.confidence_threshold = confidence_threshold
            self.max_detections = max_detections

    class DummySegmenter:
        def __init__(self, encoder_path: Path, decoder_path: Path) -> None:
            self.encoder_path = encoder_path
            self.decoder_path = decoder_path

    class DummyPipeline:
        def __init__(self, detector: DummyDetector, segmenter: DummySegmenter) -> None:
            self.detector = detector
            self.segmenter = segmenter
            self.output_dir = Path("data/raw_segmentations")
            self.mask_dir = Path("data/masks")
            self.overlay_dir = Path("data/overlays")
            self.report_dir = Path("data/reports")

    monkeypatch.setattr(inference_module, "YoloOnnxDetector", DummyDetector)
    monkeypatch.setattr(inference_module, "SamOnnxSegmenter", DummySegmenter)
    monkeypatch.setattr(inference_module, "FoodSegmentationPipeline", DummyPipeline)

    predictor = FoodPredictor.from_models_dir(tmp_path)
    pipeline = predictor._pipeline

    assert pipeline.detector.max_detections == DEFAULT_MAX_DETECTIONS
    assert "prato-do-dia-inference-" in str(pipeline.output_dir)
    assert pipeline.output_dir.name == "raw_segmentations"
    assert pipeline.mask_dir.name == "masks"
    assert pipeline.overlay_dir.name == "overlays"
    assert pipeline.report_dir.name == "reports"


def test_predict_reuses_cached_predictor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    class FakePredictor:
        def predict_bytes(self, image_bytes: bytes) -> PredictionResponse:
            return PredictionResponse(schema_version="1.0", image_width=1, image_height=1)

    def fake_from_models_dir(cls, models_dir: Path, config_path: Path | None = None) -> FakePredictor:
        calls.append(models_dir)
        return FakePredictor()

    inference_module._PREDICTOR_CACHE.clear()
    monkeypatch.setattr(FoodPredictor, "from_models_dir", classmethod(fake_from_models_dir))

    inference_module.predict(b"first", models_dir=tmp_path)
    inference_module.predict(b"second", models_dir=tmp_path)

    assert calls == [tmp_path]


def test_predict_bytes_writes_closed_temp_input_before_pipeline() -> None:
    image = np.full((8, 8, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    class FakePipeline:
        def run_image(self, image_path: Path):
            path = Path(image_path)
            assert path.exists()
            assert path.read_bytes()
            return SimpleNamespace(
                segmentations=[],
                detections=[],
                annotation_path=None,
                instance_mask_path=None,
                class_mask_path=None,
                metadata_path=None,
                overlay_path=None,
            )

    predictor = FoodPredictor(FakePipeline())
    response = predictor.predict_bytes(encoded.tobytes())

    assert response.warnings == ["no_detections"]
