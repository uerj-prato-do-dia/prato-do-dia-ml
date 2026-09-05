from pathlib import Path

import pytest

from prato_do_dia_ml.inference import (
    FoodPredictor,
    MLInvalidImageError,
    MLModelUnavailableError,
)
from prato_do_dia_ml.schema import (
    PlatePredictionResponse,
    SegmentationItem,
)


def test_plate_prediction_response_creation_and_serialization() -> None:
    items = [
        SegmentationItem(
            class_id=4,
            class_name="arroz",
            confidence=0.85,
            box=[0.1, 0.1, 0.5, 0.5],
            polygon=[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
            relative_area_percentage=12.5,
        )
    ]
    response = PlatePredictionResponse.create(inference_time_ms=15.4, items=items)

    data = response.to_dict()

    assert data["inference_time_ms"] == 15.4
    assert data["plate_detected"] is True
    assert len(data["items"]) == 1
    assert data["items"][0]["class_id"] == 4
    assert data["items"][0]["class_name"] == "arroz"
    assert data["items"][0]["relative_area_percentage"] == 12.5


def test_plate_prediction_response_flagged_false_when_small_area() -> None:
    items = [
        SegmentationItem(
            class_id=12,
            class_name="azeitona",
            confidence=0.90,
            box=[0.1, 0.1, 0.15, 0.15],
            polygon=[[0.1, 0.1], [0.15, 0.1], [0.15, 0.15]],
            relative_area_percentage=0.8,
        )
    ]
    # Total food area (0.8%) < min_area_percentage (5.0%) -> plate_detected = False
    response = PlatePredictionResponse.create(inference_time_ms=10.0, items=items, min_area_percentage=5.0)

    data = response.to_dict()

    assert data["plate_detected"] is False


def test_predict_bytes_invalid_image() -> None:
    predictor = FoodPredictor.__new__(FoodPredictor)

    with pytest.raises(MLInvalidImageError):
        predictor.predict_bytes(b"not an image")


def test_from_models_dir_reports_missing_models(tmp_path: Path) -> None:
    with pytest.raises(MLModelUnavailableError):
        FoodPredictor.from_models_dir(tmp_path)
