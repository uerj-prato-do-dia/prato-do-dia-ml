from pathlib import Path

import pytest

from prato_do_dia_ml.inference import (
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
