import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import prato_do_dia_ml.server as server_module
from prato_do_dia_ml.schema import PlatePredictionResponse, SegmentationItem
from prato_do_dia_ml.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert data["service"] == "prato-do-dia-ml"
    assert data["version"] == "1.0.0"


def test_predict_endpoint_rejects_non_image_payload(client: TestClient) -> None:
    response = client.post(
        "/v1/predict",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    assert "must be a valid JPEG or PNG" in response.json()["detail"]


def test_predict_endpoint_handles_valid_image(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyPredictor:
        def predict_bytes(self, image_bytes: bytes) -> PlatePredictionResponse:
            item = SegmentationItem(
                class_id=4,
                class_name="arroz",
                confidence=0.92,
                box=[0.1, 0.1, 0.5, 0.5],
                polygon=[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
                relative_area_percentage=15.0,
            )
            return PlatePredictionResponse.create(12.5, [item])

    monkeypatch.setattr(server_module, "get_server_predictor", lambda: DummyPredictor())

    # Create valid synthetic JPEG image
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    response = client.post(
        "/v1/predict",
        files={"file": ("meal.jpg", buf.tobytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["inference_time_ms"] == 12.5
    assert data["plate_detected"] is True
    assert len(data["items"]) == 1
    assert data["items"][0]["class_name"] == "arroz"
