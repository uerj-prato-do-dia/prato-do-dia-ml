"""FastAPI Microservice for Prato do Dia ML Food Segmentation Inference."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypedDict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from prato_do_dia_ml.inference import (
    FoodPredictor,
    MLInferenceError,
    MLInvalidImageError,
    MLModelUnavailableError,
    get_predictor,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_PREDICTOR: FoodPredictor | None = None


def get_server_predictor() -> FoodPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = get_predictor(MODELS_DIR)
    return _PREDICTOR


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup warmup and model session allocation."""
    try:
        predictor = get_server_predictor()
        # Warmup pass with 640x640 dummy image
        import numpy as np

        dummy_bgr = np.zeros((640, 640, 3), dtype=np.uint8)
        await asyncio.to_thread(predictor.predict_image, dummy_bgr)
    except Exception as exc:
        print(f"Warning: ML model warmup skipped ({exc}). Will load on demand.")
    yield


app = FastAPI(
    title="Prato do Dia ML Microservice",
    description="Single-stage YOLOv11-seg food segmentation microservice.",
    version="1.0.0",
    lifespan=lifespan,
)



class HealthStatusResponse(TypedDict):
    status: str
    service: str
    version: str
    model_loaded: bool
    model_name: str


@app.get("/health")
async def health_check() -> HealthStatusResponse:
    """Health check endpoint. Non-blocking during ongoing inferences."""
    try:
        predictor = get_server_predictor()
        is_ready = predictor.session is not None
        model_name = predictor.model_path.name
    except Exception:
        is_ready = False
        model_name = "unavailable"

    return {
        "status": "ok" if is_ready else "degraded",
        "service": "prato-do-dia-ml",
        "version": "1.0.0",
        "model_loaded": is_ready,
        "model_name": model_name,
    }


@app.post("/v1/predict")
async def predict_meal(file: UploadFile = File(...)) -> JSONResponse:  # noqa: B008
    """Predict food items and segmentation polygons from an uploaded meal image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid JPEG or PNG image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image payload.")

    try:
        predictor = get_server_predictor()
        # Dispatch CPU inference to threadpool via asyncio.to_thread to keep Event Loop responsive
        response = await asyncio.to_thread(predictor.predict_bytes, image_bytes)
        return JSONResponse(content=response.to_dict())
    except MLInvalidImageError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {exc}") from exc
    except MLModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"ML model unavailable: {exc}") from exc
    except MLInferenceError as exc:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc
