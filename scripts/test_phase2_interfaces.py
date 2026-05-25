"""Smoke test real Phase 2 ONNX sessions and one pipeline invocation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import YoloOnnxDetector
from src.pipeline import FoodSegmentationPipeline
from src.segmenter import SamOnnxSegmenter


def main() -> None:
    image_path = _first_input_image()
    detector = YoloOnnxDetector(
        "models/yolov11_food.onnx",
        confidence_threshold=0.05,
        max_detections=3,
    )
    segmenter = SamOnnxSegmenter(
        "models/sam2.1_hiera_tiny.encoder.onnx",
        "models/sam2.1_hiera_tiny.decoder.onnx",
    )
    pipeline = FoodSegmentationPipeline(detector, segmenter)
    result = pipeline.run_image(image_path)

    if not result.annotation_path.exists():
        raise AssertionError(f"annotation file was not written: {result.annotation_path}")

    print(
        f"phase 2 inference smoke test passed "
        f"image={image_path.name} detections={len(result.detections)} "
        f"segmentations={len(result.segmentations)} output={result.annotation_path}"
    )


def _first_input_image() -> Path:
    input_dir = Path("data/input")
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        matches = sorted(input_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError("no test image found in data/input")


if __name__ == "__main__":
    main()
