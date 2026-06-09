from __future__ import annotations

from pathlib import Path

import pytest

from prato_do_dia_ml.detector import YoloOnnxDetector
from prato_do_dia_ml.pipeline import FoodSegmentationPipeline
from prato_do_dia_ml.segmenter import SamOnnxSegmenter


@pytest.mark.onnx
def test_real_onnx_pipeline_writes_outputs(tmp_path) -> None:
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
    pipeline = FoodSegmentationPipeline(
        detector,
        segmenter,
        output_dir=tmp_path / "raw_segmentations",
        mask_dir=tmp_path / "masks",
        overlay_dir=tmp_path / "overlays",
        report_dir=tmp_path / "reports",
    )

    result = pipeline.run_image(image_path)

    assert result.annotation_path.exists()
    assert result.instance_mask_path.exists()
    assert result.class_mask_path.exists()
    assert result.metadata_path.exists()
    assert result.overlay_path.exists()


def _first_input_image() -> Path:
    input_dir = Path("data/input")
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        matches = sorted(input_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError("no test image found in data/input")
