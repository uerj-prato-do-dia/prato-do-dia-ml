"""Run the Prato do Dia ONNX segmentation pipeline for one image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.pipeline import FoodSegmentationPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--max-detections", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.confidence is not None or args.max_detections is not None:
        config = _with_cli_overrides(config, args.confidence, args.max_detections)
    pipeline = FoodSegmentationPipeline.from_config(config)
    result = pipeline.run_image(args.image)

    print(
        f"saved {result.annotation_path} "
        f"masks={result.instance_mask_path} metadata={result.metadata_path} "
        f"detections={len(result.detections)} segmentations={len(result.segmentations)}"
    )


def _with_cli_overrides(config, confidence: float | None, max_detections: int | None):
    from dataclasses import replace

    yolo = replace(
        config.yolo,
        confidence_threshold=config.yolo.confidence_threshold if confidence is None else confidence,
        max_detections=config.yolo.max_detections if max_detections is None else max_detections,
    )
    return replace(config, yolo=yolo)


if __name__ == "__main__":
    main()
