"""Evaluate generated SAM 2 masks against deterministic ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.io_utils import input_images, validate_instance_mask_png
from src.metrics import (
    evaluate_instance_masks,
    intersection_over_union,
    rasterize_yolo_polygons,
)
from src.pipeline import FoodSegmentationPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--ground-truth-dir", type=Path)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--max-detections", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    config = _with_cli_overrides(config, args.input_dir, args.ground_truth_dir, args.confidence, args.max_detections)

    image_paths = input_images(config.paths.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"no input images found in {config.paths.input_dir}")

    ground_truth_masks = _load_ground_truth_masks(image_paths, config.paths.ground_truth_dir)

    pipeline = FoodSegmentationPipeline.from_config(config)

    per_image = []
    for image_path in image_paths:
        result = pipeline.run_image(image_path)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"could not reload image: {image_path}")

        gt_path, gt_instance_mask = ground_truth_masks[image_path.stem]
        if gt_instance_mask.shape != image_bgr.shape[:2]:
            raise ValueError(
                f"ground truth shape {gt_instance_mask.shape} does not match "
                f"image shape {image_bgr.shape[:2]} for {image_path.name}"
            )

        predicted_instances = cv2.imread(str(result.instance_mask_path), cv2.IMREAD_UNCHANGED)
        if predicted_instances is None:
            raise FileNotFoundError(f"could not load predicted mask: {result.instance_mask_path}")
        instance_metrics = evaluate_instance_masks(predicted_instances.astype(np.int32), gt_instance_mask)
        predicted_mask = rasterize_yolo_polygons(result.annotation_path, image_bgr.shape[:2])
        gt_foreground = gt_instance_mask != 0
        iou = intersection_over_union(predicted_mask, gt_foreground)
        per_image.append(
            {
                "image": str(image_path),
                "ground_truth": str(gt_path),
                "annotation": str(result.annotation_path),
                "instance_mask": str(result.instance_mask_path),
                "overlay": str(result.overlay_path),
                "detections": len(result.detections),
                "segmentations": len(result.segmentations),
                "foreground_iou": iou,
                **instance_metrics,
            }
        )

    report = {
        "image_count": len(per_image),
        "foreground_miou": sum(item["foreground_iou"] for item in per_image) / len(per_image),
        "instance_miou": sum(float(item["miou"]) for item in per_image) / len(per_image),
        "images": per_image,
    }
    report_path = config.paths.report_dir / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _ground_truth_path(ground_truth_dir: Path, image_stem: str) -> Path:
    path = ground_truth_dir / f"{image_stem}_instances.png"
    if path.exists():
        return path
    raise FileNotFoundError(f"missing ground truth for {image_stem} in {ground_truth_dir}")


def _load_ground_truth_masks(
    image_paths: list[Path],
    ground_truth_dir: Path,
) -> dict[str, tuple[Path, np.ndarray]]:
    masks = {}
    for image_path in image_paths:
        gt_path = _ground_truth_path(ground_truth_dir, image_path.stem)
        masks[image_path.stem] = (gt_path, validate_instance_mask_png(gt_path))
    return masks


def _with_cli_overrides(
    config,
    input_dir: Path | None,
    ground_truth_dir: Path | None,
    confidence: float | None,
    max_detections: int | None,
):
    from dataclasses import replace

    paths = replace(
        config.paths,
        input_dir=config.paths.input_dir if input_dir is None else input_dir,
        ground_truth_dir=config.paths.ground_truth_dir if ground_truth_dir is None else ground_truth_dir,
    )
    yolo = replace(
        config.yolo,
        confidence_threshold=config.yolo.confidence_threshold if confidence is None else confidence,
        max_detections=config.yolo.max_detections if max_detections is None else max_detections,
    )
    return replace(config, paths=paths, yolo=yolo)


if __name__ == "__main__":
    main()
