"""Experiment orchestration for reproducible segmentation evaluation."""

from __future__ import annotations

import csv
import json
import resource
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2

from prato_do_dia_ml.config import PipelineConfig
from prato_do_dia_ml.experiment_visuals import render_debug_overlay
from prato_do_dia_ml.io_utils import input_images, validate_instance_mask_png
from prato_do_dia_ml.metrics import evaluate_instance_masks, intersection_over_union
from prato_do_dia_ml.pipeline import FoodSegmentationPipeline
from prato_do_dia_ml.reproducibility import config_to_dict, set_random_seed, write_environment, write_model_versions, write_yaml


def run_experiment(
    config: PipelineConfig,
    experiment_name: str,
    outputs_dir: str | Path = "outputs/experiments",
    limit: int | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    """Run inference/evaluation and write a reproducible experiment directory."""

    set_random_seed(seed)
    experiment_dir = Path(outputs_dir) / experiment_name
    if experiment_dir.exists() and not overwrite:
        raise FileExistsError(f"experiment already exists: {experiment_dir}. Use --overwrite to replace outputs.")

    _prepare_experiment_dir(experiment_dir)
    effective_config = _config_for_experiment(config, experiment_dir)
    image_paths = _labeled_image_paths(effective_config.paths.input_dir, effective_config.paths.ground_truth_dir)
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError("no labeled images found for experiment")

    config_payload = config_to_dict(effective_config)
    config_payload["experiment"] = {"name": experiment_name, "seed": seed, "limit": limit}
    config_payload["dataset"] = {
        "images_path": str(effective_config.paths.input_dir),
        "ground_truth_path": str(effective_config.paths.ground_truth_dir),
        "labeled_image_count": len(image_paths),
        "ground_truth_format": "single-channel PNG instance IDs",
        "label_studio_source": "canonical masks imported from Label Studio",
        "label_studio_class_map": str(effective_config.paths.ground_truth_dir / "class_map.json"),
    }
    write_yaml(config_payload, experiment_dir / "config.yaml")
    write_environment(experiment_dir / "environment.txt")
    write_model_versions(effective_config, experiment_dir / "model_versions.json")

    pipeline = FoodSegmentationPipeline.from_config(effective_config)
    rows: list[dict[str, Any]] = []
    for image_path in image_paths:
        gt_path = effective_config.paths.ground_truth_dir / f"{image_path.stem}_instances.png"
        gt_mask = validate_instance_mask_png(gt_path)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"could not load image: {image_path}")
        if image_bgr.shape[:2] != gt_mask.shape:
            raise ValueError(f"shape mismatch for {image_path}: image={image_bgr.shape[:2]} gt={gt_mask.shape}")

        start = time.perf_counter()
        result = pipeline.run_image(image_path)
        runtime_seconds = time.perf_counter() - start

        prediction = cv2.imread(str(result.instance_mask_path), cv2.IMREAD_UNCHANGED)
        if prediction is None:
            raise FileNotFoundError(f"could not load prediction mask: {result.instance_mask_path}")

        metrics = evaluate_instance_masks(prediction.astype("int32"), gt_mask)
        foreground_iou = intersection_over_union(prediction != 0, gt_mask != 0)
        debug_overlay_path = effective_config.paths.overlay_dir / f"{image_path.stem}_debug.jpg"
        render_debug_overlay(image_bgr, gt_mask, prediction, result.detections, debug_overlay_path)

        rows.append(
            {
                "image": str(image_path),
                "ground_truth": str(gt_path),
                "prediction_mask": str(result.instance_mask_path),
                "debug_overlay": str(debug_overlay_path),
                "metadata": str(result.metadata_path),
                "detections": len(result.detections),
                "segmentations": len(result.segmentations),
                "runtime_seconds": runtime_seconds,
                "peak_rss_mb": _peak_rss_mb(),
                "foreground_iou": foreground_iou,
                **metrics,
            }
        )

    report = _build_report(rows, config_payload)
    (experiment_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_metrics_csv(rows, experiment_dir / "metrics.csv")
    _write_summary(report, experiment_dir / "reports" / "summary.md")
    _copy_ranked_overlays(rows, experiment_dir / "reports")
    return experiment_dir


def compare_experiments(experiments_dir: str | Path, output_csv: str | Path | None = None) -> list[dict[str, Any]]:
    """Load experiment metrics and return a benchmark table."""

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(Path(experiments_dir).glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        aggregate = metrics["aggregate"]
        rows.append(
            {
                "experiment": metrics_path.parent.name,
                "image_count": aggregate["image_count"],
                "mean_runtime_seconds": aggregate["mean_runtime_seconds"],
                "peak_rss_mb": aggregate["peak_rss_mb"],
                "mean_iou": aggregate["mean_instance_iou"],
                "mean_dice": aggregate["mean_dice"],
                "false_positives": aggregate["false_positives"],
                "missed_regions": aggregate["missed_regions"],
                "qualitative_overlay_quality": "manual_review_required",
            }
        )

    if output_csv is not None:
        _write_dict_rows(rows, Path(output_csv))
    return rows


def _config_for_experiment(config: PipelineConfig, experiment_dir: Path) -> PipelineConfig:
    paths = replace(
        config.paths,
        output_dir=experiment_dir / "predictions" / "annotations",
        mask_dir=experiment_dir / "predictions" / "masks",
        overlay_dir=experiment_dir / "predictions" / "overlays",
        report_dir=experiment_dir / "predictions" / "metadata",
    )
    return replace(config, paths=paths)


def _prepare_experiment_dir(experiment_dir: Path) -> None:
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    for path in (
        experiment_dir / "predictions" / "masks",
        experiment_dir / "predictions" / "overlays",
        experiment_dir / "predictions" / "metadata",
        experiment_dir / "predictions" / "annotations",
        experiment_dir / "reports" / "best_cases",
        experiment_dir / "reports" / "worst_cases",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _labeled_image_paths(input_dir: Path, ground_truth_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(input_images(input_dir), key=_natural_key)
        if (ground_truth_dir / f"{path.stem}_instances.png").exists()
    ]


def _natural_key(path: Path) -> tuple[str, int]:
    digits = "".join(char for char in path.stem if char.isdigit())
    prefix = path.stem.rstrip("0123456789")
    return prefix, int(digits) if digits else -1


def _build_report(rows: list[dict[str, Any]], config_payload: dict[str, Any]) -> dict[str, Any]:
    aggregate = {
        "image_count": len(rows),
        "mean_runtime_seconds": _mean(row["runtime_seconds"] for row in rows),
        "peak_rss_mb": max((float(row["peak_rss_mb"]) for row in rows), default=0.0),
        "mean_foreground_iou": _mean(row["foreground_iou"] for row in rows),
        "mean_instance_iou": _mean(row["miou"] for row in rows),
        "mean_dice": _mean(row["mean_dice"] for row in rows),
        "mean_boundary_f": _mean(row["mean_boundary_f"] for row in rows),
        "mean_precision": _mean(row["precision"] for row in rows),
        "mean_recall": _mean(row["recall"] for row in rows),
        "false_positives": int(sum(row["false_positives"] for row in rows)),
        "missed_regions": int(sum(row["missed_instances"] for row in rows)),
        "mean_area_error": _mean(row["mean_area_error"] for row in rows),
    }
    return {"config": config_payload, "aggregate": aggregate, "images": rows}


def _write_metrics_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    scalar_rows = []
    for row in rows:
        scalar_rows.append({key: value for key, value in row.items() if key != "matches"})
    _write_dict_rows(scalar_rows, output_path)


def _write_dict_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(report: dict[str, Any], output_path: Path) -> None:
    aggregate = report["aggregate"]
    lines = [
        "# Experiment Summary",
        "",
        "## Aggregate Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in aggregate.items():
        lines.append(f"| {key} | {_format_metric(value)} |")

    sorted_rows = sorted(report["images"], key=lambda row: row["miou"])
    lines.extend(
        [
            "",
            "## Worst Cases",
            "",
            "| image | instance IoU | dice | missed | fp |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted_rows[:5]:
        lines.append(
            f"| {Path(row['image']).name} | {_format_metric(row['miou'])} | "
            f"{_format_metric(row['mean_dice'])} | {row['missed_instances']} | {row['false_positives']} |"
        )
    lines.extend(
        [
            "",
            "## Best Cases",
            "",
            "| image | instance IoU | dice | missed | fp |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in reversed(sorted_rows[-5:]):
        lines.append(
            f"| {Path(row['image']).name} | {_format_metric(row['miou'])} | "
            f"{_format_metric(row['mean_dice'])} | {row['missed_instances']} | {row['false_positives']} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_ranked_overlays(rows: list[dict[str, Any]], reports_dir: Path) -> None:
    sorted_rows = sorted(rows, key=lambda row: row["miou"])
    for rank, row in enumerate(sorted_rows[:5], start=1):
        _copy_overlay(row, reports_dir / "worst_cases", rank)
    for rank, row in enumerate(reversed(sorted_rows[-5:]), start=1):
        _copy_overlay(row, reports_dir / "best_cases", rank)


def _copy_overlay(row: dict[str, Any], output_dir: Path, rank: int) -> None:
    source = Path(row["debug_overlay"])
    if not source.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_dir / f"{rank:02d}_{Path(row['image']).stem}_iou_{row['miou']:.3f}.jpg")


def _mean(values: Any) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports kilobytes; macOS reports bytes. This repo is developed on Linux.
    rss = float(usage.ru_maxrss)
    if rss > 10_000_000:
        return rss / (1024 * 1024)
    return rss / 1024
