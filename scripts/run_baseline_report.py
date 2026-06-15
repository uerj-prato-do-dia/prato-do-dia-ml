"""Run a reproducible baseline evaluation from the dataset manifest."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prato_do_dia_ml.config import PipelineConfig, load_config
from prato_do_dia_ml.io_utils import validate_instance_mask_png
from prato_do_dia_ml.metrics import (
    boundary_f_score,
    dice_score,
    evaluate_instance_masks,
    intersection_over_union,
)
from prato_do_dia_ml.pipeline import FoodSegmentationPipeline

METRIC_COLUMNS = [
    "image_id",
    "image_path",
    "mask_path",
    "foreground_iou",
    "instance_iou",
    "dice",
    "boundary_f_score",
    "precision",
    "recall",
    "false_positives",
    "missed_instances",
    "area_error",
    "component_count",
    "status",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/baseline_v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/yolo11_sam2_baseline.toml"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    rows = selected_manifest_rows(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No baseline_eval rows with existing masks were found.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _config_for_output(load_config(args.config), output_dir)
    copy_snapshots(args.config, output_dir)

    pipeline = FoodSegmentationPipeline.from_config(config)
    metric_rows: list[dict[str, str]] = []
    warnings: list[str] = []
    for row in rows:
        try:
            metric_rows.append(evaluate_image(row, pipeline))
        except Exception as exc:
            if args.fail_fast:
                raise
            warnings.append(f"{row['image_id']}: {type(exc).__name__}")
            metric_rows.append(failed_metric_row(row, exc))

    write_metrics_by_image(output_dir / "metrics_by_image.csv", metric_rows)
    write_metrics_json(
        output_dir / "metrics.json",
        args.config,
        metric_rows,
        warnings,
    )
    write_failure_notes(output_dir / "failure_notes.md", metric_rows)
    print(f"Baseline report: {output_dir}")
    print(f"Images: {len(metric_rows)}")
    print(f"Succeeded: {sum(row['status'] == 'ok' for row in metric_rows)}")
    print(f"Failed: {sum(row['status'] == 'failed' for row in metric_rows)}")


def selected_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    selected = []
    for row in rows:
        mask_path = PROJECT_ROOT / row.get("mask_path", "")
        if row.get("split") == "baseline_eval" and row.get("mask_path") and mask_path.exists():
            selected.append(row)
    return sorted(selected, key=lambda row: row["image_id"])


def evaluate_image(row: dict[str, str], pipeline: FoodSegmentationPipeline) -> dict[str, str]:
    image_path = PROJECT_ROOT / row["image_path"]
    mask_path = PROJECT_ROOT / row["mask_path"]
    result = pipeline.run_image(image_path)

    ground_truth = validate_instance_mask_png(mask_path)
    predicted_instances = cv2.imread(str(result.instance_mask_path), cv2.IMREAD_UNCHANGED)
    if predicted_instances is None:
        raise FileNotFoundError(f"could not load predicted mask: {result.instance_mask_path}")
    if predicted_instances.shape != ground_truth.shape:
        raise ValueError(
            f"predicted shape {predicted_instances.shape} does not match ground truth shape {ground_truth.shape}"
        )

    predicted_instances = predicted_instances.astype(np.int32)
    pred_foreground = predicted_instances != 0
    gt_foreground = ground_truth != 0
    instance_metrics = evaluate_instance_masks(predicted_instances, ground_truth)
    return {
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "mask_path": row["mask_path"],
        "foreground_iou": _fmt(intersection_over_union(pred_foreground, gt_foreground)),
        "instance_iou": _fmt(float(instance_metrics["miou"])),
        "dice": _fmt(dice_score(pred_foreground, gt_foreground)),
        "boundary_f_score": _fmt(boundary_f_score(pred_foreground, gt_foreground)),
        "precision": _fmt(float(instance_metrics["precision"])),
        "recall": _fmt(float(instance_metrics["recall"])),
        "false_positives": str(instance_metrics["false_positives"]),
        "missed_instances": str(instance_metrics["missed_instances"]),
        "area_error": _fmt(float(instance_metrics["mean_area_error"])),
        "component_count": str(len(result.segmentations)),
        "status": "ok",
        "notes": "",
    }


def failed_metric_row(row: dict[str, str], exc: Exception) -> dict[str, str]:
    return {
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "mask_path": row["mask_path"],
        "foreground_iou": "",
        "instance_iou": "",
        "dice": "",
        "boundary_f_score": "",
        "precision": "",
        "recall": "",
        "false_positives": "",
        "missed_instances": "",
        "area_error": "",
        "component_count": "",
        "status": "failed",
        "notes": f"{type(exc).__name__}: {str(exc)[:160]}",
    }


def write_metrics_by_image(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_json(
    path: Path,
    config_path: Path,
    rows: list[dict[str, str]],
    warnings: list[str],
) -> None:
    successful = [row for row in rows if row["status"] == "ok"]
    metric_names = [
        "foreground_iou",
        "instance_iou",
        "dice",
        "boundary_f_score",
        "precision",
        "recall",
        "area_error",
    ]
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "image_count": len(rows),
        "successful_images": len(successful),
        "failed_images": len(rows) - len(successful),
        "mean_metrics": {name: _mean_metric(successful, name) for name in metric_names},
        "median_metrics": {name: _median_metric(successful, name) for name in metric_names},
        "model_info": model_manifest_summary(),
        "warnings": warnings,
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def write_failure_notes(path: Path, rows: list[dict[str, str]]) -> None:
    lines = ["# Baseline v1 failure notes", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['image_id']}",
                "",
                f"- Foreground IoU: {row['foreground_iou']}",
                f"- Instance IoU: {row['instance_iou']}",
                f"- False positives: {row['false_positives']}",
                f"- Missed instances: {row['missed_instances']}",
                "- Visual notes:",
                "  - TODO",
                "- Likely failure type:",
                "  - [ ] false_positive",
                "  - [ ] false_negative",
                "  - [ ] bad_box",
                "  - [ ] mask_leak",
                "  - [ ] merged_foods",
                "  - [ ] wrong_class",
                "  - [ ] poor_image_quality",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_snapshots(config_path: Path, output_dir: Path) -> None:
    shutil.copyfile(config_path, output_dir / "config_snapshot.toml")
    manifest_path = Path("models/model_manifest.json")
    destination = output_dir / "model_manifest_snapshot.json"
    if manifest_path.exists():
        shutil.copyfile(manifest_path, destination)
    else:
        destination.write_text('{"warning": "model_manifest_missing"}\n', encoding="utf-8")


def _config_for_output(config: PipelineConfig, output_dir: Path) -> PipelineConfig:
    paths = replace(
        config.paths,
        output_dir=output_dir / "raw_segmentations",
        mask_dir=output_dir / "masks",
        overlay_dir=output_dir / "overlays",
        report_dir=output_dir / "reports",
    )
    return replace(config, paths=paths)


def model_manifest_summary() -> dict[str, object]:
    path = Path("models/model_manifest.json")
    if not path.exists():
        return {"manifest_present": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "manifest_present": True,
        "schema_version": data.get("schema_version"),
        "models": data.get("models", []),
    }


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _mean_metric(rows: list[dict[str, str]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if row[name]]
    if not values:
        return None
    return float(sum(values) / len(values))


def _median_metric(rows: list[dict[str, str]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if row[name]]
    if not values:
        return None
    return float(median(values))


def _fmt(value: float) -> str:
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
