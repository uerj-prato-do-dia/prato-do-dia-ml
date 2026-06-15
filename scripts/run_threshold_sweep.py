"""Run YOLO confidence/NMS threshold sweep over the baseline_eval split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prato_do_dia_ml.config import PipelineConfig, load_config
from prato_do_dia_ml.pipeline import FoodSegmentationPipeline
from scripts.run_baseline_report import evaluate_image, failed_metric_row, selected_manifest_rows

SWEEP_COLUMNS = [
    "config_id",
    "yolo_conf",
    "nms_iou",
    "image_count",
    "succeeded",
    "failed",
    "foreground_iou",
    "instance_iou",
    "dice",
    "boundary_f_score",
    "precision",
    "recall",
    "false_positives",
    "missed_instances",
]

DEFAULT_CONF_VALUES = [0.05, 0.08, 0.10, 0.15, 0.20]
DEFAULT_NMS_VALUES = [0.35, 0.45, 0.55]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/threshold_sweep_v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/yolo11_sam2_baseline.toml"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conf-values", default=",".join(str(value) for value in DEFAULT_CONF_VALUES))
    parser.add_argument("--nms-values", default=",".join(str(value) for value in DEFAULT_NMS_VALUES))
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    rows = selected_manifest_rows(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No baseline_eval rows with existing masks were found.")

    config = load_config(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = sweep_configs(parse_float_list(args.conf_values), parse_float_list(args.nms_values))
    results = [
        run_config(config, rows, output_dir, config_id, yolo_conf, nms_iou, fail_fast=args.fail_fast)
        for config_id, yolo_conf, nms_iou in configs
    ]
    write_sweep_results(output_dir / "sweep_results.csv", results)
    write_best(output_dir, results)
    print(f"Threshold sweep: {output_dir}")
    print(f"Configs: {len(results)}")
    print(f"Images per config: {len(rows)}")
    print(f"Best foreground_iou: {best_result(results, 'foreground_iou')['config_id']}")
    print(f"Best instance_iou: {best_result(results, 'instance_iou')['config_id']}")
    print(f"Best recall: {best_result(results, 'recall')['config_id']}")


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one float value is required")
    return values


def sweep_configs(conf_values: list[float], nms_values: list[float]) -> list[tuple[str, float, float]]:
    configs = []
    for yolo_conf in conf_values:
        for nms_iou in nms_values:
            config_id = f"conf_{yolo_conf:.2f}_nms_{nms_iou:.2f}".replace(".", "p")
            configs.append((config_id, yolo_conf, nms_iou))
    return configs


def run_config(
    base_config: PipelineConfig,
    manifest_rows: list[dict[str, str]],
    output_dir: Path,
    config_id: str,
    yolo_conf: float,
    nms_iou: float,
    *,
    fail_fast: bool,
) -> dict[str, str]:
    config_output_dir = output_dir / config_id
    config_output_dir.mkdir(parents=True, exist_ok=True)
    config = config_for_sweep(base_config, config_output_dir, yolo_conf, nms_iou)
    write_config_snapshot(config_output_dir / "config_snapshot.json", config_id, yolo_conf, nms_iou)
    pipeline = FoodSegmentationPipeline.from_config(config)

    image_rows = []
    for row in manifest_rows:
        try:
            image_rows.append(evaluate_image(row, pipeline))
        except Exception as exc:
            if fail_fast:
                raise
            image_rows.append(failed_metric_row(row, exc))
    return aggregate_config_result(config_id, yolo_conf, nms_iou, image_rows)


def config_for_sweep(
    config: PipelineConfig,
    output_dir: Path,
    yolo_conf: float,
    nms_iou: float,
) -> PipelineConfig:
    paths = replace(
        config.paths,
        output_dir=output_dir / "raw_segmentations",
        mask_dir=output_dir / "masks",
        overlay_dir=output_dir / "overlays",
        report_dir=output_dir / "reports",
    )
    yolo = replace(config.yolo, confidence_threshold=yolo_conf, nms_iou_threshold=nms_iou)
    return replace(config, paths=paths, yolo=yolo)


def aggregate_config_result(
    config_id: str,
    yolo_conf: float,
    nms_iou: float,
    image_rows: list[dict[str, str]],
) -> dict[str, str]:
    successful = [row for row in image_rows if row["status"] == "ok"]
    return {
        "config_id": config_id,
        "yolo_conf": f"{yolo_conf:.4f}",
        "nms_iou": f"{nms_iou:.4f}",
        "image_count": str(len(image_rows)),
        "succeeded": str(len(successful)),
        "failed": str(len(image_rows) - len(successful)),
        "foreground_iou": _mean(successful, "foreground_iou"),
        "instance_iou": _mean(successful, "instance_iou"),
        "dice": _mean(successful, "dice"),
        "boundary_f_score": _mean(successful, "boundary_f_score"),
        "precision": _mean(successful, "precision"),
        "recall": _mean(successful, "recall"),
        "false_positives": str(sum(int(row["false_positives"]) for row in successful if row["false_positives"])),
        "missed_instances": str(sum(int(row["missed_instances"]) for row in successful if row["missed_instances"])),
    }


def write_sweep_results(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SWEEP_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_best(output_dir: Path, rows: list[dict[str, str]]) -> None:
    for metric, filename in [
        ("foreground_iou", "best_by_foreground_iou.json"),
        ("instance_iou", "best_by_instance_iou.json"),
        ("recall", "best_by_recall.json"),
    ]:
        best = best_result(rows, metric)
        (output_dir / filename).write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")


def best_result(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
    return max(rows, key=lambda row: float(row[metric] or 0.0))


def write_config_snapshot(path: Path, config_id: str, yolo_conf: float, nms_iou: float) -> None:
    path.write_text(
        json.dumps(
            {
                "config_id": config_id,
                "yolo_conf": yolo_conf,
                "nms_iou": nms_iou,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _mean(rows: list[dict[str, str]], metric: str) -> str:
    values = [float(row[metric]) for row in rows if row[metric]]
    if not values:
        return ""
    return f"{sum(values) / len(values):.6f}"


if __name__ == "__main__":
    main()
