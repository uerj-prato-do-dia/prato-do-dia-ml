"""Run zero-shot Ultralytics segmentation models on the baseline_eval split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prato_do_dia_ml.io_utils import validate_instance_mask_png
from prato_do_dia_ml.metrics import (
    boundary_f_score,
    dice_score,
    evaluate_instance_masks,
    intersection_over_union,
)
from scripts.run_baseline_report import selected_manifest_rows
from scripts.run_threshold_sweep import parse_float_list

FOOD_PROMPTS = [
    "rice",
    "beans",
    "meat",
    "chicken",
    "salad",
    "tomato",
    "pasta",
    "potato",
    "egg",
    "french fries",
    "vegetables",
    "beef",
    "pork",
    "fish",
]

YOLO11_MODELS = [
    ("yolo11m_seg", "yolo11m-seg.pt", "closed_vocab"),
    ("yolo11l_seg", "yolo11l-seg.pt", "closed_vocab"),
    ("yolo11x_seg", "yolo11x-seg.pt", "closed_vocab"),
]

YOLOE_MODELS = [
    ("yoloe26s_seg_food", "yoloe-26s-seg.pt", "open_vocab_food_prompts"),
    ("yoloe26m_seg_food", "yoloe-26m-seg.pt", "open_vocab_food_prompts"),
]

SUMMARY_COLUMNS = [
    "config_id",
    "model_name",
    "model_file",
    "model_type",
    "conf",
    "iou",
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

IMAGE_METRIC_COLUMNS = [
    "config_id",
    "model_name",
    "model_type",
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
    "component_count",
    "predicted_classes",
    "status",
    "notes",
]

FAILURE_COLUMNS = [
    "config_id",
    "model_name",
    "image_id",
    "stage",
    "error_type",
    "message",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    model_type: str
    prompts: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunSpec:
    config_id: str
    model: ModelSpec
    conf: float
    iou: float


def main() -> None:
    args = parse_args()
    rows = selected_manifest_rows(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No baseline_eval rows with existing masks were found.")

    if args.config is not None:
        config_model, config_conf, config_iou = load_experiment_config(args.config)
        model_specs = [config_model]
        conf_values = [config_conf]
        iou_values = [config_iou]
    else:
        model_specs = selected_model_specs(args)
        conf_values = parse_float_list(args.conf_values)
        iou_values = parse_float_list(args.iou_values)
    if not model_specs and not args.include_mobilesam_smoke:
        raise SystemExit("No models selected. Use --include-yolo11, --include-yoloe, or --include-mobilesam-smoke.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_specs = make_run_specs(
        model_specs,
        conf_values,
        iou_values,
    )
    all_metric_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    failure_rows: list[dict[str, str]] = []

    for run_spec in run_specs:
        summary, image_rows, failures = run_model_config(
            run_spec,
            rows,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            device=args.device,
        )
        summary_rows.append(summary)
        all_metric_rows.extend(image_rows)
        failure_rows.extend(failures)

    mobile_sam_result = None
    if args.include_mobilesam_smoke:
        mobile_sam_result = run_mobilesam_smoke(
            args.models_dir / "mobile_sam.pt",
            rows[0],
            args.output_dir / "mobile_sam_smoke",
            args.device,
        )
        if mobile_sam_result.get("status") != "ok":
            failure_rows.append(
                {
                    "config_id": "mobile_sam_smoke",
                    "model_name": "mobile_sam",
                    "image_id": rows[0]["image_id"],
                    "stage": "mobile_sam_smoke",
                    "error_type": str(mobile_sam_result.get("error_type", "")),
                    "message": str(mobile_sam_result.get("message", ""))[:240],
                }
            )

    write_csv(args.output_dir / "summary.csv", SUMMARY_COLUMNS, summary_rows)
    write_csv(args.output_dir / "metrics_by_image.csv", IMAGE_METRIC_COLUMNS, all_metric_rows)
    write_csv(args.output_dir / "failures.csv", FAILURE_COLUMNS, failure_rows)
    write_summary_json(args.output_dir / "summary.json", args, rows, summary_rows, failure_rows, mobile_sam_result)

    print(f"Ultralytics zero-shot output: {args.output_dir}")
    print(f"Images: {len(rows)}")
    print(f"Model/config runs: {len(summary_rows)}")
    print(f"Failures: {len(failure_rows)}")
    if summary_rows:
        print(f"Best foreground_iou: {best_result(summary_rows, 'foreground_iou')['config_id']}")
        print(f"Best instance_iou: {best_result(summary_rows, 'instance_iou')['config_id']}")
        print(f"Best recall: {best_result(summary_rows, 'recall')['config_id']}")
        balanced = best_with_min_precision(summary_rows, 0.75)
        if balanced is not None:
            print(f"Best recall with precision >= 0.75: {balanced['config_id']}")
        else:
            print("Best recall with precision >= 0.75: none")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_manifest.csv"))
    parser.add_argument("--models-dir", type=Path, default=Path("external_models/ultralytics"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/ultralytics_zero_shot_v1"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--conf-values", default="0.01,0.03,0.05")
    parser.add_argument("--iou-values", default="0.30,0.45,0.60")
    parser.add_argument("--include-yolo11", action="store_true")
    parser.add_argument("--include-yoloe", action="store_true")
    parser.add_argument("--include-mobilesam-smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def load_experiment_config(config_path: Path) -> tuple[ModelSpec, float, float]:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    ultralytics = data.get("ultralytics")
    if not isinstance(ultralytics, dict):
        raise ValueError("Ultralytics experiment config must include an [ultralytics] table.")
    prompts = data.get("prompts", {})
    prompt_classes = prompts.get("classes", []) if isinstance(prompts, dict) else []
    if not isinstance(prompt_classes, list):
        raise ValueError("prompts.classes must be a list.")

    model_path = required_string(ultralytics, "model_path")
    model_name = str(ultralytics.get("model_name") or Path(model_path).stem.replace("-", "_"))
    model_type = str(ultralytics.get("model_type") or "open_vocab_food_prompts")
    confidence = required_float(ultralytics, "confidence_threshold")
    nms_iou = required_float(ultralytics, "nms_iou_threshold")
    return (
        ModelSpec(model_name, model_path, model_type, tuple(str(item) for item in prompt_classes)),
        confidence,
        nms_iou,
    )


def required_string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"ultralytics.{key} must be a non-empty string.")
    return value


def required_float(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"ultralytics.{key} must be a number.")
    return float(value)


def selected_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    include_yolo11 = bool(args.include_yolo11)
    include_yoloe = bool(args.include_yoloe)
    if not include_yolo11 and not include_yoloe and not args.include_mobilesam_smoke:
        include_yolo11 = True
        include_yoloe = True

    specs: list[ModelSpec] = []
    if include_yolo11:
        specs.extend(ModelSpec(name, filename, model_type) for name, filename, model_type in YOLO11_MODELS)
    if include_yoloe:
        specs.extend(
            ModelSpec(name, filename, model_type, tuple(FOOD_PROMPTS)) for name, filename, model_type in YOLOE_MODELS
        )
    return specs


def make_run_specs(model_specs: list[ModelSpec], conf_values: list[float], iou_values: list[float]) -> list[RunSpec]:
    specs = []
    for model in model_specs:
        for conf in conf_values:
            for iou in iou_values:
                specs.append(RunSpec(config_id=config_id(model.name, conf, iou), model=model, conf=conf, iou=iou))
    return specs


def config_id(model_name: str, conf: float, iou: float) -> str:
    return f"{model_name}_conf_{conf:.2f}_iou_{iou:.2f}".replace(".", "p")


def run_model_config(
    run_spec: RunSpec,
    manifest_rows: list[dict[str, str]],
    *,
    models_dir: Path,
    output_dir: Path,
    device: str,
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    run_dir = output_dir / "runs" / run_spec.config_id
    predictions_dir = run_dir / "predictions"
    overlays_dir = run_dir / "overlays"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(run_dir / "run_config.json", run_spec)

    image_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    try:
        model = load_ultralytics_model(resolve_model_path(models_dir, run_spec.model.filename), run_spec.model.prompts)
    except Exception as exc:
        failures.append(failure_row(run_spec, "", "model_load", exc))
        for row in manifest_rows:
            image_rows.append(failed_image_row(run_spec, row, exc))
        summary = aggregate_run(run_spec, image_rows)
        write_csv(run_dir / "metrics_by_image.csv", IMAGE_METRIC_COLUMNS, image_rows)
        return summary, image_rows, failures

    for row in manifest_rows:
        try:
            image_rows.append(evaluate_image(run_spec, model, row, predictions_dir, overlays_dir, device))
        except Exception as exc:
            failures.append(failure_row(run_spec, row["image_id"], "image_inference", exc))
            image_rows.append(failed_image_row(run_spec, row, exc))

    write_csv(run_dir / "metrics_by_image.csv", IMAGE_METRIC_COLUMNS, image_rows)
    return aggregate_run(run_spec, image_rows), image_rows, failures


def load_ultralytics_model(model_path: Path, prompts: tuple[str, ...]) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Install it with: uv pip install -U ultralytics") from exc

    model = YOLO(str(model_path))
    if prompts:
        set_yoloe_prompts(model, list(prompts))
    return model


def resolve_model_path(models_dir: Path, filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    if models_dir != Path("."):
        return models_dir / path.name
    if path.parent != Path("."):
        return path
    return models_dir / path


def set_yoloe_prompts(model: Any, prompts: list[str]) -> None:
    if not hasattr(model, "set_classes"):
        raise RuntimeError("YOLOE model does not expose set_classes; update ultralytics before running YOLOE.")
    if hasattr(model, "get_text_pe"):
        model.set_classes(prompts, model.get_text_pe(prompts))
    else:
        model.set_classes(prompts)


def evaluate_image(
    run_spec: RunSpec,
    model: Any,
    row: dict[str, str],
    predictions_dir: Path,
    overlays_dir: Path,
    device: str,
) -> dict[str, str]:
    image_path = PROJECT_ROOT / row["image_path"]
    mask_path = PROJECT_ROOT / row["mask_path"]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not load image: {image_path}")

    results = model.predict(
        source=str(image_path),
        conf=run_spec.conf,
        iou=run_spec.iou,
        device=device,
        verbose=False,
    )
    result = results[0]
    predicted_instances, predictions = result_to_instance_mask(result, image.shape[:2])
    prediction_path = predictions_dir / f"{row['image_id']}_predictions.json"
    mask_output_path = predictions_dir / f"{row['image_id']}_instances.png"
    prediction_path.write_text(json.dumps(predictions, indent=2, sort_keys=True), encoding="utf-8")
    cv2.imwrite(str(mask_output_path), predicted_instances.astype(np.uint16))
    write_overlay(image, predicted_instances, overlays_dir / f"{row['image_id']}_overlay.jpg")

    ground_truth = validate_instance_mask_png(mask_path)
    if predicted_instances.shape != ground_truth.shape:
        raise ValueError(
            f"predicted shape {predicted_instances.shape} does not match ground truth shape {ground_truth.shape}"
        )

    pred_foreground = predicted_instances != 0
    gt_foreground = ground_truth != 0
    instance_metrics = evaluate_instance_masks(predicted_instances.astype(np.int32), ground_truth)
    predicted_classes = sorted({str(item["class_name"]) for item in predictions["instances"]})
    return {
        "config_id": run_spec.config_id,
        "model_name": run_spec.model.name,
        "model_type": run_spec.model.model_type,
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "mask_path": row["mask_path"],
        "foreground_iou": fmt(intersection_over_union(pred_foreground, gt_foreground)),
        "instance_iou": fmt(float(instance_metrics["miou"])),
        "dice": fmt(dice_score(pred_foreground, gt_foreground)),
        "boundary_f_score": fmt(boundary_f_score(pred_foreground, gt_foreground)),
        "precision": fmt(float(instance_metrics["precision"])),
        "recall": fmt(float(instance_metrics["recall"])),
        "false_positives": str(instance_metrics["false_positives"]),
        "missed_instances": str(instance_metrics["missed_instances"]),
        "component_count": str(len(predictions["instances"])),
        "predicted_classes": ";".join(predicted_classes),
        "status": "ok",
        "notes": "",
    }


def result_to_instance_mask(result: Any, image_shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image_shape
    instance_mask = np.zeros((height, width), dtype=np.uint16)
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    polygons = [] if masks is None or masks.xy is None else list(masks.xy)
    class_ids = [] if boxes is None or boxes.cls is None else [int(value) for value in boxes.cls.cpu().numpy().tolist()]
    confidences = (
        [] if boxes is None or boxes.conf is None else [float(value) for value in boxes.conf.cpu().numpy().tolist()]
    )

    instances: list[dict[str, Any]] = []
    for index, polygon in enumerate(polygons, start=1):
        points = np.asarray(polygon, dtype=np.float32)
        if points.size == 0:
            continue
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        pixel_points = np.round(points).astype(np.int32)
        if len(pixel_points) < 3:
            continue
        cv2.fillPoly(instance_mask, [pixel_points], index)
        class_id = class_ids[index - 1] if index - 1 < len(class_ids) else -1
        confidence = confidences[index - 1] if index - 1 < len(confidences) else None
        normalized_polygon = [
            [
                float(np.clip(x / max(width - 1, 1), 0.0, 1.0)),
                float(np.clip(y / max(height - 1, 1), 0.0, 1.0)),
            ]
            for x, y in pixel_points
        ]
        instances.append(
            {
                "instance_id": index,
                "class_id": class_id,
                "class_name": str(names.get(class_id, class_id)),
                "confidence": confidence,
                "polygon": normalized_polygon,
                "point_count": int(len(pixel_points)),
            }
        )
    return instance_mask, {"instances": instances}


def write_overlay(image_bgr: np.ndarray, instance_mask: np.ndarray, path: Path) -> None:
    overlay = image_bgr.copy()
    colors = [
        (0, 255, 255),
        (255, 0, 255),
        (0, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
        (128, 255, 0),
    ]
    for index, instance_id in enumerate([int(value) for value in np.unique(instance_mask) if value != 0]):
        color = colors[index % len(colors)]
        mask = instance_mask == instance_id
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array(color)).astype(np.uint8)
    cv2.imwrite(str(path), overlay)


def run_mobilesam_smoke(model_path: Path, row: dict[str, str], output_dir: Path, device: str) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"model file not found: {model_path}")
        from ultralytics import SAM

        model = SAM(str(model_path))
        results = model.predict(source=str(PROJECT_ROOT / row["image_path"]), device=device, verbose=False)
        result = {
            "status": "ok",
            "model_file": str(model_path),
            "image_id": row["image_id"],
            "result_count": len(results),
            "note": "MobileSAM smoke only; it is not directly comparable without prompts or detector boxes.",
        }
    except Exception as exc:  # pragma: no cover - exercised only in optional heavy smoke runs
        result = {
            "status": "failed",
            "model_file": str(model_path),
            "image_id": row["image_id"],
            "error_type": type(exc).__name__,
            "message": str(exc)[:240],
        }
    (output_dir / "smoke_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def aggregate_run(run_spec: RunSpec, rows: list[dict[str, str]]) -> dict[str, str]:
    successful = [row for row in rows if row["status"] == "ok"]
    return {
        "config_id": run_spec.config_id,
        "model_name": run_spec.model.name,
        "model_file": run_spec.model.filename,
        "model_type": run_spec.model.model_type,
        "conf": f"{run_spec.conf:.4f}",
        "iou": f"{run_spec.iou:.4f}",
        "image_count": str(len(rows)),
        "succeeded": str(len(successful)),
        "failed": str(len(rows) - len(successful)),
        "foreground_iou": mean_metric(successful, "foreground_iou"),
        "instance_iou": mean_metric(successful, "instance_iou"),
        "dice": mean_metric(successful, "dice"),
        "boundary_f_score": mean_metric(successful, "boundary_f_score"),
        "precision": mean_metric(successful, "precision"),
        "recall": mean_metric(successful, "recall"),
        "false_positives": str(sum(int(row["false_positives"]) for row in successful if row["false_positives"])),
        "missed_instances": str(sum(int(row["missed_instances"]) for row in successful if row["missed_instances"])),
    }


def best_result(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
    return max(rows, key=lambda row: float(row[metric] or 0.0))


def best_with_min_precision(rows: list[dict[str, str]], min_precision: float) -> dict[str, str] | None:
    candidates = [row for row in rows if float(row["precision"] or 0.0) >= min_precision]
    if not candidates:
        return None
    return best_result(candidates, "recall")


def write_summary_json(
    path: Path,
    args: argparse.Namespace,
    manifest_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    failure_rows: list[dict[str, str]],
    mobile_sam_result: dict[str, object] | None,
) -> None:
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "manifest": str(args.manifest),
        "config": str(args.config) if args.config is not None else None,
        "models_dir": str(args.models_dir),
        "device": args.device,
        "image_count": len(manifest_rows),
        "run_count": len(summary_rows),
        "failure_count": len(failure_rows),
        "food_prompts": FOOD_PROMPTS,
        "best_by_foreground_iou": best_result(summary_rows, "foreground_iou") if summary_rows else None,
        "best_by_instance_iou": best_result(summary_rows, "instance_iou") if summary_rows else None,
        "best_by_recall": best_result(summary_rows, "recall") if summary_rows else None,
        "best_recall_with_precision_gte_0p75": best_with_min_precision(summary_rows, 0.75) if summary_rows else None,
        "mobile_sam_smoke": mobile_sam_result,
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def write_run_config(path: Path, run_spec: RunSpec) -> None:
    path.write_text(
        json.dumps(
            {
                "config_id": run_spec.config_id,
                "model": {
                    "name": run_spec.model.name,
                    "filename": run_spec.model.filename,
                    "model_type": run_spec.model.model_type,
                    "prompts": list(run_spec.model.prompts),
                },
                "conf": run_spec.conf,
                "iou": run_spec.iou,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def failure_row(run_spec: RunSpec, image_id: str, stage: str, exc: Exception) -> dict[str, str]:
    return {
        "config_id": run_spec.config_id,
        "model_name": run_spec.model.name,
        "image_id": image_id,
        "stage": stage,
        "error_type": type(exc).__name__,
        "message": str(exc)[:240],
    }


def failed_image_row(run_spec: RunSpec, row: dict[str, str], exc: Exception) -> dict[str, str]:
    return {
        "config_id": run_spec.config_id,
        "model_name": run_spec.model.name,
        "model_type": run_spec.model.model_type,
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
        "component_count": "",
        "predicted_classes": "",
        "status": "failed",
        "notes": f"{type(exc).__name__}: {str(exc)[:160]}",
    }


def mean_metric(rows: list[dict[str, str]], metric: str) -> str:
    values = [float(row[metric]) for row in rows if row[metric]]
    if not values:
        return ""
    return fmt(sum(values) / len(values))


def fmt(value: float) -> str:
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
