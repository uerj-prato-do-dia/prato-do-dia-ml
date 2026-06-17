"""Run exploratory YOLOE pseudo-labeling over ingested unlabeled images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prato_do_dia_ml.io_utils import input_images, load_image_bgr
from scripts.run_ultralytics_zero_shot import (
    RunSpec,
    config_id,
    load_experiment_config,
    load_ultralytics_model,
    resolve_model_path,
    result_to_instance_mask,
    write_overlay,
)

SUMMARY_COLUMNS = [
    "image_id",
    "image_path",
    "prediction_count",
    "high_confidence_count",
    "predicted_classes",
    "high_confidence_classes",
    "overlay_path",
    "pseudo_label_json",
    "pseudo_label_yolo_txt",
    "status",
    "notes",
]

FAILURE_COLUMNS = ["image_path", "stage", "error_type", "message"]


@dataclass(frozen=True)
class PseudoLabelResult:
    image_id: str
    image_path: Path
    prediction_count: int
    high_confidence_count: int
    predicted_classes: tuple[str, ...]
    high_confidence_classes: tuple[str, ...]
    overlay_path: Path
    pseudo_label_json: Path
    pseudo_label_yolo_txt: Path
    status: str = "ok"
    notes: str = ""


def main() -> None:
    args = parse_args()
    results, failures = run_pseudo_labeling(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        high_confidence_threshold=args.high_confidence_threshold,
        device=args.device,
    )
    print("Unlabeled pseudo-labeling")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Images: {len(results) + len(failures)}")
    print(f"Succeeded: {len(results)}")
    print(f"Failed: {len(failures)}")
    print(f"High-confidence instances: {sum(item.high_confidence_count for item in results)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/unlabeled"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pseudo_labels/unlabeled_v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/yoloe26s_food_balanced.toml"))
    parser.add_argument("--high-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def run_pseudo_labeling(
    *,
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    high_confidence_threshold: float = 0.85,
    device: str = "cpu",
) -> tuple[list[PseudoLabelResult], list[dict[str, str]]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not 0.0 <= high_confidence_threshold <= 1.0:
        raise ValueError("high_confidence_threshold must be in [0, 1]")

    image_paths = input_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"no supported images found in {input_dir}")

    model_spec, confidence, nms_iou = load_experiment_config(config_path)
    run_spec = RunSpec(
        config_id=config_id(model_spec.name, confidence, nms_iou),
        model=model_spec,
        conf=confidence,
        iou=nms_iou,
    )
    model = load_ultralytics_model(resolve_model_path(Path("."), model_spec.filename), model_spec.prompts)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    overlays_dir = output_dir / "overlays"
    pseudo_json_dir = output_dir / "pseudo_labels" / "json"
    pseudo_yolo_dir = output_dir / "pseudo_labels" / "yolo_txt"
    for directory in (predictions_dir, overlays_dir, pseudo_json_dir, pseudo_yolo_dir):
        directory.mkdir(parents=True, exist_ok=True)

    results: list[PseudoLabelResult] = []
    failures: list[dict[str, str]] = []
    for image_path in image_paths:
        try:
            results.append(
                pseudo_label_image(
                    image_path=image_path,
                    model=model,
                    run_spec=run_spec,
                    high_confidence_threshold=high_confidence_threshold,
                    predictions_dir=predictions_dir,
                    overlays_dir=overlays_dir,
                    pseudo_json_dir=pseudo_json_dir,
                    pseudo_yolo_dir=pseudo_yolo_dir,
                    device=device,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "image_path": str(image_path),
                    "stage": "pseudo_label_image",
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:240],
                }
            )

    write_summary_csv(output_dir / "summary.csv", results)
    write_failures_csv(output_dir / "failures.csv", failures)
    write_run_metadata(
        output_dir / "run_metadata.json", config_path, run_spec, high_confidence_threshold, results, failures
    )
    return results, failures


def pseudo_label_image(
    *,
    image_path: Path,
    model: Any,
    run_spec: RunSpec,
    high_confidence_threshold: float,
    predictions_dir: Path,
    overlays_dir: Path,
    pseudo_json_dir: Path,
    pseudo_yolo_dir: Path,
    device: str,
) -> PseudoLabelResult:
    image_bgr = load_image_bgr(image_path)
    inference_results = model.predict(
        source=str(image_path),
        conf=run_spec.conf,
        iou=run_spec.iou,
        device=device,
        verbose=False,
    )
    instance_mask, predictions = result_to_instance_mask(inference_results[0], image_bgr.shape[:2])
    all_instances = list(predictions["instances"])
    high_confidence_instances = filter_high_confidence_instances(all_instances, high_confidence_threshold)

    prediction_path = predictions_dir / f"{image_path.stem}_predictions.json"
    overlay_path = overlays_dir / f"{image_path.stem}_overlay.jpg"
    pseudo_json_path = pseudo_json_dir / f"{image_path.stem}.json"
    pseudo_yolo_path = pseudo_yolo_dir / f"{image_path.stem}.txt"

    prediction_payload = {
        "image_id": image_path.stem,
        "image_path": str(image_path),
        "config_id": run_spec.config_id,
        "instances": all_instances,
    }
    prediction_path.write_text(json.dumps(prediction_payload, indent=2, sort_keys=True), encoding="utf-8")
    write_overlay(image_bgr, instance_mask, overlay_path)
    write_pseudo_label_json(
        pseudo_json_path, image_path, run_spec.config_id, high_confidence_threshold, high_confidence_instances
    )
    write_pseudo_label_yolo_txt(pseudo_yolo_path, high_confidence_instances)

    return PseudoLabelResult(
        image_id=image_path.stem,
        image_path=image_path,
        prediction_count=len(all_instances),
        high_confidence_count=len(high_confidence_instances),
        predicted_classes=classes_from_instances(all_instances),
        high_confidence_classes=classes_from_instances(high_confidence_instances),
        overlay_path=overlay_path,
        pseudo_label_json=pseudo_json_path,
        pseudo_label_yolo_txt=pseudo_yolo_path,
    )


def filter_high_confidence_instances(instances: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    filtered = []
    for instance in instances:
        confidence = instance.get("confidence")
        if confidence is not None and float(confidence) >= threshold:
            filtered.append(instance)
    return filtered


def classes_from_instances(instances: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({str(instance.get("class_name", "")) for instance in instances if instance.get("class_name")}))


def write_pseudo_label_json(
    path: Path,
    image_path: Path,
    config_id_value: str,
    threshold: float,
    instances: list[dict[str, Any]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "config_id": config_id_value,
                "pseudo_label_policy": {
                    "confidence_threshold": threshold,
                    "review_required": True,
                    "training_gold": False,
                },
                "instances": instances,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def write_pseudo_label_yolo_txt(path: Path, instances: list[dict[str, Any]]) -> None:
    lines = []
    for instance in instances:
        polygon = instance.get("polygon") or []
        if len(polygon) < 3:
            continue
        class_id = int(instance["class_id"])
        values = [str(class_id)]
        for x, y in polygon:
            values.extend((f"{float(x):.6f}", f"{float(y):.6f}"))
        lines.append(" ".join(values))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_summary_csv(path: Path, rows: list[PseudoLabelResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_id": row.image_id,
                    "image_path": str(row.image_path),
                    "prediction_count": row.prediction_count,
                    "high_confidence_count": row.high_confidence_count,
                    "predicted_classes": ";".join(row.predicted_classes),
                    "high_confidence_classes": ";".join(row.high_confidence_classes),
                    "overlay_path": str(row.overlay_path),
                    "pseudo_label_json": str(row.pseudo_label_json),
                    "pseudo_label_yolo_txt": str(row.pseudo_label_yolo_txt),
                    "status": row.status,
                    "notes": row.notes,
                }
            )


def write_failures_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FAILURE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_run_metadata(
    path: Path,
    config_path: Path,
    run_spec: RunSpec,
    high_confidence_threshold: float,
    results: list[PseudoLabelResult],
    failures: list[dict[str, str]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "config_path": str(config_path),
                "config_id": run_spec.config_id,
                "high_confidence_threshold": high_confidence_threshold,
                "image_count": len(results) + len(failures),
                "succeeded": len(results),
                "failed": len(failures),
                "high_confidence_instances": sum(result.high_confidence_count for result in results),
                "training_gold": False,
                "review_required": True,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
