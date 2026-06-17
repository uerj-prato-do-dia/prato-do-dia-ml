from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.audit_dataset_images import AuditThresholds, audit_manifest
from scripts.audit_ground_truth import audit_ground_truth
from scripts.create_dataset_manifest import build_manifest, write_manifest
from scripts.ingest_unlabeled_dataset import discover_input_files, ingest_dataset
from scripts.pseudo_label_unlabeled import filter_high_confidence_instances, write_pseudo_label_yolo_txt
from scripts.render_overlay_review import (
    load_review_items,
    parse_mask_errors,
    render_review,
    review_csv_row,
    serialize_mask_errors,
)
from scripts.run_threshold_sweep import aggregate_config_result, best_result, parse_float_list, sweep_configs
from scripts.run_ultralytics_zero_shot import (
    ModelSpec,
    RunSpec,
    best_with_min_precision,
    config_id,
    load_experiment_config,
    make_run_specs,
    resolve_model_path,
    selected_model_specs,
)


def test_build_manifest_discovers_images_and_preserves_manual_metadata(tmp_path: Path) -> None:
    input_dir = tmp_path / "data" / "input"
    ground_truth_dir = tmp_path / "data" / "ground_truth"
    manifest_path = tmp_path / "data" / "dataset_manifest.csv"
    input_dir.mkdir(parents=True)
    ground_truth_dir.mkdir(parents=True)
    _write_image(input_dir / "imagem2.jpg", size=(640, 640), value=180)
    _write_image(input_dir / "imagem1.jpg", size=(640, 640), value=180)
    _write_mask(ground_truth_dir / "imagem1_instances.png", size=(640, 640))

    write_manifest(
        manifest_path,
        [
            {
                "image_id": "imagem1",
                "image_path": str(input_dir / "imagem1.jpg"),
                "mask_path": "",
                "source": "own",
                "split": "candidate",
                "foods": "rice;beans",
                "plate_type": "white_plate",
                "lighting": "indoor",
                "angle": "top_down",
                "quality": "ok",
                "notes": "manual",
            }
        ],
    )

    rows = build_manifest(input_dir, ground_truth_dir, manifest_path)

    assert [row["image_id"] for row in rows] == ["imagem1", "imagem2"]
    assert rows[0]["mask_path"].endswith("imagem1_instances.png")
    assert rows[0]["foods"] == "rice;beans"
    assert rows[0]["notes"] == "manual"
    assert rows[1]["split"] == "candidate"
    assert rows[1]["notes"] == "auto_discovered"


def test_build_manifest_preserves_bad_manual_classification(tmp_path: Path) -> None:
    input_dir = tmp_path / "data" / "input"
    ground_truth_dir = tmp_path / "data" / "ground_truth"
    manifest_path = tmp_path / "data" / "dataset_manifest.csv"
    input_dir.mkdir(parents=True)
    ground_truth_dir.mkdir(parents=True)
    _write_image(input_dir / "imagem9.jpg", size=(640, 640), value=180)

    write_manifest(
        manifest_path,
        [
            {
                "image_id": "imagem9",
                "image_path": str(input_dir / "imagem9.jpg"),
                "mask_path": "",
                "source": "own",
                "split": "bad",
                "foods": "unknown",
                "plate_type": "unknown",
                "lighting": "unknown",
                "angle": "unknown",
                "quality": "bad",
                "notes": "not_top_down_bad_collection_example",
            }
        ],
    )

    rows = build_manifest(input_dir, ground_truth_dir, manifest_path)

    assert rows[0]["split"] == "bad"
    assert rows[0]["quality"] == "bad"
    assert rows[0]["notes"] == "not_top_down_bad_collection_example"


def test_audit_manifest_flags_small_dark_low_contrast_and_missing_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "small_dark.jpg"
    manifest_path = tmp_path / "manifest.csv"
    _write_image(image_path, size=(32, 32), value=20)
    manifest_path.write_text(
        "\n".join(
            [
                "image_id,image_path,mask_path,source,split,foods,plate_type,lighting,angle,quality,notes",
                f"small_dark,{image_path},,own,candidate,unknown,unknown,unknown,unknown,unknown,test",
            ]
        ),
        encoding="utf-8",
    )

    rows = audit_manifest(
        manifest_path,
        AuditThresholds(min_width=512, min_height=512, blur_score=80, too_dark=50, too_bright=220, low_contrast=25),
    )

    flags = rows[0]["flags"].split(";")
    assert "small_image" in flags
    assert "too_dark" in flags
    assert "low_contrast" in flags
    assert "missing_mask" in flags


def test_ground_truth_audit_valid_mask_returns_ok(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    mask_path = tmp_path / "mask.png"
    manifest_path = tmp_path / "manifest.csv"
    _write_image(image_path, size=(100, 100), value=180)
    _write_mask(mask_path, size=(100, 100), instance_count=2)
    _write_manifest(manifest_path, image_path=image_path, mask_path=mask_path)

    rows = audit_ground_truth(manifest_path)

    assert rows[0]["dimension_match"] == "true"
    assert rows[0]["instance_count"] == "2"
    assert rows[0]["flags"] == "ok"


def test_ground_truth_audit_flags_missing_empty_dimension_and_single_instance(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    empty_mask_path = tmp_path / "empty.png"
    small_mask_path = tmp_path / "small.png"
    missing_mask_path = tmp_path / "missing.png"
    manifest_path = tmp_path / "manifest.csv"
    _write_image(image_path, size=(100, 100), value=180)
    _write_empty_mask(empty_mask_path, size=(100, 100))
    _write_mask(small_mask_path, size=(50, 50), instance_count=1)
    _write_manifest_rows(
        manifest_path,
        [
            ("missing", image_path, missing_mask_path),
            ("empty", image_path, empty_mask_path),
            ("mismatch", image_path, small_mask_path),
        ],
    )

    rows = {row["image_id"]: row for row in audit_ground_truth(manifest_path)}

    assert "missing_mask" in rows["missing"]["flags"].split(";")
    assert "empty_mask" in rows["empty"]["flags"].split(";")
    assert "dimension_mismatch" in rows["mismatch"]["flags"].split(";")
    assert "single_instance_only" in rows["mismatch"]["flags"].split(";")


def test_threshold_sweep_helpers_parse_generate_and_select_best() -> None:
    assert parse_float_list("0.05,0.10") == [0.05, 0.10]
    assert sweep_configs([0.05, 0.10], [0.35]) == [
        ("conf_0p05_nms_0p35", 0.05, 0.35),
        ("conf_0p10_nms_0p35", 0.10, 0.35),
    ]
    rows = [
        {"config_id": "a", "foreground_iou": "0.1", "instance_iou": "0.4", "recall": "0.2"},
        {"config_id": "b", "foreground_iou": "0.2", "instance_iou": "0.3", "recall": "0.5"},
    ]
    assert best_result(rows, "foreground_iou")["config_id"] == "b"
    assert best_result(rows, "instance_iou")["config_id"] == "a"


def test_threshold_sweep_aggregates_fake_metrics() -> None:
    result = aggregate_config_result(
        "conf_0p05_nms_0p35",
        0.05,
        0.35,
        [
            {
                "foreground_iou": "0.100000",
                "instance_iou": "0.200000",
                "dice": "0.300000",
                "boundary_f_score": "0.400000",
                "precision": "1.000000",
                "recall": "0.500000",
                "false_positives": "0",
                "missed_instances": "2",
                "status": "ok",
            },
            {
                "foreground_iou": "0.300000",
                "instance_iou": "0.400000",
                "dice": "0.500000",
                "boundary_f_score": "0.600000",
                "precision": "0.500000",
                "recall": "0.250000",
                "false_positives": "1",
                "missed_instances": "3",
                "status": "ok",
            },
        ],
    )

    assert result["foreground_iou"] == "0.200000"
    assert result["recall"] == "0.375000"
    assert result["false_positives"] == "1"
    assert result["missed_instances"] == "5"


def test_ultralytics_zero_shot_helpers_generate_ids_and_configs() -> None:
    model = ModelSpec("yoloe26s_seg_food", "yoloe-26s-seg.pt", "open_vocab_food_prompts", ("rice", "beans"))

    assert config_id(model.name, 0.01, 0.45) == "yoloe26s_seg_food_conf_0p01_iou_0p45"
    assert make_run_specs([model], [0.01, 0.03], [0.45]) == [
        RunSpec("yoloe26s_seg_food_conf_0p01_iou_0p45", model, 0.01, 0.45),
        RunSpec("yoloe26s_seg_food_conf_0p03_iou_0p45", model, 0.03, 0.45),
    ]


def test_ultralytics_zero_shot_model_selection_defaults_to_comparable_models() -> None:
    class Args:
        include_yolo11 = False
        include_yoloe = False
        include_mobilesam_smoke = False

    specs = selected_model_specs(Args())

    assert [spec.name for spec in specs] == [
        "yolo11m_seg",
        "yolo11l_seg",
        "yolo11x_seg",
        "yoloe26s_seg_food",
        "yoloe26m_seg_food",
    ]
    assert specs[-1].prompts


def test_ultralytics_zero_shot_best_with_min_precision() -> None:
    rows = [
        {"config_id": "high_recall_low_precision", "precision": "0.50", "recall": "0.90"},
        {"config_id": "balanced", "precision": "0.80", "recall": "0.70"},
        {"config_id": "conservative", "precision": "0.95", "recall": "0.40"},
    ]

    assert best_with_min_precision(rows, 0.75)["config_id"] == "balanced"
    assert best_with_min_precision(rows, 0.99) is None


def test_ultralytics_zero_shot_loads_experiment_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[ultralytics]
model_name = "yoloe26s_seg_food"
model_path = "external_models/ultralytics/yoloe-26s-seg.pt"
model_type = "open_vocab_food_prompts"
confidence_threshold = 0.10
nms_iou_threshold = 0.30

[prompts]
classes = ["rice", "beans"]
""".strip(),
        encoding="utf-8",
    )

    model, confidence, nms_iou = load_experiment_config(config_path)

    assert model.name == "yoloe26s_seg_food"
    assert model.filename == "external_models/ultralytics/yoloe-26s-seg.pt"
    assert model.prompts == ("rice", "beans")
    assert confidence == 0.10
    assert nms_iou == 0.30


def test_ultralytics_zero_shot_resolves_model_paths() -> None:
    assert resolve_model_path(Path("models-dir"), "yoloe.pt") == Path("models-dir/yoloe.pt")
    assert resolve_model_path(Path("models-dir"), "external_models/ultralytics/yoloe.pt") == Path("models-dir/yoloe.pt")
    assert resolve_model_path(Path("."), "external_models/ultralytics/yoloe.pt") == Path(
        "external_models/ultralytics/yoloe.pt"
    )


def test_unlabeled_ingestion_hashes_valid_images_and_reports_rejections(tmp_path: Path) -> None:
    input_dir = tmp_path / "dataset"
    output_dir = tmp_path / "data" / "raw" / "unlabeled"
    report_dir = tmp_path / "outputs" / "dataset_ingestion"
    valid_path = input_dir / "phone meal.JPG"
    corrupt_path = input_dir / "broken.jpg"
    _write_image(valid_path, size=(64, 48), value=180)
    corrupt_path.write_text("not an image", encoding="utf-8")

    ingested, rejected = ingest_dataset(input_dir=input_dir, output_dir=output_dir, report_dir=report_dir)

    assert len(ingested) == 1
    assert len(ingested[0].image_id) == 64
    assert ingested[0].output_path.name == f"{ingested[0].sha256}.jpg"
    assert ingested[0].output_path.exists()
    assert rejected[0].source_path == corrupt_path
    assert (report_dir / "ingested_images.csv").exists()
    assert (report_dir / "rejected_images.csv").exists()


def test_unlabeled_ingestion_discovers_images_recursively(tmp_path: Path) -> None:
    input_dir = tmp_path / "dataset"
    _write_image(input_dir / "a.jpg", size=(16, 16), value=100)
    _write_image(input_dir / "nested" / "b.png", size=(16, 16), value=120)
    (input_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    assert [path.name for path in discover_input_files(input_dir)] == ["a.jpg", "b.png"]


def test_pseudo_label_filters_and_writes_yolo_txt(tmp_path: Path) -> None:
    instances = [
        {
            "class_id": 0,
            "class_name": "rice",
            "confidence": 0.91,
            "polygon": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]],
        },
        {
            "class_id": 1,
            "class_name": "beans",
            "confidence": 0.40,
            "polygon": [[0.3, 0.3], [0.4, 0.3], [0.4, 0.4]],
        },
    ]

    filtered = filter_high_confidence_instances(instances, 0.85)
    output_path = tmp_path / "labels.txt"
    write_pseudo_label_yolo_txt(output_path, filtered)

    assert [item["class_name"] for item in filtered] == ["rice"]
    assert output_path.read_text(encoding="utf-8") == "0 0.100000 0.100000 0.200000 0.100000 0.200000 0.200000\n"


def test_overlay_review_loads_summary_and_marks_no_prediction(tmp_path: Path) -> None:
    summary_path = _write_overlay_review_summary(tmp_path)

    items = load_review_items(summary_path)

    assert [item.image_id for item in items] == ["image_a", "image_b"]
    assert items[0].prediction_count == 0
    assert items[0].mask_errors == ("no_prediction",)
    assert items[0].next_action == "needs_review"
    assert items[0].overlay_exists
    assert items[1].predicted_classes == "rice;beans"


def test_overlay_review_template_rows_include_review_defaults(tmp_path: Path) -> None:
    item = load_review_items(_write_overlay_review_summary(tmp_path))[0]

    row = review_csv_row(item)

    assert row["image_id"] == "image_a"
    assert row["prediction_count"] == 0
    assert row["image_condition"] == ""
    assert row["mask_errors"] == '["no_prediction"]'
    assert row["next_action"] == "needs_review"
    assert row["notes"] == ""


def test_overlay_review_serializes_mask_errors_as_json_arrays() -> None:
    assert serialize_mask_errors(()) == "[]"
    assert serialize_mask_errors(("missed_food", "hallucination")) == '["missed_food", "hallucination"]'
    assert parse_mask_errors('["missed_food", "hallucination"]') == ("missed_food", "hallucination")
    with pytest.raises(ValueError, match="unknown mask_errors values"):
        serialize_mask_errors(("bad_error",))
    with pytest.raises(ValueError, match="mask_errors must be a JSON array"):
        parse_mask_errors("missed_food")


def test_overlay_review_renders_static_html_template_and_manifest(tmp_path: Path) -> None:
    summary_path = _write_overlay_review_summary(tmp_path)
    output_dir = tmp_path / "outputs" / "reviews" / "unlabeled_v1_overlay_review"
    items = load_review_items(summary_path)

    render_review(items, output_dir, summary_path)

    template = (output_dir / "review_template.csv").read_text(encoding="utf-8")
    manifest = (output_dir / "review_manifest.json").read_text(encoding="utf-8")
    html = (output_dir / "index.html").read_text(encoding="utf-8")

    assert "image_id,image_path,overlay_path,prediction_count" in template
    assert "image_condition,mask_errors,next_action" in template
    assert "image_a" in template
    assert ',"[""no_prediction""]",needs_review' in template
    assert '"image_count": 2' in manifest
    assert '"missing_overlay_count": 0' in manifest
    assert "classFilter" in html
    assert "conditionFilter" in html
    assert "maskErrorFilter" in html
    assert "Export review CSV" in html
    assert 'data-field="mask_errors"' in html
    assert "annotate_hard means" in html
    assert "rerun_lower_threshold" not in html
    assert "keep_for_robustness" not in html
    assert "mask_quality" not in html
    assert "image_quality" not in html
    assert "image_b" in html
    assert "image_b_predictions.json" in html


def test_overlay_review_missing_summary_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="pseudo-label summary not found"):
        load_review_items(tmp_path / "missing.csv")


def test_overlay_review_records_missing_overlays(tmp_path: Path) -> None:
    summary_path = _write_overlay_review_summary(tmp_path, missing_overlay=True)
    output_dir = tmp_path / "review"
    items = load_review_items(summary_path)

    render_review(items, output_dir, summary_path)

    assert not items[1].overlay_exists
    html = (output_dir / "index.html").read_text(encoding="utf-8")
    manifest = (output_dir / "review_manifest.json").read_text(encoding="utf-8")
    assert "Missing overlay" in html
    assert '"missing_overlay_count": 1' in manifest


def _write_image(path: Path, *, size: tuple[int, int], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_mask(path: Path, *, size: tuple[int, int], instance_count: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    for index in range(instance_count):
        start = 10 + index * 20
        mask[start : start + 10, start : start + 10] = index + 1
    assert cv2.imwrite(str(path), mask)


def _write_empty_mask(path: Path, *, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    assert cv2.imwrite(str(path), mask)


def _write_manifest(manifest_path: Path, *, image_path: Path, mask_path: Path) -> None:
    _write_manifest_rows(manifest_path, [("image", image_path, mask_path)])


def _write_manifest_rows(manifest_path: Path, rows: list[tuple[str, Path, Path]]) -> None:
    manifest_path.write_text(
        "\n".join(
            ["image_id,image_path,mask_path,source,split,foods,plate_type,lighting,angle,quality,notes"]
            + [
                f"{image_id},{image_path},{mask_path},own,baseline_eval,unknown,unknown,unknown,unknown,unknown,test"
                for image_id, image_path, mask_path in rows
            ]
        ),
        encoding="utf-8",
    )


def _write_overlay_review_summary(tmp_path: Path, *, missing_overlay: bool = False) -> Path:
    base_dir = tmp_path / "outputs" / "pseudo_labels" / "unlabeled_v1"
    overlays_dir = base_dir / "overlays"
    predictions_dir = base_dir / "predictions"
    pseudo_json_dir = base_dir / "pseudo_labels" / "json"
    pseudo_yolo_dir = base_dir / "pseudo_labels" / "yolo_txt"
    image_dir = tmp_path / "data" / "raw" / "unlabeled"
    for directory in [overlays_dir, predictions_dir, pseudo_json_dir, pseudo_yolo_dir, image_dir]:
        directory.mkdir(parents=True)

    _write_image(image_dir / "image_a.jpg", size=(32, 32), value=120)
    _write_image(image_dir / "image_b.jpg", size=(32, 32), value=140)
    _write_image(overlays_dir / "image_a_overlay.jpg", size=(32, 32), value=160)
    if not missing_overlay:
        _write_image(overlays_dir / "image_b_overlay.jpg", size=(32, 32), value=180)
    for image_id in ["image_a", "image_b"]:
        (predictions_dir / f"{image_id}_predictions.json").write_text("{}", encoding="utf-8")
        (pseudo_json_dir / f"{image_id}.json").write_text("{}", encoding="utf-8")
        (pseudo_yolo_dir / f"{image_id}.txt").write_text("", encoding="utf-8")

    image_b_overlay = overlays_dir / "image_b_overlay.jpg"
    if missing_overlay:
        image_b_overlay = overlays_dir / "missing_overlay.jpg"
    summary_path = base_dir / "summary.csv"
    summary_path.write_text(
        "\n".join(
            [
                (
                    "image_id,image_path,prediction_count,high_confidence_count,predicted_classes,"
                    "high_confidence_classes,overlay_path,pseudo_label_json,pseudo_label_yolo_txt,status,notes"
                ),
                (
                    f"image_a,{image_dir / 'image_a.jpg'},0,0,,,"
                    f"{overlays_dir / 'image_a_overlay.jpg'},{pseudo_json_dir / 'image_a.json'},"
                    f"{pseudo_yolo_dir / 'image_a.txt'},ok,"
                ),
                (
                    f"image_b,{image_dir / 'image_b.jpg'},2,1,rice;beans,rice,"
                    f"{image_b_overlay},{pseudo_json_dir / 'image_b.json'},"
                    f"{pseudo_yolo_dir / 'image_b.txt'},ok,"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path
