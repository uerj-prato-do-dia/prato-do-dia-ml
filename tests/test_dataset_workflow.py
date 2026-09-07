from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts.audit_dataset_images import AuditThresholds, audit_manifest
from scripts.audit_ground_truth import audit_ground_truth
from scripts.create_dataset_manifest import build_manifest, write_manifest
from scripts.ingest_unlabeled_dataset import discover_input_files, ingest_dataset


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


def test_sync_classes_txt_generates_canonical_file(tmp_path: Path) -> None:
    from prato_do_dia_ml.schema import get_canonical_classes, sync_all_classes_txt, write_classes_txt

    classes = get_canonical_classes()
    assert len(classes) == 16
    assert classes[0] == "tomate"
    assert classes[4] == "arroz"

    output_file = tmp_path / "classes.txt"
    write_classes_txt(output_file)
    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 16
    assert lines[0] == "tomate"

    synced = sync_all_classes_txt(project_root=tmp_path)
    assert len(synced) == 3
    for p in synced:
        assert p.exists()
        assert p.read_text(encoding="utf-8").splitlines()[0] == "tomate"


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
