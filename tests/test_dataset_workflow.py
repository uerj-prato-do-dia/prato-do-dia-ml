from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts.audit_dataset_images import AuditThresholds, audit_manifest
from scripts.create_dataset_manifest import build_manifest, write_manifest


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


def _write_image(path: Path, *, size: tuple[int, int], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_mask(path: Path, *, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    assert cv2.imwrite(str(path), mask)
