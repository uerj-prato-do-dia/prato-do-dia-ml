"""Audit dataset image quality from the dataset manifest."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


AUDIT_COLUMNS = [
    "image_id",
    "image_path",
    "exists",
    "width",
    "height",
    "file_size_kb",
    "format",
    "blur_score",
    "brightness",
    "contrast",
    "has_mask",
    "flags",
]


@dataclass(frozen=True)
class AuditThresholds:
    min_width: int = 512
    min_height: int = 512
    blur_score: float = 80.0
    too_dark: float = 50.0
    too_bright: float = 220.0
    low_contrast: float = 25.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_audit/images_quality.csv"))
    parser.add_argument("--min-width", type=int, default=512)
    parser.add_argument("--min-height", type=int, default=512)
    parser.add_argument("--blur-score", type=float, default=80.0)
    parser.add_argument("--too-dark", type=float, default=50.0)
    parser.add_argument("--too-bright", type=float, default=220.0)
    parser.add_argument("--low-contrast", type=float, default=25.0)
    args = parser.parse_args()

    thresholds = AuditThresholds(
        min_width=args.min_width,
        min_height=args.min_height,
        blur_score=args.blur_score,
        too_dark=args.too_dark,
        too_bright=args.too_bright,
        low_contrast=args.low_contrast,
    )
    rows = audit_manifest(args.manifest, thresholds)
    write_audit(args.output, rows)
    print_summary(rows, args.output)


def audit_manifest(manifest_path: Path, thresholds: AuditThresholds | None = None) -> list[dict[str, str]]:
    thresholds = thresholds or AuditThresholds()
    with manifest_path.open(newline="", encoding="utf-8") as file:
        manifest_rows = list(csv.DictReader(file))
    return [audit_row(row, thresholds) for row in manifest_rows]


def audit_row(row: dict[str, str], thresholds: AuditThresholds | None = None) -> dict[str, str]:
    thresholds = thresholds or AuditThresholds()
    image_path = PROJECT_ROOT / row["image_path"]
    mask_path = PROJECT_ROOT / row.get("mask_path", "") if row.get("mask_path") else None
    has_mask = bool(mask_path and mask_path.exists())
    flags: list[str] = []

    if not image_path.exists():
        return {
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "exists": "false",
            "width": "",
            "height": "",
            "file_size_kb": "",
            "format": "",
            "blur_score": "",
            "brightness": "",
            "contrast": "",
            "has_mask": str(has_mask).lower(),
            "flags": "missing_image",
        }

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return _error_row(row, image_path, has_mask, "decode_error")

    height, width = image.shape[:2]
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    brightness = float(np.mean(grayscale))
    contrast = float(np.std(grayscale))

    if width < thresholds.min_width or height < thresholds.min_height:
        flags.append("small_image")
    if blur_score < thresholds.blur_score:
        flags.append("maybe_blurry")
    if brightness < thresholds.too_dark:
        flags.append("too_dark")
    if brightness > thresholds.too_bright:
        flags.append("too_bright")
    if contrast < thresholds.low_contrast:
        flags.append("low_contrast")
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        flags.append("unsupported_format")
    if not has_mask:
        flags.append("missing_mask")
    if not flags:
        flags.append("ok")

    return {
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "exists": "true",
        "width": str(width),
        "height": str(height),
        "file_size_kb": f"{image_path.stat().st_size / 1024:.1f}",
        "format": image_path.suffix.lower().lstrip(".").upper(),
        "blur_score": f"{blur_score:.2f}",
        "brightness": f"{brightness:.2f}",
        "contrast": f"{contrast:.2f}",
        "has_mask": str(has_mask).lower(),
        "flags": ";".join(flags),
    }


def _error_row(row: dict[str, str], image_path: Path, has_mask: bool, flag: str) -> dict[str, str]:
    return {
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "exists": "true",
        "width": "",
        "height": "",
        "file_size_kb": f"{image_path.stat().st_size / 1024:.1f}",
        "format": image_path.suffix.lower().lstrip(".").upper(),
        "blur_score": "",
        "brightness": "",
        "contrast": "",
        "has_mask": str(has_mask).lower(),
        "flags": flag,
    }


def write_audit(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=AUDIT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    def count_flag(flag: str) -> int:
        return sum(flag in row["flags"].split(";") for row in rows)

    with_masks = sum(row["has_mask"] == "true" for row in rows)
    print("Dataset audit")
    print(f"Images: {len(rows)}")
    print(f"With masks: {with_masks}")
    print(f"Missing masks: {len(rows) - with_masks}")
    print(f"Maybe blurry: {count_flag('maybe_blurry')}")
    print(f"Too dark: {count_flag('too_dark')}")
    print(f"Low contrast: {count_flag('low_contrast')}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
