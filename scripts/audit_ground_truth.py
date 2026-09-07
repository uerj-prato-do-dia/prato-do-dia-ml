"""Audit ground-truth instance masks used for evaluation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_COLUMNS = [
    "image_id",
    "image_path",
    "mask_path",
    "image_width",
    "image_height",
    "mask_width",
    "mask_height",
    "mask_exists",
    "dimension_match",
    "unique_values",
    "instance_count",
    "foreground_area_px",
    "foreground_ratio",
    "flags",
]


@dataclass(frozen=True)
class GroundTruthThresholds:
    very_small_foreground: float = 0.02
    very_large_foreground: float = 0.95
    tiny_instance_ratio: float = 0.001
    tiny_instance_count: int = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_audit/ground_truth_quality.csv"))
    args = parser.parse_args()

    rows = audit_ground_truth(args.manifest)
    write_ground_truth_audit(args.output, rows)
    print_summary(rows, args.output)


def audit_ground_truth(
    manifest_path: Path,
    thresholds: GroundTruthThresholds | None = None,
) -> list[dict[str, str]]:
    thresholds = thresholds or GroundTruthThresholds()
    with manifest_path.open(newline="", encoding="utf-8") as file:
        manifest_rows = [row for row in csv.DictReader(file) if row.get("mask_path")]
    return [audit_ground_truth_row(row, thresholds) for row in manifest_rows]


def audit_ground_truth_row(
    row: dict[str, str],
    thresholds: GroundTruthThresholds | None = None,
) -> dict[str, str]:
    thresholds = thresholds or GroundTruthThresholds()
    image_path = PROJECT_ROOT / row["image_path"]
    mask_path = PROJECT_ROOT / row["mask_path"]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image_height = image.shape[0] if image is not None else 0
    image_width = image.shape[1] if image is not None else 0

    if not mask_path.exists():
        return _ground_truth_row(
            row,
            image_width=image_width,
            image_height=image_height,
            mask_width=0,
            mask_height=0,
            mask_exists=False,
            dimension_match=False,
            unique_values=[],
            flags=["missing_mask"],
        )

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2:
        return _ground_truth_row(
            row,
            image_width=image_width,
            image_height=image_height,
            mask_width=0,
            mask_height=0,
            mask_exists=True,
            dimension_match=False,
            unique_values=[],
            flags=["decode_error"],
        )

    mask_height, mask_width = mask.shape[:2]
    unique_values = [int(value) for value in np.unique(mask)]
    foreground = mask != 0
    foreground_area = int(foreground.sum())
    total_area = int(mask.size)
    foreground_ratio = foreground_area / max(total_area, 1)
    instance_ids = [value for value in unique_values if value != 0]
    dimension_match = image_width == mask_width and image_height == mask_height and image is not None
    flags: list[str] = []

    if foreground_area == 0:
        flags.append("empty_mask")
    if not dimension_match:
        flags.append("dimension_mismatch")
    if len(instance_ids) == 1:
        flags.append("single_instance_only")
    if foreground_ratio < thresholds.very_small_foreground:
        flags.append("very_small_foreground")
    if foreground_ratio > thresholds.very_large_foreground:
        flags.append("very_large_foreground")
    if _tiny_instance_count(mask, instance_ids, thresholds.tiny_instance_ratio) >= thresholds.tiny_instance_count:
        flags.append("too_many_tiny_instances")
    if not flags:
        flags.append("ok")

    return _ground_truth_row(
        row,
        image_width=image_width,
        image_height=image_height,
        mask_width=mask_width,
        mask_height=mask_height,
        mask_exists=True,
        dimension_match=dimension_match,
        unique_values=unique_values,
        flags=flags,
        instance_count=len(instance_ids),
        foreground_area_px=foreground_area,
        foreground_ratio=foreground_ratio,
    )


def _ground_truth_row(
    row: dict[str, str],
    *,
    image_width: int,
    image_height: int,
    mask_width: int,
    mask_height: int,
    mask_exists: bool,
    dimension_match: bool,
    unique_values: list[int],
    flags: list[str],
    instance_count: int = 0,
    foreground_area_px: int = 0,
    foreground_ratio: float = 0.0,
) -> dict[str, str]:
    return {
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "mask_path": row.get("mask_path", ""),
        "image_width": str(image_width or ""),
        "image_height": str(image_height or ""),
        "mask_width": str(mask_width or ""),
        "mask_height": str(mask_height or ""),
        "mask_exists": str(mask_exists).lower(),
        "dimension_match": str(dimension_match).lower(),
        "unique_values": ";".join(str(value) for value in unique_values),
        "instance_count": str(instance_count),
        "foreground_area_px": str(foreground_area_px),
        "foreground_ratio": f"{foreground_ratio:.6f}",
        "flags": ";".join(flags),
    }


def _tiny_instance_count(mask: np.ndarray, instance_ids: list[int], tiny_instance_ratio: float) -> int:
    total_area = max(int(mask.size), 1)
    tiny_count = 0
    for instance_id in instance_ids:
        ratio = int((mask == instance_id).sum()) / total_area
        if ratio < tiny_instance_ratio:
            tiny_count += 1
    return tiny_count


def write_ground_truth_audit(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=GROUND_TRUTH_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    def count_flag(flag: str) -> int:
        return sum(flag in row["flags"].split(";") for row in rows)

    print("Ground-truth audit")
    print(f"Rows with masks: {len(rows)}")
    print(f"Missing masks: {count_flag('missing_mask')}")
    print(f"Dimension mismatches: {count_flag('dimension_mismatch')}")
    print(f"Empty masks: {count_flag('empty_mask')}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
