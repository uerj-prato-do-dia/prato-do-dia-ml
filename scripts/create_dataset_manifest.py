"""Create or update the dataset manifest deterministically."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prato_do_dia_ml.io_utils import SUPPORTED_IMAGE_SUFFIXES, input_images

MANIFEST_COLUMNS = [
    "image_id",
    "image_path",
    "mask_path",
    "source",
    "split",
    "foods",
    "plate_type",
    "lighting",
    "angle",
    "quality",
    "notes",
]

MANUAL_COLUMNS = [
    "source",
    "split",
    "foods",
    "plate_type",
    "lighting",
    "angle",
    "quality",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/input"))
    parser.add_argument("--ground-truth-dir", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--output", type=Path, default=Path("data/dataset_manifest.csv"))
    parser.add_argument("--new-default-split", default="candidate")
    args = parser.parse_args()

    rows = build_manifest(
        args.input_dir,
        args.ground_truth_dir,
        args.output,
        new_default_split=args.new_default_split,
    )
    write_manifest(args.output, rows)
    with_masks = sum(1 for row in rows if row["mask_path"])
    print(f"Dataset manifest: {args.output}")
    print(f"Images: {len(rows)}")
    print(f"With masks: {with_masks}")


def build_manifest(
    input_dir: Path,
    ground_truth_dir: Path,
    existing_manifest: Path | None = None,
    *,
    new_default_split: str = "candidate",
) -> list[dict[str, str]]:
    existing_rows = read_existing_manifest(existing_manifest) if existing_manifest is not None else {}
    rows: list[dict[str, str]] = []

    for image_path in input_images(input_dir):
        image_id = image_path.stem
        existing = existing_rows.get(image_id, {})
        mask_path = find_mask_for_image(ground_truth_dir, image_id)
        row = default_row(image_id, image_path, mask_path, new_default_split)
        for column in MANUAL_COLUMNS:
            if existing.get(column):
                row[column] = existing[column]
        if existing.get("image_path") and Path(existing["image_path"]) != image_path:
            row["image_path"] = relative_path(image_path)
        if mask_path is not None:
            row["mask_path"] = relative_path(mask_path)
            if not existing.get("split"):
                row["split"] = "baseline_eval"
        rows.append(row)

    return sorted(rows, key=lambda row: row["image_id"])


def read_existing_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = {}
        for row in reader:
            image_id = row.get("image_id", "")
            if image_id:
                rows[image_id] = {column: row.get(column, "") for column in MANIFEST_COLUMNS}
        return rows


def default_row(image_id: str, image_path: Path, mask_path: Path | None, new_default_split: str) -> dict[str, str]:
    return {
        "image_id": image_id,
        "image_path": relative_path(image_path),
        "mask_path": relative_path(mask_path) if mask_path is not None else "",
        "source": "own",
        "split": "baseline_eval" if mask_path is not None else new_default_split,
        "foods": "unknown",
        "plate_type": "unknown",
        "lighting": "unknown",
        "angle": "unknown",
        "quality": "unknown",
        "notes": "auto_discovered",
    }


def find_mask_for_image(ground_truth_dir: Path, image_id: str) -> Path | None:
    candidates = [
        ground_truth_dir / f"{image_id}_instances.png",
        ground_truth_dir / f"{image_id}_instance.png",
        ground_truth_dir / f"{image_id}.png",
    ]
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidates.append(ground_truth_dir / f"{image_id}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
