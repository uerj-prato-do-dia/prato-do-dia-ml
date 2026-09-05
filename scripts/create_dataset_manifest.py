#!/usr/bin/env python3
"""Dataset Manifest and YAML Generator for YOLOv11-seg Training.

Validates label files in data/processed_640, checks strict class ID ranges [0, 15],
applies anti-leakage prefix partitioning (70% train, 20% val, 10% test), and generates data/dataset.yaml.

Usage:
    python3 prato-do-dia-ml/scripts/create_dataset_manifest.py --input-dir data/processed_640 --output-dir data
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

# Resolve project root dynamically (prato-do-dia)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed_640"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"

CLASS_NAMES = [
    "tomate",
    "salada_verde",
    "feijao",
    "batata_frita",
    "arroz",
    "carne_moida",
    "pure_batata",
    "farofa",
    "cenoura",
    "ovo_frito",
    "massa_macarrao",
    "frango_grelhado",
    "azeitona",
    "batata_palha",
    "estrogonofe",
    "carne_bovina_bife",
]

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


def extract_meal_prefix(filename_stem: str) -> str:
    """Extract base meal prefix to prevent data leakage across splits."""
    parts = filename_stem.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return filename_stem


def validate_yolo_txt_file(txt_path: Path) -> tuple[bool, str]:
    """Validate YOLO segmentation TXT format and strict [0, 15] class ID range."""
    if not txt_path.exists():
        return False, "File does not exist"

    lines = [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return False, "Empty label file"

    for idx, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            return False, f"Line {idx}: Invalid polygon coordinate format"

        try:
            class_id = int(parts[0])
            if not (0 <= class_id <= 15):
                return False, f"Line {idx}: Invalid class_id {class_id} (must be in range 0..15)"

            coords = [float(v) for v in parts[1:]]
            for c in coords:
                if not (0.0 <= c <= 1.0):
                    return False, f"Line {idx}: Coordinate {c} out of normalized bounds [0.0, 1.0]"
        except ValueError as exc:
            return False, f"Line {idx}: Non-numeric values found ({exc})"

    return True, "Valid"


def write_manifest(manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to dataset_manifest.csv."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(
    input_dir: Path,
    ground_truth_dir: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Build dataset manifest CSV rows for dataset images."""
    existing_meta: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("image_id"):
                    existing_meta[row["image_id"]] = dict(row)

    images = sorted([p for p in input_dir.glob("*.jpg")])
    rows: list[dict[str, Any]] = []

    for img_path in images:
        image_id = img_path.stem
        prev = existing_meta.get(image_id, {})

        mask_candidate = ground_truth_dir / f"{image_id}_instances.png"
        mask_path_str = str(mask_candidate) if mask_candidate.exists() else prev.get("mask_path", "")

        rows.append(
            {
                "image_id": image_id,
                "image_path": str(img_path),
                "mask_path": mask_path_str,
                "source": prev.get("source", "own"),
                "split": prev.get("split", "candidate"),
                "foods": prev.get("foods", "unknown"),
                "plate_type": prev.get("plate_type", "unknown"),
                "lighting": prev.get("lighting", "unknown"),
                "angle": prev.get("angle", "unknown"),
                "quality": prev.get("quality", "unknown"),
                "notes": prev.get("notes", "auto_discovered"),
            }
        )

    write_manifest(manifest_path, rows)
    return rows


def create_dataset(
    input_dir: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    images = sorted([p for p in input_dir.glob("*.jpg")])
    if not images:
        print(f"No JPG images found in {input_dir}")
        return

    valid_pairs: list[tuple[Path, Path]] = []
    rejected: list[tuple[Path, str]] = []

    for img_path in images:
        txt_path = input_dir / f"{img_path.stem}.txt"
        is_valid, reason = validate_yolo_txt_file(txt_path)
        if is_valid:
            valid_pairs.append((img_path, txt_path))
        else:
            rejected.append((img_path, reason))

    print(f"Found {len(images)} images in {input_dir}")
    print(f"Valid image-label pairs: {len(valid_pairs)}")

    if not valid_pairs:
        print("No valid image-label pairs to process.")
        return

    meal_groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for img_path, txt_path in valid_pairs:
        prefix = extract_meal_prefix(img_path.stem)
        meal_groups[prefix].append((img_path, txt_path))

    meal_prefixes = sorted(list(meal_groups.keys()))
    random.seed(seed)
    random.shuffle(meal_prefixes)

    num_meals = len(meal_prefixes)
    n_train = max(1, int(num_meals * train_ratio))
    n_val = max(1, int(num_meals * val_ratio))

    splits = {
        "train": set(meal_prefixes[:n_train]),
        "val": set(meal_prefixes[n_train : n_train + n_val]),
        "test": set(meal_prefixes[n_train + n_val :]),
    }

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}

    for split_name, prefixes in splits.items():
        for prefix in prefixes:
            for img_path, txt_path in meal_groups[prefix]:
                dst_img = output_dir / "images" / split_name / img_path.name
                dst_txt = output_dir / "labels" / split_name / txt_path.name

                shutil.copy2(img_path, dst_img)
                shutil.copy2(txt_path, dst_txt)
                counts[split_name] += 1

    yaml_lines = [
        "# Canonical Prato do Dia Dataset Manifest",
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        yaml_lines.append(f"  {idx}: {name}")

    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print("\n=========================================================")
    print("   MANIFESTO DE DATASET YOLOv11-seg GERADO COM SUCESSO")
    print("=========================================================")
    print(f" Manifesto YAML: {yaml_path.resolve()}")
    print(f" Imagens em Train: {counts['train']}")
    print(f" Imagens em Val:   {counts['val']}")
    print(f" Imagens em Test:  {counts['test']}")
    print("=========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YOLOv11-seg dataset manifest and dataset.yaml.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Input directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output dataset directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility")
    args = parser.parse_args()

    create_dataset(args.input_dir, args.output_dir, seed=args.seed)
