"""Validate a local UECFoodPixComplete dataset checkout."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from skimage.io import imread


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uec-root", type=Path, required=True)
    parser.add_argument("--category-file", type=Path)
    parser.add_argument("--train-list", type=Path)
    parser.add_argument("--test-list", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-class-pixels", action="store_true")
    args = parser.parse_args()

    stats = validate_dataset(
        args.uec_root,
        category_file=args.category_file,
        train_list=args.train_list,
        test_list=args.test_list,
        limit=args.limit,
        per_class_pixels=args.per_class_pixels,
    )
    print(json.dumps(stats, indent=2))


def validate_dataset(
    root: Path,
    category_file: Path | None = None,
    train_list: Path | None = None,
    test_list: Path | None = None,
    limit: int | None = None,
    per_class_pixels: bool = False,
) -> dict[str, object]:
    category_file = category_file or _find_first(root, ("category.txt", "../category.txt"))
    train_list = train_list or _find_first(root, ("train.txt", "train9000.txt", "../train.txt", "../train9000.txt"))
    test_list = test_list or _find_first(root, ("test.txt", "test1000.txt", "../test.txt", "../test1000.txt"))
    required_files = [category_file, train_list, test_list]
    required_dirs = [root / "train" / "img", root / "train" / "mask", root / "test" / "img", root / "test" / "mask"]
    missing = [str(path) for path in required_files if not path.is_file()]
    missing.extend(str(path) for path in required_dirs if not path.is_dir())
    if missing:
        raise SystemExit("Invalid UECFoodPixComplete structure. Missing:\n" + "\n".join(missing))

    categories = read_categories(category_file)
    train_ids = read_ids(train_list)
    test_ids = read_ids(test_list)
    checked_train = _check_split(root, "train", train_ids, limit, per_class_pixels)
    checked_test = _check_split(root, "test", test_ids, limit, per_class_pixels)

    missing_files = checked_train["missing_files"] + checked_test["missing_files"]
    if missing_files:
        raise SystemExit("Missing image/mask files:\n" + "\n".join(missing_files[:50]))

    unique_values = sorted(set(checked_train["unique_values"]) | set(checked_test["unique_values"]))
    if any(value < 0 or value > 102 for value in unique_values):
        raise SystemExit(f"Mask values outside expected range 0..102: {unique_values}")

    pixel_counts = Counter()
    pixel_counts.update(checked_train["pixel_counts"])
    pixel_counts.update(checked_test["pixel_counts"])

    return {
        "root": str(root),
        "category_file": str(category_file),
        "train_list": str(train_list),
        "test_list": str(test_list),
        "train_images": len(train_ids),
        "test_images": len(test_ids),
        "checked_train_images": checked_train["checked"],
        "checked_test_images": checked_test["checked"],
        "categories": len(categories),
        "unique_mask_values_found": unique_values,
        "per_class_pixel_counts": dict(sorted(pixel_counts.items())) if per_class_pixels else "skipped",
    }


def read_categories(path: Path) -> dict[int, str]:
    categories: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if parts[0].lower() == "id":
            continue
        if len(parts) != 2:
            raise SystemExit(f"Invalid category line: {line!r}")
        categories[int(parts[0])] = parts[1]
    if not categories:
        raise SystemExit(f"No categories found in {path}")
    return categories


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_red_mask(path: Path) -> np.ndarray:
    try:
        mask = imread(path)
    except FileNotFoundError as exc:
        raise SystemExit(f"Could not read mask: {path}") from exc
    if mask.ndim == 2:
        return mask.astype(np.int32, copy=False)
    if mask.ndim == 3:
        # skimage loads color PNGs as RGB/RGBA, so channel 0 is the red channel.
        return mask[:, :, 0].astype(np.int32, copy=False)
    raise SystemExit(f"Unsupported mask shape {mask.shape}: {path}")


def align_mask_to_image(mask: np.ndarray, image_shape: tuple[int, int], image_id: str) -> np.ndarray:
    """Return mask aligned to image HxW, fixing known transposed masks."""

    if mask.shape == image_shape:
        return mask
    if mask.T.shape == image_shape:
        return mask.T
    raise SystemExit(f"Shape mismatch for {image_id}: image={image_shape} mask={mask.shape}")


def _find_first(root: Path, candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        path = (root / candidate).resolve()
        if path.exists():
            return path
    return (root / candidates[0]).resolve()


def _check_split(
    root: Path,
    split: str,
    ids: list[str],
    limit: int | None,
    per_class_pixels: bool,
) -> dict[str, object]:
    missing_files: list[str] = []
    unique_values: set[int] = set()
    pixel_counts: Counter[int] = Counter()
    checked = 0

    for image_id in ids[:limit]:
        image_path = root / split / "img" / f"{image_id}.jpg"
        mask_path = root / split / "mask" / f"{image_id}.png"
        if not image_path.is_file():
            missing_files.append(str(image_path))
        if not mask_path.is_file():
            missing_files.append(str(mask_path))
        if missing_files:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = read_red_mask(mask_path)
        if image is None:
            raise SystemExit(f"Could not read image: {image_path}")
        mask = align_mask_to_image(mask, image.shape[:2], image_id)

        values, counts = np.unique(mask, return_counts=True)
        unique_values.update(int(value) for value in values)
        if per_class_pixels:
            pixel_counts.update({int(value): int(count) for value, count in zip(values, counts, strict=True)})
        checked += 1

    return {
        "checked": checked,
        "missing_files": missing_files,
        "unique_values": unique_values,
        "pixel_counts": pixel_counts,
    }


if __name__ == "__main__":
    main()
