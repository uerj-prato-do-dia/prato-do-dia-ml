"""Convert UECFoodPixComplete masks to YOLO segmentation format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from validate_uec_dataset import align_mask_to_image, read_categories, read_ids, read_red_mask

PRATO_RELEVANT_IDS = [1, 27, 31, 34, 40, 55, 60, 61, 63, 68, 80, 86, 87, 98]
OPTIONAL_OTHERS_ID = 101


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uec-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category-file", type=Path)
    parser.add_argument("--train-list", type=Path)
    parser.add_argument("--test-list", type=Path)
    parser.add_argument("--mode", choices=("full", "prato_relevant", "binary_food"), default="prato_relevant")
    parser.add_argument("--include-others", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--symlink-images", action="store_true")
    parser.add_argument("--min-area-pixels", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=1.5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.copy_images and args.symlink_images:
        raise SystemExit("Choose only one: --copy-images or --symlink-images")
    if args.min_area_pixels < 0:
        raise SystemExit("--min-area-pixels must be non-negative")
    if args.epsilon < 0:
        raise SystemExit("--epsilon must be non-negative")

    report = convert_dataset(
        uec_root=args.uec_root,
        output=args.output,
        category_file=args.category_file,
        train_list=args.train_list,
        test_list=args.test_list,
        mode=args.mode,
        include_others=args.include_others,
        copy_images=args.copy_images,
        symlink_images=args.symlink_images or not args.copy_images,
        min_area_pixels=args.min_area_pixels,
        epsilon=args.epsilon,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2))


def convert_dataset(
    *,
    uec_root: Path,
    output: Path,
    category_file: Path | None,
    train_list: Path | None,
    test_list: Path | None,
    mode: str,
    include_others: bool,
    copy_images: bool,
    symlink_images: bool,
    min_area_pixels: int,
    epsilon: float,
    limit: int | None,
) -> dict[str, object]:
    from validate_uec_dataset import _find_first

    category_file = category_file or _find_first(uec_root, ("category.txt", "../category.txt"))
    train_list = train_list or _find_first(uec_root, ("train.txt", "train9000.txt", "../train.txt", "../train9000.txt"))
    test_list = test_list or _find_first(uec_root, ("test.txt", "test1000.txt", "../test.txt", "../test1000.txt"))
    categories = read_categories(category_file)
    selected_original_ids = _selected_class_ids(categories, mode, include_others)
    original_to_yolo = {original_id: yolo_id for yolo_id, original_id in enumerate(selected_original_ids)}
    names = ["food"] if mode == "binary_food" else [categories[original_id] for original_id in selected_original_ids]

    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_reports = {}
    split_reports["train"] = _convert_split(
        uec_root=uec_root,
        source_split="train",
        yolo_split="train",
        ids=read_ids(train_list)[:limit],
        output=output,
        original_to_yolo=original_to_yolo,
        binary_food=mode == "binary_food",
        copy_images=copy_images,
        symlink_images=symlink_images,
        min_area_pixels=min_area_pixels,
        epsilon=epsilon,
    )
    split_reports["val"] = _convert_split(
        uec_root=uec_root,
        source_split="test",
        yolo_split="val",
        ids=read_ids(test_list)[:limit],
        output=output,
        original_to_yolo=original_to_yolo,
        binary_food=mode == "binary_food",
        copy_images=copy_images,
        symlink_images=symlink_images,
        min_area_pixels=min_area_pixels,
        epsilon=epsilon,
    )

    data_yaml = output / "data.yaml"
    data_yaml.write_text(_data_yaml(output, names), encoding="utf-8")

    categories_used = {
        "mode": mode,
        "include_others": include_others,
        "classes": [
            {"uec_id": original_id, "yolo_id": yolo_id, "name": categories[original_id]}
            for original_id, yolo_id in original_to_yolo.items()
        ],
        "yolo_names": names,
    }
    (output / "categories_used.json").write_text(json.dumps(categories_used, indent=2), encoding="utf-8")

    report = {
        "uec_root": str(uec_root),
        "category_file": str(category_file),
        "train_list": str(train_list),
        "test_list": str(test_list),
        "output": str(output),
        "mode": mode,
        "include_others": include_others,
        "min_area_pixels": min_area_pixels,
        "epsilon": epsilon,
        "limit": limit,
        "class_count": len(names),
        "splits": split_reports,
    }
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _convert_split(
    *,
    uec_root: Path,
    source_split: str,
    yolo_split: str,
    ids: list[str],
    output: Path,
    original_to_yolo: dict[int, int],
    binary_food: bool,
    copy_images: bool,
    symlink_images: bool,
    min_area_pixels: int,
    epsilon: float,
) -> dict[str, int]:
    images_converted = 0
    label_lines = 0
    skipped_empty = 0

    for image_id in ids:
        source_image = uec_root / source_split / "img" / f"{image_id}.jpg"
        source_mask = uec_root / source_split / "mask" / f"{image_id}.png"
        if not source_image.is_file() or not source_mask.is_file():
            raise SystemExit(f"Missing source image/mask for {source_split} id={image_id}")

        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        mask = read_red_mask(source_mask)
        if image is None:
            raise SystemExit(f"Could not read image: {source_image}")
        height, width = image.shape[:2]
        mask = align_mask_to_image(mask, (height, width), image_id)

        output_image = output / "images" / yolo_split / f"{image_id}.jpg"
        _link_or_copy(source_image, output_image, copy_images=copy_images, symlink_images=symlink_images)

        lines = _mask_to_yolo_lines(mask, width, height, original_to_yolo, min_area_pixels, epsilon, binary_food)
        (output / "labels" / yolo_split / f"{image_id}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        images_converted += 1
        label_lines += len(lines)
        if not lines:
            skipped_empty += 1

    return {"images": images_converted, "label_lines": label_lines, "empty_label_files": skipped_empty}


def _mask_to_yolo_lines(
    mask: np.ndarray,
    width: int,
    height: int,
    original_to_yolo: dict[int, int],
    min_area_pixels: int,
    epsilon: float,
    binary_food: bool,
) -> list[str]:
    lines: list[str] = []
    if binary_food:
        return _binary_food_lines(mask, width, height, min_area_pixels, epsilon)

    for original_id in sorted(original_to_yolo):
        binary = (mask == original_id).astype(np.uint8)
        if not np.any(binary):
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_pixels:
                continue
            approximated = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0 else contour
            points = approximated.reshape(-1, 2)
            if len(points) < 3:
                continue
            values = [str(original_to_yolo[original_id])]
            for x, y in points:
                x_clipped = float(np.clip(x, 0, width - 1)) / max(width - 1, 1)
                y_clipped = float(np.clip(y, 0, height - 1)) / max(height - 1, 1)
                values.extend((f"{x_clipped:.6f}", f"{y_clipped:.6f}"))
            lines.append(" ".join(values))
    return lines


def _binary_food_lines(
    mask: np.ndarray,
    width: int,
    height: int,
    min_area_pixels: int,
    epsilon: float,
) -> list[str]:
    binary = (mask != 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_pixels:
            continue
        approximated = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0 else contour
        points = approximated.reshape(-1, 2)
        if len(points) < 3:
            continue
        values = ["0"]
        for x, y in points:
            x_clipped = float(np.clip(x, 0, width - 1)) / max(width - 1, 1)
            y_clipped = float(np.clip(y, 0, height - 1)) / max(height - 1, 1)
            values.extend((f"{x_clipped:.6f}", f"{y_clipped:.6f}"))
        lines.append(" ".join(values))
    return lines


def _selected_class_ids(categories: dict[int, str], mode: str, include_others: bool) -> list[int]:
    if mode == "binary_food":
        return [class_id for class_id in sorted(categories) if class_id != 0]
    if mode == "full":
        return [class_id for class_id in sorted(categories) if class_id != 0]
    selected = list(PRATO_RELEVANT_IDS)
    if include_others:
        selected.append(OPTIONAL_OTHERS_ID)
    return [class_id for class_id in selected if class_id in categories]


def _link_or_copy(source: Path, destination: Path, *, copy_images: bool, symlink_images: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_images:
        shutil.copy2(source, destination)
        return
    if symlink_images:
        try:
            destination.symlink_to(source.resolve())
            return
        except OSError:
            shutil.copy2(source, destination)
            return
    shutil.copy2(source, destination)


def _data_yaml(output: Path, names: list[str]) -> str:
    lines = [
        f"path: {output.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for index, name in enumerate(names):
        lines.append(f"  {index}: {name}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
