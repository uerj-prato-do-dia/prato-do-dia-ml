"""Convert Label Studio brush exports into ground-truth PNG masks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import input_images

MASK_NAME_RE = re.compile(
    r"task-(?P<task>\d+)-annotation-\d+-by-\d+-label-(?P<label>.+)-(?P<index>\d+)\.png$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/input"))
    parser.add_argument("--brush-dir", type=Path, default=Path("data/annotation_exports/labelstudio/brush_masks"))
    parser.add_argument(
        "--coco-json",
        type=Path,
        default=Path("data/annotation_exports/labelstudio/result_coco.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--threshold", type=int, default=1)
    args = parser.parse_args()

    if args.threshold < 1 or args.threshold > 255:
        raise ValueError("--threshold must be in [1, 255]")

    class_name_to_id = _load_class_mapping(args.coco_json)
    task_to_image = _task_to_image_paths(args.input_dir)
    task_masks = _group_masks_by_task(args.brush_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    imported: list[dict[str, object]] = []
    for task_id in sorted(task_masks):
        image_path = task_to_image.get(task_id)
        if image_path is None:
            raise ValueError(f"task {task_id} has brush masks but no matching input image by order")

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"could not load input image: {image_path}")

        height, width = image_bgr.shape[:2]
        instance_mask = np.zeros((height, width), dtype=np.uint16)
        class_mask = np.zeros((height, width), dtype=np.uint16)
        instances: list[dict[str, object]] = []

        for instance_id, mask_info in enumerate(task_masks[task_id], start=1):
            mask = cv2.imread(str(mask_info.path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"could not load brush mask: {mask_info.path}")
            if mask.shape != (height, width):
                raise ValueError(
                    f"mask shape {mask.shape} does not match image shape {(height, width)}: {mask_info.path}"
                )

            binary = mask >= args.threshold
            if not np.any(binary):
                continue
            class_id = class_name_to_id[mask_info.label]
            instance_mask[binary] = instance_id
            class_mask[binary] = class_id
            instances.append(
                {
                    "instance_id": instance_id,
                    "class_id": class_id,
                    "label": mask_info.label,
                    "source": str(mask_info.path),
                    "area_px": int(binary.sum()),
                }
            )

        instance_path = args.output_dir / f"{image_path.stem}_instances.png"
        class_path = args.output_dir / f"{image_path.stem}_classes.png"
        metadata_path = args.output_dir / f"{image_path.stem}_labelstudio.json"
        _write_png(instance_path, instance_mask)
        _write_png(class_path, class_mask)
        metadata_path.write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "image": str(image_path),
                    "instance_mask": str(instance_path),
                    "class_mask": str(class_path),
                    "instances": instances,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        imported.append({"task_id": task_id, "image": image_path.name, "instances": len(instances)})

    class_map_path = args.output_dir / "class_map.json"
    class_map_path.write_text(json.dumps(class_name_to_id, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"imported": imported, "class_map": str(class_map_path)}, indent=2))


class MaskInfo:
    def __init__(self, path: Path, label: str, index: int) -> None:
        self.path = path
        self.label = label
        self.index = index


def _load_class_mapping(coco_json: Path) -> dict[str, int]:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    mapping = {str(category["name"]): int(category["id"]) for category in data.get("categories", [])}
    if not mapping:
        raise ValueError(f"no categories found in {coco_json}")
    return mapping


def _task_to_image_paths(input_dir: Path) -> dict[int, Path]:
    paths = sorted(input_images(input_dir), key=_natural_image_key)
    return {index: path for index, path in enumerate(paths, start=1)}


def _natural_image_key(path: Path) -> tuple[str, int]:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        return path.stem, -1
    return path.stem[: match.start()], int(match.group(1))


def _group_masks_by_task(brush_dir: Path) -> dict[int, list[MaskInfo]]:
    grouped: dict[int, list[MaskInfo]] = defaultdict(list)
    for path in sorted(brush_dir.glob("*.png")):
        match = MASK_NAME_RE.match(path.name)
        if not match:
            raise ValueError(f"unexpected Label Studio mask filename: {path.name}")
        task_id = int(match.group("task"))
        grouped[task_id].append(
            MaskInfo(
                path=path,
                label=match.group("label"),
                index=int(match.group("index")),
            )
        )

    for task_id, masks in grouped.items():
        masks.sort(key=lambda item: (item.label, item.index, item.path.name))
    return dict(grouped)


def _write_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"failed to write PNG: {path}")


if __name__ == "__main__":
    main()
