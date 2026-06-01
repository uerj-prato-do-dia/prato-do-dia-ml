"""Render previews for converted UECFoodPixComplete YOLO segmentation labels."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

PALETTE = (
    (0, 0, 255),
    (0, 180, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
    (255, 255, 0),
    (80, 80, 255),
    (80, 255, 80),
    (255, 80, 80),
    (40, 160, 220),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/uecfoodpixcomplete/outputs/previews"))
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = _read_names(args.converted_root / "data.yaml")
    image_paths = sorted((args.converted_root / "images" / args.split).glob("*.jpg"))
    if args.random:
        rng = random.Random(args.seed)
        rng.shuffle(image_paths)
    image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No images found for split {args.split} in {args.converted_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        label_path = args.converted_root / "labels" / args.split / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read image: {image_path}")
        preview = image.copy()
        if label_path.exists():
            _draw_label_file(preview, label_path, names)
        output_path = args.output_dir / args.split / f"{image_path.stem}_preview.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), preview):
            raise SystemExit(f"Failed to write preview: {output_path}")
    print(f"saved {len(image_paths)} previews to {args.output_dir / args.split}")


def _draw_label_file(image: np.ndarray, label_path: Path, names: dict[int, str]) -> None:
    height, width = image.shape[:2]
    fill = image.copy()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
        if len(coords) < 6 or len(coords) % 2:
            raise SystemExit(f"Invalid YOLO polygon at {label_path}:{line_number}")
        points = np.array(
            [
                [
                    int(round(np.clip(x, 0.0, 1.0) * (width - 1))),
                    int(round(np.clip(y, 0.0, 1.0) * (height - 1))),
                ]
                for x, y in zip(coords[0::2], coords[1::2], strict=True)
            ],
            dtype=np.int32,
        )
        color = PALETTE[class_id % len(PALETTE)]
        cv2.fillPoly(fill, [points], color)
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
        x, y, _, _ = cv2.boundingRect(points)
        cv2.putText(
            image,
            names.get(class_id, str(class_id)),
            (x, max(y - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    blended = cv2.addWeighted(fill, 0.35, image, 0.65, 0.0)
    image[:, :] = blended


def _read_names(data_yaml: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    in_names = False
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip() == "names:":
            in_names = True
            continue
        if not in_names or not line.startswith("  "):
            continue
        key, value = line.strip().split(":", maxsplit=1)
        names[int(key)] = value.strip()
    return names


if __name__ == "__main__":
    main()
