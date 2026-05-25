"""Smoke test for the preprocessing scaffold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import letterbox_image, normalize_bgr_to_rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/test_rect_input.png"))
    parser.add_argument("--output", type=Path, default=Path("data/test_letterbox_output.png"))
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    args.input.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        image = np.zeros((320, 800, 3), dtype=np.uint8)
        image[:, :400] = (0, 0, 255)
        image[:, 400:] = (0, 255, 0)
        cv2.imwrite(str(args.input), image)

    image_bgr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"could not load image: {args.input}")

    result = letterbox_image(image_bgr, size=args.size)
    normalized = normalize_bgr_to_rgb(result.image)

    expected_shape = (args.size, args.size, 3)
    if result.image.shape != expected_shape:
        raise AssertionError(f"expected {expected_shape}, got {result.image.shape}")
    if normalized.dtype != np.float32 or normalized.min() < 0.0 or normalized.max() > 1.0:
        raise AssertionError("normalization must return float32 RGB values in [0, 1]")

    original_ratio = image_bgr.shape[1] / image_bgr.shape[0]
    resized_ratio = result.resized_width / result.resized_height
    if not np.isclose(original_ratio, resized_ratio, rtol=0.01):
        raise AssertionError("letterbox resize changed the image aspect ratio")

    cv2.imwrite(str(args.output), result.image)
    print(
        f"saved {args.output} "
        f"shape={result.image.shape} scale={result.scale:.6f} "
        f"pad_left={result.pad_left} pad_top={result.pad_top}"
    )


if __name__ == "__main__":
    main()
