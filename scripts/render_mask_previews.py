"""Render color previews for single-channel mask PNGs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


PALETTE_BGR = np.array(
    [
        (0, 0, 0),
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
        (180, 120, 40),
        (120, 60, 200),
        (40, 220, 180),
        (220, 220, 80),
        (180, 80, 120),
    ],
    dtype=np.uint8,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=Path, default=Path("data/masks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mask_previews"))
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--image-dir", type=Path, default=Path("data/input"))
    parser.add_argument("--overlay", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    mask_paths = sorted(args.mask_dir.glob(args.pattern))
    if not mask_paths:
        raise FileNotFoundError(f"no masks found in {args.mask_dir} with pattern {args.pattern}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"could not load mask: {mask_path}")
        if mask.ndim != 2:
            raise ValueError(f"expected single-channel mask, got shape {mask.shape}: {mask_path}")

        preview = colorize_mask(mask)
        if args.overlay:
            image_path = _matching_image_path(args.image_dir, mask_path.stem)
            if image_path is not None:
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is not None and image_bgr.shape[:2] == mask.shape:
                    preview = blend_on_image(image_bgr, preview, mask, args.alpha)

        output_path = args.output_dir / f"{mask_path.stem}_preview.png"
        if not cv2.imwrite(str(output_path), preview):
            raise RuntimeError(f"failed to write preview: {output_path}")
        rendered.append(str(output_path))

    print(f"saved {len(rendered)} previews to {args.output_dir}")


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map integer IDs to stable BGR colors."""

    ids = mask.astype(np.int64, copy=False)
    preview = PALETTE_BGR[ids % len(PALETTE_BGR)]
    preview[ids == 0] = (0, 0, 0)
    return preview.astype(np.uint8, copy=False)


def blend_on_image(image_bgr: np.ndarray, preview_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay colorized mask pixels on the source image."""

    output = image_bgr.copy()
    foreground = mask != 0
    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, preview_bgr, alpha, 0.0)
    output[foreground] = blended[foreground]
    return output


def _matching_image_path(image_dir: Path, mask_stem: str) -> Path | None:
    image_stem = mask_stem.removesuffix("_instances").removesuffix("_classes").removesuffix("_class")
    for suffix in (".png", ".jpg", ".jpeg"):
        path = image_dir / f"{image_stem}{suffix}"
        if path.exists():
            return path
    return None


if __name__ == "__main__":
    main()
