"""Image loading and validation helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".heic")


def load_image_bgr(
    path: str | Path,
    background_rgb: tuple[int, int, int] = (0, 0, 0),
    allow_alpha: bool = False,
) -> np.ndarray:
    """Load an image as BGR uint8 with EXIF orientation correction and strict 3-channel conversion."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"could not load image: {image_path}")

    try:
        with Image.open(image_path) as img:
            transposed = ImageOps.exif_transpose(img)
            rgb = transposed.convert("RGB")
            bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
            return bgr
    except Exception as exc:
        raise ValueError(f"could not load image {image_path}: {exc}") from exc


def input_images(input_dir: str | Path) -> list[Path]:
    """Return supported input images in deterministic order."""

    directory = Path(input_dir)
    paths: list[Path] = []
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        paths.extend(directory.glob(f"*{suffix}"))
        paths.extend(directory.glob(f"*{suffix.upper()}"))
    return sorted(set(paths))


def validate_instance_mask_png(path: str | Path) -> np.ndarray:
    """Load a single-channel PNG instance mask and reject lossy/RGB labels."""

    mask_path = Path(path)
    if mask_path.suffix.lower() != ".png":
        raise ValueError(f"ground truth must be a single-channel PNG: {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"could not load ground-truth mask: {mask_path}")
    if mask.ndim != 2:
        raise ValueError(f"ground truth must be single-channel, got shape {mask.shape}: {mask_path}")
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"ground truth mask must use integer IDs: {mask_path}")
    if np.min(mask) < 0:
        raise ValueError(f"ground truth mask contains negative IDs: {mask_path}")
    return mask.astype(np.int32, copy=False)
