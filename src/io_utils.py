"""Image loading and validation helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".heic")


def load_image_bgr(
    path: str | Path,
    background_rgb: tuple[int, int, int] = (0, 0, 0),
    allow_alpha: bool = True,
) -> np.ndarray:
    """Load an image as BGR uint8, replacing transparent pixels if present."""

    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"could not load image: {image_path}")
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8: {image_path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError(f"unsupported image shape for {image_path}: {image.shape}")
    if image.shape[2] == 3:
        return image
    if image.shape[2] != 4:
        raise ValueError(f"unsupported channel count for {image_path}: {image.shape[2]}")
    if not allow_alpha:
        raise ValueError(f"alpha channel is not allowed: {image_path}")

    bgr = image[:, :, :3].copy()
    alpha = image[:, :, 3]
    background_bgr = np.array(background_rgb[::-1], dtype=np.uint8)
    bgr[alpha == 0] = background_bgr
    return bgr


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
