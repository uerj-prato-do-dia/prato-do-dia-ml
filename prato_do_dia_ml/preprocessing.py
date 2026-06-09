"""Image preprocessing utilities for model input preparation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxResult:
    """Letterboxed image plus the transform metadata needed to map coordinates."""

    image: np.ndarray
    scale: float
    pad_left: int
    pad_top: int
    resized_width: int
    resized_height: int


def letterbox_image(
    image: np.ndarray,
    size: int | tuple[int, int] = 640,
    padding_value: int | Sequence[int] = 114,
    interpolation: int = cv2.INTER_LINEAR,
) -> LetterboxResult:
    """Resize an image with padding while preserving aspect ratio exactly.

    Args:
        image: Input image as HxW or HxWxC numpy array.
        size: Output size as an integer square size or ``(height, width)``.
        padding_value: Border value passed to OpenCV. Use BGR order for color images.
        interpolation: OpenCV interpolation flag used for the proportional resize.

    Returns:
        A ``LetterboxResult`` containing the padded image and transform metadata.
    """

    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D grayscale or 3D color array")

    source_height, source_width = image.shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise ValueError("image dimensions must be positive")

    target_height, target_width = _normalize_size(size)
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    pad_width = target_width - resized_width
    pad_height = target_height - resized_height
    if pad_width < 0 or pad_height < 0:
        raise RuntimeError("letterbox resize exceeded requested output size")

    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    letterboxed = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_value,
    )

    return LetterboxResult(
        image=letterboxed,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        resized_width=resized_width,
        resized_height=resized_height,
    )


def normalize_bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to RGB float32 values in the [0, 1] range."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape HxWx3")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32) / 255.0


def _normalize_size(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        target_height = target_width = size
    else:
        if len(size) != 2:
            raise ValueError("size must be an int or a (height, width) tuple")
        target_height, target_width = size

    if target_height <= 0 or target_width <= 0:
        raise ValueError("target size must be positive")

    return target_height, target_width
