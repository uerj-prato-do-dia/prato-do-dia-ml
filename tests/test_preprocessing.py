from __future__ import annotations

import numpy as np

from src.preprocessing import letterbox_image, normalize_bgr_to_rgb


def test_letterbox_preserves_aspect_ratio() -> None:
    image_bgr = np.zeros((320, 800, 3), dtype=np.uint8)
    image_bgr[:, :400] = (0, 0, 255)
    image_bgr[:, 400:] = (0, 255, 0)

    result = letterbox_image(image_bgr, size=640)
    normalized = normalize_bgr_to_rgb(result.image)

    assert result.image.shape == (640, 640, 3)
    assert normalized.dtype == np.float32
    assert float(normalized.min()) >= 0.0
    assert float(normalized.max()) <= 1.0

    original_ratio = image_bgr.shape[1] / image_bgr.shape[0]
    resized_ratio = result.resized_width / result.resized_height
    assert np.isclose(original_ratio, resized_ratio, rtol=0.01)
    assert result.pad_top == 192
    assert result.pad_left == 0


def test_normalize_converts_bgr_to_rgb() -> None:
    image_bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)

    normalized = normalize_bgr_to_rgb(image_bgr)

    np.testing.assert_allclose(normalized[0, 0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
