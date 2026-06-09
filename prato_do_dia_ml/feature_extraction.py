"""Per-instance feature extraction from generated segmentation masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage import color, feature, measure


def extract_instance_features(
    image_bgr: np.ndarray,
    instance_mask: np.ndarray,
    image_name: str,
) -> pd.DataFrame:
    """Return one feature row per non-zero instance ID."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape HxWx3")
    if instance_mask.shape != image_bgr.shape[:2]:
        raise ValueError("instance mask shape must match image")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_lab = color.rgb2lab(image_rgb)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, P=8, R=1, method="uniform")
    height, width = instance_mask.shape
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)

    rows: list[dict[str, float | int | str]] = []
    for instance_id in [int(value) for value in np.unique(instance_mask) if value != 0]:
        mask = instance_mask == instance_id
        props = measure.regionprops(mask.astype(np.uint8))[0]
        rgb_pixels = image_rgb[mask]
        hsv_pixels = image_hsv[mask]
        lab_pixels = image_lab[mask]
        centroid_y, centroid_x = props.centroid
        centroid = np.array([centroid_x, centroid_y], dtype=np.float32)
        lbp_hist, _ = np.histogram(lbp[mask], bins=np.arange(0, 11), density=True)
        min_row, min_col, max_row, max_col = props.bbox
        bbox_height = max(max_row - min_row, 1)
        bbox_width = max(max_col - min_col, 1)
        gray_crop = gray[min_row:max_row, min_col:max_col].copy()
        mask_crop = mask[min_row:max_row, min_col:max_col]
        gray_crop[~mask_crop] = 0
        glcm = feature.graycomatrix(gray_crop, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        convex_area = props.area_convex if hasattr(props, "area_convex") else props.convex_area

        row: dict[str, float | int | str] = {
            "image": image_name,
            "instance_id": instance_id,
            "area_px": int(props.area),
            "perimeter_px": float(props.perimeter),
            "convex_area_px": int(convex_area),
            "circularity": float(4.0 * np.pi * props.area / max(props.perimeter**2, 1e-6)),
            "eccentricity": float(props.eccentricity),
            "aspect_ratio": float(bbox_width / bbox_height),
            "solidity": float(props.solidity),
            "centroid_x": float(centroid_x),
            "centroid_y": float(centroid_y),
            "center_distance_norm": float(np.linalg.norm(centroid - center) / max(width, height)),
            "relative_area": float(props.area / (height * width)),
            "glcm_contrast": float(feature.graycoprops(glcm, "contrast")[0, 0]),
            "glcm_homogeneity": float(feature.graycoprops(glcm, "homogeneity")[0, 0]),
            "glcm_energy": float(feature.graycoprops(glcm, "energy")[0, 0]),
            "glcm_correlation": float(feature.graycoprops(glcm, "correlation")[0, 0]),
        }
        for channel, name in enumerate(("r", "g", "b")):
            row[f"rgb_{name}_mean"] = float(rgb_pixels[:, channel].mean())
            row[f"rgb_{name}_std"] = float(rgb_pixels[:, channel].std())
        for channel, name in enumerate(("h", "s", "v")):
            row[f"hsv_{name}_mean"] = float(hsv_pixels[:, channel].mean())
            row[f"hsv_{name}_std"] = float(hsv_pixels[:, channel].std())
        for channel, name in enumerate(("l", "a", "b")):
            row[f"lab_{name}_mean"] = float(lab_pixels[:, channel].mean())
            row[f"lab_{name}_std"] = float(lab_pixels[:, channel].std())
        for index, value in enumerate(lbp_hist):
            row[f"lbp_{index}"] = float(value)
        rows.append(row)

    return pd.DataFrame(rows)


def save_features_csv(rows: list[pd.DataFrame], output_path: str | Path) -> Path:
    """Save concatenated feature rows to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    table.to_csv(path, index=False)
    return path
