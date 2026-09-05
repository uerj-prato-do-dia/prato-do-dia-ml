"""Deterministic cleanup and overlap resolution for instance masks."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from prato_do_dia_ml.annotations import mask_to_polygon
from prato_do_dia_ml.schema import SegmentationMask


def remove_small_components(mask: np.ndarray, min_component_px: int = 64) -> np.ndarray:
    """Remove small connected components from binary boolean mask."""
    binary = mask.astype(bool)
    labels, count = ndimage.label(binary)
    kept = np.zeros(binary.shape, dtype=bool)
    for label_id in range(1, count + 1):
        component = labels == label_id
        if int(component.sum()) >= min_component_px:
            kept |= component
    return kept


def fill_small_holes(mask: np.ndarray, fill_holes_px: int = 64) -> np.ndarray:
    """Fill small holes in binary boolean mask."""
    kept = mask.astype(bool)
    if fill_holes_px > 0 and np.any(kept):
        inverse_labels, inverse_count = ndimage.label(~kept)
        border_labels = set(np.unique(inverse_labels[0, :]))
        border_labels.update(np.unique(inverse_labels[-1, :]))
        border_labels.update(np.unique(inverse_labels[:, 0]))
        border_labels.update(np.unique(inverse_labels[:, -1]))
        for label_id in range(1, inverse_count + 1):
            if label_id in border_labels:
                continue
            hole = inverse_labels == label_id
            if int(hole.sum()) <= fill_holes_px:
                kept |= hole
    return kept


def cleanup_mask(mask: np.ndarray, min_component_px: int = 64, fill_holes_px: int = 64) -> np.ndarray:
    """Remove small connected components and fill small holes."""
    kept = remove_small_components(mask, min_component_px)
    return fill_small_holes(kept, fill_holes_px)


def postprocess_segmentations(
    segmentations: tuple[SegmentationMask, ...],
    image_shape: tuple[int, int],
    min_mask_area_ratio: float = 0.001,
    fill_holes_px: int = 64,
    remove_components_px: int = 64,
) -> tuple[SegmentationMask, ...]:
    """Clean masks, remove tiny instances, and resolve overlapping pixels."""

    if not segmentations:
        return ()

    height, width = image_shape
    scale_factor = (height * width) / (640.0 * 640.0)
    scaled_fill_holes = max(0, int(round(fill_holes_px * scale_factor)))
    scaled_remove_components = max(0, int(round(remove_components_px * scale_factor)))

    min_area_px = max(1, int(round(height * width * min_mask_area_ratio)))
    cleaned: list[SegmentationMask] = []
    for segmentation in segmentations:
        if segmentation.mask is None:
            continue
        mask = cleanup_mask(
            segmentation.mask,
            min_component_px=max(min_area_px, scaled_remove_components),
            fill_holes_px=scaled_fill_holes,
        )
        area_px = int(mask.sum())
        if area_px < min_area_px:
            continue
        polygon = tuple(mask_to_polygon(mask, width, height))
        if len(polygon) < 3:
            continue
        cleaned.append(
            segmentation.with_updates(
                mask=mask,
                polygon=polygon,
                area_px=area_px,
            )
        )

    return resolve_overlaps(tuple(cleaned), image_shape)


def resolve_overlaps(
    segmentations: tuple[SegmentationMask, ...],
    image_shape: tuple[int, int],
) -> tuple[SegmentationMask, ...]:
    """Assign overlapping pixels to the strongest instance deterministically."""

    if not segmentations:
        return ()

    height, width = image_shape
    owner = np.full((height, width), -1, dtype=np.int32)
    ordered = sorted(
        enumerate(segmentations),
        key=lambda item: (
            item[1].sam_iou_prediction,
            item[1].yolo_confidence,
            item[1].area_px,
        ),
    )
    for index, segmentation in ordered:
        if segmentation.mask is None:
            continue
        owner[segmentation.mask.astype(bool)] = index

    resolved: list[SegmentationMask] = []
    for index, segmentation in enumerate(segmentations):
        mask = owner == index
        area_px = int(mask.sum())
        if area_px == 0:
            continue
        polygon = tuple(mask_to_polygon(mask, width, height))
        if len(polygon) < 3:
            continue
        resolved.append(
            segmentation.with_updates(
                instance_id=len(resolved) + 1,
                mask=mask,
                polygon=polygon,
                area_px=area_px,
            )
        )

    return tuple(resolved)


def mask_to_instance_image(segmentations: tuple[SegmentationMask, ...], image_shape: tuple[int, int]) -> np.ndarray:
    """Render processed masks to a single-channel instance-ID image."""

    output = np.zeros(image_shape, dtype=np.uint16)
    for segmentation in segmentations:
        if segmentation.mask is None:
            continue
        output[segmentation.mask.astype(bool)] = segmentation.instance_id
    return output


def mask_to_class_image(segmentations: tuple[SegmentationMask, ...], image_shape: tuple[int, int]) -> np.ndarray:
    """Render processed masks to a single-channel class-ID image."""

    output = np.zeros(image_shape, dtype=np.uint16)
    for segmentation in segmentations:
        if segmentation.mask is None:
            continue
        output[segmentation.mask.astype(bool)] = segmentation.class_id + 1
    return output


def smooth_polygon(mask: np.ndarray, width: int, height: int) -> tuple[tuple[float, float], ...]:
    """Return a contour-smoothed polygon without changing the metric mask."""

    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ()
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(contour, True)
    points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return tuple((float(x / max(width - 1, 1)), float(y / max(height - 1, 1))) for x, y in points)
