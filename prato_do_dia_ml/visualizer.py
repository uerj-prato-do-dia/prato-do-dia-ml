"""Visual overlays for generated YOLO segmentation polygons."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from prato_do_dia_ml.metrics import parse_yolo_polygon_line

OVERLAY_COLORS_BGR = (
    (0, 0, 255),
    (0, 180, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
    (255, 255, 0),
    (80, 80, 255),
    (80, 255, 80),
)


def overlay_yolo_polygons(
    image_bgr: np.ndarray,
    txt_path: str | Path,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay YOLO TXT polygons and bounding boxes on an OpenCV BGR image."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape HxWx3")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"YOLO annotation not found: {path}")

    output = image_bgr.copy()
    fill_layer = output.copy()
    height, width = image_bgr.shape[:2]

    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue

        class_id, points = parse_yolo_polygon_line(line, index + 1)
        color = OVERLAY_COLORS_BGR[index % len(OVERLAY_COLORS_BGR)]
        pixel_points = np.array(
            [
                [
                    int(round(np.clip(x, 0.0, 1.0) * (width - 1))),
                    int(round(np.clip(y, 0.0, 1.0) * (height - 1))),
                ]
                for x, y in points
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(fill_layer, [pixel_points], color)
        cv2.polylines(output, [pixel_points], isClosed=True, color=color, thickness=2)
        x, y, box_width, box_height = cv2.boundingRect(pixel_points)
        cv2.rectangle(output, (x, y), (x + box_width, y + box_height), color, 2)
        cv2.putText(
            output,
            str(class_id),
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return cv2.addWeighted(fill_layer, alpha, output, 1.0 - alpha, 0.0)
