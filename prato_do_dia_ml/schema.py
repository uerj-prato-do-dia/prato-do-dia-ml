"""Typed data structures and response schemas for YOLOv11-seg inference."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

CLASS_ID_TO_NAME: dict[int, str] = {
    0: "tomate",
    1: "salada_verde",
    2: "feijao",
    3: "batata_frita",
    4: "arroz",
    5: "carne_moida",
    6: "pure_batata",
    7: "farofa",
    8: "cenoura",
    9: "ovo_frito",
    10: "massa_macarrao",
    11: "frango_grelhado",
    12: "azeitona",
    13: "batata_palha",
    14: "estrogonofe",
    15: "carne_bovina_bife",
}


def get_canonical_classes() -> list[str]:
    """Return ordered list of canonical food class names [index 0..N-1]."""
    return [CLASS_ID_TO_NAME[i] for i in sorted(CLASS_ID_TO_NAME.keys())]


def write_classes_txt(output_path: str | Path) -> Path:
    """Write canonical classes.txt file from Python Single Source of Truth."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = get_canonical_classes()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def sync_all_classes_txt(project_root: str | Path | None = None) -> list[Path]:
    """Synchronize all project classes.txt files against Python Single Source of Truth."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    root = Path(project_root)

    target_paths = [
        root / "prato-do-dia-ml" / "data" / "classes.txt",
        root / "data" / "processed" / "labels" / "classes.txt",
        root / "data" / "raw" / "classes.txt",
    ]

    updated: list[Path] = []
    for target in target_paths:
        updated.append(write_classes_txt(target))
    return updated


BBoxXYXY = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class Detection:
    """Single detector result in original image pixel coordinates."""

    class_id: int
    confidence: float
    bbox_xyxy: BBoxXYXY
    prompt_xyxy: BBoxXYXY | None = None


@dataclass(frozen=True)
class SegmentationMask:
    """Single segmentation result with a normalized YOLO polygon."""

    class_id: int
    confidence: float
    polygon: tuple[Point, ...]
    mask: np.ndarray | None = None
    instance_id: int = 0
    bbox_xyxy: BBoxXYXY | None = None
    yolo_confidence: float = 0.0
    sam_iou_prediction: float = 0.0
    area_px: int = 0

    def with_updates(self, **changes: object) -> SegmentationMask:
        return replace(self, **changes)


@dataclass(frozen=True)
class BoundingBox:
    """Normalized bounding box coordinates [x_min, y_min, x_max, y_max] in range [0.0, 1.0]."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def to_list(self) -> list[float]:
        return [
            round(self.x_min, 6),
            round(self.y_min, 6),
            round(self.x_max, 6),
            round(self.y_max, 6),
        ]


@dataclass(frozen=True)
class SegmentationItem:
    """Single segmented food item prediction."""

    class_id: int
    class_name: str
    confidence: float
    box: list[float]
    polygon: list[list[float]]
    relative_area_percentage: float

    @property
    def proposal_class_id(self) -> int:
        return self.class_id

    @property
    def label(self) -> str:
        return self.class_name

    @property
    def bbox(self) -> list[float]:
        return self.box

    @property
    def area_px(self) -> int:
        return int(self.relative_area_percentage * 100)

    @property
    def instance_id(self) -> int:
        return self.class_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "box": [round(float(v), 6) for v in self.box],
            "polygon": [[round(float(x), 6), round(float(y), 6)] for x, y in self.polygon],
            "relative_area_percentage": round(float(self.relative_area_percentage), 2),
        }


@dataclass(frozen=True)
class PlatePredictionResponse:
    """API-facing prediction response payload."""

    inference_time_ms: float
    plate_detected: bool
    items: list[SegmentationItem] = field(default_factory=list)

    @property
    def instances(self) -> list[SegmentationItem]:
        return self.items

    @property
    def detections(self) -> list[SegmentationItem]:
        return self.items

    @property
    def segmentations(self) -> list[SegmentationItem]:
        return self.items

    @property
    def artifacts(self) -> dict[str, Any]:
        return {}

    @property
    def image_width(self) -> int:
        return 640

    @property
    def image_height(self) -> int:
        return 640

    @property
    def model_info(self) -> ModelInfoRef:
        return ModelInfoRef(pipeline="yolo11_onnx_cpu", version="1.0.0")

    @classmethod
    def create(
        cls,
        inference_time_ms: float,
        items: list[SegmentationItem],
        min_area_percentage: float = 5.0,
    ) -> PlatePredictionResponse:
        """Construct response with deterministic plate_detected heuristic."""
        total_food_area = sum(item.relative_area_percentage for item in items)
        plate_detected = len(items) > 0 and total_food_area >= min_area_percentage
        return cls(
            inference_time_ms=round(float(inference_time_ms), 2),
            plate_detected=plate_detected,
            items=items,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "inference_time_ms": self.inference_time_ms,
            "plate_detected": self.plate_detected,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class ModelInfoRef:
    pipeline: str
    version: str
