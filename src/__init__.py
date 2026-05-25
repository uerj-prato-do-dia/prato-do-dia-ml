"""Minimal Prato do Dia computer vision scaffold."""

from src.annotations import write_yolo_segmentation_txt
from src.detector import YoloOnnxDetector
from src.metrics import intersection_over_union, rasterize_yolo_polygons
from src.pipeline import FoodSegmentationPipeline
from src.preprocessing import LetterboxResult, letterbox_image, normalize_bgr_to_rgb
from src.schema import Detection, PipelineResult, SegmentationMask
from src.segmenter import SamOnnxSegmenter
from src.visualizer import overlay_yolo_polygons

__all__ = [
    "Detection",
    "FoodSegmentationPipeline",
    "LetterboxResult",
    "PipelineResult",
    "SamOnnxSegmenter",
    "SegmentationMask",
    "YoloOnnxDetector",
    "intersection_over_union",
    "letterbox_image",
    "normalize_bgr_to_rgb",
    "overlay_yolo_polygons",
    "rasterize_yolo_polygons",
    "write_yolo_segmentation_txt",
]
