"""Minimal Prato do Dia computer vision scaffold."""

from prato_do_dia_ml.annotations import write_yolo_segmentation_txt
from prato_do_dia_ml.detector import YoloOnnxDetector
from prato_do_dia_ml.metrics import intersection_over_union, rasterize_yolo_polygons
from prato_do_dia_ml.pipeline import FoodSegmentationPipeline
from prato_do_dia_ml.preprocessing import LetterboxResult, letterbox_image, normalize_bgr_to_rgb
from prato_do_dia_ml.schema import Detection, PipelineResult, SegmentationMask
from prato_do_dia_ml.segmenter import SamOnnxSegmenter
from prato_do_dia_ml.visualizer import overlay_yolo_polygons

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
