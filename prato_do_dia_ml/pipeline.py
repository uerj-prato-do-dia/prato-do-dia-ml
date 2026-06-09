"""Pipeline for YOLOv11 detection followed by SAM 2 segmentation."""

from __future__ import annotations

from pathlib import Path

import cv2

from prato_do_dia_ml.annotations import write_mask_png, write_metadata_json, write_yolo_segmentation_txt
from prato_do_dia_ml.config import PipelineConfig
from prato_do_dia_ml.detector import YoloOnnxDetector
from prato_do_dia_ml.io_utils import load_image_bgr
from prato_do_dia_ml.postprocessing import mask_to_class_image, mask_to_instance_image, postprocess_segmentations
from prato_do_dia_ml.schema import PipelineResult
from prato_do_dia_ml.segmenter import SamOnnxSegmenter
from prato_do_dia_ml.visualizer import overlay_yolo_polygons


class FoodSegmentationPipeline:
    """Orchestrates preprocessing, detection, segmentation, and TXT output."""

    def __init__(
        self,
        detector: YoloOnnxDetector,
        segmenter: SamOnnxSegmenter,
        output_dir: str | Path = "data/raw_segmentations",
        mask_dir: str | Path = "data/masks",
        overlay_dir: str | Path = "data/overlays",
        report_dir: str | Path = "data/reports",
        background_rgb: tuple[int, int, int] = (0, 0, 0),
        allow_alpha_input: bool = True,
        min_mask_area_ratio: float = 0.001,
        fill_holes_px: int = 64,
        remove_components_px: int = 64,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.output_dir = Path(output_dir)
        self.mask_dir = Path(mask_dir)
        self.overlay_dir = Path(overlay_dir)
        self.report_dir = Path(report_dir)
        self.background_rgb = background_rgb
        self.allow_alpha_input = allow_alpha_input
        self.min_mask_area_ratio = min_mask_area_ratio
        self.fill_holes_px = fill_holes_px
        self.remove_components_px = remove_components_px

    @classmethod
    def from_config(cls, config: PipelineConfig) -> FoodSegmentationPipeline:
        """Build the ONNX pipeline from typed config."""

        detector = YoloOnnxDetector(
            config.yolo.model_path,
            input_size=config.yolo.imgsz,
            confidence_threshold=config.yolo.confidence_threshold,
            iou_threshold=config.yolo.nms_iou_threshold,
            max_detections=config.yolo.max_detections,
        )
        segmenter = SamOnnxSegmenter(
            config.sam2.encoder_path,
            config.sam2.decoder_path,
            input_size=config.sam2.imgsz,
            mask_threshold=config.sam2.mask_threshold,
        )
        return cls(
            detector,
            segmenter,
            output_dir=config.paths.output_dir,
            mask_dir=config.paths.mask_dir,
            overlay_dir=config.paths.overlay_dir,
            report_dir=config.paths.report_dir,
            background_rgb=config.image.background_rgb,
            allow_alpha_input=config.image.allow_alpha_input,
            min_mask_area_ratio=config.postprocess.min_mask_area_ratio,
            fill_holes_px=config.postprocess.fill_holes_px,
            remove_components_px=config.postprocess.remove_components_px,
        )

    def run_image(self, image_path: str | Path) -> PipelineResult:
        """Run one image through the model pipeline."""

        path = Path(image_path)
        image_bgr = load_image_bgr(
            path,
            background_rgb=self.background_rgb,
            allow_alpha=self.allow_alpha_input,
        )

        detections = self.detector.detect(image_bgr)
        raw_segmentations = self.segmenter.segment(image_bgr, detections)
        segmentations = postprocess_segmentations(
            raw_segmentations,
            image_bgr.shape[:2],
            min_mask_area_ratio=self.min_mask_area_ratio,
            fill_holes_px=self.fill_holes_px,
            remove_components_px=self.remove_components_px,
        )
        annotation_path = self.output_dir / f"{path.stem}.txt"
        write_yolo_segmentation_txt(segmentations, annotation_path)

        instance_mask_path = self.mask_dir / f"{path.stem}_instances.png"
        class_mask_path = self.mask_dir / f"{path.stem}_class.png"
        metadata_path = self.report_dir / f"{path.stem}.json"
        overlay_path = self.overlay_dir / f"{path.stem}_overlay.jpg"

        instance_image = mask_to_instance_image(segmentations, image_bgr.shape[:2])
        class_image = mask_to_class_image(segmentations, image_bgr.shape[:2])
        write_mask_png(instance_image, instance_mask_path)
        write_mask_png(class_image, class_mask_path)
        write_metadata_json(
            image_path=path,
            width=image_bgr.shape[1],
            height=image_bgr.shape[0],
            model_versions={
                "yolo": str(self.detector.model_path),
                "sam_encoder": str(self.segmenter.encoder_path),
                "sam_decoder": str(self.segmenter.decoder_path),
            },
            segmentations=segmentations,
            output_path=metadata_path,
        )

        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay = overlay_yolo_polygons(image_bgr, annotation_path)
        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(f"failed to write overlay: {overlay_path}")

        return PipelineResult(
            image_path=path,
            annotation_path=annotation_path,
            instance_mask_path=instance_mask_path,
            class_mask_path=class_mask_path,
            metadata_path=metadata_path,
            overlay_path=overlay_path,
            detections=detections,
            segmentations=segmentations,
        )
