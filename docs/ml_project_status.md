# ML project status

## Dataset

The current curated dataset has 8 annotated `baseline_eval` images with
ground-truth instance masks. These images are useful for fast regression checks
and experiment comparisons, but the set is still too small for final model
selection.

`imagem9` and `imagem10` are intentionally marked as bad/non-top-down examples.
They are useful for robustness or capture-quality discussion, but they should
not be treated as normal baseline or training images unless the project scope
changes.

New top-down phone images still need to be collected, audited, and annotated.

## Baselines

The historical baseline is the YOLO11 + SAM2 ONNX pipeline. Its original
8-image metrics were weak, especially recall:

- Mean foreground IoU: `0.129779`
- Mean instance IoU: `0.316586`
- Mean Dice: `0.212984`
- Mean precision: `0.950000`
- Mean recall: `0.284177`

Threshold tuning improved YOLO11 + SAM2 enough to keep it as a useful historical
reference:

- `configs/experiments/yolo11_sam2_balanced.toml`
- `configs/experiments/yolo11_sam2_recall.toml`

The strongest current experimental baseline is Ultralytics YOLOE zero-shot with
food prompts:

- `configs/experiments/yoloe26s_food_balanced.toml`
- `configs/experiments/yoloe26m_food_recall.toml`

## Current Conclusion

YOLOE is much better than the YOLO11 + SAM2 baseline for mask proposals and
pre-annotation on the current 8-image set. It gives stronger food foreground
coverage, better rice/beans separation, and much higher recall.

The class predictions are still unreliable and too generic for direct nutrition
estimation. Some similar or touching foods are merged, boundaries remain
imperfect, and Brazilian food categories are not represented well enough.

The project should not replace the production/default pipeline yet. Treat YOLOE
as the best current experiment path for annotation acceleration and further
evaluation.

## Next Steps

1. Collect 10-15 new top-down phone photos.
2. Pull the images from the Android collector flow.
3. Run the image-quality audit.
4. Annotate selected good images.
5. Re-run YOLOE and YOLO11 + SAM2 baseline comparisons.
6. Review whether a Label Studio pre-annotation workflow should use
   `yoloe26s_food_balanced` or `yoloe26m_food_recall`.
7. Consider fine-tuning only after enough annotated data exists.
