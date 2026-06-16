# Ultralytics zero-shot v1

This experiment compares downloaded Ultralytics segmentation models against the
8 annotated `baseline_eval` images in `data/dataset_manifest.csv`. It does not
train models and does not replace the project default YOLO11 + SAM2 ONNX
pipeline.

For the current 8-image results and selected YOLOE configs, see
`docs/ultralytics_zero_shot_results_v1.md`.

Generated outputs go under:

```text
outputs/experiments/ultralytics_zero_shot_v1/
```

The `outputs/` directory is local experiment output and should not be committed
by default.

## Expected model files

The script expects these files:

```text
external_models/ultralytics/mobile_sam.pt
external_models/ultralytics/yolo11m-seg.pt
external_models/ultralytics/yolo11l-seg.pt
external_models/ultralytics/yolo11x-seg.pt
external_models/ultralytics/yoloe-26s-seg.pt
external_models/ultralytics/yoloe-26m-seg.pt
```

If Ultralytics is not installed in the local environment:

```bash
uv pip install -U ultralytics
```

## Models

YOLO11 segmentation models are closed-vocabulary models with COCO-like class
names. They can still be useful as a sanity check, but likely failure modes are
wrong generic classes such as `bowl`, `cup`, `dining table`, or missed
food-specific regions.

YOLOE segmentation models are more relevant for this experiment because they
support food text prompts. The script uses:

```text
rice, beans, meat, chicken, salad, tomato, pasta, potato, egg,
french fries, vegetables, beef, pork, fish
```

MobileSAM is not treated as directly comparable by default. It is primarily a
prompted segmentation model and is more useful as a second stage after detector
boxes. The script can run a one-image CPU smoke check to confirm the downloaded
checkpoint loads.

## Quick smoke

Run one baseline image at one confidence/IoU setting:

```bash
uv run python scripts/run_ultralytics_zero_shot.py \
  --limit 1 \
  --conf-values 0.05 \
  --iou-values 0.45 \
  --include-yolo11 \
  --include-yoloe \
  --include-mobilesam-smoke
```

## Full 8-image comparison

Run the CPU-only YOLOE sweep:

```bash
uv run python scripts/run_ultralytics_zero_shot.py \
  --limit 8 \
  --conf-values 0.01,0.03,0.05,0.08,0.10 \
  --iou-values 0.30,0.45,0.60 \
  --include-yoloe \
  --device cpu
```

Run the broader CPU-only zero-shot comparison:

```bash
uv run python scripts/run_ultralytics_zero_shot.py \
  --limit 8 \
  --conf-values 0.01,0.03,0.05 \
  --iou-values 0.30,0.45,0.60 \
  --include-yolo11 \
  --include-yoloe \
  --include-mobilesam-smoke \
  --device cpu
```

This runs 45 comparable model/config combinations:

- 3 YOLO11 segmentation checkpoints.
- 2 YOLOE segmentation checkpoints.
- 3 confidence thresholds.
- 3 NMS IoU thresholds.

MobileSAM smoke is reported separately and is not included in the aggregate
metric ranking.

## Selected YOLOE configs

After reviewing the full zero-shot results, the two selected YOLOE configs were
captured as reproducible TOML descriptors:

```bash
uv run python scripts/run_ultralytics_zero_shot.py \
  --config configs/experiments/yoloe26s_food_balanced.toml \
  --output-dir outputs/experiments/ultralytics_zero_shot_v1/yoloe26s_food_balanced

uv run python scripts/run_ultralytics_zero_shot.py \
  --config configs/experiments/yoloe26m_food_recall.toml \
  --output-dir outputs/experiments/ultralytics_zero_shot_v1/yoloe26m_food_recall
```

The first config is the preferred balanced zero-shot baseline candidate. The
second config is useful for aggressive pre-annotation.

For a faster run, keep all models but use one NMS IoU:

```bash
uv run python scripts/run_ultralytics_zero_shot.py \
  --limit 8 \
  --conf-values 0.01,0.03,0.05 \
  --iou-values 0.45 \
  --include-yolo11 \
  --include-yoloe
```

## Outputs

```text
outputs/experiments/ultralytics_zero_shot_v1/
  summary.csv
  summary.json
  metrics_by_image.csv
  failures.csv
  runs/
    yolo11m_seg_conf_0p01_iou_0p45/
      predictions/
      overlays/
      metrics_by_image.csv
    yoloe26s_seg_food_conf_0p01_iou_0p45/
      predictions/
      overlays/
      metrics_by_image.csv
```

Each prediction JSON preserves predicted class IDs, class names, confidence, and
instance IDs. This is important for diagnosing whether closed-vocabulary models
are detecting generic objects instead of food regions.

## Metrics

The script converts Ultralytics segmentation polygons into instance-ID masks and
reuses the project metrics:

- `foreground_iou`
- `instance_iou`
- `dice`
- `boundary_f_score`
- `precision`
- `recall`
- `false_positives`
- `missed_instances`

Class labels are preserved for analysis, but evaluation is based on foreground
and object-level masks because the current ground truth is an instance mask, not
a directly comparable semantic class map.

## Comparing with selected YOLO11 + SAM2 configs

Compare zero-shot outputs against:

```text
outputs/experiments/config_comparison_v1/recall/
outputs/experiments/config_comparison_v1/balanced/
```

Use `summary.csv` for aggregate ranking and `metrics_by_image.csv` plus overlays
to inspect per-image failures. The most useful short list is:

- Best `foreground_iou`.
- Best `instance_iou`.
- Best `recall`.
- Best recall where `precision >= 0.75`.

These results are still based on only 8 images. Treat them as model-selection
evidence for the next experiment, not as final model quality.
