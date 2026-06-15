# Threshold sweep

The baseline has high precision and low recall. It usually avoids many random
food predictions, but it misses many true food regions. Before fine-tuning, run
a small threshold sweep to see whether detector confidence and NMS settings can
increase recall without destroying precision.

## Run

Quick subset:

```bash
uv run python scripts/run_threshold_sweep.py --limit 3
```

Custom sweep:

```bash
uv run python scripts/run_threshold_sweep.py \
  --conf-values 0.05,0.10,0.15 \
  --nms-values 0.35,0.45
```

Full default sweep:

```bash
uv run python scripts/run_threshold_sweep.py
```

Default values:

- YOLO confidence: `0.05`, `0.08`, `0.10`, `0.15`, `0.20`
- NMS IoU: `0.35`, `0.45`, `0.55`

Total default configs: 15.

## Outputs

Generated outputs are ignored by Git:

```text
outputs/experiments/threshold_sweep_v1/
  sweep_results.csv
  best_by_foreground_iou.json
  best_by_instance_iou.json
  best_by_recall.json
```

Each config also writes intermediate predictions, masks, overlays, reports, and
a small config snapshot under its own subdirectory.

## Reading results

`sweep_results.csv` has one row per threshold configuration.

Important columns:

- `foreground_iou`: food/background overlap.
- `instance_iou`: matched instance overlap.
- `precision`: matched predictions divided by predicted instances.
- `recall`: matched ground-truth instances divided by annotated instances.
- `false_positives`: total unmatched predictions.
- `missed_instances`: total unmatched ground-truth instances.

For the current failure pattern, prioritize recall improvement while watching
precision and false positives. A useful next config should increase recall
meaningfully without causing broad plate/background detections to dominate.

## Decision

If no threshold setting materially improves recall, the next model-quality work
should be dataset expansion and food-domain fine-tuning rather than more
threshold tuning.
