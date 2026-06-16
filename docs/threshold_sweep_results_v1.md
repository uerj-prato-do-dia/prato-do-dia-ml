# Threshold sweep results v1

The threshold sweep over the 8 annotated `baseline_eval` images found much
better operating points than the original baseline. These results are useful
for choosing short-term experiment configs, but they are based on a tiny dataset
and may overfit `baseline_v1`.

## Summary

| config | yolo_conf | nms_iou | foreground_iou | instance_iou | dice | precision | recall | false_positives | missed_instances |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original baseline | 0.15 | 0.45 | 0.129779 | 0.316586 | 0.212984 | 0.950000 | 0.284177 | - | - |
| recall: `conf_0p01_nms_0p60` | 0.01 | 0.60 | 0.407598 | 0.504995 | 0.550319 | 0.650151 | 0.849355 | 39 | 8 |
| foreground IoU: `conf_0p01_nms_0p45` | 0.01 | 0.45 | 0.420957 | 0.503965 | 0.563997 | 0.638420 | 0.815873 | 41 | 10 |
| balanced: `conf_0p03_nms_0p30` | 0.03 | 0.30 | 0.325164 | 0.416089 | 0.463572 | 0.795343 | 0.713442 | 17 | 14 |

## Selected configs

Two experiment configs are intentionally added without replacing the default
baseline config:

- `configs/experiments/yolo11_sam2_recall.toml`: `yolo_conf = 0.01`,
  `nms_iou = 0.60`.
- `configs/experiments/yolo11_sam2_balanced.toml`: `yolo_conf = 0.03`,
  `nms_iou = 0.30`.

The recall config is best for research and error analysis because it exposes
more candidate food regions and reduces missed detections. The tradeoff is a
large increase in false positives and lower precision, so it is not the safest
app/demo default.

The balanced config is better for app/demo testing because it keeps precision
above `0.75`, cuts false positives from 39-41 down to 17, and still improves
recall substantially compared with the original baseline.

## Tradeoff

The original baseline is too conservative: precision is high, but recall is low.
Lowering the YOLO confidence threshold greatly improves recall and segmentation
overlap, but it also admits more weak detections. Raising NMS IoU can preserve
more overlapping candidates, which helps recall, while stricter NMS and a
slightly higher confidence threshold reduce noisy detections.

For the next short-term work:

- Use the recall config for failure analysis, annotation review, and model
  debugging.
- Use the balanced config for app/demo testing until a larger annotated dataset
  supports a better decision.
- Do not treat either config as final model quality. The sweep used only 8
  images and may be tuned to this small baseline set.

## Rendering comparable overlays

The existing baseline report script already runs only `baseline_eval` images
with existing masks. Use it with explicit configs and output directories:

```bash
uv run python scripts/run_baseline_report.py \
  --config configs/experiments/yolo11_sam2_recall.toml \
  --output-dir outputs/experiments/config_comparison_v1/recall

uv run python scripts/run_baseline_report.py \
  --config configs/experiments/yolo11_sam2_balanced.toml \
  --output-dir outputs/experiments/config_comparison_v1/balanced
```

Expected generated structure:

```text
outputs/experiments/config_comparison_v1/
  recall/
    overlays/
    metrics_by_image.csv
  balanced/
    overlays/
    metrics_by_image.csv
```

Generated files under `outputs/` are local experiment artifacts and should not
be committed by default.

To create a small comparison summary after both reports run:

```bash
python - <<'PY'
import csv
import json
from pathlib import Path

root = Path("outputs/experiments/config_comparison_v1")
rows = []
for label in ("recall", "balanced"):
    metrics = json.loads((root / label / "metrics.json").read_text())
    mean = metrics["mean_metrics"]
    rows.append({
        "config": label,
        "image_count": metrics["image_count"],
        "successful_images": metrics["successful_images"],
        "failed_images": metrics["failed_images"],
        "foreground_iou": mean["foreground_iou"],
        "instance_iou": mean["instance_iou"],
        "dice": mean["dice"],
        "precision": mean["precision"],
        "recall": mean["recall"],
    })

with (root / "comparison_summary.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
PY
```
