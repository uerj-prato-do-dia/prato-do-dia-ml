# Baseline evaluation

The baseline report measures the current YOLO11 + SAM2 ONNX pipeline on images
listed in `data/dataset_manifest.csv`.

This is a reproducibility workflow, not a claim of final model quality. The
current dataset is too small for broad conclusions.

## Run

Create or refresh the manifest first:

```bash
uv run python scripts/create_dataset_manifest.py
```

Run the image quality audit:

```bash
uv run python scripts/audit_dataset_images.py
```

Run a quick baseline subset:

```bash
uv run python scripts/run_baseline_report.py --limit 3
```

Run the full baseline set:

```bash
uv run python scripts/run_baseline_report.py
```

Optional paths:

```bash
uv run python scripts/run_baseline_report.py \
  --manifest data/dataset_manifest.csv \
  --output-dir outputs/experiments/baseline_v1 \
  --config configs/experiments/yolo11_sam2_baseline.toml
```

## Outputs

Generated outputs are ignored by Git by default:

```text
outputs/experiments/baseline_v1/
  config_snapshot.toml
  model_manifest_snapshot.json
  metrics.json
  metrics_by_image.csv
  failure_notes.md
  overlays/
```

The runner also writes intermediate masks, raw segmentations, and per-image
metadata under the same output directory so it does not pollute `data/`.

## Metrics

`metrics_by_image.csv` has one row per image.

- `foreground_iou`: overlap between all predicted food pixels and all annotated
  food pixels.
- `instance_iou`: mean matched-instance IoU after assignment.
- `dice`: foreground Dice/F1 overlap.
- `boundary_f_score`: boundary agreement with a small pixel tolerance.
- `precision`: matched predictions divided by predicted instances.
- `recall`: matched ground-truth instances divided by annotated instances.
- `false_positives`: predicted instances that did not match a ground-truth
  instance.
- `missed_instances`: annotated instances that were not matched.
- `area_error`: mean absolute relative area error for matched instances.
- `status`: `ok` or `failed`.
- `notes`: concise error text for failed images.

`metrics.json` aggregates mean and median metrics, records the config path, git
commit, model manifest summary, and warnings.

## Failure notes

`failure_notes.md` is generated as a template. After reviewing overlays, fill in
the visual notes and likely failure type checkboxes for each image.

Common failure types:

- false positive;
- false negative;
- bad box;
- mask leak;
- merged foods;
- wrong class;
- poor image quality.

## Interpretation

Do not overfit decisions to a tiny baseline set. Use the report to find obvious
failure modes, prioritize annotation, and compare future changes against the same
images. Strong claims require a larger held-out test split.
