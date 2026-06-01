# UECFoodPixComplete Isolated Experiment

This directory is an external benchmark/sandbox for UECFoodPixComplete. It must
not be mixed with the Prato do Dia Label Studio dataset, class map, masks, or
training labels.

UECFoodPixComplete has JPG images and PNG masks. The class label is stored in
the red channel only; `0` is background and `1..102` are food categories.

## Dataset Location

Keep the raw public dataset outside the main Prato do Dia dataset. Recommended:

```text
datasets_external/uecfoodpixcomplete/data/UECFoodPIXCOMPLETE/
```

Expected structure:

```text
UECFoodPIXCOMPLETE/
  train/img/*.jpg
  train/mask/*.png
  test/img/*.jpg
  test/mask/*.png
  category.txt
  train.txt
  test.txt
```

`datasets_external/` is ignored by Git.

## Validate

```bash
python experiments/uecfoodpixcomplete/validate_uec_dataset.py \
  --uec-root datasets_external/uecfoodpixcomplete/data/UECFoodPIXCOMPLETE \
  --limit 100 \
  --per-class-pixels
```

Remove `--limit` for a full validation pass.

## Convert To YOLO Segmentation

Smoke subset using project-relevant UEC classes:

```bash
python experiments/uecfoodpixcomplete/convert_uec_to_yolo_seg.py \
  --uec-root datasets_external/uecfoodpixcomplete/data/UECFoodPIXCOMPLETE \
  --output experiments/uecfoodpixcomplete/converted_sample \
  --mode prato_relevant \
  --limit 100 \
  --symlink-images
```

Full 102-class conversion:

```bash
python experiments/uecfoodpixcomplete/convert_uec_to_yolo_seg.py \
  --uec-root datasets_external/uecfoodpixcomplete/data/UECFoodPIXCOMPLETE \
  --output experiments/uecfoodpixcomplete/converted_full \
  --mode full \
  --symlink-images
```

Binary foreground conversion for food-vs-background segmentation pretraining:

```bash
python experiments/uecfoodpixcomplete/convert_uec_to_yolo_seg.py \
  --uec-root datasets_external/uecfoodpixcomplete/data/UECFoodPIXCOMPLETE \
  --output experiments/uecfoodpixcomplete/converted_binary_food_sample \
  --mode binary_food \
  --limit 500 \
  --symlink-images
```

This mode maps every non-background UEC pixel to a single YOLO class named
`food`. It is the most relevant UEC experiment for Prato do Dia segmentation
quality because it avoids forcing Japanese food classes onto Brazilian meals.

The converter writes:

```text
images/train/
images/val/
labels/train/
labels/val/
data.yaml
categories_used.json
conversion_report.json
```

`train.txt` becomes YOLO `train`; `test.txt` becomes YOLO `val`.

## Preview Conversion

Always inspect previews before training:

```bash
python experiments/uecfoodpixcomplete/preview_masks.py \
  --converted-root experiments/uecfoodpixcomplete/converted_sample \
  --split train \
  --limit 30 \
  --random
```

Previews are written to:

```text
experiments/uecfoodpixcomplete/outputs/previews/
```

## Train Smoke Test

Install the training dependency group only when you are ready to train:

```bash
uv sync --group train
```

CPU-safe smoke run:

```bash
bash experiments/uecfoodpixcomplete/train.sh \
  experiments/uecfoodpixcomplete/converted_sample/data.yaml
```

Equivalent command:

```bash
yolo segment train \
  model=yolov8n-seg.pt \
  data=experiments/uecfoodpixcomplete/converted_sample/data.yaml \
  imgsz=640 \
  epochs=10 \
  batch=2 \
  device=cpu \
  workers=2 \
  project=experiments/uecfoodpixcomplete/outputs/runs \
  name=yolov8n_seg_uec_smoke
```

Longer CPU experiment:

```bash
EPOCHS=50 BATCH=2 WORKERS=2 RUN_NAME=yolov8n_seg_uec_50ep \
  bash experiments/uecfoodpixcomplete/train.sh \
  experiments/uecfoodpixcomplete/converted_sample/data.yaml
```

`yolo11n-seg.pt` can be used by setting `MODEL=yolo11n-seg.pt` if available.

## Evaluate

```bash
bash experiments/uecfoodpixcomplete/evaluate.sh \
  experiments/uecfoodpixcomplete/outputs/runs/yolov8n_seg_uec_smoke/weights/best.pt \
  experiments/uecfoodpixcomplete/converted_sample/data.yaml
```

## Notes And Limitations

- This is a pipeline and pretraining/benchmark experiment, not the final Prato
  do Dia dataset.
- UECFoodPixComplete classes are mostly Japanese food categories and do not map
  cleanly to Brazilian plate-food categories.
- Prefer `binary_food` for segmentation pretraining/benchmarking and `full`
  for stress-testing multi-class conversion. Use `prato_relevant` only for
  quick filtering experiments; it leaves many UEC images empty.
- Do not rename or merge UEC classes into Prato do Dia classes in this
  experiment.
- Do not commit raw UEC data, converted image symlinks/copies, YOLO runs, or
  model weights.
- Python 3.11+ is expected. Current scripts require `opencv-python` and
  `numpy`; training additionally requires `ultralytics`.
