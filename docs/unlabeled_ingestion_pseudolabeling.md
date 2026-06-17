# Unlabeled ingestion and pseudo-labeling

This workflow ingests raw phone images from `dataset/`, normalizes them as
hash-addressed JPEGs, and runs exploratory YOLOE pseudo-labeling for manual
inspection.

It does not create ground truth. Pseudo-labels are model output and must stay
separate from verified annotation data.

## Architecture

Existing project pieces reused by this workflow:

- `prato_do_dia_ml.io_utils.load_image_bgr`: image decode, BGR conversion, alpha
  handling, and validation.
- `prato_do_dia_ml.io_utils.input_images`: deterministic supported-image
  discovery for processed image directories.
- `prato_do_dia_ml.preprocessing.letterbox_image`: model-input resize logic used
  by the production YOLO11/SAM2 path.
- `FoodSegmentationPipeline`: historical YOLO11 + SAM2 ONNX inference pipeline.
- `scripts/run_ultralytics_zero_shot.py`: YOLOE model/config loading,
  prediction-to-mask conversion, and overlay rendering.

The ingest script does not store letterboxed images. The repo's production
model preprocessing applies letterbox at inference time so predictions can be
mapped back to the original image geometry. Storing already-letterboxed images
would turn model-input tensors into dataset samples and make later annotation
less useful.

## Ingest Raw Images

Put raw files in:

```text
dataset/
```

Run:

```bash
uv run python scripts/ingest_unlabeled_dataset.py
```

Default output:

```text
data/raw/unlabeled/
  <sha256>.jpg

outputs/dataset_ingestion/unlabeled_v1/
  ingested_images.csv
  rejected_images.csv
  summary.json
```

Behavior:

- Decodes each supported image.
- Rejects corrupt or unsupported files.
- Converts valid files to normalized JPEG.
- Uses the SHA-256 of the normalized JPEG bytes as the stable image ID.
- Writes images to `data/raw/unlabeled/`.
- Leaves source files untouched by default.

To move invalid files out of `dataset/` instead of only reporting them:

```bash
uv run python scripts/ingest_unlabeled_dataset.py --move-invalid
```

Use that option only when you are comfortable moving source files.

## Exploratory Pseudo-labeling

Run the current balanced YOLOE config:

```bash
uv run python scripts/pseudo_label_unlabeled.py \
  --config configs/experiments/yoloe26s_food_balanced.toml \
  --high-confidence-threshold 0.85
```

For more aggressive proposals:

```bash
uv run python scripts/pseudo_label_unlabeled.py \
  --config configs/experiments/yoloe26m_food_recall.toml \
  --high-confidence-threshold 0.85
```

Default output:

```text
outputs/pseudo_labels/unlabeled_v1/
  summary.csv
  failures.csv
  run_metadata.json
  predictions/
    <image_id>_predictions.json
  overlays/
    <image_id>_overlay.jpg
  pseudo_labels/
    json/
      <image_id>.json
    yolo_txt/
      <image_id>.txt
```

`predictions/` contains all model predictions at the config confidence
threshold. `pseudo_labels/` contains only instances whose confidence is at least
the high-confidence threshold.

## Manual overlay review

Manual overlay review is the human triage step between exploratory
pseudo-labeling and any future annotation work. It answers four questions:

1. What is the physical condition of the source image?
2. Did the model produce useful mask proposals?
3. What should happen to this image next?
4. Are there notes an annotator or future experiment should know?

This step does not create ground truth. It only produces a local review CSV that
helps decide which images should move into a separate manual annotation flow.

Generate the local static review page:

```bash
uv run python scripts/render_overlay_review.py
```

Default input:

```text
outputs/pseudo_labels/unlabeled_v1/summary.csv
```

Default output:

```text
outputs/reviews/unlabeled_v1_overlay_review/
  index.html
  review_template.csv
  review_manifest.json
```

### Review page

Open `index.html` directly in the browser. No server is required. The page uses
relative links to the normalized image, overlay, prediction JSON, pseudo-label
JSON, and YOLO TXT files; it does not copy generated artifacts.

The page supports filtering by predicted class, image condition, mask error,
minimum prediction count, images with no detections, and images with
high-confidence pseudo-labels. Review fields are stored in browser
`localStorage` and can be exported with **Export review CSV**.

Use the filters to work in passes:

1. Start with all images and reject obvious bad inputs.
2. Review images with no detections.
3. Review images with high-confidence pseudo-labels.
4. Filter by common predicted classes to spot systematic model behavior.
5. Export the CSV after the review pass.

### Review fields

Each image has these manual review fields:

```text
image_condition,mask_errors,next_action,notes
```

#### `image_condition`

Use this to describe only the physical condition of the input photo. Do not use
this field to decide whether the image is useful; that decision belongs in
`next_action`.

- `optimal`: complete plate, top-down or close to top-down, good lighting, and
  good focus.
- `suboptimal_angle`: perspective is severe enough to distort food area.
- `suboptimal_lighting`: harsh shadows or overexposure cover important regions.
- `suboptimal_focus`: motion blur or focus issues compromise food boundaries.
- `invalid_content`: not food, not a recognizable plate, or otherwise invalid.

#### `mask_errors`

Use this to mark all observed model failure modes. This is a multi-label field:
select every error that applies. If the overlay is visually acceptable, leave the
list empty.

- `undersegmented`: the model merged multiple food regions into one mask.
- `oversegmented`: the model split one food region into too many fragments.
- `missed_food`: relevant food is visible but not detected.
- `background_leak`: masks invade plate, table, background, or non-food areas.
- `hallucination`: the model predicted a class that is not present in the image.
- `no_prediction`: the model produced no detections for this image.

In exported CSV files, `mask_errors` is serialized as a JSON array string:

```csv
mask_errors
"[""missed_food"",""hallucination""]"
```

`mask_errors=[]` means no visible mask error was marked by the reviewer. It does
not mean the pseudo-label is verified ground truth.

#### `next_action`

Use this to decide the next operational step.

- `annotate_standard`: image is physically suitable for the normal annotation
  flow. Use this for `optimal` images that should become verified ground truth.
- `annotate_hard`: image is suboptimal but useful. It still needs ground truth,
  but should be routed into a hard/robustness evaluation split rather than the
  normal baseline set.
- `retake`: the meal/case is useful, but the photo should be captured again.
  Use for important examples with blur, bad angle, severe shadow, or cropped
  plate.
- `discard`: exclude this image from the workflow. Use for invalid content,
  pure junk, unusable duplicates, or files that should not remain in the data
  lake.
- `needs_review`: default state. Use when the image has not been reviewed yet or
  needs a second opinion.

Quick decision guide:

```text
Optimal image + worth verified labels -> annotate_standard
Suboptimal image + useful hard case -> annotate_hard
Important case but photo should be recaptured -> retake
Bad and not useful -> discard
Not decided yet -> needs_review
```

Examples:

```text
Good image, model missed food:
image_condition=optimal
mask_errors=["missed_food"]
next_action=annotate_standard

Bad angle, still useful for robustness:
image_condition=suboptimal_angle
mask_errors=["missed_food", "hallucination"]
next_action=annotate_hard

Invalid image:
image_condition=invalid_content
mask_errors=[]
next_action=discard

Useful meal, poor capture:
image_condition=suboptimal_focus
mask_errors=[]
next_action=retake
```

#### `notes`

Use free text for anything the dropdowns do not capture. Good notes are short
and actionable, for example:

- `rice and beans merged`
- `plate leak on right side`
- `good candidate for annotation`
- `retake with top-down angle`

### After exporting the review CSV

The exported CSV is a triage artifact only. It must not be copied into
`data/ground_truth/`, and it must not update `data/dataset_manifest.csv`
automatically.

Recommended follow-up:

1. Keep the exported CSV under `outputs/reviews/`.
2. Select rows with `next_action=annotate_standard` for the normal manual
   annotation workflow.
3. Select rows with `next_action=annotate_hard` for manual annotation plus a
   hard/robustness evaluation split.
4. Use `next_action=retake` as a capture checklist.
5. Never treat pseudo-label JSON/TXT files as verified masks without human
   correction.

`outputs/reviews/` is local generated output and is ignored by Git.

## Safety Rules

- Do not copy pseudo-labels into `data/ground_truth/`.
- Do not train on pseudo-labels as if they were gold labels.
- Use overlays for manual review before annotation or training decisions.
- Treat `data/raw/unlabeled/` as unverified input, not as `baseline_eval`.
- Keep `outputs/pseudo_labels/` ignored and local unless a deliberately small
  review artifact is explicitly selected.

Pseudo-labels are useful for pre-annotation and triage. They are not reliable
enough for direct nutrition estimation or final classification.
