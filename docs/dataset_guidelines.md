# Dataset guidelines

This dataset is still small. Treat it as a baseline evaluation set and a guide
for what to collect next, not as training data for a production model.

## Collection targets

Short term: collect and annotate 30-50 images.

Medium term: collect and annotate 100-150 images.

Fine-tuning target: collect and annotate 200-300 images before drawing strong
conclusions about model quality.

## Splits

Current manifest splits:

- `baseline_eval`: images with ground-truth masks used for repeatable baseline
  metrics.
- `candidate`: collected images that still need annotation or quality review.
- `bad`: images kept as examples of unacceptable quality.

Later, when the dataset is large enough, introduce `train`, `val`, and `test`.
Do not move the current tiny baseline set into training without first creating a
new held-out evaluation set.

## Controlled and robust collection

Aim for roughly:

- 70% controlled images: top-down, full plate visible, good lighting, sharp.
- 30% robustness cases: mild shadows, slanted angle, partial occlusion, varied
  plates, mixed foods, different phone cameras.

Avoid making the robustness set mostly unusable. The food boundaries still need
to be annotatable.

## Image quality checklist

Prefer images with:

- top-down or near top-down angle;
- full plate visible;
- sharp focus;
- enough resolution for food boundaries;
- no severe motion blur;
- no severe darkness;
- no overexposure;
- no harsh shadows crossing important boundaries;
- food regions that can be annotated consistently.

Mark weak images in `data/dataset_manifest.csv` using `quality` values such as
`maybe_blurry`, `too_dark`, or `bad`.

## Annotation rules

- Annotate one instance per food region.
- Do not annotate the plate, tray, table, cutlery, or background.
- If two portions of the same food are visibly separated, annotate them as
  separate instances.
- If foods touch but still have a visible boundary, split them.
- If foods are truly mixed with no reliable visual boundary, annotate the mixed
  region as one instance and note it in the manifest.
- Sauces and small fragments should be annotated only when they are visually
  meaningful for the meal; otherwise include them with the nearest food region
  and note the limitation.

Use single-channel PNG instance masks for evaluation. Background must be `0`;
food instances should use positive integer IDs.

## Naming

Existing files use `imagem1.jpg`, `imagem2.jpg`, and so on. New collection can
keep that convention for continuity, but `img_0001.jpg`, `img_0002.jpg`, etc. is
preferred once the dataset grows.

The manifest `image_id` should match the image filename stem and remain stable.

## Manifest and audit

Update the manifest after adding images or masks:

```bash
uv run python scripts/create_dataset_manifest.py
```

Run the local quality audit:

```bash
uv run python scripts/audit_dataset_images.py
```

The audit writes ignored local output to:

```text
outputs/dataset_audit/images_quality.csv
```

Use the audit flags to decide whether an image belongs in `baseline_eval`,
`candidate`, or `bad`.
