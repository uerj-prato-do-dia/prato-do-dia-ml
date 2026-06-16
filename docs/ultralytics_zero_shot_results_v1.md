# Ultralytics zero-shot results v1

## Purpose

This experiment tested downloaded Ultralytics segmentation models as zero-shot
food mask proposal generators. The goal was to compare them against the current
YOLO11 + SAM2 ONNX baseline without training, replacing the production/default
pipeline, or changing API/mobile code.

The outputs were generated under:

```text
outputs/experiments/ultralytics_zero_shot_v1/
```

Generated outputs remain local experiment artifacts and should not be committed
by default.

## Dataset Scope

The comparison used the 8 annotated `baseline_eval` images from
`data/dataset_manifest.csv`. These images have ground-truth instance masks and
were already used for the previous YOLO11 + SAM2 baseline reports.

The result is strong enough to guide the next experiment, but it is not final
model-quality evidence. Validate again after adding more top-down annotated
phone photos.

## Models Tested

Closed-vocabulary segmentation models:

- `external_models/ultralytics/yolo11m-seg.pt`
- `external_models/ultralytics/yolo11l-seg.pt`
- `external_models/ultralytics/yolo11x-seg.pt`

Open-vocabulary YOLOE segmentation models with food prompts:

- `external_models/ultralytics/yoloe-26s-seg.pt`
- `external_models/ultralytics/yoloe-26m-seg.pt`

MobileSAM was only treated as a load/smoke check. It is not directly comparable
without detector boxes or other prompts.

Food prompts:

```text
rice, beans, meat, chicken, salad, tomato, pasta, potato, egg,
french fries, vegetables, beef, pork, fish
```

## Comparison With YOLO11 + SAM2

The previous YOLO11 + SAM2 baseline was weak on the same 8 images:

| pipeline | foreground_iou | instance_iou | dice | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11 + SAM2 original baseline | 0.129779 | 0.316586 | 0.212984 | 0.950000 | 0.284177 |
| YOLO11 + SAM2 balanced threshold config | 0.325164 | 0.416089 | 0.463572 | 0.795343 | 0.713442 |
| YOLO11 + SAM2 recall threshold config | 0.407598 | 0.504995 | 0.550319 | 0.650151 | 0.849355 |

YOLOE zero-shot is clearly better for food mask proposals and pre-annotation on
this small dataset, especially for food foreground coverage and rice/beans
separation.

## Top Configurations

| config | model_file | conf | iou | foreground_iou | instance_iou | dice | boundary_f_score | precision | recall | false_positives | missed_instances |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yoloe26s_seg_food_conf_0p10_iou_0p30` | `yoloe-26s-seg.pt` | 0.10 | 0.30 | 0.880418 | 0.700691 | 0.933743 | 0.520543 | 0.846230 | 0.871181 | 11 | 7 |
| `yoloe26m_seg_food_conf_0p08_iou_0p30` | `yoloe-26m-seg.pt` | 0.08 | 0.30 | 0.907760 | 0.663230 | 0.950492 | 0.553895 | 0.800824 | 0.930208 | 20 | 3 |
| `yoloe26m_seg_food_conf_0p01_iou_0p30` | `yoloe-26m-seg.pt` | 0.01 | 0.30 | - | - | - | - | 0.419576 | 0.958333 | 132 | 1 |

`yoloe26s_seg_food_conf_0p10_iou_0p30` is the best balanced/general config. It
keeps precision high, recall high, and false positives modest. It is now
captured as:

```text
configs/experiments/yoloe26s_food_balanced.toml
```

`yoloe26m_seg_food_conf_0p08_iou_0p30` is the best recall-constrained config
with precision above `0.75`. It catches more true food regions, but accepts more
false positives. It is now captured as:

```text
configs/experiments/yoloe26m_food_recall.toml
```

`yoloe26m_seg_food_conf_0p01_iou_0p30` has the highest raw recall, but it is not
recommended for practical use. Precision drops to `0.419576` and false positives
rise to `132`, making review and downstream use too noisy.

## Visual Review Summary

The two selected YOLOE configs looked good visually:

- They separate rice and beans much better than the previous YOLO11 + SAM2
  baseline.
- They invade plate/background only slightly.
- They are clearly better for mask proposals and pre-annotation.
- They sometimes merge very similar or touching foods.
- Some predicted classes are unreliable or too generic.

## Strengths

- Strong food foreground segmentation.
- Better rice/beans separation.
- Much better recall than YOLO11 + SAM2.
- Useful for pre-annotation and annotation acceleration.
- Less plate/background leakage than the previous baseline.

## Weaknesses

- Class labels are still unreliable.
- Some foods remain merged, especially similar or touching regions.
- Some oversegmentation remains.
- Boundaries are still imperfect.
- Brazilian food categories are not represented well enough.
- Not ready for direct nutrition estimation without review or fine-tuning.

## Recommendation

Use `yoloe26s_food_balanced` as the main zero-shot baseline candidate.

Use `yoloe26m_food_recall` for aggressive pre-annotation where human review is
expected and extra proposals are acceptable.

Keep YOLO11 + SAM2 as a historical baseline for comparison. Do not replace the
production/default pipeline yet.

Before making product decisions, rerun this evaluation after adding more
top-down annotated phone photos.
