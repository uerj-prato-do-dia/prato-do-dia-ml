# Baseline v1 failure analysis

This document summarizes the manually reviewed failure notes for the baseline
YOLO11 + SAM2 ONNX pipeline on the current `baseline_eval` split.

## Metric summary

| Metric | Mean |
| --- | ---: |
| Foreground IoU | 0.129779 |
| Instance IoU | 0.316586 |
| Dice | 0.212984 |
| Boundary F-score | 0.370098 |
| Precision | 0.950000 |
| Recall | 0.284177 |

## Interpretation

The baseline behaves conservatively: high precision but low recall. It tends to
miss food regions more than it invents correct food regions. The main failure is
not image quality for images 1-8, but domain mismatch and poor food-specific
detection capability.

The generic pretrained detector often predicts broad or wrong semantic classes,
then SAM segments those broad proposals. This creates plate/background masks,
bad boxes, merged foods, and weak instance separation. The current images are
usable for baseline evaluation, but the model is not yet food-domain aligned.

## Per-image summary

| image_id | foreground_iou | instance_iou | primary_failure | key_notes |
| --- | ---: | ---: | --- | --- |
| imagem1 | 0.207061 | 0.522831 | domain_mismatch / false_negative / bad_box | Rice/garnish was partially detected with wrong class; beans, fries, lettuce, tomato, meat, onion rings mostly missed; some plate/background leakage. |
| imagem2 | 0.138339 | 0.271751 | domain_mismatch / plate_false_positive / false_negative | Large mask covers plate/table; potatoes partly detected but meat, asparagus, tomatoes, and green vegetables are missed. |
| imagem3 | 0.117158 | 0.477362 | domain_mismatch / plate_false_positive / merged_foods | Plate/border is detected as food; carrot/lettuce/meat region is merged; rice and beans missed. |
| imagem4 | 0.356690 | 0.252788 | merged_foods / mask_leak / wrong_class | One broad detection covers ground meat, tomato, rice, egg, and plate area without instance separation. |
| imagem5 | 0.009467 | 0.444973 | mask_leak / merged_foods / false_negative | Large mask leaks over plate and multiple foods; rice, beans, chicken, egg, lettuce, tomato, and olive are not separated. |
| imagem6 | 0.006543 | 0.009021 | near_total_failure / plate_false_positive / false_negative | Almost all food instances are missed; model predicts a large wrong-class plate/border region. |
| imagem7 | 0.023850 | 0.034005 | merged_foods / plate_false_positive / false_negative | One large wrong-class region covers plate and all food; rice, shoestring potatoes, and strogonoff are not separated. |
| imagem8 | 0.179124 | 0.519958 | domain_mismatch / merged_foods / false_negative | Some overlap with food regions, but broad wrong-class masks merge rice, greens, egg, pork, and plate/background. |

## Common failure modes

- `domain_mismatch`: generic detector classes do not map well to the target
  plate-food categories.
- `wrong_class`: detected regions are semantically wrong for the meal analysis
  task.
- `false_negative`: many visible food regions are not proposed at all.
- `bad_box`: proposal boxes are too broad or poorly aligned.
- `mask_leak`: SAM masks leak into plate/background when proposals are broad.
- `merged_foods`: multiple food instances are collapsed into one detection.
- `plate_false_positive`: plate, table, or border regions are detected as food.
- `weak_instance_separation`: even when foreground overlaps, individual foods
  are not separated well enough for nutrition mapping.

## Next experimental implication

The next experiment should focus on increasing recall without destroying
precision. Before fine-tuning, run a threshold sweep over YOLO confidence and
NMS IoU, and audit ground-truth masks to ensure evaluation validity.

If threshold changes do not materially improve recall, the next scientific step
is likely food-domain data collection and fine-tuning rather than more pipeline
plumbing.
