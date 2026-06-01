#!/usr/bin/env bash
set -euo pipefail

DATA_YAML="${1:-experiments/uecfoodpixcomplete/converted_sample/data.yaml}"
MODEL="${MODEL:-yolov8n-seg.pt}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-2}"
RUN_NAME="${RUN_NAME:-yolov8n_seg_uec_smoke}"

yolo segment train \
  model="${MODEL}" \
  data="${DATA_YAML}" \
  imgsz=640 \
  epochs="${EPOCHS}" \
  batch="${BATCH}" \
  device=cpu \
  workers="${WORKERS}" \
  project=experiments/uecfoodpixcomplete/outputs/runs \
  name="${RUN_NAME}"
