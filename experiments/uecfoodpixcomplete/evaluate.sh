#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?Usage: ./evaluate.sh path/to/best.pt [data.yaml]}"
DATA_YAML="${2:-experiments/uecfoodpixcomplete/converted_sample/data.yaml}"

yolo segment val \
  model="${MODEL_PATH}" \
  data="${DATA_YAML}" \
  imgsz=640 \
  device=cpu \
  project=experiments/uecfoodpixcomplete/outputs/val
