# Model Acquisition Guide

Since local development runs on CPU, models must be in ONNX format before being placed in the `models/` directory.

## 1. Exporting YOLOv11 (Detector)
The YOLO detector is lightweight and can be exported directly. Run the following in a fresh Python environment or Colab:

\`\`\`python
# Requires: pip install ultralytics
from ultralytics import YOLO

model = YOLO("yolo11n.pt") 
path = model.export(format="onnx", opset=14, dynamic=False)
print(f"Exported to: {path}")
\`\`\`
Rename the output to `yolov11_food.onnx` and move it to `models/`.

## 2. Acquiring SAM 2.1 (Segmenter)
Exporting the SAM 2 architecture (Hiera) into ONNX is complex because it must be split into two separate models: the Image Encoder and the Prompt Decoder. We download the pre-exported ONNX weights directly from Hugging Face.

Install the model-acquisition dependency group first:

\`\`\`bash
uv sync --group models
\`\`\`

\`\`\`python
import os
import shutil
from huggingface_hub import hf_hub_download

repo = "vietanhdev/segment-anything-2-onnx-models"
os.makedirs("models", exist_ok=True)

# Download Encoder
enc_path = hf_hub_download(repo_id=repo, filename="sam2_hiera_tiny.encoder.onnx")
shutil.copy(enc_path, "models/sam2.1_hiera_tiny.encoder.onnx")

# Download Decoder
dec_path = hf_hub_download(repo_id=repo, filename="sam2_hiera_tiny.decoder.onnx")
shutil.copy(dec_path, "models/sam2.1_hiera_tiny.decoder.onnx")

print("SAM 2.1 ONNX weights downloaded to models/ directory.")
\`\`\`
