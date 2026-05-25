"""Extract per-instance features from generated segmentation masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.feature_extraction import extract_instance_features, save_features_csv
from src.io_utils import input_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--output", type=Path, default=Path("data/features/features.csv"))
    args = parser.parse_args()

    config = load_config(args.config)
    rows = []
    for image_path in input_images(config.paths.input_dir):
        mask_path = config.paths.mask_dir / f"{image_path.stem}_instances.png"
        if not mask_path.exists():
            continue
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        instance_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image_bgr is None:
            raise FileNotFoundError(f"could not load image: {image_path}")
        if instance_mask is None:
            raise FileNotFoundError(f"could not load instance mask: {mask_path}")
        rows.append(extract_instance_features(image_bgr, instance_mask, image_path.name))

    output_path = save_features_csv(rows, args.output)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
