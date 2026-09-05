#!/usr/bin/env python3
"""Automated image screening script to filter sharp, zenithal meal photos from data/raw to data/interim."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim" / "zenitais"


def calculate_sharpness(image_bgr: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def evaluate_image(
    image_path: Path,
    min_sharpness: float = 80.0,
    max_aspect_ratio: float = 1.8,
) -> tuple[bool, str, float]:
    """Evaluate image sharpness, aspect ratio, and EXIF orientation."""
    try:
        with Image.open(image_path) as pil_img:
            pil_img.load()
            transposed = ImageOps.exif_transpose(pil_img)
            w, h = transposed.size
            rgb = transposed.convert("RGB")
            bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        return False, f"Falha ao ler a imagem: {exc}", 0.0

    # Aspect ratio check (zenithal photos should not be extreme panoramas)
    aspect_ratio = max(w / h, h / w)
    if aspect_ratio > max_aspect_ratio:
        return False, f"Proporção panorâmica excessiva ({aspect_ratio:.2f} > {max_aspect_ratio})", 0.0

    # Sharpness check
    sharpness = calculate_sharpness(bgr)
    if sharpness < min_sharpness:
        return False, f"Imagem desfocada/borrada (nitidez: {sharpness:.1f} < {min_sharpness})", sharpness

    return True, f"Aprovada (nitidez: {sharpness:.1f}, proporção: {aspect_ratio:.2f})", sharpness


def filter_images(
    raw_dir: Path = RAW_DIR,
    output_dir: Path = INTERIM_DIR,
    min_sharpness: float = 80.0,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png"}
    files = sorted([p for p in raw_dir.glob("*") if p.suffix.lower() in image_extensions])

    if not files:
        print(f"⚠️ Nenhuma imagem encontrada em {raw_dir}.")
        return 0, 0

    approved_count = 0
    rejected_count = 0

    print(f"🔍 Analisando {len(files)} imagens de {raw_dir}...")

    for img_path in files:
        is_valid, reason, score = evaluate_image(img_path, min_sharpness=min_sharpness)
        if is_valid:
            approved_count += 1
            dest = output_dir / img_path.name
            dest.write_bytes(img_path.read_bytes())
            print(f"  ✅ [APROVADA] {img_path.name} -> {reason}")
        else:
            rejected_count += 1
            print(f"  ❌ [REJEITADA] {img_path.name} -> {reason}")

    print("\n📊 Resultado da Triagem:")
    print(f"  - Total analisadas: {len(files)}")
    print(f"  - Aprovadas (copiadas para {output_dir.relative_to(PROJECT_ROOT)}): {approved_count}")
    print(f"  - Rejeitadas (filtradas): {rejected_count}")

    return approved_count, rejected_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Filtra fotos nítidas e zenitais de data/raw para data/interim.")
    parser.add_argument("--min-sharpness", type=float, default=80.0, help="Limiar mínimo de nitidez (Laplacian variance)")
    args = parser.parse_args()

    filter_images(min_sharpness=args.min_sharpness)
    return 0


if __name__ == "__main__":
    sys.exit(main())
