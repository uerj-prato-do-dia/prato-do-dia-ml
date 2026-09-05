#!/usr/bin/env python3
"""Near-duplicate image detection script using Perceptual Hashing (pHash)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim" / "zenitais"
DUPLICATES_DIR = DATA_DIR / "interim" / "duplicatas"


def compute_phash(image_path: Path, hash_size: int = 8) -> np.ndarray:
    """Compute Perceptual Hash (pHash) for an image."""
    with Image.open(image_path) as pil_img:
        pil_img.load()
        transposed = ImageOps.exif_transpose(pil_img)
        gray = cv2.cvtColor(np.array(transposed.convert("RGB")), cv2.COLOR_RGB2GRAY)

    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_low_freq = dct[:hash_size, :hash_size]
    avg = np.mean(dct_low_freq[1:])  # Ignore DC coefficient
    return dct_low_freq > avg


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """Compute Hamming Distance between two 64-bit boolean hash arrays."""
    return int(np.count_nonzero(hash1 != hash2))


def deduplicate(
    input_dir: Path = INTERIM_DIR,
    duplicates_dir: Path = DUPLICATES_DIR,
    max_distance: int = 5,
) -> tuple[int, int]:
    duplicates_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png"}
    files = sorted([p for p in input_dir.glob("*") if p.suffix.lower() in image_extensions])

    if not files:
        print(f"⚠️ Nenhuma imagem encontrada em {input_dir}.")
        return 0, 0

    hashes: list[tuple[Path, np.ndarray]] = []
    duplicate_files: set[Path] = set()

    print(f"🔍 Analisando {len(files)} imagens em {input_dir} para remoção de duplicatas...")

    for img_path in files:
        try:
            h = compute_phash(img_path)
        except Exception as exc:
            print(f"⚠️ Erro ao calcular pHash para {img_path.name}: {exc}")
            continue

        is_dup = False
        for original_path, orig_hash in hashes:
            dist = hamming_distance(h, orig_hash)
            if dist <= max_distance:
                is_dup = True
                duplicate_files.add(img_path)
                print(f"  👯‍♂️ [DUPLICATA] {img_path.name} é quase idêntica a {original_path.name} (distância pHash: {dist} <= {max_distance})")
                break

        if not is_dup:
            hashes.append((img_path, h))

    for dup_path in duplicate_files:
        target = duplicates_dir / dup_path.name
        dup_path.rename(target)

    kept_count = len(files) - len(duplicate_files)
    print("\n📊 Resultado da Deduplicação:")
    print(f"  - Total analisadas: {len(files)}")
    print(f"  - Imagens únicas mantidas: {kept_count}")
    print(f"  - Duplicatas segregadas (movidas para {duplicates_dir.relative_to(PROJECT_ROOT)}): {len(duplicate_files)}")

    return kept_count, len(duplicate_files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Identifica e segrega imagens quase idênticas (near-duplicates) usando pHash.")
    parser.add_argument("--max-distance", type=int, default=5, help="Distância máxima de Hamming para considerar duplicata (default: 5)")
    args = parser.parse_args()

    deduplicate(max_distance=args.max_distance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
