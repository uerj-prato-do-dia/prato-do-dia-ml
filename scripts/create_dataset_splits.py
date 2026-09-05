#!/usr/bin/env python3
"""Generates deterministic train/val/test split files (70% / 20% / 10%) for YOLO training."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_IMAGES_DIR = DATA_DIR / "processed" / "images"
SPLITS_DIR = DATA_DIR / "splits"


def create_splits(
    images_dir: Path = PROCESSED_IMAGES_DIR,
    splits_dir: Path = SPLITS_DIR,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
) -> tuple[int, int, int]:
    splits_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in image_extensions])

    if not image_paths:
        print(f"⚠️ Nenhuma imagem encontrada em {images_dir}. Gerando arquivos de split vazios.")
        (splits_dir / "train.txt").write_text("", encoding="utf-8")
        (splits_dir / "val.txt").write_text("", encoding="utf-8")
        (splits_dir / "test.txt").write_text("", encoding="utf-8")
        return (0, 0, 0)

    # Shuffle deterministically
    random.seed(seed)
    shuffled = list(image_paths)
    random.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_files = shuffled[:train_count]
    val_files = shuffled[train_count : train_count + val_count]
    test_files = shuffled[train_count + val_count :]

    def _write_split_file(filename: str, files: list[Path]) -> None:
        rel_paths = [str(f.relative_to(DATA_DIR)) for f in files]
        (splits_dir / filename).write_text("\n".join(rel_paths) + ("\n" if rel_paths else ""), encoding="utf-8")

    _write_split_file("train.txt", train_files)
    _write_split_file("val.txt", val_files)
    _write_split_file("test.txt", test_files)

    print(f"✔ Splits gerados em {splits_dir}:")
    print(f"  - Treino (70%): {len(train_files)} imagens")
    print(f"  - Validação (20%): {len(val_files)} imagens")
    print(f"  - Teste (10%): {len(test_files)} imagens")

    return (len(train_files), len(val_files), len(test_files))


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera os arquivos de split train.txt, val.txt e test.txt.")
    parser.add_argument("--seed", type=int, default=42, help="Semente randômica para reprodutibilidade")
    args = parser.parse_args()

    create_splits(seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
