"""Dataset curation tools: image quality screening, Laplacian sharpness, and pHash deduplication."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageOps


class QualityEvaluation(NamedTuple):
    is_valid: bool
    reason: str
    sharpness: float
    aspect_ratio: float


def calculate_sharpness(image_bgr: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def evaluate_image_quality(
    image_path: Path,
    min_sharpness: float = 80.0,
    max_aspect_ratio: float = 1.8,
) -> QualityEvaluation:
    """Evaluate image sharpness, aspect ratio, and EXIF orientation."""
    try:
        with Image.open(image_path) as pil_img:
            pil_img.load()
            transposed = ImageOps.exif_transpose(pil_img)
            w, h = transposed.size
            rgb = transposed.convert("RGB")
            bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        return QualityEvaluation(False, f"Falha ao ler a imagem: {exc}", 0.0, 0.0)

    aspect_ratio = max(w / h, h / w)
    if aspect_ratio > max_aspect_ratio:
        return QualityEvaluation(
            False, f"Proporção panorâmica excessiva ({aspect_ratio:.2f} > {max_aspect_ratio})", 0.0, aspect_ratio
        )

    sharpness = calculate_sharpness(bgr)
    if sharpness < min_sharpness:
        return QualityEvaluation(
            False, f"Imagem desfocada/borrada (nitidez: {sharpness:.1f} < {min_sharpness})", sharpness, aspect_ratio
        )

    return QualityEvaluation(
        True, f"Aprovada (nitidez: {sharpness:.1f}, proporção: {aspect_ratio:.2f})", sharpness, aspect_ratio
    )


def compute_phash(image_path: Path, hash_size: int = 8) -> np.ndarray:
    """Compute Perceptual Hash (pHash) for an image."""
    with Image.open(image_path) as pil_img:
        pil_img.load()
        transposed = ImageOps.exif_transpose(pil_img)
        gray = cv2.cvtColor(np.array(transposed.convert("RGB")), cv2.COLOR_RGB2GRAY)

    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_low_freq = dct[:hash_size, :hash_size]
    avg = np.mean(dct_low_freq[1:])
    return dct_low_freq > avg


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """Compute Hamming Distance between two 64-bit boolean hash arrays."""
    return int(np.count_nonzero(hash1 != hash2))


def filter_zenithal_images(
    raw_dir: Path,
    output_dir: Path,
    min_sharpness: float = 80.0,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png"}
    files = sorted([p for p in raw_dir.glob("*") if p.suffix.lower() in image_extensions])

    if not files:
        return 0, 0

    approved_count = 0
    rejected_count = 0

    for img_path in files:
        eval_result = evaluate_image_quality(img_path, min_sharpness=min_sharpness)
        if eval_result.is_valid:
            approved_count += 1
            dest = output_dir / img_path.name
            dest.write_bytes(img_path.read_bytes())
        else:
            rejected_count += 1

    return approved_count, rejected_count


def deduplicate_dataset(
    input_dir: Path,
    duplicates_dir: Path,
    max_distance: int = 5,
) -> tuple[int, int]:
    duplicates_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png"}
    files = sorted([p for p in input_dir.glob("*") if p.suffix.lower() in image_extensions])

    if not files:
        return 0, 0

    hashes: list[tuple[Path, np.ndarray]] = []
    duplicate_files: set[Path] = set()

    for img_path in files:
        try:
            h = compute_phash(img_path)
        except Exception:
            continue

        is_dup = False
        for _original_path, orig_hash in hashes:
            dist = hamming_distance(h, orig_hash)
            if dist <= max_distance:
                is_dup = True
                duplicate_files.add(img_path)
                break

        if not is_dup:
            hashes.append((img_path, h))

    for dup_path in duplicate_files:
        target = duplicates_dir / dup_path.name
        dup_path.rename(target)

    kept_count = len(files) - len(duplicate_files)
    return kept_count, len(duplicate_files)
