#!/usr/bin/env python3
"""Automated script to download, verify SHA-256 hashes, and position ONNX model files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SCRIPTS_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPTS_DIR.parent
MODELS_DIR = ML_ROOT / "models"


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_manifest(models_dir: Path) -> Path | None:
    manifest_paths = [
        models_dir / "models_manifest.json",
        models_dir / "model_manifest.json",
    ]
    for p in manifest_paths:
        if p.exists():
            return p
    return None


def download_and_verify(
    models_dir: Path = MODELS_DIR,
    force: bool = False,
) -> bool:
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = find_manifest(models_dir)

    if not manifest_path:
        logging.error("No models_manifest.json found in %s", models_dir)
        return False

    logging.info("Reading manifest from: %s", manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_entries = manifest.get("models", [])

    all_valid = True

    for entry in model_entries:
        filename = entry.get("filename")
        expected_sha = entry.get("sha256")
        _expected_size = entry.get("size_bytes")
        role = entry.get("role", "unknown")

        if not filename or not expected_sha:
            logging.warning("Skipping invalid manifest entry: %s", entry)
            continue

        target_path = models_dir / filename
        logging.info("Checking model [%s]: %s ...", role, filename)

        if target_path.exists() and not force:
            actual_sha = calculate_sha256(target_path)
            if actual_sha.lower() == expected_sha.lower():
                logging.info("✔ Model '%s' is valid (SHA-256 match).", filename)
                continue
            else:
                logging.warning("✘ SHA-256 mismatch for '%s'! Recalculating...", filename)

        # In production, download from remote storage bucket (e.g. S3 / GCS / Releases)
        # Here we verify existing or mock local storage
        if not target_path.exists():
            logging.error("✘ Model file '%s' missing from %s. Please place valid file or download from storage.", filename, models_dir)
            all_valid = False
            continue

        actual_sha = calculate_sha256(target_path)
        if actual_sha.lower() == expected_sha.lower():
            logging.info("✔ Successfully verified SHA-256 for '%s'.", filename)
        else:
            logging.error("✘ SHA-256 validation failed for '%s'! Expected: %s, Got: %s", filename, expected_sha, actual_sha)
            all_valid = False

    return all_valid


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify SHA-256 checksums of ONNX model files.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="Path to models directory")
    parser.add_argument("--force", action="store_true", help="Force re-verification of all model files")

    args = parser.parse_args()
    success = download_and_verify(models_dir=args.models_dir, force=args.force)

    if success:
        logging.info("All ONNX models are present and SHA-256 verified successfully!")
        return 0
    else:
        logging.error("Model verification failed! Please check missing or corrupted files.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
