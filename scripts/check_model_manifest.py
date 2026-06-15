from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check or regenerate the ONNX model manifest.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--write", action="store_true", help="Rewrite model_manifest.json with current hashes.")
    args = parser.parse_args()

    manifest = build_manifest(args.models_dir)
    manifest_path = args.models_dir / "model_manifest.json"
    if args.write:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected != manifest:
        raise SystemExit("model manifest does not match current model files")


def build_manifest(models_dir: Path) -> dict[str, object]:
    entries = [
        ("yolov11_food.onnx", "detector"),
        ("sam2.1_hiera_tiny.encoder.onnx", "sam_encoder"),
        ("sam2.1_hiera_tiny.decoder.onnx", "sam_decoder"),
    ]
    return {
        "schema_version": "1.0",
        "models": [
            {
                "filename": filename,
                "role": role,
                "size_bytes": (models_dir / filename).stat().st_size,
                "sha256": sha256_file(models_dir / filename),
            }
            for filename, role in entries
        ],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
