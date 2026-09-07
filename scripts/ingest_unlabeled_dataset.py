"""Ingest raw unlabeled images into a hash-addressed dataset area."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from prato_do_dia_ml.io_utils import SUPPORTED_IMAGE_SUFFIXES, load_image_bgr

INGEST_COLUMNS = [
    "image_id",
    "source_path",
    "output_path",
    "sha256",
    "width",
    "height",
    "file_size_bytes",
    "status",
    "notes",
]

REJECT_COLUMNS = [
    "source_path",
    "status",
    "error_type",
    "message",
    "rejected_path",
]


@dataclass(frozen=True)
class IngestedImage:
    image_id: str
    source_path: Path
    output_path: Path
    sha256: str
    width: int
    height: int
    file_size_bytes: int
    status: str = "ok"
    notes: str = "normalized_jpeg_hash_id"


@dataclass(frozen=True)
class RejectedImage:
    source_path: Path
    status: str
    error_type: str
    message: str
    rejected_path: Path | None = None


def main() -> None:
    args = parse_args()
    images, rejected = ingest_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        jpeg_quality=args.jpeg_quality,
        move_invalid=args.move_invalid,
    )
    print("Unlabeled dataset ingestion")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Accepted: {len(images)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Manifest: {args.report_dir / 'ingested_images.csv'}")
    print(f"Rejected report: {args.report_dir / 'rejected_images.csv'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/unlabeled"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/dataset_ingestion/unlabeled_v1"))
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--move-invalid",
        action="store_true",
        help="Move undecodable files to report-dir/rejected instead of leaving source files untouched.",
    )
    return parser.parse_args(argv)


def ingest_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    report_dir: Path,
    jpeg_quality: int = 95,
    move_invalid: bool = False,
) -> tuple[list[IngestedImage], list[RejectedImage]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_dir}")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    ingested: list[IngestedImage] = []
    rejected: list[RejectedImage] = []
    for source_path in discover_input_files(input_dir):
        try:
            ingested.append(ingest_image(source_path, output_dir, jpeg_quality=jpeg_quality))
        except Exception as exc:
            rejected.append(handle_rejected_file(source_path, report_dir, exc, move_invalid=move_invalid))

    write_ingest_manifest(report_dir / "ingested_images.csv", ingested)
    write_rejected_manifest(report_dir / "rejected_images.csv", rejected)
    write_ingest_summary(report_dir / "summary.json", input_dir, output_dir, ingested, rejected)
    return ingested, rejected


def discover_input_files(input_dir: Path) -> list[Path]:
    suffixes = {suffix.lower() for suffix in SUPPORTED_IMAGE_SUFFIXES} | {".webp", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def ingest_image(source_path: Path, output_dir: Path, *, jpeg_quality: int) -> IngestedImage:
    image_bgr = load_image_bgr(source_path, background_rgb=(0, 0, 0), allow_alpha=True)
    height, width = image_bgr.shape[:2]
    success, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not success:
        raise RuntimeError(f"failed to encode normalized JPEG for {source_path}")

    payload = encoded.tobytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    output_path = output_dir / f"{sha256}.jpg"
    if not output_path.exists():
        output_path.write_bytes(payload)

    return IngestedImage(
        image_id=sha256,
        source_path=source_path,
        output_path=output_path,
        sha256=sha256,
        width=width,
        height=height,
        file_size_bytes=len(payload),
    )


def handle_rejected_file(source_path: Path, report_dir: Path, exc: Exception, *, move_invalid: bool) -> RejectedImage:
    rejected_path = None
    if move_invalid:
        rejected_dir = report_dir / "rejected"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        rejected_path = rejected_dir / source_path.name
        if rejected_path.exists():
            rejected_path = (
                rejected_dir
                / f"{source_path.stem}_{hashlib.sha256(str(source_path).encode()).hexdigest()[:8]}{source_path.suffix}"
            )
        shutil.move(str(source_path), rejected_path)
    return RejectedImage(
        source_path=source_path,
        status="rejected",
        error_type=type(exc).__name__,
        message=str(exc)[:240],
        rejected_path=rejected_path,
    )


def write_ingest_manifest(path: Path, rows: list[IngestedImage]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=INGEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_id": row.image_id,
                    "source_path": str(row.source_path),
                    "output_path": str(row.output_path),
                    "sha256": row.sha256,
                    "width": row.width,
                    "height": row.height,
                    "file_size_bytes": row.file_size_bytes,
                    "status": row.status,
                    "notes": row.notes,
                }
            )


def write_rejected_manifest(path: Path, rows: list[RejectedImage]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REJECT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_path": str(row.source_path),
                    "status": row.status,
                    "error_type": row.error_type,
                    "message": row.message,
                    "rejected_path": "" if row.rejected_path is None else str(row.rejected_path),
                }
            )


def write_ingest_summary(
    path: Path,
    input_dir: Path,
    output_dir: Path,
    ingested: list[IngestedImage],
    rejected: list[RejectedImage],
) -> None:
    path.write_text(
        "{\n"
        f'  "timestamp": "{datetime.now(UTC).isoformat()}",\n'
        f'  "input_dir": "{input_dir}",\n'
        f'  "output_dir": "{output_dir}",\n'
        f'  "accepted": {len(ingested)},\n'
        f'  "rejected": {len(rejected)}\n'
        "}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
