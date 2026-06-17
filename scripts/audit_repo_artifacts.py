"""Dry-run audit for large model, binary, and log artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

MODEL_EXTENSIONS = {
    ".bin",
    ".ckpt",
    ".engine",
    ".h5",
    ".joblib",
    ".onnx",
    ".pb",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
}

LOG_EXTENSIONS = {
    ".log",
    ".jsonl",
    ".out",
    ".trace",
}

BINARY_OR_ARCHIVE_EXTENSIONS = {
    ".7z",
    ".arrow",
    ".gz",
    ".npy",
    ".npz",
    ".parquet",
    ".tar",
    ".tgz",
    ".zip",
}

DEFAULT_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}


@dataclass(frozen=True)
class ArtifactFinding:
    path: str
    size_mb: float
    modified_at: str
    category: str


def main() -> None:
    args = parse_args()
    findings = scan_artifacts(
        root=args.root,
        min_size_mb=args.min_size_mb,
        skip_dirs=set(args.skip_dir),
    )
    write_findings(findings, args.output_format, args.output)
    if args.output is None:
        print_summary(findings, args.min_size_mb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List large model weights, binary artifacts, and logs. Dry-run only; deletes nothing."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root to scan.")
    parser.add_argument("--min-size-mb", type=float, default=50.0, help="Minimum file size to report.")
    parser.add_argument(
        "--output-format",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output file.")
    parser.add_argument(
        "--skip-dir",
        action="append",
        default=sorted(DEFAULT_SKIP_DIRS),
        help="Directory name to skip. Can be passed multiple times.",
    )
    return parser.parse_args()


def scan_artifacts(root: Path, min_size_mb: float, skip_dirs: set[str]) -> list[ArtifactFinding]:
    root = root.resolve()
    min_size_bytes = int(min_size_mb * 1024 * 1024)
    findings: list[ArtifactFinding] = []
    for path in sorted(root.rglob("*")):
        if should_skip(path, root, skip_dirs) or not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError as exc:
            print(f"warning: cannot stat {path}: {exc}", file=sys.stderr)
            continue
        if stat.st_size < min_size_bytes:
            continue
        category = categorize(path)
        if category == "other_large_file":
            continue
        findings.append(
            ArtifactFinding(
                path=path.relative_to(root).as_posix(),
                size_mb=round(stat.st_size / 1024 / 1024, 2),
                modified_at=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                category=category,
            )
        )
    return sorted(findings, key=lambda item: (-item.size_mb, item.path))


def should_skip(path: Path, root: Path, skip_dirs: set[str]) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in skip_dirs for part in relative_parts[:-1])


def categorize(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in MODEL_EXTENSIONS:
        return "model_weight_or_checkpoint"
    if suffix in LOG_EXTENSIONS:
        return "large_log"
    if suffix in BINARY_OR_ARCHIVE_EXTENSIONS:
        return "large_binary_or_archive"
    return "other_large_file"


def write_findings(findings: list[ArtifactFinding], output_format: str, output: Path | None) -> None:
    if output_format == "json":
        payload = json.dumps([asdict(finding) for finding in findings], indent=2)
    elif output_format == "csv":
        payload = findings_to_csv(findings)
    else:
        payload = findings_to_table(findings)

    if output is None:
        print(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")


def findings_to_csv(findings: list[ArtifactFinding]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["path", "size_mb", "modified_at", "category"], lineterminator="\n")
    writer.writeheader()
    for finding in findings:
        writer.writerow(asdict(finding))
    return buffer.getvalue().rstrip("\n")


def findings_to_table(findings: list[ArtifactFinding]) -> str:
    if not findings:
        return "No large model, binary, or log artifacts found."
    rows = [["size_mb", "modified_at", "category", "path"]]
    rows.extend(
        [[f"{finding.size_mb:.2f}", finding.modified_at, finding.category, finding.path] for finding in findings]
    )
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return "\n".join("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows)


def print_summary(findings: list[ArtifactFinding], min_size_mb: float) -> None:
    print()
    print(f"Findings: {len(findings)} files >= {min_size_mb:.1f} MB")
    print("Dry-run only: inspect paths before deleting or moving anything.")


if __name__ == "__main__":
    main()
