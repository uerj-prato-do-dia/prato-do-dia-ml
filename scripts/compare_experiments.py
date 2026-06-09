"""Compare metrics across saved experiment runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prato_do_dia_ml.experiment import compare_experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/experiments/benchmark.csv"))
    args = parser.parse_args()

    rows = compare_experiments(args.experiments_dir, args.output_csv)
    if not rows:
        raise FileNotFoundError(f"no metrics.json files found in {args.experiments_dir}")

    print(_markdown_table(rows))
    print(f"saved {args.output_csv}")


def _markdown_table(rows: list[dict[str, object]]) -> str:
    fields = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row[field]) for field in fields) + " |")
    return "\n".join(lines)


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
