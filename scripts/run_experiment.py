"""Run a reproducible YOLO11 + SAM 2 segmentation experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    output_dir = run_experiment(
        config=load_config(args.config),
        experiment_name=args.experiment_name,
        outputs_dir=args.outputs_dir,
        limit=args.limit,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"saved experiment to {output_dir}")


if __name__ == "__main__":
    main()
