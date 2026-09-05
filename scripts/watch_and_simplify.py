#!/usr/bin/env python3
"""Real-time Auto-Simplifier Daemon for X-AnyLabeling.

Monitors data/processed_640 for newly saved .json annotation files from X-AnyLabeling
and automatically applies Douglas-Peucker polygon simplification in real time.

Usage:
    python3 prato-do-dia-ml/scripts/watch_and_simplify.py --labels-dir data/processed_640 --epsilon-ratio 0.008
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add scripts directory and package root to sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR.parent))

try:
    from simplify_yolo_labels import (
        convert_labelme_to_yolo_segmentation,
        simplify_json_file,
    )
except ImportError:
    from scripts.simplify_yolo_labels import (
        convert_labelme_to_yolo_segmentation,
        simplify_json_file,
    )
except ModuleNotFoundError:
    from simplify_yolo_labels import (
        convert_labelme_to_yolo_segmentation,
        simplify_json_file,
    )

PROJECT_ROOT = SCRIPTS_DIR.parent.parent
DEFAULT_LABELS_DIR = PROJECT_ROOT / "data" / "processed" / "labels"


def start_file_watcher(labels_dir: Path, epsilon_ratio: float = 0.008, interval: float = 1.0) -> None:
    labels_dir = labels_dir.resolve()
    if not labels_dir.exists():
        print(f"Directory not found: {labels_dir}")
        return

    print("========================================================")
    print("   AUTO-SIMPLIFICADOR EM TEMPO REAL PARA X-ANYLABELING")
    print("========================================================")
    print(f" Monitorando pasta:  {labels_dir}")
    print(f" Tolerância epsilon: {epsilon_ratio}")
    print(" Status:            Aguardando salvamentos no X-AnyLabeling...\n")

    # Store last modification times
    mtimes: dict[Path, float] = {}

    try:
        while True:
            json_files = sorted(labels_dir.glob("*.json"))
            for json_path in json_files:
                if json_path.name.endswith(".bak"):
                    continue

                current_mtime = json_path.stat().st_mtime
                last_mtime = mtimes.get(json_path)

                # Process if file is new or modified
                if last_mtime is None or current_mtime > last_mtime:
                    mtimes[json_path] = current_mtime
                    try:
                        polys, orig_v, simp_v = simplify_json_file(
                            json_path, epsilon_ratio=epsilon_ratio, backup=True, dry_run=False
                        )
                        if polys > 0 and orig_v != simp_v:
                            # Also keep matching YOLO TXT in sync
                            convert_labelme_to_yolo_segmentation(json_path, labels_dir)
                            # Update stored mtime after write
                            mtimes[json_path] = json_path.stat().st_mtime
                            red = ((orig_v - simp_v) / orig_v * 100.0) if orig_v > 0 else 0.0
                            print(
                                f"⚡ [{time.strftime('%H:%M:%S')}] Auto-simplificado: {json_path.name} ({orig_v} -> {simp_v} vértices, -{red:.1f}%)"
                            )
                    except Exception as exc:
                        print(f"Erro ao processar {json_path.name}: {exc}")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nAuto-simplificador finalizado pelo usuário.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Auto-Simplifier Daemon for X-AnyLabeling.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=DEFAULT_LABELS_DIR,
        help="Directory to watch for .json files",
    )
    parser.add_argument(
        "--epsilon-ratio",
        type=float,
        default=0.008,
        help="Tolerance ratio (default: 0.008)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)",
    )

    args = parser.parse_args()
    start_file_watcher(args.labels_dir, epsilon_ratio=args.epsilon_ratio, interval=args.interval)
