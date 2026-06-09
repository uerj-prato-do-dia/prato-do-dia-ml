from __future__ import annotations

import json

from prato_do_dia_ml.experiment import compare_experiments


def test_compare_experiments_reads_metrics(tmp_path) -> None:
    metrics_path = tmp_path / "smoke" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "image_count": 2,
                    "mean_runtime_seconds": 1.25,
                    "peak_rss_mb": 512.0,
                    "mean_instance_iou": 0.5,
                    "mean_dice": 0.66,
                    "false_positives": 1,
                    "missed_regions": 3,
                },
                "images": [],
            }
        ),
        encoding="utf-8",
    )

    rows = compare_experiments(tmp_path)

    assert len(rows) == 1
    assert rows[0]["experiment"] == "smoke"
    assert rows[0]["mean_iou"] == 0.5
