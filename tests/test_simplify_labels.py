import math

import numpy as np

from scripts.simplify_yolo_labels import simplify_polygon


def test_simplify_noisy_polygon() -> None:
    """Test Douglas-Peucker simplification on a synthetic 100-point noisy circle."""
    num_points = 100
    center_x, center_y = 0.5, 0.5
    radius = 0.3

    # Generate 100-vertex circle polygon with small high-frequency noise
    noisy_coords: list[float] = []
    np.random.seed(42)
    for i in range(num_points):
        angle = (2.0 * math.pi * i) / num_points
        # Add micro-noise (+/- 0.005) to simulate raw SAM 2 edge jitter
        noise_r = radius + float(np.random.uniform(-0.005, 0.005))
        px = max(0.0, min(1.0, center_x + noise_r * math.cos(angle)))
        py = max(0.0, min(1.0, center_y + noise_r * math.sin(angle)))
        noisy_coords.extend([px, py])

    orig_count = len(noisy_coords) // 2
    assert orig_count == 100

    target_class_id = 4  # arroz
    cid, simp_coords, orig_v, simp_v = simplify_polygon(
        class_id=target_class_id,
        coords=noisy_coords,
        epsilon_ratio=0.0025,
        img_w=640.0,
        img_h=640.0,
    )

    # 1. Assert retention of canonical class ID
    assert cid == target_class_id

    # 2. Assert vertex reduction (100 points reduced significantly to < 30)
    assert orig_v == 100
    assert simp_v < orig_v
    assert 4 <= simp_v <= 30

    # 3. Assert all simplified coordinates remain within normalized bounds [0.0, 1.0]
    for val in simp_coords:
        assert 0.0 <= val <= 1.0


def test_convert_labelme_to_yolo_segmentation(tmp_path) -> None:
    """Test Labelme JSON conversion with accent removal, space sanitization, and warning log."""
    import json

    from scripts.simplify_yolo_labels import convert_labelme_to_yolo_segmentation

    json_data = {
        "imageWidth": 640,
        "imageHeight": 640,
        "shapes": [
            {
                "label": "0: tomate",
                "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
            },
            {
                "label": "Frango Grelhado",  # Needs sanitization to frango_grelhado (class 11)
                "points": [[100, 100], [200, 100], [200, 200]],
            },
            {
                "label": "categoria_invalida_xyz",  # Unknown label, should log warning and skip
                "points": [[0, 0], [10, 10], [20, 20]],
            },
        ],
    }

    json_path = tmp_path / "test_meal.json"
    json_path.write_text(json.dumps(json_data), encoding="utf-8")

    txt_path = convert_labelme_to_yolo_segmentation(json_path, tmp_path)
    assert txt_path.exists()

    lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # 2 valid shapes converted, 1 invalid skipped

    # Check class IDs: 0 (tomate) and 11 (frango_grelhado)
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("11 ")
