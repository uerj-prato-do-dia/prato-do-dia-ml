# AGENTS.md

## Project Context: Prato do Dia
"Prato do Dia" is a dietary tracking application. Users take top-down photos of their meals using an in-app UI overlay to center the plate. The backend processes these images to identify food items, estimate quantities, calculate nutritional proportions, and log the data into a nutritional history database.

## Current Engineering Focus
1.  **Algorithm/Model Pipeline:** Implement a hybrid computer vision pipeline using `YOLOv11` (for food detection and bounding box generation) combined with `SAM 2` (Segment Anything Model 2 for precise masking). 
2.  **Preprocessing Standardization:** Strict separation of concerns between client-side compression and server-side tensor preparation.
3.  **Hardware Optimization:** Local development relies strictly on CPU execution. The pipeline must export and run the final models using ONNX Runtime to minimize inference latency.
4.  **Data Annotation:** Output segmentations in YOLO TXT format to simplify Git versioning and pipeline integration.
5.  **Architecture & Refactoring:** Simplify the repository using Occam's razor. Eliminate unnecessary abstractions.
6.  **Dependency Management:** Use `uv` and `pyproject.toml` exclusively.
7.  **Phase 3:** Metric evaluation (Intersection over Union) and visual validation of SAM 2 generated masks against deterministic ground truth.

## Repository Layout & Conventions
*   **Simplicity (Occam's Razor):** Keep the directory structure flat and intuitive. Do not over-engineer the initial architecture.
*   **Version Control:** DO NOT create an `old`, `archive`, or `deprecated` folder. Delete dead code and rely on Git history.
*   **Changelog:** All structural and model changes must be documented in `CHANGELOG.md` following standard "Keep a Changelog" formatting.
*   **Naming:** Use `snake_case` for Python scripts and directories. Use `PascalCase` for classes.

## Agent Workflow Guidelines

### 1. Refactoring & Environment Tasks
*   **Goal:** Clean up the repository and manage dependencies modernly.
*   **Constraints:** 
    *   Delete unused files instead of moving them. Ensure no breaking changes occur in the active execution path. 
    *   Update `CHANGELOG.md` with removed modules.
    *   Manage dependencies using `uv add <package>` instead of `pip install`. Do not use or maintain a `requirements.txt` file; rely on `pyproject.toml` and `uv.lock`.
*   **Done when:** The codebase contains only the necessary files for the active pipeline, no dead code exists, and the environment is entirely reproducible via `uv sync`.

### 2. Preprocessing Pipeline Implementation
*   **Goal:** Implement robust server-side image preprocessing before model inference.
*   **Constraints:**
    *   Assume the mobile client handles EXIF rotation and initial downscaling.
    *   Implement **Letterboxing**: Resize images while maintaining aspect ratio and padding the rest to fit the exact required square resolution (e.g., 640x640 for YOLO, 1024x1024 for SAM 2). Do not use standard `cv2.resize` that distorts aspect ratios.
    *   Ensure proper channel ordering (BGR to RGB conversion if using OpenCV).
    *   Apply necessary tensor normalization specific to the ONNX exported weights.
*   **Done when:** Raw rectangular images are correctly letterboxed, converted to RGB, and normalized into tensors without geometric distortion.

### 3. YOLOv11 + SAM 2 Segmentation Pipeline
*   **Goal:** Set up the segmentation, auto-annotation, and manual correction pipeline optimized for CPU inference.
*   **Constraints:** 
    *   The pipeline must process top-down images sequentially: YOLOv11 detects food items and generates bounding boxes -> Bounding boxes are fed as prompts to SAM 2 -> SAM 2 generates precise masks.
    *   **Inference Engine:** Once weights are defined, native PyTorch inference must be replaced with `onnxruntime` to handle severe CPU bottlenecks.
    *   **Annotation Format:** Outputs from SAM 2 must be saved strictly in **YOLO Segmentation (TXT)** format (class_id x1 y1 x2 y2 ...) inside a `data/raw_segmentations/` directory to prevent Git conflicts and simplify the integration with YOLO training.
    *   Feature extraction logic must be isolated in its own module (e.g., `src/feature_extraction.py`).
*   **Done when:** An input image successfully passes through YOLOv11 and SAM 2 (via ONNX Runtime), produces a YOLO TXT segmentation mask, saves the annotations to disk, and features can be successfully extracted from the labeled data.

## Review and Verification
Before finalizing any code changes:
1. Verify that the changes strictly adhere to the Occam's razor principle (simplest working implementation).
2. Ensure no dead code was "archived" instead of deleted.
3. Confirm that all dependency changes are reflected in `pyproject.toml` via `uv`.
4. Ensure preprocessing scripts explicitly use Letterboxing to prevent aspect ratio distortion.
