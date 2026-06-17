"""Render a static manual-review UI for pseudo-label overlays."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REVIEW_COLUMNS = [
    "image_id",
    "image_path",
    "overlay_path",
    "prediction_count",
    "high_confidence_count",
    "predicted_classes",
    "high_confidence_classes",
    "image_condition",
    "mask_errors",
    "next_action",
    "notes",
]

SUMMARY_REQUIRED_COLUMNS = [
    "image_id",
    "image_path",
    "prediction_count",
    "high_confidence_count",
    "predicted_classes",
    "high_confidence_classes",
    "overlay_path",
    "pseudo_label_json",
    "pseudo_label_yolo_txt",
]

IMAGE_CONDITION_OPTIONS = [
    "",
    "optimal",
    "suboptimal_angle",
    "suboptimal_lighting",
    "suboptimal_focus",
    "invalid_content",
]

MASK_ERROR_OPTIONS = [
    "undersegmented",
    "oversegmented",
    "missed_food",
    "background_leak",
    "hallucination",
    "no_prediction",
]

NEXT_ACTION_OPTIONS = [
    "needs_review",
    "annotate_standard",
    "annotate_hard",
    "retake",
    "discard",
]


@dataclass(frozen=True)
class OverlayReviewItem:
    image_id: str
    image_path: str
    overlay_path: str
    prediction_count: int
    high_confidence_count: int
    predicted_classes: str
    high_confidence_classes: str
    pseudo_label_json: str
    pseudo_label_yolo_txt: str
    overlay_exists: bool
    image_condition: str = ""
    mask_errors: tuple[str, ...] = ()
    next_action: str = "needs_review"
    notes: str = ""


def main() -> None:
    args = parse_args()
    items = load_review_items(args.summary)
    render_review(items, args.output_dir, args.summary)
    print(f"Overlay review: {args.output_dir / 'index.html'}")
    print(f"Review template: {args.output_dir / 'review_template.csv'}")
    print(f"Images: {len(items)}")
    print(f"Missing overlays: {sum(not item.overlay_exists for item in items)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("outputs/pseudo_labels/unlabeled_v1/summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reviews/unlabeled_v1_overlay_review"))
    return parser.parse_args(argv)


def load_review_items(summary_path: Path) -> list[OverlayReviewItem]:
    if not summary_path.exists():
        raise FileNotFoundError(f"pseudo-label summary not found: {summary_path}")

    with summary_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing_columns = [column for column in SUMMARY_REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(f"summary is missing required columns: {', '.join(missing_columns)}")
        return [review_item_from_summary_row(row) for row in reader]


def review_item_from_summary_row(row: dict[str, str]) -> OverlayReviewItem:
    prediction_count = int(row["prediction_count"])
    overlay_path = row["overlay_path"]
    return OverlayReviewItem(
        image_id=row["image_id"],
        image_path=row["image_path"],
        overlay_path=overlay_path,
        prediction_count=prediction_count,
        high_confidence_count=int(row["high_confidence_count"]),
        predicted_classes=row["predicted_classes"],
        high_confidence_classes=row["high_confidence_classes"],
        pseudo_label_json=row["pseudo_label_json"],
        pseudo_label_yolo_txt=row["pseudo_label_yolo_txt"],
        overlay_exists=path_exists_from_project_root(overlay_path),
        mask_errors=("no_prediction",) if prediction_count == 0 else (),
    )


def path_exists_from_project_root(path_value: str) -> bool:
    path = Path(path_value)
    if path.is_absolute():
        return path.exists()
    return (PROJECT_ROOT / path).exists()


def render_review(items: list[OverlayReviewItem], output_dir: Path, summary_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_review_template(output_dir / "review_template.csv", items)
    write_review_manifest(output_dir / "review_manifest.json", items, summary_path)
    (output_dir / "index.html").write_text(render_html(items, output_dir, summary_path), encoding="utf-8")


def write_review_template(path: Path, items: list[OverlayReviewItem]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for item in items:
            writer.writerow(review_csv_row(item))


def review_csv_row(item: OverlayReviewItem) -> dict[str, str | int]:
    return {
        "image_id": item.image_id,
        "image_path": item.image_path,
        "overlay_path": item.overlay_path,
        "prediction_count": item.prediction_count,
        "high_confidence_count": item.high_confidence_count,
        "predicted_classes": item.predicted_classes,
        "high_confidence_classes": item.high_confidence_classes,
        "image_condition": item.image_condition,
        "mask_errors": serialize_mask_errors(item.mask_errors),
        "next_action": item.next_action,
        "notes": item.notes,
    }


def serialize_mask_errors(mask_errors: list[str] | tuple[str, ...]) -> str:
    validate_mask_errors(mask_errors)
    return json.dumps(list(mask_errors), ensure_ascii=False)


def parse_mask_errors(value: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"mask_errors must be a JSON array: {value}") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError(f"mask_errors must be a JSON array of strings: {value}")
    validate_mask_errors(parsed)
    return tuple(parsed)


def validate_mask_errors(mask_errors: list[str] | tuple[str, ...]) -> None:
    invalid = sorted(set(mask_errors) - set(MASK_ERROR_OPTIONS))
    if invalid:
        raise ValueError(f"unknown mask_errors values: {', '.join(invalid)}")


def write_review_manifest(path: Path, items: list[OverlayReviewItem], summary_path: Path) -> None:
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "summary_path": str(summary_path),
        "image_count": len(items),
        "missing_overlay_count": sum(not item.overlay_exists for item in items),
        "missing_overlays": [item.overlay_path for item in items if not item.overlay_exists],
        "review_template": "review_template.csv",
        "review_html": "index.html",
        "review_policy": {
            "local_only": True,
            "updates_ground_truth": False,
            "updates_dataset_manifest": False,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def render_html(items: list[OverlayReviewItem], output_dir: Path, summary_path: Path) -> str:
    ui_items = [ui_item(item, output_dir) for item in items]
    class_options = sorted(
        {class_name for item in items for class_name in split_classes(item.predicted_classes) if class_name}
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prato do Dia Overlay Review</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f5f3;
      color: #1f2933;
    }}
    body {{
      margin: 0;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(245, 245, 243, 0.96);
      border-bottom: 1px solid #d8d8d2;
      padding: 16px 24px;
      backdrop-filter: blur(8px);
    }}
    h1 {{
      font-size: 22px;
      margin: 0 0 4px;
      letter-spacing: 0;
    }}
    .meta {{
      color: #5f6b76;
      font-size: 13px;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 14px;
      align-items: end;
    }}
    label {{
      display: grid;
      gap: 5px;
      font-size: 12px;
      font-weight: 650;
      color: #394550;
    }}
    select, input, textarea, button {{
      font: inherit;
      border: 1px solid #c7c9c3;
      border-radius: 6px;
      background: #fff;
      color: #1f2933;
    }}
    select, input {{
      min-height: 36px;
      padding: 6px 8px;
    }}
    input[type="checkbox"] {{
      min-height: auto;
    }}
    button {{
      min-height: 38px;
      padding: 8px 12px;
      cursor: pointer;
      background: #1f5f5b;
      border-color: #1f5f5b;
      color: #fff;
      font-weight: 700;
    }}
    main {{
      padding: 24px;
    }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
      color: #4a5560;
      font-size: 13px;
    }}
    .pill {{
      border: 1px solid #d8d8d2;
      border-radius: 999px;
      background: #fff;
      padding: 5px 9px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #d8d8d2;
      border-radius: 8px;
      overflow: hidden;
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-width: 0;
    }}
    .card.hidden {{
      display: none;
    }}
    .thumb {{
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #20252b;
      display: block;
    }}
    .missing {{
      display: grid;
      place-items: center;
      width: 100%;
      aspect-ratio: 4 / 3;
      background: #2f3740;
      color: #fff;
      font-weight: 700;
    }}
    .content {{
      padding: 12px;
      display: grid;
      gap: 10px;
    }}
    .id {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      word-break: break-all;
    }}
    .kv {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      font-size: 13px;
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-size: 13px;
    }}
    a {{
      color: #1f5f5b;
      font-weight: 650;
    }}
    .review {{
      display: grid;
      gap: 8px;
      padding: 12px;
      border-top: 1px solid #ecece8;
      background: #fafafa;
    }}
    textarea {{
      min-height: 64px;
      padding: 8px;
      resize: vertical;
    }}
    fieldset {{
      border: 1px solid #d7dad4;
      border-radius: 6px;
      display: grid;
      gap: 6px;
      margin: 0;
      padding: 8px;
    }}
    legend {{
      font-size: 12px;
      font-weight: 650;
      color: #394550;
      padding: 0 4px;
    }}
    .check {{
      align-items: center;
      display: flex;
      flex-direction: row;
      gap: 6px;
      font-size: 12px;
      font-weight: 500;
    }}
    .guidance {{
      background: #edf5f2;
      border: 1px solid #c9ded7;
      border-radius: 6px;
      color: #244541;
      font-size: 13px;
      padding: 10px 12px;
      margin-bottom: 16px;
    }}
    .empty {{
      padding: 48px 12px;
      text-align: center;
      color: #66717d;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Prato do Dia Overlay Review</h1>
    <div class="meta">Source: {escape(str(summary_path))} · Local review only · Does not update ground truth</div>
    <div class="toolbar">
      <label>Class
        <select id="classFilter">
          <option value="">All classes</option>
          {render_class_options(class_options)}
        </select>
      </label>
      <label>Image condition
        <select id="conditionFilter">
          <option value="">All conditions</option>
          {render_options([option for option in IMAGE_CONDITION_OPTIONS if option])}
        </select>
      </label>
      <label>Mask error
        <select id="maskErrorFilter">
          <option value="">All mask errors</option>
          {render_options(MASK_ERROR_OPTIONS)}
        </select>
      </label>
      <label>Minimum predictions
        <input id="minPredictions" type="number" min="0" value="0">
      </label>
      <label>
        <span>No detections only</span>
        <input id="noDetectionsOnly" type="checkbox">
      </label>
      <label>
        <span>High confidence only</span>
        <input id="highConfidenceOnly" type="checkbox">
      </label>
      <button id="exportCsv" type="button">Export review CSV</button>
    </div>
  </header>
  <main>
    <div class="guidance">
      annotate_hard means: annotate with ground truth and route to a hard/robustness evaluation set.
      It is not a qualitative holding bucket.
    </div>
    <div class="stats" id="stats"></div>
    <section class="grid" id="grid"></section>
    <div class="empty" id="emptyState" hidden>No cards match the current filters.</div>
  </main>
  <script>
    const reviewItems = {json.dumps(ui_items, ensure_ascii=False)};
    const imageConditionOptions = {json.dumps(IMAGE_CONDITION_OPTIONS)};
    const maskErrorOptions = {json.dumps(MASK_ERROR_OPTIONS)};
    const nextActionOptions = {json.dumps(NEXT_ACTION_OPTIONS)};
    const storageKey = 'prato-do-dia-overlay-review-v2';
    let reviewState = loadState();

    function loadState() {{
      try {{
        return JSON.parse(localStorage.getItem(storageKey) || '{{}}');
      }} catch (_error) {{
        return {{}};
      }}
    }}

    function saveState() {{
      localStorage.setItem(storageKey, JSON.stringify(reviewState));
    }}

    function defaultReview(item) {{
      return {{
        image_condition: item.image_condition || '',
        mask_errors: Array.isArray(item.mask_errors) ? item.mask_errors : [],
        next_action: item.next_action || 'needs_review',
        notes: item.notes || ''
      }};
    }}

    function reviewFor(item) {{
      return {{...defaultReview(item), ...(reviewState[item.image_id] || {{}})}};
    }}

    function optionTags(options, selected) {{
      return options.map(value => {{
        const label = value === '' ? 'Select...' : value;
        const isSelected = value === selected ? ' selected' : '';
        return `<option value="${{escapeAttr(value)}}"${{isSelected}}>${{escapeHtml(label)}}</option>`;
      }}).join('');
    }}

    function maskErrorCheckboxes(item, selected) {{
      const selectedSet = new Set(selected || []);
      return maskErrorOptions.map(value => {{
        const isChecked = selectedSet.has(value) ? ' checked' : '';
        return `<label class="check">
          <input type="checkbox" data-field="mask_errors" data-mask-error="${{escapeAttr(value)}}"
            data-image-id="${{escapeAttr(item.image_id)}}"${{isChecked}}>
          <span>${{escapeHtml(value)}}</span>
        </label>`;
      }}).join('');
    }}

    function renderCards() {{
      const grid = document.getElementById('grid');
      grid.innerHTML = reviewItems.map(item => {{
        const review = reviewFor(item);
        const imageHtml = item.overlay_exists
          ? `<a href="${{escapeAttr(item.overlay_href)}}"><img class="thumb" src="${{escapeAttr(item.overlay_href)}}" alt="Overlay for ${{escapeAttr(item.image_id)}}"></a>`
          : `<div class="missing">Missing overlay</div>`;
        return `
          <article class="card" id="card-${{escapeAttr(item.image_id)}}" data-image-id="${{escapeAttr(item.image_id)}}">
            ${{imageHtml}}
            <div class="content">
              <div class="id">${{escapeHtml(item.image_id)}}</div>
              <div class="kv">
                <div><strong>Predictions</strong><br>${{item.prediction_count}}</div>
                <div><strong>High confidence</strong><br>${{item.high_confidence_count}}</div>
              </div>
              <div><strong>Classes</strong><br>${{escapeHtml(item.predicted_classes || 'none')}}</div>
              <div><strong>High-confidence classes</strong><br>${{escapeHtml(item.high_confidence_classes || 'none')}}</div>
              <div class="links">
                <a href="${{escapeAttr(item.image_href)}}">image</a>
                <a href="${{escapeAttr(item.overlay_href)}}">overlay</a>
                <a href="${{escapeAttr(item.prediction_href)}}">prediction JSON</a>
                <a href="${{escapeAttr(item.pseudo_json_href)}}">pseudo JSON</a>
                <a href="${{escapeAttr(item.pseudo_yolo_href)}}">YOLO TXT</a>
              </div>
            </div>
            <div class="review">
              <label>Image condition
                <select data-field="image_condition" data-image-id="${{escapeAttr(item.image_id)}}">
                  ${{optionTags(imageConditionOptions, review.image_condition)}}
                </select>
              </label>
              <fieldset>
                <legend>Mask errors</legend>
                ${{maskErrorCheckboxes(item, review.mask_errors)}}
              </fieldset>
              <label>Next action
                <select data-field="next_action" data-image-id="${{escapeAttr(item.image_id)}}">
                  ${{optionTags(nextActionOptions, review.next_action)}}
                </select>
              </label>
              <label>Notes
                <textarea data-field="notes" data-image-id="${{escapeAttr(item.image_id)}}">${{escapeHtml(review.notes)}}</textarea>
              </label>
            </div>
          </article>`;
      }}).join('');

      grid.querySelectorAll('select[data-field], textarea[data-field], input[data-field]').forEach(input => {{
        input.addEventListener('input', event => {{
          const target = event.target;
          const imageId = target.dataset.imageId;
          const field = target.dataset.field;
          const item = reviewItems.find(candidate => candidate.image_id === imageId);
          const review = reviewFor(item);
          if (field === 'mask_errors') {{
            const selected = Array.from(document.querySelectorAll(`input[data-field="mask_errors"][data-image-id="${{cssEscape(imageId)}}"]:checked`))
              .map(input => input.dataset.maskError);
            reviewState[imageId] = {{...review, mask_errors: selected}};
          }} else {{
            reviewState[imageId] = {{...review, [field]: target.value}};
          }}
          saveState();
          applyFilters();
        }});
      }});
      applyFilters();
    }}

    function applyFilters() {{
      const classFilter = document.getElementById('classFilter').value;
      const conditionFilter = document.getElementById('conditionFilter').value;
      const maskErrorFilter = document.getElementById('maskErrorFilter').value;
      const minPredictions = Number(document.getElementById('minPredictions').value || 0);
      const noDetectionsOnly = document.getElementById('noDetectionsOnly').checked;
      const highConfidenceOnly = document.getElementById('highConfidenceOnly').checked;
      let visible = 0;
      for (const item of reviewItems) {{
        const card = document.getElementById(`card-${{item.image_id}}`);
        const classes = item.predicted_classes.split(';').filter(Boolean);
        const review = reviewFor(item);
        const errors = Array.isArray(review.mask_errors) ? review.mask_errors : [];
        const show = (!classFilter || classes.includes(classFilter))
          && (!conditionFilter || review.image_condition === conditionFilter)
          && (!maskErrorFilter || errors.includes(maskErrorFilter))
          && item.prediction_count >= minPredictions
          && (!noDetectionsOnly || item.prediction_count === 0)
          && (!highConfidenceOnly || item.high_confidence_count > 0);
        card.classList.toggle('hidden', !show);
        if (show) visible += 1;
      }}
      document.getElementById('emptyState').hidden = visible !== 0;
      renderStats(visible);
    }}

    function renderStats(visible) {{
      const totalPredictions = reviewItems.reduce((sum, item) => sum + item.prediction_count, 0);
      const highConfidence = reviewItems.reduce((sum, item) => sum + item.high_confidence_count, 0);
      const noDetections = reviewItems.filter(item => item.prediction_count === 0).length;
      const missingOverlays = reviewItems.filter(item => !item.overlay_exists).length;
      document.getElementById('stats').innerHTML = [
        `${{visible}} visible`,
        `${{reviewItems.length}} images`,
        `${{totalPredictions}} predictions`,
        `${{highConfidence}} high-confidence`,
        `${{noDetections}} no detections`,
        `${{missingOverlays}} missing overlays`
      ].map(value => `<span class="pill">${{escapeHtml(value)}}</span>`).join('');
    }}

    function exportCsv() {{
      const columns = {json.dumps(REVIEW_COLUMNS)};
      const rows = reviewItems.map(item => {{
        const review = reviewFor(item);
        return {{
          image_id: item.image_id,
          image_path: item.image_path,
          overlay_path: item.overlay_path,
          prediction_count: item.prediction_count,
          high_confidence_count: item.high_confidence_count,
          predicted_classes: item.predicted_classes,
          high_confidence_classes: item.high_confidence_classes,
          image_condition: review.image_condition,
          mask_errors: JSON.stringify(review.mask_errors || []),
          next_action: review.next_action,
          notes: review.notes
        }};
      }});
      const csv = [columns.join(',')].concat(rows.map(row => columns.map(column => csvEscape(row[column])).join(','))).join('\\n') + '\\n';
      const blob = new Blob([csv], {{type: 'text/csv;charset=utf-8'}});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'overlay_review_export.csv';
      link.click();
      URL.revokeObjectURL(link.href);
    }}

    function csvEscape(value) {{
      const text = String(value ?? '');
      if (/[",\\n]/.test(text)) return `"${{text.replaceAll('"', '""')}}"`;
      return text;
    }}

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, char => ({{
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
      }}[char]));
    }}

    function escapeAttr(value) {{
      return escapeHtml(value);
    }}

    function cssEscape(value) {{
      if (window.CSS && CSS.escape) return CSS.escape(value);
      return String(value).replace(/"/g, '\\\\"');
    }}

    for (const id of ['classFilter', 'conditionFilter', 'maskErrorFilter', 'minPredictions', 'noDetectionsOnly', 'highConfidenceOnly']) {{
      document.getElementById(id).addEventListener('input', applyFilters);
    }}
    document.getElementById('exportCsv').addEventListener('click', exportCsv);
    renderCards();
  </script>
</body>
</html>
"""


def render_class_options(class_options: list[str]) -> str:
    return render_options(class_options)


def render_options(options: list[str]) -> str:
    return "\n".join(f'<option value="{escape(option)}">{escape(option)}</option>' for option in options)


def ui_item(item: OverlayReviewItem, output_dir: Path) -> dict[str, object]:
    return {
        **asdict(item),
        "image_href": relative_href(output_dir, item.image_path),
        "overlay_href": relative_href(output_dir, item.overlay_path),
        "prediction_href": relative_href(output_dir, prediction_json_path(item)),
        "pseudo_json_href": relative_href(output_dir, item.pseudo_label_json),
        "pseudo_yolo_href": relative_href(output_dir, item.pseudo_label_yolo_txt),
    }


def prediction_json_path(item: OverlayReviewItem) -> str:
    overlay_path = Path(item.overlay_path)
    predictions_dir = overlay_path.parent.parent / "predictions"
    return str(predictions_dir / f"{item.image_id}_predictions.json")


def relative_href(output_dir: Path, path_value: str) -> str:
    path = Path(path_value)
    target = path if path.is_absolute() else PROJECT_ROOT / path
    return os.path.relpath(target, output_dir).replace(os.sep, "/")


def split_classes(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


if __name__ == "__main__":
    main()
