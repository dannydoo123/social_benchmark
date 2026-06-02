from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.local_classifier import TARGET_FIELDS, load_classifier
from social_benchmark.pipeline.text_features import label_value


def write_threshold_report(
    training_jsonl: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    *,
    thresholds: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90),
) -> dict[str, Any]:
    rows = _read_jsonl(training_jsonl)
    classifier = load_classifier(model_path)
    predictions = _batch_predictions(classifier, rows)
    report = {"examples": len(rows), "model": str(model_path), "thresholds": {}}
    for threshold in thresholds:
        report["thresholds"][f"{threshold:.2f}"] = {
            field: _field_threshold_metrics(rows, predictions, field, threshold)
            for field in TARGET_FIELDS
        }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def export_active_learning_queue(
    input_jsonl: str | Path,
    model_paths: list[str | Path],
    output_path: str | Path,
    *,
    max_rows: int = 200,
) -> int:
    rows = _read_jsonl(input_jsonl)
    classifiers = [(Path(path).stem, load_classifier(path)) for path in model_paths]
    scored = [_active_learning_row(row, classifiers) for row in rows]
    scored.sort(key=lambda row: (-int(row["review_priority_score"]), row.get("source_item_id", "")))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_priority_score",
        "disagreement_fields",
        "low_confidence_fields",
        "source_platform",
        "thread_id",
        "source_item_id",
        "url",
        "model_id",
        "provider_id",
        "product_id",
        "task_category",
        "aspect_category",
        "evidence_type",
        "polarity_score",
        "firsthand_flag",
        "text",
        "model_predictions_json",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in scored[:max_rows]:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return min(max_rows, len(scored))


def _field_threshold_metrics(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    field: str,
    threshold: float,
) -> dict[str, Any]:
    answered = []
    for row, prediction in zip(rows, predictions):
        field_prediction = prediction.get(field, {})
        confidence = float(field_prediction.get("confidence") or 0.0)
        if confidence < threshold:
            continue
        answered.append((label_value(row.get(field)), str(field_prediction.get("label") or ""), confidence))
    correct = sum(actual == predicted for actual, predicted, _ in answered)
    return {
        "threshold": threshold,
        "answered": len(answered),
        "coverage": len(answered) / len(rows) if rows else 0.0,
        "precision": correct / len(answered) if answered else 0.0,
        "mean_confidence": sum(confidence for _, _, confidence in answered) / len(answered) if answered else 0.0,
    }


def _active_learning_row(row: dict[str, Any], classifiers: list[tuple[str, Any]]) -> dict[str, Any]:
    predictions = {name: classifier.predict_row(row) for name, classifier in classifiers}
    disagreement_fields = []
    low_confidence_fields = []
    for field in TARGET_FIELDS:
        labels = {
            str(prediction.get(field, {}).get("label") or "")
            for prediction in predictions.values()
            if prediction.get(field, {}).get("label") not in (None, "")
        }
        confidences = [
            float(prediction.get(field, {}).get("confidence") or 0.0)
            for prediction in predictions.values()
            if field in prediction
        ]
        if len(labels) > 1:
            disagreement_fields.append(field)
        if confidences and max(confidences) < 0.65:
            low_confidence_fields.append(field)
    priority = len(disagreement_fields) * 3 + len(low_confidence_fields)
    return {
        **row,
        "text": row.get("text") or row.get("evidence_text") or "",
        "review_priority_score": priority,
        "disagreement_fields": ",".join(disagreement_fields),
        "low_confidence_fields": ",".join(low_confidence_fields),
        "model_predictions_json": json.dumps(predictions, sort_keys=True),
    }


def _batch_predictions(classifier: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if classifier.__class__.__name__ == "HFEmbeddingClassifier":
        from social_benchmark.pipeline.hf_classifier import _predict_field
        from social_benchmark.pipeline.text_features import model_text

        vectors = classifier.embedder.encode([model_text(row) for row in rows])
        return [
            {field: _predict_field(model, vector) for field, model in classifier.models.items()}
            for vector in vectors
        ]
    return [classifier.predict_row(row) for row in rows]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
