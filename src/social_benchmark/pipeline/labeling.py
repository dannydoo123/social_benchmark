from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.local_classifier import load_classifier
from social_benchmark.pipeline.models import AspectCategory, EvidenceType, TaskCategory


LABEL_FIELDS = [
    "review_id",
    "source_platform",
    "community_id",
    "thread_id",
    "source_item_id",
    "url",
    "model_id",
    "provider_id",
    "product_id",
    "inference_profile",
    "task_category",
    "aspect_category",
    "evidence_type",
    "claim_type",
    "polarity_score",
    "extractor_confidence",
    "firsthand_flag",
    "regression_flag",
    "hallucination_flag",
    "refusal_flag",
    "value_flag",
    "evidence_text",
    "classifier_task_category",
    "classifier_task_confidence",
    "classifier_aspect_category",
    "classifier_aspect_confidence",
    "classifier_evidence_type",
    "classifier_evidence_confidence",
    "classifier_polarity_score",
    "classifier_polarity_confidence",
    "classifier_firsthand_flag",
    "classifier_firsthand_confidence",
    "classifier_disagreement_count",
    "reviewed_flag",
    "human_excluded_from_scoring",
    "human_exclusion_reason",
    "human_provider_id",
    "human_model_id",
    "human_product_id",
    "human_inference_profile",
    "human_task_category",
    "human_aspect_category",
    "human_evidence_type",
    "human_polarity_score",
    "human_firsthand_flag",
    "human_notes",
]

MIN_SUGGESTION_ACCURACY = 0.65


def export_labeling_queue(
    observations_path: str | Path,
    output_path: str | Path,
    max_rows: int = 200,
    confidence_below: float = 0.72,
    include_neutral: bool = True,
    classifier_model_path: str | Path | None = None,
    raw_items_path: str | Path | None = None,
    context_output_path: str | Path | None = None,
    excluded_review_csv_paths: list[str | Path] | None = None,
) -> int:
    observations = _read_jsonl(observations_path)
    classifier = load_classifier(classifier_model_path) if classifier_model_path else None
    excluded_keys = _reviewed_observation_keys(excluded_review_csv_paths or [])
    raw_by_source_id = _raw_item_index(raw_items_path) if raw_items_path else {}
    candidates = [
        _row_with_classifier_suggestions(observation, classifier, _context_text_for_observation(observation, raw_by_source_id))
        for observation in observations
        if _needs_review(observation, confidence_below=confidence_below, include_neutral=include_neutral)
        and _observation_key(observation) not in excluded_keys
    ]
    candidates = _diverse_priority_order(
        candidates,
    )[:max_rows]
    candidates = [_with_review_id(candidate) for candidate in candidates]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for observation in candidates:
            row = {field: observation.get(field, "") for field in LABEL_FIELDS}
            writer.writerow(row)
    if context_output_path:
        if raw_items_path is None:
            raise ValueError("raw_items_path is required when exporting review context")
        export_review_context(candidates, raw_items_path, context_output_path)
    return len(candidates)


def export_review_context(
    rows: list[dict[str, Any]],
    raw_items_path: str | Path,
    output_path: str | Path,
) -> int:
    raw_by_source_id = {
        str(record.get("source_id") or ""): record
        for record in _read_jsonl(raw_items_path)
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            raw = raw_by_source_id.get(str(row.get("source_item_id") or ""), {})
            record = {
                "review_id": row.get("review_id", ""),
                "source_platform": row.get("source_platform", ""),
                "community_id": row.get("community_id", ""),
                "thread_id": row.get("thread_id", ""),
                "source_item_id": row.get("source_item_id", ""),
                "url": row.get("url", ""),
                "model_id": row.get("model_id", ""),
                "provider_id": row.get("provider_id", ""),
                "product_id": row.get("product_id", ""),
                "inference_profile": row.get("inference_profile", ""),
                "evidence_text": row.get("evidence_text", ""),
                "raw_title": raw.get("title", ""),
                "raw_body": raw.get("body", ""),
                "raw_full_text": _full_text(raw),
                "raw_parent_id": raw.get("parent_id", ""),
                "raw_metadata": raw.get("metadata", {}),
                "machine_label": {
                    "task_category": row.get("task_category", ""),
                    "aspect_category": row.get("aspect_category", ""),
                    "evidence_type": row.get("evidence_type", ""),
                    "claim_type": row.get("claim_type", ""),
                    "polarity_score": row.get("polarity_score", ""),
                    "firsthand_flag": row.get("firsthand_flag", ""),
                    "extractor_confidence": row.get("extractor_confidence", ""),
                },
                "classifier_suggestion": {
                    "task_category": row.get("classifier_task_category", ""),
                    "task_confidence": row.get("classifier_task_confidence", ""),
                    "aspect_category": row.get("classifier_aspect_category", ""),
                    "aspect_confidence": row.get("classifier_aspect_confidence", ""),
                    "evidence_type": row.get("classifier_evidence_type", ""),
                    "evidence_confidence": row.get("classifier_evidence_confidence", ""),
                    "polarity_score": row.get("classifier_polarity_score", ""),
                    "polarity_confidence": row.get("classifier_polarity_confidence", ""),
                    "firsthand_flag": row.get("classifier_firsthand_flag", ""),
                    "firsthand_confidence": row.get("classifier_firsthand_confidence", ""),
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def _needs_review(observation: dict[str, Any], confidence_below: float, include_neutral: bool) -> bool:
    confidence = float(observation.get("extractor_confidence") or 0)
    if confidence < confidence_below:
        return True
    if include_neutral and observation.get("claim_type") == "neutral":
        return True
    return False


def _diverse_priority_order(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_candidates = sorted(
        observations,
        key=lambda item: (
            _priority_bucket(item),
            -int(item.get("classifier_disagreement_count") or 0),
            float(item.get("extractor_confidence") or 0),
            item.get("model_id") or "",
            item.get("aspect_category") or "",
        ),
    )
    selected: list[dict[str, Any]] = []
    remaining = list(sorted_candidates)
    seen_pairs: set[tuple[str, str]] = set()

    while remaining:
        selected_index = 0
        for index, item in enumerate(remaining):
            pair = (str(item.get("model_id") or ""), str(item.get("aspect_category") or ""))
            if pair not in seen_pairs:
                selected_index = index
                break
        item = remaining.pop(selected_index)
        seen_pairs.add((str(item.get("model_id") or ""), str(item.get("aspect_category") or "")))
        selected.append(item)
    return selected


def _priority_bucket(observation: dict[str, Any]) -> int:
    disagreement_count = int(observation.get("classifier_disagreement_count") or 0)
    if disagreement_count >= 2:
        return 0
    confidence = float(observation.get("extractor_confidence") or 0)
    if confidence < 0.60:
        return 1
    if observation.get("claim_type") == "neutral":
        return 2
    if not observation.get("firsthand_flag"):
        return 3
    return 4


def _row_with_classifier_suggestions(
    observation: dict[str, Any],
    classifier: Any | None,
    context_text: str = "",
) -> dict[str, Any]:
    row = dict(observation)
    if classifier is None:
        return row

    prediction = classifier.predict_row(
        {
            **observation,
            "text": str(observation.get("evidence_text") or observation.get("text") or ""),
            "context_text": context_text,
        }
    )
    row["classifier_task_category"] = _suggested_label(
        classifier,
        "task_category",
        _validated_label(prediction.get("task_category", {}).get("label"), {category.value for category in TaskCategory}),
    )
    row["classifier_task_confidence"] = _suggested_confidence(classifier, "task_category", prediction.get("task_category", {}).get("confidence"))
    row["classifier_aspect_category"] = _suggested_label(
        classifier,
        "aspect_category",
        _validated_label(prediction.get("aspect_category", {}).get("label"), {category.value for category in AspectCategory}),
    )
    row["classifier_aspect_confidence"] = _suggested_confidence(
        classifier,
        "aspect_category",
        prediction.get("aspect_category", {}).get("confidence"),
    )
    row["classifier_evidence_type"] = _suggested_label(
        classifier,
        "evidence_type",
        _validated_label(prediction.get("evidence_type", {}).get("label"), {category.value for category in EvidenceType}),
    )
    row["classifier_evidence_confidence"] = _suggested_confidence(
        classifier,
        "evidence_type",
        prediction.get("evidence_type", {}).get("confidence"),
    )
    row["classifier_polarity_score"] = _suggested_label(
        classifier,
        "polarity_score",
        _validated_polarity(prediction.get("polarity_score", {}).get("label")),
    )
    row["classifier_polarity_confidence"] = _suggested_confidence(
        classifier,
        "polarity_score",
        prediction.get("polarity_score", {}).get("confidence"),
    )
    row["classifier_firsthand_flag"] = _suggested_label(
        classifier,
        "firsthand_flag",
        _validated_bool(prediction.get("firsthand_flag", {}).get("label")),
    )
    row["classifier_firsthand_confidence"] = _suggested_confidence(
        classifier,
        "firsthand_flag",
        prediction.get("firsthand_flag", {}).get("confidence"),
    )
    row["classifier_disagreement_count"] = _classifier_disagreement_count(row)
    return row


def _classifier_disagreement_count(row: dict[str, Any]) -> int:
    disagreements = 0
    comparisons = [
        ("task_category", "classifier_task_category"),
        ("aspect_category", "classifier_aspect_category"),
        ("evidence_type", "classifier_evidence_type"),
        ("polarity_score", "classifier_polarity_score"),
        ("firsthand_flag", "classifier_firsthand_flag"),
    ]
    for machine_field, classifier_field in comparisons:
        classifier_value = row.get(classifier_field)
        if classifier_value in (None, ""):
            continue
        if _normalized_value(row.get(machine_field)) != _normalized_value(classifier_value):
            disagreements += 1
    return disagreements


def _validated_label(value: Any, allowed: set[str]) -> str:
    label = str(value or "").strip()
    return label if label in allowed else ""


def _validated_polarity(value: Any) -> str:
    label = str(value or "").strip()
    return label if label in {"-2", "-1", "0", "1", "2"} else ""


def _validated_bool(value: Any) -> str:
    label = str(value or "").strip().lower()
    return label if label in {"true", "false"} else ""


def _rounded_confidence(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _normalized_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value or "").strip().lower()


def _suggested_label(classifier: Any, field: str, value: str) -> str:
    return value if _field_meets_quality_gate(classifier, field) else ""


def _suggested_confidence(classifier: Any, field: str, value: Any) -> str:
    return _rounded_confidence(value) if _field_meets_quality_gate(classifier, field) else ""


def _field_meets_quality_gate(classifier: Any, field: str) -> bool:
    metrics = getattr(classifier, "field_metrics", {}).get(field)
    if not metrics:
        return True
    try:
        return float(metrics.get("accuracy") or 0.0) >= MIN_SUGGESTION_ACCURACY
    except (TypeError, ValueError):
        return False


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _raw_item_index(path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("source_id") or ""): record
        for record in _read_jsonl(path)
    }


def _reviewed_observation_keys(paths: list[str | Path]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not _row_counts_as_reviewed(row):
                    continue
                keys.add(_observation_key(row))
    return keys


def _row_counts_as_reviewed(row: dict[str, Any]) -> bool:
    if _is_truthy(row.get("reviewed_flag")) or _is_truthy(row.get("human_excluded_from_scoring")):
        return True
    human_fields = [
        "human_model_id",
        "human_provider_id",
        "human_product_id",
        "human_inference_profile",
        "human_task_category",
        "human_aspect_category",
        "human_evidence_type",
        "human_polarity_score",
        "human_firsthand_flag",
        "human_notes",
    ]
    return any(str(row.get(field) or "").strip() for field in human_fields)


def _observation_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_item_id") or "").strip(),
        str(row.get("model_id") or "").strip(),
        str(row.get("evidence_text") or row.get("text") or "").strip(),
    )


def _with_review_id(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    if not enriched.get("review_id"):
        enriched["review_id"] = uuid.uuid4().hex
    return enriched


def _full_text(raw: dict[str, Any]) -> str:
    title = str(raw.get("title") or "").strip()
    body = str(raw.get("body") or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def _context_text_for_observation(observation: dict[str, Any], raw_by_source_id: dict[str, dict[str, Any]]) -> str:
    raw = raw_by_source_id.get(str(observation.get("source_item_id") or ""), {})
    return _full_text(raw)


def _is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}
