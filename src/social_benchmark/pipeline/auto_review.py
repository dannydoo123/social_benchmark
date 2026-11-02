from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.labeling import LABEL_FIELDS
from social_benchmark.pipeline.local_classifier import TARGET_FIELDS, load_classifier
from social_benchmark.pipeline.models import AspectCategory, EvidenceType, TaskCategory

ABSTAIN_LABEL = "__abstain__"
DEFAULT_MODEL_CANDIDATES = (
    Path("datasets/training/reddit_auto_review_hn_round8.joblib"),
    Path("datasets/training/high_precision_ensemble.joblib"),
    Path("datasets/training/local_nb_model.json"),
)
FIELD_TO_HUMAN_FIELD = {
    "task_category": "human_task_category",
    "aspect_category": "human_aspect_category",
    "evidence_type": "human_evidence_type",
    "polarity_score": "human_polarity_score",
    "firsthand_flag": "human_firsthand_flag",
}
FIELD_CONFIDENCE_GATES = {
    "task_category": 0.68,
    "aspect_category": 0.70,
    "evidence_type": 0.70,
    "polarity_score": 0.80,
    "firsthand_flag": 0.80,
}
VALID_LABELS = {
    "task_category": {item.value for item in TaskCategory},
    "aspect_category": {item.value for item in AspectCategory},
    "evidence_type": {item.value for item in EvidenceType},
    "polarity_score": {"-2", "-1", "0", "1", "2"},
    "firsthand_flag": {"true", "false"},
}


def auto_review_labels(
    *,
    labels_csv: str | Path,
    output_csv: str | Path,
    report_path: str | Path,
    model_path: str | Path | None = None,
    context_jsonl: str | Path | None = None,
    source_platform: str = "reddit",
    min_extractor_confidence: float = 0.72,
    min_model_confidence: float = 0.0,
    max_iterations: int = 3,
    accept_neutral: bool = False,
) -> dict[str, Any]:
    """Convert review queue rows into strict model-reviewed CSV rows.

    This is intentionally high precision: rows are accepted only when the
    extractor and trained classifier agree on every scoring field. All other
    rows are marked excluded so they cannot enter training by accident.
    """
    rows = _read_csv(labels_csv)
    context = _context_index(context_jsonl) if context_jsonl else {}
    resolved_model_path = Path(model_path) if model_path else _default_model_path()
    classifier = load_classifier(resolved_model_path)

    reviewed_rows: list[dict[str, Any]] = []
    previous_signature: list[tuple[Any, ...]] | None = None
    stable = False
    iterations_run = 0
    for iteration in range(max(1, max_iterations)):
        iterations_run = iteration + 1
        reviewed_rows = [
            _review_row(
                row,
                classifier,
                context.get(str(row.get("review_id") or "")),
                source_platform=source_platform,
                min_extractor_confidence=min_extractor_confidence,
                min_model_confidence=min_model_confidence,
                accept_neutral=accept_neutral,
            )
            for row in rows
        ]
        signature = [_decision_signature(row) for row in reviewed_rows]
        if signature == previous_signature:
            stable = True
            break
        previous_signature = signature

    _write_csv(output_csv, reviewed_rows)
    report = _report(
        rows=reviewed_rows,
        model_path=resolved_model_path,
        iterations_run=iterations_run,
        stable=stable or iterations_run == 1,
        min_extractor_confidence=min_extractor_confidence,
        min_model_confidence=min_model_confidence,
        accept_neutral=accept_neutral,
        classifier=classifier,
    )
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def auto_review_observations(
    *,
    observations_jsonl: str | Path,
    output_csv: str | Path,
    report_path: str | Path,
    model_path: str | Path | None = None,
    context_jsonl: str | Path | None = None,
    source_platform: str = "reddit",
    min_extractor_confidence: float = 0.72,
    min_model_confidence: float = 0.0,
    max_iterations: int = 3,
    accept_neutral: bool = False,
) -> dict[str, Any]:
    rows = [_observation_to_label_row(row, index) for index, row in enumerate(_read_jsonl(observations_jsonl), start=1)]
    temp_labels = Path(output_csv).with_suffix(".input_rows.tmp.csv")
    _write_csv(temp_labels, rows)
    try:
        return auto_review_labels(
            labels_csv=temp_labels,
            output_csv=output_csv,
            report_path=report_path,
            model_path=model_path,
            context_jsonl=context_jsonl,
            source_platform=source_platform,
            min_extractor_confidence=min_extractor_confidence,
            min_model_confidence=min_model_confidence,
            max_iterations=max_iterations,
            accept_neutral=accept_neutral,
        )
    finally:
        try:
            temp_labels.unlink()
        except OSError:
            pass


def _review_row(
    row: dict[str, Any],
    classifier: Any,
    context_row: dict[str, Any] | None,
    *,
    source_platform: str,
    min_extractor_confidence: float,
    min_model_confidence: float,
    accept_neutral: bool,
) -> dict[str, Any]:
    reviewed = dict(row)
    reviewed["reviewed_flag"] = ""
    reviewed["human_excluded_from_scoring"] = ""
    reviewed["human_exclusion_reason"] = ""
    reviewed["human_notes"] = ""

    exclusion = _precheck_exclusion(
        reviewed,
        source_platform=source_platform,
        min_extractor_confidence=min_extractor_confidence,
        accept_neutral=accept_neutral,
    )
    if exclusion:
        return _exclude(reviewed, exclusion)

    prediction = classifier.predict_row(
        {
            **reviewed,
            "text": str(reviewed.get("evidence_text") or ""),
            "context_text": _context_text(context_row),
        }
    )
    for field in TARGET_FIELDS:
        predicted = prediction.get(field) or {}
        predicted_label = _normalized_value(predicted.get("label"))
        if predicted_label in {"", ABSTAIN_LABEL}:
            return _exclude(reviewed, f"model_abstained_{field}")
        if predicted_label not in VALID_LABELS[field]:
            return _exclude(reviewed, f"invalid_model_label_{field}")
        confidence = _float(predicted.get("confidence"))
        gate = max(FIELD_CONFIDENCE_GATES[field], min_model_confidence)
        if confidence < gate:
            return _exclude(reviewed, f"low_model_confidence_{field}")
        machine_label = _normalized_value(reviewed.get(field))
        if predicted_label != machine_label:
            return _exclude(reviewed, f"model_disagreement_{field}")

    reviewed["reviewed_flag"] = "true"
    reviewed["human_provider_id"] = str(reviewed.get("provider_id") or "").strip()
    reviewed["human_model_id"] = str(reviewed.get("model_id") or "").strip()
    reviewed["human_product_id"] = str(reviewed.get("product_id") or "").strip()
    reviewed["human_inference_profile"] = str(reviewed.get("inference_profile") or "").strip()
    for field, human_field in FIELD_TO_HUMAN_FIELD.items():
        reviewed[human_field] = _normalized_value(reviewed.get(field))
    reviewed["human_notes"] = "auto_review:model_agreed_with_extractor"
    return reviewed


def _precheck_exclusion(
    row: dict[str, Any],
    *,
    source_platform: str,
    min_extractor_confidence: float,
    accept_neutral: bool,
) -> str:
    if source_platform and _normalized_value(row.get("source_platform")) != source_platform:
        return "source_platform_mismatch"
    if not _normalized_value(row.get("provider_id")):
        return "missing_provider_id"
    if not _normalized_value(row.get("model_id")):
        return "missing_model_id"
    if len(str(row.get("evidence_text") or "").split()) < 6:
        return "short_evidence"
    if not accept_neutral and _normalized_value(row.get("claim_type")) == "neutral":
        return "neutral_claim"
    if _float(row.get("extractor_confidence")) < min_extractor_confidence:
        return "low_extractor_confidence"
    for field in TARGET_FIELDS:
        if _normalized_value(row.get(field)) not in VALID_LABELS[field]:
            return f"invalid_extractor_label_{field}"
    return ""


def _exclude(row: dict[str, Any], reason: str) -> dict[str, Any]:
    excluded = dict(row)
    excluded["reviewed_flag"] = ""
    excluded["human_excluded_from_scoring"] = "true"
    excluded["human_exclusion_reason"] = reason
    excluded["human_notes"] = f"auto_review_excluded:{reason}"
    return excluded


def _report(
    *,
    rows: list[dict[str, Any]],
    model_path: Path,
    iterations_run: int,
    stable: bool,
    min_extractor_confidence: float,
    min_model_confidence: float,
    accept_neutral: bool,
    classifier: Any,
) -> dict[str, Any]:
    accepted = [row for row in rows if _is_truthy(row.get("reviewed_flag"))]
    excluded = [row for row in rows if _is_truthy(row.get("human_excluded_from_scoring"))]
    reasons = Counter(str(row.get("human_exclusion_reason") or "") for row in excluded)
    return {
        "rows": len(rows),
        "accepted": len(accepted),
        "excluded": len(excluded),
        "acceptance_rate_pct": _pct(len(accepted), len(rows)),
        "iterations_run": iterations_run,
        "stable": stable,
        "model_path": str(model_path),
        "gates": {
            "min_extractor_confidence": min_extractor_confidence,
            "min_model_confidence": min_model_confidence,
            "field_confidence_gates": FIELD_CONFIDENCE_GATES,
            "accept_neutral": accept_neutral,
            "require_model_extractor_agreement": True,
        },
        "exclusion_reasons": dict(sorted(reasons.items())),
        "model_field_quality": _model_quality(classifier),
    }


def _model_quality(classifier: Any) -> dict[str, Any]:
    quality: dict[str, Any] = {}
    for field, metrics in getattr(classifier, "field_metrics", {}).items():
        if not isinstance(metrics, dict):
            continue
        quality[field] = {
            key: metrics[key]
            for key in ("accuracy", "precision", "coverage", "answered", "abstained", "total")
            if key in metrics
        }
    return quality


def _decision_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _normalized_value(row.get("reviewed_flag")),
        _normalized_value(row.get("human_excluded_from_scoring")),
        _normalized_value(row.get("human_exclusion_reason")),
        tuple(_normalized_value(row.get(field)) for field in FIELD_TO_HUMAN_FIELD.values()),
    )


def _default_model_path() -> Path:
    for path in DEFAULT_MODEL_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No default review model found; pass --model explicitly")


def _observation_to_label_row(observation: dict[str, Any], index: int) -> dict[str, Any]:
    row = {field: observation.get(field, "") for field in LABEL_FIELDS}
    row["review_id"] = str(observation.get("review_id") or f"auto-{index:06d}")
    row["source_platform"] = str(observation.get("source_platform") or "")
    row["community_id"] = str(observation.get("community_id") or "")
    row["thread_id"] = str(observation.get("thread_id") or "")
    row["source_item_id"] = str(observation.get("source_item_id") or "")
    row["url"] = str(observation.get("url") or "")
    row["model_id"] = str(observation.get("model_id") or "")
    row["provider_id"] = str(observation.get("provider_id") or "")
    row["product_id"] = str(observation.get("product_id") or "")
    row["inference_profile"] = str(observation.get("inference_profile") or "")
    row["task_category"] = _normalized_value(observation.get("task_category"))
    row["aspect_category"] = _normalized_value(observation.get("aspect_category"))
    row["evidence_type"] = _normalized_value(observation.get("evidence_type"))
    row["claim_type"] = _normalized_value(observation.get("claim_type"))
    row["polarity_score"] = _normalized_value(observation.get("polarity_score"))
    row["extractor_confidence"] = str(observation.get("extractor_confidence") or "")
    row["firsthand_flag"] = _normalized_value(observation.get("firsthand_flag"))
    row["evidence_text"] = str(observation.get("evidence_text") or observation.get("text") or "")
    return row


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(LABEL_FIELDS)
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _context_index(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {str(row.get("review_id") or ""): row for row in rows if row.get("review_id")}


def _context_text(context_row: dict[str, Any] | None) -> str:
    if not context_row:
        return ""
    return str(context_row.get("raw_full_text") or "").strip()


def _normalized_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value or "").strip().lower()


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round((numerator / denominator) * 100, 2)


def _is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}
