from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.classifier_experiments import EmbeddingVectorCache, grouped_holdout_indexes
from social_benchmark.pipeline.local_classifier import TARGET_FIELDS
from social_benchmark.pipeline.models import (
    AspectCategory,
    ClaimType,
    EvidenceType,
    Observation,
    SourcePlatform,
    TaskCategory,
    WeightComponents,
)
from social_benchmark.pipeline.routed_classifier import DEFAULT_FIELD_CONFIG, _build_features_by_field
from social_benchmark.pipeline.scoring import ScoreAggregator
from social_benchmark.pipeline.structured_classifier import _field_model
from social_benchmark.pipeline.text_features import label_value, model_text

# Calibrated-confidence gates with measured >=80% out-of-fold precision
# (datasets/evaluation/gated_precision_*_round8_2026-06-12.json).
DEFAULT_GATES = {
    "firsthand_flag": 0.50,
    "aspect_category": 0.70,
    "evidence_type": 0.90,
    "task_category": 0.80,
    "polarity_score": 0.80,  # sign-level (3-class) head
}

EVIDENCE_SPAN_LIMIT = 300
EVIDENCE_SAMPLES_PER_CELL = 4


def build_score_snapshot(
    *,
    training_jsonl: str | Path,
    observation_paths: list[str | Path],
    reviewed_paths: list[str | Path],
    output_path: str | Path,
    field_config: dict[str, dict[str, Any]] | None = None,
    gates: dict[str, float] | None = None,
    embedding_cache_dir: str | Path | None = None,
    methodology_paths: list[str | Path] | None = None,
    calibration_runs: int = 4,
    web_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a versioned, gate-filtered score snapshot for the dashboard.

    Human-reviewed rows enter scoring with their reviewed labels. Unreviewed
    rows enter only for fields whose calibrated classifier confidence clears
    the per-field gate; polarity uses the sign-level head and never assigns
    machine magnitudes.
    """
    gates = {**DEFAULT_GATES, **(gates or {})}
    config = field_config or DEFAULT_FIELD_CONFIG
    training_rows = _read_jsonl(training_jsonl)
    reviewed = _load_reviewed_rows(reviewed_paths)
    raw_observations = _load_observations(observation_paths)

    human_rows, machine_candidates, dropped_by_review = _split_by_review(raw_observations, reviewed)
    machine_rows, machine_rejected = _gate_machine_rows(
        machine_candidates,
        training_rows,
        config,
        gates,
        embedding_cache_dir=embedding_cache_dir,
        calibration_runs=calibration_runs,
    )
    pool = [_to_observation(row) for row in human_rows] + [_to_observation(row) for row, _ in machine_rows]
    task_trusted_flags = [True] * len(human_rows) + [details["task_trusted"] for _, details in machine_rows]

    aggregator = ScoreAggregator()
    leaderboard = _leaderboard_rows(aggregator, pool)
    overall = aggregator.overall_scores(pool)
    tasks = _per_task_rows(aggregator, pool, task_trusted_flags)
    evidence = _evidence_samples(pool)

    snapshot = {
        "snapshot_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gates": gates,
        "corpus": {
            "human_reviewed_included": len(human_rows),
            "machine_included": len(machine_rows),
            "machine_rejected_by_gates": machine_rejected,
            "excluded_by_review": dropped_by_review,
            "training_examples": len(training_rows),
            "models": len({row.model_id for row in pool}),
            "threads": len({row.thread_id for row in pool if row.thread_id}),
        },
        "overall": [
            {"model_id": model_id, "score": round(score, 2)}
            for model_id, score in sorted(overall.items(), key=lambda item: item[1], reverse=True)
        ],
        "leaderboard": leaderboard,
        "tasks": tasks,
        "evidence": evidence,
        "methodology": _methodology_block(methodology_paths or [], gates, len(training_rows)),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2, sort_keys=False), encoding="utf-8")
    if web_output_path:
        web_output = Path(web_output_path)
        web_output.parent.mkdir(parents=True, exist_ok=True)
        web_output.write_text(json.dumps(snapshot), encoding="utf-8")
    return snapshot


def _split_by_review(
    raw_observations: list[dict[str, Any]],
    reviewed: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    human_rows: list[dict[str, Any]] = []
    machine_candidates: list[dict[str, Any]] = []
    dropped = 0
    for row in raw_observations:
        key = (str(row.get("source_item_id")), str(row.get("model_id")))
        review = reviewed.get(key)
        if review is None:
            machine_candidates.append(row)
            continue
        if str(review.get("human_excluded_from_scoring", "")).lower() == "true":
            dropped += 1
            continue
        human_rows.append(_apply_review(row, review))
    return human_rows, machine_candidates, dropped


def _apply_review(row: dict[str, Any], review: dict[str, str]) -> dict[str, Any]:
    updated = dict(row)
    updated["human_labeled_flag"] = True
    for source, target in (
        ("human_model_id", "model_id"),
        ("human_provider_id", "provider_id"),
        ("human_task_category", "task_category"),
        ("human_aspect_category", "aspect_category"),
        ("human_evidence_type", "evidence_type"),
    ):
        value = (review.get(source) or "").strip()
        if value:
            updated[target] = value
    polarity = (review.get("human_polarity_score") or "").strip()
    if polarity != "":
        updated["polarity_score"] = int(float(polarity))
    firsthand = (review.get("human_firsthand_flag") or "").strip()
    if firsthand != "":
        updated["firsthand_flag"] = firsthand.lower() == "true"
    return updated


def _gate_machine_rows(
    candidates: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    config: dict[str, dict[str, Any]],
    gates: dict[str, float],
    *,
    embedding_cache_dir: str | Path | None,
    calibration_runs: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], int]:
    """Predict fields for unreviewed rows; keep rows whose aspect and sign
    polarity clear their calibrated gates."""
    if not candidates:
        return [], 0
    cache = EmbeddingVectorCache(embedding_cache_dir) if embedding_cache_dir else None
    train_texts = [model_text(row, use_context=False, use_metadata=False) for row in training_rows]
    candidate_texts = [str(row.get("evidence_text") or "") for row in candidates]

    train_features = _build_features_by_field(train_texts, config, cache)
    candidate_features = _build_features_by_field(candidate_texts, config, cache)

    labels_by_field = {field: [label_value(row.get(field)) for row in training_rows] for field in TARGET_FIELDS}
    labels_by_field["polarity_score"] = [
        str(max(-1, min(1, int(float(value))))) for value in labels_by_field["polarity_score"]
    ]
    strategies = {field: str(config[field]["strategy"]) for field in TARGET_FIELDS}
    strategies["polarity_score"] = "flat"  # sign-level head

    predictions: dict[str, list[tuple[str, float]]] = {}
    for field in TARGET_FIELDS:
        calibrator = _fit_calibrator(
            train_features[field],
            labels_by_field[field],
            training_rows,
            strategies[field],
            field,
            runs=calibration_runs,
        )
        model = _field_model(field, strategies[field])
        model.fit(train_features[field], labels_by_field[field])
        field_predictions = []
        for features in candidate_features[field]:
            details = model.predict_details(features)
            confidence = float(details["confidence"])
            if calibrator is not None:
                confidence = float(calibrator.predict([confidence])[0])
            field_predictions.append((str(details["label"]), confidence))
        predictions[field] = field_predictions

    kept: list[tuple[dict[str, Any], dict[str, Any]]] = []
    rejected = 0
    for index, row in enumerate(candidates):
        aspect_label, aspect_conf = predictions["aspect_category"][index]
        polarity_label, polarity_conf = predictions["polarity_score"][index]
        if aspect_conf < gates["aspect_category"] or polarity_conf < gates["polarity_score"]:
            rejected += 1
            continue
        task_label, task_conf = predictions["task_category"][index]
        evidence_label, evidence_conf = predictions["evidence_type"][index]
        firsthand_label, firsthand_conf = predictions["firsthand_flag"][index]
        updated = dict(row)
        updated["aspect_category"] = aspect_label
        updated["polarity_score"] = int(polarity_label)
        task_trusted = task_conf >= gates["task_category"]
        if task_trusted:
            updated["task_category"] = task_label
        if evidence_conf >= gates["evidence_type"]:
            updated["evidence_type"] = evidence_label
        if firsthand_conf >= gates["firsthand_flag"]:
            updated["firsthand_flag"] = firsthand_label == "true"
        kept.append((updated, {"task_trusted": task_trusted}))
    return kept, rejected


def _fit_calibrator(
    features: list[list[float]],
    labels: list[str],
    training_rows: list[dict[str, Any]],
    strategy: str,
    field: str,
    *,
    runs: int,
) -> Any | None:
    from sklearn.isotonic import IsotonicRegression

    splits = grouped_holdout_indexes(training_rows, runs=runs, group_field="thread_id")
    confidences: list[float] = []
    corrects: list[float] = []
    for train_indexes, test_indexes in splits:
        model = _field_model(field, strategy)
        model.fit([features[i] for i in train_indexes], [labels[i] for i in train_indexes])
        for index in test_indexes:
            details = model.predict_details(features[index])
            confidences.append(float(details["confidence"]))
            corrects.append(float(str(details["label"]) == labels[index]))
    if len(set(confidences)) < 2:
        return None
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(confidences, corrects)
    return calibrator


def _to_observation(row: dict[str, Any]) -> Observation:
    weights_data = row.get("weights") or {}
    weights = WeightComponents(
        source_quality_weight=float(weights_data.get("source_quality_weight", 1.0)),
        firsthand_weight=1.25 if _bool(row.get("firsthand_flag")) else 0.75,
        author_credibility_weight=float(weights_data.get("author_credibility_weight", 1.0)),
        corroboration_weight=float(weights_data.get("corroboration_weight", 1.0)),
        recency_weight=float(weights_data.get("recency_weight", 1.0)),
        engagement_weight=float(weights_data.get("engagement_weight", 1.0)),
        duplicate_penalty=float(weights_data.get("duplicate_penalty", 1.0)),
    )
    published_at = row.get("published_at")
    return Observation(
        source_platform=SourcePlatform(str(row.get("source_platform") or "hacker_news")),
        community_id=str(row.get("community_id") or ""),
        thread_id=str(row["thread_id"]) if row.get("thread_id") else None,
        source_item_id=str(row.get("source_item_id") or ""),
        author_id_hash=str(row["author_id_hash"]) if row.get("author_id_hash") else None,
        url=str(row.get("url") or ""),
        published_at=datetime.fromisoformat(published_at) if isinstance(published_at, str) else None,
        model_id=str(row.get("model_id") or ""),
        provider_id=str(row.get("provider_id") or ""),
        model_version_or_alias=str(row.get("model_version_or_alias") or ""),
        task_category=_enum_or_default(TaskCategory, row.get("task_category"), TaskCategory.GENERAL),
        aspect_category=_enum_or_default(AspectCategory, row.get("aspect_category"), AspectCategory.SATISFACTION),
        evidence_type=_enum_or_default(EvidenceType, row.get("evidence_type"), EvidenceType.HEARSAY),
        claim_type=_enum_or_default(ClaimType, row.get("claim_type"), ClaimType.NEUTRAL),
        polarity_score=int(row.get("polarity_score") or 0),
        severity_score=float(row.get("severity_score") or 0.0),
        extractor_confidence=float(row.get("extractor_confidence") or 0.0),
        firsthand_flag=_bool(row.get("firsthand_flag")),
        comparative_flag=_bool(row.get("comparative_flag")),
        regression_flag=_bool(row.get("regression_flag")),
        hallucination_flag=_bool(row.get("hallucination_flag")),
        refusal_flag=_bool(row.get("refusal_flag")),
        value_flag=_bool(row.get("value_flag")),
        weights=weights,
        human_labeled_flag=_bool(row.get("human_labeled_flag")),
        evidence_text=str(row.get("evidence_text") or ""),
        product_id=row.get("product_id") or None,
        inference_profile=row.get("inference_profile") or None,
    )


def _leaderboard_rows(aggregator: ScoreAggregator, pool: list[Observation]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for observation in pool:
        by_cell[(observation.model_id, observation.aspect_category.value)].append(observation)
    provider_by_model = {observation.model_id: observation.provider_id for observation in pool}

    rows = []
    for snapshot in aggregator.aspect_scores(pool):
        group = by_cell[(snapshot.model_id, snapshot.aspect_category)]
        human = sum(1 for item in group if item.human_labeled_flag)
        firsthand = sum(1 for item in group if item.firsthand_flag)
        rows.append(
            {
                "model_id": snapshot.model_id,
                "provider_id": provider_by_model.get(snapshot.model_id, ""),
                "aspect": snapshot.aspect_category,
                "score": round(snapshot.score, 2),
                "ci": [round(snapshot.confidence_low or 0.0, 2), round(snapshot.confidence_high or 0.0, 2)],
                "ess": round(snapshot.effective_n, 1),
                "weighted_n": round(snapshot.weighted_n, 1),
                "n_observations": len(group),
                "n_threads": len({item.thread_id for item in group if item.thread_id}),
                "n_authors": len({item.author_id_hash for item in group if item.author_id_hash}),
                "firsthand_ratio": round(firsthand / len(group), 3) if group else 0.0,
                "human_share": round(human / len(group), 3) if group else 0.0,
                "warnings": snapshot.warnings,
                "publishable": snapshot.publishable,
                "tier": _trust_tier(snapshot.effective_n),
            }
        )
    return rows


def _trust_tier(ess: float) -> str:
    if ess < 30:
        return "insufficient"
    if ess < 150:
        return "provisional"
    return "ranked"


def _per_task_rows(
    aggregator: ScoreAggregator,
    pool: list[Observation],
    task_trusted_flags: list[bool],
) -> dict[str, list[dict[str, Any]]]:
    trusted = [observation for observation, flag in zip(pool, task_trusted_flags) if flag]
    by_task: dict[str, list[Observation]] = defaultdict(list)
    for observation in trusted:
        by_task[observation.task_category.value].append(observation)
    result: dict[str, list[dict[str, Any]]] = {}
    for task, group in sorted(by_task.items()):
        result[task] = [
            {
                "model_id": snapshot.model_id,
                "aspect": snapshot.aspect_category,
                "score": round(snapshot.score, 2),
                "ci": [round(snapshot.confidence_low or 0.0, 2), round(snapshot.confidence_high or 0.0, 2)],
                "ess": round(snapshot.effective_n, 1),
                "tier": _trust_tier(snapshot.effective_n),
            }
            for snapshot in aggregator.aspect_scores(group)
        ]
    return result


def _evidence_samples(pool: list[Observation]) -> dict[str, list[dict[str, Any]]]:
    by_cell: dict[str, list[Observation]] = defaultdict(list)
    for observation in pool:
        by_cell[f"{observation.model_id}|{observation.aspect_category.value}"].append(observation)
    samples: dict[str, list[dict[str, Any]]] = {}
    for key, group in by_cell.items():
        ranked = sorted(group, key=lambda item: item.final_weight, reverse=True)[:EVIDENCE_SAMPLES_PER_CELL]
        samples[key] = [
            {
                "span": item.evidence_text[:EVIDENCE_SPAN_LIMIT],
                "url": item.url,
                "polarity": item.polarity_score,
                "evidence_type": item.evidence_type.value,
                "firsthand": item.firsthand_flag,
                "human_labeled": item.human_labeled_flag,
            }
            for item in ranked
        ]
    return samples


def _methodology_block(
    artifact_paths: list[str | Path],
    gates: dict[str, float],
    training_examples: int,
) -> dict[str, Any]:
    artifacts = {}
    for path in artifact_paths:
        path = Path(path)
        if path.exists():
            artifacts[path.stem] = json.loads(path.read_text(encoding="utf-8")).get("fields", {})
    return {
        "gates": gates,
        "training_examples": training_examples,
        "gated_precision_artifacts": artifacts,
        "pipeline": [
            "Official Hacker News API collection (Algolia + Firebase)",
            "Rule-based span extraction with model/product/profile identity resolution",
            "Human-reviewed training corpus labeled against docs/labeling-guide.md",
            "Multi-encoder classifier with leakage-safe isotonic confidence calibration",
            "Per-field confidence gates: below-gate predictions abstain or go to review",
            "Weighted scoring with thread/author/community caps, Wilson-style intervals, and effective sample size",
        ],
    }


def _load_reviewed_rows(paths: list[str | Path]) -> dict[tuple[str, str], dict[str, str]]:
    reviewed: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (str(row.get("source_item_id") or ""), str(row.get("model_id") or ""))
                reviewed[key] = row
    return reviewed


def _load_observations(paths: list[str | Path]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_jsonl(path):
            key = (
                str(row.get("source_item_id") or ""),
                str(row.get("model_id") or ""),
                str(row.get("evidence_type") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def _enum_or_default(enum_cls: Any, value: Any, default: Any) -> Any:
    try:
        return enum_cls(str(value))
    except ValueError:
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
