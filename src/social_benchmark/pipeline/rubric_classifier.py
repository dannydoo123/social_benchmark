from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.classifier_experiments import (
    EmbeddingVectorCache,
    _encode_with_cache,
    grouped_holdout_indexes,
)
from social_benchmark.pipeline.embeddings import DEFAULT_EMBEDDING_MODEL, HuggingFaceTextEmbedder
from social_benchmark.pipeline.local_classifier import TARGET_FIELDS
from social_benchmark.pipeline.structured_classifier import DEFAULT_FIELD_STRATEGIES, _field_model, _macro_f1
from social_benchmark.pipeline.text_features import label_value, model_text

LABEL_RUBRICS = {
    "task_category": {
        "general": "General model quality or use without a specific task.",
        "coding": "Writing, editing, debugging, reviewing, or understanding source code.",
        "api_developer_workflow": "Using an API, SDK, rate limits, latency, deployment, or developer integration.",
        "agents": "Autonomous or multi-step agents using tools and completing actions.",
        "writing": "Writing, rewriting, editing, summarizing, or creative prose.",
        "research": "Finding, synthesizing, verifying, or citing information.",
        "data_analysis": "Analyzing tables, statistics, datasets, SQL results, or quantitative data.",
        "long_context": "Working with very long documents, conversations, repositories, or context windows.",
        "multimodal": "Understanding or generating images, audio, video, or mixed media.",
        "roleplay": "Roleplay, character interaction, or fictional persona behavior.",
    },
    "aspect_category": {
        "satisfaction": "Overall satisfaction, preference, enjoyment, or disappointment.",
        "trust_reliability": "Reliability, predictability, trust, safety, or dependable behavior.",
        "task_fit": "How capable or suitable the model is for a specific task.",
        "regression_stability": "The model became worse, changed behavior, or regressed over time.",
        "hallucination_safety": "Fabrication, hallucination, factual errors, or unsafe invented claims.",
        "refusal_acceptance": "Whether refusals are excessive, appropriate, or acceptable.",
        "value": "Price, cost, subscription value, usage limits, or value for money.",
        "developer_ergonomics": "API, SDK, tooling, latency, rate limits, deployment, or integration experience.",
    },
    "evidence_type": {
        "firsthand_usage": "The author directly used, tested, deployed, or paid for the model.",
        "comparative_evaluation": "The author compares multiple models or products based on evaluation.",
        "bug_regression_report": "A model became worse, broke a workflow, or negatively changed over time.",
        "integration_failure": "An API, SDK, latency, rate limit, deployment, or tooling integration failed.",
        "benchmark_anecdote": "A benchmark, score, test suite, or benchmark-like result is discussed.",
        "hearsay": "Secondhand opinion, speculation, or commentary without direct use.",
        "release_update_reaction": "Reaction to a provider announcement, release, update, or release notes.",
        "pricing_value_comment": "Comment about price, subscription cost, limits, or value for money.",
    },
    "polarity_score": {
        "-2": "Strong negative complaint: severe failure, unusable behavior, or major regression.",
        "-1": "Mild negative complaint or negative task fit.",
        "0": "Neutral, descriptive, ambiguous, or without a clear quality judgment.",
        "1": "Mild positive praise or positive task fit.",
        "2": "Strong positive praise or clear model-quality endorsement.",
    },
    "firsthand_flag": {
        "false": "The author did not directly use or evaluate the model; this is secondhand.",
        "true": "The author directly used, tested, paid for, deployed, or evaluated the model.",
    },
}


def run_rubric_embedding_bakeoff(
    training_jsonl: str | Path,
    output_path: str | Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    runs: int = 8,
    group_field: str = "thread_id",
    embedding_cache_dir: str | Path | None = None,
    rubric_scales: tuple[float, ...] = (1.0, 3.0, 10.0),
    field_rubric_scales: dict[str, float] | None = None,
    embedder: Any | None = None,
    field_strategies: dict[str, str] | None = None,
) -> dict[str, Any]:
    examples = _read_jsonl(training_jsonl)
    resolved_embedder = embedder or HuggingFaceTextEmbedder(model_name=embedding_model)
    cache = EmbeddingVectorCache(embedding_cache_dir) if embedding_cache_dir else None
    texts = [model_text(row, use_context=False, use_metadata=False) for row in examples]
    vectors = _encode_with_cache(texts, embedding_model=embedding_model, embedder=resolved_embedder, cache=cache)
    rubric_vectors = {
        field: resolved_embedder.encode(list(definitions.values()))
        for field, definitions in LABEL_RUBRICS.items()
    }
    splits = grouped_holdout_indexes(examples, runs=runs, group_field=group_field)
    strategies = field_strategies or DEFAULT_FIELD_STRATEGIES
    scale_variants: dict[str, dict[str, float]] = {
        str(scale): {field: scale for field in TARGET_FIELDS}
        for scale in rubric_scales
    }
    if field_rubric_scales is not None:
        scale_variants["field_selected"] = {
            field: float(field_rubric_scales.get(field, 0.0))
            for field in TARGET_FIELDS
        }
    variants = {}
    for name, scales in scale_variants.items():
        fields = {}
        for field in TARGET_FIELDS:
            field_vectors = [
                _rubric_features(vector, rubric_vectors[field], scale=scales[field])
                for vector in vectors
            ]
            fields[field] = _evaluate_field(examples, field_vectors, splits, field, strategies.get(field, "flat"))
        variants[name] = {
            "field_rubric_scales": scales,
            "fields": fields,
            "mean_macro_f1": sum(values["macro_f1"] for values in fields.values()) / len(fields),
        }
    result = {
        "backend": "rubric_conditioned_embedding",
        "embedding_model": embedding_model,
        "examples": len(examples),
        "evaluation": {"group_field": group_field, "runs": runs},
        "field_strategies": strategies,
        "variants": variants,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _rubric_features(vector: list[float], rubric_vectors: list[list[float]], *, scale: float) -> list[float]:
    return [float(value) for value in vector] + [scale * _cosine(vector, rubric) for rubric in rubric_vectors]


def _evaluate_field(
    examples: list[dict[str, Any]],
    vectors: list[list[float]],
    splits: list[tuple[list[int], list[int]]],
    field: str,
    strategy: str,
) -> dict[str, Any]:
    actual: list[str] = []
    predicted: list[str] = []
    for train_indexes, test_indexes in splits:
        model = _field_model(field, strategy)
        model.fit(
            [vectors[index] for index in train_indexes],
            [label_value(examples[index].get(field)) for index in train_indexes],
        )
        actual.extend(label_value(examples[index].get(field)) for index in test_indexes)
        predicted.extend(model.predict(vectors[index]) for index in test_indexes)
    return {
        "accuracy": sum(left == right for left, right in zip(actual, predicted)) / len(actual) if actual else 0.0,
        "macro_f1": _macro_f1(actual, predicted),
        "evaluated": len(actual),
    }


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
