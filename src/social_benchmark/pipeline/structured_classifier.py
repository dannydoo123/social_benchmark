from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.classifier_experiments import (
    EmbeddingVectorCache,
    _encode_with_cache,
    grouped_holdout_indexes,
)
from social_benchmark.pipeline.constraint_resolver import CONSTRAINT_SETS, resolve_prediction_labels
from social_benchmark.pipeline.embeddings import DEFAULT_EMBEDDING_MODEL, HuggingFaceTextEmbedder
from social_benchmark.pipeline.local_classifier import TARGET_FIELDS
from social_benchmark.pipeline.text_features import label_value, model_text

TASK_GROUPS = {
    "general": "general",
    "coding": "technical",
    "api_developer_workflow": "technical",
    "agents": "technical",
    "data_analysis": "knowledge",
    "research": "knowledge",
    "long_context": "knowledge",
    "writing": "creative",
    "multimodal": "creative",
    "roleplay": "creative",
}
ASPECT_GROUPS = {
    "satisfaction": "experience",
    "value": "experience",
    "developer_ergonomics": "experience",
    "task_fit": "capability",
    "trust_reliability": "reliability",
    "regression_stability": "reliability",
    "hallucination_safety": "safety",
    "refusal_acceptance": "safety",
}
EVIDENCE_GROUPS = {
    "firsthand_usage": "usage",
    "comparative_evaluation": "evaluation",
    "benchmark_anecdote": "evaluation",
    "bug_regression_report": "failure",
    "integration_failure": "failure",
    "hearsay": "external",
    "release_update_reaction": "external",
    "pricing_value_comment": "commercial",
}
HIERARCHIES = {
    "task_category": TASK_GROUPS,
    "aspect_category": ASPECT_GROUPS,
    "evidence_type": EVIDENCE_GROUPS,
}
POLARITY_VALUES = (-2, -1, 0, 1, 2)
DEFAULT_FIELD_STRATEGIES = {
    "task_category": "flat",
    "aspect_category": "flat",
    "evidence_type": "flat",
    "polarity_score": "ordinal",
    "firsthand_flag": "flat",
}


class HierarchicalClassifier:
    def __init__(self, label_groups: dict[str, str]) -> None:
        self.label_groups = label_groups
        self.group_model: Any = None
        self.label_models: dict[str, Any] = {}
        self.group_default = ""
        self.label_defaults: dict[str, str] = {}

    def fit(self, vectors: list[list[float]], labels: list[str]) -> None:
        groups = [self.label_groups.get(label, label) for label in labels]
        self.group_default = _most_common(groups)
        self.group_model = _fit_logistic(vectors, groups)
        grouped_vectors: dict[str, list[list[float]]] = defaultdict(list)
        grouped_labels: dict[str, list[str]] = defaultdict(list)
        for vector, label, group in zip(vectors, labels, groups):
            grouped_vectors[group].append(vector)
            grouped_labels[group].append(label)
        for group, group_labels in grouped_labels.items():
            self.label_defaults[group] = _most_common(group_labels)
            self.label_models[group] = _fit_logistic(grouped_vectors[group], group_labels)

    def predict(self, vector: list[float]) -> str:
        group = _predict(self.group_model, vector, self.group_default)
        return _predict(self.label_models.get(group), vector, self.label_defaults.get(group, group))


class OrdinalPolarityClassifier:
    def __init__(self) -> None:
        self.threshold_models: list[Any] = []
        self.default = "0"

    def fit(self, vectors: list[list[float]], labels: list[str]) -> None:
        numeric = [int(label) for label in labels]
        self.default = str(_most_common(numeric))
        self.threshold_models = [
            _fit_logistic(vectors, [str(int(value > threshold)) for value in numeric])
            for threshold in POLARITY_VALUES[:-1]
        ]

    def predict(self, vector: list[float]) -> str:
        return self.predict_details(vector)["label"]

    def predict_details(self, vector: list[float]) -> dict[str, Any]:
        value = POLARITY_VALUES[0]
        probabilities = []
        for threshold, model in zip(POLARITY_VALUES[:-1], self.threshold_models):
            probability = _positive_probability(model, vector)
            probabilities.append(probability)
            if probability >= 0.5:
                value = threshold + 1
        confidence = sum(abs(probability - 0.5) * 2 for probability in probabilities) / len(probabilities)
        return {"label": str(value), "confidence": confidence}


def run_structured_embedding_bakeoff(
    training_jsonl: str | Path,
    output_path: str | Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    runs: int = 8,
    group_field: str = "thread_id",
    embedding_cache_dir: str | Path | None = None,
    embedder: Any | None = None,
    field_strategies: dict[str, str] | None = None,
) -> dict[str, Any]:
    examples = _read_jsonl(training_jsonl)
    texts = [model_text(row, use_context=False, use_metadata=False) for row in examples]
    cache = EmbeddingVectorCache(embedding_cache_dir) if embedding_cache_dir else None
    vectors = _encode_with_cache(
        texts,
        embedding_model=embedding_model,
        embedder=embedder or HuggingFaceTextEmbedder(model_name=embedding_model),
        cache=cache,
    )
    splits = grouped_holdout_indexes(examples, runs=runs, group_field=group_field)
    strategies = field_strategies or DEFAULT_FIELD_STRATEGIES
    fields = {
        field: _evaluate_field(examples, vectors, splits, field, strategies.get(field, "flat"))
        for field in TARGET_FIELDS
    }
    result = {
        "backend": "structured_embedding",
        "embedding_model": embedding_model,
        "examples": len(examples),
        "evaluation": {"group_field": group_field, "runs": runs},
        "field_strategies": strategies,
        "fields": fields,
        "mean_macro_f1": sum(values["macro_f1"] for values in fields.values()) / len(fields),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def run_constraint_bakeoff(
    training_jsonl: str | Path,
    output_path: str | Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    runs: int = 8,
    group_field: str = "thread_id",
    embedding_cache_dir: str | Path | None = None,
    embedder: Any | None = None,
    field_strategies: dict[str, str] | None = None,
) -> dict[str, Any]:
    examples = _read_jsonl(training_jsonl)
    texts = [model_text(row, use_context=False, use_metadata=False) for row in examples]
    cache = EmbeddingVectorCache(embedding_cache_dir) if embedding_cache_dir else None
    vectors = _encode_with_cache(
        texts,
        embedding_model=embedding_model,
        embedder=embedder or HuggingFaceTextEmbedder(model_name=embedding_model),
        cache=cache,
    )
    splits = grouped_holdout_indexes(examples, runs=runs, group_field=group_field)
    strategies = field_strategies or DEFAULT_FIELD_STRATEGIES
    actual = {field: [] for field in TARGET_FIELDS}
    predicted_by_set = {
        name: {field: [] for field in TARGET_FIELDS}
        for name in CONSTRAINT_SETS
    }
    for train_indexes, test_indexes in splits:
        models = {}
        for field in TARGET_FIELDS:
            models[field] = _field_model(field, strategies.get(field, "flat"))
            models[field].fit(
                [vectors[index] for index in train_indexes],
                [label_value(examples[index].get(field)) for index in train_indexes],
            )
        for index in test_indexes:
            labels = {field: models[field].predict(vectors[index]) for field in TARGET_FIELDS}
            for field in TARGET_FIELDS:
                actual[field].append(label_value(examples[index].get(field)))
            for name, rules in CONSTRAINT_SETS.items():
                resolved = resolve_prediction_labels(labels, enabled_rules=rules)
                for field in TARGET_FIELDS:
                    predicted_by_set[name][field].append(resolved[field])
    variants = {}
    for name, predictions in predicted_by_set.items():
        fields = {
            field: {
                "accuracy": _accuracy(actual[field], predictions[field]),
                "macro_f1": _macro_f1(actual[field], predictions[field]),
                "evaluated": len(actual[field]),
                "confusion": dict(
                    sorted(Counter(f"{left} -> {right}" for left, right in zip(actual[field], predictions[field])).items())
                ),
            }
            for field in TARGET_FIELDS
        }
        variants[name] = {
            "fields": fields,
            "mean_macro_f1": sum(values["macro_f1"] for values in fields.values()) / len(fields),
        }
    result = {
        "backend": "structured_embedding_constraints",
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
        labels = [label_value(examples[index].get(field)) for index in train_indexes]
        model = _field_model(field, strategy)
        model.fit([vectors[index] for index in train_indexes], labels)
        actual.extend(label_value(examples[index].get(field)) for index in test_indexes)
        predicted.extend(model.predict(vectors[index]) for index in test_indexes)
    return {
        "accuracy": _accuracy(actual, predicted),
        "macro_f1": _macro_f1(actual, predicted),
        "evaluated": len(actual),
        "confusion": dict(sorted(Counter(f"{left} -> {right}" for left, right in zip(actual, predicted)).items())),
    }


def _field_model(field: str, strategy: str) -> Any:
    if strategy == "ordinal" and field == "polarity_score":
        return OrdinalPolarityClassifier()
    if strategy == "hierarchical" and field in HIERARCHIES:
        return HierarchicalClassifier(HIERARCHIES[field])
    if strategy != "flat":
        raise ValueError(f"Unsupported strategy {strategy!r} for {field}.")
    return FlatClassifier()


class FlatClassifier:
    def __init__(self) -> None:
        self.model: Any = None
        self.default = ""

    def fit(self, vectors: list[list[float]], labels: list[str]) -> None:
        self.default = _most_common(labels)
        self.model = _fit_logistic(vectors, labels)

    def predict(self, vector: list[float]) -> str:
        return self.predict_details(vector)["label"]

    def predict_details(self, vector: list[float]) -> dict[str, Any]:
        if self.model is None:
            return {"label": str(self.default), "confidence": 1.0}
        probabilities = self.model.predict_proba([vector])[0]
        return {
            "label": str(self.model.classes_[int(probabilities.argmax())]),
            "confidence": float(max(probabilities)),
        }


def _fit_logistic(vectors: list[list[float]], labels: list[str]) -> Any:
    if not labels or len(set(labels)) < 2:
        return None
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(class_weight="balanced", max_iter=1200)
    model.fit(vectors, labels)
    return model


def _predict(model: Any, vector: list[float], default: str) -> str:
    return str(model.predict([vector])[0]) if model is not None else str(default)


def _positive_probability(model: Any, vector: list[float]) -> float:
    if model is None:
        return 0.0
    classes = [str(value) for value in model.classes_]
    probabilities = model.predict_proba([vector])[0]
    return float(probabilities[classes.index("1")]) if "1" in classes else 0.0


def _most_common(values: list[Any]) -> Any:
    return Counter(values).most_common(1)[0][0] if values else ""


def _accuracy(actual: list[str], predicted: list[str]) -> float:
    return sum(left == right for left, right in zip(actual, predicted)) / len(actual) if actual else 0.0


def _macro_f1(actual: list[str], predicted: list[str]) -> float:
    labels = sorted(set(actual) | set(predicted))
    scores = []
    for label in labels:
        true_positive = sum(left == label and right == label for left, right in zip(actual, predicted))
        false_positive = sum(left != label and right == label for left, right in zip(actual, predicted))
        false_negative = sum(left == label and right != label for left, right in zip(actual, predicted))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
