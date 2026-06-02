from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.embeddings import DEFAULT_EMBEDDING_MODEL, HuggingFaceTextEmbedder, TextEmbedder
from social_benchmark.pipeline.local_classifier import TARGET_FIELDS
from social_benchmark.pipeline.text_features import label_value, model_text


class HFEmbeddingClassifier:
    def __init__(
        self,
        models: dict[str, Any] | None = None,
        field_metrics: dict[str, Any] | None = None,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedder: TextEmbedder | None = None,
    ) -> None:
        self.models = models or {}
        self.field_metrics = field_metrics or {}
        self.model_name = model_name
        self.embedder = embedder or HuggingFaceTextEmbedder(model_name=model_name)

    def fit(self, examples: list[dict[str, Any]], target_fields: tuple[str, ...] = TARGET_FIELDS) -> None:
        texts = [model_text(example) for example in examples]
        vectors = self.embedder.encode(texts)
        self.models = {field: _fit_field(vectors, examples, field) for field in target_fields}

    def predict(self, text: str) -> dict[str, Any]:
        return self.predict_row({"text": text})

    def predict_row(self, row: dict[str, Any]) -> dict[str, Any]:
        vectors = self.embedder.encode([model_text(row)])
        vector = vectors[0] if vectors else []
        return {field: _predict_field(model, vector) for field, model in self.models.items()}

    def save(self, path: str | Path) -> None:
        joblib = _joblib()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"backend": "hf_embedding", "model_name": self.model_name, "field_metrics": self.field_metrics, "models": self.models},
            output,
        )

    @classmethod
    def load(cls, path: str | Path) -> "HFEmbeddingClassifier":
        payload = _joblib().load(path)
        return cls(
            models=payload["models"],
            field_metrics=payload.get("field_metrics") or {},
            model_name=payload.get("model_name") or DEFAULT_EMBEDDING_MODEL,
        )


def train_hf_classifier(
    training_jsonl: str | Path,
    output_path: str | Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    runs: int = 8,
) -> int:
    examples = _read_jsonl(training_jsonl)
    classifier = HFEmbeddingClassifier(model_name=model_name)
    classifier.fit(examples)
    classifier.field_metrics = evaluate_hf_examples(examples, model_name=model_name, runs=runs).get("fields", {})
    classifier.save(output_path)
    return len(examples)


def evaluate_hf_classifier(
    training_jsonl: str | Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    runs: int = 8,
) -> dict[str, Any]:
    return evaluate_hf_examples(_read_jsonl(training_jsonl), model_name=model_name, runs=runs)


def write_hf_evaluation(
    training_jsonl: str | Path,
    output_path: str | Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    runs: int = 8,
) -> dict[str, Any]:
    metrics = evaluate_hf_classifier(training_jsonl, model_name=model_name, runs=runs)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def evaluate_hf_examples(
    examples: list[dict[str, Any]],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    runs: int = 8,
    embedder: TextEmbedder | None = None,
) -> dict[str, Any]:
    resolved_embedder = embedder or HuggingFaceTextEmbedder(model_name=model_name)
    texts = [model_text(example) for example in examples]
    vectors = resolved_embedder.encode(texts)
    fields: dict[str, Any] = {}
    for field in TARGET_FIELDS:
        aggregate_actual: list[str] = []
        aggregate_predicted: list[str] = []
        run_accuracies: list[float] = []
        for seed in range(runs):
            train_indexes, test_indexes = _repeatable_holdout_indexes(examples, field, seed=seed)
            if not train_indexes or not test_indexes:
                continue
            model = _fit_field([vectors[index] for index in train_indexes], [examples[index] for index in train_indexes], field)
            actual = [label_value(examples[index].get(field)) for index in test_indexes]
            predicted = [_predict_field(model, vectors[index])["label"] for index in test_indexes]
            aggregate_actual.extend(actual)
            aggregate_predicted.extend(predicted)
            run_accuracies.append(_accuracy(actual, predicted))
        fields[field] = {
            "accuracy": _accuracy(aggregate_actual, aggregate_predicted),
            "macro_f1": _macro_f1(aggregate_actual, aggregate_predicted),
            "runs": len(run_accuracies),
            "run_accuracy_min": min(run_accuracies) if run_accuracies else 0.0,
            "run_accuracy_max": max(run_accuracies) if run_accuracies else 0.0,
            "evaluated": len(aggregate_actual),
            "confusion": _confusion(aggregate_actual, aggregate_predicted),
        }
    return {"backend": "hf_embedding", "embedding_model": model_name, "examples": len(examples), "fields": fields, "runs_requested": runs}


def _fit_field(vectors: list[list[float]], examples: list[dict[str, Any]], field: str) -> Any:
    labels = [label_value(example.get(field)) for example in examples]
    if not labels:
        return {"constant": ""}
    if len(set(labels)) < 2:
        return {"constant": labels[0]}
    _, linear_model = _sklearn()
    model = linear_model.LogisticRegression(class_weight="balanced", max_iter=1200)
    model.fit(vectors, labels)
    return model


def _predict_field(model: Any, vector: list[float]) -> dict[str, Any]:
    if isinstance(model, dict) and "constant" in model:
        return {"label": model["constant"], "confidence": 1.0}
    label = str(model.predict([vector])[0])
    probabilities = model.predict_proba([vector])[0]
    return {"label": label, "confidence": float(max(probabilities))}


def _repeatable_holdout_indexes(examples: list[dict[str, Any]], field: str, *, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[label_value(example.get(field))].append(index)
    train: list[int] = []
    test: list[int] = []
    for indexes in grouped.values():
        shuffled = list(indexes)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            train.extend(shuffled)
            continue
        test_count = max(1, round(len(shuffled) * 0.25))
        test_count = min(test_count, len(shuffled) - 1)
        test.extend(shuffled[:test_count])
        train.extend(shuffled[test_count:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


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


def _confusion(actual: list[str], predicted: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(f"{left} -> {right}" for left, right in zip(actual, predicted)).items()))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _joblib():
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to use the Hugging Face embedding classifier backend.") from exc
    return joblib


def _sklearn():
    try:
        from sklearn import linear_model, metrics
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to use the Hugging Face embedding classifier backend.") from exc
    return metrics, linear_model
