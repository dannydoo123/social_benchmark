from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.local_classifier import TARGET_FIELDS

DEFAULT_CONFIG = {"use_context": False, "use_metadata": False, "use_char_features": True}
EXPERIMENT_CONFIGS = {
    "evidence_word": {"use_context": False, "use_metadata": False, "use_char_features": False},
    "evidence_word_char": {"use_context": False, "use_metadata": False, "use_char_features": True},
    "context_word_char": {"use_context": True, "use_metadata": False, "use_char_features": True},
    "context_metadata_word_char": DEFAULT_CONFIG,
}


class SklearnTextClassifier:
    def __init__(
        self,
        models: dict[str, Any] | None = None,
        field_metrics: dict[str, Any] | None = None,
        config: dict[str, bool] | None = None,
    ) -> None:
        self.models = models or {}
        self.field_metrics = field_metrics or {}
        self.config = config or DEFAULT_CONFIG

    def fit(self, examples: list[dict[str, Any]], target_fields: tuple[str, ...] = TARGET_FIELDS) -> None:
        self.models = {field: _fit_field(examples, field, self.config) for field in target_fields}

    def predict(self, text: str) -> dict[str, Any]:
        return self.predict_row({"text": text})

    def predict_row(self, row: dict[str, Any]) -> dict[str, Any]:
        text = _model_text(row, self.config)
        return {field: _predict_field(model, text) for field, model in self.models.items()}

    def save(self, path: str | Path) -> None:
        joblib = _joblib()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": self.config, "field_metrics": self.field_metrics, "models": self.models}, output)

    @classmethod
    def load(cls, path: str | Path) -> "SklearnTextClassifier":
        payload = _joblib().load(path)
        return cls(
            models=payload["models"],
            field_metrics=payload.get("field_metrics") or {},
            config=payload.get("config") or DEFAULT_CONFIG,
        )


def train_sklearn_classifier(training_jsonl: str | Path, output_path: str | Path, *, runs: int = 8) -> int:
    examples = _read_jsonl(training_jsonl)
    classifier = SklearnTextClassifier()
    classifier.fit(examples)
    classifier.field_metrics = evaluate_sklearn_examples(examples, runs=runs).get("fields", {})
    classifier.save(output_path)
    return len(examples)


def evaluate_sklearn_classifier(training_jsonl: str | Path, *, runs: int = 8) -> dict[str, Any]:
    return evaluate_sklearn_examples(_read_jsonl(training_jsonl), runs=runs)


def evaluate_sklearn_examples(
    examples: list[dict[str, Any]],
    *,
    runs: int = 8,
    config: dict[str, bool] | None = None,
) -> dict[str, Any]:
    resolved_config = config or DEFAULT_CONFIG
    metrics: dict[str, Any] = {}
    for field in TARGET_FIELDS:
        aggregate_actual: list[str] = []
        aggregate_predicted: list[str] = []
        run_accuracies: list[float] = []
        for seed in range(runs):
            train, test = _repeatable_holdout(examples, field, seed=seed)
            if not train or not test:
                continue
            model = _fit_field(train, field, resolved_config)
            actual = [_label_value(row.get(field)) for row in test]
            predicted = [_predict_field(model, _model_text(row, resolved_config))["label"] for row in test]
            aggregate_actual.extend(actual)
            aggregate_predicted.extend(predicted)
            run_accuracies.append(_accuracy(actual, predicted))
        metrics[field] = {
            "accuracy": _accuracy(aggregate_actual, aggregate_predicted),
            "macro_f1": _macro_f1(aggregate_actual, aggregate_predicted),
            "runs": len(run_accuracies),
            "run_accuracy_min": min(run_accuracies) if run_accuracies else 0.0,
            "run_accuracy_max": max(run_accuracies) if run_accuracies else 0.0,
            "evaluated": len(aggregate_actual),
            "confusion": _confusion(aggregate_actual, aggregate_predicted),
        }
    return {"config": resolved_config, "examples": len(examples), "fields": metrics, "runs_requested": runs}


def evaluate_sklearn_variants(training_jsonl: str | Path, *, runs: int = 8) -> dict[str, Any]:
    examples = _read_jsonl(training_jsonl)
    return {
        "examples": len(examples),
        "runs_requested": runs,
        "variants": {
            name: evaluate_sklearn_examples(examples, runs=runs, config=config)
            for name, config in EXPERIMENT_CONFIGS.items()
        },
    }


def write_sklearn_variant_evaluation(training_jsonl: str | Path, output_path: str | Path, *, runs: int = 8) -> dict[str, Any]:
    metrics = evaluate_sklearn_variants(training_jsonl, runs=runs)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def write_sklearn_evaluation(training_jsonl: str | Path, output_path: str | Path, *, runs: int = 8) -> dict[str, Any]:
    metrics = evaluate_sklearn_classifier(training_jsonl, runs=runs)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _fit_field(examples: list[dict[str, Any]], field: str, config: dict[str, bool]) -> Any:
    sklearn_pipeline, feature_extraction, linear_model = _sklearn()
    texts = [_model_text(example, config) for example in examples]
    labels = [_label_value(example.get(field)) for example in examples]
    if len(set(labels)) < 2:
        return {"constant": labels[0] if labels else ""}
    transformers = [
        ("word", feature_extraction.TfidfVectorizer(ngram_range=(1, 2), max_features=16000, sublinear_tf=True)),
    ]
    if config["use_char_features"]:
        transformers.append(
            (
                "char",
                feature_extraction.TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=16000,
                    sublinear_tf=True,
                ),
            )
        )
    features = sklearn_pipeline.FeatureUnion(transformers)
    model = sklearn_pipeline.Pipeline(
        [
            ("features", features),
            ("classifier", linear_model.LogisticRegression(class_weight="balanced", max_iter=1200)),
        ]
    )
    model.fit(texts, labels)
    return model


def _predict_field(model: Any, text: str) -> dict[str, Any]:
    if isinstance(model, dict) and "constant" in model:
        return {"label": model["constant"], "confidence": 1.0}
    label = str(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    return {"label": label, "confidence": float(max(probabilities))}


def _model_text(row: dict[str, Any], config: dict[str, bool]) -> str:
    evidence = str(row.get("text") or row.get("evidence_text") or "").strip()
    context = str(row.get("context_text") or row.get("raw_full_text") or "").strip()
    metadata = " ".join(
        f"{field}={str(row.get(field) or '').strip().lower()}"
        for field in ("provider_id", "model_id", "product_id", "inference_profile", "source_platform")
        if row.get(field)
    )
    parts = [evidence, evidence]
    if config["use_context"]:
        parts.append(context)
    if config["use_metadata"]:
        parts.append(metadata)
    return "\n".join(part for part in parts if part)


def _repeatable_holdout(examples: list[dict[str, Any]], field: str, *, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[_label_value(example.get(field))].append(example)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for rows in grouped.values():
        shuffled = list(rows)
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


def _label_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _joblib():
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to use the sklearn classifier backend.") from exc
    return joblib


def _sklearn():
    try:
        from sklearn import feature_extraction, linear_model, pipeline
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to use the sklearn classifier backend.") from exc
    return pipeline, feature_extraction.text, linear_model
