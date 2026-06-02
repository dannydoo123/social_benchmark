from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.embeddings import DEFAULT_EMBEDDING_MODEL, HuggingFaceTextEmbedder
from social_benchmark.pipeline.hf_classifier import HFEmbeddingClassifier
from social_benchmark.pipeline.local_classifier import LocalNaiveBayesClassifier, TARGET_FIELDS
from social_benchmark.pipeline.sklearn_classifier import SklearnTextClassifier
from social_benchmark.pipeline.text_features import label_value

ABSTAIN_LABEL = "__abstain__"
PRECISION_FIRST_FIELD_CONFIDENCE = {
    "task_category": 0.62,
    "aspect_category": 0.68,
    "evidence_type": 0.68,
    "polarity_score": 0.78,
    "firsthand_flag": 0.78,
}


class HighPrecisionClassifier:
    def __init__(
        self,
        backends: dict[str, Any] | None = None,
        field_metrics: dict[str, Any] | None = None,
        *,
        min_confidence: float = 0.62,
        min_agreement: int = 2,
        include_hf: bool = True,
        hf_model_name: str = DEFAULT_EMBEDDING_MODEL,
        field_min_confidence: dict[str, float] | None = None,
    ) -> None:
        self.backends = backends or {}
        self.field_metrics = field_metrics or {}
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement
        self.include_hf = include_hf
        self.hf_model_name = hf_model_name
        self.field_min_confidence = field_min_confidence or {}

    def fit(self, examples: list[dict[str, Any]], target_fields: tuple[str, ...] = TARGET_FIELDS) -> None:
        nb = LocalNaiveBayesClassifier()
        nb.fit(examples, target_fields=target_fields)
        sklearn = SklearnTextClassifier()
        sklearn.fit(examples, target_fields=target_fields)
        self.backends = {"naive_bayes": nb, "sklearn_tfidf_logistic": sklearn}
        if self.include_hf:
            hf = HFEmbeddingClassifier(model_name=self.hf_model_name)
            hf.fit(examples, target_fields=target_fields)
            self.backends["hf_embedding_logistic"] = hf

    def predict(self, text: str) -> dict[str, Any]:
        return self.predict_row({"text": text})

    def predict_row(self, row: dict[str, Any]) -> dict[str, Any]:
        backend_predictions = {name: backend.predict_row(row) for name, backend in self.backends.items()}
        return {
            field: _combine_field_predictions(
                [prediction[field] for prediction in backend_predictions.values() if field in prediction],
                min_confidence=self.field_min_confidence.get(field, self.min_confidence),
                min_agreement=self.min_agreement,
            )
            for field in TARGET_FIELDS
        }

    def save(self, path: str | Path) -> None:
        joblib = _joblib()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        hf_backend = self.backends.get("hf_embedding_logistic")
        loaded_embedder = None
        if hf_backend is not None:
            loaded_embedder = hf_backend.embedder
            hf_backend.embedder = HuggingFaceTextEmbedder(model_name=hf_backend.model_name)
        try:
            joblib.dump(
                {
                    "backend": "high_precision_ensemble",
                    "backends": self.backends,
                    "field_metrics": self.field_metrics,
                    "min_confidence": self.min_confidence,
                    "min_agreement": self.min_agreement,
                    "include_hf": self.include_hf,
                    "hf_model_name": self.hf_model_name,
                    "field_min_confidence": self.field_min_confidence,
                },
                output,
            )
        finally:
            if hf_backend is not None:
                hf_backend.embedder = loaded_embedder

    @classmethod
    def load(cls, path: str | Path) -> "HighPrecisionClassifier":
        payload = _joblib().load(path)
        return cls(
            backends=payload["backends"],
            field_metrics=payload.get("field_metrics") or {},
            min_confidence=float(payload.get("min_confidence", 0.62)),
            min_agreement=int(payload.get("min_agreement", 2)),
            include_hf=bool(payload.get("include_hf", True)),
            hf_model_name=payload.get("hf_model_name") or DEFAULT_EMBEDDING_MODEL,
            field_min_confidence=payload.get("field_min_confidence") or {},
        )


def train_high_precision_classifier(
    training_jsonl: str | Path,
    output_path: str | Path,
    *,
    runs: int = 8,
    min_confidence: float = 0.62,
    min_agreement: int = 2,
    include_hf: bool = True,
    hf_model_name: str = DEFAULT_EMBEDDING_MODEL,
    field_min_confidence: dict[str, float] | None = None,
) -> int:
    examples = _read_jsonl(training_jsonl)
    classifier = HighPrecisionClassifier(
        min_confidence=min_confidence,
        min_agreement=min_agreement,
        include_hf=include_hf,
        hf_model_name=hf_model_name,
        field_min_confidence=field_min_confidence,
    )
    classifier.fit(examples)
    classifier.field_metrics = evaluate_high_precision_examples(
        examples,
        runs=runs,
        min_confidence=min_confidence,
        min_agreement=min_agreement,
        include_hf=include_hf,
        hf_model_name=hf_model_name,
        field_min_confidence=field_min_confidence,
    ).get("fields", {})
    classifier.save(output_path)
    return len(examples)


def evaluate_high_precision_classifier(
    training_jsonl: str | Path,
    *,
    runs: int = 8,
    min_confidence: float = 0.62,
    min_agreement: int = 2,
    include_hf: bool = True,
    hf_model_name: str = DEFAULT_EMBEDDING_MODEL,
    field_min_confidence: dict[str, float] | None = None,
) -> dict[str, Any]:
    return evaluate_high_precision_examples(
        _read_jsonl(training_jsonl),
        runs=runs,
        min_confidence=min_confidence,
        min_agreement=min_agreement,
        include_hf=include_hf,
        hf_model_name=hf_model_name,
        field_min_confidence=field_min_confidence,
    )


def write_high_precision_evaluation(
    training_jsonl: str | Path,
    output_path: str | Path,
    *,
    runs: int = 8,
    min_confidence: float = 0.62,
    min_agreement: int = 2,
    include_hf: bool = True,
    hf_model_name: str = DEFAULT_EMBEDDING_MODEL,
    field_min_confidence: dict[str, float] | None = None,
) -> dict[str, Any]:
    metrics = evaluate_high_precision_classifier(
        training_jsonl,
        runs=runs,
        min_confidence=min_confidence,
        min_agreement=min_agreement,
        include_hf=include_hf,
        hf_model_name=hf_model_name,
        field_min_confidence=field_min_confidence,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def evaluate_high_precision_examples(
    examples: list[dict[str, Any]],
    *,
    runs: int = 8,
    min_confidence: float = 0.62,
    min_agreement: int = 2,
    include_hf: bool = True,
    hf_model_name: str = DEFAULT_EMBEDDING_MODEL,
    field_min_confidence: dict[str, float] | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field in TARGET_FIELDS:
        aggregate_actual: list[str] = []
        aggregate_predicted: list[str] = []
        abstained = 0
        total = 0
        run_precisions: list[float] = []
        for seed in range(runs):
            train, test = _repeatable_holdout(examples, field, seed=seed)
            if not train or not test:
                continue
            classifier = HighPrecisionClassifier(
                min_confidence=min_confidence,
                min_agreement=min_agreement,
                include_hf=include_hf,
                hf_model_name=hf_model_name,
                field_min_confidence=field_min_confidence,
            )
            classifier.fit(train, target_fields=(field,))
            actual = [label_value(row.get(field)) for row in test]
            predicted = [classifier.predict_row(row)[field]["label"] for row in test]
            total += len(predicted)
            abstained += sum(label == ABSTAIN_LABEL for label in predicted)
            answered_pairs = [(left, right) for left, right in zip(actual, predicted) if right != ABSTAIN_LABEL]
            aggregate_actual.extend(left for left, _ in answered_pairs)
            aggregate_predicted.extend(right for _, right in answered_pairs)
            run_precisions.append(_accuracy([left for left, _ in answered_pairs], [right for _, right in answered_pairs]))
        fields[field] = {
            "precision": _accuracy(aggregate_actual, aggregate_predicted),
            "coverage": (total - abstained) / total if total else 0.0,
            "answered": total - abstained,
            "abstained": abstained,
            "total": total,
            "runs": len(run_precisions),
            "run_precision_min": min(run_precisions) if run_precisions else 0.0,
            "run_precision_max": max(run_precisions) if run_precisions else 0.0,
            "confusion": _confusion(aggregate_actual, aggregate_predicted),
        }
    return {
        "backend": "high_precision_ensemble",
        "examples": len(examples),
        "fields": fields,
        "hf_model_name": hf_model_name if include_hf else None,
        "include_hf": include_hf,
        "min_agreement": min_agreement,
        "min_confidence": min_confidence,
        "field_min_confidence": field_min_confidence or {},
        "runs_requested": runs,
    }


def _combine_field_predictions(
    predictions: list[dict[str, Any]],
    *,
    min_confidence: float,
    min_agreement: int,
) -> dict[str, Any]:
    if not predictions:
        return {"label": ABSTAIN_LABEL, "confidence": 0.0, "agreement": 0}
    grouped: dict[str, list[float]] = defaultdict(list)
    for prediction in predictions:
        grouped[str(prediction.get("label", ""))].append(float(prediction.get("confidence", 0.0)))
    label, confidences = max(grouped.items(), key=lambda item: (len(item[1]), sum(item[1]) / len(item[1])))
    confidence = sum(confidences) / len(confidences)
    agreement = len(confidences)
    if agreement < min_agreement or confidence < min_confidence:
        return {
            "label": ABSTAIN_LABEL,
            "confidence": confidence,
            "agreement": agreement,
            "candidate_label": label,
        }
    return {"label": label, "confidence": confidence, "agreement": agreement}


def _repeatable_holdout(examples: list[dict[str, Any]], field: str, *, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[label_value(example.get(field))].append(example)
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


def _confusion(actual: list[str], predicted: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(f"{left} -> {right}" for left, right in zip(actual, predicted)).items()))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _joblib():
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to use the high-precision classifier backend.") from exc
    return joblib
