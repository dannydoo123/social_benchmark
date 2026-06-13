from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.classifier_experiments import EmbeddingVectorCache, _encode_with_cache, grouped_holdout_indexes
from social_benchmark.pipeline.embeddings import HuggingFaceTextEmbedder
from social_benchmark.pipeline.local_classifier import TARGET_FIELDS
from social_benchmark.pipeline.rubric_classifier import LABEL_RUBRICS, _rubric_features
from social_benchmark.pipeline.structured_classifier import _field_model, _macro_f1
from social_benchmark.pipeline.text_features import label_value, model_text

DEFAULT_FIELD_CONFIG = {
    "task_category": {
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "backend": "auto",
        "rubric_scale": 10.0,
        "strategy": "flat",
    },
    "aspect_category": {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "backend": "auto",
        "rubric_scale": 1.0,
        "strategy": "flat",
    },
    "evidence_type": {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "backend": "auto",
        "rubric_scale": 0.0,
        "strategy": "flat",
    },
    "polarity_score": {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "backend": "auto",
        "rubric_scale": 3.0,
        "strategy": "ordinal",
    },
    "firsthand_flag": {
        "embedding_model": "bert-base-uncased",
        "backend": "transformers",
        "rubric_scale": 0.0,
        "strategy": "flat",
    },
}


class RoutedRubricClassifier:
    def __init__(
        self,
        *,
        field_config: dict[str, dict[str, Any]] | None = None,
        embedders: dict[str, Any] | None = None,
    ) -> None:
        self.field_config = field_config or DEFAULT_FIELD_CONFIG
        self.embedders = embedders or {}
        self.models: dict[str, Any] = {}
        self.rubric_vectors: dict[str, list[list[float]]] = {}

    def fit(self, examples: list[dict[str, Any]]) -> None:
        texts = [model_text(row, use_context=False, use_metadata=False) for row in examples]
        for field in TARGET_FIELDS:
            config = self.field_config[field]
            features = self._field_features(field, texts)
            model = _field_model(field, str(config["strategy"]))
            model.fit(features, [label_value(row.get(field)) for row in examples])
            self.models[field] = model

    def predict_row(self, row: dict[str, Any]) -> dict[str, dict[str, Any]]:
        text = model_text(row, use_context=False, use_metadata=False)
        predictions = {}
        for field in TARGET_FIELDS:
            features = self._field_features(field, [text])[0]
            predictions[field] = self.models[field].predict_details(features)
        return predictions

    def _field_features(self, field: str, texts: list[str]) -> list[list[float]]:
        config = self.field_config[field]
        features = self._encoder_rubric_features(
            field,
            str(config["embedding_model"]),
            str(config["backend"]),
            float(config["rubric_scale"]),
            texts,
        )
        for extra in config.get("extra_embedding_models", []):
            extra_features = self._encoder_rubric_features(
                field,
                str(extra["embedding_model"]),
                str(extra.get("backend", "auto")),
                float(extra.get("rubric_scale", config["rubric_scale"])),
                texts,
            )
            features = [row + extra_row for row, extra_row in zip(features, extra_features)]
        nli_model = config.get("nli_model")
        if nli_model:
            nli_scale = float(config.get("nli_scale", 1.0))
            scorer = self._nli_scorer(field, str(nli_model))
            nli_vectors = scorer.encode(texts)
            features = [
                row + [value * nli_scale for value in nli_row]
                for row, nli_row in zip(features, nli_vectors)
            ]
        return features

    def _encoder_rubric_features(
        self,
        field: str,
        model_name: str,
        backend: str,
        scale: float,
        texts: list[str],
    ) -> list[list[float]]:
        key = _encoder_key({"embedding_model": model_name, "backend": backend})
        embedder = self._encoder(model_name, backend)
        rubric_key = f"{field}|{key}"
        if rubric_key not in self.rubric_vectors:
            if field in self.rubric_vectors and model_name == str(self.field_config[field]["embedding_model"]):
                # backward compatibility with models saved before multi-encoder stacking
                self.rubric_vectors[rubric_key] = self.rubric_vectors[field]
            else:
                self.rubric_vectors[rubric_key] = embedder.encode(list(LABEL_RUBRICS[field].values()))
        rubric_vectors = self.rubric_vectors[rubric_key]
        vectors = embedder.encode(texts)
        return [_rubric_features(vector, rubric_vectors, scale=scale) for vector in vectors]

    def _encoder(self, model_name: str, backend: str) -> Any:
        key = _encoder_key({"embedding_model": model_name, "backend": backend})
        if key not in self.embedders:
            self.embedders[key] = HuggingFaceTextEmbedder(model_name=model_name, backend=backend)
        return self.embedders[key]

    def _nli_scorer(self, field: str, model_name: str) -> Any:
        from social_benchmark.pipeline.nli_features import NliRubricScorer

        key = f"nli|{model_name}|{field}"
        if key not in self.embedders:
            self.embedders[key] = NliRubricScorer(
                model_name=model_name,
                hypotheses=list(LABEL_RUBRICS[field].values()),
            )
        return self.embedders[key]

    def save(self, path: str | Path) -> None:
        import joblib

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "backend": "routed_rubric",
                "field_config": self.field_config,
                "models": self.models,
                "rubric_vectors": self.rubric_vectors,
            },
            output,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RoutedRubricClassifier":
        import joblib

        payload = joblib.load(path)
        classifier = cls(field_config=payload["field_config"])
        classifier.models = payload["models"]
        classifier.rubric_vectors = payload["rubric_vectors"]
        return classifier

def train_routed_rubric_classifier(
    training_jsonl: str | Path,
    output_path: str | Path,
    field_config: dict[str, dict[str, Any]] | None = None,
) -> int:
    examples = _read_jsonl(training_jsonl)
    classifier = RoutedRubricClassifier(field_config=field_config)
    classifier.fit(examples)
    classifier.save(output_path)
    return len(examples)


def run_routed_rubric_bakeoff(
    training_jsonl: str | Path,
    output_path: str | Path,
    *,
    runs: int = 8,
    group_field: str = "thread_id",
    embedding_cache_dir: str | Path | None = None,
    field_config: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    examples = _read_jsonl(training_jsonl)
    config = field_config or DEFAULT_FIELD_CONFIG
    texts = [model_text(row, use_context=False, use_metadata=False) for row in examples]
    cache = EmbeddingVectorCache(embedding_cache_dir) if embedding_cache_dir else None
    features_by_field = _build_features_by_field(texts, config, cache)
    splits = grouped_holdout_indexes(examples, runs=runs, group_field=group_field)
    fields = {}
    for field in TARGET_FIELDS:
        actual: list[str] = []
        predicted: list[str] = []
        for train_indexes, test_indexes in splits:
            model = _field_model(field, str(config[field]["strategy"]))
            model.fit(
                [features_by_field[field][index] for index in train_indexes],
                [label_value(examples[index].get(field)) for index in train_indexes],
            )
            actual.extend(label_value(examples[index].get(field)) for index in test_indexes)
            predicted.extend(model.predict(features_by_field[field][index]) for index in test_indexes)
        fields[field] = {
            "accuracy": sum(left == right for left, right in zip(actual, predicted)) / len(actual) if actual else 0.0,
            "macro_f1": _macro_f1(actual, predicted),
            "evaluated": len(actual),
        }
    result = {
        "backend": "routed_rubric",
        "examples": len(examples),
        "evaluation": {
            "group_field": group_field,
            "groups": len({str(row.get(group_field) or row.get("source_item_id") or "") for row in examples}),
            "runs": runs,
        },
        "field_config": config,
        "fields": fields,
        "mean_macro_f1": sum(values["macro_f1"] for values in fields.values()) / len(fields),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def run_gated_precision_bakeoff(
    training_jsonl: str | Path,
    output_path: str | Path,
    *,
    runs: int = 8,
    group_field: str = "thread_id",
    embedding_cache_dir: str | Path | None = None,
    field_config: dict[str, dict[str, Any]] | None = None,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    calibrate: bool = False,
    calibration_runs: int = 3,
) -> dict[str, Any]:
    """Measure out-of-fold precision and coverage of confidence-gated predictions.

    Precision here is accuracy on the subset of holdout rows whose prediction
    confidence reaches the threshold; coverage is the fraction of rows kept.

    With ``calibrate`` the raw confidence is mapped through an isotonic
    regression fit on inner out-of-fold predictions inside each outer train
    split, so calibration never sees the holdout rows.
    """
    examples = _read_jsonl(training_jsonl)
    config = field_config or DEFAULT_FIELD_CONFIG
    texts = [model_text(row, use_context=False, use_metadata=False) for row in examples]
    cache = EmbeddingVectorCache(embedding_cache_dir) if embedding_cache_dir else None
    features_by_field = _build_features_by_field(texts, config, cache)
    splits = grouped_holdout_indexes(examples, runs=runs, group_field=group_field)
    fields: dict[str, Any] = {}
    for field in TARGET_FIELDS:
        outcomes: list[tuple[str, str, float]] = []
        for train_indexes, test_indexes in splits:
            calibrator = None
            if calibrate:
                calibrator = _fit_confidence_calibrator(
                    field,
                    config,
                    examples,
                    features_by_field[field],
                    train_indexes,
                    group_field=group_field,
                    runs=calibration_runs,
                )
            model = _field_model(field, str(config[field]["strategy"]))
            model.fit(
                [features_by_field[field][index] for index in train_indexes],
                [label_value(examples[index].get(field)) for index in train_indexes],
            )
            for index in test_indexes:
                details = model.predict_details(features_by_field[field][index])
                confidence = float(details["confidence"])
                if calibrator is not None:
                    confidence = float(calibrator.predict([confidence])[0])
                outcomes.append(
                    (
                        label_value(examples[index].get(field)),
                        str(details["label"]),
                        confidence,
                    )
                )
        fields[field] = {
            "evaluated": len(outcomes),
            "thresholds": {
                f"{threshold:.2f}": _gated_metrics(outcomes, threshold)
                for threshold in thresholds
            },
        }
    result = {
        "backend": "routed_rubric_gated_calibrated" if calibrate else "routed_rubric_gated",
        "examples": len(examples),
        "evaluation": {
            "group_field": group_field,
            "groups": len({str(row.get(group_field) or row.get("source_item_id") or "") for row in examples}),
            "runs": runs,
            "calibrated": calibrate,
        },
        "field_config": config,
        "fields": fields,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _build_features_by_field(
    texts: list[str],
    config: dict[str, dict[str, Any]],
    cache: EmbeddingVectorCache | None,
) -> dict[str, list[list[float]]]:
    """Build per-field feature rows: encoder embedding + rubric similarities + optional NLI scores."""
    vectors_by_encoder: dict[str, list[list[float]]] = {}
    embedder_by_encoder: dict[str, Any] = {}
    features_by_field: dict[str, list[list[float]]] = {}

    def _encoder_features(field: str, model_name: str, backend: str, scale: float) -> list[list[float]]:
        key = _encoder_key({"embedding_model": model_name, "backend": backend})
        if key not in vectors_by_encoder:
            embedder = HuggingFaceTextEmbedder(model_name=model_name, backend=backend)
            embedder_by_encoder[key] = embedder
            vectors_by_encoder[key] = _encode_with_cache(
                texts,
                embedding_model=key,
                embedder=embedder,
                cache=cache,
            )
        rubric_vectors = embedder_by_encoder[key].encode(list(LABEL_RUBRICS[field].values()))
        return [
            _rubric_features(vector, rubric_vectors, scale=scale)
            for vector in vectors_by_encoder[key]
        ]

    for field in TARGET_FIELDS:
        field_settings = config[field]
        features = _encoder_features(
            field,
            str(field_settings["embedding_model"]),
            str(field_settings["backend"]),
            float(field_settings["rubric_scale"]),
        )
        for extra in field_settings.get("extra_embedding_models", []):
            extra_features = _encoder_features(
                field,
                str(extra["embedding_model"]),
                str(extra.get("backend", "auto")),
                float(extra.get("rubric_scale", field_settings["rubric_scale"])),
            )
            features = [row + extra_row for row, extra_row in zip(features, extra_features)]
        nli_model = field_settings.get("nli_model")
        if nli_model:
            from social_benchmark.pipeline.nli_features import NliRubricScorer

            nli_scale = float(field_settings.get("nli_scale", 1.0))
            scorer = NliRubricScorer(
                model_name=str(nli_model),
                hypotheses=list(LABEL_RUBRICS[field].values()),
            )
            nli_vectors = _encode_with_cache(
                texts,
                embedding_model=f"nli|{nli_model}|{field}",
                embedder=scorer,
                cache=cache,
            )
            features = [
                row + [value * nli_scale for value in nli_row]
                for row, nli_row in zip(features, nli_vectors)
            ]
        features_by_field[field] = features
    return features_by_field


def _fit_confidence_calibrator(
    field: str,
    config: dict[str, dict[str, Any]],
    examples: list[dict[str, Any]],
    field_features: list[list[float]],
    train_indexes: list[int],
    *,
    group_field: str,
    runs: int,
) -> Any | None:
    """Fit isotonic confidence->P(correct) on inner out-of-fold predictions."""
    from sklearn.isotonic import IsotonicRegression

    train_examples = [examples[index] for index in train_indexes]
    inner_splits = grouped_holdout_indexes(train_examples, runs=runs, group_field=group_field)
    confidences: list[float] = []
    corrects: list[float] = []
    for inner_train, inner_test in inner_splits:
        model = _field_model(field, str(config[field]["strategy"]))
        model.fit(
            [field_features[train_indexes[index]] for index in inner_train],
            [label_value(train_examples[index].get(field)) for index in inner_train],
        )
        for index in inner_test:
            details = model.predict_details(field_features[train_indexes[index]])
            confidences.append(float(details["confidence"]))
            corrects.append(float(str(details["label"]) == label_value(train_examples[index].get(field))))
    if len(set(confidences)) < 2:
        return None
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(confidences, corrects)
    return calibrator


def _gated_metrics(outcomes: list[tuple[str, str, float]], threshold: float) -> dict[str, float]:
    covered = [(actual, predicted) for actual, predicted, confidence in outcomes if confidence >= threshold]
    correct = sum(actual == predicted for actual, predicted in covered)
    return {
        "coverage": len(covered) / len(outcomes) if outcomes else 0.0,
        "covered": len(covered),
        "precision": correct / len(covered) if covered else 0.0,
    }


def _encoder_key(config: dict[str, Any]) -> str:
    return f"{config['embedding_model']}|{config['backend']}"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
