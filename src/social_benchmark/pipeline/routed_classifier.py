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
            embedder = self._embedder(field)
            vectors = embedder.encode(texts)
            rubric_vectors = embedder.encode(list(LABEL_RUBRICS[field].values()))
            self.rubric_vectors[field] = rubric_vectors
            features = [_rubric_features(vector, rubric_vectors, scale=float(config["rubric_scale"])) for vector in vectors]
            model = _field_model(field, str(config["strategy"]))
            model.fit(features, [label_value(row.get(field)) for row in examples])
            self.models[field] = model

    def predict_row(self, row: dict[str, Any]) -> dict[str, dict[str, Any]]:
        text = model_text(row, use_context=False, use_metadata=False)
        predictions = {}
        for field in TARGET_FIELDS:
            config = self.field_config[field]
            vector = self._embedder(field).encode([text])[0]
            features = _rubric_features(vector, self.rubric_vectors[field], scale=float(config["rubric_scale"]))
            predictions[field] = self.models[field].predict_details(features)
        return predictions

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

    def _embedder(self, field: str) -> Any:
        config = self.field_config[field]
        key = _encoder_key(config)
        if key not in self.embedders:
            self.embedders[key] = HuggingFaceTextEmbedder(
                model_name=str(config["embedding_model"]),
                backend=str(config["backend"]),
            )
        return self.embedders[key]


def train_routed_rubric_classifier(training_jsonl: str | Path, output_path: str | Path) -> int:
    examples = _read_jsonl(training_jsonl)
    classifier = RoutedRubricClassifier()
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
    vectors_by_encoder: dict[str, list[list[float]]] = {}
    embedder_by_encoder: dict[str, Any] = {}
    features_by_field = {}
    for field in TARGET_FIELDS:
        field_settings = config[field]
        key = _encoder_key(field_settings)
        if key not in vectors_by_encoder:
            embedder = HuggingFaceTextEmbedder(
                model_name=str(field_settings["embedding_model"]),
                backend=str(field_settings["backend"]),
            )
            embedder_by_encoder[key] = embedder
            vectors_by_encoder[key] = _encode_with_cache(
                texts,
                embedding_model=key,
                embedder=embedder,
                cache=cache,
            )
        rubric_vectors = embedder_by_encoder[key].encode(list(LABEL_RUBRICS[field].values()))
        features_by_field[field] = [
            _rubric_features(vector, rubric_vectors, scale=float(field_settings["rubric_scale"]))
            for vector in vectors_by_encoder[key]
        ]
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


def _encoder_key(config: dict[str, Any]) -> str:
    return f"{config['embedding_model']}|{config['backend']}"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
