from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.classifier_experiments import EmbeddingVectorCache, _encode_with_cache, grouped_holdout_indexes
from social_benchmark.pipeline.embeddings import HuggingFaceTextEmbedder
from social_benchmark.pipeline.local_classifier import TARGET_FIELDS
from social_benchmark.pipeline.routed_classifier import DEFAULT_FIELD_CONFIG, _encoder_key, _read_jsonl
from social_benchmark.pipeline.rubric_classifier import LABEL_RUBRICS, _rubric_features
from social_benchmark.pipeline.structured_classifier import _field_model, _macro_f1
from social_benchmark.pipeline.text_features import label_value, model_text

DEFAULT_CHAIN_ORDERS = (
    ("firsthand_flag", "evidence_type", "aspect_category", "task_category", "polarity_score"),
    ("firsthand_flag", "evidence_type", "task_category", "aspect_category", "polarity_score"),
)


def run_soft_chain_bakeoff(
    training_jsonl: str | Path,
    output_path: str | Path,
    *,
    runs: int = 8,
    inner_runs: int = 4,
    group_field: str = "thread_id",
    embedding_cache_dir: str | Path | None = None,
    field_config: dict[str, dict[str, Any]] | None = None,
    chain_orders: tuple[tuple[str, ...], ...] = DEFAULT_CHAIN_ORDERS,
    embedders: dict[str, Any] | None = None,
) -> dict[str, Any]:
    for order in chain_orders:
        _validate_order(order)
    examples = _read_jsonl(training_jsonl)
    config = field_config or DEFAULT_FIELD_CONFIG
    features_by_field = _field_features(
        examples,
        config=config,
        embedding_cache_dir=embedding_cache_dir,
        embedders=embedders,
    )
    splits = grouped_holdout_indexes(examples, runs=runs, group_field=group_field)
    variants = {}
    for order in chain_orders:
        fields = _evaluate_chain(
            examples,
            features_by_field=features_by_field,
            splits=splits,
            order=order,
            config=config,
            group_field=group_field,
            inner_runs=inner_runs,
        )
        name = " -> ".join(order)
        variants[name] = {
            "order": list(order),
            "fields": fields,
            "mean_macro_f1": sum(values["macro_f1"] for values in fields.values()) / len(fields),
        }
    selective_order = DEFAULT_CHAIN_ORDERS[0]
    selective_dependencies = {"evidence_type": ("firsthand_flag",)}
    fields = _evaluate_chain(
        examples,
        features_by_field=features_by_field,
        splits=splits,
        order=selective_order,
        config=config,
        group_field=group_field,
        inner_runs=inner_runs,
        dependencies=selective_dependencies,
    )
    variants["selective: firsthand_flag -> evidence_type"] = {
        "order": list(selective_order),
        "dependencies": {field: list(upstream) for field, upstream in selective_dependencies.items()},
        "fields": fields,
        "mean_macro_f1": sum(values["macro_f1"] for values in fields.values()) / len(fields),
    }
    result = {
        "backend": "soft_classifier_chain",
        "examples": len(examples),
        "evaluation": {
            "group_field": group_field,
            "groups": len({str(row.get(group_field) or row.get("source_item_id") or "") for row in examples}),
            "runs": runs,
            "inner_runs": inner_runs,
        },
        "field_config": config,
        "variants": variants,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _evaluate_chain(
    examples: list[dict[str, Any]],
    *,
    features_by_field: dict[str, list[list[float]]],
    splits: list[tuple[list[int], list[int]]],
    order: tuple[str, ...],
    config: dict[str, dict[str, Any]],
    group_field: str,
    inner_runs: int,
    dependencies: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    actual = {field: [] for field in TARGET_FIELDS}
    predicted = {field: [] for field in TARGET_FIELDS}
    for train_indexes, test_indexes in splits:
        train_probabilities = {index: {} for index in train_indexes}
        test_probabilities = {index: {} for index in test_indexes}
        inner_splits = _grouped_kfold_indexes(examples, train_indexes, folds=inner_runs, group_field=group_field)
        for field in order:
            upstream_fields = dependencies.get(field, ()) if dependencies is not None else order[: order.index(field)]
            labels = [label_value(examples[index].get(field)) for index in train_indexes]
            model = _field_model(field, str(config[field]["strategy"]))
            model.fit(
                [
                    _augmented(features_by_field[field][index], train_probabilities[index], upstream_fields)
                    for index in train_indexes
                ],
                labels,
            )
            for index in test_indexes:
                vector = _augmented(features_by_field[field][index], test_probabilities[index], upstream_fields)
                predicted[field].append(model.predict(vector))
                test_probabilities[index][field] = _probability_vector(model, vector, field)
                actual[field].append(label_value(examples[index].get(field)))

            oof_probabilities: dict[int, list[float]] = {}
            for inner_train, inner_test in inner_splits:
                inner_model = _field_model(field, str(config[field]["strategy"]))
                inner_model.fit(
                    [
                        _augmented(features_by_field[field][index], train_probabilities[index], upstream_fields)
                        for index in inner_train
                    ],
                    [label_value(examples[index].get(field)) for index in inner_train],
                )
                for index in inner_test:
                    vector = _augmented(features_by_field[field][index], train_probabilities[index], upstream_fields)
                    oof_probabilities[index] = _probability_vector(inner_model, vector, field)
            for index in train_indexes:
                train_probabilities[index][field] = oof_probabilities[index]
    return {
        field: {
            "accuracy": _accuracy(actual[field], predicted[field]),
            "macro_f1": _macro_f1(actual[field], predicted[field]),
            "evaluated": len(actual[field]),
        }
        for field in TARGET_FIELDS
    }


def _field_features(
    examples: list[dict[str, Any]],
    *,
    config: dict[str, dict[str, Any]],
    embedding_cache_dir: str | Path | None,
    embedders: dict[str, Any] | None,
) -> dict[str, list[list[float]]]:
    texts = [model_text(row, use_context=False, use_metadata=False) for row in examples]
    cache = EmbeddingVectorCache(embedding_cache_dir) if embedding_cache_dir else None
    resolved_embedders = embedders or {}
    vectors_by_encoder: dict[str, list[list[float]]] = {}
    features_by_field = {}
    for field in TARGET_FIELDS:
        settings = config[field]
        key = _encoder_key(settings)
        if key not in resolved_embedders:
            resolved_embedders[key] = HuggingFaceTextEmbedder(
                model_name=str(settings["embedding_model"]),
                backend=str(settings["backend"]),
            )
        if key not in vectors_by_encoder:
            vectors_by_encoder[key] = _encode_with_cache(
                texts,
                embedding_model=key,
                embedder=resolved_embedders[key],
                cache=cache,
            )
        rubric_vectors = resolved_embedders[key].encode(list(LABEL_RUBRICS[field].values()))
        features_by_field[field] = [
            _rubric_features(vector, rubric_vectors, scale=float(settings["rubric_scale"]))
            for vector in vectors_by_encoder[key]
        ]
    return features_by_field


def _probability_vector(model: Any, vector: list[float], field: str) -> list[float]:
    labels = list(LABEL_RUBRICS[field])
    if getattr(model, "model", None) is None:
        default = str(getattr(model, "default", ""))
        return [1.0 if label == default else 0.0 for label in labels]
    classes = [str(value) for value in model.model.classes_]
    probabilities = model.model.predict_proba([vector])[0]
    by_label = {label: float(probability) for label, probability in zip(classes, probabilities)}
    return [by_label.get(label, 0.0) for label in labels]


def _augmented(base: list[float], probabilities: dict[str, list[float]], upstream_fields: tuple[str, ...]) -> list[float]:
    return [*base, *(value for field in upstream_fields for value in probabilities[field])]


def _validate_order(order: tuple[str, ...]) -> None:
    if len(order) != len(TARGET_FIELDS) or set(order) != set(TARGET_FIELDS):
        raise ValueError("Chain order must contain every target field exactly once.")
    if order[-1] != "polarity_score":
        raise ValueError("Ordinal polarity must be last because it does not expose class probabilities.")


def _grouped_kfold_indexes(
    examples: list[dict[str, Any]],
    indexes: list[int],
    *,
    folds: int,
    group_field: str,
) -> list[tuple[list[int], list[int]]]:
    groups: dict[str, list[int]] = {}
    for index in indexes:
        row = examples[index]
        key = str(row.get(group_field) or row.get("source_item_id") or f"row:{index}")
        groups.setdefault(key, []).append(index)
    keys = sorted(groups)
    random.Random(0).shuffle(keys)
    fold_count = min(max(2, folds), len(keys))
    partitions = [keys[offset::fold_count] for offset in range(fold_count)]
    return [
        (
            [index for key in keys if key not in test_keys for index in groups[key]],
            [index for key in test_keys for index in groups[key]],
        )
        for test_keys in map(set, partitions)
    ]


def _accuracy(actual: list[str], predicted: list[str]) -> float:
    return sum(left == right for left, right in zip(actual, predicted)) / len(actual) if actual else 0.0
