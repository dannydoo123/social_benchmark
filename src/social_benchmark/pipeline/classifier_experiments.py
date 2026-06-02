from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.embeddings import _text_hash
from social_benchmark.pipeline.embeddings import HuggingFaceTextEmbedder
from social_benchmark.pipeline.local_classifier import TARGET_FIELDS
from social_benchmark.pipeline.text_features import label_value, model_text

DEFAULT_EMBEDDING_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
)
TEXT_MODES = ("evidence_only", "augmented")


def run_frozen_embedding_bakeoff(
    training_jsonl: str | Path,
    output_path: str | Path,
    *,
    embedding_models: tuple[str, ...] = DEFAULT_EMBEDDING_MODELS,
    text_modes: tuple[str, ...] = TEXT_MODES,
    runs: int = 4,
    test_fraction: float = 0.25,
    group_field: str = "source_item_id",
    embedding_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    examples = _read_jsonl(training_jsonl)
    splits = grouped_holdout_indexes(examples, runs=runs, test_fraction=test_fraction, group_field=group_field)
    embedding_cache = EmbeddingVectorCache(embedding_cache_dir) if embedding_cache_dir else None
    result: dict[str, Any] = {
        "examples": len(examples),
        "evaluation": {
            "group_field": group_field,
            "groups": len({_group_key(row, index, group_field=group_field) for index, row in enumerate(examples)}),
            "runs": runs,
            "test_fraction": test_fraction,
            "embedding_cache_dir": str(embedding_cache_dir) if embedding_cache_dir else None,
        },
        "tfidf": {},
        "embedding_models": {},
    }
    for mode in text_modes:
        result["tfidf"][mode] = evaluate_tfidf_grouped(examples, splits=splits, text_mode=mode)
    for embedding_model in embedding_models:
        result["embedding_models"][embedding_model] = {}
        for mode in text_modes:
            result["embedding_models"][embedding_model][mode] = evaluate_embeddings_grouped(
                examples,
                embedding_model=embedding_model,
                splits=splits,
                text_mode=mode,
                embedding_cache=embedding_cache,
            )
    result["ranking"] = rank_experiments(result)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def evaluate_tfidf_grouped(
    examples: list[dict[str, Any]],
    *,
    splits: list[tuple[list[int], list[int]]],
    text_mode: str,
) -> dict[str, Any]:
    feature_extraction, linear_model, pipeline = _sklearn()
    texts = [_text_for_mode(row, text_mode) for row in examples]

    def fit_predict(field: str, train_indexes: list[int], test_indexes: list[int]) -> list[str]:
        labels = [label_value(examples[index].get(field)) for index in train_indexes]
        if len(set(labels)) < 2:
            return [labels[0] if labels else "" for _ in test_indexes]
        features = pipeline.FeatureUnion(
            [
                ("word", feature_extraction.TfidfVectorizer(ngram_range=(1, 2), max_features=16000, sublinear_tf=True)),
                (
                    "char",
                    feature_extraction.TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=16000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        model = pipeline.Pipeline(
            [
                ("features", features),
                ("classifier", linear_model.LogisticRegression(class_weight="balanced", max_iter=1200)),
            ]
        )
        model.fit([texts[index] for index in train_indexes], labels)
        return [str(value) for value in model.predict([texts[index] for index in test_indexes])]

    return _evaluate_fields(examples, splits=splits, fit_predict=fit_predict, backend="tfidf_logistic", text_mode=text_mode)


def evaluate_embeddings_grouped(
    examples: list[dict[str, Any]],
    *,
    embedding_model: str,
    splits: list[tuple[list[int], list[int]]],
    text_mode: str,
    embedding_cache: "EmbeddingVectorCache | None" = None,
    embedder: Any | None = None,
) -> dict[str, Any]:
    _, linear_model, _ = _sklearn()
    texts = [_text_for_mode(row, text_mode) for row in examples]
    resolved_embedder = embedder or HuggingFaceTextEmbedder(model_name=embedding_model)
    vectors = _encode_with_cache(texts, embedding_model=embedding_model, embedder=resolved_embedder, cache=embedding_cache)

    def fit_predict(field: str, train_indexes: list[int], test_indexes: list[int]) -> list[str]:
        labels = [label_value(examples[index].get(field)) for index in train_indexes]
        if len(set(labels)) < 2:
            return [labels[0] if labels else "" for _ in test_indexes]
        model = linear_model.LogisticRegression(class_weight="balanced", max_iter=1200)
        model.fit([vectors[index] for index in train_indexes], labels)
        return [str(value) for value in model.predict([vectors[index] for index in test_indexes])]

    result = _evaluate_fields(examples, splits=splits, fit_predict=fit_predict, backend="embedding_logistic", text_mode=text_mode)
    result["embedding_model"] = embedding_model
    result["embedding_cache"] = embedding_cache.summary(embedding_model) if embedding_cache else None
    return result


class EmbeddingVectorCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._vectors_by_model: dict[str, dict[str, list[float]]] = {}
        self._stats_by_model: dict[str, dict[str, int]] = {}

    def vectors_for(self, model_name: str) -> dict[str, list[float]]:
        if model_name not in self._vectors_by_model:
            self._vectors_by_model[model_name] = self._read_model_cache(model_name)
            self._stats_by_model[model_name] = {"hits": 0, "misses": 0, "stored": len(self._vectors_by_model[model_name])}
        return self._vectors_by_model[model_name]

    def record_hits(self, model_name: str, count: int) -> None:
        self._stats(model_name)["hits"] += count

    def record_misses(self, model_name: str, count: int) -> None:
        self._stats(model_name)["misses"] += count

    def flush(self, model_name: str) -> None:
        vectors = self.vectors_for(model_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {"embedding_model": model_name, "vectors": vectors}
        self._cache_path(model_name).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        self._stats(model_name)["stored"] = len(vectors)

    def summary(self, model_name: str) -> dict[str, int | str]:
        stats = dict(self._stats(model_name))
        stats["path"] = str(self._cache_path(model_name))
        return stats

    def _stats(self, model_name: str) -> dict[str, int]:
        self.vectors_for(model_name)
        return self._stats_by_model[model_name]

    def _read_model_cache(self, model_name: str) -> dict[str, list[float]]:
        path = self._cache_path(model_name)
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("embedding_model") != model_name:
            return {}
        return {
            str(text_hash): [float(value) for value in vector]
            for text_hash, vector in dict(payload.get("vectors") or {}).items()
            if isinstance(vector, list)
        }

    def _cache_path(self, model_name: str) -> Path:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_") or "embedding_model"
        return self.cache_dir / f"{name}.json"


def _encode_with_cache(
    texts: list[str],
    *,
    embedding_model: str,
    embedder: Any,
    cache: EmbeddingVectorCache | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    if cache is None:
        return _encode_unique_texts(texts, embedder)

    stored = cache.vectors_for(embedding_model)
    hashes = [_text_hash(text) for text in texts]
    missing_text_by_hash: dict[str, str] = {}
    hits = 0
    for text_hash, text in zip(hashes, texts):
        if text_hash in stored:
            hits += 1
        else:
            missing_text_by_hash.setdefault(text_hash, text)
    cache.record_hits(embedding_model, hits)
    cache.record_misses(embedding_model, len(missing_text_by_hash))
    if missing_text_by_hash:
        new_hashes = list(missing_text_by_hash)
        vectors = embedder.encode([missing_text_by_hash[text_hash] for text_hash in new_hashes])
        for text_hash, vector in zip(new_hashes, vectors):
            stored[text_hash] = [float(value) for value in vector]
        cache.flush(embedding_model)
    return [stored[text_hash] for text_hash in hashes]


def _encode_unique_texts(texts: list[str], embedder: Any) -> list[list[float]]:
    unique_texts = list(dict.fromkeys(texts))
    vectors_by_text = {
        text: [float(value) for value in vector]
        for text, vector in zip(unique_texts, embedder.encode(unique_texts))
    }
    return [vectors_by_text[text] for text in texts]


def grouped_holdout_indexes(
    examples: list[dict[str, Any]],
    *,
    runs: int = 4,
    test_fraction: float = 0.25,
    group_field: str = "source_item_id",
) -> list[tuple[list[int], list[int]]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(examples):
        groups.setdefault(_group_key(row, index, group_field=group_field), []).append(index)
    keys = sorted(groups)
    test_count = max(1, round(len(keys) * test_fraction))
    splits = []
    for seed in range(runs):
        shuffled = list(keys)
        random.Random(seed).shuffle(shuffled)
        test_keys = set(shuffled[:test_count])
        train_indexes = [index for key in shuffled[test_count:] for index in groups[key]]
        test_indexes = [index for key in shuffled[:test_count] for index in groups[key]]
        splits.append((train_indexes, test_indexes))
    return splits


def rank_experiments(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mode, metrics in result["tfidf"].items():
        rows.append(_ranking_row("tfidf_logistic", mode, metrics))
    for checkpoint, modes in result["embedding_models"].items():
        for mode, metrics in modes.items():
            rows.append(_ranking_row(checkpoint, mode, metrics))
    return sorted(rows, key=lambda row: row["mean_macro_f1"], reverse=True)


def _evaluate_fields(
    examples: list[dict[str, Any]],
    *,
    splits: list[tuple[list[int], list[int]]],
    fit_predict: Any,
    backend: str,
    text_mode: str,
) -> dict[str, Any]:
    fields = {}
    for field in TARGET_FIELDS:
        actual: list[str] = []
        predicted: list[str] = []
        run_accuracies = []
        for train_indexes, test_indexes in splits:
            run_actual = [label_value(examples[index].get(field)) for index in test_indexes]
            run_predicted = fit_predict(field, train_indexes, test_indexes)
            actual.extend(run_actual)
            predicted.extend(run_predicted)
            run_accuracies.append(_accuracy(run_actual, run_predicted))
        fields[field] = {
            "accuracy": _accuracy(actual, predicted),
            "macro_f1": _macro_f1(actual, predicted),
            "evaluated": len(actual),
            "run_accuracy_min": min(run_accuracies) if run_accuracies else 0.0,
            "run_accuracy_max": max(run_accuracies) if run_accuracies else 0.0,
            "confusion": _confusion(actual, predicted),
        }
    return {
        "backend": backend,
        "text_mode": text_mode,
        "fields": fields,
        "mean_macro_f1": _mean_macro_f1(fields),
    }


def _ranking_row(name: str, mode: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "text_mode": mode,
        "mean_macro_f1": metrics["mean_macro_f1"],
        "field_macro_f1": {field: values["macro_f1"] for field, values in metrics["fields"].items()},
    }


def _text_for_mode(row: dict[str, Any], mode: str) -> str:
    if mode == "evidence_only":
        return model_text(row, use_context=False, use_metadata=False)
    if mode == "augmented":
        return model_text(row, use_context=True, use_metadata=True)
    raise ValueError(f"Unsupported text mode: {mode}")


def _group_key(row: dict[str, Any], index: int, *, group_field: str = "source_item_id") -> str:
    if group_field == "thread_id":
        return str(row.get("thread_id") or row.get("source_item_id") or f"row:{index}")
    if group_field == "source_item_id":
        return str(row.get("source_item_id") or row.get("thread_id") or f"row:{index}")
    return str(row.get(group_field) or row.get("source_item_id") or row.get("thread_id") or f"row:{index}")


def _mean_macro_f1(fields: dict[str, Any]) -> float:
    values = [metrics["macro_f1"] for metrics in fields.values()]
    return sum(values) / len(values) if values else 0.0


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


def _sklearn():
    try:
        from sklearn import feature_extraction, linear_model, pipeline
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to run classifier experiments.") from exc
    return feature_extraction.text, linear_model, pipeline
