from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.embeddings import DEFAULT_EMBEDDING_MODEL
from social_benchmark.pipeline.local_classifier import evaluate_classifier
from social_benchmark.pipeline.sklearn_classifier import evaluate_sklearn_classifier


def compare_classifier_backends(
    training_jsonl: str | Path,
    output_path: str | Path,
    *,
    runs: int = 8,
    hf_model_name: str | None = None,
    include_hf: bool = True,
    include_ensemble: bool = True,
    ensemble_min_confidence: float = 0.62,
    ensemble_min_agreement: int = 2,
    ensemble_field_min_confidence: dict[str, float] | None = None,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "training": str(training_jsonl),
        "runs": runs,
        "backends": {"naive_bayes": evaluate_classifier(training_jsonl)},
    }
    try:
        comparison["backends"]["sklearn_tfidf_logistic"] = evaluate_sklearn_classifier(training_jsonl, runs=runs)
    except Exception as exc:  # noqa: BLE001 - comparison should report unavailable optional backends.
        comparison["backends"]["sklearn_tfidf_logistic"] = {
            "available": False,
            "error": str(exc),
            "hint": "Install scikit-learn to evaluate the current sklearn baseline.",
        }
    if include_hf:
        try:
            from social_benchmark.pipeline.hf_classifier import evaluate_hf_classifier

            comparison["backends"]["hf_embedding_logistic"] = evaluate_hf_classifier(
                training_jsonl,
                model_name=hf_model_name or DEFAULT_EMBEDDING_MODEL,
                runs=runs,
            )
        except Exception as exc:  # noqa: BLE001 - comparison should report unavailable optional backends.
            comparison["backends"]["hf_embedding_logistic"] = {
                "available": False,
                "error": str(exc),
                "hint": "Install optional Hugging Face dependencies and make the embedding model available locally.",
            }
    if include_ensemble:
        try:
            from social_benchmark.pipeline.high_precision_classifier import evaluate_high_precision_classifier

            comparison["backends"]["high_precision_ensemble"] = evaluate_high_precision_classifier(
                training_jsonl,
                runs=runs,
                min_confidence=ensemble_min_confidence,
                min_agreement=ensemble_min_agreement,
                include_hf=include_hf,
                hf_model_name=hf_model_name or DEFAULT_EMBEDDING_MODEL,
                field_min_confidence=ensemble_field_min_confidence,
            )
        except Exception as exc:  # noqa: BLE001 - comparison should report unavailable optional backends.
            comparison["backends"]["high_precision_ensemble"] = {
                "available": False,
                "error": str(exc),
                "hint": "Install scikit-learn and optional Hugging Face dependencies for the ensemble backend.",
            }
    comparison["summary"] = _summary(comparison["backends"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    return comparison


def _summary(backends: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for backend, metrics in backends.items():
        fields = metrics.get("fields") or {}
        field_scores = {
            field: {
                "accuracy": values.get("accuracy", values.get("precision", 0.0)),
                "macro_f1": values.get("macro_f1"),
                "coverage": values.get("coverage"),
                "evaluated": values.get("evaluated", values.get("total", 0)),
            }
            for field, values in fields.items()
        }
        comparable_scores = [score["macro_f1"] for score in field_scores.values() if score.get("macro_f1") is not None]
        rows[backend] = {
            "available": metrics.get("available", True),
            "examples": metrics.get("examples", 0),
            "error": metrics.get("error"),
            "mean_macro_f1": sum(comparable_scores) / len(comparable_scores) if comparable_scores else None,
            "fields": field_scores,
        }
    return rows
