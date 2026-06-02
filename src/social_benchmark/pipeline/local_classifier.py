from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TARGET_FIELDS = ("task_category", "aspect_category", "evidence_type", "polarity_score", "firsthand_flag")
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*")
FIRSTHAND_HINTS = ("i ", "my ", "me ", "we ", "our ", "i've ", "i’d ", "i'd ", "i used", "i tried", "we used")
POSITIVE_HINTS = ("great", "better", "best", "useful", "worth it", "helpful", "accurate", "amazing", "good enough")
NEGATIVE_HINTS = ("bad", "worse", "broken", "slow", "frustrating", "annoying", "expensive", "wrong", "unsafe")
FEATURE_HINTS = {
    "__feature_firsthand__": ("i used", "i tried", "my workflow", "we used", "in production", "i've", "i have"),
    "__feature_compare__": ("better than", "worse than", "compared to", "vs ", " versus "),
    "__feature_value__": ("price", "pricing", "cost", "worth", "cheap", "expensive", "subscription"),
    "__feature_security__": ("sandbox", "safe", "unsafe", "security", "vulnerability", "rce", "secrets", "trusted", "untrusted"),
    "__feature_hallucination__": ("hallucinated", "made up", "invented", "fabricated", "wrong claims"),
    "__feature_regression__": ("regression", "got worse", "used to", "broke", "stopped working"),
    "__feature_refusal__": ("refused", "won't answer", "too cautious", "censored"),
    "__feature_api__": ("api", "sdk", "latency", "rate limit", "timeout", "integration", "deploy"),
    "__feature_coding__": ("code", "coding", "python", "typescript", "debug", "repo", "terminal"),
    "__feature_research__": ("research", "search", "citations", "paper", "web search"),
    "__feature_writing__": ("write", "writing", "draft", "essay", "tone", "copy"),
    "__feature_agents__": ("agent", "agents", "tool use", "workflow", "autonomous", "loop"),
}
BINARY_METADATA_FIELDS = ("provider_id", "model_id", "product_id", "inference_profile", "source_platform")


class LocalNaiveBayesClassifier:
    def __init__(self, models: dict[str, Any] | None = None, field_metrics: dict[str, Any] | None = None) -> None:
        self.models = models or {}
        self.field_metrics = field_metrics or {}

    def fit(self, examples: list[dict[str, Any]], target_fields: tuple[str, ...] = TARGET_FIELDS) -> None:
        self.models = {field: _train_field(examples, field) for field in target_fields}

    def predict(self, text: str) -> dict[str, Any]:
        return self.predict_row({"text": text})

    def predict_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return {field: _predict_field(model, row) for field, model in self.models.items()}

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"field_metrics": self.field_metrics, "models": self.models}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LocalNaiveBayesClassifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(models=payload["models"], field_metrics=payload.get("field_metrics") or {})


def train_classifier(training_jsonl: str | Path, output_path: str | Path) -> int:
    examples = _read_jsonl(training_jsonl)
    classifier = LocalNaiveBayesClassifier()
    classifier.fit(examples)
    classifier.field_metrics = _evaluate_classifier_examples(examples).get("fields", {})
    classifier.save(output_path)
    return len(examples)


def load_classifier(path: str | Path):
    if Path(path).suffix == ".joblib":
        try:
            import joblib

            payload = joblib.load(path)
            if isinstance(payload, dict) and payload.get("backend") == "hf_embedding":
                from social_benchmark.pipeline.hf_classifier import HFEmbeddingClassifier

                return HFEmbeddingClassifier.load(path)
            if isinstance(payload, dict) and payload.get("backend") == "high_precision_ensemble":
                from social_benchmark.pipeline.high_precision_classifier import HighPrecisionClassifier

                return HighPrecisionClassifier.load(path)
        except Exception:
            pass
        from social_benchmark.pipeline.sklearn_classifier import SklearnTextClassifier

        return SklearnTextClassifier.load(path)
    return LocalNaiveBayesClassifier.load(path)


def predict_jsonl(model_path: str | Path, input_jsonl: str | Path, output_path: str | Path) -> int:
    classifier = load_classifier(model_path)
    rows = _read_jsonl(input_jsonl)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            text = str(row.get("text") or row.get("evidence_text") or "")
            prediction = classifier.predict_row(row)
            handle.write(json.dumps({"text": text, "prediction": prediction}, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return len(rows)


def evaluate_classifier(training_jsonl: str | Path) -> dict[str, Any]:
    examples = _read_jsonl(training_jsonl)
    return _evaluate_classifier_examples(examples)


def _evaluate_classifier_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(examples) < 2:
        return {"examples": len(examples), "fields": {}}
    fields: dict[str, Any] = {}
    for field in TARGET_FIELDS:
        correct = 0
        total = 0
        for index, example in enumerate(examples):
            train = [item for train_index, item in enumerate(examples) if train_index != index]
            model = _train_field(train, field)
            predicted = _predict_field(model, example)
            actual = _label_value(example.get(field))
            correct += int(predicted["label"] == actual)
            total += 1
        fields[field] = {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}
    return {"examples": len(examples), "fields": fields}


def write_classifier_evaluation(training_jsonl: str | Path, output_path: str | Path) -> dict[str, Any]:
    metrics = evaluate_classifier(training_jsonl)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _train_field(examples: list[dict[str, Any]], field: str) -> dict[str, Any]:
    class_doc_counts = Counter()
    class_token_counts: dict[str, Counter] = defaultdict(Counter)
    class_total_tokens = Counter()
    vocabulary: set[str] = set()

    for example in examples:
        label = _label_value(example.get(field))
        tokens = _feature_tokens(example)
        class_doc_counts[label] += 1
        class_token_counts[label].update(tokens)
        class_total_tokens[label] += len(tokens)
        vocabulary.update(tokens)

    return {
        "classes": dict(class_doc_counts),
        "token_counts": {label: dict(counts) for label, counts in class_token_counts.items()},
        "total_tokens": dict(class_total_tokens),
        "vocabulary": sorted(vocabulary),
        "majority": class_doc_counts.most_common(1)[0][0] if class_doc_counts else "",
    }


def _predict_field(model: dict[str, Any], row_or_text: dict[str, Any] | str) -> dict[str, Any]:
    classes = model.get("classes") or {}
    if not classes:
        return {"label": "", "confidence": 0.0}
    if isinstance(row_or_text, str):
        tokens = _feature_tokens({"text": row_or_text})
    else:
        tokens = _feature_tokens(row_or_text)
    vocabulary = set(model.get("vocabulary") or [])
    vocab_size = max(1, len(vocabulary))
    total_docs = sum(int(count) for count in classes.values())
    scores: dict[str, float] = {}
    for label, doc_count in classes.items():
        score = math.log((int(doc_count) + 1) / (total_docs + len(classes)))
        token_counts = model.get("token_counts", {}).get(label, {})
        total_tokens = int(model.get("total_tokens", {}).get(label, 0))
        for token in tokens:
            score += math.log((int(token_counts.get(token, 0)) + 1) / (total_tokens + vocab_size))
        scores[label] = score

    best_label = max(scores, key=scores.get)
    confidence = _softmax_confidence(scores, best_label)
    return {"label": best_label, "confidence": confidence}


def _softmax_confidence(scores: dict[str, float], best_label: str) -> float:
    max_score = max(scores.values())
    exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
    total = sum(exp_scores.values())
    return exp_scores[best_label] / total if total else 0.0


def _tokens(text: str) -> list[str]:
    base_tokens = [token.lower() for token in TOKEN_RE.findall(text) if len(token) > 1]
    bigrams = [f"{base_tokens[index]}__{base_tokens[index + 1]}" for index in range(len(base_tokens) - 1)]
    return base_tokens + bigrams


def _feature_tokens(row: dict[str, Any]) -> list[str]:
    evidence_text = str(row.get("text") or row.get("evidence_text") or "")
    context_text = str(row.get("context_text") or row.get("raw_full_text") or "")
    combined_text = f"{evidence_text}\n{context_text}".strip()
    tokens = _tokens(evidence_text) + _tokens(evidence_text)
    if context_text:
        tokens.extend(_tokens(context_text))
    lowered = combined_text.lower()
    if evidence_text.strip().endswith("?"):
        tokens.append("__feature_question__")
    if any(hint in lowered for hint in FIRSTHAND_HINTS):
        tokens.append("__feature_first_person__")
    if any(hint in lowered for hint in POSITIVE_HINTS):
        tokens.append("__feature_positive__")
    if any(hint in lowered for hint in NEGATIVE_HINTS):
        tokens.append("__feature_negative__")
    for feature_token, hints in FEATURE_HINTS.items():
        if any(hint in lowered for hint in hints):
            tokens.append(feature_token)
    for field in BINARY_METADATA_FIELDS:
        value = str(row.get(field) or "").strip().lower()
        if value:
            tokens.append(f"__meta_{field}__{value}")
    return tokens


def _label_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
