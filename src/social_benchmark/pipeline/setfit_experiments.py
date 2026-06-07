from __future__ import annotations

import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.classifier_experiments import grouped_holdout_indexes
from social_benchmark.pipeline.local_classifier import TARGET_FIELDS
from social_benchmark.pipeline.text_features import label_value, model_text

DEFAULT_SETFIT_CHECKPOINTS = (
    ("BAAI/bge-small-en-v1.5", "augmented"),
    ("sentence-transformers/all-mpnet-base-v2", "evidence_only"),
)


def run_setfit_bakeoff(
    training_jsonl: str | Path,
    output_path: str | Path,
    *,
    checkpoints: tuple[tuple[str, str], ...] = DEFAULT_SETFIT_CHECKPOINTS,
    fields: tuple[str, ...] = TARGET_FIELDS,
    test_fraction: float = 0.25,
    num_epochs: int = 1,
    num_iterations: int = 4,
    batch_size: int = 16,
    max_steps: int = -1,
    group_field: str = "source_item_id",
) -> dict[str, Any]:
    examples = _read_jsonl(training_jsonl)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    train_indexes, test_indexes = grouped_holdout_indexes(
        examples,
        runs=1,
        test_fraction=test_fraction,
        group_field=group_field,
    )[0]
    result: dict[str, Any] = {
        "examples": len(examples),
        "evaluation": {
            "group_field": group_field,
            "runs": 1,
            "test_fraction": test_fraction,
            "train_examples": len(train_indexes),
            "test_examples": len(test_indexes),
        },
        "training": {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "num_iterations": num_iterations,
            "max_steps": max_steps,
        },
        "checkpoints": {},
    }
    for checkpoint, text_mode in checkpoints:
        checkpoint_result = {"checkpoint": checkpoint, "text_mode": text_mode, "fields": {}}
        result["checkpoints"][checkpoint] = checkpoint_result
        for field in fields:
            checkpoint_result["fields"][field] = _train_evaluate_field(
                examples,
                train_indexes=train_indexes,
                test_indexes=test_indexes,
                checkpoint=checkpoint,
                text_mode=text_mode,
                field=field,
                num_epochs=num_epochs,
                num_iterations=num_iterations,
                batch_size=batch_size,
                max_steps=max_steps,
            )
            checkpoint_result["mean_macro_f1"] = _mean_macro_f1(checkpoint_result["fields"])
            _write_result(output, result)
        checkpoint_result["mean_macro_f1"] = _mean_macro_f1(checkpoint_result["fields"])
    result["ranking"] = sorted(
        [
            {"checkpoint": checkpoint, "text_mode": values["text_mode"], "mean_macro_f1": values["mean_macro_f1"]}
            for checkpoint, values in result["checkpoints"].items()
        ],
        key=lambda row: row["mean_macro_f1"],
        reverse=True,
    )
    _write_result(output, result)
    return result


def parse_checkpoint_specs(specs: list[str]) -> tuple[tuple[str, str], ...]:
    if not specs:
        return DEFAULT_SETFIT_CHECKPOINTS
    parsed = []
    for spec in specs:
        checkpoint, separator, text_mode = spec.partition("|")
        if not separator or text_mode not in {"evidence_only", "augmented"}:
            raise ValueError("SetFit checkpoint specs must use CHECKPOINT|evidence_only or CHECKPOINT|augmented.")
        parsed.append((checkpoint, text_mode))
    return tuple(parsed)


def _train_evaluate_field(
    examples: list[dict[str, Any]],
    *,
    train_indexes: list[int],
    test_indexes: list[int],
    checkpoint: str,
    text_mode: str,
    field: str,
    num_epochs: int,
    num_iterations: int,
    batch_size: int,
    max_steps: int,
) -> dict[str, Any]:
    Dataset, SetFitModel, Trainer, TrainingArguments = _setfit()
    train_texts = [_text_for_mode(examples[index], text_mode) for index in train_indexes]
    test_texts = [_text_for_mode(examples[index], text_mode) for index in test_indexes]
    train_labels = [label_value(examples[index].get(field)) for index in train_indexes]
    test_labels = [label_value(examples[index].get(field)) for index in test_indexes]
    classes = sorted(set(train_labels))
    class_to_id = {label: index for index, label in enumerate(classes)}
    model = SetFitModel.from_pretrained(checkpoint, labels=classes)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_iterations=num_iterations,
            max_steps=max_steps,
            save_strategy="no",
            logging_strategy="no",
            report_to="none",
            show_progress_bar=False,
        ),
        train_dataset=Dataset.from_dict({"text": train_texts, "label": [class_to_id[label] for label in train_labels]}),
    )
    trainer.train()
    predicted = [str(value) for value in model.predict(test_texts)]
    metrics = {
        "accuracy": _accuracy(test_labels, predicted),
        "macro_f1": _macro_f1(test_labels, predicted),
        "evaluated": len(test_labels),
        "classes_in_train": classes,
        "classes_only_in_test": sorted(set(test_labels) - set(classes)),
        "confusion": _confusion(test_labels, predicted),
    }
    del trainer
    del model
    gc.collect()
    return metrics


def _text_for_mode(row: dict[str, Any], mode: str) -> str:
    if mode == "evidence_only":
        return model_text(row, use_context=False, use_metadata=False)
    if mode == "augmented":
        return model_text(row, use_context=True, use_metadata=True)
    raise ValueError(f"Unsupported text mode: {mode}")


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


def _write_result(output: Path, result: dict[str, Any]) -> None:
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def _setfit():
    try:
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install setfit to run SetFit experiments: py -3.12 -m pip install setfit") from exc
    return Dataset, SetFitModel, Trainer, TrainingArguments
