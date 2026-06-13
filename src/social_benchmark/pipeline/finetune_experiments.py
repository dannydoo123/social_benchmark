from __future__ import annotations

import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.classifier_experiments import grouped_holdout_indexes
from social_benchmark.pipeline.local_classifier import TARGET_FIELDS
from social_benchmark.pipeline.text_features import label_value, model_text

DEFAULT_FINETUNE_CHECKPOINT = "BAAI/bge-small-en-v1.5"


def run_finetuned_encoder_bakeoff(
    training_jsonl: str | Path,
    output_path: str | Path,
    *,
    checkpoint: str = DEFAULT_FINETUNE_CHECKPOINT,
    fields: tuple[str, ...] = TARGET_FIELDS,
    runs: int = 4,
    test_fraction: float = 0.25,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    unfrozen_layers: int = 2,
    group_field: str = "thread_id",
) -> dict[str, Any]:
    """Fine-tune a shared encoder with one linear head per field.

    The encoder keeps only its top ``unfrozen_layers`` transformer layers
    trainable; embeddings and lower layers stay frozen so the small corpus
    cannot wash out pretrained features. Evaluation is grouped-holdout like
    the frozen bakeoffs so numbers are directly comparable.
    """
    examples = _read_jsonl(training_jsonl)
    texts = [model_text(row, use_context=False, use_metadata=False) for row in examples]
    labels_by_field = {
        field: [label_value(row.get(field)) for row in examples] for field in fields
    }
    splits = grouped_holdout_indexes(
        examples, runs=runs, test_fraction=test_fraction, group_field=group_field
    )
    outcomes: dict[str, dict[str, list[str]]] = {
        field: {"actual": [], "predicted": []} for field in fields
    }
    for train_indexes, test_indexes in splits:
        predictions = _train_and_predict(
            texts,
            labels_by_field,
            train_indexes=train_indexes,
            test_indexes=test_indexes,
            checkpoint=checkpoint,
            fields=fields,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            unfrozen_layers=unfrozen_layers,
        )
        for field in fields:
            outcomes[field]["actual"].extend(labels_by_field[field][i] for i in test_indexes)
            outcomes[field]["predicted"].extend(predictions[field])
    field_results = {
        field: {
            "accuracy": _accuracy(data["actual"], data["predicted"]),
            "macro_f1": _macro_f1(data["actual"], data["predicted"]),
            "evaluated": len(data["actual"]),
            "confusion": _confusion(data["actual"], data["predicted"]),
        }
        for field, data in outcomes.items()
    }
    result = {
        "backend": "finetuned_shared_encoder",
        "checkpoint": checkpoint,
        "examples": len(examples),
        "evaluation": {
            "group_field": group_field,
            "runs": runs,
            "test_fraction": test_fraction,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "unfrozen_layers": unfrozen_layers,
        },
        "fields": field_results,
        "mean_macro_f1": sum(v["macro_f1"] for v in field_results.values()) / len(field_results),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _train_and_predict(
    texts: list[str],
    labels_by_field: dict[str, list[str]],
    *,
    train_indexes: list[int],
    test_indexes: list[int],
    checkpoint: str,
    fields: tuple[str, ...],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    unfrozen_layers: int,
) -> dict[str, list[str]]:
    torch, nn, AutoModel, AutoTokenizer = _transformers()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    encoder = AutoModel.from_pretrained(checkpoint).to(device)
    _freeze_lower_layers(encoder, unfrozen_layers)

    classes_by_field = {
        field: sorted({labels_by_field[field][i] for i in train_indexes}) for field in fields
    }
    hidden = encoder.config.hidden_size
    heads = nn.ModuleDict(
        {field: nn.Linear(hidden, len(classes_by_field[field])) for field in fields}
    ).to(device)

    # Fresh linear heads need a much higher learning rate than the
    # pretrained encoder layers to converge in a few epochs.
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": learning_rate},
            {"params": list(heads.parameters()), "lr": 1e-3},
        ]
    )
    loss_fn = nn.CrossEntropyLoss()

    def _encode(batch_texts: list[str]):
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden_states = encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    label_ids = {
        field: {label: i for i, label in enumerate(classes_by_field[field])} for field in fields
    }
    encoder.train()
    heads.train()
    order = list(train_indexes)
    generator = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        permutation = torch.randperm(len(order), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            batch = [order[i] for i in permutation[start : start + batch_size]]
            optimizer.zero_grad()
            pooled = _encode([texts[i] for i in batch])
            loss = None
            for field in fields:
                targets = torch.tensor(
                    [label_ids[field][labels_by_field[field][i]] for i in batch], device=device
                )
                field_loss = loss_fn(heads[field](pooled), targets)
                loss = field_loss if loss is None else loss + field_loss
            loss.backward()
            optimizer.step()

    encoder.eval()
    heads.eval()
    predictions: dict[str, list[str]] = {field: [] for field in fields}
    with torch.no_grad():
        for start in range(0, len(test_indexes), batch_size):
            batch = test_indexes[start : start + batch_size]
            pooled = _encode([texts[i] for i in batch])
            for field in fields:
                choices = heads[field](pooled).argmax(dim=-1).tolist()
                predictions[field].extend(classes_by_field[field][c] for c in choices)

    del encoder, heads, optimizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return predictions


def _freeze_lower_layers(encoder: Any, unfrozen_layers: int) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    layers = None
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layers = encoder.encoder.layer
    if layers is not None and unfrozen_layers > 0:
        for layer in layers[-unfrozen_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    if hasattr(encoder, "pooler") and encoder.pooler is not None:
        for parameter in encoder.pooler.parameters():
            parameter.requires_grad = True


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


def _transformers():
    try:
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install torch and transformers to run fine-tune experiments.") from exc
    return torch, nn, AutoModel, AutoTokenizer
