from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


FIELD_MAP = {
    "provider_id": "human_provider_id",
    "model_id": "human_model_id",
    "product_id": "human_product_id",
    "inference_profile": "human_inference_profile",
    "task_category": "human_task_category",
    "aspect_category": "human_aspect_category",
    "evidence_type": "human_evidence_type",
    "polarity_score": "human_polarity_score",
    "firsthand_flag": "human_firsthand_flag",
}

EXTRA_FIELDS = {
    "human_excluded_from_scoring": "human_excluded_from_scoring",
    "human_exclusion_reason": "human_exclusion_reason",
    "human_notes": "human_notes",
    "reviewed_flag": "reviewed_flag",
}


def evaluate_label_csv(labels_csv: str | Path) -> dict[str, Any]:
    rows = _read_csv(labels_csv)
    metrics: dict[str, Any] = {"rows": len(rows), "fields": {}}
    for machine_field, human_field in FIELD_MAP.items():
        labeled = [row for row in rows if (row.get(human_field) or "").strip()]
        if not labeled:
            metrics["fields"][machine_field] = {"labeled": 0}
            continue
        correct = sum(
            1
            for row in labeled
            if _normalize(row.get(machine_field), machine_field) == _normalize(row.get(human_field), machine_field)
        )
        metrics["fields"][machine_field] = {
            "labeled": len(labeled),
            "correct": correct,
            "accuracy": correct / len(labeled),
            "confusion": _confusion(labeled, machine_field, human_field),
        }
    return metrics


def write_label_evaluation(labels_csv: str | Path, output_path: str | Path) -> dict[str, Any]:
    metrics = evaluate_label_csv(labels_csv)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def apply_reviewed_labels(
    observations_path: str | Path,
    labels_csv: str | Path,
    output_path: str | Path,
) -> int:
    observations = _read_jsonl(observations_path)
    labels = _label_index(_read_csv(labels_csv))
    updated: list[dict[str, Any]] = []
    count = 0
    for observation in observations:
        key = _row_key(observation)
        row = labels.get(key)
        if row:
            changed = _apply_row(observation, row)
            count += int(changed)
        updated.append(observation)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for observation in updated:
            handle.write(json.dumps(observation, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return count


def _apply_row(observation: dict[str, Any], row: dict[str, str]) -> bool:
    changed = False
    for machine_field, human_field in FIELD_MAP.items():
        value = (row.get(human_field) or "").strip()
        if not value:
            continue
        observation[machine_field] = _typed_value(value, machine_field)
        changed = True
    for observation_field, row_field in EXTRA_FIELDS.items():
        value = row.get(row_field)
        if value in (None, ""):
            continue
        observation[observation_field] = _typed_value(value, observation_field)
        changed = True
    if changed or _is_truthy(row.get("reviewed_flag")):
        observation["human_labeled_flag"] = True
        observation["human_notes"] = row.get("human_notes", "")
    return changed or _is_truthy(row.get("reviewed_flag"))


def _label_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    relevant_fields = [*FIELD_MAP.values(), *EXTRA_FIELDS.values()]
    return {_row_key(row): row for row in rows if any((row.get(field) or "").strip() for field in relevant_fields)}


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("source_platform") or ""),
        str(row.get("source_item_id") or ""),
        str(row.get("model_id") or ""),
        str(row.get("task_category") or ""),
        str(row.get("aspect_category") or ""),
    )


def _confusion(rows: list[dict[str, str]], machine_field: str, human_field: str) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        predicted = _normalize(row.get(machine_field), machine_field)
        actual = _normalize(row.get(human_field), machine_field)
        counts[f"{predicted} -> {actual}"] += 1
    return dict(counts.most_common(20))


def _normalize(value: str | None, field: str) -> str:
    if field == "firsthand_flag":
        return str(_typed_value(value or "", field)).lower()
    if field == "polarity_score":
        return str(_typed_value(value or "0", field))
    return (value or "").strip()


def _typed_value(value: str, field: str) -> Any:
    if field == "firsthand_flag":
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if field == "human_excluded_from_scoring":
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if field == "reviewed_flag":
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if field == "polarity_score":
        try:
            return int(value)
        except ValueError:
            return 0
    return value


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
