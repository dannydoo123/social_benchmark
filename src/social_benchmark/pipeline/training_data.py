from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


TRAINING_FIELDS = [
    "text",
    "context_text",
    "provider_id",
    "model_id",
    "product_id",
    "inference_profile",
    "task_category",
    "aspect_category",
    "evidence_type",
    "polarity_score",
    "firsthand_flag",
    "source_platform",
    "source_item_id",
    "url",
]


def build_training_jsonl(
    labels_csv: str | Path,
    output_path: str | Path,
    context_jsonl: str | Path | None = None,
) -> int:
    rows = _read_csv(labels_csv)
    context_by_review_id = _context_index(context_jsonl) if context_jsonl else {}
    examples = [
        _training_example(row, context_by_review_id.get(row.get("review_id", "")))
        for row in rows
        if _has_human_label(row) and not _is_truthy(row.get("human_excluded_from_scoring"))
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return len(examples)


def _training_example(row: dict[str, str], context_row: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "text": row.get("evidence_text", ""),
        "context_text": _context_text(context_row),
        "provider_id": row.get("human_provider_id") or row.get("provider_id") or "",
        "model_id": row.get("human_model_id") or row.get("model_id"),
        "product_id": row.get("human_product_id") or row.get("product_id") or "",
        "inference_profile": row.get("human_inference_profile") or row.get("inference_profile") or "",
        "task_category": row.get("human_task_category") or row.get("task_category"),
        "aspect_category": row.get("human_aspect_category") or row.get("aspect_category"),
        "evidence_type": row.get("human_evidence_type") or row.get("evidence_type"),
        "polarity_score": _int_or_default(row.get("human_polarity_score"), row.get("polarity_score")),
        "firsthand_flag": _bool_or_default(row.get("human_firsthand_flag"), row.get("firsthand_flag")),
        "source_platform": row.get("source_platform", ""),
        "source_item_id": row.get("source_item_id", ""),
        "url": row.get("url", ""),
    }


def _has_human_label(row: dict[str, str]) -> bool:
    human_fields = [
        "human_model_id",
        "human_provider_id",
        "human_product_id",
        "human_inference_profile",
        "human_task_category",
        "human_aspect_category",
        "human_evidence_type",
        "human_polarity_score",
        "human_firsthand_flag",
    ]
    return any((row.get(field) or "").strip() for field in human_fields)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _context_index(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {str(row.get("review_id") or ""): row for row in rows if row.get("review_id")}


def _context_text(context_row: dict[str, Any] | None) -> str:
    if not context_row:
        return ""
    return str(context_row.get("raw_full_text") or "").strip()


def _int_or_default(value: str | None, default: str | None) -> int:
    raw = value if value not in (None, "") else default
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return 0


def _bool_or_default(value: str | None, default: str | None) -> bool:
    raw = value if value not in (None, "") else default
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}
