from __future__ import annotations

from typing import Any


TEXT_METADATA_FIELDS = ("provider_id", "model_id", "product_id", "inference_profile", "source_platform")


def model_text(row: dict[str, Any], *, use_context: bool = True, use_metadata: bool = True) -> str:
    evidence = str(row.get("text") or row.get("evidence_text") or "").strip()
    context = str(row.get("context_text") or row.get("raw_full_text") or "").strip()
    metadata = " ".join(
        f"{field}={str(row.get(field) or '').strip().lower()}"
        for field in TEXT_METADATA_FIELDS
        if row.get(field)
    )
    parts = [evidence]
    if use_context:
        parts.append(context)
    if use_metadata:
        parts.append(metadata)
    return "\n".join(part for part in parts if part)


def label_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)
