from __future__ import annotations

from typing import Any

CONSTRAINT_SETS = {
    "none": (),
    "firsthand_consistency": ("firsthand_consistency",),
    "high_purity": ("firsthand_consistency", "integration_shape", "pricing_aspect", "release_firsthand"),
}


def resolve_prediction_labels(
    predictions: dict[str, str],
    *,
    enabled_rules: tuple[str, ...] = CONSTRAINT_SETS["high_purity"],
) -> dict[str, str]:
    resolved = dict(predictions)
    evidence = resolved.get("evidence_type", "")
    if "firsthand_consistency" in enabled_rules:
        if evidence == "firsthand_usage":
            resolved["firsthand_flag"] = "true"
        elif evidence == "hearsay":
            resolved["firsthand_flag"] = "false"
    if "integration_shape" in enabled_rules and evidence == "integration_failure":
        resolved["aspect_category"] = "developer_ergonomics"
        resolved["task_category"] = "api_developer_workflow"
    if "pricing_aspect" in enabled_rules and evidence == "pricing_value_comment":
        resolved["aspect_category"] = "value"
    if "release_firsthand" in enabled_rules and evidence == "release_update_reaction":
        resolved["firsthand_flag"] = "false"
    return resolved


def resolve_prediction(
    predictions: dict[str, dict[str, Any]],
    *,
    enabled_rules: tuple[str, ...] = CONSTRAINT_SETS["high_purity"],
) -> dict[str, dict[str, Any]]:
    labels = {field: str(value.get("label") or "") for field, value in predictions.items()}
    resolved_labels = resolve_prediction_labels(labels, enabled_rules=enabled_rules)
    resolved = {field: dict(value) for field, value in predictions.items()}
    for field, label in resolved_labels.items():
        if field not in resolved:
            resolved[field] = {"label": label, "confidence": 1.0}
        elif label != labels.get(field):
            resolved[field]["label"] = label
            resolved[field]["constraint_applied"] = True
    return resolved
