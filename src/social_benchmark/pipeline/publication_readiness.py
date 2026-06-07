from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_MIN_FIELD_MACRO_F1 = 0.70
DEFAULT_MIN_WORST_FIELD_MACRO_F1 = 0.60
DEFAULT_MIN_EXAMPLES = 300
DEFAULT_MIN_GROUPS = 75


def assess_publication_readiness(
    result: dict[str, Any],
    *,
    min_field_macro_f1: float = DEFAULT_MIN_FIELD_MACRO_F1,
    min_worst_field_macro_f1: float = DEFAULT_MIN_WORST_FIELD_MACRO_F1,
    min_examples: int = DEFAULT_MIN_EXAMPLES,
    min_groups: int = DEFAULT_MIN_GROUPS,
) -> dict[str, Any]:
    fields = result.get("fields") or {}
    field_scores = {field: float(metrics.get("macro_f1") or 0.0) for field, metrics in fields.items()}
    failures = []
    if int(result.get("examples") or 0) < min_examples:
        failures.append(f"examples<{min_examples}")
    groups = int((result.get("evaluation") or {}).get("groups") or 0)
    if groups < min_groups:
        failures.append(f"groups<{min_groups}")
    if field_scores and sum(field_scores.values()) / len(field_scores) < min_field_macro_f1:
        failures.append(f"mean_macro_f1<{min_field_macro_f1}")
    for field, score in field_scores.items():
        if score < min_worst_field_macro_f1:
            failures.append(f"{field}<{min_worst_field_macro_f1}")
    return {
        "ready": not failures,
        "failures": failures,
        "examples": int(result.get("examples") or 0),
        "groups": groups,
        "field_macro_f1": field_scores,
    }


def write_publication_readiness(result_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    assessment = assess_publication_readiness(result)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(assessment, indent=2, sort_keys=True), encoding="utf-8")
    return assessment
