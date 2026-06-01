from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def observation_report(observations_path: str | Path) -> dict[str, Any]:
    observations = _read_jsonl(observations_path)
    total = len(observations)
    if not observations:
        return {"observations": 0}

    report: dict[str, Any] = {
        "observations": total,
        "source_platform": _counter(observations, "source_platform"),
        "model_id": _counter(observations, "model_id"),
        "aspect_category": _counter(observations, "aspect_category"),
        "task_category": _counter(observations, "task_category"),
        "evidence_type": _counter(observations, "evidence_type"),
        "claim_type": _counter(observations, "claim_type"),
        "flags": {
            flag: sum(1 for item in observations if item.get(flag))
            for flag in [
                "firsthand_flag",
                "comparative_flag",
                "regression_flag",
                "hallucination_flag",
                "refusal_flag",
                "value_flag",
            ]
        },
        "confidence": _confidence_summary(observations),
        "evidence_text": _evidence_text_summary(observations),
        "quality_warnings": [],
    }

    if report["flags"]["firsthand_flag"] / total < 0.05:
        report["quality_warnings"].append("low_firsthand_detection")
    if report["claim_type"].get("neutral", 0) / total > 0.60:
        report["quality_warnings"].append("neutral_overproduction")
    if report["evidence_type"].get("hearsay", 0) / total > 0.50:
        report["quality_warnings"].append("hearsay_overproduction")
    if report["confidence"]["avg"] < 0.65:
        report["quality_warnings"].append("low_average_extractor_confidence")
    return report


def write_observation_report(observations_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    report = observation_report(observations_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _counter(observations: list[dict[str, Any]], field: str, limit: int = 25) -> dict[str, int]:
    return dict(Counter(str(item.get(field, "")) for item in observations).most_common(limit))


def _confidence_summary(observations: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(item.get("extractor_confidence") or 0) for item in observations]
    return {
        "avg": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "below_0_60": sum(1 for value in values if value < 0.60),
        "below_0_72": sum(1 for value in values if value < 0.72),
    }


def _evidence_text_summary(observations: list[dict[str, Any]]) -> dict[str, float]:
    lengths = [len(str(item.get("evidence_text") or "").split()) for item in observations]
    return {
        "avg_words": sum(lengths) / len(lengths),
        "min_words": min(lengths),
        "max_words": max(lengths),
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

