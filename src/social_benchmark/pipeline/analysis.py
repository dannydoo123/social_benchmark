from __future__ import annotations

import csv
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


def source_yield_report(
    *,
    source_platform: str,
    raw_items_path: str | Path | None = None,
    observations_path: str | Path | None = None,
    review_queue_path: str | Path | None = None,
    training_paths: list[str | Path] | None = None,
    snapshot_path: str | Path | None = None,
) -> dict[str, Any]:
    """Report collection-to-benchmark yield for one source platform."""
    raw_rows = _read_jsonl(raw_items_path) if raw_items_path else []
    observation_rows = _read_jsonl(observations_path) if observations_path else []
    queue_rows = _read_csv(review_queue_path) if review_queue_path else []
    training_rows = _read_many_jsonl(training_paths or [])
    snapshot = _read_json(snapshot_path) if snapshot_path else {}

    raw_source = _count_source(raw_rows, source_platform, default_to_source=True)
    observations_source = _count_source(observation_rows, source_platform)
    queue_source = _count_source(queue_rows, source_platform)
    training_source = _count_source(training_rows, source_platform)
    source_mix = (snapshot.get("corpus") or {}).get("source_mix") or {}
    benchmark_source = int(source_mix.get(source_platform) or training_source)
    benchmark_total = int(sum(int(value or 0) for value in source_mix.values()) or len(training_rows))

    report: dict[str, Any] = {
        "source_platform": source_platform,
        "raw_items": {
            "total": len(raw_rows),
            "source": raw_source,
        },
        "observations": {
            "total": len(observation_rows),
            "source": observations_source,
            "raw_to_observation_pct": _pct(observations_source, raw_source),
        },
        "review_queue": {
            "total": len(queue_rows),
            "source": queue_source,
            "raw_to_queue_pct": _pct(queue_source, raw_source),
            "observation_to_queue_pct": _pct(queue_source, observations_source),
        },
        "reviewed_training": {
            "total": len(training_rows),
            "source": training_source,
            "queue_to_training_pct": _pct(training_source, queue_source),
            "raw_to_training_pct": _pct(training_source, raw_source),
        },
        "benchmark": {
            "total": benchmark_total,
            "source": benchmark_source,
            "source_share_pct": _pct(benchmark_source, benchmark_total),
        },
    }
    report["source_breakdown"] = {
        "raw_items": _source_counter(raw_rows, default_source=source_platform if raw_items_path else ""),
        "observations": _source_counter(observation_rows),
        "review_queue": _source_counter(queue_rows),
        "reviewed_training": _source_counter(training_rows),
        "benchmark": source_mix,
    }
    report["notes"] = _yield_notes(report)
    return report


def write_source_yield_report(
    *,
    source_platform: str,
    output_path: str | Path,
    raw_items_path: str | Path | None = None,
    observations_path: str | Path | None = None,
    review_queue_path: str | Path | None = None,
    training_paths: list[str | Path] | None = None,
    snapshot_path: str | Path | None = None,
) -> dict[str, Any]:
    report = source_yield_report(
        source_platform=source_platform,
        raw_items_path=raw_items_path,
        observations_path=observations_path,
        review_queue_path=review_queue_path,
        training_paths=training_paths,
        snapshot_path=snapshot_path,
    )
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


def _read_many_jsonl(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    return rows


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _count_source(rows: list[dict[str, Any]], source_platform: str, *, default_to_source: bool = False) -> int:
    if default_to_source and rows and all("source_platform" not in row and "platform" not in row for row in rows):
        return len(rows)
    return sum(1 for row in rows if _row_source(row) == source_platform)


def _source_counter(rows: list[dict[str, Any]], *, default_source: str = "") -> dict[str, int]:
    if default_source and rows and all("source_platform" not in row and "platform" not in row for row in rows):
        return {default_source: len(rows)}
    return dict(Counter(_row_source(row) for row in rows).most_common())


def _row_source(row: dict[str, Any]) -> str:
    return str(row.get("source_platform") or row.get("platform") or "")


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round((numerator / denominator) * 100, 2)


def _yield_notes(report: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if report["raw_items"]["source"] == 0:
        notes.append("No raw rows for this source were provided, so yield cannot be measured yet.")
    if report["observations"]["source"] == 0 and report["raw_items"]["source"] > 0:
        notes.append("Raw rows produced no extracted observations; inspect source targeting and extractor rules.")
    if report["reviewed_training"]["source"] == 0:
        notes.append("No reviewed training rows for this source are included yet.")
    return notes

