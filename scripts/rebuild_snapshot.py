"""Re-score the dashboard snapshot with canonical model resolution, a provider
rollup, fixed overall tiering, and recalibrated ESS tiers.

This runs off the human-reviewed training corpus only (no classifier/embeddings),
so it is fast and deterministic. It reuses the production scoring engine
(:class:`ScoreAggregator`) and the snapshot helpers, and normalizes every
observation through :mod:`model_resolver` so unversioned mentions ("Claude")
roll up to the provider instead of fragmenting the per-model board.

Usage:
    python scripts/rebuild_snapshot.py [training.jsonl] [--out web/public/snapshot.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from social_benchmark.pipeline.model_resolver import UNSPECIFIED_MODEL, is_registered, resolve
from social_benchmark.pipeline.score_snapshot import (
    EVIDENCE_SAMPLES_PER_CELL,
    EVIDENCE_SPAN_LIMIT,
    DEFAULT_GATES,
    _methodology_block,
    _read_jsonl,
    _to_observation,
)
from social_benchmark.pipeline.models import Observation
from social_benchmark.pipeline.scoring import (
    ScoreAggregator,
    contribution_capped_weights,
    effective_sample_size,
)

# Recalibrated for the current single-source corpus (~1.1k reviewed obs). The
# original 30/150 cutoffs were tuned for a much larger multi-platform corpus and
# left every model "insufficient"; these let well-evidenced providers/models rank.
RANKED_ESS = 50.0
PROVISIONAL_ESS = 15.0

DEFAULT_TRAINING = "datasets/training/hn_manual_training_threaded_round8_2026-06-12_merged.jsonl"
METHODOLOGY_ARTIFACT = "datasets/evaluation/gated_precision_polflat_calibrated_round8_2026-06-12.json"


def tier_for(ess: float) -> str:
    if ess >= RANKED_ESS:
        return "ranked"
    if ess >= PROVISIONAL_ESS:
        return "provisional"
    return "insufficient"


def to_observations(rows: list[dict]) -> tuple[list[Observation], dict, Counter]:
    """Resolve + convert rows. Returns (model_pool, model->provider, coverage counter)."""
    model_pool: list[Observation] = []
    model_provider: dict[str, str] = {}
    coverage = Counter()
    unspecified_provider = Counter()
    unregistered = Counter()
    for row in rows:
        res = resolve(str(row.get("model_id") or ""), str(row.get("provider_id") or ""))
        coverage[res.status] += 1
        data = dict(row)
        data["provider_id"] = res.provider_id
        data["human_labeled_flag"] = True
        data["evidence_text"] = row.get("evidence_text") or row.get("text") or ""
        data["model_id"] = res.model_id or UNSPECIFIED_MODEL
        observation = _to_observation(data)  # provider_id already set on data above
        model_pool.append(observation)
        if res.model_id is None:
            unspecified_provider[res.provider_id] += 1
        else:
            model_provider[res.model_id] = res.provider_id
            if not is_registered(res.model_id):
                unregistered[res.model_id] += 1
    coverage_block = {
        "by_status": dict(coverage),
        "unspecified_by_provider": dict(unspecified_provider.most_common()),
        "unregistered_versioned": dict(unregistered.most_common(40)),
    }
    return model_pool, model_provider, coverage_block


def overall_with_tiers(
    aggregator: ScoreAggregator,
    observations: list[Observation],
) -> list[dict]:
    """Overall score per entity (model_id), tiered on the entity's TOTAL evidence
    rather than its weakest aspect."""
    overall = aggregator.overall_scores(observations)
    by_entity: dict[str, list[Observation]] = defaultdict(list)
    for obs in observations:
        by_entity[obs.model_id].append(obs)
    rows = []
    for entity, score in overall.items():
        group = by_entity[entity]
        ess = effective_sample_size(contribution_capped_weights(group))
        rows.append(
            {
                "model_id": entity,
                "score": round(score, 2),
                "ess": round(ess, 1),
                "n_observations": len(group),
                "n_threads": len({o.thread_id for o in group if o.thread_id}),
                "tier": tier_for(ess),
            }
        )
    rows.sort(key=lambda r: (-(r["tier"] == "ranked"), -(r["tier"] == "provisional"), -r["score"]))
    return rows


def leaderboard_rows(aggregator: ScoreAggregator, pool: list[Observation], provider_of: dict[str, str]) -> list[dict]:
    by_cell: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for obs in pool:
        by_cell[(obs.model_id, obs.aspect_category.value)].append(obs)
    rows = []
    for snap in aggregator.aspect_scores(pool):
        group = by_cell[(snap.model_id, snap.aspect_category)]
        firsthand = sum(1 for o in group if o.firsthand_flag)
        rows.append(
            {
                "model_id": snap.model_id,
                "provider_id": provider_of.get(snap.model_id, ""),
                "aspect": snap.aspect_category,
                "score": round(snap.score, 2),
                "ci": [round(snap.confidence_low or 0.0, 2), round(snap.confidence_high or 0.0, 2)],
                "ess": round(snap.effective_n, 1),
                "weighted_n": round(snap.weighted_n, 1),
                "n_observations": len(group),
                "n_threads": len({o.thread_id for o in group if o.thread_id}),
                "n_authors": len({o.author_id_hash for o in group if o.author_id_hash}),
                "firsthand_ratio": round(firsthand / len(group), 3) if group else 0.0,
                "human_share": 1.0,
                "warnings": snap.warnings,
                "publishable": snap.publishable,
                "tier": tier_for(snap.effective_n),
            }
        )
    return rows


def provider_board(aggregator: ScoreAggregator, pool: list[Observation], unspecified_by_provider: dict[str, int]) -> list[dict]:
    """Roll every observation up to its provider (versioned + unversioned)."""
    provider_pool = [dataclasses.replace(obs, model_id=obs.provider_id) for obs in pool]
    aspects_by_provider: dict[str, list[dict]] = defaultdict(list)
    cells: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for obs in provider_pool:
        cells[(obs.model_id, obs.aspect_category.value)].append(obs)
    for snap in aggregator.aspect_scores(provider_pool):
        aspects_by_provider[snap.model_id].append(
            {
                "aspect": snap.aspect_category,
                "score": round(snap.score, 2),
                "ci": [round(snap.confidence_low or 0.0, 2), round(snap.confidence_high or 0.0, 2)],
                "ess": round(snap.effective_n, 1),
                "n_observations": len(cells[(snap.model_id, snap.aspect_category)]),
                "tier": tier_for(snap.effective_n),
            }
        )

    overall = overall_with_tiers(aggregator, provider_pool)
    models_per_provider: dict[str, set[str]] = defaultdict(set)
    for obs in pool:
        if obs.model_id != UNSPECIFIED_MODEL:
            models_per_provider[obs.provider_id].add(obs.model_id)

    rows = []
    for entry in overall:
        provider = entry["model_id"]
        rows.append(
            {
                "provider_id": provider,
                "score": entry["score"],
                "ess": entry["ess"],
                "n_observations": entry["n_observations"],
                "n_threads": entry["n_threads"],
                "n_models": len(models_per_provider.get(provider, set())),
                "unspecified_observations": unspecified_by_provider.get(provider, 0),
                "tier": entry["tier"],
                "aspects": sorted(aspects_by_provider.get(provider, []), key=lambda a: a["aspect"]),
            }
        )
    return rows


def evidence_samples(pool: list[Observation]) -> dict[str, list[dict]]:
    by_cell: dict[str, list[Observation]] = defaultdict(list)
    for obs in pool:
        by_cell[f"{obs.model_id}|{obs.aspect_category.value}"].append(obs)
        by_cell[f"{obs.provider_id}|{obs.aspect_category.value}"].append(obs)
    samples: dict[str, list[dict]] = {}
    for key, group in by_cell.items():
        ranked = sorted(group, key=lambda o: o.final_weight, reverse=True)[:EVIDENCE_SAMPLES_PER_CELL]
        samples[key] = [
            {
                "span": o.evidence_text[:EVIDENCE_SPAN_LIMIT],
                "url": o.url,
                "polarity": o.polarity_score,
                "evidence_type": o.evidence_type.value,
                "firsthand": o.firsthand_flag,
                "human_labeled": o.human_labeled_flag,
            }
            for o in ranked
        ]
    return samples


def task_rows(aggregator: ScoreAggregator, pool: list[Observation]) -> dict[str, list[dict]]:
    by_task: dict[str, list[Observation]] = defaultdict(list)
    for obs in pool:
        if obs.model_id != UNSPECIFIED_MODEL:
            by_task[obs.task_category.value].append(obs)
    result: dict[str, list[dict]] = {}
    for task, group in sorted(by_task.items()):
        result[task] = [
            {
                "model_id": snap.model_id,
                "aspect": snap.aspect_category,
                "score": round(snap.score, 2),
                "ci": [round(snap.confidence_low or 0.0, 2), round(snap.confidence_high or 0.0, 2)],
                "ess": round(snap.effective_n, 1),
                "tier": tier_for(snap.effective_n),
            }
            for snap in aggregator.aspect_scores(group)
        ]
    return result


def main() -> None:
    global RANKED_ESS, PROVISIONAL_ESS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", nargs="?", default=DEFAULT_TRAINING)
    parser.add_argument("--out", default="web/public/snapshot.json")
    parser.add_argument("--ranked-ess", type=float, default=RANKED_ESS)
    parser.add_argument("--provisional-ess", type=float, default=PROVISIONAL_ESS)
    args = parser.parse_args()

    RANKED_ESS, PROVISIONAL_ESS = args.ranked_ess, args.provisional_ess

    rows = _read_jsonl(args.training)
    full_pool, provider_of, coverage = to_observations(rows)
    model_pool = [obs for obs in full_pool if obs.model_id != UNSPECIFIED_MODEL]

    aggregator = ScoreAggregator()
    leaderboard = leaderboard_rows(aggregator, model_pool, provider_of)
    overall = overall_with_tiers(aggregator, model_pool)
    providers = provider_board(aggregator, full_pool, coverage["unspecified_by_provider"])
    tasks = task_rows(aggregator, model_pool)
    evidence = evidence_samples(full_pool)

    snapshot = {
        "snapshot_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gates": DEFAULT_GATES,
        "tier_thresholds": {"ranked_ess": RANKED_ESS, "provisional_ess": PROVISIONAL_ESS},
        "corpus": {
            "human_reviewed_included": len(full_pool),
            "machine_included": 0,
            "machine_rejected_by_gates": 0,
            "excluded_by_review": 0,
            "training_examples": len(rows),
            "models": len({o.model_id for o in model_pool}),
            "providers": len({o.provider_id for o in full_pool}),
            "unspecified_observations": sum(coverage["unspecified_by_provider"].values()),
            "threads": len({o.thread_id for o in full_pool if o.thread_id}),
        },
        "overall": overall,
        "providers": providers,
        "leaderboard": leaderboard,
        "tasks": tasks,
        "evidence": evidence,
        "coverage": coverage,
        "methodology": _methodology_block([METHODOLOGY_ARTIFACT], DEFAULT_GATES, len(rows)),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot), encoding="utf-8")

    print(
        json.dumps(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "models": snapshot["corpus"]["models"],
                "providers": snapshot["corpus"]["providers"],
                "unspecified_obs": snapshot["corpus"]["unspecified_observations"],
                "coverage": coverage["by_status"],
                "provider_top": [
                    {"provider": p["provider_id"], "score": p["score"], "ess": p["ess"], "tier": p["tier"], "n": p["n_observations"]}
                    for p in providers[:6]
                ],
                "out": str(out),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
