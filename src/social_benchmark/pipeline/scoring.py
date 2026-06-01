from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt

from social_benchmark.pipeline.models import AspectCategory, Observation, ScoreSnapshot


DEFAULT_OVERALL_WEIGHTS = {
    AspectCategory.SATISFACTION.value: 0.20,
    AspectCategory.TRUST_RELIABILITY.value: 0.18,
    AspectCategory.TASK_FIT.value: 0.22,
    AspectCategory.VALUE.value: 0.12,
    AspectCategory.REGRESSION_STABILITY.value: 0.12,
    AspectCategory.HALLUCINATION_SAFETY.value: 0.10,
    AspectCategory.REFUSAL_ACCEPTANCE.value: 0.06,
}


class ScoreAggregator:
    def aspect_scores(self, observations: list[Observation]) -> list[ScoreSnapshot]:
        grouped: dict[tuple[str, str], list[Observation]] = defaultdict(list)
        for observation in observations:
            grouped[(observation.model_id, observation.aspect_category.value)].append(observation)

        snapshots: list[ScoreSnapshot] = []
        for (model_id, aspect), group in sorted(grouped.items()):
            adjusted_weights = contribution_capped_weights(group)
            aspect_values = [self._score_for_aspect(item, aspect) for item in group]
            weighted_sum = sum(weight * value for weight, value in zip(adjusted_weights, aspect_values))
            total_weight = sum(adjusted_weights)
            score = weighted_sum / total_weight if total_weight else 0.0
            confidence_low, confidence_high = weighted_score_interval(aspect_values, adjusted_weights)
            warnings = confidence_warnings(group, adjusted_weights)
            publication_blockers = score_publication_blockers(
                effective_n=effective_sample_size(adjusted_weights),
                confidence_low=confidence_low,
                confidence_high=confidence_high,
                warnings=warnings,
            )
            snapshots.append(
                ScoreSnapshot(
                    model_id=model_id,
                    aspect_category=aspect,
                    score=score,
                    weighted_n=sum(item.final_weight for item in group),
                    effective_n=effective_sample_size(adjusted_weights),
                    confidence_low=confidence_low,
                    confidence_high=confidence_high,
                    warnings=warnings,
                    source_mix=source_mix(group, adjusted_weights),
                    capped_weighted_n=total_weight,
                    publishable=not publication_blockers,
                    publication_blockers=publication_blockers,
                )
            )
        return snapshots

    def overall_scores(self, observations: list[Observation]) -> dict[str, float]:
        by_model_aspect = {
            (snapshot.model_id, snapshot.aspect_category): snapshot.score
            for snapshot in self.aspect_scores(observations)
        }
        model_ids = sorted({observation.model_id for observation in observations})
        scores: dict[str, float] = {}
        for model_id in model_ids:
            weighted_sum = 0.0
            used_weight = 0.0
            for aspect, weight in DEFAULT_OVERALL_WEIGHTS.items():
                score = by_model_aspect.get((model_id, aspect))
                if score is None:
                    continue
                weighted_sum += score * weight
                used_weight += weight
            if used_weight:
                scores[model_id] = weighted_sum / used_weight
        return scores

    def complaint_rate(
        self,
        observations: list[Observation],
        flag_name: str,
    ) -> dict[str, tuple[float, tuple[float, float]]]:
        weighted_flags: dict[str, float] = defaultdict(float)
        weighted_total: dict[str, float] = defaultdict(float)
        counts: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
        for observation in observations:
            model_id = observation.model_id
            flagged = bool(getattr(observation, flag_name))
            weighted_total[model_id] += observation.final_weight
            if flagged:
                weighted_flags[model_id] += observation.final_weight
            successes, total = counts[model_id]
            counts[model_id] = (successes + int(flagged), total + 1)

        rates: dict[str, tuple[float, tuple[float, float]]] = {}
        for model_id, total_weight in weighted_total.items():
            rate = weighted_flags[model_id] / total_weight if total_weight else 0.0
            successes, total = counts[model_id]
            rates[model_id] = (rate, wilson_interval(successes, total))
        return rates

    def _score_for_aspect(self, observation: Observation, aspect: str) -> float:
        if aspect == AspectCategory.HALLUCINATION_SAFETY.value:
            return 0.0 if observation.hallucination_flag else observation.mapped_score
        if aspect == AspectCategory.REFUSAL_ACCEPTANCE.value:
            return 0.0 if observation.refusal_flag else observation.mapped_score
        if aspect == AspectCategory.REGRESSION_STABILITY.value:
            return 0.0 if observation.regression_flag else observation.mapped_score
        return observation.mapped_score


def effective_sample_size(weights: list[float]) -> float:
    if not weights:
        return 0.0
    total = sum(weights)
    squared = sum(weight * weight for weight in weights)
    return (total * total / squared) if squared else 0.0


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denominator = 1 + z * z / total
    center = (p_hat + z * z / (2 * total)) / denominator
    margin = z * sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def weighted_score_interval(values: list[float], weights: list[float], z: float = 1.96) -> tuple[float, float]:
    total_weight = sum(weights)
    if not values or not total_weight:
        return (0.0, 0.0)
    mean = sum(value * weight for value, weight in zip(values, weights)) / total_weight
    n_eff = effective_sample_size(weights)
    if n_eff <= 1:
        return (max(0.0, mean), min(100.0, mean))
    variance = sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights)) / total_weight
    margin = z * sqrt(variance / n_eff)
    return (max(0.0, mean - margin), min(100.0, mean + margin))


def contribution_capped_weights(observations: list[Observation]) -> list[float]:
    weights = [item.final_weight for item in observations]
    weights = _scale_cap(observations, weights, lambda item: item.thread_id or "", 0.12)
    weights = _scale_cap(observations, weights, lambda item: item.author_id_hash or "", 0.15)
    weights = _scale_cap(observations, weights, lambda item: item.community_id, 0.35)
    weights = _scale_cap(observations, weights, lambda item: item.source_platform.value, 0.60)
    return weights


def source_mix(observations: list[Observation], weights: list[float] | None = None) -> dict[str, float]:
    effective_weights = weights or [item.final_weight for item in observations]
    total_weight = sum(effective_weights)
    if not total_weight:
        return {}
    platform_weights = Counter()
    for observation, weight in zip(observations, effective_weights):
        platform_weights[observation.source_platform.value] += weight
    return {platform: weight / total_weight for platform, weight in sorted(platform_weights.items())}


def confidence_warnings(observations: list[Observation], weights: list[float] | None = None) -> list[str]:
    warnings: list[str] = []
    weights = weights or [item.final_weight for item in observations]
    n_eff = effective_sample_size(weights)
    if n_eff < 30:
        warnings.append("low_effective_sample_size")

    total_weight = sum(weights)
    if not total_weight:
        return warnings

    platform_weights = Counter()
    community_weights = Counter()
    thread_weights = Counter()
    author_weights = Counter()
    for observation, weight in zip(observations, weights):
        platform_weights[observation.source_platform.value] += weight
        community_weights[observation.community_id] += weight
        if observation.thread_id:
            thread_weights[observation.thread_id] += weight
        if observation.author_id_hash:
            author_weights[observation.author_id_hash] += weight

    if _max_share(platform_weights, total_weight) > 0.60:
        warnings.append("platform_overconcentrated")
    if _max_share(community_weights, total_weight) > 0.35:
        warnings.append("community_overconcentrated")
    if _max_share(thread_weights, total_weight) > 0.12:
        warnings.append("thread_overconcentrated")
    if _max_share(author_weights, total_weight) > 0.15:
        warnings.append("author_overconcentrated")
    return warnings


def score_publication_blockers(
    effective_n: float,
    confidence_low: float | None,
    confidence_high: float | None,
    warnings: list[str],
    max_ci_width: float = 30.0,
) -> list[str]:
    blockers: list[str] = []
    if effective_n < 30:
        blockers.append("effective_sample_size_below_30")
    if "platform_overconcentrated" in warnings:
        blockers.append("platform_overconcentrated")
    if "community_overconcentrated" in warnings:
        blockers.append("community_overconcentrated")
    if "thread_overconcentrated" in warnings:
        blockers.append("thread_overconcentrated")
    if "author_overconcentrated" in warnings:
        blockers.append("author_overconcentrated")
    if confidence_low is not None and confidence_high is not None and confidence_high - confidence_low > max_ci_width:
        blockers.append("confidence_interval_too_wide")
    return blockers


def _max_share(counter: Counter, total_weight: float) -> float:
    if not counter or not total_weight:
        return 0.0
    return max(counter.values()) / total_weight


def _scale_cap(
    observations: list[Observation],
    weights: list[float],
    key_func,
    max_share: float,
) -> list[float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, observation in enumerate(observations):
        key = key_func(observation)
        if key:
            groups[key].append(index)
    if len(groups) <= 1:
        return weights

    total_weight = sum(weights)
    if not total_weight:
        return weights
    cap = total_weight * max_share
    adjusted = list(weights)
    for indexes in groups.values():
        group_weight = sum(adjusted[index] for index in indexes)
        if group_weight > cap and group_weight:
            scale = cap / group_weight
            for index in indexes:
                adjusted[index] *= scale
    return adjusted
