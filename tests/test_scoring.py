import unittest

from social_benchmark.pipeline.models import (
    AspectCategory,
    ClaimType,
    EvidenceType,
    Observation,
    SourcePlatform,
    TaskCategory,
    WeightComponents,
)
from social_benchmark.pipeline.scoring import ScoreAggregator, effective_sample_size, wilson_interval


class ScoringTest(unittest.TestCase):
    def test_aspect_score_uses_weighted_mapped_score(self):
        observations = [
            _observation("gpt-4o", 2, 2.0),
            _observation("gpt-4o", -2, 1.0),
        ]

        scores = ScoreAggregator().aspect_scores(observations)

        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores[0].score, (2.0 * 100 + 1.0 * 0) / 3.0)

    def test_effective_sample_size(self):
        self.assertAlmostEqual(effective_sample_size([1, 1, 1, 1]), 4.0)
        self.assertLess(effective_sample_size([10, 1, 1, 1]), 4.0)

    def test_wilson_interval_bounds(self):
        low, high = wilson_interval(3, 10)
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 1)
        self.assertLess(low, high)

    def test_thread_cap_reduces_single_thread_dominance(self):
        observations = [
            *[_observation("gpt-4o", -2, 1.0, thread_id="bad-thread") for _ in range(10)],
            _observation("gpt-4o", 2, 1.0, thread_id="good-thread"),
        ]

        score = ScoreAggregator().aspect_scores(observations)[0]

        self.assertGreater(score.score, 25)
        self.assertLess(score.capped_weighted_n, score.weighted_n)
        self.assertIsNotNone(score.confidence_low)
        self.assertIsNotNone(score.confidence_high)
        self.assertFalse(score.publishable)
        self.assertIn("effective_sample_size_below_30", score.publication_blockers)


def _observation(model_id: str, polarity: int, weight: float, thread_id: str = "t1") -> Observation:
    return Observation(
        source_platform=SourcePlatform.HACKER_NEWS,
        community_id="hacker_news",
        thread_id=thread_id,
        source_item_id=f"{model_id}-{polarity}-{weight}",
        author_id_hash=None,
        url="",
        published_at=None,
        model_id=model_id,
        provider_id="openai",
        model_version_or_alias=model_id,
        task_category=TaskCategory.CODING,
        aspect_category=AspectCategory.SATISFACTION,
        evidence_type=EvidenceType.FIRSTHAND_USAGE,
        claim_type=ClaimType.PRAISE if polarity > 0 else ClaimType.COMPLAINT,
        polarity_score=polarity,
        severity_score=1,
        extractor_confidence=0.9,
        firsthand_flag=True,
        comparative_flag=False,
        regression_flag=False,
        hallucination_flag=False,
        refusal_flag=False,
        value_flag=False,
        weights=WeightComponents(source_quality_weight=weight),
    )


if __name__ == "__main__":
    unittest.main()
