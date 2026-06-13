import json
import tempfile
import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.score_snapshot import (
    _apply_review,
    _load_observations,
    _split_by_review,
    _to_observation,
    _trust_tier,
)


def _observation_row(item_id: str, model: str, polarity: int = 1) -> dict:
    return {
        "source_platform": "hacker_news",
        "community_id": "hacker_news",
        "thread_id": "t1",
        "source_item_id": item_id,
        "author_id_hash": f"author-{item_id}",
        "url": f"https://news.ycombinator.com/item?id={item_id}",
        "model_id": model,
        "provider_id": "anthropic",
        "model_version_or_alias": model,
        "task_category": "coding",
        "aspect_category": "satisfaction",
        "evidence_type": "firsthand_usage",
        "claim_type": "praise",
        "polarity_score": polarity,
        "severity_score": 0.5,
        "extractor_confidence": 0.6,
        "firsthand_flag": True,
        "comparative_flag": False,
        "regression_flag": False,
        "hallucination_flag": False,
        "refusal_flag": False,
        "value_flag": False,
        "weights": {"source_quality_weight": 1.15},
        "evidence_text": "Claude nailed the refactor on the first try.",
    }


class ScoreSnapshotTest(unittest.TestCase):
    def test_trust_tiers(self):
        self.assertEqual(_trust_tier(10), "insufficient")
        self.assertEqual(_trust_tier(60), "provisional")
        self.assertEqual(_trust_tier(200), "ranked")

    def test_review_application_overrides_labels_and_marks_human(self):
        row = _observation_row("1", "claude")
        review = {
            "human_model_id": "claude-opus-4.5",
            "human_task_category": "writing",
            "human_polarity_score": "-1",
            "human_firsthand_flag": "False",
        }
        updated = _apply_review(row, review)
        self.assertEqual(updated["model_id"], "claude-opus-4.5")
        self.assertEqual(updated["task_category"], "writing")
        self.assertEqual(updated["polarity_score"], -1)
        self.assertFalse(updated["firsthand_flag"])
        self.assertTrue(updated["human_labeled_flag"])

    def test_split_drops_excluded_and_routes_unreviewed_to_machine(self):
        rows = [_observation_row("1", "claude"), _observation_row("2", "claude"), _observation_row("3", "gpt-5")]
        reviewed = {
            ("1", "claude"): {"human_excluded_from_scoring": "True"},
            ("2", "claude"): {"human_excluded_from_scoring": "False", "human_polarity_score": "2"},
        }
        human, machine, dropped = _split_by_review(rows, reviewed)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(human), 1)
        self.assertEqual(human[0]["polarity_score"], 2)
        self.assertEqual(len(machine), 1)
        self.assertEqual(machine[0]["source_item_id"], "3")

    def test_observation_round_trip_and_firsthand_weight(self):
        observation = _to_observation(_observation_row("9", "claude", polarity=2))
        self.assertEqual(observation.mapped_score, 100.0)
        self.assertEqual(observation.weights.firsthand_weight, 1.25)
        row = _observation_row("9", "claude")
        row["firsthand_flag"] = False
        self.assertEqual(_to_observation(row).weights.firsthand_weight, 0.75)

    def test_observation_dedup_by_item_model_evidence(self):
        path = Path(tempfile.gettempdir()) / f"obs_{uuid.uuid4().hex}.jsonl"
        rows = [_observation_row("1", "claude"), _observation_row("1", "claude"), _observation_row("1", "gpt-5")]
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        try:
            loaded = _load_observations([path])
        finally:
            path.unlink()
        self.assertEqual(len(loaded), 2)


if __name__ == "__main__":
    unittest.main()
