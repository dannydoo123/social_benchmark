import json
import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.analysis import observation_report, source_yield_report


class ObservationReportTest(unittest.TestCase):
    def test_reports_quality_warnings(self):
        temp_path = Path(".test_tmp") / f"report_{uuid.uuid4().hex}.jsonl"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(
            json.dumps(
                {
                    "source_platform": "hacker_news",
                    "model_id": "claude",
                    "aspect_category": "satisfaction",
                    "task_category": "general",
                    "evidence_type": "hearsay",
                    "claim_type": "neutral",
                    "extractor_confidence": 0.5,
                    "evidence_text": "Claude release",
                    "firsthand_flag": False,
                    "comparative_flag": False,
                    "regression_flag": False,
                    "hallucination_flag": False,
                    "refusal_flag": False,
                    "value_flag": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        report = observation_report(temp_path)

        self.assertEqual(report["observations"], 1)
        self.assertIn("low_firsthand_detection", report["quality_warnings"])
        self.assertIn("neutral_overproduction", report["quality_warnings"])

    def test_reports_source_yield_percentages(self):
        temp_dir = Path(".test_tmp") / f"source_yield_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        raw = temp_dir / "raw.jsonl"
        observations = temp_dir / "observations.jsonl"
        queue = temp_dir / "queue.csv"
        training = temp_dir / "training.jsonl"
        snapshot = temp_dir / "snapshot.json"

        raw.write_text(
            json.dumps({"platform": "reddit", "source_item_id": "r1"}) + "\n"
            + json.dumps({"platform": "reddit", "source_item_id": "r2"}) + "\n"
            + json.dumps({"platform": "reddit", "source_item_id": "r3"}) + "\n"
            + json.dumps({"platform": "reddit", "source_item_id": "r4"}) + "\n",
            encoding="utf-8",
        )
        observations.write_text(
            json.dumps({"source_platform": "reddit", "source_item_id": "r1"}) + "\n"
            + json.dumps({"source_platform": "reddit", "source_item_id": "r2"}) + "\n",
            encoding="utf-8",
        )
        queue.write_text(
            "source_platform,source_item_id\n"
            "reddit,r1\n"
            "reddit,r2\n",
            encoding="utf-8",
        )
        training.write_text(
            json.dumps({"source_platform": "reddit", "source_item_id": "r1"}) + "\n"
            + json.dumps({"source_platform": "hacker_news", "source_item_id": "h1"}) + "\n",
            encoding="utf-8",
        )
        snapshot.write_text(
            json.dumps({"corpus": {"source_mix": {"reddit": 1, "hacker_news": 3}}}),
            encoding="utf-8",
        )

        report = source_yield_report(
            source_platform="reddit",
            raw_items_path=raw,
            observations_path=observations,
            review_queue_path=queue,
            training_paths=[training],
            snapshot_path=snapshot,
        )

        self.assertEqual(report["raw_items"]["source"], 4)
        self.assertEqual(report["observations"]["raw_to_observation_pct"], 50.0)
        self.assertEqual(report["reviewed_training"]["raw_to_training_pct"], 25.0)
        self.assertEqual(report["benchmark"]["source_share_pct"], 25.0)

    def test_reports_zero_source_yield_without_source_files(self):
        report = source_yield_report(source_platform="reddit")

        self.assertEqual(report["raw_items"]["source"], 0)
        self.assertEqual(report["benchmark"]["source_share_pct"], None)
        self.assertIn("No raw rows for this source were provided", report["notes"][0])


if __name__ == "__main__":
    unittest.main()

