import csv
import json
import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.label_feedback import apply_reviewed_labels, evaluate_label_csv


class LabelFeedbackTest(unittest.TestCase):
    def test_evaluates_and_applies_reviewed_labels(self):
        temp_dir = Path(".test_tmp") / f"feedback_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        labels = temp_dir / "labels.csv"
        observations = temp_dir / "observations.jsonl"
        output = temp_dir / "applied.jsonl"

        row = {
            "source_platform": "hacker_news",
            "source_item_id": "1",
            "model_id": "claude",
            "task_category": "general",
            "aspect_category": "satisfaction",
            "evidence_type": "hearsay",
            "polarity_score": "0",
            "firsthand_flag": "false",
            "human_polarity_score": "1",
            "human_firsthand_flag": "true",
        }
        with labels.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        observations.write_text(json.dumps(row) + "\n", encoding="utf-8")

        metrics = evaluate_label_csv(labels)
        updated = apply_reviewed_labels(observations, labels, output)
        applied = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(metrics["fields"]["polarity_score"]["accuracy"], 0.0)
        self.assertEqual(updated, 1)
        self.assertEqual(applied["polarity_score"], 1)
        self.assertTrue(applied["firsthand_flag"])
        self.assertTrue(applied["human_labeled_flag"])

    def test_applies_exclusion_only_rows(self):
        temp_dir = Path(".test_tmp") / f"feedback_exclusion_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        labels = temp_dir / "labels.csv"
        observations = temp_dir / "observations.jsonl"
        output = temp_dir / "applied.jsonl"

        row = {
            "source_platform": "hacker_news",
            "source_item_id": "2",
            "model_id": "claude-opus-4.8",
            "task_category": "general",
            "aspect_category": "satisfaction",
            "human_excluded_from_scoring": "true",
            "human_exclusion_reason": "factual_release_or_adoption",
            "reviewed_flag": "true",
        }
        with labels.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        observations.write_text(json.dumps(row) + "\n", encoding="utf-8")

        updated = apply_reviewed_labels(observations, labels, output)
        applied = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(updated, 1)
        self.assertTrue(applied["human_labeled_flag"])
        self.assertTrue(applied["human_excluded_from_scoring"])
        self.assertEqual(applied["human_exclusion_reason"], "factual_release_or_adoption")


if __name__ == "__main__":
    unittest.main()
