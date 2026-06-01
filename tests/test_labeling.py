import csv
import json
import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.labeling import export_labeling_queue
from social_benchmark.pipeline.local_classifier import train_classifier


class LabelingExportTest(unittest.TestCase):
    def test_exports_low_confidence_observations(self):
        temp_path = Path(".test_tmp") / f"labeling_{uuid.uuid4().hex}"
        temp_path.mkdir(parents=True)
        observations = temp_path / "observations.jsonl"
        output = temp_path / "labels.csv"
        observations.write_text(
            json.dumps(
                {
                    "source_platform": "hacker_news",
                    "community_id": "hacker_news",
                    "source_item_id": "1",
                    "model_id": "claude",
                    "task_category": "coding",
                    "aspect_category": "satisfaction",
                    "evidence_type": "hearsay",
                    "claim_type": "neutral",
                    "polarity_score": 0,
                    "extractor_confidence": 0.5,
                    "evidence_text": "Claude was okay for code.",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        written = export_labeling_queue(observations, output)

        self.assertEqual(written, 1)
        with output.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["model_id"], "claude")
        self.assertEqual(rows[0]["human_notes"], "")

    def test_exports_classifier_suggestions(self):
        temp_path = Path(".test_tmp") / f"labeling_model_{uuid.uuid4().hex}"
        temp_path.mkdir(parents=True)
        observations = temp_path / "observations.jsonl"
        training = temp_path / "training.jsonl"
        model = temp_path / "model.json"
        output = temp_path / "labels.csv"
        observations.write_text(
            json.dumps(
                {
                    "source_platform": "hacker_news",
                    "community_id": "hacker_news",
                    "source_item_id": "1",
                    "model_id": "claude-opus-4.8",
                    "task_category": "general",
                    "aspect_category": "satisfaction",
                    "evidence_type": "hearsay",
                    "claim_type": "neutral",
                    "polarity_score": 0,
                    "extractor_confidence": 0.5,
                    "firsthand_flag": False,
                    "regression_flag": False,
                    "hallucination_flag": False,
                    "refusal_flag": False,
                    "value_flag": False,
                    "evidence_text": "I used Claude Opus 4.8 for debugging and it was great.",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        training.write_text(
            json.dumps(
                {
                    "text": "I used Claude Opus 4.8 for debugging and it was great.",
                    "task_category": "coding",
                    "aspect_category": "satisfaction",
                    "evidence_type": "firsthand_usage",
                    "polarity_score": 1,
                    "firsthand_flag": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        train_classifier(training, model)

        written = export_labeling_queue(observations, output, classifier_model_path=model)

        self.assertEqual(written, 1)
        with output.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["classifier_task_category"], "coding")
        self.assertEqual(rows[0]["classifier_evidence_type"], "firsthand_usage")
        self.assertEqual(rows[0]["classifier_polarity_score"], "1")
        self.assertEqual(rows[0]["classifier_firsthand_flag"], "true")

    def test_exports_review_context_sidecar(self):
        temp_path = Path(".test_tmp") / f"labeling_context_{uuid.uuid4().hex}"
        temp_path.mkdir(parents=True)
        observations = temp_path / "observations.jsonl"
        raw = temp_path / "raw.jsonl"
        output = temp_path / "labels.csv"
        context_output = temp_path / "review_context.jsonl"
        observations.write_text(
            json.dumps(
                {
                    "source_platform": "hacker_news",
                    "community_id": "hacker_news",
                    "thread_id": "1",
                    "source_item_id": "1",
                    "url": "https://news.ycombinator.com/item?id=1",
                    "model_id": "claude",
                    "provider_id": "anthropic",
                    "task_category": "coding",
                    "aspect_category": "satisfaction",
                    "evidence_type": "hearsay",
                    "claim_type": "neutral",
                    "polarity_score": 0,
                    "extractor_confidence": 0.5,
                    "evidence_text": "Claude was okay for code.",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raw.write_text(
            json.dumps(
                {
                    "platform": "hacker_news",
                    "source_id": "1",
                    "title": "Ask HN: Claude for code",
                    "body": "Claude was okay for code.",
                    "url": "https://news.ycombinator.com/item?id=1",
                    "metadata": {"is_root_story": True},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        written = export_labeling_queue(
            observations,
            output,
            raw_items_path=raw,
            context_output_path=context_output,
        )

        self.assertEqual(written, 1)
        with output.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows[0]["review_id"])
        with context_output.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(records[0]["review_id"], rows[0]["review_id"])
        self.assertEqual(records[0]["raw_title"], "Ask HN: Claude for code")
        self.assertIn("Claude was okay for code.", records[0]["raw_full_text"])

    def test_excludes_already_reviewed_rows(self):
        temp_path = Path(".test_tmp") / f"labeling_exclude_{uuid.uuid4().hex}"
        temp_path.mkdir(parents=True)
        observations = temp_path / "observations.jsonl"
        reviewed = temp_path / "reviewed.csv"
        output = temp_path / "labels.csv"
        observations.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "source_platform": "hacker_news",
                            "community_id": "hacker_news",
                            "source_item_id": "1",
                            "model_id": "claude",
                            "task_category": "coding",
                            "aspect_category": "satisfaction",
                            "evidence_type": "hearsay",
                            "claim_type": "neutral",
                            "polarity_score": 0,
                            "extractor_confidence": 0.5,
                            "evidence_text": "Claude was okay for code.",
                        }
                    ),
                    json.dumps(
                        {
                            "source_platform": "hacker_news",
                            "community_id": "hacker_news",
                            "source_item_id": "2",
                            "model_id": "gpt-5",
                            "task_category": "coding",
                            "aspect_category": "satisfaction",
                            "evidence_type": "hearsay",
                            "claim_type": "neutral",
                            "polarity_score": 0,
                            "extractor_confidence": 0.5,
                            "evidence_text": "GPT-5 was okay for code.",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        reviewed.write_text(
            ",".join(
                [
                    "source_item_id",
                    "model_id",
                    "evidence_text",
                    "reviewed_flag",
                    "human_task_category",
                ]
            )
            + "\n"
            + ",".join(["1", "claude", "Claude was okay for code.", "true", "coding"])
            + "\n",
            encoding="utf-8",
        )

        written = export_labeling_queue(
            observations,
            output,
            excluded_review_csv_paths=[reviewed],
        )

        self.assertEqual(written, 1)
        with output.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["source_item_id"], "2")


if __name__ == "__main__":
    unittest.main()
