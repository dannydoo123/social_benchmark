import csv
import json
import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.auto_review import auto_review_labels
from social_benchmark.pipeline.labeling import LABEL_FIELDS
from social_benchmark.pipeline.local_classifier import LocalNaiveBayesClassifier


class AutoReviewTest(unittest.TestCase):
    def test_accepts_only_stable_model_extractor_agreement(self):
        temp_dir = Path(".test_tmp") / f"auto_review_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        model = temp_dir / "model.json"
        labels = temp_dir / "queue.csv"
        output = temp_dir / "reviewed.csv"
        report_path = temp_dir / "report.json"

        examples = [
            {
                "text": "I used Claude Code for Python debugging and it was useful in production.",
                "provider_id": "anthropic",
                "model_id": "claude",
                "source_platform": "reddit",
                "task_category": "coding",
                "aspect_category": "task_fit",
                "evidence_type": "firsthand_usage",
                "polarity_score": 1,
                "firsthand_flag": True,
            },
            {
                "text": "Someone said ChatGPT hallucinated citations during research and was unreliable.",
                "provider_id": "openai",
                "model_id": "gpt",
                "source_platform": "reddit",
                "task_category": "research",
                "aspect_category": "hallucination_safety",
                "evidence_type": "hearsay",
                "polarity_score": -1,
                "firsthand_flag": False,
            },
        ]
        classifier = LocalNaiveBayesClassifier()
        classifier.fit(examples)
        classifier.save(model)

        with labels.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "review_id": "accept-1",
                    "source_platform": "reddit",
                    "source_item_id": "r1",
                    "thread_id": "t1",
                    "url": "https://reddit.com/r/test/comments/1",
                    "provider_id": "anthropic",
                    "model_id": "claude",
                    "task_category": "coding",
                    "aspect_category": "task_fit",
                    "evidence_type": "firsthand_usage",
                    "claim_type": "praise",
                    "polarity_score": "1",
                    "extractor_confidence": "0.93",
                    "firsthand_flag": "true",
                    "evidence_text": "I used Claude Code for Python debugging and it was useful in production.",
                }
            )
            writer.writerow(
                {
                    "review_id": "exclude-1",
                    "source_platform": "reddit",
                    "source_item_id": "r2",
                    "thread_id": "t2",
                    "url": "https://reddit.com/r/test/comments/2",
                    "provider_id": "openai",
                    "model_id": "gpt",
                    "task_category": "research",
                    "aspect_category": "hallucination_safety",
                    "evidence_type": "hearsay",
                    "claim_type": "neutral",
                    "polarity_score": "-1",
                    "extractor_confidence": "0.95",
                    "firsthand_flag": "false",
                    "evidence_text": "Someone said ChatGPT hallucinated citations during research and was unreliable.",
                }
            )

        report = auto_review_labels(
            labels_csv=labels,
            output_csv=output,
            report_path=report_path,
            model_path=model,
            max_iterations=3,
        )
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertTrue(report["stable"])
        self.assertEqual(report["iterations_run"], 2)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["excluded"], 1)
        self.assertEqual(rows[0]["reviewed_flag"], "true")
        self.assertEqual(rows[0]["human_task_category"], "coding")
        self.assertEqual(rows[0]["human_polarity_score"], "1")
        self.assertEqual(rows[1]["human_excluded_from_scoring"], "true")
        self.assertEqual(rows[1]["human_exclusion_reason"], "neutral_claim")
        self.assertTrue(json.loads(report_path.read_text(encoding="utf-8"))["stable"])


if __name__ == "__main__":
    unittest.main()
