import json
import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.analysis import observation_report


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


if __name__ == "__main__":
    unittest.main()

