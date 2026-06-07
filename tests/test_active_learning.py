import json
import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.active_learning import write_fixed_model_evaluation, write_threshold_report
from social_benchmark.pipeline.local_classifier import train_classifier


class ActiveLearningTest(unittest.TestCase):
    def test_writes_threshold_report(self):
        examples = [
            self._example("Python debugging worked.", "coding", True),
            self._example("API timeout failed.", "api_developer_workflow", False),
        ]
        temp_dir = Path(".test_tmp") / f"threshold_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        training = temp_dir / "training.jsonl"
        model = temp_dir / "model.json"
        report = temp_dir / "report.json"
        training.write_text("\n".join(json.dumps(row) for row in examples) + "\n", encoding="utf-8")
        train_classifier(training, model)

        result = write_threshold_report(training, model, report, thresholds=(0.0,))

        self.assertEqual(result["examples"], 2)
        self.assertIn("task_category", result["thresholds"]["0.00"])

    def test_evaluates_fixed_model_without_retraining(self):
        examples = [
            self._example("Python debugging worked.", "coding", True),
            self._example("API timeout failed.", "api_developer_workflow", False),
        ]
        temp_dir = Path(".test_tmp") / f"fixed_eval_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        training = temp_dir / "training.jsonl"
        model = temp_dir / "model.json"
        report = temp_dir / "report.json"
        training.write_text("\n".join(json.dumps(row) for row in examples) + "\n", encoding="utf-8")
        train_classifier(training, model)

        result = write_fixed_model_evaluation(training, model, report)

        self.assertEqual(result["examples"], 2)
        self.assertIn("macro_f1", result["fields"]["task_category"])
        self.assertTrue(report.exists())

    @staticmethod
    def _example(text, task, firsthand):
        return {
            "text": text,
            "task_category": task,
            "aspect_category": "satisfaction",
            "evidence_type": "firsthand_usage",
            "polarity_score": 1,
            "firsthand_flag": firsthand,
        }


if __name__ == "__main__":
    unittest.main()
