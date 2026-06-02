import importlib.util
import json
import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.local_classifier import load_classifier
from social_benchmark.pipeline.sklearn_classifier import evaluate_sklearn_examples, train_sklearn_classifier


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
class SklearnClassifierTest(unittest.TestCase):
    def test_trains_loads_and_evaluates_field_models(self):
        examples = [
            self._example("I use Claude for Python debugging and it works well.", "coding", "satisfaction", "firsthand_usage", 1, True),
            self._example("Claude is useful for refactoring code.", "coding", "satisfaction", "firsthand_usage", 1, True),
            self._example("GPT API timeouts broke our deploy.", "api_developer_workflow", "developer_ergonomics", "integration_failure", -1, True),
            self._example("Gemini SDK latency is frustrating.", "api_developer_workflow", "developer_ergonomics", "integration_failure", -1, False),
        ]
        temp_dir = Path(".test_tmp") / f"sklearn_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        training = temp_dir / "training.jsonl"
        model = temp_dir / "model.joblib"
        training.write_text("\n".join(json.dumps(item) for item in examples) + "\n", encoding="utf-8")

        count = train_sklearn_classifier(training, model, runs=2)
        classifier = load_classifier(model)
        metrics = evaluate_sklearn_examples(examples, runs=2)
        prediction = classifier.predict("Python code debugging was useful")

        self.assertEqual(count, 4)
        self.assertEqual(metrics["runs_requested"], 2)
        self.assertEqual(prediction["task_category"]["label"], "coding")
        self.assertGreater(prediction["task_category"]["confidence"], 0.5)

    @staticmethod
    def _example(text, task, aspect, evidence, polarity, firsthand):
        return {
            "text": text,
            "task_category": task,
            "aspect_category": aspect,
            "evidence_type": evidence,
            "polarity_score": polarity,
            "firsthand_flag": firsthand,
        }


if __name__ == "__main__":
    unittest.main()
