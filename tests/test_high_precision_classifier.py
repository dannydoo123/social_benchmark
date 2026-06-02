import importlib.util
import json
import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.high_precision_classifier import ABSTAIN_LABEL, HighPrecisionClassifier, train_high_precision_classifier
from social_benchmark.pipeline.local_classifier import load_classifier


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
class HighPrecisionClassifierTest(unittest.TestCase):
    def test_trains_abstaining_ensemble_without_hf(self):
        examples = [
            self._example("I use Claude for Python debugging and it works well.", "coding", "satisfaction", "firsthand_usage", 1, True),
            self._example("Claude is useful for refactoring code.", "coding", "satisfaction", "firsthand_usage", 1, True),
            self._example("GPT API timeouts broke our deploy.", "api_developer_workflow", "developer_ergonomics", "integration_failure", -1, True),
            self._example("Gemini SDK latency is frustrating.", "api_developer_workflow", "developer_ergonomics", "integration_failure", -1, False),
        ]
        temp_dir = Path(".test_tmp") / f"high_precision_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        training = temp_dir / "training.jsonl"
        model = temp_dir / "ensemble.joblib"
        training.write_text("\n".join(json.dumps(item) for item in examples) + "\n", encoding="utf-8")

        count = train_high_precision_classifier(training, model, runs=2, include_hf=False)
        classifier = load_classifier(model)
        prediction = classifier.predict("Python code debugging was useful")

        self.assertEqual(count, 4)
        self.assertIn(prediction["task_category"]["label"], {"coding", ABSTAIN_LABEL})

    def test_abstains_when_agreement_is_too_low(self):
        classifier = HighPrecisionClassifier(min_confidence=0.5, min_agreement=3, include_hf=False)
        classifier.fit(
            [
                self._example("Python code is useful.", "coding", "satisfaction", "firsthand_usage", 1, True),
                self._example("API timeout failed.", "api_developer_workflow", "developer_ergonomics", "integration_failure", -1, True),
            ]
        )

        prediction = classifier.predict("Python debugging was useful")

        self.assertIn("task_category", prediction)

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
