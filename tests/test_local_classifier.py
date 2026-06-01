import json
import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.local_classifier import (
    LocalNaiveBayesClassifier,
    evaluate_classifier,
    predict_jsonl,
    train_classifier,
)


class LocalClassifierTest(unittest.TestCase):
    def test_trains_predicts_and_evaluates(self):
        examples = [
            {
                "text": "Claude Code is great for debugging Python.",
                "task_category": "coding",
                "aspect_category": "satisfaction",
                "evidence_type": "firsthand_usage",
                "polarity_score": 1,
                "firsthand_flag": True,
            },
            {
                "text": "GPT API timeout failures in production.",
                "task_category": "api_developer_workflow",
                "aspect_category": "developer_ergonomics",
                "evidence_type": "integration_failure",
                "polarity_score": -1,
                "firsthand_flag": True,
            },
            {
                "text": "Release notes mention Gemini improvements.",
                "task_category": "general",
                "aspect_category": "satisfaction",
                "evidence_type": "release_update_reaction",
                "polarity_score": 0,
                "firsthand_flag": False,
            },
        ]
        temp_dir = Path(".test_tmp") / f"classifier_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        training = temp_dir / "training.jsonl"
        model = temp_dir / "model.json"
        predictions = temp_dir / "predictions.jsonl"
        training.write_text("\n".join(json.dumps(item) for item in examples) + "\n", encoding="utf-8")

        count = train_classifier(training, model)
        metrics = evaluate_classifier(training)
        predict_count = predict_jsonl(model, training, predictions)
        classifier = LocalNaiveBayesClassifier.load(model)
        prediction = classifier.predict("Claude Code debugging works well")

        self.assertEqual(count, 3)
        self.assertEqual(metrics["examples"], 3)
        self.assertEqual(predict_count, 3)
        self.assertIn("task_category", prediction)
        self.assertTrue(prediction["task_category"]["label"])

    def test_predict_row_uses_context_and_metadata(self):
        examples = [
            {
                "text": "This seems unsafe.",
                "context_text": "I used Claude Code in production and it created an RCE vulnerability in my app sandbox.",
                "model_id": "claude",
                "product_id": "claude-code",
                "provider_id": "anthropic",
                "task_category": "coding",
                "aspect_category": "trust_reliability",
                "evidence_type": "firsthand_usage",
                "polarity_score": -2,
                "firsthand_flag": True,
            },
            {
                "text": "This was worth it.",
                "context_text": "I used DeepSeek through OpenRouter and the price was cheap enough to be worth it.",
                "model_id": "deepseek",
                "product_id": "openrouter",
                "provider_id": "deepseek",
                "task_category": "general",
                "aspect_category": "value",
                "evidence_type": "pricing_value_comment",
                "polarity_score": 1,
                "firsthand_flag": True,
            },
        ]
        classifier = LocalNaiveBayesClassifier()
        classifier.fit(examples)

        prediction = classifier.predict_row(
            {
                "text": "Seems unsafe.",
                "context_text": "I used Claude Code and it bypassed the sandbox with shell exec.",
                "model_id": "claude",
                "product_id": "claude-code",
                "provider_id": "anthropic",
            }
        )

        self.assertEqual(prediction["aspect_category"]["label"], "trust_reliability")
        self.assertEqual(prediction["firsthand_flag"]["label"], "true")


if __name__ == "__main__":
    unittest.main()
