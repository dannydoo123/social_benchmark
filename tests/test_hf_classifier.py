import json
import importlib.util
import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.hf_classifier import HFEmbeddingClassifier, evaluate_hf_examples
from social_benchmark.pipeline.embeddings import HuggingFaceTextEmbedder
from social_benchmark.pipeline.model_comparison import compare_classifier_backends


class FakeEmbedder:
    model_name = "fake-local-embeddings"

    def encode(self, texts):
        self.texts = list(texts)
        return [self._vector(text.lower()) for text in texts]

    @staticmethod
    def _vector(text):
        return [
            1.0 if any(term in text for term in ("python", "code", "debug")) else 0.0,
            1.0 if any(term in text for term in ("api", "timeout", "sdk")) else 0.0,
            1.0 if any(term in text for term in ("use", "used", "production")) else 0.0,
            1.0 if any(term in text for term in ("bad", "broken", "frustrating")) else 0.0,
        ]


class HFClassifierTest(unittest.TestCase):
    def test_embedder_accepts_explicit_device(self):
        embedder = HuggingFaceTextEmbedder(model_name="unused", backend="transformers", device="cpu")

        self.assertEqual(embedder.device, "cpu")

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
    def test_trains_and_predicts_with_injected_embedder(self):
        examples = [
            self._example("I use Claude for Python debugging and it works well.", "coding", "satisfaction", "firsthand_usage", 1, True),
            self._example("Claude is useful for refactoring code.", "coding", "satisfaction", "firsthand_usage", 1, True),
            self._example("GPT API timeouts broke our deploy.", "api_developer_workflow", "developer_ergonomics", "integration_failure", -1, True),
            self._example("Gemini SDK latency is frustrating.", "api_developer_workflow", "developer_ergonomics", "integration_failure", -1, False),
        ]

        classifier = HFEmbeddingClassifier(embedder=FakeEmbedder(), model_name=FakeEmbedder.model_name)
        classifier.fit(examples)
        prediction = classifier.predict("Python code debugging was useful")
        metrics = evaluate_hf_examples(examples, model_name=FakeEmbedder.model_name, runs=2, embedder=FakeEmbedder())

        self.assertEqual(prediction["task_category"]["label"], "coding")
        self.assertGreater(prediction["task_category"]["confidence"], 0.5)
        self.assertEqual(metrics["backend"], "hf_embedding")
        self.assertEqual(metrics["runs_requested"], 2)

    def test_compare_classifier_backends_can_skip_hf(self):
        temp_dir = Path(".test_tmp") / f"compare_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        training = temp_dir / "training.jsonl"
        output = temp_dir / "comparison.json"
        examples = [
            self._example("Claude code debugging works.", "coding", "satisfaction", "firsthand_usage", 1, True),
            self._example("GPT API timeout failed.", "api_developer_workflow", "developer_ergonomics", "integration_failure", -1, True),
        ]
        training.write_text("\n".join(json.dumps(item) for item in examples) + "\n", encoding="utf-8")

        comparison = compare_classifier_backends(training, output, runs=1, include_hf=False)

        self.assertIn("naive_bayes", comparison["backends"])
        self.assertIn("sklearn_tfidf_logistic", comparison["backends"])
        self.assertNotIn("hf_embedding_logistic", comparison["backends"])
        self.assertIn("available", comparison["summary"]["sklearn_tfidf_logistic"])
        self.assertTrue(output.exists())

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
    def test_evidence_only_mode_excludes_context_and_metadata(self):
        embedder = FakeEmbedder()
        examples = [
            {
                **self._example("Evidence one", "coding", "satisfaction", "firsthand_usage", 1, True),
                "context_text": "Context one",
                "model_id": "claude",
            },
            {
                **self._example("Evidence two", "general", "value", "hearsay", 0, False),
                "context_text": "Context two",
                "model_id": "gpt",
            },
        ]
        classifier = HFEmbeddingClassifier(embedder=embedder, text_mode="evidence_only")

        classifier.fit(examples, target_fields=("firsthand_flag",))

        self.assertEqual(embedder.texts, ["Evidence one", "Evidence two"])

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
