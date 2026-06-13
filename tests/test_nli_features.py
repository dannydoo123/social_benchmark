import unittest
from unittest import mock

from social_benchmark.pipeline import nli_features
from social_benchmark.pipeline.routed_classifier import DEFAULT_FIELD_CONFIG, _build_features_by_field, _gated_metrics


class FakeScorer:
    def __init__(self, model_name="fake", hypotheses=None, **_kwargs) -> None:
        self.model_name = model_name
        self.hypotheses = list(hypotheses or [])

    def encode(self, texts):
        return [[0.5] * len(self.hypotheses) for _ in texts]


class FakeEmbedder:
    def __init__(self, model_name="fake-embedder", backend="auto") -> None:
        self.model_name = model_name

    def encode(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class NliFeatureTest(unittest.TestCase):
    def test_nli_features_append_scaled_scores(self):
        config = {
            field: dict(settings, nli_model="fake-nli", nli_scale=2.0)
            if field == "aspect_category"
            else dict(settings)
            for field, settings in DEFAULT_FIELD_CONFIG.items()
        }
        with mock.patch.object(nli_features, "NliRubricScorer", FakeScorer), mock.patch(
            "social_benchmark.pipeline.routed_classifier.HuggingFaceTextEmbedder",
            FakeEmbedder,
        ):
            features = _build_features_by_field(["some text"], config, cache=None)

        aspect_row = features["aspect_category"][0]
        evidence_row = features["evidence_type"][0]
        # aspect gains 8 NLI dimensions (one per aspect label), each 0.5 * 2.0
        self.assertEqual(len(aspect_row), len(evidence_row) + 8)
        self.assertEqual(aspect_row[-8:], [1.0] * 8)

    def test_gated_metrics_coverage_and_precision(self):
        outcomes = [("a", "a", 0.9), ("a", "b", 0.4), ("b", "b", 0.7)]
        metrics = _gated_metrics(outcomes, 0.6)
        self.assertAlmostEqual(metrics["coverage"], 2 / 3)
        self.assertEqual(metrics["covered"], 2)
        self.assertEqual(metrics["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
