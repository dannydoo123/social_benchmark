import unittest

from social_benchmark.pipeline.classifier_experiments import grouped_holdout_indexes
from social_benchmark.pipeline.soft_chain_classifier import _evaluate_chain, _probability_vector, run_soft_chain_bakeoff
from social_benchmark.pipeline.structured_classifier import FlatClassifier


class FakeEmbedder:
    def encode(self, texts):
        return [[float("code" in text.lower()), float("bad" in text.lower())] for text in texts]


class SoftChainClassifierTest(unittest.TestCase):
    def test_probability_vector_uses_stable_rubric_label_order(self):
        model = FlatClassifier()
        model.fit([[0.0], [1.0]], ["false", "true"])

        probabilities = _probability_vector(model, [1.0], "firsthand_flag")

        self.assertEqual(len(probabilities), 2)
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[1], probabilities[0])

    def test_bakeoff_uses_grouped_oof_chain_features(self):
        config = {
            field: {"embedding_model": "fake", "backend": "auto", "rubric_scale": 0.0, "strategy": "flat"}
            for field in ("task_category", "aspect_category", "evidence_type", "polarity_score", "firsthand_flag")
        }
        config["polarity_score"]["strategy"] = "ordinal"
        rows = []
        for group in range(8):
            positive = group % 2 == 0
            rows.append(
                {
                    "text": "code works" if positive else "general is bad",
                    "thread_id": f"thread-{group}",
                    "task_category": "coding" if positive else "general",
                    "aspect_category": "task_fit" if positive else "satisfaction",
                    "evidence_type": "firsthand_usage" if positive else "hearsay",
                    "polarity_score": "1" if positive else "-1",
                    "firsthand_flag": "true" if positive else "false",
                }
            )
        features = {
            field: FakeEmbedder().encode([row["text"] for row in rows])
            for field in config
        }
        fields = _evaluate_chain(
            rows,
            features_by_field=features,
            splits=grouped_holdout_indexes(rows, runs=2, group_field="thread_id"),
            order=("firsthand_flag", "evidence_type", "aspect_category", "task_category", "polarity_score"),
            config=config,
            group_field="thread_id",
            inner_runs=2,
        )

        self.assertEqual(fields["task_category"]["evaluated"], 4)
        self.assertGreater(fields["task_category"]["macro_f1"], 0.0)

    def test_rejects_chain_with_ordinal_polarity_before_downstream_fields(self):
        with self.assertRaises(ValueError):
            run_soft_chain_bakeoff(
                "missing.jsonl",
                "unused.json",
                chain_orders=(("polarity_score", "firsthand_flag", "evidence_type", "aspect_category", "task_category"),),
            )


if __name__ == "__main__":
    unittest.main()
