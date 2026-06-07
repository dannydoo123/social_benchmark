import unittest

from social_benchmark.pipeline.routed_classifier import RoutedRubricClassifier


class FakeEmbedder:
    def encode(self, texts):
        return [[float("code" in text.lower()), float("bad" in text.lower())] for text in texts]


class RoutedClassifierTest(unittest.TestCase):
    def test_fits_and_predicts_with_specialized_embedders(self):
        config = {
            "task_category": {"embedding_model": "fake", "backend": "auto", "rubric_scale": 1.0, "strategy": "flat"},
            "aspect_category": {"embedding_model": "fake", "backend": "auto", "rubric_scale": 1.0, "strategy": "flat"},
            "evidence_type": {"embedding_model": "fake", "backend": "auto", "rubric_scale": 1.0, "strategy": "flat"},
            "polarity_score": {"embedding_model": "fake", "backend": "auto", "rubric_scale": 1.0, "strategy": "ordinal"},
            "firsthand_flag": {"embedding_model": "fake", "backend": "auto", "rubric_scale": 1.0, "strategy": "flat"},
        }
        examples = [
            self._example("code works", "coding", "task_fit", "firsthand_usage", "1", "true"),
            self._example("code is bad", "coding", "task_fit", "firsthand_usage", "-1", "true"),
            self._example("general works", "general", "satisfaction", "hearsay", "1", "false"),
            self._example("general is bad", "general", "satisfaction", "hearsay", "-1", "false"),
        ]
        classifier = RoutedRubricClassifier(field_config=config, embedders={"fake|auto": FakeEmbedder()})

        classifier.fit(examples)
        prediction = classifier.predict_row({"text": "code works"})

        self.assertEqual(prediction["task_category"]["label"], "coding")
        self.assertGreater(prediction["task_category"]["confidence"], 0.0)
        self.assertEqual(set(prediction), set(config))

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
