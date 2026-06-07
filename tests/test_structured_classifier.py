import importlib.util
import unittest

from social_benchmark.pipeline.structured_classifier import (
    HierarchicalClassifier,
    OrdinalPolarityClassifier,
)


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
class StructuredClassifierTest(unittest.TestCase):
    def test_hierarchical_classifier_routes_then_classifies(self):
        vectors = [[-2.0], [-1.0], [1.0], [2.0]]
        labels = ["general", "general", "coding", "api_developer_workflow"]
        classifier = HierarchicalClassifier(
            {
                "general": "general",
                "coding": "technical",
                "api_developer_workflow": "technical",
            }
        )

        classifier.fit(vectors, labels)

        self.assertEqual(classifier.predict([-1.5]), "general")
        self.assertIn(classifier.predict([1.5]), {"coding", "api_developer_workflow"})

    def test_ordinal_polarity_preserves_order(self):
        vectors = [[-2.0], [-1.0], [0.0], [1.0], [2.0]] * 3
        labels = ["-2", "-1", "0", "1", "2"] * 3
        classifier = OrdinalPolarityClassifier()

        classifier.fit(vectors, labels)
        predictions = [int(classifier.predict(vector)) for vector in [[-2.0], [0.0], [2.0]]]

        self.assertLessEqual(predictions[0], predictions[1])
        self.assertLessEqual(predictions[1], predictions[2])


if __name__ == "__main__":
    unittest.main()
