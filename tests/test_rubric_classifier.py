import unittest

from social_benchmark.pipeline.rubric_classifier import _rubric_features


class RubricClassifierTest(unittest.TestCase):
    def test_appends_scaled_rubric_similarities(self):
        features = _rubric_features([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], scale=3.0)

        self.assertEqual(features[:2], [1.0, 0.0])
        self.assertAlmostEqual(features[2], 3.0)
        self.assertAlmostEqual(features[3], 0.0)


if __name__ == "__main__":
    unittest.main()
