import unittest

from social_benchmark.pipeline.publication_readiness import assess_publication_readiness


class PublicationReadinessTest(unittest.TestCase):
    def test_rejects_weak_or_small_evaluation(self):
        result = {
            "examples": 100,
            "evaluation": {"groups": 20},
            "fields": {"task_category": {"macro_f1": 0.5}},
        }

        assessment = assess_publication_readiness(result)

        self.assertFalse(assessment["ready"])
        self.assertIn("examples<300", assessment["failures"])
        self.assertIn("task_category<0.6", assessment["failures"])

    def test_accepts_result_that_meets_gates(self):
        result = {
            "examples": 500,
            "evaluation": {"groups": 100},
            "fields": {
                "task_category": {"macro_f1": 0.72},
                "aspect_category": {"macro_f1": 0.70},
            },
        }

        self.assertTrue(assess_publication_readiness(result)["ready"])


if __name__ == "__main__":
    unittest.main()
