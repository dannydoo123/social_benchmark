import unittest

from social_benchmark.pipeline.constraint_resolver import resolve_prediction, resolve_prediction_labels


class ConstraintResolverTest(unittest.TestCase):
    def test_enforces_firsthand_consistency(self):
        resolved = resolve_prediction_labels(
            {"evidence_type": "firsthand_usage", "firsthand_flag": "false"},
            enabled_rules=("firsthand_consistency",),
        )

        self.assertEqual(resolved["firsthand_flag"], "true")

    def test_high_purity_integration_shape(self):
        resolved = resolve_prediction_labels(
            {
                "evidence_type": "integration_failure",
                "aspect_category": "task_fit",
                "task_category": "coding",
            },
            enabled_rules=("integration_shape",),
        )

        self.assertEqual(resolved["aspect_category"], "developer_ergonomics")
        self.assertEqual(resolved["task_category"], "api_developer_workflow")

    def test_marks_changed_prediction(self):
        resolved = resolve_prediction(
            {
                "evidence_type": {"label": "hearsay", "confidence": 0.8},
                "firsthand_flag": {"label": "true", "confidence": 0.7},
            },
            enabled_rules=("firsthand_consistency",),
        )

        self.assertEqual(resolved["firsthand_flag"]["label"], "false")
        self.assertTrue(resolved["firsthand_flag"]["constraint_applied"])


if __name__ == "__main__":
    unittest.main()
