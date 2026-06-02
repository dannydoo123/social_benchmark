import unittest

from social_benchmark.pipeline.setfit_experiments import parse_checkpoint_specs


class SetFitExperimentTest(unittest.TestCase):
    def test_parses_checkpoint_specs(self):
        self.assertEqual(
            parse_checkpoint_specs(["BAAI/bge-small-en-v1.5|augmented"]),
            (("BAAI/bge-small-en-v1.5", "augmented"),),
        )

    def test_rejects_invalid_checkpoint_specs(self):
        with self.assertRaises(ValueError):
            parse_checkpoint_specs(["BAAI/bge-small-en-v1.5"])


if __name__ == "__main__":
    unittest.main()
