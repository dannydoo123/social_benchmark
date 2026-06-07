import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from social_benchmark.pipeline.setfit_experiments import parse_checkpoint_specs, run_setfit_bakeoff


class SetFitExperimentTest(unittest.TestCase):
    def test_parses_checkpoint_specs(self):
        self.assertEqual(
            parse_checkpoint_specs(["BAAI/bge-small-en-v1.5|augmented"]),
            (("BAAI/bge-small-en-v1.5", "augmented"),),
        )

    def test_rejects_invalid_checkpoint_specs(self):
        with self.assertRaises(ValueError):
            parse_checkpoint_specs(["BAAI/bge-small-en-v1.5"])

    @patch("social_benchmark.pipeline.setfit_experiments._train_evaluate_field")
    def test_bakeoff_can_group_by_thread(self, train_evaluate):
        train_evaluate.return_value = {"macro_f1": 0.5}
        temp_dir = Path(".test_tmp") / f"setfit_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        training = temp_dir / "training.jsonl"
        output = temp_dir / "result.json"
        training.write_text(
            "\n".join(
                [
                    '{"text":"a","thread_id":"thread-a","source_item_id":"1"}',
                    '{"text":"b","thread_id":"thread-a","source_item_id":"2"}',
                    '{"text":"c","thread_id":"thread-b","source_item_id":"3"}',
                    '{"text":"d","thread_id":"thread-b","source_item_id":"4"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = run_setfit_bakeoff(
            training,
            output,
            checkpoints=(("test/checkpoint", "evidence_only"),),
            fields=("firsthand_flag",),
            group_field="thread_id",
        )

        self.assertEqual(result["evaluation"]["group_field"], "thread_id")


if __name__ == "__main__":
    unittest.main()
