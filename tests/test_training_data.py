import csv
import json
import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.training_data import build_training_jsonl, merge_training_jsonl


class TrainingDataTest(unittest.TestCase):
    def test_builds_jsonl_from_human_labels(self):
        temp_dir = Path(".test_tmp") / f"training_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        labels = temp_dir / "labels.csv"
        output = temp_dir / "training.jsonl"
        with labels.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "evidence_text",
                    "model_id",
                    "product_id",
                    "inference_profile",
                    "task_category",
                    "aspect_category",
                    "evidence_type",
                    "polarity_score",
                    "firsthand_flag",
                    "source_platform",
                    "thread_id",
                    "source_item_id",
                    "url",
                    "human_polarity_score",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "evidence_text": "Claude Code with Opus 4.8 in ultracode mode was great for debugging.",
                    "model_id": "claude-opus-4.8",
                    "product_id": "claude-code",
                    "inference_profile": "ultracode",
                    "task_category": "coding",
                    "aspect_category": "satisfaction",
                    "evidence_type": "hearsay",
                    "polarity_score": "0",
                    "firsthand_flag": "false",
                    "source_platform": "hacker_news",
                    "thread_id": "thread-1",
                    "source_item_id": "1",
                    "url": "https://news.ycombinator.com/item?id=1",
                    "human_polarity_score": "1",
                }
            )

        written = build_training_jsonl(labels, output)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(written, 1)
        self.assertEqual(rows[0]["polarity_score"], 1)
        self.assertEqual(rows[0]["model_id"], "claude-opus-4.8")
        self.assertEqual(rows[0]["product_id"], "claude-code")
        self.assertEqual(rows[0]["inference_profile"], "ultracode")
        self.assertEqual(rows[0]["thread_id"], "thread-1")

    def test_skips_excluded_rows(self):
        temp_dir = Path(".test_tmp") / f"training_skip_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        labels = temp_dir / "labels.csv"
        output = temp_dir / "training.jsonl"
        with labels.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "evidence_text",
                    "model_id",
                    "task_category",
                    "aspect_category",
                    "evidence_type",
                    "polarity_score",
                    "firsthand_flag",
                    "source_platform",
                    "source_item_id",
                    "url",
                    "human_model_id",
                    "human_excluded_from_scoring",
                    "human_exclusion_reason",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "evidence_text": "This is a release note, not a quality signal.",
                    "model_id": "claude-opus-4.8",
                    "task_category": "general",
                    "aspect_category": "satisfaction",
                    "evidence_type": "release_update_reaction",
                    "polarity_score": "0",
                    "firsthand_flag": "false",
                    "source_platform": "hacker_news",
                    "source_item_id": "1",
                    "url": "https://news.ycombinator.com/item?id=1",
                    "human_model_id": "claude-opus-4.8",
                    "human_excluded_from_scoring": "true",
                    "human_exclusion_reason": "factual_release_or_adoption",
                }
            )

        written = build_training_jsonl(labels, output)

        self.assertEqual(written, 0)
        self.assertFalse(output.read_text(encoding="utf-8").strip())

    def test_includes_context_text_when_sidecar_is_provided(self):
        temp_dir = Path(".test_tmp") / f"training_context_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        labels = temp_dir / "labels.csv"
        context = temp_dir / "context.jsonl"
        output = temp_dir / "training.jsonl"
        with labels.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "review_id",
                    "evidence_text",
                    "model_id",
                    "task_category",
                    "aspect_category",
                    "evidence_type",
                    "polarity_score",
                    "firsthand_flag",
                    "source_platform",
                    "source_item_id",
                    "url",
                    "human_model_id",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "review_id": "review-1",
                    "evidence_text": "Claude was unsafe here.",
                    "model_id": "claude",
                    "task_category": "coding",
                    "aspect_category": "trust_reliability",
                    "evidence_type": "hearsay",
                    "polarity_score": "-1",
                    "firsthand_flag": "false",
                    "source_platform": "hacker_news",
                    "source_item_id": "1",
                    "url": "https://news.ycombinator.com/item?id=1",
                    "human_model_id": "claude",
                }
            )
        context.write_text(
            json.dumps({"review_id": "review-1", "raw_full_text": "I used Claude in production and it felt unsafe."}) + "\n",
            encoding="utf-8",
        )

        written = build_training_jsonl(labels, output, context_jsonl=context)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(written, 1)
        self.assertEqual(rows[0]["context_text"], "I used Claude in production and it felt unsafe.")

    def test_accepts_utf8_bom_context_sidecar(self):
        temp_dir = Path(".test_tmp") / f"training_context_bom_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        labels = temp_dir / "labels.csv"
        context = temp_dir / "context.jsonl"
        output = temp_dir / "training.jsonl"
        labels.write_text(
            "review_id,evidence_text,model_id,task_category,aspect_category,evidence_type,polarity_score,firsthand_flag,human_model_id\n"
            "review-1,Claude worked,claude,coding,task_fit,firsthand_usage,1,true,claude\n",
            encoding="utf-8",
        )
        context.write_text(
            json.dumps({"review_id": "review-1", "raw_full_text": "Full context"}) + "\n",
            encoding="utf-8-sig",
        )

        written = build_training_jsonl(labels, output, context_jsonl=context)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(written, 1)
        self.assertEqual(rows[0]["context_text"], "Full context")

    def test_preserves_intentionally_blank_reviewed_model(self):
        temp_dir = Path(".test_tmp") / f"training_blank_model_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        labels = temp_dir / "labels.csv"
        output = temp_dir / "training.jsonl"
        labels.write_text(
            "reviewed_flag,evidence_text,model_id,product_id,task_category,aspect_category,evidence_type,polarity_score,firsthand_flag,human_model_id,human_product_id,human_task_category\n"
            "true,Claude Code worked,claude,claude-code,coding,task_fit,firsthand_usage,1,true,,claude-code,coding\n",
            encoding="utf-8",
        )

        written = build_training_jsonl(labels, output)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(written, 1)
        self.assertEqual(rows[0]["model_id"], "")
        self.assertEqual(rows[0]["product_id"], "claude-code")

    def test_merges_deduplicates_and_excludes_threads(self):
        temp_dir = Path(".test_tmp") / f"training_merge_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        first = temp_dir / "first.jsonl"
        second = temp_dir / "second.jsonl"
        output = temp_dir / "merged.jsonl"
        first.write_text(
            json.dumps({"source_item_id": "1", "thread_id": "train", "model_id": "claude", "text": "same", "polarity_score": 0})
            + "\n"
            + json.dumps({"source_item_id": "2", "thread_id": "gold", "model_id": "gpt", "text": "held out"})
            + "\n",
            encoding="utf-8",
        )
        second.write_text(
            json.dumps({"source_item_id": "1", "thread_id": "train", "model_id": "claude", "text": "same", "polarity_score": 1})
            + "\n",
            encoding="utf-8",
        )

        written = merge_training_jsonl([first, second], output, excluded_thread_ids={"gold"})
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(written, 1)
        self.assertEqual(rows[0]["polarity_score"], 1)


if __name__ == "__main__":
    unittest.main()
