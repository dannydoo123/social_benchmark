# Round-Two Review Processing Report

Date: 2026-06-07

## Input

- Reviewed queue: `datasets/queues/targeted_training_review_round2_2026-06-06_filled.csv`
- Rows reviewed: 300
- Rows accepted for training: 194
- Rows excluded: 106
- Accepted threads: 29

The reviewed batch passed label validation. It has no overlap with the five
held-out gold evaluation threads.

## Applied Training Data

- Converted batch: `datasets/training/targeted_training_review_round2_2026-06-06.jsonl`
- Merged training set: `datasets/training/hn_manual_training_threaded_round2_2026-06-07_merged.jsonl`
- Merged observations: 547
- Merged threads: 69
- Held-out gold thread overlap: 0

Reviewed blank identity values are preserved. This matters for product-level
evidence such as Claude Code or NotebookLM where the exact underlying model is
not stated.

## Strict Thread-Grouped Evaluation

Result: `datasets/training/frozen_embedding_bakeoff_thread_grouped_round2_2026-06-07.json`

Best configuration remains `BAAI/bge-small-en-v1.5` with `evidence_only` text.

| Metric | Previous | Round two | Change |
|---|---:|---:|---:|
| Mean macro F1 | 0.3961 | 0.3954 | -0.0007 |
| Task | 0.3337 | 0.3074 | -0.0264 |
| Aspect | 0.2876 | 0.2998 | +0.0122 |
| Evidence | 0.3786 | 0.3675 | -0.0111 |
| Polarity | 0.2942 | 0.2882 | -0.0060 |
| Firsthand | 0.6864 | 0.7143 | +0.0278 |

The result is effectively flat overall, with meaningful improvement in aspect
and firsthand classification offset by weaker task, evidence, and polarity
classification.

## Candidate Decision

- Candidate model: `datasets/training/bge_small_evidence_only_round2_2026-06-07_model.joblib`
- Production model remains: `datasets/training/bge_small_evidence_only_2026-06-06_merged_model.joblib`

Do not promote the round-two candidate yet because its strict thread-grouped
mean macro F1 did not exceed the existing model. Do not repeatedly evaluate
against the isolated gold set while iterating.

## Recommended Next Training Review

Target the weak and confused labels rather than collecting a broad random
batch:

- Increase clearly distinguishable examples for `task_category`, especially
  `long_context`, `data_analysis`, `multimodal`, and `research`.
- Add difficult contrasts between `firsthand_usage`,
  `comparative_evaluation`, `benchmark_anecdote`, and
  `bug_regression_report`.
- Add balanced polarity examples, especially neutral and strong positive or
  negative evidence with explicit wording.
- Preserve thread grouping and keep all gold threads excluded from training.
