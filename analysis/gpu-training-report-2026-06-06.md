# GPU Training Report

Date: 2026-06-06

## Environment

- GPU: NVIDIA GeForce RTX 3070
- VRAM: 8 GB
- PyTorch: `2.11.0+cu128`
- CUDA available: `true`
- SentenceTransformer device: `cuda:0`

## Training Data

The stricter experiments used:

```text
datasets/training/hn_manual_training_threaded_v4.jsonl
```

- Reviewed rows: `232`
- Hacker News threads: `25`
- Evaluation grouping: `thread_id`

The dataset remains highly imbalanced. In particular:

- `refusal_acceptance`: `1` row
- `bug_regression_report`: `1` row
- `regression_stability`: `5` rows

These classes cannot be evaluated or learned reliably yet.

## Thread-Grouped Frozen Embedding Result

Artifact:

```text
datasets/training/frozen_embedding_bakeoff_thread_grouped_rtx3070.json
```

Eight thread-grouped holdout runs:

| Rank | Backend | Input Mode | Mean Macro F1 |
| ---: | --- | --- | ---: |
| 1 | `BAAI/bge-small-en-v1.5` | evidence only | `0.411` |
| 2 | `BAAI/bge-small-en-v1.5` | augmented | `0.405` |
| 3 | TF-IDF logistic | augmented | `0.303` |
| 4 | TF-IDF logistic | evidence only | `0.281` |

BGE Small evidence-only field macro F1:

- task category: `0.435`
- aspect category: `0.295`
- evidence type: `0.241`
- polarity: `0.343`
- firsthand: `0.738`

## GPU SetFit Pilot

Artifact:

```text
datasets/training/setfit_bge_thread_grouped_rtx3070_20steps.json
```

Configuration:

- checkpoint: `BAAI/bge-small-en-v1.5`
- input mode: evidence only
- grouped by: `thread_id`
- fields: `firsthand_flag`, `polarity_score`
- epochs: `1`
- iterations: `1`
- maximum steps: `20`

Results:

| Field | Frozen BGE Macro F1 | SetFit Macro F1 |
| --- | ---: | ---: |
| firsthand | `0.738` | `0.693` |
| polarity | `0.343` | `0.181` |

The RTX 3070 reduced each 20-step field training run to roughly three seconds, but SetFit did not improve quality.

## Production Candidate

Trained artifact:

```text
datasets/training/bge_small_evidence_only_threaded_v4_model.joblib
```

The artifact uses frozen `BAAI/bge-small-en-v1.5` embeddings with separate logistic heads and persists `text_mode=evidence_only`.

## Decision

- Keep frozen BGE Small evidence-only as the current production candidate.
- Do not promote the SetFit pilot.
- Do not freeze extractor `v1` yet.
- Collect targeted labels before further fine-tuning.

Highest-priority labels:

1. `bug_regression_report`
2. `regression_stability`
3. `refusal_acceptance`
4. `comparative_evaluation`
5. polarity classes `-2` and `2`

The next useful training checkpoint is after at least 500 total reviewed rows and at least 25 examples for every class intended for benchmark scoring.
