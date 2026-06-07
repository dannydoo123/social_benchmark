# Targeted Review Processing Report

Date: 2026-06-06

## Reviewed Batch

Input:

```text
datasets/queues/targeted_training_review_2026-06-06_filled.csv
```

- Reviewed rows: `240`
- Included for training: `171`
- Excluded from scoring: `69`
- Invalid required labels: `0`
- Duplicate review IDs: `0`

Exclusion reasons:

- not about model quality: `32`
- too vague or speculative: `18`
- factual release or adoption: `11`
- off topic: `5`
- duplicate or low value: `3`

## Merged Training Data

New reviewed training artifact:

```text
datasets/training/targeted_training_review_2026-06-06.jsonl
```

Merged and deduplicated artifact:

```text
datasets/training/hn_manual_training_threaded_2026-06-06_merged.jsonl
```

- Unique reviewed observations: `353`
- Threads: `44`
- Gold evaluation thread overlap: `0`

Important remaining class shortages:

- refusal acceptance: `1`
- integration failure: `7`
- hallucination safety: `10`
- bug regression report: `12`
- regression stability: `13`

## Thread-Grouped Evaluation

Artifact:

```text
datasets/training/frozen_embedding_bakeoff_thread_grouped_2026-06-06_merged.json
```

Best configuration remains:

```text
BAAI/bge-small-en-v1.5
input mode: evidence only
mean macro F1: 0.396
```

Compared with the earlier 232-row evaluation:

| Field | Earlier Macro F1 | Merged Macro F1 | Change |
| --- | ---: | ---: | ---: |
| task category | `0.435` | `0.334` | `-0.101` |
| aspect category | `0.295` | `0.288` | `-0.007` |
| evidence type | `0.241` | `0.379` | `+0.138` |
| polarity | `0.343` | `0.294` | `-0.049` |
| firsthand | `0.738` | `0.686` | `-0.052` |

The expanded evaluation covers more diverse threads and corrected labels, so the earlier and merged scores are not directly comparable as a pure training-data ablation. Evidence-type performance improved substantially. Task, polarity, and firsthand remain the next active-learning priorities.

## Updated Production Candidate

```text
datasets/training/bge_small_evidence_only_2026-06-06_merged_model.joblib
```

The gold evaluation CSV remains untouched and must not be used for training:

```text
datasets/queues/gold_eval_review_2026-06-06.csv
```
