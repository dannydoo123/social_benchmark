# Dataset Manifest

| Artifact | Purpose | Approximate rows |
|---|---|---:|
| `training/hn_manual_training.jsonl` | Combined compact training set | 324 |
| `reviewed/label_queue_hn_all_combined_assisted_v4_human_reviewed.csv` | Primary human-reviewed batch | 300 |
| `reviewed/label_queue_hn_all_combined_assisted_v5_human_reviewed.csv` | Follow-up human-reviewed batch | 76 |
| `reviewed/label_queue_hn_all_combined_assisted_v6_ai_reviewed.csv` | AI-reviewed rows requiring audit | 10 |
| `queues/label_queue_hn_expanded_2026-06-01_v3.csv` | Latest queue for the next review batch | 300 |
| `queues/label_queue_hn_expanded_2026-06-01_v3_context.jsonl` | Full context sidecar for the latest queue | 300 |
| `evaluation/hn_sklearn_eval_v1.json` | Context-heavy logistic baseline | 324 |
| `evaluation/hn_sklearn_eval_v2.json` | Evidence-only logistic baseline | 324 |
| `evaluation/hn_sklearn_variants_v1.json` | Feature representation comparison | 324 |

## Excluded Artifacts

- `data/raw/`: official API snapshots, currently about 43 MB.
- `data/processed/`: reproducible observations, scores, and superseded queues.
- `data/training/*.joblib`: reproducible local classifier binaries.
- `data/training/hn_local_nb_model*.json`: superseded Naive Bayes binaries.
