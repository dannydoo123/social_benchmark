# Portable Training Dataset

This directory contains the compact, Git-tracked artifacts needed to continue
classifier development on another machine.

## Scope

- Hacker News only.
- Reviewed labels and compact training JSONL are included.
- The latest fresh review queue and context sidecar are included.
- Evaluation reports are included for cross-machine comparisons.
- Raw corpus snapshots, generated observations, scores, and trained model
  binaries remain excluded from Git because they are reproducible and bulky.

## Privacy And Use

The queue context sidecar contains Hacker News text for review. Keep this
repository private unless the context sidecar is removed. Preserve source URLs
and source item IDs when using reviewed evidence.

## Contents

```text
reviewed/
  label_queue_hn_all_combined_assisted_v4_human_reviewed.csv
  label_queue_hn_all_combined_assisted_v5_human_reviewed.csv
  label_queue_hn_all_combined_assisted_v6_ai_reviewed.csv
training/
  hn_manual_training.jsonl
evaluation/
  hn_sklearn_eval_v1.json
  hn_sklearn_eval_v2.json
  hn_sklearn_variants_v1.json
queues/
  label_queue_hn_expanded_2026-06-01_v3.csv
  label_queue_hn_expanded_2026-06-01_v3_context.jsonl
```

The `v6_ai_reviewed` rows are intentionally separate from human-reviewed rows
so they can be audited before being trusted as ground truth.

## Laptop Setup

Use Python 3.11 or 3.12 for the Hugging Face phase:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Run the local classifier tests and evaluation:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
python -m social_benchmark.pipeline.cli evaluate-sklearn-classifier `
  --training datasets/training/hn_manual_training.jsonl `
  --out data/training/hn_sklearn_eval_laptop.json `
  --runs 8
```

Load `datasets/queues/label_queue_hn_expanded_2026-06-01_v3.csv` into the
review UI to add the next human-reviewed batch.
