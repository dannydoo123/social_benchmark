# Review Workflow

## What To Review

Start with rows from:

```text
data/processed/label_queue_publishability.csv
```

This workflow supports Hacker News and Reddit-derived rows. Reddit rows should be reviewed before they are merged into the public benchmark snapshot.

Prioritize:

1. Rows where the text clearly says the author used or tested the model.
2. Rows marked `neutral` that actually sound positive or negative.
3. Rows where `evidence_type` looks wrong, especially release notes vs real user experience.
4. Rows where model, task, or aspect is clearly misclassified.

The UI pre-fills review fields from the machine label. Edit only the values that are wrong, or press Accept to mark a row reviewed with no corrections.

## UI

The first React screen is a label review workspace under:

```text
web/
```

It supports:

- loading exported CSV, JSON, or JSONL label rows
- reviewing one evidence span at a time
- accepting machine labels
- correcting provider, model, product, profile, task, aspect, evidence type, polarity, and firsthand flag
- exporting reviewed CSV or JSON

## Commands

Generate a label queue:

```powershell
$env:PYTHONPATH='src'
python -m social_benchmark.pipeline.cli export-labels --observations data/processed/combined_observations_publishability.jsonl --out data/processed/label_queue_publishability.csv --max-rows 200
```

Generate a Reddit pilot queue with source context:

```powershell
$env:PYTHONPATH='src'
python -m social_benchmark.pipeline.cli fetch-config --config config/reddit_sources.json --out data/raw/reddit_pilot.jsonl
python -m social_benchmark.pipeline.cli process-jsonl --raw data/raw/reddit_pilot.jsonl --observations-out data/processed/reddit_observations.jsonl --scores-out data/processed/reddit_scores.json
python -m social_benchmark.pipeline.cli export-labels --observations data/processed/reddit_observations.jsonl --raw data/raw/reddit_pilot.jsonl --out datasets/queues/reddit_review_queue.csv --context-out datasets/queues/reddit_context.jsonl --max-rows 300
python -m social_benchmark.pipeline.cli auto-review-labels --labels datasets/queues/reddit_review_queue.csv --context datasets/queues/reddit_context.jsonl --out datasets/queues/reddit_auto_reviewed.csv --report-out data/processed/reddit_auto_review_report.json --source-platform reddit --max-iterations 3
python -m social_benchmark.pipeline.cli report-source-yield --source-platform reddit --raw data/raw/reddit_pilot.jsonl --observations data/processed/reddit_observations.jsonl --review-queue datasets/queues/reddit_review_queue.csv --out data/processed/reddit_yield_pre_review.json
```

If `reddit_auto_review_report.json` shows `accepted: 0`, stop there and do not
merge Reddit into the benchmark. The auto-reviewer is intentionally strict; a
zero-row result means the current extractor/model agreement is not reliable
enough for unattended inclusion.

Run the review UI:

```powershell
cd web
npm install
npm run dev
```

After exporting reviewed CSV from the UI:

```powershell
$env:PYTHONPATH='src'
python -m social_benchmark.pipeline.cli evaluate-labels --labels data/processed/label_queue_reviewed.csv --out data/processed/label_eval.json
python -m social_benchmark.pipeline.cli build-training-data --labels data/processed/label_queue_reviewed.csv --out data/training/extractor_training.jsonl
python -m social_benchmark.pipeline.cli train-local-classifier --training data/training/extractor_training.jsonl --model-out data/training/local_nb_model.json
```

After Reddit review, merge it into the benchmark and rebuild the static dashboard artifacts:

```powershell
$env:PYTHONPATH='src'
python -m social_benchmark.pipeline.cli build-training-data --labels datasets/queues/reddit_auto_reviewed.csv --context datasets/queues/reddit_context.jsonl --out datasets/training/reddit_reviewed.jsonl
python -m social_benchmark.pipeline.cli merge-training-data --input datasets/training/hn_manual_training_threaded_round8_2026-06-12_merged.jsonl --input datasets/training/reddit_reviewed.jsonl --out datasets/training/benchmark_reviewed_with_reddit.jsonl
python scripts/rebuild_snapshot.py datasets/training/benchmark_reviewed_with_reddit.jsonl --out web/public/snapshot.json --observations-out web/public/observations.json
python -m social_benchmark.pipeline.cli report-source-yield --source-platform reddit --raw data/raw/reddit_pilot.jsonl --observations data/processed/reddit_observations.jsonl --review-queue datasets/queues/reddit_review_queue.csv --training datasets/training/benchmark_reviewed_with_reddit.jsonl --snapshot web/public/snapshot.json --out data/processed/reddit_yield_post_review.json
```
