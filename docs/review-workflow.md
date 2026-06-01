# Review Workflow

## What To Review

Start with rows from:

```text
data/processed/label_queue_publishability.csv
```

This workflow currently assumes Hacker News-derived rows only.

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
