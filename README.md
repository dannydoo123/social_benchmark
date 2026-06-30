# Social Benchmark

Social Benchmark is a benchmark and decision-support pipeline for evaluating LLMs from real user evidence in public technical communities. It turns raw community discussion into structured observations, scores, review queues, and model comparison artifacts.

## Scope

- Primary benchmark source: Hacker News
- Additional collectors and experiments: Reddit and GitHub
- Output formats: JSONL, JSON summaries, review CSVs, and benchmark snapshots

## Repository Layout

| Area | Purpose |
|---|---|
| `src/social_benchmark/` | Python package and CLI implementation |
| `tests/` | Test suite |
| `web/` | React benchmark/review UI |
| `config/` | Source and model registry files |
| `datasets/` | Reviewed labels, training sets, evaluation outputs, and caches |
| `db/migrations/` | Initial PostgreSQL/Supabase schema |
| `docs/` | Review workflow and methodology docs |
| `analysis/` | Research notes and experiment reports |

## Setup

```powershell
cd C:\coding\social_benchmark\social_benchmark
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

Optional extras:

```powershell
pip install -e ".[hf]"
pip install -e ".[setfit]"
```

Install the web workspace when you need the review UI:

```powershell
cd web
npm install
```

## Common Commands

The main entrypoint is the `sb-pipeline` CLI.

Fetch and process data:

```powershell
sb-pipeline fetch-hn --kind top --limit 50 --comments 100 --out data/raw/hn_top.jsonl
sb-pipeline process-jsonl --raw data/raw/hn_top.jsonl --observations-out data/processed/observations.jsonl --scores-out data/processed/scores.json
sb-pipeline export-labels --observations data/processed/observations.jsonl --out data/processed/label_queue.csv --max-rows 200
```

Train and compare classifiers:

```powershell
sb-pipeline build-training-data --labels data/processed/label_queue.csv --out data/training/extractor_training.jsonl
sb-pipeline train-sklearn-classifier --training data/training/extractor_training.jsonl --model-out data/training/sklearn_model.joblib
sb-pipeline compare-classifiers --training data/training/extractor_training.jsonl --out data/training/classifier_comparison.json --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

Run the frozen embedding bake-off:

```powershell
sb-pipeline run-frozen-embedding-bakeoff --training datasets/training/hn_manual_training.jsonl --out datasets/training/frozen_embedding_bakeoff_grouped.json
```

## Web UI

```powershell
cd web
npm run dev
```

The web app is used for review workflows and the benchmark dashboard shell.

## Testing

```powershell
python -m pytest
```

## Key Docs

- `description.md`
- `plan.md`
- `claude.md`
- `codex.md`
- `data-pipeline.md`
- `docs/review-workflow.md`
- `docs/labeling-guide.md`
