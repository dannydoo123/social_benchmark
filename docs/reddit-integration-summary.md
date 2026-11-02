# Reddit Integration Summary

Date: 2026-06-26

## Current Status

Reddit support is implemented in the pipeline, but no real Reddit benchmark rows
have been collected yet in this workspace.

Current benchmark source mix:

- Hacker News: 1,114 reviewed observations
- Reddit: 0 reviewed observations
- Reddit benchmark share: 0.0%

The attempted Reddit fetch failed because unauthenticated Reddit JSON endpoints
returned HTTP 403 from this environment, and no Reddit OAuth environment
variables were configured.

Missing credentials:

- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_REFRESH_TOKEN`
- `REDDIT_USERNAME`
- `REDDIT_PASSWORD`

## Implemented Changes

### Reddit Collection

Added a Reddit client that can fetch subreddit listings, subreddit searches,
posts, and comments, mapping them into the existing `RawItem` pipeline format.

Main files:

- `src/social_benchmark/pipeline/clients/reddit.py`
- `src/social_benchmark/pipeline/collection.py`
- `src/social_benchmark/pipeline/cli.py`
- `config/reddit_sources.json`

Supported commands:

```powershell
python -m social_benchmark.pipeline.cli fetch-reddit-subreddit --subreddit ClaudeAI --limit 10 --comments 100 --out data/raw/reddit_subreddit.jsonl
python -m social_benchmark.pipeline.cli fetch-reddit-search --subreddit ClaudeAI --query "better than" --limit 10 --comments 100 --out data/raw/reddit_search.jsonl
python -m social_benchmark.pipeline.cli fetch-config --config config/reddit_sources.json --out data/raw/reddit_pilot.jsonl
```

### Reddit Source Config

The Reddit pilot config was split into measurable query groups instead of one
broad OR query:

- `comparative_better_than`
- `comparative_vs`
- `regression_stability`
- `hallucination_refusal`
- `coding_agents`
- `value_limits`

This lets the yield report show which query families produce useful benchmark
rows.

### Snapshot And Dashboard

The snapshot rebuild path now supports multiple reviewed JSONL files and emits
source-mix metadata.

Main files:

- `scripts/rebuild_snapshot.py`
- `src/social_benchmark/pipeline/score_snapshot.py`
- `src/social_benchmark/pipeline/supabase_loader.py`
- `web/src/Dashboard.tsx`
- `web/src/dataSource.ts`

The dashboard now shows dynamic source mix instead of assuming a Hacker News-only
corpus.

### Supabase Migration

Updated the dashboard snapshot migration so the database schema matches the
current loader and frontend expectations.

Main file:

- `db/migrations/002_score_snapshots.sql`

The migration now defines:

- `score_snapshots`
- `leaderboard_rows`
- `task_rows`
- `evidence_samples`
- dashboard `observations`
- `latest_snapshot` view

### Yield Reporting

Added a source-yield report so Reddit usage can be measured after each run.

Main files:

- `src/social_benchmark/pipeline/analysis.py`
- `src/social_benchmark/pipeline/cli.py`
- `tests/test_analysis.py`

Command:

```powershell
python -m social_benchmark.pipeline.cli report-source-yield `
  --source-platform reddit `
  --raw data/raw/reddit_pilot.jsonl `
  --observations data/processed/reddit_observations.jsonl `
  --review-queue datasets/queues/reddit_review_queue.csv `
  --training datasets/training/benchmark_reviewed_with_reddit.jsonl `
  --snapshot web/public/snapshot.json `
  --out data/processed/reddit_yield_post_review.json
```

The report includes:

- raw-to-observation yield
- raw-to-review-queue yield
- queue-to-reviewed-training yield
- raw-to-reviewed-training yield
- Reddit share of the benchmark snapshot

### Automated Review

Added a strict no-human-review path for Reddit rows.

Main files:

- `src/social_benchmark/pipeline/auto_review.py`
- `src/social_benchmark/pipeline/cli.py`
- `tests/test_auto_review.py`

Commands:

```powershell
python -m social_benchmark.pipeline.cli auto-review-labels `
  --labels datasets/queues/reddit_review_queue.csv `
  --context datasets/queues/reddit_context.jsonl `
  --out datasets/queues/reddit_auto_reviewed.csv `
  --report-out data/processed/reddit_auto_review_report.json `
  --source-platform reddit `
  --max-iterations 3
```

```powershell
python -m social_benchmark.pipeline.cli auto-review-observations `
  --observations data/processed/reddit_observations.jsonl `
  --out datasets/queues/reddit_auto_reviewed.csv `
  --report-out data/processed/reddit_auto_review_report.json `
  --source-platform reddit `
  --max-iterations 3
```

The auto-reviewer accepts a row only when:

- source platform matches
- provider and model IDs are present
- evidence text is long enough
- extractor confidence passes the gate
- claim is not neutral, unless explicitly allowed
- trained model does not abstain
- trained model and extractor agree on every benchmark field
- repeated passes are stable

Rows that fail are marked:

```text
human_excluded_from_scoring=true
```

That prevents rejected rows from entering training accidentally.

## Reliability Findings

The existing high-precision model was evaluated and found too weak on some fields
for broad unattended inclusion.

A current verifier model was trained from the latest HN reviewed corpus:

```text
datasets/training/reddit_auto_review_hn_round8.joblib
```

Evaluation summary:

- task category precision: about 83%
- aspect category precision: about 82%
- evidence type precision: about 86%
- polarity precision: about 83%
- firsthand precision: about 89%

Relaxed policies produced too many incorrect rows when tested against existing
filled HN review CSVs. The accepted policy is therefore strict all-field
agreement. If this produces zero accepted Reddit rows, the correct action is to
keep Reddit at 0% for that run rather than weakening the gates.

An experimental stricter model was also generated:

```text
datasets/training/reddit_auto_review_hn_round8_strict085.joblib
```

It evaluated poorly because coverage collapsed and should not be used.

## End-To-End Reddit Flow

Once Reddit credentials are configured:

```powershell
$env:PYTHONPATH='src'

python -m social_benchmark.pipeline.cli fetch-config `
  --config config/reddit_sources.json `
  --out data/raw/reddit_pilot.jsonl

python -m social_benchmark.pipeline.cli process-jsonl `
  --raw data/raw/reddit_pilot.jsonl `
  --observations-out data/processed/reddit_observations.jsonl `
  --scores-out data/processed/reddit_scores.json

python -m social_benchmark.pipeline.cli auto-review-observations `
  --observations data/processed/reddit_observations.jsonl `
  --out datasets/queues/reddit_auto_reviewed.csv `
  --report-out data/processed/reddit_auto_review_report.json `
  --source-platform reddit `
  --max-iterations 3

python -m social_benchmark.pipeline.cli build-training-data `
  --labels datasets/queues/reddit_auto_reviewed.csv `
  --out datasets/training/reddit_reviewed.jsonl

python -m social_benchmark.pipeline.cli merge-training-data `
  --input datasets/training/hn_manual_training_threaded_round8_2026-06-12_merged.jsonl `
  --input datasets/training/reddit_reviewed.jsonl `
  --out datasets/training/benchmark_reviewed_with_reddit.jsonl

python scripts/rebuild_snapshot.py `
  datasets/training/benchmark_reviewed_with_reddit.jsonl `
  --out web/public/snapshot.json `
  --observations-out web/public/observations.json

python -m social_benchmark.pipeline.cli report-source-yield `
  --source-platform reddit `
  --raw data/raw/reddit_pilot.jsonl `
  --observations data/processed/reddit_observations.jsonl `
  --review-queue datasets/queues/reddit_auto_reviewed.csv `
  --training datasets/training/benchmark_reviewed_with_reddit.jsonl `
  --snapshot web/public/snapshot.json `
  --out data/processed/reddit_yield_post_review.json
```

If `reddit_auto_review_report.json` shows `accepted: 0`, stop and do not merge
Reddit into the benchmark.

## Verification

Completed checks:

- Python test suite: 90 tests passed
- TypeScript check: `npx tsc --noEmit` passed
- `git diff --check` passed except CRLF line-ending warnings

Known local issue:

- `npm run build` still fails at Vite/esbuild with `spawn EPERM`. TypeScript
  succeeds, so this appears to be a local process-permission issue rather than a
  code/type issue.

## Next Required Step

Configure Reddit API credentials, then rerun the end-to-end Reddit flow.

Without credentials, this environment cannot fetch Reddit data because Reddit is
blocking unauthenticated JSON access with HTTP 403.
