# Labeling Guide

## Goal

Review exported observation rows and correct the fields needed to train the local extractor.

Focus on the `evidence_text` column. Do not infer from the full thread unless the exported span clearly depends on the title for context.

Review batches may come from Hacker News or Reddit. Reddit rows should be treated as benchmark-eligible only after human review; unreviewed Reddit observations are for queue triage, not public scoring.

## Human Fields

The review UI pre-fills human fields from the machine label. Edit only fields that need correction. In raw CSV review, blank human fields still mean the machine label is acceptable.

- `reviewed_flag`: `true` when the row has been manually checked even if no correction was needed.
- `human_provider_id`: provider, such as `anthropic`, `openai`, `google`, `meta`, or `mistral`.
- `human_model_id`: canonical model, such as `claude-opus-4.8`, `gpt-5.5`, `gemini-3.5-flash`, `llama`.
- `human_product_id`: product or interface, such as `claude-code`, `chatgpt`, `openai-api`, `gemini-api`, `cursor`.
- `human_inference_profile`: stated run configuration, such as `ultracode`, `high_effort`, `low_effort`, or `thinking`.
- `human_task_category`: one of `coding`, `writing`, `research`, `agents`, `roleplay`, `data_analysis`, `long_context`, `multimodal`, `api_developer_workflow`, `general`.
- `human_aspect_category`: one of `satisfaction`, `trust_reliability`, `task_fit`, `regression_stability`, `hallucination_safety`, `refusal_acceptance`, `value`, `developer_ergonomics`.
- `human_evidence_type`: one of `firsthand_usage`, `comparative_evaluation`, `bug_regression_report`, `integration_failure`, `benchmark_anecdote`, `hearsay`, `release_update_reaction`, `pricing_value_comment`.
- `human_polarity_score`: `-2`, `-1`, `0`, `1`, or `2`.
- `human_firsthand_flag`: `true` or `false`.
- `human_notes`: short note for unclear cases.

Do not label a product as the model when the base model is stated. For example, "Claude Code with Opus 4.8 in ultracode mode" should be `model_id=claude-opus-4.8`, `product_id=claude-code`, and `inference_profile=ultracode`.

## Polarity

- `2`: strong praise, clear model-quality endorsement.
- `1`: mild praise or positive task fit.
- `0`: neutral, descriptive, ambiguous, or no clear quality claim.
- `-1`: mild complaint or negative task fit.
- `-2`: strong complaint, severe regression, unusable behavior, or repeated failure.

## Firsthand

Mark `true` only when the author appears to have used, tested, paid for, deployed, or directly evaluated the model.

Examples:

- `I used Claude Code for this migration` -> true
- `We see GPT-5 timeouts in production` -> true
- `People say Gemini is bad` -> false
- `The release notes claim better coding` -> false

## Evidence Type

Use `release_update_reaction` for provider announcements or reactions to a new release unless the text also contains direct usage evidence.

Use `integration_failure` for API, SDK, latency, rate-limit, deployment, or tooling issues.

Use `bug_regression_report` when the text says a model became worse, broke existing workflows, or changed behavior negatively over time.

Use `hearsay` when the text is opinion, speculation, or secondhand commentary without direct usage.

## Review Batch Size

For the first useful local classifier, review 200 rows:

- 100 low-confidence rows
- 50 neutral rows that look like they may hide sentiment
- 50 high-impact rows from high-engagement threads

## After Review

Run:

```powershell
python -m social_benchmark.pipeline.cli evaluate-labels --labels data/processed/label_queue.csv --out data/processed/label_eval.json
python -m social_benchmark.pipeline.cli build-training-data --labels data/processed/label_queue.csv --out data/training/extractor_training.jsonl
python -m social_benchmark.pipeline.cli train-local-classifier --training data/training/extractor_training.jsonl --model-out data/training/local_nb_model.json
```

For Reddit benchmark inclusion without manual review, run the strict auto-reviewer first. It accepts only rows where the trained model and extractor agree on every benchmark field; everything else is marked excluded from scoring.

```powershell
python -m social_benchmark.pipeline.cli auto-review-labels --labels datasets/queues/reddit_review_queue.csv --context datasets/queues/reddit_context.jsonl --out datasets/queues/reddit_auto_reviewed.csv --report-out data/processed/reddit_auto_review_report.json --source-platform reddit --max-iterations 3
python -m social_benchmark.pipeline.cli build-training-data --labels datasets/queues/reddit_auto_reviewed.csv --context datasets/queues/reddit_context.jsonl --out datasets/training/reddit_reviewed.jsonl
python -m social_benchmark.pipeline.cli merge-training-data --input datasets/training/hn_manual_training_threaded_round8_2026-06-12_merged.jsonl --input datasets/training/reddit_reviewed.jsonl --out datasets/training/benchmark_reviewed_with_reddit.jsonl
python scripts/rebuild_snapshot.py datasets/training/benchmark_reviewed_with_reddit.jsonl --out web/public/snapshot.json --observations-out web/public/observations.json
python -m social_benchmark.pipeline.cli report-source-yield --source-platform reddit --raw data/raw/reddit_pilot.jsonl --observations data/processed/reddit_observations.jsonl --review-queue datasets/queues/reddit_review_queue.csv --training datasets/training/benchmark_reviewed_with_reddit.jsonl --snapshot web/public/snapshot.json --out data/processed/reddit_yield_post_review.json
```

The yield report is the canonical answer to "what percent was used": it reports raw-to-observation yield, queue-to-reviewed yield, raw-to-reviewed yield, and Reddit's share of the benchmark snapshot.

Do not weaken the auto-review gates to force Reddit volume. If the auto-review
report accepts zero rows, the correct benchmark share for that run is 0%.

The first model is only a local baseline. Its value is making extraction improvements measurable before heavier local ML dependencies are added.
