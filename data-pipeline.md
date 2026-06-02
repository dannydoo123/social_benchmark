# Data Pipeline and Metric Extraction

## Purpose

This document defines how Social Benchmark collects public evidence and turns it into structured score inputs without using paid LLM API calls for every post or comment.

The core rule is:

- Use official source APIs for collection.
- Use deterministic filters and local text analysis models for extraction.
- Use limited LLM API calls only for model release/update monitoring and periodic analysis of major satisfaction shifts.

## Cost Model

Routine evidence extraction should not depend on frontier LLM API calls. At scale, per-item LLM analysis would make historical backfills and continuous monitoring too expensive.

The target cost structure is:

- Source collection cost: official APIs such as Hacker News.
- Processing cost: local batch inference, local embeddings, database storage, scheduled jobs.
- LLM API cost: limited release/update monitoring, taxonomy updates, and occasional aggregate shift analysis.

## Pipeline Overview

```text
Official API item
  -> raw item storage
  -> text normalization
  -> source-specific cleanup
  -> model alias matching
  -> cheap relevance filtering
  -> deduplication and near-duplicate clustering
  -> local classification and extraction
  -> observation rows
  -> weighted scoring
  -> confidence intervals and warnings
```

## Source Collection

Collect source items as raw records before extraction.

Primary fields:

- platform
- community or repository
- source item id
- parent/thread id
- author hash or platform id if terms permit
- published timestamp
- url
- title
- body/text
- engagement metadata
- moderation/deleted/locked state when available

Preferred first sources:

1. Hacker News because the official Firebase API exposes stories, comments, users, scores, timestamps, and thread structure.
2. Future sources can be added after the Hacker News pipeline is stable.

## Extraction Strategy

Do not call a large LLM for every item.

Use this staged extraction approach:

1. Rules and dictionaries
   - model alias dictionary
   - provider lookup table
   - task keyword lists
   - complaint keyword lists
   - comparison markers
   - first-person usage markers

2. Local relevance classifier
   - binary classifier: useful model-quality evidence vs noise
   - runs only after cheap keyword/model filters

3. Local multi-task classifier
   - task category
   - aspect category
   - evidence type
   - polarity score
   - firsthand flag
   - regression flag
   - hallucination complaint flag
   - refusal complaint flag
   - value flag

4. Human labeling workflow
   - review uncertain examples
   - label high-impact threads
   - label disagreement cases
   - periodically retrain local models

Current local tooling supports exporting uncertain/neutral observations to CSV and converting reviewed labels into supervised training JSONL.

Current model backends:

- dependency-free Naive Bayes baseline
- sklearn TF-IDF logistic baseline
- Hugging Face embedding-backed logistic classifiers using local embeddings

Use `docs/labeling-guide.md` when reviewing exported rows.

Reviewed-label workflow:

```text
observations.jsonl
  -> export-labels
  -> label_queue.csv
  -> human review
  -> evaluate-labels
  -> apply-labels
  -> build-training-data
  -> train-local-classifier
```

The current local classifier is a dependency-free Naive Bayes baseline. It is useful for regression testing and bootstrapping, but should eventually be replaced or supplemented by a stronger local encoder classifier.

Before extraction, apply source-specific cleanup:

- remove raw URLs from Hacker News comment spans
- keep exact evidence spans after cleanup for audit and labeling

## Local Model Design

Use local supervised models rather than generative extraction for routine processing.

Recommended first architecture:

- model/entity linker: alias dictionary plus canonical model table
- encoder: MiniLM, DeBERTa, RoBERTa, BERT, or Sentence-BERT style model
- heads:
  - relevance: binary
  - task category: multi-class
  - aspect category: multi-label
  - polarity: ordinal class from -2 to 2
  - evidence type: multi-class
  - flags: binary outputs

Initial labels can combine:

- manual labels
- weak labels from rules
- reviewed high-confidence examples
- reviewed uncertain examples

Target first labeled dataset:

- 500 examples for schema validation
- 2,000 examples for first useful local classifier
- 10,000+ examples for stable production extraction

## Observation Schema

Each observation is a specific evidence item about one model, one task category, and one aspect.

Model identity is split into separate dimensions:

- `provider_id`: the company or lab responsible for the model, such as `anthropic`, `openai`, or `google`.
- `model_id`: the canonical base or snapshot model being judged, such as `claude-opus-4.8`, `gpt-5.5`, or `gemini-3.5-flash`.
- `product_id`: the surface or wrapper used to access the model, such as `claude-code`, `chatgpt`, `openai-api`, or `gemini-api`.
- `inference_profile`: the run configuration when stated, such as `ultracode`, `high_effort`, `low_effort`, or `thinking`.

Scores should aggregate by `model_id` and `provider_id` first. Do not create a separate model for every effort level or product surface. Use `product_id` and `inference_profile` for filtering, diagnostics, and explaining why one use case performed differently.

Example:

```text
"Claude Code with Opus 4.8 in ultracode mode nailed it"
provider_id = anthropic
model_id = claude-opus-4.8
product_id = claude-code
inference_profile = ultracode
```

```text
observation {
  source_platform
  community_id
  thread_id
  source_item_id
  author_id_hash
  url
  published_at
  evidence_text

  model_id
  provider_id
  model_version_or_alias
  product_id
  inference_profile

  task_category
  aspect_category
  evidence_type
  claim_type

  polarity_score        # -2, -1, 0, 1, 2
  severity_score        # 0.0 to 1.0
  extractor_confidence  # 0.0 to 1.0

  firsthand_flag
  comparative_flag
  regression_flag
  hallucination_flag
  refusal_flag
  value_flag

  source_quality_weight
  firsthand_weight
  author_credibility_weight
  corroboration_weight
  recency_weight
  engagement_weight
  duplicate_penalty

  final_weight
  duplicate_cluster_id
  extractor_model_name
  extractor_model_version
  human_labeled_flag
}
```

## Metric Categories

Use these primary scoring categories:

- Satisfaction
- Trust and reliability
- Task fit
- Regression stability
- Hallucination safety
- Refusal acceptance
- Value
- Developer ergonomics

Task categories:

- coding
- writing
- research
- agents
- roleplay
- data analysis
- long-context work
- multimodal work
- API/developer workflow

Evidence types:

- firsthand usage
- comparative evaluation
- bug/regression report
- integration failure
- benchmark anecdote
- hearsay
- release/update reaction
- pricing/value comment

## Weighting

Observation weight:

```text
final_weight =
  source_quality_weight
  * firsthand_weight
  * author_credibility_weight
  * corroboration_weight
  * recency_weight
  * engagement_weight
  * duplicate_penalty
```

Default weight guidance:

- firsthand usage should outrank hearsay
- technical sources should outrank low-context social chatter
- corroborated claims should outrank isolated claims
- recent evidence should matter more for fast-moving models
- duplicate/campaign-like clusters should be heavily capped

## Score Aggregation

Map polarity to a 0-100 score:

```text
mapped_score = ((polarity_score + 2) / 4) * 100
```

Aspect score:

```text
aspect_score(model, aspect, window) =
  sum(final_weight * mapped_score) / sum(final_weight)
```

Complaint-rate categories:

```text
complaint_rate =
  weighted complaint observations / weighted relevant observations

positive_score =
  100 * (1 - complaint_rate)
```

Use complaint-rate logic for:

- hallucination safety
- refusal acceptance
- regression stability

Default overall score:

```text
Overall =
  0.20 Satisfaction
+ 0.18 Trust
+ 0.22 TaskFit
+ 0.12 Value
+ 0.12 RegressionStability
+ 0.10 HallucinationSafety
+ 0.06 RefusalAcceptance
```

Personalized profiles should adjust weights, not rewrite observations.

Scores must include a publishability gate. Compute scores for internal analysis, but block public display when:

- effective sample size is below 30
- one platform, community, or thread remains overconcentrated after caps
- confidence interval width is too large

The implementation exposes `publishable` and `publication_blockers` on score snapshots.

## Confidence and Warnings

Effective sample size:

```text
n_eff = (sum(weights)^2) / sum(weights^2)
```

Use Wilson intervals for binary complaint rates.

Use thread-week block bootstrap intervals for weighted aspect scores.

The current implementation also emits an approximate weighted confidence interval for aspect scores until the block bootstrap job exists.

Apply contribution caps before scoring:

- thread cap
- author cap
- community cap
- platform cap

Keep uncapped and capped weighted sample counts so the dashboard can explain how much evidence was downweighted.

Show low-confidence warnings when:

- n_eff < 30
- one platform contributes more than 60% of weighted evidence
- one community contributes more than 35% of weighted evidence
- one thread contributes more than 20% of weighted evidence
- recent score movement is not corroborated across source families

## Deduplication and Manipulation Controls

Apply these stages before scoring:

1. Exact source id/url deduplication
2. Text hash deduplication
3. SimHash or MinHash near-duplicate clustering
4. Local embedding similarity clustering
5. Author/thread/community caps
6. Burst detection for campaign-like behavior

The implementation includes an initial embedding sidecar flow:

```text
training or observation JSONL
  -> embed-jsonl
  -> embeddings.jsonl
  -> cluster-embeddings
  -> embedding_clusters.json
```

These clusters should feed duplicate review and campaign detection before any public scoring use.

Strong regression claims should require corroboration across at least two platform families.

## Limited LLM API Usage

LLM APIs may be used for model release and update monitoring, not routine per-post extraction.

Allowed LLM-assisted jobs:

- monitor official model release notes
- summarize provider update announcements
- map release dates to observed satisfaction changes
- suggest new model aliases after release announcements
- propose taxonomy updates for new capabilities
- analyze aggregate score shifts after a major release

Do not use LLM APIs to classify every post/comment/issue.

Release/update records should store:

- provider
- model
- release_or_update_name
- announced_at
- effective_at
- source_url
- capability_claims
- pricing_or_limit_changes
- deprecation_or_routing_changes
- expected affected categories
- notes

These records can explain why user satisfaction or dissatisfaction changes after a known model release, routing change, pricing change, safety-policy update, or API behavior change.

## Reprocessing Rule

Store raw items, candidate flags, extraction output, extractor version, and human labels separately.

Store the exact evidence span on every observation. This is required for audit views, manual labeling, and retraining local extraction models without reselling or republishing raw source text in bulk.

This allows old data to be reprocessed when:

- model aliases change
- local classifiers improve
- scoring formulas change
- new metric categories are added
- release-monitoring data adds context to historical shifts
