# claude.md

## Project Description

This project is a human-perceived LLM benchmark and decision-support product. It evaluates models from real user evidence in public technical communities, then presents multidimensional scores with provenance, confidence intervals, and personalized recommendations.

The product should answer practical questions such as which model fits a workflow, which models are becoming unstable, which models users trust, and which models provide the best value.

## Tech Stack

- Frontend: React, Next.js
- Backend: Python, FastAPI
- Database: PostgreSQL via Supabase
- Data systems: official API collectors, embedding pipeline, vector search, spam/manipulation detection, statistical scoring engine
- Dashboard: leaderboard, model detail pages, regression alerts, evidence views, personalized comparisons

## Do

- Use official APIs and legally clear data access.
- Preserve provenance for every evidence item.
- Keep scores multidimensional instead of collapsing everything into one universal score.
- Show confidence intervals, effective sample size, source mix, and firsthand ratio.
- Separate satisfaction, trust, task fit, regression risk, hallucination complaints, refusal complaints, and value.
- Prefer structured sampling, caps, and cluster detection for manipulation resistance.
- Keep public scores open and monetize decision-support layers.
- Treat regression claims as strong only when corroborated across at least two platform families.
- Build the pipeline end to end with one source before adding high-volume sources.
- Use migrations for schema changes and keep scoring formulas testable.

## Do Not

- Do not build a simple sentiment-only ranking board.
- Do not sell score manipulation, paid placement, or hidden sponsorship influence.
- Do not scrape where official APIs or terms disallow it.
- Do not expose or resell raw community text beyond what source terms permit.
- Do not let one platform, community, author, or thread dominate a score.
- Do not hide low-confidence or source-concentration warnings.
- Do not mix unsafe-request refusals with excessive-refusal complaints.
- Do not add broad abstractions before the first source-to-score path works.

## MCP Server Usage

- Prefer project-local files and repository state before external lookup.
- Use Supabase MCP/tools only for database schema, migrations, auth, storage, or hosted project inspection.
- Do not use GitHub as a dataset source in the current MVP; Hacker News is the only active source.
- Use GitHub MCP/plugin tools for PRs, issues, repository metadata, CI checks, and publishing work.
- Use browser/web lookup only for current external facts, API docs, pricing, legal terms, or source-specific changes.
- Record any external assumption that affects data access, scoring, or compliance.

## Project Docs Usage

- Read `README.md` first for the product summary and current repository status.
- Read `plan.md` when choosing build order, MVP scope, milestones, or next implementation tasks.
- Read `data-pipeline.md` before designing schema, collectors, extraction models, scoring jobs, confidence intervals, deduplication, or release/update monitoring.
- Read `config/model_registry.json` before changing model/provider aliases, product/interface mappings, or inference profile labels.
- Read `docs/web-dashboard.md` before changing the web dashboard, the score-snapshot shape, model identity/provider rollup, tier thresholds, or the Supabase schema/loader.
- Read `description.md` only when deeper product/research rationale is needed.
- Keep `codex.md` synchronized with this file so Codex sessions can load the same guidance.

## Current Task

- Grow the reviewed Hacker News corpus (1,114 observations as of
  2026-06-12, rounds 5-8 merged): fetch fresh data, pipeline-label with the
  trained routed rubric classifier, review labels with LLM evaluation, and
  merge reviewed rows into training. All five fields now have an >=80%
  calibrated gated-precision operating point; the multi-task fine-tuned
  encoder (`run-finetuned-encoder-bakeoff`) ties the frozen stack with a
  single model.
- Re-run thread-grouped classifier experiments and calibrated gated
  precision (`run-gated-precision-bakeoff --calibrated`) on each expanded
  corpus; assess publication readiness after each round.
- The selected candidate is the multi-encoder stacked routed rubric
  classifier with a flat polarity head
  (`datasets/training/routed_variant_stacked_polflat_2026-06-12.json`,
  mean macro F1 `0.4700`; trained model
  `routed_stacked_polflat_round5_2026-06-12_model.joblib`). Firsthand,
  aspect, evidence, and 3-class sign polarity clear the 80% gated-precision
  bar; task does not yet. See
  `analysis/round5-polarity-iteration-report-2026-06-12.md`.
- Next: fine-tuned shared encoder at ~1,200 examples, polarity hard
  negatives in review rounds, locked 300-observation holdout.

## Next Task Recording

- When the classifier passes publication-readiness gates, build the scoring
  engine snapshot job and scaffold the dashboard stack (Next.js, FastAPI,
  PostgreSQL/Supabase) with the observation schema in migrations.
- Architecture options if data growth stalls: evidence-to-rubric
  cross-encoder, stronger instruction-tuned embedding checkpoints, or
  per-field data augmentation.

## Key Scoring Notes

- Observation score: -2 to 2 per model, task, and aspect.
- Aspect score: weighted normalized 0-100.
- Weight components: source quality, firsthand evidence, author credibility, corroboration, and recency.
- Default overall score combines satisfaction, trust, task fit, value, regression stability, hallucination safety, and refusal acceptance.
- Confidence warnings should trigger for low effective sample size or overconcentrated source mix.

## Data Pipeline Notes

- Do not use paid LLM API calls for routine per-post extraction.
- Use official source APIs for collection, then local rules, local classifiers, local embeddings, and human labeling for extraction.
- Store raw source items separately from extracted observations so improved classifiers can reprocess historical data.
- Keep `provider_id`, `model_id`, `product_id`, and `inference_profile` separate. Products such as Claude Code or ChatGPT are not model IDs when the base model is known.
- LLM API calls are allowed for monitoring official model releases, update notes, pricing/limit changes, alias discovery, and aggregate shift analysis.
- Release/update records should help explain why satisfaction or dissatisfaction changes after provider announcements or model behavior changes.

## Update - 2026-06-11

Superseding note: read `analysis/stacked-encoder-iteration-report-2026-06-11.md`
first. The corpus is now `859` accepted observations across `157` threads
(`datasets/training/hn_manual_training_threaded_round4_2026-06-11_merged.jsonl`).
The selected candidate is the multi-encoder stacked routed classifier with an
NLI polarity feature (`datasets/training/routed_variant_stacked_v2_2026-06-11.json`,
trained model `datasets/training/routed_stacked_round4_2026-06-11_model.joblib`)
at strict thread-grouped mean macro F1 `0.4639`. The selective soft chain is
retired. Use `fetch-hn-search` for targeted collection and
`run-gated-precision-bakeoff` for production-trust measurement. Firsthand is
deployable at 80% precision / full coverage; polarity is the weakest field.

## Current Classifier Handoff - 2026-06-07

This section is the authoritative starting point for the next Codex session.
It supersedes the older `Current Task` and `Next Task Recording` sections above
for classifier and benchmark-quality work.

### Current State

- Active source scope remains Hacker News only.
- CUDA is available on an NVIDIA RTX 3070 with 8 GB VRAM.
- Current isolated reviewed training set:
  `datasets/training/hn_manual_training_threaded_round2_2026-06-07_merged.jsonl`
- Training set size: `547` accepted observations across `69` threads.
- The five permanently held-out gold threads have zero overlap with training.
- Do not merge the gold set into training or repeatedly tune against it.
- Full test suite status: `69` tests passed.

### Accuracy Progress

The original round-two frozen BGE evidence-only classifier reached:

```text
strict thread-grouped mean macro F1: 0.3954
```

The selected routed rubric architecture reached:

```text
strict thread-grouped mean macro F1: 0.4169
absolute improvement: +0.0215
relative improvement: approximately +5.4%
```

The retained selective soft-chain experiment reached:

```text
strict thread-grouped mean macro F1: 0.4217
absolute improvement over routed rubric: +0.0048
```

Only the `firsthand_flag` probability vector is passed to `evidence_type`.
This improved evidence macro F1 from `0.3675` to `0.3916` without changing the
other field results. Broad chains and retrieval augmentation were rejected.

Selected routed field results:

| Field | Encoder and head | Macro F1 |
|---|---|---:|
| task category | MPNet plus strong rubric features | `0.3396` |
| aspect category | BGE Small plus light rubric features | `0.3078` |
| evidence type | BGE Small flat head | `0.3675` |
| polarity | BGE Small plus ordinal threshold head | `0.3048` |
| firsthand | raw BERT specialized binary head | `0.7647` |

The routed model is better but is not publication-ready. The publication gate
currently fails because there are only `69` thread groups and task, aspect,
evidence, and polarity remain below `0.60` macro F1.

### Current Candidate And Commands

Selected trained candidate:

```text
datasets/training/routed_rubric_round2_2026-06-07_model.joblib
```

Strict evaluation:

```text
datasets/training/routed_rubric_thread_grouped_2026-06-07.json
```

Publication assessment:

```text
datasets/evaluation/routed_rubric_publication_readiness_2026-06-07.json
```

Reproduce the selected evaluation:

```powershell
$env:PYTHONPATH='src'
python -m social_benchmark.pipeline.cli run-routed-rubric-bakeoff --training datasets/training/hn_manual_training_threaded_round2_2026-06-07_merged.jsonl --out datasets/training/routed_rubric_thread_grouped_2026-06-07.json --runs 8 --group-field thread_id --embedding-cache-dir datasets/training/embedding_cache
```

Train the selected routed candidate:

```powershell
$env:PYTHONPATH='src'
python -m social_benchmark.pipeline.cli train-routed-rubric-classifier --training datasets/training/hn_manual_training_threaded_round2_2026-06-07_merged.jsonl --model-out datasets/training/routed_rubric_round2_2026-06-07_model.joblib
```

Run all tests:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

### Experiments Already Tried

Do not repeat these without a materially different design:

- Hierarchical routing for task, aspect, and evidence regressed to `0.3773`.
- Hard dependency constraints regressed because evidence mistakes propagated
  into otherwise correct fields. Keep constraints audit-only.
- Raw BERT across every field reached only `0.3608`, but BERT was best for the
  specialized firsthand field.
- MPNet across every field reached only `0.3815`, but MPNet was best for task.
- Broad SetFit pilots did not beat frozen embeddings.
- Adding broad reviewed batches without targeted selection produced nearly flat
  accuracy.

Keep only changes that improve the same strict eight-run thread-grouped
evaluation or provide required correctness infrastructure.

### Next Architectural Experiments

Implement and benchmark changes one at a time. Run focused tests after each
code step and the strict thread-grouped evaluation after each behavior change.
Revert or leave experimental-only anything that worsens results.

Recommended order:

1. Soft classifier chain.
   - Use out-of-fold probability vectors from earlier fields as features for
     later fields.
   - Suggested order: firsthand, evidence, aspect, task, polarity.
   - Do not use ground-truth upstream labels at evaluation or inference time.
   - Compare multiple chain orders or an ensemble of chains.

   Status: completed. Retain only the selective `firsthand_flag` to
   `evidence_type` dependency. Broad chains damaged downstream fields.

2. Retrieval-augmented classification.
   - Retrieve nearest reviewed examples using cached embeddings.
   - Add neighbor label distributions, similarity, and disagreement as model
     features.
   - Preserve thread-grouped evaluation so neighbors from held-out threads do
     not leak into training.

   Status: completed and rejected. The best tested variant regressed to
   `0.4004` mean macro F1. Do not repeat without a materially different
   retrieval representation or training design.

3. Evidence-to-rubric cross-encoder.
   - Start with evidence type and aspect because they remain weak.
   - Convert every reviewed row into positive evidence/rubric pairs plus hard
     negative pairs from commonly confused labels.
   - Test a small NLI-trained DeBERTa or ModernBERT-sized encoder using CUDA,
     mixed precision, small batches, and gradient accumulation.

4. Shared multi-task encoder.
   - One encoder with five specialized heads.
   - Keep polarity ordinal.
   - Prefer partially frozen layers or parameter-efficient fine-tuning because
     the dataset is small and VRAM is limited.

5. Mixture-of-experts confidence router.
   - Assemble only experts that improve held-out results.
   - Easy/high-confidence rows can use cheap heads.
   - Ambiguous rows can use the cross-encoder or abstain for review.

### Next Data Work

Architecture alone will not reach publication quality. After the next two
cheap architecture experiments, generate targeted review queues for:

- `long_context`, `multimodal`, and `data_analysis`
- `regression_stability`, `hallucination_safety`, and `refusal_acceptance`
- `integration_failure`, `bug_regression_report`, and
  `release_update_reaction`
- hard-negative contrasts:
  task fit versus satisfaction, firsthand usage versus comparative evaluation,
  regression report versus ordinary complaint, and neutral versus mild
  positive/negative polarity

Build a new representative locked holdout with at least `300` accepted
observations across at least `75` threads. Never use that holdout for training.

### Publication Gates

Use `assess-publication-readiness`. Do not call an automatic classifier
publication-ready until all required gates pass:

- representative locked holdout with at least `300` accepted observations
- at least `75` independent thread groups
- target mean macro F1 at least `0.70`
- no scored field below `0.60` macro F1
- high-confidence automatic predictions calibrated to at least `90%` precision
- no tuning against the permanently locked holdout

### Files To Read First

Read these before making classifier decisions:

- `analysis/architecture-upgrade-report-2026-06-07.md`
  - authoritative experiment results and retained/rejected architecture choices
- `analysis/soft-chain-and-retrieval-report-2026-06-07.md`
  - latest selective chain improvement and rejected retrieval experiment
- `analysis/round2-review-processing-report-2026-06-07.md`
  - latest reviewed batch composition and accuracy movement
- `analysis/gold-evaluation-report-2026-06-06.md`
  - locked gold-set weaknesses and handling rules
- `docs/classification-roadmap.md`
  - classifier workflow and selected routed architecture
- `docs/labeling-guide.md`
  - authoritative label meanings
- `src/social_benchmark/pipeline/routed_classifier.py`
  - selected routed rubric model
- `src/social_benchmark/pipeline/rubric_classifier.py`
  - label definitions and rubric feature experiments
- `src/social_benchmark/pipeline/structured_classifier.py`
  - ordinal polarity and rejected hierarchy experiment infrastructure
- `src/social_benchmark/pipeline/constraint_resolver.py`
  - optional consistency audit; do not enable hard overrides by default
- `src/social_benchmark/pipeline/publication_readiness.py`
  - publication-quality gates

### Working Rules For The Next Session

- Preserve all user changes in the dirty worktree.
- Do not delete or overwrite reviewed CSVs.
- Do not train on or repeatedly evaluate the permanently isolated gold set.
- Always group evaluation by `thread_id`.
- Use the cached embeddings in `datasets/training/embedding_cache`.
- Use CUDA when it materially accelerates encoder work.
- Test after each implementation step instead of coding the entire experiment
  before verification.
- Report per-field changes, not only aggregate mean macro F1.
