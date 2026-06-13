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
