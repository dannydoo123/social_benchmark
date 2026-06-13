# Project Plan

## Goal

Build a human-perceived LLM benchmark and decision-support product that ranks and explains model quality using real user evidence from high-signal public communities.

## Product Principles

- Measure user-perceived quality, not just synthetic benchmark ability.
- Keep scores multidimensional: satisfaction, trust, task fit, regression stability, hallucination safety, refusal acceptance, and value.
- Make every score traceable to evidence, source mix, confidence interval, and sample quality.
- Monetize alerts, comparisons, reports, team views, and API access, not score manipulation.

## MVP Scope

1. Define the canonical observation schema.
2. Build collectors for official, high-signal sources.
3. Normalize, deduplicate, and classify evidence.
4. Implement weighted scoring with confidence intervals.
5. Ship a public leaderboard with provenance and uncertainty.
6. Add personalized comparison profiles after the global scoring pipeline works.

## Priority Sources

- Hacker News: Ask HN, launch posts, model comparison discussions.

Current MVP source scope is Hacker News only. Other sources remain future expansion candidates.

## Initial Data Model

- Source platform, community, thread, author, timestamp, URL, and raw text reference.
- Model mentioned, provider, task category, aspect label, sentiment score, evidence type.
- Firsthand flag, corroboration group, duplicate cluster, trust features, moderation status.
- Weight components: source quality, firsthand quality, author credibility, corroboration, recency.

## Scoring Milestones

1. Observation score from -2 to 2 per aspect.
2. Aspect scores normalized to 0-100.
3. Task-fit portfolio across coding, writing, research, agents, and roleplay.
4. Regression risk comparing recent and longer-window complaint rates.
5. Hallucination and refusal complaint rates converted into positive safety/acceptance scores.
6. Overall default score with configurable user weights.
7. Thread-week bootstrap confidence intervals and effective sample size.

## Anti-Manipulation Milestones

- Exact ID/URL deduplication.
- Near-duplicate clustering with SimHash or MinHash.
- Embedding similarity clustering for repost and campaign detection.
- Author/thread/community caps.
- Cross-platform corroboration requirement for strong regression claims.
- Low-confidence warnings for weak sample size or overconcentrated source mix.

## Dashboard Milestones

- Public leaderboard with multidimensional scores.
- Model detail pages with trend charts and source mix.
- Evidence drill-down without reselling raw community text.
- Regression alerts.
- Personalized profiles for agent builders, researchers, writers, enterprise users, creative users, local/privacy users, and budget-sensitive users.
- Paid exports, PDF reports, team dashboards, and API access.

## Recommended Build Order

1. Repository scaffold: Next.js frontend, FastAPI backend, PostgreSQL/Supabase, shared types.
2. Database schema and migration workflow.
3. One end-to-end Hacker News collector because the source is structured, official, and currently in scope.
4. Normalization and observation storage.
5. Manual labeling workflow for early ground truth.
6. Baseline classifier for model/task/aspect/evidence extraction.
7. Scoring engine and confidence interval job.
8. Minimal dashboard showing scores, sample counts, confidence, and evidence links.
9. Add additional sources only after Hacker News extraction quality, labeling, and scoring are stable.
10. Add paid decision-support features only after public scoring is credible.

## Current Status (2026-06-11)

The Hacker News collection, extraction, labeling, and classifier-experiment
pipeline is built and iterating. Reviewed training data stands at 547
observations across 69 threads. The selected classifier candidate is the
routed rubric architecture (per-field encoders plus rubric features) with a
selective firsthand-to-evidence soft chain, at thread-grouped mean macro F1
`0.4217`. The publication-readiness gate still rejects it: too few thread
groups and task/aspect/evidence/polarity below per-field thresholds.

## Current Task

- Expand the reviewed corpus: pull fresh Hacker News data, label with the
  pipeline classifier, review labels with LLM evaluation, and merge into
  training data.
- Re-run embedding and architecture experiments on the expanded corpus until
  per-field quality passes the publication-readiness gate.
- When improvements plateau, change architecture (e.g. evidence-to-rubric
  cross-encoder) or research new methods before adding more data.

## Next Task

- Once the classifier passes publication gates, return to the scoring engine
  and dashboard milestones: scaffold Next.js/FastAPI/Supabase, define the
  observation schema in migrations, and ship the minimal leaderboard.
