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
- Use GitHub MCP/plugin tools for PRs, issues, repository metadata, CI checks, and publishing work.
- Use browser/web lookup only for current external facts, API docs, pricing, legal terms, or source-specific changes.
- Record any external assumption that affects data access, scoring, or compliance.

## Current Task

- Read `description.md`.
- Create `plan.md` describing the build priorities.
- Create this `claude.md` as compact project guidance under 100 lines.

## Next Task Recording

- Scaffold the repo with Next.js, FastAPI, PostgreSQL/Supabase, and shared project conventions.
- Define the first database schema for sources, communities, authors, threads, evidence items, models, observations, duplicate clusters, score snapshots, and confidence intervals.
- Implement the first official-API collector, preferably GitHub or Hacker News before Reddit.
- Add a small manual labeling flow to validate model, task, aspect, evidence type, and score extraction.

## Key Scoring Notes

- Observation score: -2 to 2 per model, task, and aspect.
- Aspect score: weighted normalized 0-100.
- Weight components: source quality, firsthand evidence, author credibility, corroboration, and recency.
- Default overall score combines satisfaction, trust, task fit, value, regression stability, hallucination safety, and refusal acceptance.
- Confidence warnings should trigger for low effective sample size or overconcentrated source mix.
