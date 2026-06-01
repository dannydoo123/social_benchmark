# Initial Signal Test

This is a historical mixed-source experiment. The current MVP scope is Hacker News only; GitHub is no longer part of the active collection plan.

## Inputs

The test used the current local pipeline against live official APIs:

- Hacker News: 10 top stories
- Hacker News expanded: same 10 stories plus up to 100 comments per story
- GitHub: 10 issue-search results for `gpt in:title,body type:issue updated:>=2026-01-01`

Generated files are under `data/raw/` and `data/processed/`, which are intentionally ignored by Git.

## Output Summary

Title-only Hacker News:

- Raw items: 10
- Observations: 1
- Result: too sparse for scoring because story titles rarely contain user experience evidence.

Hacker News with comments:

- Raw items: 569
- Observations: 54
- Dominant models: Claude, Claude Code, GPT-5, Gemini, Claude Opus
- Dominant aspects: satisfaction, developer ergonomics, value
- Dominant evidence type: hearsay, then integration/pricing/release reaction

GitHub issue search:

- Raw items: 10
- Observations before span-local extraction fix: 516
- Observations after span-local extraction fix: 40
- Result: GitHub metadata and bodies are useful, but broad search is noisy and long issue bodies can overgenerate without span-local extraction.

Combined run:

- Raw items: 579
- Observations after source cleanup: 79
- Source mix: 54 Hacker News observations, 40 GitHub observations
- Dominant aspects: satisfaction, developer ergonomics, value
- Dominant task categories: coding, general, agents, API/developer workflow
- Claim types after broadening the initial polarity lexicon: 68 neutral, 17 complaint, 9 praise

After adding source-specific cleanup, the same combined sample produced 79 observations:

- Source mix: 46 Hacker News observations, 33 GitHub observations
- Claim types: 54 neutral, 17 complaint, 8 praise
- Quality warnings: low firsthand detection, neutral overproduction, low average extractor confidence

After adding the standalone firsthand detector:

- Firsthand observations: 11 of 79
- Remaining quality warnings: neutral overproduction, low average extractor confidence
- All sample scores remain unpublished because effective sample size and concentration thresholds are not met

## Ratings

Hacker News title-only signal: 2/10

- Good for detecting provider release announcements.
- Poor for satisfaction scoring without comments.

Hacker News with comments signal: 6/10

- Good source for release reactions and comparative discussion.
- Needs comment-level filtering, thread caps, and better firsthand detection.

GitHub broad issue search signal: 4/10

- Good structured source, but generic search pulls unrelated repos and task trackers.
- Should be driven by curated repos, labels, and source allowlists rather than broad keyword search.

Current rule extractor quality: 4/10

- Good enough as a bootstrap data generator.
- Too many neutral observations.
- Weak at separating real model-quality claims from boilerplate, task lists, and release mentions.

Current scoring reliability on this sample: 2/10

- The score math works, but sample size is far too small.
- All outputs correctly show low-confidence and source-concentration warnings.
- Scores should not be considered product-quality until there is enough corroborated evidence.

Pipeline architecture after this test: 7/10

- Official API collection works.
- Local-only extraction works.
- JSONL processing and scoring work.
- The span-local extraction fix reduced overgeneration from 516 to 40 observations on 10 GitHub issues.

## Improvements

Highest priority:

1. Use HN comments by default. HN story titles alone should feed release/update monitoring, not satisfaction scoring.
2. Strip noisy boilerplate, quote blocks, and generated task text before extraction.
3. Add source-item and thread caps directly in scoring so one thread cannot dominate.
4. Add a manual labeling command that exports uncertain/high-impact observations for review.

Extractor improvements:

1. Train a local supervised classifier from reviewed examples.
2. Treat release announcements separately from user satisfaction evidence.
3. Improve firsthand detection beyond simple phrase matching.
4. Improve polarity beyond keyword counts, especially for technical complaints.
5. Keep extraction span-local and preserve the exact evidence span for audits.

Scoring improvements:

1. Do not publish scores when `effective_n < 30`.
2. Separate attention volume from quality scores.
3. Require cross-source corroboration for strong regression claims.
4. Add confidence intervals to the emitted score JSON.
5. Add per-source, per-thread, and per-author contribution caps before aggregation.
