# Community observatory implementation

## Current result

The public research view is now the default entry point. The baseline/review workspace remains accessible. Filters include model search, category, community, 7/30-day window, and vendor-community exclusion (on by default). Model rows drill into source-linked claim excerpts. Candidate labels are explicitly separate from validated scores.

## Claim semantics

`claim_extraction.py` v2 preserves clause offsets, splits local aspects, supports guarded adjacent references and direct-parent pronouns, and marks explicit named comparisons as relative preferences. Code, quotes, ambiguous comparisons and mixed same-aspect sentiment abstain. Exact version strings do not imply official registry verification. Labels remain unvalidated candidates. See [extraction-evaluation.md](extraction-evaluation.md) for measured iterations, sampling limits and the pending independent review queue.

## Bias control

Communities are grouped as general, workflow, vendor, or unclassified. These describe focus, not quality. Vendor sources are excluded by default from the UI sample. The internal diagnostic in `community_balance.py` averages claims within author/thread, then threads within author, authors within community, communities within cohort. Vendor-focused data contributes at most 20%; vendor-only samples produce no aggregate. Diagnostic results are saved locally in `data/live/balance-diagnostics.json`, never exposed as validated public scores. Cross-thread author dependence is reduced; near-duplicate campaign detection, interval estimation, and leave-one-community-out sensitivity remain unfinished.

## Collection and scheduling

- HN story/comment collection plus recent comment search via Algolia, up to 1,000 results per query per run. Search limits do not imply full coverage.
- Reddit public Atom feeds: genuine imported records; full JSON API is blocked without approved credentials. RSS is partial and does not expose all replies, parent IDs, or engagement. HTTP 429 trips a circuit breaker; a persisted community cursor rotates the next run. No proxy or access-control bypass.
- Free Arctic Shift archive collector: `scripts/collect_reddit_archive.py` rotates community/model searches, retrieves up to 100 recent comments per thread from up to five recent matching threads per query, and keeps original dates, thread/parent IDs and archive provenance. The scheduled run is capped at 30 requests (one request per second at most); manual trials can use up to 100. This is a bounded sample, not a complete archive. Access denial triggers a 24-hour cooldown; other failures defer four hours. Existing raw IDs are not overwritten by potentially stale archive text. Archive availability and lag are not guaranteed; source health records the latest publication timestamp returned. No paid provider or model-training job is enabled.
- OpenAI developer Discourse: public search and topic JSON, explicitly vendor-focused. The routine refresh collects returned topic posts; the historical backfill additionally paginates the topic's returned post-ID stream.
- Existing four-hour Windows task calls `run-live-refresh.ps1`, which now invokes HN, Reddit feed, HN comment-search, forum collection, then rebuilds `community-perspective.json`. The local development server reads the feed every minute. This remains a local-machine schedule, not hosted production scheduling.
- Raw sources are checkpointed in `data/live/evidence.sqlite`. Training data is excluded. Data is filtered to the configured rolling window, deduplicated by model/category/text, and source failures remain visible.

Targets in `config/live_sources.json` are 1,000 unique candidate chunks each for GPT-6-Astra and GPT-6.1-Sol. Validated chunks are tracked separately and remain zero pending independent review. A chunk is not an independent person or conversation. The target has not been reached.

## Historical conversation backfill

`scripts/backfill_model_conversations.py` searches 90 days of HN comments, retrieves full indexed conversation trees for discovered threads, paginates public Discourse conversations, and imports public Lemmy search matches. Thread context may predate the search window. Parent and thread IDs are preserved; an unrelated reply does not inherit a model label merely because the thread mentions that model. Source failures are recorded in `data/live/broad-collection-health.json`.

The connected GitHub search supplied public exact-name issue searches and comments from selected active issues. `scripts/import_github_evidence.py` imports these locally with `exclude_from_public_scoring: true`. Issue trackers overrepresent bugs, configuration problems and generated diagnostic text; account type does not establish human authorship. These records require relevance and attribution review before contributing opinions. This connector import is a one-time backfill, not part of the four-hour schedule.

The Sources view exposes `collection-coverage.json` through the public feed: historical exact-name records, comments-only counts, distinct texts and platform mix. These are separate from the rolling 30-day candidate claims. Exact mentions, unique text, independent experiences and validated opinions are different measures. Raw conversation bodies remain local. Large thread archives must never be presented as equally large samples of model opinions.

## Design sources

Read and applied Emil Kowalski design engineering, Impeccable Operate/craft-floor guidance, and Taste app-screen guidance. Instruction snapshots are stored under `design-references`. Product brief overrode marketing-oriented choices. No decorative animation, imagery, invented numbers, or sales sections.

Upstream sources:
- https://github.com/emilkowalski/skills/tree/main/skills/emil-design-eng
- https://github.com/pbakaus/impeccable
- https://github.com/Leonxlnx/taste-skill

Build and tests passed. Browser access host was unavailable; native Chrome selection was rejected because the existing window was unrelated private repository content. No visual QA or skill browser-inspector pass is claimed.
