# Fresh community perspective

The training baseline is used only by the existing verifier; it is never copied into the rolling evidence store. No automatic retraining or self-label feedback loop is enabled.

Run from the repository root:

    $env:PYTHONPATH='src'
    python scripts/refresh_live_evidence.py

The SQLite store is `data/live/evidence.sqlite`; the atomically published feed is `web/public/live-perspective.json`. Raw records retain parents, thread, community, author hashes, timestamps and engagement. Evidence rows retain parent context and review provenance. Raw author handles are local only; the public feed exposes links and aggregate counts.

`config/live_sources.json` sets the four-hour cadence, rolling 30-day window, query and community lists, and bounded per-run collection limits. These limits make this a pilot, not exhaustive coverage. Category tags are deterministic exploratory filters, not independently validated classifications. Model aliases require explicit model names; provider/product-only mentions remain ambiguous.

`scripts/install-refresh-schedule.ps1 -Hours 4` installs the local Windows task (1 is also supported). The computer must be on and the task account must have source credentials. The local development website reads the refreshed file every minute. A deployed static build does NOT update merely because the local JSON changes: production still needs a hosted collector plus serving/storage integration or a publish-on-refresh job.

Collection failures are reported by source and preserve prior data. Missing timestamps, future dates and evidence outside the rolling window cannot enter summaries. Exact-text duplicates are removed; one author/thread contribution and equal thread/community weighting reduce concentration. Near-duplicate/campaign detection is still needed.

Screening uses the frozen classifier and rule extractor agreement. This is not calibrated validation on new models or communities. Public scores remain withheld until fresh-domain validation is implemented and passes. Current minimum coverage checks require 30 threads, 30 authors, 3 communities across 2 platforms, and at least 5 threads per community; these are conservative initial policy settings, not a statistical guarantee. Community summaries are diagnostic only. Independent labeled holdouts, uncertainty intervals and domain-specific reliability calibration remain required before numeric publication.

Reddit may require OAuth environment credentials; blocked sources are never treated as evidence of neutral sentiment. No training was performed because the immediate gap is fresh, independently validated data rather than more fitting on the same baseline.

Verified API references: https://github.com/HackerNews/API and https://hn.algolia.com/api .
