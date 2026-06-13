# Web Dashboard & Supabase Backend

Reference for the public benchmark dashboard (`web/`), its Supabase data store,
and the scoring/identity pipeline that feeds it. Read this before changing the
dashboard, the snapshot shape, the model-resolution logic, or the Supabase
schema.

Last updated: 2026-06-13.

---

## 1. What this is

A React (Vite) dashboard that presents the human-perceived LLM benchmark from
reviewed Hacker News evidence. It reads a versioned **score snapshot** from
Supabase (live) and falls back to a static snapshot file when Supabase is
unreachable. It is read-only to the public; all writes happen offline.

Source of truth for *scores* is the snapshot JSON produced by the re-scoring
job; Supabase is a serving layer loaded from that JSON.

---

## 2. Pages

The dashboard (`web/src/Dashboard.tsx`) has five views. **Providers** is the
landing page because the per-model corpus is still thin.

| Page | What it shows | Why |
| --- | --- | --- |
| **Providers** | Provider-level ranking (anthropic, openai, …) with overall score, ESS, tier, per-aspect breakdown, evidence receipts. | The only board with enough evidence to reach `ranked` today. |
| **Models** | Versioned models only (e.g. `claude-opus-4.8`), per-aspect + overall. | Specific-model view; caps at `provisional` until the corpus grows. |
| **Fields** | Per-task (coding, writing, …) comparison across aspects. | "Benchmarks by field". Task-gated observations only. |
| **Evidence** | Filterable table of every reviewed observation with provenance. | Show-your-work / audit. |
| **Methodology** | Pipeline, classifier-evolution timeline, calibrated gated-precision table, **registry-coverage band**, ranking rules. | "What technological effort went in." |

A persistent **data-health band** (corpus size, providers, threads, unversioned
share, single-source caveat) sits above every page.

---

## 3. Data flow

```
HN Algolia/Firebase API
  → rule extraction (span + raw model/product/profile mention)
  → human-reviewed training corpus (datasets/training/*_merged.jsonl)
  → scripts/rebuild_snapshot.py   (resolve identity, score, roll up, tier)
       → web/public/snapshot.json        (static fallback)
       → web/public/observations.json    (evidence-explorer fallback)
  → python -m social_benchmark.pipeline.cli load-supabase   (serve live)
  → web/src/dataSource.ts  (Supabase first, static fallback)
```

The production scoring engine (`build_score_snapshot` in
`src/social_benchmark/pipeline/score_snapshot.py`) still runs the full gated
classifier path. `scripts/rebuild_snapshot.py` is a **faster human-only**
re-scorer that reuses the same `ScoreAggregator` and adds identity resolution +
provider rollup; it does not run embeddings, so it is the quick path for
re-scoring after data or registry changes.

---

## 4. Model identity & provider rollup (the important part)

**Problem:** ~81% of HN mentions name only a provider/family ("Claude",
"Gemini", "DeepSeek") with no version. Treating those as a model created a fake
`claude` bucket (363 obs) that both fragmented and diluted the real per-model
board, and produced overlap between `claude`, `claude-opus`, and
`claude-opus-4.8`.

**Resolver:** `src/social_benchmark/pipeline/model_resolver.py`
- Resolves the provider from `config/model_registry.json` first, then a
  first-token alias map (`gpt`→openai, `qwen`→alibaba, `kimi`→moonshot, …). This
  fixes extractor mislabels like `qwen-3.6 → anthropic` and `gpt-oss-120b → meta`.
- A mention is a **specific model** only if it carries a version token (a digit).
  `claude-opus` / `gemini` → unversioned; `claude-opus-4.8` / `gemini-3-flash` /
  `o3` → versioned.
- Resolution `status`: `registry` (in registry) · `versioned_unregistered`
  (real model, refresh the registry from provider docs) · `unversioned` (rolls
  up to provider only) · `unknown` (empty/unattributable).

**Rollup rules:**
- **Providers board** aggregates *all* of a provider's observations (versioned +
  unversioned). This is why providers reach `ranked` (anthropic ESS ~611).
- **Models board** shows versioned models only; unversioned evidence never lands
  on a specific model's score.
- The Methodology **registry-coverage band** surfaces the status split and the
  top `versioned_unregistered` models so the registry can be kept current —
  this is how new models enter the board.

**Adding/curating models:** edit `config/model_registry.json` (refresh from
official provider docs), then re-score. No code change needed for a new versioned
model to appear; registry entries add the official display name and mark it
`registry` instead of `versioned_unregistered`.

---

## 5. Scoring & tiers

- Observation polarity (−2..2) → 0–100 via `((p+2)/4)*100`; weighted by source
  quality, firsthand, author credibility, corroboration, recency; thread/author/
  community/platform concentration caps applied (`contribution_capped_weights`).
- Aspect score = weighted mean; CI = weighted-score interval; ESS = effective
  sample size of capped weights.
- **Overall tier uses the entity's TOTAL evidence**, not its weakest aspect.
  (The old UI took `Math.min` across aspects, which made every model look
  "insufficient" — fixed.)
- **Tiers recalibrated for the current single-source corpus** (in
  `scripts/rebuild_snapshot.py`): `ranked` ESS ≥ 50, `provisional` ESS ≥ 15,
  else `insufficient`. The original 30/150 cutoffs were for a much larger
  multi-platform corpus and left everything insufficient. Insufficient rows are
  never ranked and are shown greyed.

---

## 6. Supabase schema

Project ref `urnnulxipkmkwpkhqkip`. Migrations:
`benchmark_snapshot_schema`, `add_provider_rollup_columns`.

| Table | Purpose |
| --- | --- |
| `score_snapshots` | One row per snapshot (`is_current` flags the live one). jsonb columns: `gates`, `tier_thresholds`, `corpus`, `overall`, `providers`, `coverage`, `methodology`. |
| `leaderboard_rows` | Per versioned-model × aspect cell (score, ci, ess, counts, warnings, tier). |
| `task_rows` | Per task × model × aspect. |
| `evidence_samples` | Top evidence spans per `model|aspect` and `provider|aspect`. |
| `observations` | Reviewed observations for the evidence explorer. |

**RLS:** every table has `enable row level security` + a `select using (true)`
policy. There are **no** insert/update/delete policies, so the publishable/anon
key is read-only in the browser. Writes use the service-role key (bypasses RLS)
or the Supabase MCP (privileged). Never add a public write policy.

---

## 7. Web data layer

- `web/src/supabaseClient.ts` — singleton client; project URL + publishable key
  default baked in, overridable via `VITE_SUPABASE_URL` / `VITE_SUPABASE_ANON_KEY`
  (see `web/.env.example`).
- `web/src/dataSource.ts` — `getSnapshot()` reconstructs the `Snapshot` from
  Supabase (meta + leaderboard + task + evidence), falling back to
  `/snapshot.json`; `getObservations()` queries `observations`, falling back to
  `/observations.json`. Evidence receipts fall back to the static snapshot when
  `evidence_samples` is empty.
- A header pill shows whether the current view is `live · Supabase` or
  `static snapshot`.

---

## 8. How to refresh after new data or a registry change

```bash
# 1. Re-score (human-only, fast — no classifier/embeddings).
python scripts/rebuild_snapshot.py \
  datasets/training/<latest>_merged.jsonl \
  --out web/public/snapshot.json
# (also regenerate web/public/observations.json with resolved providers — see
#  the snippet in scripts/ history, or extend rebuild_snapshot to emit it.)

# 2. Load Supabase with a service-role key (bypasses RLS, no context cost).
export SUPABASE_URL=https://urnnulxipkmkwpkhqkip.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
python -m social_benchmark.pipeline.cli load-supabase \
  --snapshot web/public/snapshot.json \
  --observations datasets/training/<latest>_merged.jsonl

# 3. Verify + ship.
cd web && npm run build
```

Without a service-role key, the initial seed was applied via the Supabase MCP
(`scripts/gen_seed_sql.py` generates batched INSERT SQL; the `_seed/` output is
disposable and git-ignored/removed after use).

---

## 9. Known limitations / TODO

- **Single source (Hacker News).** All scores carry a single-source caveat;
  regression claims need a second platform family (per `claude.md`). Highest-value
  next step alongside more volume.
- **Per-model board is thin** — provisional at best. Provider rollup is the
  usable board today; the HN historical backfill + a second source are the path
  to ranked individual models.
- **Registry lag.** 153 `versioned_unregistered` models await `model_registry.json`
  entries (top: `claude-opus-4.7`, `gpt-4`, `gemini-3`). Curate from provider docs.
- **Methodology gated-precision table** renders from the static fallback only on
  the live path — the Supabase `methodology.gated_precision_artifacts` was seeded
  empty to keep the MCP load light. Push the full artifact via the service-role
  loader to make it live.
- **Tasks/Fields** are versioned-model only; a provider-level task rollup is not
  built yet.
- `public.rls_auto_enable()` is a pre-existing SECURITY DEFINER function flagged
  by the security advisor — not created by this work; review separately.

---

## 10. Key files

| File | Role |
| --- | --- |
| `web/src/Dashboard.tsx` | All five pages + components. |
| `web/src/dataSource.ts` | Supabase-or-static loader + `Snapshot` types. |
| `web/src/supabaseClient.ts` | Read-only Supabase client. |
| `src/social_benchmark/pipeline/model_resolver.py` | Identity resolution + provider derivation. |
| `scripts/rebuild_snapshot.py` | Fast human-only re-score with rollup + tiers. |
| `src/social_benchmark/pipeline/supabase_loader.py` + CLI `load-supabase` | Service-role loader. |
| `src/social_benchmark/pipeline/score_snapshot.py` | Full gated-classifier snapshot builder. |
| `config/model_registry.json` | Canonical providers/models (curate from docs). |
