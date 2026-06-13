# Round 5/6 Merge, Polarity Rework, Calibration, And Extractor Fix Report

Date: 2026-06-12

## Round 5 Review And Merge

Round 5 queue (200 rows from `hn_targeted_round5_2026-06-11.jsonl`) was
LLM-reviewed against `docs/labeling-guide.md` via
`datasets/queues/fill_round5_2026-06-11.py`:

- accepted 96, excluded 104 (48% yield, down from round 4's 63%)
- dominant junk: duplicate spans of the same comment-model pair (~25 rows)
  and hardware/self-hosting requirement talk (~10 rows); the rest were
  vendor pitches, meta commentary, and product-strategy threads
- merged corpus: 859 -> 955 examples, 157 -> 185 threads
  (`hn_manual_training_threaded_round5_2026-06-11_merged.jsonl`)

## Strict Eight-Run Thread-Grouped Results (955 examples)

Selected stacked config re-run (`routed_stacked_v2_round5_2026-06-11.json`):

| Field | Round 4 (859) | Round 5 (955) |
|---|---:|---:|
| task category | 0.3676 | 0.3643 |
| aspect category | 0.3969 | 0.4207 |
| evidence type | 0.4283 | 0.4171 |
| polarity (ordinal) | 0.3479 | 0.3329 |
| firsthand | 0.7786 | 0.7626 |
| **mean** | **0.4639** | **0.4595** |

Frozen-feature gains from pure data growth have stalled, as predicted in
the round 4 report.

## Gated Precision (out-of-fold, 955 examples)

`gated_precision_stacked_round5_2026-06-11.json`:

| Field | 80%+ precision at | Coverage |
|---|---|---:|
| firsthand | t=0.60 (p=0.83) | 71% |
| firsthand | t=0.50 (p=0.78, just below bar) | 100% |
| aspect | t=0.60 (p=0.85) | 13% |
| evidence | t=0.70 (p=0.81) | 7% |
| task | not reached below t=0.95 (p=0.86 at 1.8%) | — |
| polarity | not reachable (p<=0.45 at usable coverage) | — |

Aspect and evidence gated coverage both improved vs round 4 (10% and 2%).
Firsthand slipped slightly below the 80% bar at full coverage but clears it
at 71% coverage.

## Polarity Rework Experiments

| Variant | Polarity macro F1 |
|---|---:|
| ordinal head (selected config) | 0.3329 |
| flat 5-class head on same stacked features | 0.3433 |
| ordinal head on 3-class collapse (-1/0/+1) | 0.1996 (ordinal head degenerates) |
| flat head on 3-class collapse (-1/0/+1) | **0.4770** |

The flat head beats ordinal at 5 classes and the 3-class collapse adds a
further large gain: the -2/+2 magnitude distinction, not sign, is the main
polarity bottleneck. Label distribution (5-class): -2:87, -1:288, 0:113,
+1:372, +2:95. Production option: publish sign-level polarity from the
classifier and keep magnitude review-gated.

## Isotonic Confidence Calibration (new `--calibrated` gated bakeoff flag)

`run-gated-precision-bakeoff --calibrated` now maps raw confidences through
an isotonic regression fit on inner out-of-fold predictions inside each
outer train split (leakage-safe). Calibrated results on 955 examples with
the flat-polarity config
(`gated_precision_polflat_calibrated_round5_2026-06-12.json`):

| Field | >=80% precision at | Coverage |
|---|---|---:|
| firsthand | t=0.80 (p=0.84) | 66% |
| firsthand | t=0.90 (p=0.90) | 27% |
| aspect | t=0.80 (p=0.84) | 13% |
| evidence | t=0.90 (p=0.82) | 4% |
| task | t=0.90 (p=0.90) | 2% |
| polarity 5-class | not reachable | — |
| polarity 3-class (collapsed corpus) | t=0.80 (p=0.76, just below) | 6% |

Calibration makes thresholds interpretable as probabilities and lets task
reach the 80% bar for the first time (tiny coverage). 3-class polarity at
t=0.70 gives p=0.72 at 21% coverage — closest polarity has come to
production use.

## Extractor Fixes (`extractors/rules.py`, all 76 tests pass)

- Observations now collapse task/aspect variants of the same
  comment-model-evidence triple to the best-scoring span, eliminating the
  duplicate review rows that wasted ~12% of round 5 effort.
- New `hardware_infra_requirements` exclusion: spans with two or more
  hardware/self-hosting signals (GGUF, VRAM, quantization, llama.cpp, ...)
  and no firsthand/regression/hallucination/refusal signal are dropped, and
  such spans are score-penalized when picking evidence spans.
- Gemini-protocol and Groq-hosting context filters were already in place
  from the prior session.

## Round 6 Data Round (first with the fixed extractor)

Fetch targeted polarity-heavy language (12 queries, recency-sorted,
exclusion list v4 with 436 threads): 85 stories, 3,317 raw items, 198
observations (1.47 rows per item-model pair vs 1.86 in round 5 — the
task/aspect collapse working). Queue of 145 rows reviewed via
`fill_round6_2026-06-12.py`: 68 accepted, 77 excluded (47% yield; no
Gemini-protocol or hardware junk remained — exclusions were now dominated
by Show-HN vendor pitches and duplicates across overlapping windows).

Merged corpus: 955 -> 1,023 examples.

## Round 6 Strict Eight-Run Thread-Grouped Results (1,023 examples, flat polarity)

| Field | Round 5 (955, ordinal) | Round 6 (1,023, flat) |
|---|---:|---:|
| task category | 0.3643 | 0.3729 |
| aspect category | 0.4207 | 0.4117 |
| evidence type | 0.4171 | 0.4552 |
| polarity | 0.3329 | 0.3337 |
| firsthand | 0.7626 | 0.7762 |
| **mean** | 0.4595 | **0.4700** (new best) |

## Round 6 Calibrated Gated Precision (1,023 examples)

| Field | >=80% precision at | Coverage |
|---|---|---:|
| firsthand | t=0.60 (p=0.81) | 91% |
| firsthand | t=0.80 (p=0.89) | 53% |
| firsthand | t=0.90 (p=0.93) | 30% |
| aspect | t=0.80 (p=0.82) | 11% |
| evidence | t=0.90 (p=0.85) | 5% |
| task | not reached (p=0.77 at 1.2%) | — |
| polarity 5-class | not reached (p=0.63 at 0.9%) | — |

Firsthand is now production-deployable well past the 80% bar: 81% precision
at 91% coverage, 89% at 53%. Publication gates still fail (mean 0.47 < 0.70).

## Round 6 3-Class Polarity Calibrated Gated Precision (1,023 examples)

On the collapsed-sign corpus
(`gated_precision_pol3flat_calibrated_round6_2026-06-12.json`), polarity
clears the 80% bar for the first time:

| Threshold | Precision | Coverage |
|---|---:|---:|
| 0.70 | 0.71 | 23% |
| 0.80 | 0.77 | 4.2% |
| 0.90 | **0.83** | 2.6% |
| 0.95 | **1.00** | 1.4% |

Production interpretation: sign-level polarity can be auto-published for a
small high-confidence slice with review fallback; magnitude (-2/+2) stays
review-only.

## SetFit Fine-Tune Pilot (1,023 examples, single thread-grouped split)

| Field | SetFit bge-small 300 steps | Frozen stacked baseline |
|---|---:|---:|
| task category | 0.2681 | 0.3729 |
| polarity 3-class | 0.4674 | 0.4770 |

Contrastive fine-tuning of bge-small at this scale does not beat the
multi-encoder frozen ensemble. The fine-tuned-encoder lever should wait for
the ~1,200-example corpus and use the planned multi-task partially-frozen
recipe rather than per-field SetFit.

## Round 7 Data Round (task-category targeted)

Fetch targeted non-coding task categories (10 queries, recency-sorted,
exclusion list v5 with 521 threads): 68 stories, 2,132 raw items, 110
observations, 80-row queue. Reviewed via `fill_round7_2026-06-12.py`: 31
accepted, 49 excluded (39% yield — recency-sorted Show-HN vendor threads
dominate; relevance-sorted queries likely yield better).
Merged corpus: 1,023 -> 1,054 examples.

Strict bakeoff mean macro F1 `0.4622` (within noise of round 6's 0.4700
given only 31 new rows). Calibrated gated precision improved again:

| Field | >=80% precision at | Coverage |
|---|---|---:|
| firsthand | t=0.50 (p=0.80) | **97%** |
| firsthand | t=0.80 (p=0.87) | 59% |
| aspect | t=0.70 (p=0.81) | 11% |
| aspect | t=0.80 (p=0.92) | 7% |
| evidence | t=0.90 (p=0.81) | 5% |
| polarity 3-class | t=0.80 (p=0.90) | 3.1% |
| polarity 3-class | t=0.90 (p=0.95) | 1.8% |
| task | t=0.90 (p=0.88) | 0.8% |

Firsthand now meets the 80% bar at essentially full coverage; 3-class sign
polarity is solidly above the bar on its high-confidence slice.

## Fine-Tuned Shared Encoder (new `run-finetuned-encoder-bakeoff`)

New multi-task fine-tune harness: shared transformer, one linear head per
field (head lr 1e-3, encoder lr 2e-5), bottom layers frozen, mean pooling,
thread-grouped 4-run eval on CUDA. Results on the 1,054 corpus vs the
frozen stacked baseline (0.4622):

| Variant | Mean | Task | Aspect | Evidence | Polarity | Firsthand |
|---|---:|---:|---:|---:|---:|---:|
| bge-small, 2 layers, lr 2e-5 heads (bug) | 0.1694 | — | — | — | — | — |
| bge-small, 2 layers, head-lr fix | 0.3514 | 0.21 | 0.31 | 0.30 | 0.24 | 0.70 |
| bge-base, 4 layers, 5 epochs | 0.4508 | 0.35 | 0.39 | 0.42 | 0.35 | 0.75 |
| **bge-base, 6 layers, 8 epochs** | **0.4682** | 0.31 | 0.42 | 0.46 | **0.37** | **0.79** |
| frozen stacked baseline | 0.4622 | 0.37 | 0.41 | 0.43 | 0.33 | 0.77 |

The fine-tuned encoder now beats the frozen four-encoder stack overall and
is the best polarity and firsthand result to date. Only task category
regresses — a hybrid (fine-tuned encoder for four fields, frozen stack for
task) is the obvious next candidate. SetFit per-field contrastive tuning
remains worse than both.

## Round 8 Data Round (relevance-sorted) And Final Status

Round 8 fetch went back to relevance sorting (10 task/polarity queries,
exclusion v6 with 589 threads): 65 stories, 2,232 raw items, 179
observations, 140-row queue, 60 accepted (43% yield; review caught one new
extractor gap — "grok hosted model" meaning Groq — now added to
`GROQ_HOSTING_SIGNALS`). Merged corpus: 1,054 -> **1,114 examples**.

Strict eight-run results on 1,114: frozen stacked flat mean `0.4696`;
fine-tuned bge-base (6 layers, 8 epochs) mean `0.4700` with better polarity
(0.364 vs 0.339) and better task (0.393 vs 0.372). The deeper v3 recipe
(8 layers, 12 epochs, 1,054 corpus) hit mean `0.4702` with firsthand
`0.8019` — the first field to cross 0.80 macro F1.

Round 8 calibrated gated precision (frozen flat config, 1,114):

| Field | >=80% precision at | Coverage |
|---|---|---:|
| firsthand | t=0.50 (p=0.80) | 96% |
| firsthand | t=0.80 (p=0.88) | 58% |
| aspect | t=0.70 (p=0.89) | 10% |
| evidence | t=0.90 (p=0.82) | 5% |
| polarity 3-class | t=0.80 (p=0.81) | 5.1% |
| task | t=0.80 (p=0.79, at the bar) | 6.2% |
| task | t=0.90 (p=0.90) | 1.7% |

Every field now has a >=80%-precision operating point (task right at the
boundary at 6% coverage, comfortably above at 1.7%).

## Next

1. Decide product policy: publish sign-level polarity gated at t>=0.90 with
   review fallback; magnitude review-gated.
2. Build gated prediction into the labeling/export path and the scoring
   engine snapshot job (claude.md "Next Task Recording").
3. Make the fine-tuned encoder a first-class candidate: per-fold
   fine-tuning inside the gated bakeoff (avoid leakage), or hybrid routing
   (fine-tuned encoder for polarity/firsthand/task, frozen stack for
   aspect/evidence).
4. Lock the 300-observation holdout.
2. Continue data rounds toward ~1,200 examples, then start the fine-tuned
   shared encoder (multi-task heads, partially frozen, CUDA).
3. Polarity-specific hard negatives in the next review round; consider a
   dedicated sign head plus magnitude head trained only on non-neutral rows.
4. Lock the 300-observation holdout once the corpus stabilizes.
