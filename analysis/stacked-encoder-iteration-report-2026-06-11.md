# Stacked Encoder And Data Expansion Iteration Report

Date: 2026-06-11

## Summary

Two targeted review rounds expanded the corpus from `547` to `859` accepted
observations (`69` to `157` threads). Combined with multi-encoder feature
stacking and a zero-shot NLI polarity feature, strict thread-grouped mean
macro F1 improved from `0.4217` to `0.4639`, a relative gain of about `10%`.
Confidence-gated precision now supports production use of `firsthand_flag`
at full coverage and selective use of aspect/task predictions.

## Data Rounds

Both rounds used the new `fetch-hn-search` command (official HN Algolia API)
with queries targeted at weak categories, excluded all previously used thread
IDs, were pipeline-labeled with the trained routed candidate, and were
reviewed by LLM evaluation against `docs/labeling-guide.md`.

| Round | Threads fetched | Observations | Reviewed | Accepted | Excluded |
|---|---:|---:|---:|---:|---:|
| 3 (2026-06-11) | 77 | 1,929 | 220 | 173 | 47 |
| 4 (2026-06-11) | 90 | 1,990 | 220 | 139 | 81 |

Round 4 contained more junk spans: comments about the Gemini protocol
(`gemini://`) misattributed to Google Gemini, the Groq hosting provider
extracted as the Grok model, and infrastructure/hardware discussion. The
extractor should learn these distinctions; see Next Steps.

Machine-vs-reviewer agreement (round 3): aspect `45%`, evidence `28%`,
confirming the rule-based labels are weak supervision at best.

## Experiments (all strict eight-run thread-grouped evaluation)

| Change | Corpus | Mean macro F1 | Decision |
|---|---|---:|---|
| Session start: routed + selective chain | 547 | `0.4217` | superseded |
| New checkpoints bakeoff (bge-base, gte-base, e5-base) | 547 | best `0.3936` standalone | informative only |
| Routed variant: gte polarity ordinal, e5 evidence | 547 | `0.4229` | retained |
| Soft chain on the new routing | 547 | `0.4213` | chain dropped |
| Best-mix routing (bge aspect/evidence, gte polarity) | 720 | `0.4356` | retained |
| NLI xsmall features on all weak fields | 720 | `0.4212` | rejected |
| NLI xsmall on polarity only, scale 3.0 | 720 | `0.4385` | retained |
| NLI deberta-v3-small | 720 | `0.4206` | rejected |
| Best-mix + NLI polarity | 859 | `0.4255` | superseded |
| Multi-encoder stacking (4 encoders, weak fields) | 859 | `0.4586` | retained |
| Stacking incl. firsthand | 859 | `0.4639` | **selected** |
| StandardScaler in logistic heads | 859 | `0.4486` | reverted |

Selected per-field results on 859 examples:

| Field | Macro F1 |
|---|---:|
| task category | `0.3676` |
| aspect category | `0.3969` |
| evidence type | `0.4283` |
| polarity | `0.3479` |
| firsthand | `0.7786` |

Notes:

- The firsthand-to-evidence soft chain only helped the weaker bge-small
  evidence head; with e5/stacked evidence it regresses. Chain retired.
- Zero-shot NLI entailment against label rubrics helps only ordinal polarity;
  the smaller `nli-deberta-v3-xsmall` beats `nli-deberta-v3-small` here.
- Feature standardization destroys the tuned rubric-scale weighting.

## Gated Precision (out-of-fold, selected config, 859 examples)

`run-gated-precision-bakeoff` measures precision/coverage of
confidence-gated predictions:

| Field | 80%+ precision available at | Coverage |
|---|---|---:|
| firsthand | threshold 0.50 (p=0.80) | `100%` |
| firsthand | threshold 0.70 (p=0.93) | `43%` |
| aspect | threshold 0.60 (p=0.83) | `10%` |
| task | threshold 0.80 (p=0.76) | `13%` |
| evidence | threshold 0.80 (p=0.80) | `2%` |
| polarity | not reachable (p≈0.42 max at usable coverage) | — |

Production-trust interpretation: firsthand classification is deployable
today at the 80% bar. Aspect/task can auto-label a small high-confidence
slice with review fallback. Polarity must remain human/LLM-reviewed.

## New Infrastructure

- `fetch-hn-search`: targeted thread collection via the official Algolia API.
- `run-gated-precision-bakeoff`: leakage-safe precision/coverage measurement.
- `--field-config` on routed/soft-chain/train CLIs: encoder routing,
  `extra_embedding_models` stacking, and `nli_model`/`nli_scale` per field.
- `nli_features.NliRubricScorer`: cached zero-shot NLI rubric scoring.
- `RoutedRubricClassifier` now trains/predicts with stacking and NLI, so the
  selected candidate is usable for labeling exports.

## Artifacts

- Selected evaluation: `datasets/training/routed_stacked_v2_round4_2026-06-11.json`
- Trained candidate: `datasets/training/routed_stacked_round4_2026-06-11_model.joblib`
- Selected field config: `datasets/training/routed_variant_stacked_v2_2026-06-11.json`
- Gated precision: `datasets/evaluation/gated_precision_stacked_round4_2026-06-11.json`
- Readiness: `datasets/evaluation/routed_stacked_publication_readiness_2026-06-11.json`
- Reviewed rounds: `datasets/queues/targeted_training_review_round{3,4}_2026-06-11_filled.csv`
- Merged corpus: `datasets/training/hn_manual_training_threaded_round4_2026-06-11_merged.jsonl`

## Publication Gate Status

Still failing: mean `0.4639 < 0.70`; task/aspect/evidence/polarity below
`0.60`. Firsthand passes its bar.

## Next Steps (priority order)

1. Polarity rework: the ordinal head is the weakest and worst calibrated.
   Try a flat 5-class head on stacked features, polarity-specific hard
   negatives in review rounds, and collapsing to 3 classes (-1/0/+1) as a
   sensitivity check on whether the -2/+2 distinction is the bottleneck.
2. Extractor fixes from review findings: filter Gemini-protocol threads,
   stop matching Groq-the-host as Grok-the-model, drop pure hardware/infra
   spans. This raises queue yield (round 4 wasted 37% of review effort).
3. Continue data rounds toward the 300-observation locked holdout and 75+
   thread requirement; rounds cost ~2 hours each with the current tooling.
4. Fine-tuned shared encoder (multi-task heads, partially frozen, CUDA)
   once the corpus passes ~1,200 examples; frozen-feature gains are slowing.
5. Per-field confidence calibration (isotonic on out-of-fold probabilities)
   to widen the ≥80% precision coverage without accuracy gains.
