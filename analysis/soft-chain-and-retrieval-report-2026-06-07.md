# Soft Chain And Retrieval Experiment Report

Date: 2026-06-07

All results use the same 547 reviewed observations, 69 Hacker News threads,
eight thread-grouped holdout runs, and no held-out gold threads.

## Baseline

Selected routed rubric classifier:

- Mean macro F1: `0.416885`

## Soft Classifier Chain

Training features use inner grouped out-of-fold probabilities. Holdout
predictions never use ground-truth upstream labels.

| Variant | Mean macro F1 | Decision |
|---|---:|---|
| Full chain: firsthand, evidence, aspect, task, polarity | `0.419487` | Rejected as a full chain |
| Full chain: firsthand, evidence, task, aspect, polarity | `0.415559` | Rejected |
| Selective dependency: firsthand to evidence only | `0.421692` | Retained experiment |

The selective dependency improved evidence-type macro F1 from `0.367515` to
`0.391554`, an absolute gain of `0.024039`. All other field results remained
identical to the routed baseline. Mean macro F1 improved by `0.004808`.

Full chains were not retained as candidate behavior because downstream
dependencies reduced aspect and polarity quality.

Artifact:

- `datasets/training/soft_chain_routed_thread_grouped_2026-06-07.json`

## Retrieval Augmentation

The retrieval experiment added nearest-neighbor label distributions,
similarity summaries, and disagreement features. Training-row retrieval
excluded all examples from the same thread, and holdout rows retrieved only
from outer-training threads.

The best tested neighbor count reached only `0.400360` mean macro F1. This
regressed substantially against the routed baseline, so the implementation and
benchmark artifact were removed.

## Decision

Retain the leakage-safe soft-chain bakeoff infrastructure and the selective
firsthand-to-evidence result. Do not promote a broad chain or retrieval
augmentation. The next architecture experiment should start with an
evidence-to-rubric cross-encoder after targeted review data is expanded.
