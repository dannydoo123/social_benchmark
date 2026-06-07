# Classifier Architecture Upgrade Report

Date: 2026-06-07

All results use the same 547 reviewed observations, 69 Hacker News threads,
eight thread-grouped holdout runs, and no held-out gold threads.

## Baseline

Frozen `BAAI/bge-small-en-v1.5`, evidence-only input, and one flat logistic
head per field:

- Mean macro F1: `0.3954`

## Staged Experiments

| Architecture | Mean macro F1 | Decision |
|---|---:|---|
| Hierarchical task/aspect/evidence plus ordinal polarity | `0.3773` | Rejected |
| Flat heads plus ordinal polarity | `0.3968` | Retained |
| Hard dependency constraints | `0.3950` to `0.3954` | Audit-only |
| Field-specific BGE rubric features | `0.4023` | Retained |
| Raw BERT rubric model for all fields | `0.3608` | Rejected globally |
| MPNet rubric model for all fields | `0.3815` | Rejected globally |
| Routed specialized encoders and rubric features | `0.4169` | Selected candidate |

Hierarchical routing and hard constraints propagated upstream mistakes into
otherwise correct fields. They remain available as experiments and consistency
audits, but are not active prediction overrides.

## Selected Routed Architecture

| Field | Encoder | Rubric scale | Head | Macro F1 |
|---|---|---:|---|---:|
| task category | `all-mpnet-base-v2` | `10` | flat logistic | `0.3396` |
| aspect category | `bge-small-en-v1.5` | `1` | flat logistic | `0.3078` |
| evidence type | `bge-small-en-v1.5` | `0` | flat logistic | `0.3675` |
| polarity | `bge-small-en-v1.5` | `3` | ordinal thresholds | `0.3048` |
| firsthand | `bert-base-uncased` | `0` | binary logistic | `0.7647` |

Strict mean macro F1 improved from `0.3954` to `0.4169`, an absolute gain of
`0.0215` and a relative gain of approximately `5.4%`.

The publication-readiness gate still rejects the candidate because the current
evaluation has only `69` thread groups and task, aspect, evidence, and polarity
remain below the minimum per-field threshold.

## Artifacts

- Strict result:
  `datasets/training/routed_rubric_thread_grouped_2026-06-07.json`
- Trained candidate:
  `datasets/training/routed_rubric_round2_2026-06-07_model.joblib`
- Publication readiness:
  `datasets/evaluation/routed_rubric_publication_readiness_2026-06-07.json`
- Rejected hierarchy result:
  `datasets/training/structured_bge_thread_grouped_2026-06-07.json`
- Constraint comparison:
  `datasets/training/constraint_bakeoff_bge_thread_grouped_2026-06-07.json`
- Rubric comparisons:
  `datasets/training/rubric_selected_bge_thread_grouped_2026-06-07.json`
  `datasets/training/rubric_selected_bert_base_thread_grouped_2026-06-07.json`
  `datasets/training/rubric_selected_mpnet_thread_grouped_2026-06-07.json`

## Decision

Use the routed rubric classifier as the next candidate for review assistance
and active learning. It is not yet publication-grade automatic labeling:
aspect, evidence, and polarity remain below acceptable publication thresholds.
Do not repeatedly evaluate it against the permanently isolated gold set.
