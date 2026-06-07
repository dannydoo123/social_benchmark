# Gold Evaluation Report

Date: 2026-06-06

## Gold Set

Reviewed input:

```text
datasets/queues/gold_eval_review_2026-06-06_filled.csv
```

- Reviewed rows: `60`
- Accepted observations: `46`
- Excluded observations: `14`
- Held-out Hacker News threads: `5`
- Training overlap: `0`

The gold set was not merged into training and the evaluated model was not retrained on gold.

## Evaluated Model

```text
datasets/training/bge_small_evidence_only_2026-06-06_merged_model.joblib
```

Evaluation artifact:

```text
datasets/evaluation/bge_small_evidence_only_2026-06-06_gold_eval.json
```

## Results

Mean macro F1: `0.360`

| Field | Accuracy | Macro F1 |
| --- | ---: | ---: |
| task category | `0.500` | `0.366` |
| aspect category | `0.435` | `0.333` |
| evidence type | `0.413` | `0.182` |
| polarity | `0.304` | `0.261` |
| firsthand | `0.761` | `0.661` |

## Main Errors

- Comparative evaluations are commonly predicted as firsthand usage or benchmark anecdotes.
- Strong complaints with polarity `-2` are commonly softened to `-1`.
- Task-fit claims are commonly routed to hallucination safety or developer ergonomics.
- Writing and long-context tasks are commonly routed to research.
- False firsthand claims remain overpredicted.

## Interpretation

This is a targeted stress-test set, not a representative random production sample:

- `21` of `46` accepted observations have polarity `-2`.
- No accepted observations have neutral polarity.
- The batch emphasizes agents, comparisons, failures, value, and hallucination claims.

The result is useful for identifying weaknesses but should not be reported as general production accuracy.

## Identity Audit

Thirteen accepted rows did not explicitly confirm a human model ID. Several notes indicate that the machine-labeled model may be incidental or that the actual agent is unspecified.

Audit queue:

```text
datasets/queues/gold_eval_identity_audit_2026-06-06.csv
```

The completed audit confirmed that all `13` rows intentionally lack a canonical model ID. They are product/provider-level or unspecified-agent evidence and must remain excluded from model-specific benchmark scores.

Model-scoring-eligible gold subset:

```text
datasets/evaluation/gold_eval_model_scoring_eligible_2026-06-06.csv
```

- Eligible rows: `33`
- Identity-audited rows excluded from model-specific scoring: `13`

These identity decisions do not change the five extractor-field metrics above.

## Decision

- Keep the gold set permanently excluded from training.
- Do not tune repeatedly against this gold set.
- Prioritize new training labels for comparative evaluation, strong negative polarity, task fit, writing, and long-context work.
- Build a second representative holdout later using a random thread sample.
