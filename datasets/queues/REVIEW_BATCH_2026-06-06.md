# Review Batch 2026-06-06

## Review Order

Review the training batch first:

```text
datasets/queues/targeted_training_review_2026-06-06.csv
```

- Rows: `240`
- Threads: `35`
- Ordered with sparse classes, model disagreements, extreme polarity, and likely failures first.
- After review, this batch may be converted into training data.

Review the gold evaluation batch separately:

```text
datasets/queues/gold_eval_review_2026-06-06.csv
```

- Rows: `60`
- Threads: `5`
- Every thread is excluded from the training batch.
- Never merge this batch into training data. Use it only for final model evaluation and threshold selection.

## Context Sidecars

Matching full-context files:

```text
datasets/queues/targeted_training_review_2026-06-06_context.jsonl
datasets/queues/gold_eval_review_2026-06-06_context.jsonl
```

Use the sidecar when the evidence span is unclear. Match records using `review_id`.

## Labeling Priorities

Pay extra attention to:

1. Whether the row should be excluded from scoring entirely.
2. `bug_regression_report` and `regression_stability`.
3. Hallucination and refusal claims.
4. Direct comparisons between models.
5. Strong polarity values `-2` and `2`.
6. Whether the author directly used or tested the model.

Review according to:

```text
docs/labeling-guide.md
```

## Review UI

```powershell
cd web
npm run dev
```

Load one CSV at a time and export the reviewed result before loading the next batch.
