# Round-Two Targeted Review Instructions

## Files

Review:

```text
datasets/queues/targeted_training_review_round2_2026-06-06.csv
```

Context sidecar:

```text
datasets/queues/targeted_training_review_round2_2026-06-06_context.jsonl
```

- Rows: `300`
- Threads: `29`
- Maximum rows from one thread: `35`
- Previous-review overlap: `0`
- Gold-thread overlap: `0`

This is training data. Do not add rows from the gold evaluation files.

## Primary Goal

Correct the failure modes found in the held-out gold evaluation:

1. Comparative evaluations mislabeled as firsthand usage or benchmark anecdotes.
2. Strong complaints labeled too mildly.
3. Task-fit claims confused with hallucination safety or developer ergonomics.
4. Writing and long-context work confused with research.
5. Secondhand claims incorrectly marked firsthand.

## Review Procedure

For every row, decide in this order:

1. Should the row be excluded from scoring?
2. Which model or product is actually being judged?
3. What is the primary task?
4. What is the primary quality aspect?
5. What form of evidence supports the claim?
6. Is the author reporting direct experience?
7. What is the polarity and its strength?

When the visible span is unclear, find the matching `review_id` in the context sidecar.

## Exclusion

Exclude rows that do not contain a usable quality judgment:

- release or adoption facts without a quality judgment
- questions without an answer or conclusion
- instructions, prompts, or feature descriptions
- general AI discussion
- a model mentioned only incidentally
- unsupported speculation
- duplicate or extremely weak evidence

Prefer exclusion over forcing an ambiguous row into a scoring label.

## Model Identity

Label the model being judged, not every model mentioned.

If the text judges a product but does not identify the underlying model:

```text
human_provider_id = known provider, if clear
human_model_id = blank
human_product_id = product being judged
```

If a model is only incidental comparison context, do not assign the claim to that model.

## Comparative Evaluation

Use `comparative_evaluation` when the central evidence is a comparison:

- better than
- worse than
- compared with
- versus
- preferred one model after testing both

Direct comparison remains `comparative_evaluation` even when it is firsthand.

Example:

```text
I used Claude and GPT for three months; Claude was better at refactoring.

task = coding
aspect = task_fit
evidence_type = comparative_evaluation
firsthand = true
polarity = 1 or 2
```

Do not use `benchmark_anecdote` merely because the author tested multiple models.

## Polarity

Use the strength of the author's actual judgment:

- `2`: strong endorsement, clearly best, major success
- `1`: mild praise or useful result
- `0`: neutral, mixed, descriptive, or unclear
- `-1`: mild complaint or limited weakness
- `-2`: severe failure, repeated failure, dangerous behavior, unusable result, or strong rejection

Repeated confident errors, destructive behavior, major regressions, and "unusable" claims are usually `-2`.

Do not soften a strong complaint to `-1`.

## Task Fit Versus Other Aspects

Use `task_fit` when the claim is primarily whether the model performs a task well.

```text
It writes poor prose.
aspect = task_fit
task = writing
```

Use `hallucination_safety` only for fabricated or unsupported content:

```text
It invented citations and nonexistent APIs.
aspect = hallucination_safety
```

Use `developer_ergonomics` for the surrounding technical interface:

```text
The API times out and the SDK is broken.
aspect = developer_ergonomics
```

Use `trust_reliability` for unpredictable, unsafe, or dangerous actions:

```text
The agent tried to retrieve production secrets.
aspect = trust_reliability
```

## Writing, Research, And Long Context

Use `writing` when the output is prose, editing, rewriting, tone, or composition.

Use `research` when the task is finding, checking, or synthesizing information.

Use `long_context` only when the claim specifically depends on processing or remembering large documents, repositories, or conversations.

Do not label ordinary writing as research merely because it involves facts.

## Firsthand

Mark `true` only when the author directly used, tested, deployed, paid for, or personally evaluated the system.

True:

- I used it
- we deployed it
- I compared both models
- it failed during my test

False:

- people say
- a company reportedly used it
- release notes claim
- I think it probably works

`comparative_evaluation`, `integration_failure`, and `pricing_value_comment` can each be either firsthand or secondhand.

## Evidence Type Priority

Choose the most specific evidence form:

- worsening over time or broken previous behavior: `bug_regression_report`
- API, SDK, timeout, rate-limit, or deployment failure: `integration_failure`
- direct comparison: `comparative_evaluation`
- pricing, quota, subscription, or value judgment: `pricing_value_comment`
- reaction to release news: `release_update_reaction`
- prompt/test anecdote without direct comparison: `benchmark_anecdote`
- direct usage without a more specific form: `firsthand_usage`
- secondhand or unsupported claim: `hearsay`

## Final Check

Before accepting each row, verify:

1. The row contains actual quality evidence.
2. The model or product being judged is correctly identified.
3. The task describes what the user was doing.
4. The aspect describes what quality dimension is judged.
5. The evidence type describes how the claim is supported.
6. Firsthand is based on direct use, not first-person opinion.
7. Strong failures receive strong negative polarity.
8. Another careful reviewer would likely make the same decision.

## Review UI

```powershell
cd web
npm run dev
```

Load the round-two CSV and export the completed result with a distinct filename such as:

```text
targeted_training_review_round2_2026-06-06_filled.csv
```
