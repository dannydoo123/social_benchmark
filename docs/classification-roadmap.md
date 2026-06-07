# Classification Roadmap

## Goal

Turn the current weak Naive Bayes helper into a context-aware, review-driven labeling system that can support high-quality benchmark construction.

## Planned Steps

1. Expand the raw HN corpus
- Fetch more Hacker News items and comments.
- Regenerate observations from the raw source set.

2. Improve span quality first
- Keep the rule-based extractor for candidate spans.
- Strip quote-only, question-only, title-only, and install/instruction noise.
- Keep a full source-context sidecar for review.

3. Train on reviewed labels, not raw guesses
- Build training JSONL from human-reviewed CSVs.
- Use the extracted span plus the full source context.
- Keep reviewed rows and exclusions out of training.

4. Replace the weak baseline with a stronger local classifier
- Move from plain Naive Bayes to a more feature-rich local model.
- Use metadata and context, not just the snippet text.
- Predict each field separately:
  - `task_category`
  - `aspect_category`
  - `evidence_type`
  - `polarity_score`
  - `firsthand_flag`

5. Add confidence gating and abstention
- Only surface classifier suggestions when field accuracy is good enough.
- Do not force labels on uncertain rows.
- Use disagreement count to prioritize review.

6. Use active learning
- Review rows where the machine and classifier disagree.
- Prefer low-confidence or high-disagreement rows.
- Retrain after each batch.

7. Iterate until the queue is mostly edge cases
- When the remaining rows are mostly ambiguous, stop extractor surgery.
- At that point the bottleneck is reviewed data, not the model code.

## Current Implementation Anchors

- Extractor logic: `src/social_benchmark/pipeline/extractors/rules.py`
- Training data builder: `src/social_benchmark/pipeline/training_data.py`
- Classifier: `src/social_benchmark/pipeline/local_classifier.py`
- Queue export and review flow: `src/social_benchmark/pipeline/labeling.py`

## Model Upgrade Consideration

- The current local Naive Bayes classifier is a baseline, not the final choice.
- We are considering stronger open-source alternatives, including Hugging Face-backed models such as SetFit, sentence-transformer embeddings plus a linear head, or a ModernBERT-based encoder.
- Any replacement should preserve the same review artifacts, labels, and extraction pipeline so prior work remains reusable.
- Model comparisons should be run against the same reviewed HN slices before swapping the active backend.

## Current Hugging Face Path

- `embed-jsonl` writes local embedding vectors for reviewed training rows or extracted observations.
- `cluster-embeddings` groups near-duplicate rows by cosine similarity for deduplication and manipulation review.
- `train-hf-classifier` trains one logistic classifier head per target field on local Hugging Face embeddings.
- `evaluate-hf-classifier` evaluates the embedding backend with the same repeated holdout style used by the sklearn baseline.
- `compare-classifiers` writes a single report comparing the current Naive Bayes baseline, the sklearn TF-IDF baseline, and the Hugging Face embedding backend when available.
- `train-high-precision-classifier --precision-first --skip-hf` trains the selected agreement-gated production candidate. It uses Naive Bayes plus sklearn consensus and abstains when confidence is insufficient.

The first recommended model remains `sentence-transformers/all-MiniLM-L6-v2` because it is small enough for local iteration. Larger encoder models should only replace it after the comparison report shows a measurable gain on the same reviewed HN slices.

## Reviewed HN Experiment

The first local comparison used `datasets/training/hn_manual_training.jsonl` with 324 reviewed examples.

- The sklearn TF-IDF logistic backend was the strongest standalone classifier.
- `sentence-transformers/all-MiniLM-L6-v2` added a different signal but was weaker as a standalone classifier on this dataset.
- Adding MiniLM to the ensemble reduced precision and coverage on most fields, so it remains experimental.
- The selected production candidate is a Naive Bayes plus sklearn consensus model with field-specific thresholds:
  - `task_category`: `0.62`
  - `aspect_category`: `0.68`
  - `evidence_type`: `0.68`
  - `polarity_score`: `0.78`
  - `firsthand_flag`: `0.78`

The selected ensemble is intentionally precision-first. It should abstain and send uncertain rows to review rather than force labels.

## Grouped Frozen-Embedding Bake-Off

The stricter source-item-grouped bake-off is documented in:

```text
analysis/extractor-bakeoff-report-2026-06-01.md
```

The strongest completed frozen checkpoint is `BAAI/bge-small-en-v1.5` with augmented evidence input. `sentence-transformers/all-mpnet-base-v2` with evidence-only input is the secondary candidate. A bounded GPU SetFit pilot ran successfully but did not beat frozen embeddings, so the current production candidate is the frozen BGE Small augmented classifier.

## Operating Rule

- Extract better.
- Train on reviewed data.
- Use context.
- Predict fields separately.
- Abstain when unsure.
- Retrain on disagreements.

## Routed Rubric Architecture

The current selected experimental candidate is the routed rubric classifier:

- MPNet plus strong task-label rubric features for `task_category`.
- BGE Small plus light rubric features for `aspect_category`.
- BGE Small without rubric features for `evidence_type`.
- BGE Small plus an ordinal threshold head for `polarity_score`.
- Raw BERT mean-pooled embeddings for the specialized `firsthand_flag` head.

This improved eight-run thread-grouped mean macro F1 from `0.3954` to `0.4169`.
Hierarchical routing and hard constraint overrides were tested but rejected as
active prediction behavior because they propagated upstream mistakes.

Use:

```powershell
python -m social_benchmark.pipeline.cli run-routed-rubric-bakeoff --training TRAINING.jsonl --out EVAL.json --group-field thread_id --embedding-cache-dir datasets/training/embedding_cache
python -m social_benchmark.pipeline.cli train-routed-rubric-classifier --training TRAINING.jsonl --model-out MODEL.joblib
python -m social_benchmark.pipeline.cli assess-publication-readiness --evaluation EVAL.json --out READINESS.json
```
