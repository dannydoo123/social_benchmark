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

## Operating Rule

- Extract better.
- Train on reviewed data.
- Use context.
- Predict fields separately.
- Abstain when unsure.
- Retrain on disagreements.
