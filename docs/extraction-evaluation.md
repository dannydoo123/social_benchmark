# Extraction iteration record

The v2 extractor remains a candidate generator. Neither the tests nor the small assistant-labeled diagnostic establish high accuracy on the public data distribution. No labels were automatically approved, and no model was trained on these candidates.

## What changed

- Sentence/contrast/aspect boundaries preserve exact offsets and decimal version numbers. Different aspects can receive opposite sentiment; mixed sentiment on the same aspect abstains.
- Adjacent same-post pronouns and explicit elliptical predicates can retain a subject. Paragraph breaks, intervening subjects, questions and ambiguous references clear it.
- Direct-parent pronouns require a single explicit model in the actual parent within the same thread. They do not inherit a model from an entire discussion. Published candidates include a parent source link and context excerpt.
- Named directional comparisons produce relative preferences, marked separately from absolute praise. Unnamed comparison subjects and ambiguous comparisons abstain.
- Negation, cost vocabulary, tool-calling categories, quota drain, and code-generation vocabulary have additional handling. Generic testing is not automatically coding.
- Code, explicit quotations, benchmark tables, selected diagnostic metadata, hypothetical language, and recognized sarcasm are withheld.
- Evidence ingestion now preserves paragraph, quote and code boundaries. Previously flattened records cannot be reconstructed without refetching. In particular, collapsed quote lines cause conservative false negatives.

## Iterations and evidence

The frozen sample is `datasets/evaluation/claim_extraction_real.json`: 50 seeded short real documents containing model mentions, from the local non-quarantined archive. Labels were authored by the implementing assistant before modifying the extractor. The evaluation covers specific versions only. This sampling underrepresents long posts, implicit-only replies and multilingual text, and is not independent human review.

Development: 30 documents, 4 ambiguous cases excluded, 5 expected labels. Baseline: 2 true positives, 2 false positives, 3 false negatives (50% precision / 40% recall). Iteration 1: 3 true positives, 0 false positives, 2 false negatives (100% observed precision / 60% recall). Both misses occur in a flattened quote/comment boundary. Do not extrapolate 100% precision from three predictions.

Iteration 2 fixed a synthetic adversarial failure where “good and bad for coding” was incorrectly split into two aspects. The initial held-out check used 20 documents, of which 9 were explicitly ambiguous/excluded: 3 true positives, 0 false positives and 0 false negatives in the 11 scored documents. This has too few positive labels for a meaningful accuracy claim. The holdout also does not test parent-reply attribution; those cases currently have synthetic regression coverage only.

Iteration 3 inspected target-model output from the full archive and found new failures despite the small diagnostic passing: saved metadata treated as praise, quota drain treated as speed praise, benchmark tables treated as experience, and unnamed comparison targets treated as subjects. These received fixes and regression tests. This is development feedback, not a second independent holdout.

Reproduce diagnostics with `python scripts/evaluate_claim_extraction.py --split development` or `--split holdout`. Run the full suite with `python -m unittest discover -s tests`.

## Next accuracy gate

`python scripts/build_extraction_review_queue.py` creates 200 pending documents locally: 150 candidate documents sampled across source/attribution/category/polarity strata, plus 50 non-candidate documents to detect misses. This is a deliberately stratified diagnostic sample; its pooled metrics are not population estimates. Independently label all claims and ambiguous cases, keeping threads together when splitting development and evaluation. Report precision and recall by category and attribution, uncertainty intervals, exclusion counts, and model-version errors. Do not optimize on the final evaluation split. Recollect quote-preserving source text where needed.

The public UI continues to report zero independently validated chunks. Increasing record or candidate counts is not an accuracy criterion.
