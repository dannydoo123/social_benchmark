# Extractor Classifier Bake-Off Report

Date: 2026-06-01

## Executive Summary

The extractor classifier bake-off now uses source-item-grouped holdouts rather than row-level random holdouts. This prevents multiple extracted rows from the same Hacker News item from crossing the train/test boundary.

The strongest completed frozen-embedding configuration is:

```text
BAAI/bge-small-en-v1.5
input mode: augmented evidence span
architecture: frozen sentence-transformer embedding + separate logistic head per field
mean macro F1: 0.425
```

The strongest TF-IDF baseline reached `0.358` mean macro F1 under the same grouped evaluation. BGE Small therefore improved mean macro F1 by `0.067` absolute, or approximately `18.7%` relative.

SetFit fine-tuning was installed and attempted, but CPU-local runtime was not practical. A full BGE pass exceeded one hour, and a bounded five-step one-field pilot still ran for more than four minutes before being stopped. SetFit should be evaluated on a GPU-backed environment or a deliberately sampled pilot before it becomes part of the local iteration loop.

## Dataset

Training artifact:

```text
datasets/training/hn_manual_training.jsonl
```

Reviewed examples: `324`

Current grouped evaluation key:

```text
source_item_id
```

The training artifact does not currently preserve `thread_id`. The next stricter evaluation should carry `thread_id` into training JSONL and group by thread. Duplicate-cluster grouping should also be added after embedding clusters are reviewed.

## Architecture

Each backend predicts the same five fields independently:

- `task_category`
- `aspect_category`
- `evidence_type`
- `polarity_score`
- `firsthand_flag`

Frozen embedding architecture:

```text
evidence span
  -> optional context and metadata augmentation
  -> sentence-transformer embedding
  -> one logistic classifier head per field
```

TF-IDF baseline:

```text
evidence span
  -> optional context and metadata augmentation
  -> word and character TF-IDF features
  -> one logistic classifier head per field
```

## Completed Frozen-Model Results

All rows below use the same four source-item-grouped holdout splits.

| Rank | Backend | Input Mode | Mean Macro F1 |
| ---: | --- | --- | ---: |
| 1 | `BAAI/bge-small-en-v1.5` | augmented | `0.425` |
| 2 | `BAAI/bge-small-en-v1.5` | evidence only | `0.416` |
| 3 | `sentence-transformers/all-mpnet-base-v2` | evidence only | `0.410` |
| 4 | `sentence-transformers/all-mpnet-base-v2` | augmented | `0.400` |
| 5 | `sentence-transformers/all-MiniLM-L6-v2` | evidence only | `0.398` |
| 6 | `sentence-transformers/all-MiniLM-L6-v2` | augmented | `0.395` |
| 7 | TF-IDF logistic | augmented | `0.358` |
| 8 | TF-IDF logistic | evidence only | `0.338` |

### Per-Field Macro F1

| Backend | Mode | Task | Aspect | Evidence | Polarity | Firsthand |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BGE Small | augmented | `0.265` | `0.329` | `0.406` | `0.438` | `0.690` |
| BGE Small | evidence only | `0.274` | `0.307` | `0.382` | `0.451` | `0.667` |
| MPNet | evidence only | `0.289` | `0.325` | `0.360` | `0.361` | `0.715` |
| MPNet | augmented | `0.280` | `0.306` | `0.359` | `0.372` | `0.681` |
| MiniLM | evidence only | `0.270` | `0.333` | `0.369` | `0.359` | `0.657` |
| MiniLM | augmented | `0.243` | `0.340` | `0.370` | `0.358` | `0.664` |
| TF-IDF | augmented | `0.193` | `0.327` | `0.273` | `0.285` | `0.713` |
| TF-IDF | evidence only | `0.210` | `0.288` | `0.252` | `0.256` | `0.681` |

## Interpretation

### Selected Frozen Checkpoint

Use `BAAI/bge-small-en-v1.5` with augmented input as the default frozen embedding candidate.

Reasons:

- strongest overall grouped mean macro F1
- strongest grouped evidence-type result
- strong polarity classification
- small enough for practical local inference

### Secondary Candidate

Keep `sentence-transformers/all-mpnet-base-v2` with evidence-only input as the secondary candidate.

Reasons:

- second-best checkpoint family after BGE Small
- best task-category result in the matrix
- best firsthand result in the matrix

MPNet is larger and slower, so it should only be retained if field-specific routing or an ensemble materially improves validated precision.

### Input Augmentation

Augmentation helped BGE Small but hurt MPNet and MiniLM slightly. Input formatting should remain a checkpoint-level decision rather than a universal setting.

### Why Earlier Results Changed

Earlier random row-level holdouts favored TF-IDF. The grouped evaluation is stricter and more credible because it prevents rows from the same source item from leaking into both train and test. Under grouped evaluation, every sentence-transformer checkpoint outperformed TF-IDF on mean macro F1.

## SetFit Status

SetFit `1.1.3` was installed and a reusable CLI runner was added.

CPU-local attempted runs:

1. BGE Small across all five fields with one epoch and four pair-generation iterations:
   - exceeded one hour on CPU
   - stopped
   - no completed result

2. BGE Small `firsthand_flag` pilot with one epoch, one pair-generation iteration, and `max_steps=5`:
   - still exceeded four minutes on CPU
   - stopped
   - no completed result

Conclusion:

- SetFit remains a valid next experiment.
- It is not appropriate for repeated CPU-local iteration on this machine.
- Run SetFit on GPU with bounded steps and incremental outputs.
- Start with BGE Small augmented and MPNet evidence-only.

## GPU SetFit Update

After the NVIDIA driver and CUDA-enabled PyTorch were installed, PyTorch reported:

```text
torch: 2.7.1+cu118
cuda available: true
gpu: NVIDIA GeForce GTX 1660 Ti
```

A bounded BGE Small SetFit pilot was run on `firsthand_flag`:

```text
checkpoint: BAAI/bge-small-en-v1.5
input mode: augmented
field: firsthand_flag
epochs: 1
iterations: 1
max_steps: 5
test examples: 98
macro F1: 0.640
```

Runtime was still high for the expected gain: five embedding-training steps took roughly 2.4 minutes of training time, plus setup overhead. The earlier uncapped one-field BGE pilot reached `0.662` macro F1 but took roughly 8.5 minutes of embedding training.

For comparison, frozen embeddings already reached stronger grouped `firsthand_flag` macro F1:

- BGE Small augmented: `0.690`
- MPNet evidence-only: `0.715`

Conclusion:

- GPU SetFit is now technically runnable.
- The first bounded SetFit result did not beat frozen embeddings.
- Continue with frozen BGE Small as the production candidate.
- Revisit SetFit only after more labels are available or on a faster GPU.

## Precision-First Production Use

The project also includes a Naive Bayes plus sklearn agreement-gated ensemble. It abstains when confidence or agreement is insufficient. This remains useful for producing conservative review suggestions while the BGE-based extractor is integrated.

The ensemble should not be treated as the final extractor selection until it is evaluated using the same grouped splits as the frozen checkpoint matrix.

## Generated Artifacts

Primary completed experiment:

```text
datasets/training/frozen_embedding_bakeoff_grouped.json
```

Existing model and analysis artifacts:

```text
datasets/training/bge_small_augmented_model.joblib
datasets/training/classifier_comparison.json
datasets/training/hf_embedding_model.joblib
datasets/training/high_precision_ensemble.joblib
datasets/training/hn_manual_embeddings.jsonl
datasets/training/hn_manual_embedding_clusters.json
datasets/training/local_nb_model.json
datasets/training/sklearn_model.joblib
```

## Recommended Next Steps

1. Carry `thread_id` into reviewed training JSONL and rerun grouped evaluation by thread.
2. Review embedding duplicate clusters and add duplicate-cluster-grouped evaluation.
3. Train a production BGE Small augmented classifier artifact with separate logistic heads.
4. Run GPU-backed SetFit pilots for BGE Small augmented and MPNet evidence-only only if runtime is acceptable and early pilots beat frozen embeddings.
5. Compare grouped precision, macro F1, coverage, inference latency, and memory footprint.
6. Freeze extractor version `v1` only after the grouped evaluation and abstention policy are finalized.
