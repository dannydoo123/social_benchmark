from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from social_benchmark.pipeline.analysis import write_observation_report
from social_benchmark.pipeline.active_learning import export_active_learning_queue, write_fixed_model_evaluation, write_threshold_report
from social_benchmark.pipeline.clients.github import GitHubClient
from social_benchmark.pipeline.clients.hackernews import HackerNewsClient
from social_benchmark.pipeline.collection import collect_from_config, load_source_config
from social_benchmark.pipeline.classifier_experiments import DEFAULT_EMBEDDING_MODELS, run_frozen_embedding_bakeoff
from social_benchmark.pipeline.embeddings import cluster_embedding_jsonl, write_embeddings_jsonl
from social_benchmark.pipeline.extractors.rules import RuleBasedExtractor
from social_benchmark.pipeline.hf_classifier import train_hf_classifier, write_hf_evaluation
from social_benchmark.pipeline.high_precision_classifier import (
    PRECISION_FIRST_FIELD_CONFIDENCE,
    train_high_precision_classifier,
    write_high_precision_evaluation,
)
from social_benchmark.pipeline.label_feedback import apply_reviewed_labels, write_label_evaluation
from social_benchmark.pipeline.labeling import export_labeling_queue
from social_benchmark.pipeline.local_classifier import (
    TARGET_FIELDS,
    predict_jsonl,
    train_classifier,
    write_classifier_evaluation,
)
from social_benchmark.pipeline.model_comparison import compare_classifier_backends
from social_benchmark.pipeline.models import RawItem, SourcePlatform, to_jsonable
from social_benchmark.pipeline.scoring import ScoreAggregator
from social_benchmark.pipeline.routed_classifier import run_routed_rubric_bakeoff, train_routed_rubric_classifier
from social_benchmark.pipeline.publication_readiness import write_publication_readiness
from social_benchmark.pipeline.setfit_experiments import parse_checkpoint_specs, run_setfit_bakeoff
from social_benchmark.pipeline.sklearn_classifier import (
    train_sklearn_classifier,
    write_sklearn_evaluation,
    write_sklearn_variant_evaluation,
)
from social_benchmark.pipeline.storage import read_jsonl, write_jsonl
from social_benchmark.pipeline.training_data import build_training_jsonl, merge_training_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(prog="sb-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hn_parser = subparsers.add_parser("fetch-hn", help="Fetch Hacker News stories/items")
    hn_parser.add_argument("--kind", choices=["top", "new", "best", "ask"], default="top")
    hn_parser.add_argument("--limit", type=int, default=30)
    hn_parser.add_argument("--comments", type=int, default=0, help="Fetch up to this many comments per story")
    hn_parser.add_argument("--max-depth", type=int, default=8, help="Maximum comment depth to fetch")
    hn_parser.add_argument("--out", required=True)

    gh_search_parser = subparsers.add_parser("fetch-github-search", help="Search GitHub issues")
    gh_search_parser.add_argument("--query", required=True)
    gh_search_parser.add_argument("--limit", type=int, default=30)
    gh_search_parser.add_argument("--out", required=True)

    gh_repo_parser = subparsers.add_parser("fetch-github-repo", help="Fetch GitHub repo issues")
    gh_repo_parser.add_argument("--owner", required=True)
    gh_repo_parser.add_argument("--repo", required=True)
    gh_repo_parser.add_argument("--state", default="all")
    gh_repo_parser.add_argument("--since")
    gh_repo_parser.add_argument("--limit", type=int, default=30)
    gh_repo_parser.add_argument("--include-comments", action="store_true")
    gh_repo_parser.add_argument("--out", required=True)

    config_parser = subparsers.add_parser("fetch-config", help="Fetch sources listed in a JSON config")
    config_parser.add_argument("--config", default="config/sources.json")
    config_parser.add_argument("--out", required=True)

    process_parser = subparsers.add_parser("process-jsonl", help="Extract observations and scores from raw JSONL")
    process_parser.add_argument("--raw", required=True)
    process_parser.add_argument("--observations-out", required=True)
    process_parser.add_argument("--scores-out", required=True)

    labels_parser = subparsers.add_parser("export-labels", help="Export uncertain observations for manual labeling")
    labels_parser.add_argument("--observations", required=True)
    labels_parser.add_argument("--out", required=True)
    labels_parser.add_argument("--max-rows", type=int, default=200)
    labels_parser.add_argument("--confidence-below", type=float, default=0.72)
    labels_parser.add_argument("--skip-neutral", action="store_true")
    labels_parser.add_argument("--model", help="Optional local classifier model to add review suggestions")
    labels_parser.add_argument("--raw", help="Optional raw JSONL path used to export review context sidecar")
    labels_parser.add_argument("--context-out", help="Optional JSONL sidecar with full source context for each review row")
    labels_parser.add_argument("--exclude-reviewed", action="append", default=[], help="Reviewed CSV file to exclude from future queues; may be passed multiple times")

    report_parser = subparsers.add_parser("report-observations", help="Summarize extraction output quality")
    report_parser.add_argument("--observations", required=True)
    report_parser.add_argument("--out", required=True)

    training_parser = subparsers.add_parser("build-training-data", help="Convert reviewed label CSV into JSONL")
    training_parser.add_argument("--labels", required=True)
    training_parser.add_argument("--out", required=True)
    training_parser.add_argument("--context", help="Optional review-context JSONL matched by review_id")

    merge_training_parser = subparsers.add_parser("merge-training-data", help="Merge and deduplicate training JSONL files")
    merge_training_parser.add_argument("--input", action="append", required=True)
    merge_training_parser.add_argument("--out", required=True)
    merge_training_parser.add_argument("--exclude-thread-id", action="append", default=[])

    eval_labels_parser = subparsers.add_parser("evaluate-labels", help="Compare machine labels with reviewed CSV labels")
    eval_labels_parser.add_argument("--labels", required=True)
    eval_labels_parser.add_argument("--out", required=True)

    apply_labels_parser = subparsers.add_parser("apply-labels", help="Apply reviewed CSV labels to observation JSONL")
    apply_labels_parser.add_argument("--observations", required=True)
    apply_labels_parser.add_argument("--labels", required=True)
    apply_labels_parser.add_argument("--out", required=True)

    train_parser = subparsers.add_parser("train-local-classifier", help="Train the dependency-free local NB baseline")
    train_parser.add_argument("--training", required=True)
    train_parser.add_argument("--model-out", required=True)

    eval_classifier_parser = subparsers.add_parser("evaluate-local-classifier", help="Leave-one-out eval for local NB baseline")
    eval_classifier_parser.add_argument("--training", required=True)
    eval_classifier_parser.add_argument("--out", required=True)

    train_sklearn_parser = subparsers.add_parser("train-sklearn-classifier", help="Train TF-IDF logistic field classifiers")
    train_sklearn_parser.add_argument("--training", required=True)
    train_sklearn_parser.add_argument("--model-out", required=True)
    train_sklearn_parser.add_argument("--runs", type=int, default=8)

    eval_sklearn_parser = subparsers.add_parser("evaluate-sklearn-classifier", help="Repeated holdout eval for TF-IDF logistic classifiers")
    eval_sklearn_parser.add_argument("--training", required=True)
    eval_sklearn_parser.add_argument("--out", required=True)
    eval_sklearn_parser.add_argument("--runs", type=int, default=8)

    eval_sklearn_variants_parser = subparsers.add_parser("evaluate-sklearn-variants", help="Compare TF-IDF feature representations")
    eval_sklearn_variants_parser.add_argument("--training", required=True)
    eval_sklearn_variants_parser.add_argument("--out", required=True)
    eval_sklearn_variants_parser.add_argument("--runs", type=int, default=8)

    train_hf_parser = subparsers.add_parser("train-hf-classifier", help="Train embedding-backed field classifiers with a local Hugging Face model")
    train_hf_parser.add_argument("--training", required=True)
    train_hf_parser.add_argument("--model-out", required=True)
    train_hf_parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    train_hf_parser.add_argument("--runs", type=int, default=8)
    train_hf_parser.add_argument("--text-mode", default="augmented", choices=["evidence_only", "augmented"])

    eval_hf_parser = subparsers.add_parser("evaluate-hf-classifier", help="Evaluate embedding-backed field classifiers")
    eval_hf_parser.add_argument("--training", required=True)
    eval_hf_parser.add_argument("--out", required=True)
    eval_hf_parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    eval_hf_parser.add_argument("--runs", type=int, default=8)
    eval_hf_parser.add_argument("--text-mode", default="augmented", choices=["evidence_only", "augmented"])

    embed_parser = subparsers.add_parser("embed-jsonl", help="Write local Hugging Face embeddings for JSONL rows")
    embed_parser.add_argument("--input", required=True)
    embed_parser.add_argument("--out", required=True)
    embed_parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    embed_parser.add_argument("--backend", default="auto", choices=["auto", "sentence-transformers", "transformers"])
    embed_parser.add_argument("--batch-size", type=int, default=32)

    cluster_parser = subparsers.add_parser("cluster-embeddings", help="Greedily cluster embedding JSONL rows by cosine similarity")
    cluster_parser.add_argument("--embeddings", required=True)
    cluster_parser.add_argument("--out", required=True)
    cluster_parser.add_argument("--threshold", type=float, default=0.92)

    compare_parser = subparsers.add_parser("compare-classifiers", help="Compare current classifier backends on the same reviewed training data")
    compare_parser.add_argument("--training", required=True)
    compare_parser.add_argument("--out", required=True)
    compare_parser.add_argument("--runs", type=int, default=8)
    compare_parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    compare_parser.add_argument("--skip-hf", action="store_true")
    compare_parser.add_argument("--skip-ensemble", action="store_true")
    compare_parser.add_argument("--ensemble-min-confidence", type=float, default=0.62)
    compare_parser.add_argument("--ensemble-min-agreement", type=int, default=2)

    bakeoff_parser = subparsers.add_parser("run-frozen-embedding-bakeoff", help="Compare TF-IDF and frozen embedding checkpoints with grouped holdouts")
    bakeoff_parser.add_argument("--training", required=True)
    bakeoff_parser.add_argument("--out", required=True)
    bakeoff_parser.add_argument("--runs", type=int, default=4)
    bakeoff_parser.add_argument("--embedding-model", action="append", default=[])
    bakeoff_parser.add_argument("--group-field", default="source_item_id", choices=["source_item_id", "thread_id"])
    bakeoff_parser.add_argument("--embedding-cache-dir", help="Optional directory for reusable bake-off embedding vectors")

    routed_bakeoff_parser = subparsers.add_parser("run-routed-rubric-bakeoff", help="Evaluate specialized per-field encoders and rubric features")
    routed_bakeoff_parser.add_argument("--training", required=True)
    routed_bakeoff_parser.add_argument("--out", required=True)
    routed_bakeoff_parser.add_argument("--runs", type=int, default=8)
    routed_bakeoff_parser.add_argument("--group-field", default="thread_id", choices=["source_item_id", "thread_id"])
    routed_bakeoff_parser.add_argument("--embedding-cache-dir")

    routed_train_parser = subparsers.add_parser("train-routed-rubric-classifier", help="Train the selected specialized per-field classifier")
    routed_train_parser.add_argument("--training", required=True)
    routed_train_parser.add_argument("--model-out", required=True)

    publication_parser = subparsers.add_parser("assess-publication-readiness", help="Apply minimum publication-quality gates to an evaluation")
    publication_parser.add_argument("--evaluation", required=True)
    publication_parser.add_argument("--out", required=True)

    setfit_parser = subparsers.add_parser("run-setfit-bakeoff", help="Fine-tune top sentence-transformer checkpoints with SetFit on grouped holdouts")
    setfit_parser.add_argument("--training", required=True)
    setfit_parser.add_argument("--out", required=True)
    setfit_parser.add_argument("--checkpoint", action="append", default=[], help="CHECKPOINT|evidence_only or CHECKPOINT|augmented")
    setfit_parser.add_argument("--field", action="append", default=[])
    setfit_parser.add_argument("--epochs", type=int, default=1)
    setfit_parser.add_argument("--iterations", type=int, default=4)
    setfit_parser.add_argument("--batch-size", type=int, default=16)
    setfit_parser.add_argument("--max-steps", type=int, default=-1)
    setfit_parser.add_argument("--group-field", default="source_item_id", choices=["source_item_id", "thread_id"])

    threshold_parser = subparsers.add_parser("threshold-report", help="Measure precision and coverage by confidence threshold")
    threshold_parser.add_argument("--training", required=True)
    threshold_parser.add_argument("--model", required=True)
    threshold_parser.add_argument("--out", required=True)

    gold_eval_parser = subparsers.add_parser("evaluate-fixed-classifier", help="Evaluate a trained classifier on untouched labeled JSONL")
    gold_eval_parser.add_argument("--evaluation", required=True)
    gold_eval_parser.add_argument("--model", required=True)
    gold_eval_parser.add_argument("--out", required=True)

    active_parser = subparsers.add_parser("export-active-learning", help="Export rows prioritized by model disagreement and low confidence")
    active_parser.add_argument("--input", required=True)
    active_parser.add_argument("--model", action="append", required=True)
    active_parser.add_argument("--out", required=True)
    active_parser.add_argument("--max-rows", type=int, default=200)

    train_ensemble_parser = subparsers.add_parser("train-high-precision-classifier", help="Train an agreement-gated ensemble for higher precision")
    train_ensemble_parser.add_argument("--training", required=True)
    train_ensemble_parser.add_argument("--model-out", required=True)
    train_ensemble_parser.add_argument("--runs", type=int, default=8)
    train_ensemble_parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    train_ensemble_parser.add_argument("--skip-hf", action="store_true")
    train_ensemble_parser.add_argument("--min-confidence", type=float, default=0.62)
    train_ensemble_parser.add_argument("--min-agreement", type=int, default=2)
    train_ensemble_parser.add_argument("--precision-first", action="store_true")

    eval_ensemble_parser = subparsers.add_parser("evaluate-high-precision-classifier", help="Evaluate the agreement-gated ensemble")
    eval_ensemble_parser.add_argument("--training", required=True)
    eval_ensemble_parser.add_argument("--out", required=True)
    eval_ensemble_parser.add_argument("--runs", type=int, default=8)
    eval_ensemble_parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    eval_ensemble_parser.add_argument("--skip-hf", action="store_true")
    eval_ensemble_parser.add_argument("--min-confidence", type=float, default=0.62)
    eval_ensemble_parser.add_argument("--min-agreement", type=int, default=2)
    eval_ensemble_parser.add_argument("--precision-first", action="store_true")

    predict_parser = subparsers.add_parser("predict-local-classifier", help="Predict labels for JSONL rows with text/evidence_text")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "fetch-hn":
        _fetch_hn(args)
    elif args.command == "fetch-github-search":
        _fetch_github_search(args)
    elif args.command == "fetch-github-repo":
        _fetch_github_repo(args)
    elif args.command == "fetch-config":
        _fetch_config(args)
    elif args.command == "process-jsonl":
        _process_jsonl(args)
    elif args.command == "export-labels":
        _export_labels(args)
    elif args.command == "report-observations":
        _report_observations(args)
    elif args.command == "build-training-data":
        _build_training_data(args)
    elif args.command == "merge-training-data":
        _merge_training_data(args)
    elif args.command == "evaluate-labels":
        _evaluate_labels(args)
    elif args.command == "apply-labels":
        _apply_labels(args)
    elif args.command == "train-local-classifier":
        _train_local_classifier(args)
    elif args.command == "evaluate-local-classifier":
        _evaluate_local_classifier(args)
    elif args.command == "train-sklearn-classifier":
        _train_sklearn_classifier(args)
    elif args.command == "evaluate-sklearn-classifier":
        _evaluate_sklearn_classifier(args)
    elif args.command == "evaluate-sklearn-variants":
        _evaluate_sklearn_variants(args)
    elif args.command == "train-hf-classifier":
        _train_hf_classifier(args)
    elif args.command == "evaluate-hf-classifier":
        _evaluate_hf_classifier(args)
    elif args.command == "embed-jsonl":
        _embed_jsonl(args)
    elif args.command == "cluster-embeddings":
        _cluster_embeddings(args)
    elif args.command == "compare-classifiers":
        _compare_classifiers(args)
    elif args.command == "run-frozen-embedding-bakeoff":
        _run_frozen_embedding_bakeoff(args)
    elif args.command == "run-routed-rubric-bakeoff":
        _run_routed_rubric_bakeoff(args)
    elif args.command == "train-routed-rubric-classifier":
        _train_routed_rubric_classifier(args)
    elif args.command == "assess-publication-readiness":
        _assess_publication_readiness(args)
    elif args.command == "run-setfit-bakeoff":
        _run_setfit_bakeoff(args)
    elif args.command == "threshold-report":
        _threshold_report(args)
    elif args.command == "evaluate-fixed-classifier":
        _evaluate_fixed_classifier(args)
    elif args.command == "export-active-learning":
        _export_active_learning(args)
    elif args.command == "train-high-precision-classifier":
        _train_high_precision_classifier(args)
    elif args.command == "evaluate-high-precision-classifier":
        _evaluate_high_precision_classifier(args)
    elif args.command == "predict-local-classifier":
        _predict_local_classifier(args)


def _fetch_hn(args: argparse.Namespace) -> None:
    client = HackerNewsClient()
    id_getter = {
        "top": client.top_story_ids,
        "new": client.new_story_ids,
        "best": client.best_story_ids,
        "ask": client.ask_story_ids,
    }[args.kind]
    ids = id_getter()[: args.limit]
    raw_items = []
    if args.comments:
        for story_id in ids:
            raw_items.extend(client.fetch_story_with_comments(story_id, max_comments=args.comments, max_depth=args.max_depth))
    else:
        raw_items = client.fetch_items(ids)
    count = write_jsonl(args.out, raw_items)
    print(json.dumps({"written": count, "out": args.out}))


def _fetch_github_search(args: argparse.Namespace) -> None:
    raw_items = GitHubClient().search_issue_items(args.query, max_items=args.limit)
    count = write_jsonl(args.out, raw_items)
    print(json.dumps({"written": count, "out": args.out}))


def _fetch_github_repo(args: argparse.Namespace) -> None:
    raw_items = GitHubClient().fetch_repo_issue_items(
        args.owner,
        args.repo,
        state=args.state,
        since=args.since,
        max_issues=args.limit,
        include_comments=args.include_comments,
    )
    count = write_jsonl(args.out, raw_items)
    print(json.dumps({"written": count, "out": args.out}))


def _fetch_config(args: argparse.Namespace) -> None:
    config = load_source_config(args.config)
    raw_items = collect_from_config(config)
    count = write_jsonl(args.out, raw_items)
    print(json.dumps({"written": count, "out": args.out, "config": args.config}))


def _process_jsonl(args: argparse.Namespace) -> None:
    raw_records = read_jsonl(args.raw)
    raw_items = [_raw_item_from_record(record) for record in raw_records]
    extractor = RuleBasedExtractor.default()
    observations = []
    for item in raw_items:
        observations.extend(extractor.extract_observations(item))

    aggregator = ScoreAggregator()
    scores = aggregator.aspect_scores(observations)
    overall = aggregator.overall_scores(observations)

    write_jsonl(args.observations_out, observations)
    Path(args.scores_out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.scores_out).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "aspect_scores": to_jsonable(scores),
                "overall_scores": overall,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(
        json.dumps(
            {
                "raw_items": len(raw_items),
                "observations": len(observations),
                "aspect_scores": len(scores),
                "scores_out": args.scores_out,
            }
        )
    )


def _export_labels(args: argparse.Namespace) -> None:
    count = export_labeling_queue(
        observations_path=args.observations,
        output_path=args.out,
        max_rows=args.max_rows,
        confidence_below=args.confidence_below,
        include_neutral=not args.skip_neutral,
        classifier_model_path=args.model,
        raw_items_path=args.raw,
        context_output_path=args.context_out,
        excluded_review_csv_paths=args.exclude_reviewed,
    )
    print(json.dumps({"written": count, "out": args.out}))


def _report_observations(args: argparse.Namespace) -> None:
    report = write_observation_report(args.observations, args.out)
    print(json.dumps({"observations": report.get("observations", 0), "out": args.out}))


def _build_training_data(args: argparse.Namespace) -> None:
    count = build_training_jsonl(args.labels, args.out, context_jsonl=args.context)
    print(json.dumps({"written": count, "out": args.out}))


def _merge_training_data(args: argparse.Namespace) -> None:
    count = merge_training_jsonl(args.input, args.out, excluded_thread_ids=set(args.exclude_thread_id))
    print(json.dumps({"written": count, "out": args.out, "inputs": args.input}))


def _evaluate_labels(args: argparse.Namespace) -> None:
    metrics = write_label_evaluation(args.labels, args.out)
    print(json.dumps({"rows": metrics.get("rows", 0), "out": args.out}))


def _apply_labels(args: argparse.Namespace) -> None:
    count = apply_reviewed_labels(args.observations, args.labels, args.out)
    print(json.dumps({"updated": count, "out": args.out}))


def _train_local_classifier(args: argparse.Namespace) -> None:
    count = train_classifier(args.training, args.model_out)
    print(json.dumps({"examples": count, "model_out": args.model_out}))


def _evaluate_local_classifier(args: argparse.Namespace) -> None:
    metrics = write_classifier_evaluation(args.training, args.out)
    print(json.dumps({"examples": metrics.get("examples", 0), "out": args.out}))


def _predict_local_classifier(args: argparse.Namespace) -> None:
    count = predict_jsonl(args.model, args.input, args.out)
    print(json.dumps({"predicted": count, "out": args.out}))


def _train_sklearn_classifier(args: argparse.Namespace) -> None:
    count = train_sklearn_classifier(args.training, args.model_out, runs=args.runs)
    print(json.dumps({"examples": count, "model_out": args.model_out}))


def _evaluate_sklearn_classifier(args: argparse.Namespace) -> None:
    metrics = write_sklearn_evaluation(args.training, args.out, runs=args.runs)
    print(json.dumps({"examples": metrics.get("examples", 0), "out": args.out}))


def _evaluate_sklearn_variants(args: argparse.Namespace) -> None:
    metrics = write_sklearn_variant_evaluation(args.training, args.out, runs=args.runs)
    print(json.dumps({"examples": metrics.get("examples", 0), "out": args.out}))


def _train_hf_classifier(args: argparse.Namespace) -> None:
    count = train_hf_classifier(
        args.training,
        args.model_out,
        model_name=args.embedding_model,
        runs=args.runs,
        text_mode=args.text_mode,
    )
    print(json.dumps({"examples": count, "model_out": args.model_out, "embedding_model": args.embedding_model, "text_mode": args.text_mode}))


def _evaluate_hf_classifier(args: argparse.Namespace) -> None:
    metrics = write_hf_evaluation(
        args.training,
        args.out,
        model_name=args.embedding_model,
        runs=args.runs,
        text_mode=args.text_mode,
    )
    print(json.dumps({"examples": metrics.get("examples", 0), "out": args.out, "embedding_model": args.embedding_model, "text_mode": args.text_mode}))


def _embed_jsonl(args: argparse.Namespace) -> None:
    count = write_embeddings_jsonl(
        args.input,
        args.out,
        model_name=args.embedding_model,
        backend=args.backend,
        batch_size=args.batch_size,
    )
    print(json.dumps({"written": count, "out": args.out, "embedding_model": args.embedding_model}))


def _cluster_embeddings(args: argparse.Namespace) -> None:
    result = cluster_embedding_jsonl(args.embeddings, args.out, threshold=args.threshold)
    print(json.dumps(result))


def _compare_classifiers(args: argparse.Namespace) -> None:
    comparison = compare_classifier_backends(
        args.training,
        args.out,
        runs=args.runs,
        hf_model_name=args.embedding_model,
        include_hf=not args.skip_hf,
        include_ensemble=not args.skip_ensemble,
        ensemble_min_confidence=args.ensemble_min_confidence,
        ensemble_min_agreement=args.ensemble_min_agreement,
        ensemble_field_min_confidence=None,
    )
    print(json.dumps({"backends": list(comparison["backends"].keys()), "out": args.out}))


def _run_frozen_embedding_bakeoff(args: argparse.Namespace) -> None:
    result = run_frozen_embedding_bakeoff(
        args.training,
        args.out,
        embedding_models=tuple(args.embedding_model) or DEFAULT_EMBEDDING_MODELS,
        runs=args.runs,
        group_field=args.group_field,
        embedding_cache_dir=args.embedding_cache_dir,
    )
    print(json.dumps({"examples": result["examples"], "ranked": len(result["ranking"]), "out": args.out}))


def _run_routed_rubric_bakeoff(args: argparse.Namespace) -> None:
    result = run_routed_rubric_bakeoff(
        args.training,
        args.out,
        runs=args.runs,
        group_field=args.group_field,
        embedding_cache_dir=args.embedding_cache_dir,
    )
    print(json.dumps({"examples": result["examples"], "mean_macro_f1": result["mean_macro_f1"], "out": args.out}))


def _train_routed_rubric_classifier(args: argparse.Namespace) -> None:
    count = train_routed_rubric_classifier(args.training, args.model_out)
    print(json.dumps({"examples": count, "model_out": args.model_out}))


def _assess_publication_readiness(args: argparse.Namespace) -> None:
    result = write_publication_readiness(args.evaluation, args.out)
    print(json.dumps({"ready": result["ready"], "failures": result["failures"], "out": args.out}))


def _run_setfit_bakeoff(args: argparse.Namespace) -> None:
    result = run_setfit_bakeoff(
        args.training,
        args.out,
        checkpoints=parse_checkpoint_specs(args.checkpoint),
        fields=tuple(args.field) or TARGET_FIELDS,
        num_epochs=args.epochs,
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        group_field=args.group_field,
    )
    print(json.dumps({"examples": result["examples"], "ranked": len(result["ranking"]), "out": args.out}))


def _threshold_report(args: argparse.Namespace) -> None:
    report = write_threshold_report(args.training, args.model, args.out)
    print(json.dumps({"examples": report["examples"], "out": args.out}))


def _evaluate_fixed_classifier(args: argparse.Namespace) -> None:
    report = write_fixed_model_evaluation(args.evaluation, args.model, args.out)
    print(json.dumps({"examples": report["examples"], "mean_macro_f1": report["mean_macro_f1"], "out": args.out}))


def _export_active_learning(args: argparse.Namespace) -> None:
    count = export_active_learning_queue(args.input, args.model, args.out, max_rows=args.max_rows)
    print(json.dumps({"written": count, "out": args.out}))


def _train_high_precision_classifier(args: argparse.Namespace) -> None:
    count = train_high_precision_classifier(
        args.training,
        args.model_out,
        runs=args.runs,
        min_confidence=args.min_confidence,
        min_agreement=args.min_agreement,
        include_hf=not args.skip_hf,
        hf_model_name=args.embedding_model,
        field_min_confidence=PRECISION_FIRST_FIELD_CONFIDENCE if args.precision_first else None,
    )
    print(json.dumps({"examples": count, "model_out": args.model_out}))


def _evaluate_high_precision_classifier(args: argparse.Namespace) -> None:
    metrics = write_high_precision_evaluation(
        args.training,
        args.out,
        runs=args.runs,
        min_confidence=args.min_confidence,
        min_agreement=args.min_agreement,
        include_hf=not args.skip_hf,
        hf_model_name=args.embedding_model,
        field_min_confidence=PRECISION_FIRST_FIELD_CONFIDENCE if args.precision_first else None,
    )
    print(json.dumps({"examples": metrics.get("examples", 0), "out": args.out}))


def _raw_item_from_record(record: dict) -> RawItem:
    record = dict(record)
    record["platform"] = SourcePlatform(record["platform"])
    if isinstance(record.get("published_at"), str):
        record["published_at"] = datetime.fromisoformat(record["published_at"].replace("Z", "+00:00"))
    record.pop("text", None)
    return RawItem(**record)


if __name__ == "__main__":
    main()
