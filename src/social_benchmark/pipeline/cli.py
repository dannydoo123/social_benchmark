from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from social_benchmark.pipeline.analysis import write_observation_report
from social_benchmark.pipeline.clients.github import GitHubClient
from social_benchmark.pipeline.clients.hackernews import HackerNewsClient
from social_benchmark.pipeline.collection import collect_from_config, load_source_config
from social_benchmark.pipeline.extractors.rules import RuleBasedExtractor
from social_benchmark.pipeline.label_feedback import apply_reviewed_labels, write_label_evaluation
from social_benchmark.pipeline.labeling import export_labeling_queue
from social_benchmark.pipeline.local_classifier import (
    predict_jsonl,
    train_classifier,
    write_classifier_evaluation,
)
from social_benchmark.pipeline.models import RawItem, SourcePlatform, to_jsonable
from social_benchmark.pipeline.scoring import ScoreAggregator
from social_benchmark.pipeline.sklearn_classifier import (
    train_sklearn_classifier,
    write_sklearn_evaluation,
    write_sklearn_variant_evaluation,
)
from social_benchmark.pipeline.storage import read_jsonl, write_jsonl
from social_benchmark.pipeline.training_data import build_training_jsonl


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


def _raw_item_from_record(record: dict) -> RawItem:
    record = dict(record)
    record["platform"] = SourcePlatform(record["platform"])
    if isinstance(record.get("published_at"), str):
        record["published_at"] = datetime.fromisoformat(record["published_at"].replace("Z", "+00:00"))
    record.pop("text", None)
    return RawItem(**record)


if __name__ == "__main__":
    main()
