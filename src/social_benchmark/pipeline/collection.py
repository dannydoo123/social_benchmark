from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.clients.github import GitHubClient
from social_benchmark.pipeline.clients.hackernews import HackerNewsClient
from social_benchmark.pipeline.clients.reddit import RedditClient
from social_benchmark.pipeline.models import RawItem


def load_source_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_from_config(config: dict[str, Any]) -> list[RawItem]:
    raw_items: list[RawItem] = []

    hn_sources = config.get("hacker_news", [])
    if hn_sources:
        hn_client = HackerNewsClient()
        for hn_source in hn_sources:
            raw_items.extend(_collect_hn(hn_client, hn_source))

    github_repositories = config.get("github_repositories", [])
    github_searches = config.get("github_searches", [])
    if github_repositories or github_searches:
        github_client = GitHubClient()
        for repo_source in github_repositories:
            raw_items.extend(_collect_github_repo(github_client, repo_source))

        for search_source in github_searches:
            raw_items.extend(
                github_client.search_issue_items(
                    query=search_source["query"],
                    max_items=int(search_source.get("limit", 25)),
                )
            )

    reddit_subreddits = config.get("reddit_subreddits", [])
    reddit_searches = config.get("reddit_searches", [])
    if reddit_subreddits or reddit_searches:
        reddit_client = RedditClient()
        for subreddit_source in reddit_subreddits:
            raw_items.extend(_collect_reddit_subreddit(reddit_client, subreddit_source))
        for search_source in reddit_searches:
            raw_items.extend(_collect_reddit_search(reddit_client, search_source))

    return _dedupe_raw_items(raw_items)


def _collect_hn(client: HackerNewsClient, source: dict[str, Any]) -> list[RawItem]:
    kind = source.get("kind", "top")
    limit = int(source.get("limit", 25))
    comments = int(source.get("comments", 0))
    max_depth = int(source.get("max_depth", 8))
    workers = max(1, int(source.get("workers", 8)))
    id_getter = {
        "top": client.top_story_ids,
        "new": client.new_story_ids,
        "best": client.best_story_ids,
        "ask": client.ask_story_ids,
    }[kind]
    ids = id_getter()[:limit]
    raw_items: list[RawItem] = []
    if comments:
        def fetch_story(story_id: int) -> list[RawItem]:
            return client.fetch_story_with_comments(story_id, max_comments=comments, max_depth=max_depth)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for story_items in executor.map(fetch_story, ids):
                raw_items.extend(story_items)
    else:
        raw_items.extend(client.fetch_items(ids))
    return raw_items


def _collect_github_repo(client: GitHubClient, source: dict[str, Any]) -> list[RawItem]:
    return client.fetch_repo_issue_items(
        owner=source["owner"],
        repo=source["repo"],
        state=source.get("state", "all"),
        since=source.get("since"),
        max_issues=int(source.get("limit", 50)),
        include_comments=bool(source.get("include_comments", False)),
    )


def _collect_reddit_subreddit(client: RedditClient, source: dict[str, Any]) -> list[RawItem]:
    return client.fetch_subreddit_posts(
        subreddit=source["name"],
        sort=source.get("sort", "top"),
        time_filter=source.get("time", source.get("time_filter", "month")),
        limit=int(source.get("limit", 50)),
        include_comments=bool(source.get("comments", 0)),
        max_comments=int(source.get("comments", 200)),
        max_depth=int(source.get("max_depth", 6)),
    )


def _collect_reddit_search(client: RedditClient, source: dict[str, Any]) -> list[RawItem]:
    raw_items: list[RawItem] = []
    subreddits = source.get("subreddits") or ([source["subreddit"]] if source.get("subreddit") else [])
    for subreddit in subreddits:
        raw_items.extend(
            client.search_subreddit_posts(
                subreddit=subreddit,
                query=source["query"],
                sort=source.get("sort", "relevance"),
                time_filter=source.get("time", source.get("time_filter", "month")),
                limit=int(source.get("limit", 50)),
                include_comments=bool(source.get("comments", 0)),
                max_comments=int(source.get("comments", 200)),
                max_depth=int(source.get("max_depth", 6)),
            )
        )
    return raw_items


def _dedupe_raw_items(items: list[RawItem]) -> list[RawItem]:
    seen: set[tuple[str, str]] = set()
    deduped: list[RawItem] = []
    for item in items:
        key = (item.platform.value, item.source_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
