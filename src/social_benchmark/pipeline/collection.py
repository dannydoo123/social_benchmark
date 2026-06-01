from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from social_benchmark.pipeline.clients.github import GitHubClient
from social_benchmark.pipeline.clients.hackernews import HackerNewsClient
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

    return _dedupe_raw_items(raw_items)


def _collect_hn(client: HackerNewsClient, source: dict[str, Any]) -> list[RawItem]:
    kind = source.get("kind", "top")
    limit = int(source.get("limit", 25))
    comments = int(source.get("comments", 0))
    max_depth = int(source.get("max_depth", 8))
    id_getter = {
        "top": client.top_story_ids,
        "new": client.new_story_ids,
        "best": client.best_story_ids,
        "ask": client.ask_story_ids,
    }[kind]
    ids = id_getter()[:limit]
    raw_items: list[RawItem] = []
    if comments:
        for story_id in ids:
            raw_items.extend(client.fetch_story_with_comments(story_id, max_comments=comments, max_depth=max_depth))
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
