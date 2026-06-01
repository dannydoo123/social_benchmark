from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

from social_benchmark.pipeline.clients.base import HttpJsonClient
from social_benchmark.pipeline.models import RawItem, SourcePlatform
from social_benchmark.pipeline.normalization import hash_identifier, normalize_text


class GitHubClient:
    """Client for GitHub REST API issue and issue-comment evidence."""

    def __init__(
        self,
        token: str | None = None,
        api_version: str | None = None,
        http: HttpJsonClient | None = None,
    ) -> None:
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.api_version = api_version or os.getenv("GITHUB_API_VERSION", "2026-03-10")
        self.http = http or HttpJsonClient(base_url="https://api.github.com", min_delay_seconds=0.1)

    @property
    def headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": self.api_version,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def search_issues(
        self,
        query: str,
        page: int = 1,
        per_page: int = 30,
        sort: str = "updated",
        order: str = "desc",
    ) -> dict[str, Any]:
        return self.http.get_json(
            "search/issues",
            params={"q": query, "page": page, "per_page": per_page, "sort": sort, "order": order},
            headers=self.headers,
        )

    def list_repo_issues(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        since: str | None = None,
        page: int = 1,
        per_page: int = 30,
    ) -> list[dict[str, Any]]:
        return list(
            self.http.get_json(
                f"repos/{owner}/{repo}/issues",
                params={"state": state, "since": since, "page": page, "per_page": per_page},
                headers=self.headers,
            )
            or []
        )

    def list_issue_comments(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        page: int = 1,
        per_page: int = 100,
    ) -> list[dict[str, Any]]:
        return list(
            self.http.get_json(
                f"repos/{owner}/{repo}/issues/{issue_number}/comments",
                params={"page": page, "per_page": per_page},
                headers=self.headers,
            )
            or []
        )

    def fetch_repo_issue_items(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        since: str | None = None,
        max_issues: int = 50,
        include_comments: bool = True,
    ) -> list[RawItem]:
        raw_items: list[RawItem] = []
        page = 1
        while len(raw_items) < max_issues:
            issues = self.list_repo_issues(owner, repo, state=state, since=since, page=page)
            if not issues:
                break
            for issue in issues:
                if "pull_request" in issue:
                    continue
                raw_items.append(github_issue_to_raw_item(issue, f"{owner}/{repo}"))
                if include_comments:
                    for comment in self.list_issue_comments(owner, repo, int(issue["number"])):
                        raw_items.append(github_comment_to_raw_item(comment, issue, f"{owner}/{repo}"))
                if len([item for item in raw_items if item.parent_id is None]) >= max_issues:
                    break
            page += 1
        return raw_items

    def search_issue_items(self, query: str, max_items: int = 50) -> list[RawItem]:
        response = self.search_issues(query, per_page=min(max_items, 100))
        items = response.get("items") or []
        return [github_issue_to_raw_item(item, _repo_full_name_from_issue(item)) for item in items[:max_items]]


def github_issue_to_raw_item(issue: dict[str, Any], repository: str) -> RawItem:
    source_id = str(issue["id"])
    number = str(issue.get("number") or source_id)
    user = issue.get("user") or {}
    labels = [label.get("name", "") for label in issue.get("labels") or [] if isinstance(label, dict)]
    return RawItem(
        platform=SourcePlatform.GITHUB,
        source_id=source_id,
        title=normalize_text(issue.get("title") or ""),
        body=normalize_text(issue.get("body") or ""),
        url=issue.get("html_url") or "",
        community_id=repository,
        thread_id=number,
        parent_id=None,
        author_id_hash=hash_identifier(str(user.get("id") or user.get("login") or "")),
        author_handle=user.get("login"),
        published_at=_parse_github_datetime(issue.get("created_at")),
        engagement={
            "comments": int(issue.get("comments") or 0),
            "reactions": int((issue.get("reactions") or {}).get("total_count") or 0),
        },
        metadata={
            "number": issue.get("number"),
            "state": issue.get("state"),
            "labels": labels,
            "updated_at": issue.get("updated_at"),
            "closed_at": issue.get("closed_at"),
        },
    )


def github_comment_to_raw_item(comment: dict[str, Any], issue: dict[str, Any], repository: str) -> RawItem:
    source_id = str(comment["id"])
    user = comment.get("user") or {}
    issue_number = str(issue.get("number") or issue.get("id"))
    return RawItem(
        platform=SourcePlatform.GITHUB,
        source_id=source_id,
        title=normalize_text(issue.get("title") or ""),
        body=normalize_text(comment.get("body") or ""),
        url=comment.get("html_url") or issue.get("html_url") or "",
        community_id=repository,
        thread_id=issue_number,
        parent_id=str(issue.get("id")),
        author_id_hash=hash_identifier(str(user.get("id") or user.get("login") or "")),
        author_handle=user.get("login"),
        published_at=_parse_github_datetime(comment.get("created_at")),
        engagement={"reactions": int((comment.get("reactions") or {}).get("total_count") or 0)},
        metadata={
            "issue_number": issue.get("number"),
            "issue_id": issue.get("id"),
            "updated_at": comment.get("updated_at"),
        },
    )


def _parse_github_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _repo_full_name_from_issue(issue: dict[str, Any]) -> str:
    repo_url = issue.get("repository_url") or ""
    marker = "/repos/"
    if marker in repo_url:
        return repo_url.split(marker, 1)[1]
    return "github"

