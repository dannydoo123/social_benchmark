from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from social_benchmark.pipeline.clients.base import ApiClientError, HttpJsonClient
from social_benchmark.pipeline.models import RawItem, SourcePlatform, utc_from_unix
from social_benchmark.pipeline.normalization import hash_identifier, normalize_evidence_text, normalize_text


@dataclass(slots=True)
class RedditOAuthConfig:
    client_id: str
    client_secret: str | None = None
    refresh_token: str | None = None
    username: str | None = None
    password: str | None = None

    @classmethod
    def from_env(cls) -> "RedditOAuthConfig | None":
        client_id = os.environ.get("REDDIT_CLIENT_ID")
        if not client_id:
            return None
        return cls(
            client_id=client_id,
            client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
            refresh_token=os.environ.get("REDDIT_REFRESH_TOKEN"),
            username=os.environ.get("REDDIT_USERNAME"),
            password=os.environ.get("REDDIT_PASSWORD"),
        )


class RedditClient:
    """Read-only Reddit collector for subreddit listings, search, and threads."""

    def __init__(
        self,
        http: HttpJsonClient | None = None,
        oauth_config: RedditOAuthConfig | None = None,
        user_agent: str = "social-benchmark/0.1 reddit-collector",
        min_delay_seconds: float = 1.0,
    ) -> None:
        self.oauth_config = oauth_config if oauth_config is not None else RedditOAuthConfig.from_env()
        self.user_agent = user_agent
        self.http = http or HttpJsonClient(
            base_url="https://oauth.reddit.com" if self.oauth_config else "https://www.reddit.com",
            user_agent=user_agent,
            min_delay_seconds=min_delay_seconds,
            retries=2,
        )
        self._access_token: str | None = None
        self._access_token_expires_at = 0.0

    def fetch_subreddit_posts(
        self,
        subreddit: str,
        sort: str = "top",
        time_filter: str = "month",
        limit: int = 50,
        include_comments: bool = False,
        max_comments: int = 200,
        max_depth: int = 6,
    ) -> list[RawItem]:
        posts = self._listing_posts(
            f"/r/{subreddit}/{sort}.json",
            params={"t": time_filter, "limit": min(100, max(1, limit)), "raw_json": 1},
            max_posts=limit,
        )
        return self._posts_with_optional_comments(posts, include_comments, max_comments, max_depth)

    def search_subreddit_posts(
        self,
        subreddit: str,
        query: str,
        sort: str = "relevance",
        time_filter: str = "month",
        limit: int = 50,
        include_comments: bool = False,
        max_comments: int = 200,
        max_depth: int = 6,
    ) -> list[RawItem]:
        posts = self._listing_posts(
            f"/r/{subreddit}/search.json",
            params={
                "q": query,
                "restrict_sr": 1,
                "sort": sort,
                "t": time_filter,
                "limit": min(100, max(1, limit)),
                "raw_json": 1,
            },
            max_posts=limit,
        )
        return self._posts_with_optional_comments(posts, include_comments, max_comments, max_depth)

    def fetch_thread(
        self,
        subreddit: str,
        article_id: str,
        max_comments: int = 200,
        max_depth: int = 6,
    ) -> list[RawItem]:
        payload = self._get_json(
            f"/r/{subreddit}/comments/{_strip_fullname(article_id)}.json",
            params={"limit": max_comments, "depth": max_depth, "sort": "confidence", "raw_json": 1},
        )
        return reddit_thread_to_raw_items(payload, max_comments=max_comments, max_depth=max_depth)

    def _posts_with_optional_comments(
        self,
        posts: list[dict[str, Any]],
        include_comments: bool,
        max_comments: int,
        max_depth: int,
    ) -> list[RawItem]:
        raw_items: list[RawItem] = []
        seen: set[tuple[str, str]] = set()
        for post in posts:
            if include_comments:
                subreddit = post.get("subreddit") or ""
                thread_items = self.fetch_thread(str(subreddit), str(post.get("id") or post.get("name")), max_comments=max_comments, max_depth=max_depth)
                for item in thread_items:
                    key = (item.platform.value, item.source_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    raw_items.append(item)
            else:
                item = reddit_post_to_raw_item(post)
                key = (item.platform.value, item.source_id)
                if key not in seen:
                    seen.add(key)
                    raw_items.append(item)
        return raw_items

    def _listing_posts(self, path: str, params: dict[str, Any], max_posts: int) -> list[dict[str, Any]]:
        posts: list[dict[str, Any]] = []
        after: str | None = None
        count = 0
        while len(posts) < max_posts:
            page_params = dict(params)
            page_params["after"] = after
            page_params["count"] = count
            payload = self._get_json(path, page_params)
            data = (payload or {}).get("data") or {}
            children = data.get("children") or []
            if not children:
                break
            for child in children:
                if child.get("kind") == "t3":
                    posts.append(child.get("data") or {})
                if len(posts) >= max_posts:
                    break
            after = data.get("after")
            count += len(children)
            if not after:
                break
        return posts

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        headers = self._headers()
        return self.http.get_json(path, params=params, headers=headers)

    def _headers(self) -> dict[str, str]:
        if not self.oauth_config:
            return {}
        token = self._access_token_value()
        return {"Authorization": f"Bearer {token}"}

    def _access_token_value(self) -> str:
        if self._access_token and time.time() < self._access_token_expires_at:
            return self._access_token
        if not self.oauth_config:
            raise ApiClientError("Reddit OAuth config is missing")
        token_payload = self._request_access_token(self.oauth_config)
        token = token_payload.get("access_token")
        if not token:
            raise ApiClientError("Reddit access token response did not include access_token")
        self._access_token = str(token)
        self._access_token_expires_at = time.time() + max(60, int(token_payload.get("expires_in") or 3600) - 60)
        return self._access_token

    def _request_access_token(self, config: RedditOAuthConfig) -> dict[str, Any]:
        if config.refresh_token:
            body = {"grant_type": "refresh_token", "refresh_token": config.refresh_token}
        elif config.username and config.password:
            body = {
                "grant_type": "password",
                "username": config.username,
                "password": config.password,
            }
        else:
            body = {"grant_type": "client_credentials"}

        auth = base64.b64encode(f"{config.client_id}:{config.client_secret or ''}".encode("utf-8")).decode("ascii")
        request = Request(
            "https://www.reddit.com/api/v1/access_token",
            data=urlencode(body).encode("utf-8"),
            headers={
                "Authorization": f"Basic {auth}",
                "User-Agent": self.user_agent,
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ApiClientError(f"Reddit OAuth token request failed: {exc}") from exc


def reddit_thread_to_raw_items(
    payload: Any,
    max_comments: int = 200,
    max_depth: int = 6,
) -> list[RawItem]:
    if not isinstance(payload, list) or not payload:
        return []
    post_children = ((payload[0] or {}).get("data") or {}).get("children") or []
    if not post_children:
        return []
    post = post_children[0].get("data") or {}
    root = reddit_post_to_raw_item(post)
    raw_items = [root]
    comment_listing = payload[1] if len(payload) > 1 else {}
    for comment in _flatten_comment_listing(comment_listing, max_comments=max_comments, max_depth=max_depth):
        raw_items.append(reddit_comment_to_raw_item(comment, post))
    return raw_items


def reddit_post_to_raw_item(payload: dict[str, Any]) -> RawItem:
    source_id = _fullname(payload, "t3")
    subreddit = str(payload.get("subreddit") or "").strip()
    permalink = payload.get("permalink") or f"/r/{subreddit}/comments/{payload.get('id') or ''}/"
    return RawItem(
        platform=SourcePlatform.REDDIT,
        source_id=source_id,
        title=normalize_text(payload.get("title") or ""),
        body=normalize_evidence_text(payload.get("selftext") or ""),
        url=_absolute_reddit_url(permalink),
        community_id=_community_id(subreddit),
        thread_id=source_id,
        parent_id=None,
        author_id_hash=hash_identifier(payload.get("author")),
        author_handle=payload.get("author"),
        published_at=utc_from_unix(payload.get("created_utc")),
        engagement={
            "score": int(payload.get("score") or 0),
            "ups": int(payload.get("ups") or 0),
            "num_comments": int(payload.get("num_comments") or 0),
            "upvote_ratio": float(payload.get("upvote_ratio") or 0.0),
        },
        metadata={
            "type": "post",
            "subreddit": subreddit,
            "name": source_id,
            "link_flair_text": payload.get("link_flair_text"),
            "over_18": bool(payload.get("over_18")),
            "stickied": bool(payload.get("stickied")),
            "locked": bool(payload.get("locked")),
            "is_self": bool(payload.get("is_self")),
            "is_root_story": True,
        },
    )


def reddit_comment_to_raw_item(payload: dict[str, Any], root_post: dict[str, Any]) -> RawItem:
    source_id = _fullname(payload, "t1")
    subreddit = str(payload.get("subreddit") or root_post.get("subreddit") or "").strip()
    root_source_id = _fullname(root_post, "t3")
    parent_id = payload.get("parent_id")
    return RawItem(
        platform=SourcePlatform.REDDIT,
        source_id=source_id,
        title="",
        body=normalize_evidence_text(payload.get("body") or ""),
        url=_absolute_reddit_url(payload.get("permalink") or root_post.get("permalink") or ""),
        community_id=_community_id(subreddit),
        thread_id=root_source_id,
        parent_id=str(parent_id) if parent_id else None,
        author_id_hash=hash_identifier(payload.get("author")),
        author_handle=payload.get("author"),
        published_at=utc_from_unix(payload.get("created_utc")),
        engagement={
            "score": int(payload.get("score") or 0),
            "ups": int(payload.get("ups") or 0),
        },
        metadata={
            "type": "comment",
            "subreddit": subreddit,
            "name": source_id,
            "depth": int(payload.get("_depth") or 0),
            "root_story_id": root_source_id,
            "comment_index": int(payload.get("_comment_index") or 0),
            "controversiality": int(payload.get("controversiality") or 0),
            "stickied": bool(payload.get("stickied")),
        },
    )


def _flatten_comment_listing(
    listing: dict[str, Any],
    max_comments: int,
    max_depth: int,
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    stack: list[tuple[dict[str, Any], int]] = [((child.get("data") or {}), 0) for child in reversed(((listing.get("data") or {}).get("children") or []))]
    while stack and len(flattened) < max_comments:
        payload, depth = stack.pop()
        if payload.get("body") in {"[deleted]", "[removed]"} or not payload.get("body"):
            continue
        if depth > max_depth:
            continue
        payload = dict(payload)
        payload["_depth"] = depth
        payload["_comment_index"] = len(flattened) + 1
        flattened.append(payload)
        replies = payload.get("replies")
        if isinstance(replies, dict):
            children = ((replies.get("data") or {}).get("children") or [])
            for child in reversed(children):
                if child.get("kind") == "t1":
                    stack.append((child.get("data") or {}, depth + 1))
    return flattened


def _fullname(payload: dict[str, Any], prefix: str) -> str:
    name = payload.get("name")
    if name:
        return str(name)
    return f"{prefix}_{payload.get('id') or ''}"


def _strip_fullname(value: str) -> str:
    return value.split("_", 1)[1] if "_" in value else value


def _community_id(subreddit: str) -> str:
    return f"r/{subreddit}" if subreddit else "reddit"


def _absolute_reddit_url(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return f"https://www.reddit.com{path_or_url if path_or_url.startswith('/') else '/' + path_or_url}"
