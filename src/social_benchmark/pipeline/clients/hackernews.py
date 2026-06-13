from __future__ import annotations

from collections.abc import Iterable

from social_benchmark.pipeline.clients.base import HttpJsonClient
from social_benchmark.pipeline.models import RawItem, SourcePlatform, utc_from_unix
from social_benchmark.pipeline.normalization import hash_identifier, normalize_text


class HackerNewsClient:
    """Client for the official Hacker News Firebase API."""

    def __init__(
        self,
        http: HttpJsonClient | None = None,
        search_http: HttpJsonClient | None = None,
    ) -> None:
        self.http = http or HttpJsonClient(
            base_url="https://hacker-news.firebaseio.com/v0",
            min_delay_seconds=0.02,
        )
        self.search_http = search_http or HttpJsonClient(
            base_url="https://hn.algolia.com/api/v1",
            min_delay_seconds=0.25,
        )

    def search_story_ids(
        self,
        query: str,
        max_stories: int = 30,
        min_comments: int = 0,
        sort_by_date: bool = False,
    ) -> list[int]:
        """Search stories via the official HN Algolia API and return story IDs."""
        endpoint = "search_by_date" if sort_by_date else "search"
        story_ids: list[int] = []
        page = 0
        while len(story_ids) < max_stories:
            payload = self.search_http.get_json(
                endpoint,
                params={
                    "query": query,
                    "tags": "story",
                    "hitsPerPage": min(50, max_stories),
                    "page": page,
                },
            )
            hits = (payload or {}).get("hits") or []
            if not hits:
                break
            for hit in hits:
                if int(hit.get("num_comments") or 0) < min_comments:
                    continue
                story_id = hit.get("story_id") or hit.get("objectID")
                if story_id is not None:
                    story_ids.append(int(story_id))
                if len(story_ids) >= max_stories:
                    break
            page += 1
            if page >= int((payload or {}).get("nbPages") or 0):
                break
        return story_ids

    def get_item(self, item_id: int | str) -> dict | None:
        return self.http.get_json(f"item/{item_id}.json")

    def get_user(self, user_id: str) -> dict | None:
        return self.http.get_json(f"user/{user_id}.json")

    def top_story_ids(self) -> list[int]:
        return list(self.http.get_json("topstories.json") or [])

    def new_story_ids(self) -> list[int]:
        return list(self.http.get_json("newstories.json") or [])

    def best_story_ids(self) -> list[int]:
        return list(self.http.get_json("beststories.json") or [])

    def ask_story_ids(self) -> list[int]:
        return list(self.http.get_json("askstories.json") or [])

    def max_item_id(self) -> int:
        return int(self.http.get_json("maxitem.json"))

    def fetch_items(self, item_ids: Iterable[int | str]) -> list[RawItem]:
        items: list[RawItem] = []
        for item_id in item_ids:
            payload = self.get_item(item_id)
            if payload:
                items.append(hn_item_to_raw_item(payload))
        return items

    def fetch_story_with_comments(
        self,
        story_id: int | str,
        max_comments: int = 200,
        max_depth: int = 8,
    ) -> list[RawItem]:
        root = self.get_item(story_id)
        if not root:
            return []
        raw_items = [hn_item_to_raw_item(root, thread_id=str(root["id"]), depth=0, root_story_id=str(root["id"]), comment_index=0)]
        pending: list[tuple[int | str, int]] = [(current_id, 1) for current_id in (root.get("kids") or [])]
        comment_count = 0
        while pending and comment_count < max_comments:
            current_id, depth = pending.pop(0)
            if depth > max_depth:
                continue
            item = self.get_item(current_id)
            if not item or item.get("deleted") or item.get("dead"):
                continue
            comment_count += 1
            raw_items.append(
                hn_item_to_raw_item(
                    item,
                    thread_id=str(root["id"]),
                    depth=depth,
                    root_story_id=str(root["id"]),
                    comment_index=comment_count,
                )
            )
            pending.extend((child_id, depth + 1) for child_id in (item.get("kids") or []))
        return raw_items


def hn_item_to_raw_item(
    payload: dict,
    thread_id: str | None = None,
    depth: int | None = None,
    root_story_id: str | None = None,
    comment_index: int | None = None,
) -> RawItem:
    item_type = payload.get("type", "item")
    source_id = str(payload["id"])
    parent_id = str(payload["parent"]) if payload.get("parent") is not None else None
    resolved_thread_id = thread_id or (source_id if item_type == "story" else parent_id)
    title = normalize_text(payload.get("title") or "")
    body = normalize_text(payload.get("text") or "")
    url = payload.get("url") or f"https://news.ycombinator.com/item?id={source_id}"
    return RawItem(
        platform=SourcePlatform.HACKER_NEWS,
        source_id=source_id,
        title=title,
        body=body,
        url=url,
        community_id="hacker_news",
        thread_id=resolved_thread_id,
        parent_id=parent_id,
        author_id_hash=hash_identifier(payload.get("by")),
        author_handle=payload.get("by"),
        published_at=utc_from_unix(payload.get("time")),
        engagement={
            "score": int(payload.get("score") or 0),
            "descendants": int(payload.get("descendants") or 0),
        },
        metadata={
            "type": item_type,
            "kids": payload.get("kids") or [],
            "deleted": bool(payload.get("deleted")),
            "dead": bool(payload.get("dead")),
            "depth": depth if depth is not None else (0 if item_type == "story" else None),
            "root_story_id": root_story_id or (source_id if item_type == "story" else None),
            "comment_index": comment_index,
            "is_root_story": item_type == "story",
        },
    )
