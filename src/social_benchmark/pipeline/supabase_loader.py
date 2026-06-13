"""Load a score snapshot and its observations into Supabase.

The web dashboard reads its data from Supabase (with the static
``web/public/snapshot.json`` as an offline fallback). This module pushes a
snapshot produced by :mod:`social_benchmark.pipeline.score_snapshot` into the
normalized tables created by the ``benchmark_snapshot_schema`` migration:
``score_snapshots``, ``leaderboard_rows``, ``task_rows``, ``evidence_samples``
and ``observations``.

Writes use the Supabase REST (PostgREST) API. In production, pass a service-role
key (``SUPABASE_SERVICE_ROLE_KEY``) which bypasses row-level security. For a
one-off seed against a project where only the publishable/anon key is available,
a temporary ``for insert`` RLS policy can be enabled while loading and dropped
afterwards, keeping the public posture read-only.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

BATCH_SIZE = 500

# Tables that hold per-snapshot rows; cleared for the snapshot before reload.
SNAPSHOT_TABLES = ("evidence_samples", "task_rows", "leaderboard_rows")


class SupabaseError(RuntimeError):
    pass


def load_snapshot_to_supabase(
    *,
    snapshot_path: str | Path,
    observation_paths: Iterable[str | Path] = (),
    url: str | None = None,
    key: str | None = None,
    make_current: bool = True,
) -> dict[str, int]:
    """Upsert one snapshot (and optional observations) into Supabase.

    Returns a dict of table -> rows written.
    """
    url = (url or os.environ.get("SUPABASE_URL") or "").rstrip("/")
    key = (
        key
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or ""
    )
    if not url or not key:
        raise SupabaseError("SUPABASE_URL and a Supabase key are required")

    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    snapshot_id = str(snapshot["snapshot_id"])
    client = _RestClient(url, key)

    if make_current:
        # Demote any previously current snapshot so exactly one is live.
        client.patch("score_snapshots", {"is_current": "eq.true"}, {"is_current": False})

    # Replace this snapshot's child rows for idempotent re-runs.
    for table in SNAPSHOT_TABLES:
        client.delete(table, {"snapshot_id": f"eq.{snapshot_id}"})
    client.delete("score_snapshots", {"snapshot_id": f"eq.{snapshot_id}"})

    written: dict[str, int] = {}

    client.insert(
        "score_snapshots",
        [
            {
                "snapshot_id": snapshot_id,
                "generated_at": snapshot.get("generated_at"),
                "is_current": bool(make_current),
                "gates": snapshot.get("gates", {}),
                "corpus": snapshot.get("corpus", {}),
                "overall": snapshot.get("overall", []),
                "methodology": snapshot.get("methodology", {}),
            }
        ],
    )
    written["score_snapshots"] = 1

    written["leaderboard_rows"] = client.insert(
        "leaderboard_rows", list(_leaderboard_records(snapshot, snapshot_id))
    )
    written["task_rows"] = client.insert(
        "task_rows", list(_task_records(snapshot, snapshot_id))
    )
    written["evidence_samples"] = client.insert(
        "evidence_samples", list(_evidence_records(snapshot, snapshot_id))
    )

    observation_paths = list(observation_paths)
    if observation_paths:
        # Observations are global; refresh them wholesale and tag with this snapshot.
        client.delete("observations", {"id": "gte.0"})
        written["observations"] = client.insert(
            "observations", list(_observation_records(observation_paths, snapshot_id))
        )

    return written


def _leaderboard_records(snapshot: dict[str, Any], snapshot_id: str):
    for row in snapshot.get("leaderboard", []):
        ci = row.get("ci") or [None, None]
        yield {
            "snapshot_id": snapshot_id,
            "model_id": row.get("model_id", ""),
            "provider_id": row.get("provider_id", ""),
            "aspect": row.get("aspect", ""),
            "score": row.get("score"),
            "ci_low": ci[0],
            "ci_high": ci[1],
            "ess": row.get("ess"),
            "weighted_n": row.get("weighted_n"),
            "n_observations": row.get("n_observations"),
            "n_threads": row.get("n_threads"),
            "n_authors": row.get("n_authors"),
            "firsthand_ratio": row.get("firsthand_ratio"),
            "human_share": row.get("human_share"),
            "warnings": row.get("warnings", []),
            "publishable": bool(row.get("publishable", False)),
            "tier": row.get("tier", "insufficient"),
        }


def _task_records(snapshot: dict[str, Any], snapshot_id: str):
    for task, rows in (snapshot.get("tasks") or {}).items():
        for row in rows:
            ci = row.get("ci") or [None, None]
            yield {
                "snapshot_id": snapshot_id,
                "task": task,
                "model_id": row.get("model_id", ""),
                "aspect": row.get("aspect", ""),
                "score": row.get("score"),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "ess": row.get("ess"),
                "tier": row.get("tier", "insufficient"),
            }


def _evidence_records(snapshot: dict[str, Any], snapshot_id: str):
    for cell_key, samples in (snapshot.get("evidence") or {}).items():
        model_id, _, aspect = cell_key.partition("|")
        for ord_index, sample in enumerate(samples):
            yield {
                "snapshot_id": snapshot_id,
                "model_id": model_id,
                "aspect": aspect,
                "ord": ord_index,
                "span": sample.get("span", ""),
                "url": sample.get("url", ""),
                "polarity": sample.get("polarity"),
                "evidence_type": sample.get("evidence_type"),
                "firsthand": sample.get("firsthand"),
                "human_labeled": sample.get("human_labeled"),
            }


def _observation_records(paths: list[str | Path], snapshot_id: str):
    seen: set[tuple[str, str]] = set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row.get("source_item_id") or ""), str(row.get("model_id") or ""))
                if key in seen:
                    continue
                seen.add(key)
                yield {
                    "snapshot_id": snapshot_id,
                    "source_platform": row.get("source_platform"),
                    "source_item_id": row.get("source_item_id"),
                    "thread_id": row.get("thread_id"),
                    "url": row.get("url"),
                    "model_id": row.get("model_id"),
                    "provider_id": row.get("provider_id"),
                    "product_id": row.get("product_id"),
                    "inference_profile": row.get("inference_profile"),
                    "task_category": row.get("task_category"),
                    "aspect_category": row.get("aspect_category"),
                    "evidence_type": row.get("evidence_type"),
                    "polarity_score": _maybe_int(row.get("polarity_score")),
                    "firsthand_flag": _maybe_bool(row.get("firsthand_flag")),
                    "evidence_text": row.get("evidence_text") or row.get("text") or "",
                }


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    return str(value).lower() == "true"


class _RestClient:
    def __init__(self, url: str, key: str) -> None:
        self._endpoint = f"{url}/rest/v1"
        self._headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def insert(self, table: str, rows: list[dict[str, Any]]) -> int:
        total = 0
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            self._request(
                "POST",
                f"{self._endpoint}/{table}",
                body=batch,
                extra_headers={"Prefer": "return=minimal"},
            )
            total += len(batch)
        return total

    def delete(self, table: str, filters: dict[str, str]) -> None:
        query = "&".join(f"{column}={value}" for column, value in filters.items())
        self._request("DELETE", f"{self._endpoint}/{table}?{query}")

    def patch(self, table: str, filters: dict[str, str], values: dict[str, Any]) -> None:
        query = "&".join(f"{column}={value}" for column, value in filters.items())
        self._request("PATCH", f"{self._endpoint}/{table}?{query}", body=values)

    def _request(
        self,
        method: str,
        url: str,
        *,
        body: Any | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = dict(self._headers)
        if extra_headers:
            headers.update(extra_headers)
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request) as response:  # noqa: S310 (trusted URL)
                response.read()
        except urllib.error.HTTPError as error:  # pragma: no cover - network error path
            detail = error.read().decode("utf-8", "replace")
            raise SupabaseError(f"{method} {url} failed ({error.code}): {detail}") from error
