"""Resolve raw extractor model mentions to canonical provider/model identities.

Hacker News commenters mostly write provider/family names with no version
("Claude", "Gemini", "DeepSeek"). Those unversioned mentions must not become
fake models that fragment and dilute the per-model board; instead they roll up
to the provider only. This module turns a raw ``(model_id, provider_id)`` pair
into a :class:`Resolution` that the scoring assembly uses to:

* derive the provider from :mod:`config/model_registry.json` (fixing extractor
  mislabels such as ``qwen-3.6 -> anthropic``),
* keep versioned mentions as specific models, and
* route unversioned mentions to a provider-level bucket.

A versioned mention not yet present in the registry is still scored as a model
but reported under ``versioned_unregistered`` so the registry can be refreshed
from provider docs ("reflect new models").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "config" / "model_registry.json"

# First dash-separated token -> provider, for mentions not found in the registry.
_TOKEN_PROVIDER = {
    "claude": "anthropic",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "chatgpt": "openai",
    "codex": "openai",
    "gemini": "google",
    "gemma": "google",
    "palm": "google",
    "nano": "google",
    "llama": "meta",
    "mistral": "mistral",
    "mixtral": "mistral",
    "codestral": "mistral",
    "deepseek": "deepseek",
    "qwen": "alibaba",
    "qwen3": "alibaba",
    "grok": "xai",
    "kimi": "moonshot",
}

UNSPECIFIED_MODEL = "__unspecified__"


@dataclass(frozen=True)
class Resolution:
    provider_id: str
    model_id: str | None  # None => unversioned, rolls up to provider only
    status: str  # registry | versioned_unregistered | unversioned | unknown
    versioned: bool


@lru_cache(maxsize=1)
def _registry() -> dict:
    canonical: dict[str, str] = {}  # model_id -> provider_id
    display: dict[str, str] = {}
    providers: set[str] = set()
    if _REGISTRY_PATH.exists():
        data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        for provider in data.get("providers", []):
            pid = provider["id"]
            providers.add(pid)
            for model in provider.get("models", []):
                canonical[model["id"].lower()] = pid
                display[model["id"].lower()] = model.get("display_name", model["id"])
    return {"canonical": canonical, "display": display, "providers": providers}


def _looks_versioned(model_id: str) -> bool:
    """A mention is a specific model only if it carries a version token (a digit).

    ``claude-opus`` / ``gemini`` / ``deepseek`` -> unversioned (provider bucket);
    ``claude-opus-4.8`` / ``gemini-3-flash`` / ``o3`` -> versioned model.
    """
    return bool(re.search(r"\d", model_id))


def provider_for(model_id: str, fallback_provider: str = "") -> str:
    reg = _registry()
    mid = model_id.strip().lower()
    if mid in reg["canonical"]:
        return reg["canonical"][mid]
    first_token = mid.split("-", 1)[0]
    if first_token in _TOKEN_PROVIDER:
        return _TOKEN_PROVIDER[first_token]
    fallback = (fallback_provider or "").strip().lower()
    return fallback or "unknown"


def resolve(model_id: str, provider_id: str = "") -> Resolution:
    reg = _registry()
    mid = (model_id or "").strip().lower()
    if not mid:
        return Resolution(provider_id=(provider_id or "unknown").lower(), model_id=None, status="unknown", versioned=False)

    provider = provider_for(mid, provider_id)
    if mid in reg["canonical"]:
        return Resolution(provider_id=provider, model_id=mid, status="registry", versioned=True)
    if _looks_versioned(mid):
        return Resolution(provider_id=provider, model_id=mid, status="versioned_unregistered", versioned=True)
    return Resolution(provider_id=provider, model_id=None, status="unversioned", versioned=False)


def display_name(model_id: str) -> str:
    return _registry()["display"].get(model_id.lower(), model_id)


def is_registered(model_id: str) -> bool:
    return model_id.lower() in _registry()["canonical"]
