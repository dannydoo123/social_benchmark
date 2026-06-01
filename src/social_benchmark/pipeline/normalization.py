from __future__ import annotations

import hashlib
import html
import re


TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    without_tags = TAG_RE.sub(" ", text)
    return html.unescape(without_tags)


def normalize_text(text: str | None) -> str:
    stripped = strip_html(text)
    return WHITESPACE_RE.sub(" ", stripped).strip()


def hash_identifier(value: str | None, salt: str = "social-benchmark-v0") -> str | None:
    if not value:
        return None
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return digest[:24]


def stable_text_hash(text: str) -> str:
    normalized = normalize_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

