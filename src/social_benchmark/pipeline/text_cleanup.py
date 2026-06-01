from __future__ import annotations

import re

from social_benchmark.pipeline.models import RawItem, SourcePlatform
from social_benchmark.pipeline.normalization import normalize_text


CHECKBOX_RE = re.compile(r"^\s*[-*]\s+\[[ xX]\]\s+")
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
URL_RE = re.compile(r"https?://\S+")

GITHUB_BOILERPLATE_PREFIXES = (
    "### ",
    "## ",
    "task:",
    "[task]",
    "- [ ]",
    "- [x]",
    "steps to reproduce",
    "expected behavior",
    "actual behavior",
    "environment",
    "additional context",
    "checklist",
)

LOW_SIGNAL_TITLE_PREFIXES = (
    "[task]",
    "task-",
    "chore:",
    "docs:",
    "bump ",
    "release ",
)


def clean_item_text(item: RawItem) -> tuple[str, str]:
    title = normalize_text(item.title)
    body = item.body or ""
    if item.platform == SourcePlatform.GITHUB:
        return title, clean_github_markdown(body)
    if item.platform == SourcePlatform.HACKER_NEWS:
        return title, clean_hacker_news_text(body)
    return title, normalize_text(body)


def clean_github_markdown(text: str) -> str:
    text = HTML_COMMENT_RE.sub(" ", text or "")
    text = CODE_FENCE_RE.sub(" ", text)
    kept_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if CHECKBOX_RE.match(line):
            continue
        if MARKDOWN_HEADING_RE.match(line) and not _line_mentions_model(line):
            continue
        if any(lowered.startswith(prefix) for prefix in GITHUB_BOILERPLATE_PREFIXES) and not _line_mentions_model(line):
            continue
        kept_lines.append(line)
    return normalize_text(" ".join(kept_lines))


def clean_hacker_news_text(text: str) -> str:
    return normalize_text(URL_RE.sub(" ", text or ""))


def is_low_signal_title(title: str) -> bool:
    lowered = normalize_text(title).lower()
    return any(lowered.startswith(prefix) for prefix in LOW_SIGNAL_TITLE_PREFIXES)


def _line_mentions_model(line: str) -> bool:
    lowered = line.lower()
    return any(token in lowered for token in ("gpt", "claude", "gemini", "llama", "chatgpt", "deepseek", "grok"))

