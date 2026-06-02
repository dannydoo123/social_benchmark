from __future__ import annotations

import re


USAGE_VERB_RE = re.compile(
    r"\b("
    r"use|used|using|tried|try|tested|test|ran|run|"
    r"asked|ask|prompted|prompt|switched|migrated|"
    r"deployed|deploy|integrated|integrate|built|build|"
    r"paid|subscribed|cancelled|canceled|"
    r"found|find|got|get|see|saw"
    r")\b",
    re.IGNORECASE,
)

FIRST_PERSON_USAGE_RE = re.compile(
    r"\b(i|we|i've|we've|i'd|we'd|i'm|we're)\b"
    r"(?:\W+\w+){0,8}?\W+"
    + USAGE_VERB_RE.pattern,
    re.IGNORECASE,
)

FIRSTHAND_PHRASE_RE = re.compile(
    r"\b("
    r"in production|my workflow|our workflow|our team|my app|our app|"
    r"my experience|from experience|for me|for us|in our tests|in my tests|"
    r"we use|we used|i use|i used|i tried|we tried|i tested|we tested|"
    r"been using|have been using|daily driver"
    r")\b",
    re.IGNORECASE,
)


def is_firsthand_text(text: str) -> bool:
    lowered = text.lower()
    if FIRSTHAND_PHRASE_RE.search(lowered):
        return True
    return bool(FIRST_PERSON_USAGE_RE.search(lowered))
