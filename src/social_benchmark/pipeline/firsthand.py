from __future__ import annotations

import re


FIRST_PERSON_RE = re.compile(
    r"\b("
    r"i|i've|i'd|i'm|me|my|mine|"
    r"we|we've|we'd|we're|us|our|ours"
    r")\b",
    re.IGNORECASE,
)

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
    first_person = FIRST_PERSON_RE.search(lowered)
    usage_verb = USAGE_VERB_RE.search(lowered)
    if not first_person or not usage_verb:
        return False
    return abs(first_person.start() - usage_verb.start()) <= 80

