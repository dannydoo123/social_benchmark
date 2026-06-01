from __future__ import annotations

import re
import hashlib

from social_benchmark.pipeline.models import RawItem
from social_benchmark.pipeline.normalization import normalize_text, stable_text_hash


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def exact_item_key(item: RawItem) -> str:
    return f"{item.platform.value}:{item.source_id}"


def content_fingerprint(item: RawItem) -> str:
    return stable_text_hash(item.text)


def simhash(text: str, bits: int = 64) -> int:
    tokens = TOKEN_RE.findall(normalize_text(text).lower())
    if not tokens:
        return 0
    vector = [0] * bits
    for token in tokens:
        hashed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
        for index in range(bits):
            if hashed & (1 << index):
                vector[index] += 1
            else:
                vector[index] -= 1
    result = 0
    for index, value in enumerate(vector):
        if value > 0:
            result |= 1 << index
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def near_duplicate(left: str, right: str, max_distance: int = 6) -> bool:
    return hamming_distance(simhash(left), simhash(right)) <= max_distance
