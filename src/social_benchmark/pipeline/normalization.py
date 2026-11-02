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


def normalize_evidence_text(text: str | None) -> str:
    """Preserve paragraph, quote, and code boundaries for opinion attribution.

    Keep normalize_text for fingerprints/search. This conversion is for source bodies.
    """
    if not text:
        return ''
    from html.parser import HTMLParser

    class EvidenceParser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts=[]
            self.quote=0

        def handle_starttag(self, tag, attrs):
            if tag=='blockquote':
                self.quote+=1
                self.parts.append('\n')
            elif tag in ('p','div','li','br'):
                self.parts.append('\n\n' if tag in ('p','div') else '\n')
            elif tag in ('pre','code'):
                self.parts.append('\n```\n' if tag=='pre' else '`')

        def handle_endtag(self, tag):
            if tag=='blockquote':
                self.quote=max(0,self.quote-1)
                self.parts.append('\n\n')
            elif tag in ('p','div','li'):
                self.parts.append('\n\n')
            elif tag in ('pre','code'):
                self.parts.append('\n```\n' if tag=='pre' else '`')

        def handle_data(self, data):
            self.parts.append(('> '+data.replace('\n','\n> ')) if self.quote else data)

    if re.search(r'</?(?:p|div|br|blockquote|pre|code|a|li|em|strong)\b',text,re.I):
        parser=EvidenceParser();parser.feed(text)
        text=''.join(parser.parts)
    # Plain Markdown retains its original quote and fence markers.
    text=re.sub(r'[^\S\n]+',' ',text.replace('\r\n','\n'))
    return re.sub(r'\n[ \t]+','\n',re.sub(r'\n{3,}','\n\n',text)).strip()


def hash_identifier(value: str | None, salt: str = "social-benchmark-v0") -> str | None:
    if not value:
        return None
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return digest[:24]


def stable_text_hash(text: str) -> str:
    normalized = normalize_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

