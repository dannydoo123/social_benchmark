from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from math import log1p

from social_benchmark.pipeline.catalog import ModelCatalog
from social_benchmark.pipeline.models import (
    AspectCategory,
    CandidateFeatures,
    ClaimType,
    EvidenceType,
    Observation,
    RawItem,
    SourcePlatform,
    TaskCategory,
    WeightComponents,
)
from social_benchmark.pipeline.firsthand import is_firsthand_text
from social_benchmark.pipeline.normalization import normalize_text
from social_benchmark.pipeline.text_cleanup import clean_item_text, is_low_signal_title


TASK_TERMS: dict[TaskCategory, tuple[str, ...]] = {
    TaskCategory.CODING: (
        "code",
        "coding",
        "programming",
        "debug",
        "refactor",
        "typescript",
        "python",
        "rust",
        "functional real time strategy game",
    ),
    TaskCategory.WRITING: ("write", "writing", "email", "copy", "essay", "draft", "tone"),
    TaskCategory.RESEARCH: ("research", "sources", "citations", "summarize", "paper", "search"),
    TaskCategory.AGENTS: ("agent", "tool use", "function call", "workflow", "autonomous", "browser"),
    TaskCategory.ROLEPLAY: ("roleplay", "character", "story", "creative"),
    TaskCategory.DATA_ANALYSIS: ("csv", "spreadsheet", "analysis", "chart", "sql", "dataframe"),
    TaskCategory.LONG_CONTEXT: ("long context", "context window", "pdf", "document", "long document"),
    TaskCategory.MULTIMODAL: ("image", "vision", "audio", "screenshot", "multimodal"),
    TaskCategory.API_DEV_WORKFLOW: ("api", "sdk", "rate limit", "latency", "deploy", "integration", "docs"),
}

ASPECT_TERMS: dict[AspectCategory, tuple[str, ...]] = {
    AspectCategory.SATISFACTION: (
        "best",
        "better",
        "worse",
        "great",
        "bad",
        "love",
        "hate",
        "impressed",
        "impressive",
        "pleasantly surprised",
        "surprised us",
        "nailed it",
        "creative",
        "intelligent",
        "useful",
        "excellent",
    ),
    AspectCategory.TRUST_RELIABILITY: (
        "trust",
        "reliable",
        "production",
        "stable",
        "flaky",
        "broken",
        "dependable",
        "safe",
        "unsafe",
        "sandbox",
        "threat model",
        "trusted",
        "untrusted",
        "harm",
        "hurt",
        "malware",
        "secrets",
    ),
    AspectCategory.TASK_FIT: ("good for", "bad for", "works well", "struggles with", "fit", "fit for", "good at"),
    AspectCategory.REGRESSION_STABILITY: ("regression", "got worse", "became worse", "degraded", "nerfed", "used to", "regressed"),
    AspectCategory.HALLUCINATION_SAFETY: ("hallucination", "made up", "invented", "fabricated", "fake citation", "hallucinated"),
    AspectCategory.REFUSAL_ACCEPTANCE: ("refused", "refusal", "censored", "won't answer", "over-cautious", "too cautious"),
    AspectCategory.VALUE: ("price", "pricing", "worth", "expensive", "cheap", "subscription", "cost", "value"),
    AspectCategory.DEVELOPER_ERGONOMICS: ("api", "sdk", "docs", "latency", "rate limit", "timeout", "bug", "tooling"),
}

TASK_PRIORITY = (
    TaskCategory.API_DEV_WORKFLOW,
    TaskCategory.AGENTS,
    TaskCategory.CODING,
    TaskCategory.DATA_ANALYSIS,
    TaskCategory.LONG_CONTEXT,
    TaskCategory.MULTIMODAL,
    TaskCategory.RESEARCH,
    TaskCategory.WRITING,
    TaskCategory.ROLEPLAY,
    TaskCategory.GENERAL,
)

ASPECT_PRIORITY = (
    AspectCategory.HALLUCINATION_SAFETY,
    AspectCategory.REFUSAL_ACCEPTANCE,
    AspectCategory.REGRESSION_STABILITY,
    AspectCategory.TRUST_RELIABILITY,
    AspectCategory.VALUE,
    AspectCategory.DEVELOPER_ERGONOMICS,
    AspectCategory.TASK_FIT,
    AspectCategory.SATISFACTION,
)

POSITIVE_TERMS = (
    "best",
    "better",
    "good",
    "great",
    "excellent",
    "impressive",
    "impressed",
    "pleasantly surprised",
    "surprised us",
    "nailed it",
    "creative",
    "intelligent",
    "reliable",
    "works well",
    "useful",
    "solid",
    "faster",
    "improved",
    "nice",
    "love",
    "worth it",
    "accurate",
    "helpful",
    "robust",
    "smart",
    "amazing",
    "strong",
)

NEGATIVE_TERMS = (
    "worse",
    "bad",
    "terrible",
    "poor",
    "broken",
    "flaky",
    "fails",
    "failed",
    "regressed",
    "buggy",
    "slow",
    "slower",
    "crashed",
    "timeout",
    "timeouts",
    "unusable",
    "wrong",
    "mistake",
    "hallucination",
    "hallucinated",
    "made up",
    "invented",
    "fabricated",
    "refused",
    "over-cautious",
    "expensive",
    "regression",
    "degraded",
    "disappointing",
    "frustrating",
    "annoying",
    "useless",
)

FIRSTHAND_TERMS = (
    "i used",
    "i use",
    "i tried",
    "i've tried",
    "i have tried",
    "tested",
    "i tested",
    "using it",
    "using this",
    "been using",
    "i find",
    "i found",
    "my experience",
    "we used",
    "we use",
    "we've used",
    "we have used",
    "our team",
    "in my app",
    "in production",
    "my workflow",
)

COMPARATIVE_TERMS = ("better than", "worse than", "compared to", "vs", "versus", "switched from", "compared", "passing it on")
REGRESSION_TERMS = ("regression", "got worse", "became worse", "degraded", "nerfed", "used to be")
HALLUCINATION_TERMS = ("hallucination", "hallucinated", "made up", "invented", "fabricated", "fake citation")
REFUSAL_TERMS = ("refused", "refusal", "censored", "won't answer", "over-cautious", "safety filter")
VALUE_TERMS = ("price", "pricing", "worth", "expensive", "cheap", "subscription", "cost")
BENCHMARK_TERMS = ("benchmark", "benchmarks", "leaderboard", "eval", "evaluations", "scores")
NOISE_TERMS = ("automod", "bot", "debug", "traceback", "stack trace", "stderr", "stdout", "log output")
POLARITY_IGNORE_PHRASES = ("bad actors",)
INSTRUCTION_PREFIXES = (
    "install manually:",
    "run npx ",
    "give this prompt to your agent:",
    "ktx setup",
)
RELEASE_TERMS = (
    "release",
    "update",
    "new model",
    "changed",
    "rolled out",
    "deprecated",
    "preview",
    "beta",
    "announced",
    "announcement",
    "launch",
    "launched",
    "available",
    "rollout",
    "rolled out",
    "introduced",
    "deployed",
    "adoption",
)


@dataclass(slots=True)
class RuleBasedExtractor:
    catalog: ModelCatalog
    max_observations_per_item: int = 24

    @classmethod
    def default(cls) -> "RuleBasedExtractor":
        return cls(catalog=ModelCatalog())

    def extract_features(self, item: RawItem) -> CandidateFeatures:
        return self._extract_features_for_text(item, item.text)

    def _extract_features_for_text(self, item: RawItem, source_text: str) -> CandidateFeatures:
        text = normalize_text(source_text)
        lowered = text.lower()
        mentions = self.catalog.find_mentions(text)
        product_id = self.catalog.detect_product_id(text)
        inference_profile = self.catalog.detect_inference_profile(text)
        matched_terms: list[str] = []
        task_categories = _matched_categories(lowered, TASK_TERMS, matched_terms)
        aspect_categories = _matched_categories(lowered, ASPECT_TERMS, matched_terms)

        firsthand = _contains_any(lowered, FIRSTHAND_TERMS, matched_terms) or is_firsthand_text(text)
        comparative = _contains_any(lowered, COMPARATIVE_TERMS, matched_terms)
        regression = _contains_any(lowered, REGRESSION_TERMS, matched_terms)
        hallucination = _contains_any(lowered, HALLUCINATION_TERMS, matched_terms)
        refusal = _contains_any(lowered, REFUSAL_TERMS, matched_terms)
        value = _contains_any(lowered, VALUE_TERMS, matched_terms)
        release = _contains_any(lowered, RELEASE_TERMS, matched_terms)
        benchmark = _contains_any(lowered, BENCHMARK_TERMS, matched_terms)
        noise = _contains_any(lowered, NOISE_TERMS, matched_terms)

        polarity = _polarity_score(lowered, matched_terms)
        exclude_reason = _exclude_reason(
            text,
            firsthand=firsthand,
            comparative=comparative,
            regression=regression,
            hallucination=hallucination,
            refusal=refusal,
            value=value,
            release=release,
            benchmark=benchmark,
            noise=noise,
            polarity=polarity,
        )
        evidence_types = self._evidence_types(firsthand, comparative, regression, value, release, benchmark, text)
        if not task_categories:
            task_categories = [TaskCategory.GENERAL]
        else:
            task_categories = [_primary_category(lowered, TASK_TERMS, TASK_PRIORITY)]
        if not aspect_categories:
            aspect_categories = [AspectCategory.SATISFACTION]
        else:
            aspect_categories = [_primary_category(lowered, ASPECT_TERMS, ASPECT_PRIORITY)]
        severity = min(1.0, abs(polarity) / 2 + 0.15 * len({*aspect_categories}))
        relevant = bool(mentions and not exclude_reason and (firsthand or comparative or regression or hallucination or refusal or value or release or abs(polarity) > 0))
        confidence = _confidence(mentions, matched_terms, firsthand, item)

        return CandidateFeatures(
            source_item_id=item.source_id,
            evidence_text=text,
            model_mentions=mentions,
            product_id=product_id,
            inference_profile=inference_profile,
            task_categories=task_categories,
            aspect_categories=aspect_categories,
            evidence_types=evidence_types,
            polarity_score=polarity,
            severity_score=severity,
            extractor_confidence=confidence,
            firsthand_flag=firsthand,
            comparative_flag=comparative,
            regression_flag=regression,
            hallucination_flag=hallucination,
            refusal_flag=refusal,
            value_flag=value,
            relevant=relevant,
            matched_terms=sorted(set(matched_terms)),
        )

    def extract_observations(self, item: RawItem) -> list[Observation]:
        observations_by_key: dict[tuple[str, str, str, str, str, str], tuple[Observation, float]] = {}

        for span in _candidate_spans(item):
            features = self._extract_features_for_text(item, span)
            if not features.relevant:
                continue
            self._observations_from_features(item, features, observations_by_key)

        observations = [observation for observation, _ in observations_by_key.values()]
        observations.sort(key=lambda observation: (observation.aspect_category.value, observation.model_id))
        return observations[: self.max_observations_per_item]

    def _observations_from_features(
        self,
        item: RawItem,
        features: CandidateFeatures,
        observations_by_key: dict[tuple[str, str, str, str, str, str], tuple[Observation, float]],
    ) -> None:
        weights = self._weights(item, features)
        evidence_type = features.evidence_types[0] if features.evidence_types else EvidenceType.HEARSAY
        claim_type = _claim_type(features.polarity_score)
        for mention in features.model_mentions:
            for task in features.task_categories:
                for aspect in features.aspect_categories:
                    key = (
                        mention.model_id,
                        features.product_id or "",
                        features.inference_profile or "",
                        task.value,
                        aspect.value,
                        evidence_type.value,
                    )
                    observation = Observation(
                        source_platform=item.platform,
                        community_id=item.community_id,
                        thread_id=item.thread_id,
                        source_item_id=item.source_id,
                        author_id_hash=item.author_id_hash,
                        url=item.url,
                        published_at=item.published_at,
                        model_id=mention.model_id,
                        provider_id=mention.provider_id,
                        model_version_or_alias=mention.alias,
                        product_id=features.product_id,
                        inference_profile=features.inference_profile,
                        task_category=task,
                        aspect_category=aspect,
                        evidence_type=evidence_type,
                        claim_type=claim_type,
                        polarity_score=features.polarity_score,
                        severity_score=features.severity_score,
                        extractor_confidence=features.extractor_confidence,
                        firsthand_flag=features.firsthand_flag,
                        comparative_flag=features.comparative_flag,
                        regression_flag=features.regression_flag,
                        hallucination_flag=features.hallucination_flag,
                        refusal_flag=features.refusal_flag,
                        value_flag=features.value_flag,
                        weights=weights,
                        evidence_text=features.evidence_text,
                    )
                    score = _evidence_span_score(features.evidence_text, features)
                    existing = observations_by_key.get(key)
                    if existing is None or score > existing[1]:
                        observations_by_key[key] = (observation, score)

    def _weights(self, item: RawItem, features: CandidateFeatures) -> WeightComponents:
        source_quality = {
            "github": 1.25,
            "hacker_news": 1.15,
            "stack_exchange": 1.20,
            "reddit": 1.0,
            "hugging_face": 1.15,
        }.get(item.platform.value, 1.0)
        engagement_total = sum(float(value) for value in item.engagement.values() if isinstance(value, (int, float)))
        return WeightComponents(
            source_quality_weight=source_quality,
            firsthand_weight=1.25 if features.firsthand_flag else 0.75,
            author_credibility_weight=1.0,
            corroboration_weight=1.0,
            recency_weight=1.0,
            engagement_weight=min(1.4, 1.0 + log1p(max(0.0, engagement_total)) / 20),
            duplicate_penalty=1.0,
        )

    def _evidence_types(
        self,
        firsthand: bool,
        comparative: bool,
        regression: bool,
        value: bool,
        release: bool,
        benchmark: bool,
        text: str,
    ) -> list[EvidenceType]:
        evidence: list[EvidenceType] = []
        lowered = normalize_text(text).lower()
        if regression:
            evidence.append(EvidenceType.BUG_REGRESSION_REPORT)
        if "api" in lowered or "sdk" in lowered or "integration" in lowered or "bug" in lowered:
            evidence.append(EvidenceType.INTEGRATION_FAILURE)
        if comparative:
            evidence.append(EvidenceType.COMPARATIVE_EVALUATION)
        if benchmark:
            evidence.append(EvidenceType.BENCHMARK_ANECDOTE)
        if value:
            evidence.append(EvidenceType.PRICING_VALUE_COMMENT)
        if release:
            evidence.append(EvidenceType.RELEASE_UPDATE_REACTION)
        if firsthand:
            evidence.append(EvidenceType.FIRSTHAND_USAGE)
        if not evidence:
            evidence.append(EvidenceType.HEARSAY)
        return evidence


def _matched_categories(
    text: str,
    terms_by_category: dict,
    matched_terms: list[str],
) -> list:
    matches = []
    for category, terms in terms_by_category.items():
        if _contains_any(text, terms, matched_terms):
            matches.append(category)
    return matches


def _primary_category(text: str, terms_by_category: dict, priority: tuple):
    matched = {
        category: sum(1 for term in terms if term in text)
        for category, terms in terms_by_category.items()
    }
    return max(priority, key=lambda category: (matched.get(category, 0), -priority.index(category)))


def _contains_any(text: str, terms: Iterable[str], matched_terms: list[str] | None = None) -> bool:
    found = False
    for term in terms:
        if term in text:
            found = True
            if matched_terms is not None:
                matched_terms.append(term)
    return found


def _polarity_score(text: str, matched_terms: list[str]) -> int:
    polarity_text = text
    for phrase in POLARITY_IGNORE_PHRASES:
        polarity_text = polarity_text.replace(phrase, " ")
    positive = sum(1 for term in POSITIVE_TERMS if term in polarity_text)
    negative = sum(1 for term in NEGATIVE_TERMS if term in polarity_text)
    matched_terms.extend([term for term in POSITIVE_TERMS if term in polarity_text])
    matched_terms.extend([term for term in NEGATIVE_TERMS if term in polarity_text])
    net = positive - negative
    if net >= 2:
        return 2
    if net == 1:
        return 1
    if net == 0:
        return 0
    if net == -1:
        return -1
    return -2


def _claim_type(polarity: int) -> ClaimType:
    if polarity > 0:
        return ClaimType.PRAISE
    if polarity < 0:
        return ClaimType.COMPLAINT
    return ClaimType.NEUTRAL


def _confidence(mentions: list, matched_terms: list[str], firsthand: bool, item: RawItem) -> float:
    score = 0.25
    if mentions:
        score += 0.25
    if matched_terms:
        score += min(0.25, len(set(matched_terms)) * 0.03)
    if firsthand:
        score += 0.15
    if item.body and item.title:
        score += 0.05
    return min(0.95, score)


def _evidence_span_score(text: str, features: CandidateFeatures) -> float:
    lowered = text.lower()
    score = float(len(text.split()))
    score += 5.0 * len(features.model_mentions)
    score += 2.0 * len(features.matched_terms)
    score += 3.0 * abs(features.polarity_score)
    if any(term in lowered for term in COMPARATIVE_TERMS):
        score += 6.0
    if any(term in lowered for term in FIRSTHAND_TERMS):
        score += 4.0
    if any(term in lowered for term in REGRESSION_TERMS):
        score += 3.0
    if any(term in lowered for term in ("better than", "worse than", "compared to", "vs", "versus")):
        score += 5.0
    if _looks_like_quote_only_span(text):
        score -= 8.0
    if text.count(">") >= 1:
        score -= 3.0
    if len(features.model_mentions) > 1:
        score -= 2.5 * (len(features.model_mentions) - 1)
    if lowered.startswith(("as part of ", "according to ", "anthropic says ", "openai says ", "google says ")):
        score -= 4.0
    return score


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _candidate_spans(item: RawItem) -> list[str]:
    title, body = clean_item_text(item)
    spans: list[str] = []
    catalog = ModelCatalog()
    if title and not is_low_signal_title(title) and not _skip_title_only_story_span(item, body):
        spans.append(title)

    sentences = [part.strip() for part in SENTENCE_RE.split(body) if part.strip()]
    if not sentences and body:
        sentences = [body]

    title_has_model = bool(catalog.find_mentions(title)) if title else False
    for index, sentence in enumerate(sentences):
        for candidate in _span_candidates_for_sentence(title, title_has_model, sentences, index, catalog):
            spans.append(candidate)

    return [span for span in _unique_preserving_order(spans) if _has_signal_shape(span)]


def _unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _has_signal_shape(span: str) -> bool:
    lowered = span.lower()
    if _looks_like_question_only_span(span):
        return False
    if _looks_like_instructional_span(span):
        return False
    has_quality_term = any(
        term in lowered
        for term in (
            *POSITIVE_TERMS,
            *NEGATIVE_TERMS,
            *FIRSTHAND_TERMS,
            *COMPARATIVE_TERMS,
            *REGRESSION_TERMS,
            *HALLUCINATION_TERMS,
            *REFUSAL_TERMS,
            *VALUE_TERMS,
            *RELEASE_TERMS,
            *BENCHMARK_TERMS,
            "api",
            "sdk",
            "latency",
            "rate limit",
            "docs",
        )
    )
    return has_quality_term or bool(ModelCatalog().find_mentions(span))


def _sentence_window(sentences: list[str], index: int, radius: int = 1) -> str:
    start = max(0, index - radius)
    end = min(len(sentences), index + radius + 1)
    return " ".join(sentence for sentence in sentences[start:end] if sentence)


def _span_candidates_for_sentence(
    title: str,
    title_has_model: bool,
    sentences: list[str],
    index: int,
    catalog: ModelCatalog,
) -> list[str]:
    candidates: list[str] = []
    sentence = _trim_span_sentences(sentences[index].strip())
    if len(sentence) >= 12 and _window_has_signal(sentence, catalog):
        candidates.append(_prepend_title_context(title, sentence, title_has_model, catalog))
    window = _trim_span_sentences(_sentence_window(sentences, index))
    if len(window) >= 12 and _window_has_signal(window, catalog):
        candidates.append(_prepend_title_context(title, window, title_has_model, catalog))
    return candidates


def _prepend_title_context(title: str, span: str, title_has_model: bool, catalog: ModelCatalog) -> str:
    if title and title_has_model and not catalog.find_mentions(span):
        return f"{title}. {span}"
    return span


def _window_has_signal(window: str, catalog: ModelCatalog) -> bool:
    lowered = window.lower()
    return bool(catalog.find_mentions(window)) or any(
        term in lowered
        for term in (
            *POSITIVE_TERMS,
            *NEGATIVE_TERMS,
            *FIRSTHAND_TERMS,
            *COMPARATIVE_TERMS,
            *REGRESSION_TERMS,
            *HALLUCINATION_TERMS,
            *REFUSAL_TERMS,
            *VALUE_TERMS,
            *RELEASE_TERMS,
            *BENCHMARK_TERMS,
            "api",
            "sdk",
            "latency",
            "rate limit",
            "docs",
        )
    )


def _looks_like_quote_only_span(text: str) -> bool:
    stripped = text.strip()
    if len(stripped.split()) < 20 and stripped.startswith('"') and stripped.endswith('"'):
        return True
    return stripped.startswith(("\"users will", "\"what model are you", "\"we compared"))


def _looks_like_question_only_span(text: str) -> bool:
    stripped = text.strip()
    if not stripped.endswith("?"):
        return False
    lowered = stripped.lower()
    if any(term in lowered for term in (*POSITIVE_TERMS, *NEGATIVE_TERMS, *FIRSTHAND_TERMS, *COMPARATIVE_TERMS)):
        return False
    return True


def _trim_span_sentences(text: str) -> str:
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    while sentences and _should_drop_preface_sentence(sentences[0]):
        sentences.pop(0)
    while len(sentences) > 1 and _looks_like_question_only_span(sentences[0]):
        sentences.pop(0)
    while len(sentences) > 1 and _looks_like_question_only_span(sentences[-1]):
        sentences.pop()
    return " ".join(sentences).strip()


def _is_quoted_sentence(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.startswith(">")


def _should_drop_preface_sentence(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if _is_quoted_sentence(stripped):
        return True
    if _looks_like_release_preface_sentence(lowered):
        return True
    if _looks_like_instructional_span(stripped):
        return True
    return False


def _looks_like_release_preface_sentence(text: str) -> bool:
    return text.startswith(
        (
            "as part of ",
            "a small number of organizations are currently using",
            "models of this capability level require",
            "we’re making swift progress",
            "we're making swift progress",
            "we plan to release",
            "not only that, but we plan to release",
        )
    )


def _looks_like_instructional_span(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(prefix) for prefix in INSTRUCTION_PREFIXES)


def _skip_title_only_story_span(item: RawItem, body: str) -> bool:
    metadata = item.metadata or {}
    if item.platform != SourcePlatform.HACKER_NEWS:
        return False
    if body.strip():
        return False
    if not metadata.get("is_root_story"):
        return False
    title = (item.title or "").strip().lower()
    return not title.startswith(("ask hn:", "show hn:", "tell hn:"))


def _exclude_reason(
    text: str,
    *,
    firsthand: bool,
    comparative: bool,
    regression: bool,
    hallucination: bool,
    refusal: bool,
    value: bool,
    release: bool,
    benchmark: bool,
    noise: bool,
    polarity: int,
) -> str | None:
    if _looks_non_english_or_garbled(text):
        return "non_english_or_garbled"
    if noise and not (firsthand or comparative or regression or hallucination or refusal or value or abs(polarity) > 0):
        return "tooling_or_bot_transcript"
    if release and not (firsthand or comparative or regression or hallucination or refusal or value or abs(polarity) > 0):
        return "factual_release_or_adoption"
    if benchmark and not (firsthand or comparative or regression or hallucination or refusal or value or abs(polarity) > 0):
        return "benchmark_without_user_judgment"
    if not (firsthand or comparative or regression or hallucination or refusal or value or abs(polarity) > 0):
        return "not_about_model_quality"
    return None


def _looks_non_english_or_garbled(text: str) -> bool:
    if not text:
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    ascii_letters = [ch for ch in letters if ch.isascii()]
    non_ascii_ratio = 1.0 - (len(ascii_letters) / len(letters))
    cjk_present = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    token_count = len(text.split())
    return (cjk_present and token_count < 40) or (non_ascii_ratio > 0.55 and token_count < 30)
