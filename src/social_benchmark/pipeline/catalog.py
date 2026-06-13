from __future__ import annotations

import re
from dataclasses import dataclass

from social_benchmark.pipeline.models import ModelMention


@dataclass(frozen=True, slots=True)
class ModelAlias:
    alias: str
    model_id: str
    provider_id: str


@dataclass(frozen=True, slots=True)
class ProductAlias:
    alias: str
    product_id: str
    provider_id: str


@dataclass(frozen=True, slots=True)
class InferenceProfileAlias:
    alias: str
    profile_id: str


DEFAULT_ALIASES: tuple[ModelAlias, ...] = (
    ModelAlias("gpt-5.5", "gpt-5.5", "openai"),
    ModelAlias("gpt-5.4-mini", "gpt-5.4-mini", "openai"),
    ModelAlias("gpt-5.4-nano", "gpt-5.4-nano", "openai"),
    ModelAlias("gpt-5.4", "gpt-5.4", "openai"),
    ModelAlias("gpt-5", "gpt-5", "openai"),
    ModelAlias("gpt-4.1", "gpt-4.1", "openai"),
    ModelAlias("gpt-4o", "gpt-4o", "openai"),
    ModelAlias("gpt-4", "gpt-4", "openai"),
    ModelAlias("o3", "o3", "openai"),
    ModelAlias("o4-mini", "o4-mini", "openai"),
    ModelAlias("claude opus 4.8", "claude-opus-4.8", "anthropic"),
    ModelAlias("opus 4.8", "claude-opus-4.8", "anthropic"),
    ModelAlias("claude-opus-4-8", "claude-opus-4.8", "anthropic"),
    ModelAlias("opus-4-8", "claude-opus-4.8", "anthropic"),
    ModelAlias("claude opus 4.7", "claude-opus-4.7", "anthropic"),
    ModelAlias("opus 4.7", "claude-opus-4.7", "anthropic"),
    ModelAlias("claude-opus-4-7", "claude-opus-4.7", "anthropic"),
    ModelAlias("opus-4-7", "claude-opus-4.7", "anthropic"),
    ModelAlias("claude opus 4.5", "claude-opus-4.5", "anthropic"),
    ModelAlias("opus 4.5", "claude-opus-4.5", "anthropic"),
    ModelAlias("claude sonnet 4.6", "claude-sonnet-4.6", "anthropic"),
    ModelAlias("sonnet 4.6", "claude-sonnet-4.6", "anthropic"),
    ModelAlias("claude-sonnet-4-6", "claude-sonnet-4.6", "anthropic"),
    ModelAlias("sonnet-4-6", "claude-sonnet-4.6", "anthropic"),
    ModelAlias("claude sonnet 4.7", "claude-sonnet-4.7", "anthropic"),
    ModelAlias("sonnet 4.7", "claude-sonnet-4.7", "anthropic"),
    ModelAlias("claude-sonnet-4-7", "claude-sonnet-4.7", "anthropic"),
    ModelAlias("sonnet-4-7", "claude-sonnet-4.7", "anthropic"),
    ModelAlias("claude sonnet 4.5", "claude-sonnet-4.5", "anthropic"),
    ModelAlias("sonnet 4.5", "claude-sonnet-4.5", "anthropic"),
    ModelAlias("claude haiku 4.5", "claude-haiku-4.5", "anthropic"),
    ModelAlias("haiku 4.5", "claude-haiku-4.5", "anthropic"),
    ModelAlias("claude opus", "claude-opus", "anthropic"),
    ModelAlias("claude sonnet", "claude-sonnet", "anthropic"),
    ModelAlias("claude haiku", "claude-haiku", "anthropic"),
    ModelAlias("claude", "claude", "anthropic"),
    ModelAlias("gemini 3.5 flash", "gemini-3.5-flash", "google"),
    ModelAlias("gemini 3.0 flash", "gemini-3.0-flash", "google"),
    ModelAlias("gemini 3.0 flash lite", "gemini-3.0-flash-lite", "google"),
    ModelAlias("gemini 3.0 flash-lite", "gemini-3.0-flash-lite", "google"),
    ModelAlias("gemini 3.0 pro", "gemini-3.0-pro", "google"),
    ModelAlias("gemini 3.0", "gemini-3.0", "google"),
    ModelAlias("gemini 3.1 pro", "gemini-3.1-pro", "google"),
    ModelAlias("gemini 3.1 flash-lite", "gemini-3.1-flash-lite", "google"),
    ModelAlias("gemini 3.1 flash lite", "gemini-3.1-flash-lite", "google"),
    ModelAlias("gemini 3 flash", "gemini-3-flash", "google"),
    ModelAlias("gemini 2.5", "gemini-2.5", "google"),
    ModelAlias("gemini 2.0", "gemini-2.0", "google"),
    ModelAlias("gemini", "gemini", "google"),
    ModelAlias("deepseek r1", "deepseek-r1", "deepseek"),
    ModelAlias("deepseek", "deepseek", "deepseek"),
    ModelAlias("grok", "grok", "xai"),
    ModelAlias("llama 4", "llama-4", "meta"),
    ModelAlias("llama 3", "llama-3", "meta"),
    ModelAlias("llama", "llama", "meta"),
    ModelAlias("mistral large", "mistral-large", "mistral"),
    ModelAlias("mistral", "mistral", "mistral"),
    ModelAlias("qwen", "qwen", "alibaba"),
    ModelAlias("kimi", "kimi", "moonshot"),
    ModelAlias("command r", "command-r", "cohere"),
)


DEFAULT_PRODUCT_ALIASES: tuple[ProductAlias, ...] = (
    ProductAlias("claude code", "claude-code", "anthropic"),
    ProductAlias("chatgpt", "chatgpt", "openai"),
    ProductAlias("openai api", "openai-api", "openai"),
    ProductAlias("anthropic api", "anthropic-api", "anthropic"),
    ProductAlias("gemini api", "gemini-api", "google"),
    ProductAlias("google ai studio", "google-ai-studio", "google"),
    ProductAlias("github copilot", "github-copilot", "github"),
    ProductAlias("cursor", "cursor", "cursor"),
)


DEFAULT_INFERENCE_PROFILE_ALIASES: tuple[InferenceProfileAlias, ...] = (
    InferenceProfileAlias("ultracode mode", "ultracode"),
    InferenceProfileAlias("ultra code mode", "ultracode"),
    InferenceProfileAlias("ultracode", "ultracode"),
    InferenceProfileAlias("ultra code", "ultracode"),
    InferenceProfileAlias("high effort", "high_effort"),
    InferenceProfileAlias("medium effort", "medium_effort"),
    InferenceProfileAlias("low effort", "low_effort"),
    InferenceProfileAlias("thinking mode", "thinking"),
    InferenceProfileAlias("extended thinking", "extended_thinking"),
)


class ModelCatalog:
    def __init__(
        self,
        aliases: tuple[ModelAlias, ...] = DEFAULT_ALIASES,
        product_aliases: tuple[ProductAlias, ...] = DEFAULT_PRODUCT_ALIASES,
        inference_profile_aliases: tuple[InferenceProfileAlias, ...] = DEFAULT_INFERENCE_PROFILE_ALIASES,
    ) -> None:
        self.aliases = sorted(
            aliases,
            key=lambda item: (len(item.alias), item.alias.count("-") + item.alias.count(".")),
            reverse=True,
        )
        self._patterns = [
            (
                alias,
                re.compile(rf"(?<![a-zA-Z0-9]){re.escape(alias.alias)}(?![a-zA-Z0-9])", re.IGNORECASE),
            )
            for alias in self.aliases
        ]
        self.product_aliases = sorted(
            product_aliases,
            key=lambda item: (len(item.alias), item.alias.count("-") + item.alias.count(".")),
            reverse=True,
        )
        self.inference_profile_aliases = sorted(
            inference_profile_aliases,
            key=lambda item: (len(item.alias), item.alias.count("-") + item.alias.count(".")),
            reverse=True,
        )
        self._product_patterns = [
            (
                alias,
                re.compile(rf"(?<![a-zA-Z0-9]){re.escape(alias.alias)}(?![a-zA-Z0-9])", re.IGNORECASE),
            )
            for alias in self.product_aliases
        ]
        self._inference_profile_patterns = [
            (
                alias,
                re.compile(rf"(?<![a-zA-Z0-9]){re.escape(alias.alias)}(?![a-zA-Z0-9])", re.IGNORECASE),
            )
            for alias in self.inference_profile_aliases
        ]

    def find_mentions(self, text: str) -> list[ModelMention]:
        mentions: list[ModelMention] = []
        occupied: list[tuple[int, int]] = []
        for alias, pattern in self._patterns:
            for match in pattern.finditer(text):
                span = match.span()
                if any(not (span[1] <= taken[0] or span[0] >= taken[1]) for taken in occupied):
                    continue
                mentions.append(
                    ModelMention(
                        model_id=alias.model_id,
                        provider_id=alias.provider_id,
                        alias=match.group(0),
                        start=span[0],
                        end=span[1],
                    )
                )
                occupied.append(span)
        mentions = [mention for mention in mentions if _mention_context_allows(text, mention)]
        return _drop_generic_mentions(sorted(mentions, key=lambda item: item.start))

    def detect_product_id(self, text: str) -> str | None:
        for alias, pattern in self._product_patterns:
            for match in pattern.finditer(text):
                if _product_context_allows(text, match.start(), match.end()):
                    return alias.product_id
        return None

    def detect_inference_profile(self, text: str) -> str | None:
        for alias, pattern in self._inference_profile_patterns:
            if pattern.search(text):
                return alias.profile_id
        return None


GENERIC_MODEL_IDS = {"claude", "claude-opus", "claude-sonnet", "claude-haiku", "gemini", "deepseek", "llama", "mistral"}

# Text-level signals that a matched alias is not the LLM at all.
GEMINI_PROTOCOL_SIGNALS = (
    "gemini://",
    "geminispace",
    "gemini protocol",
    "circumlunar.space",
    "gemini browser",
    "gemini capsule",
)
GROQ_HOSTING_SIGNALS = (
    "openrouter",
    "hosting the model",
    "hosted on",
    "hosted model",
    "hosting provider",
    "inference provider",
)


def _mention_context_allows(text: str, mention: "ModelMention") -> bool:
    lowered = text.lower()
    if mention.provider_id == "google" and mention.model_id.startswith("gemini"):
        if any(signal in lowered for signal in GEMINI_PROTOCOL_SIGNALS):
            return False
    if mention.model_id.startswith("grok"):
        window = lowered[max(0, mention.start - 80) : mention.end + 80]
        if any(signal in window for signal in GROQ_HOSTING_SIGNALS):
            return False
    return True


def _drop_generic_mentions(mentions: list[ModelMention]) -> list[ModelMention]:
    providers_with_specific = {
        mention.provider_id
        for mention in mentions
        if mention.model_id not in GENERIC_MODEL_IDS
    }
    return [
        mention
        for mention in mentions
        if mention.model_id not in GENERIC_MODEL_IDS or mention.provider_id not in providers_with_specific
    ]


PRODUCT_CONTEXT_TERMS = (
    "in ",
    "on ",
    "using ",
    "with ",
    "via ",
    "inside ",
    "running in ",
    "switched to ",
    "switch to ",
    "worked in ",
    "for ",
    "through ",
    "from ",
    "at ",
    "my ",
    "our ",
)


def _product_context_allows(text: str, start: int, end: int, window: int = 24) -> bool:
    lowered = text.lower()
    before = lowered[max(0, start - window):start]
    after = lowered[end:min(len(lowered), end + window)]
    if "/" in before or "/" in after:
        return False
    if " says" in after or " said" in after or " think" in after:
        return False
    return any(term in before or term in after for term in PRODUCT_CONTEXT_TERMS)
