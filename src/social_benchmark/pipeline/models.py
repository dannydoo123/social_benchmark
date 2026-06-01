from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SourcePlatform(str, Enum):
    HACKER_NEWS = "hacker_news"
    GITHUB = "github"
    STACK_EXCHANGE = "stack_exchange"
    REDDIT = "reddit"
    HUGGING_FACE = "hugging_face"


class TaskCategory(str, Enum):
    CODING = "coding"
    WRITING = "writing"
    RESEARCH = "research"
    AGENTS = "agents"
    ROLEPLAY = "roleplay"
    DATA_ANALYSIS = "data_analysis"
    LONG_CONTEXT = "long_context"
    MULTIMODAL = "multimodal"
    API_DEV_WORKFLOW = "api_developer_workflow"
    GENERAL = "general"


class AspectCategory(str, Enum):
    SATISFACTION = "satisfaction"
    TRUST_RELIABILITY = "trust_reliability"
    TASK_FIT = "task_fit"
    REGRESSION_STABILITY = "regression_stability"
    HALLUCINATION_SAFETY = "hallucination_safety"
    REFUSAL_ACCEPTANCE = "refusal_acceptance"
    VALUE = "value"
    DEVELOPER_ERGONOMICS = "developer_ergonomics"


class EvidenceType(str, Enum):
    FIRSTHAND_USAGE = "firsthand_usage"
    COMPARATIVE_EVALUATION = "comparative_evaluation"
    BUG_REGRESSION_REPORT = "bug_regression_report"
    INTEGRATION_FAILURE = "integration_failure"
    BENCHMARK_ANECDOTE = "benchmark_anecdote"
    HEARSAY = "hearsay"
    RELEASE_UPDATE_REACTION = "release_update_reaction"
    PRICING_VALUE_COMMENT = "pricing_value_comment"


class ClaimType(str, Enum):
    PRAISE = "praise"
    COMPLAINT = "complaint"
    MIXED = "mixed"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class RawItem:
    platform: SourcePlatform
    source_id: str
    title: str = ""
    body: str = ""
    url: str = ""
    community_id: str = ""
    thread_id: str | None = None
    parent_id: str | None = None
    author_id_hash: str | None = None
    author_handle: str | None = None
    published_at: datetime | None = None
    engagement: dict[str, float | int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        if self.title and self.body:
            return f"{self.title}\n\n{self.body}"
        return self.title or self.body


@dataclass(frozen=True, slots=True)
class ModelMention:
    model_id: str
    provider_id: str
    alias: str
    start: int
    end: int


@dataclass(slots=True)
class CandidateFeatures:
    source_item_id: str
    evidence_text: str = ""
    model_mentions: list[ModelMention] = field(default_factory=list)
    product_id: str | None = None
    inference_profile: str | None = None
    task_categories: list[TaskCategory] = field(default_factory=list)
    aspect_categories: list[AspectCategory] = field(default_factory=list)
    evidence_types: list[EvidenceType] = field(default_factory=list)
    polarity_score: int = 0
    severity_score: float = 0.0
    extractor_confidence: float = 0.0
    firsthand_flag: bool = False
    comparative_flag: bool = False
    regression_flag: bool = False
    hallucination_flag: bool = False
    refusal_flag: bool = False
    value_flag: bool = False
    relevant: bool = False
    matched_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WeightComponents:
    source_quality_weight: float = 1.0
    firsthand_weight: float = 1.0
    author_credibility_weight: float = 1.0
    corroboration_weight: float = 1.0
    recency_weight: float = 1.0
    engagement_weight: float = 1.0
    duplicate_penalty: float = 1.0

    @property
    def final_weight(self) -> float:
        return (
            self.source_quality_weight
            * self.firsthand_weight
            * self.author_credibility_weight
            * self.corroboration_weight
            * self.recency_weight
            * self.engagement_weight
            * self.duplicate_penalty
        )


@dataclass(slots=True)
class Observation:
    source_platform: SourcePlatform
    community_id: str
    thread_id: str | None
    source_item_id: str
    author_id_hash: str | None
    url: str
    published_at: datetime | None
    model_id: str
    provider_id: str
    model_version_or_alias: str
    task_category: TaskCategory
    aspect_category: AspectCategory
    evidence_type: EvidenceType
    claim_type: ClaimType
    polarity_score: int
    severity_score: float
    extractor_confidence: float
    firsthand_flag: bool
    comparative_flag: bool
    regression_flag: bool
    hallucination_flag: bool
    refusal_flag: bool
    value_flag: bool
    weights: WeightComponents
    duplicate_cluster_id: str | None = None
    extractor_model_name: str = "rule_based_v0"
    extractor_model_version: str = "0.1.0"
    human_labeled_flag: bool = False
    evidence_text: str = ""
    product_id: str | None = None
    inference_profile: str | None = None

    @property
    def final_weight(self) -> float:
        return self.weights.final_weight

    @property
    def mapped_score(self) -> float:
        return ((self.polarity_score + 2) / 4) * 100


@dataclass(slots=True)
class ScoreSnapshot:
    model_id: str
    aspect_category: str
    score: float
    weighted_n: float
    effective_n: float
    confidence_low: float | None = None
    confidence_high: float | None = None
    warnings: list[str] = field(default_factory=list)
    source_mix: dict[str, float] = field(default_factory=dict)
    capped_weighted_n: float | None = None
    publishable: bool = False
    publication_blockers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReleaseUpdate:
    provider_id: str
    model_id: str
    release_or_update_name: str
    announced_at: datetime | None
    effective_at: datetime | None
    source_url: str
    capability_claims: list[str] = field(default_factory=list)
    pricing_or_limit_changes: list[str] = field(default_factory=list)
    deprecation_or_routing_changes: list[str] = field(default_factory=list)
    expected_affected_categories: list[str] = field(default_factory=list)
    notes: str = ""


def utc_from_unix(timestamp: int | float | None) -> datetime | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    return value
