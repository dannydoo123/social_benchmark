"""Collection, extraction, and scoring pipeline for Social Benchmark."""

from social_benchmark.pipeline.catalog import ModelCatalog
from social_benchmark.pipeline.extractors.rules import RuleBasedExtractor
from social_benchmark.pipeline.scoring import ScoreAggregator

__all__ = ["ModelCatalog", "RuleBasedExtractor", "ScoreAggregator"]

