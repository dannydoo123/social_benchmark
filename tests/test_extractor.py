import unittest

from social_benchmark.pipeline.extractors.rules import RuleBasedExtractor
from social_benchmark.pipeline.models import AspectCategory, RawItem, SourcePlatform, TaskCategory


class RuleBasedExtractorTest(unittest.TestCase):
    def test_extracts_firsthand_negative_regression_observation(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="1",
            title="Claude got worse for coding",
            body="I used Claude Sonnet in production and it got worse at Python debugging.",
            community_id="hacker_news",
        )

        observations = RuleBasedExtractor.default().extract_observations(item)

        self.assertTrue(observations)
        self.assertIn("claude-sonnet", {observation.model_id for observation in observations})
        self.assertIn(TaskCategory.CODING, {observation.task_category for observation in observations})
        self.assertIn(
            AspectCategory.REGRESSION_STABILITY,
            {observation.aspect_category for observation in observations},
        )
        self.assertTrue(any(observation.firsthand_flag for observation in observations))
        self.assertTrue(any(observation.regression_flag for observation in observations))
        self.assertTrue(any(observation.polarity_score < 0 for observation in observations))
        self.assertTrue(any("production" in observation.evidence_text.lower() for observation in observations))

    def test_ignores_items_without_model_mentions(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="2",
            title="Generic developer tooling complaint",
            body="The docs are bad and the API is flaky.",
            community_id="hacker_news",
        )

        observations = RuleBasedExtractor.default().extract_observations(item)

        self.assertEqual(observations, [])

    def test_separates_product_model_and_inference_profile(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="3",
            title="Claude Code with Opus 4.8 nailed this",
            body=(
                "Claude Code with Opus 4.8 in ultracode mode nailed it, the best result so far. "
                "The prompt was: Create a simple but functional real time strategy game."
            ),
            community_id="hacker_news",
        )

        observations = RuleBasedExtractor.default().extract_observations(item)

        self.assertTrue(observations)
        self.assertIn("claude-opus-4.8", {observation.model_id for observation in observations})
        self.assertNotIn("claude-code", {observation.model_id for observation in observations})
        self.assertTrue(all(observation.provider_id == "anthropic" for observation in observations))
        self.assertTrue(all(observation.product_id == "claude-code" for observation in observations))
        self.assertTrue(any(observation.inference_profile == "ultracode" for observation in observations))
        self.assertIn(TaskCategory.CODING, {observation.task_category for observation in observations})

    def test_ignores_product_names_in_comparison_lists(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="7",
            title="Which model is best?",
            body="ChatGPT/Gemini/Claude/Qwen/... says: Opus 4.8 is still the best option.",
            community_id="hacker_news",
        )

        observations = RuleBasedExtractor.default().extract_observations(item)

        self.assertTrue(observations)
        self.assertTrue(all(observation.product_id in (None, "") for observation in observations))

    def test_prefers_richer_body_context_over_short_title_summary(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="4",
            title="We compared Opus 4.7 and GPT-5.5 (among others).",
            body=(
                "Opus 4.7 came up with the most creative and intelligent API design that pleasantly surprised us, "
                "especially given that GPT-5.5 was passing it on various coding benchmarks."
            ),
            community_id="hacker_news",
        )

        observations = RuleBasedExtractor.default().extract_observations(item)

        self.assertTrue(observations)
        self.assertIn("claude-opus-4.7", {observation.model_id for observation in observations})
        self.assertTrue(
            any(
                "most creative and intelligent api design" in observation.evidence_text.lower()
                for observation in observations
            )
        )
        self.assertTrue(
            any("passing it on various coding benchmarks" in observation.evidence_text.lower() for observation in observations)
        )

    def test_excludes_release_only_announcements(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="5",
            title="Claude Mythos Preview is now available",
            body="As part of Project Glasswing, a small number of organizations are currently using Claude Mythos Preview for cybersecurity work.",
            community_id="hacker_news",
        )

        observations = RuleBasedExtractor.default().extract_observations(item)

        self.assertEqual(observations, [])

    def test_excludes_benchmark_only_rows_without_user_judgment(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="6",
            title="GPT-5.5 benchmark results",
            body="GPT-5.5 scored well on multiple coding benchmarks but no one described an actual user experience.",
            community_id="hacker_news",
        )

        observations = RuleBasedExtractor.default().extract_observations(item)

        self.assertEqual(observations, [])

    def test_evidence_type_uses_selected_span_not_whole_item(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="8",
            title="Claude pricing discussion",
            body=(
                "Claude is too expensive for small teams. "
                "Our internal parser hit a bug yesterday, but that was unrelated."
            ),
            community_id="hacker_news",
        )

        observations = RuleBasedExtractor.default().extract_observations(item)

        self.assertTrue(observations)
        self.assertTrue(any(observation.evidence_type.value == "pricing_value_comment" for observation in observations))
        self.assertFalse(all(observation.evidence_type.value == "integration_failure" for observation in observations))

    def test_penalizes_quote_only_metadata_spans(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="9",
            title="Claude Opus 4.8 in Claude Code",
            body=(
                "Claude Opus 4.8 in Claude Code is bad and still hides basic failures behind polished language. "
                "\"what model are you I'm Claude Opus (claude-opus-4-8), running in Claude Code.\""
            ),
            community_id="hacker_news",
        )

        observations = RuleBasedExtractor.default().extract_observations(item)

        self.assertTrue(observations)
        self.assertTrue(any("is bad" in observation.evidence_text.lower() for observation in observations))

    def test_skips_title_only_hn_root_story_without_body(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="10",
            title="DeepSeek Slashes AI Costs to Cents",
            body="",
            community_id="hacker_news",
            thread_id="10",
            metadata={"is_root_story": True, "type": "story"},
        )

        observations = RuleBasedExtractor.default().extract_observations(item)

        self.assertEqual(observations, [])

    def test_strips_leading_quote_sentences_from_candidate_windows(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="11",
            title="",
            body=(
                "> And fast mode for Opus 4.8 is now three times cheaper than it was for previous models. "
                "this is what I'm happy about, if true. "
                "Opus 4.7 is frustratingly slow."
            ),
            community_id="hacker_news",
        )

        observations = RuleBasedExtractor.default().extract_observations(item)

        self.assertTrue(observations)
        self.assertFalse(any(observation.evidence_text.strip().startswith(">") for observation in observations))
        self.assertTrue(any("frustratingly slow" in observation.evidence_text.lower() for observation in observations))

    def test_skips_instructional_install_spans(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="12",
            title="Show HN: Ktx",
            body=(
                "Install manually: npm install -g @kaelio/ktx ktx setup "
                "Or give this prompt to your agent: Run npx skills add Kaelio/ktx --skill ktx."
            ),
            community_id="hacker_news",
        )

        observations = RuleBasedExtractor.default().extract_observations(item)

        self.assertEqual(observations, [])

    def test_maps_sandbox_and_safety_language_to_trust_reliability(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="13",
            title="",
            body="I used Claude in a separate sandbox and it still feels unsafe to trust with secrets on my machine.",
            community_id="hacker_news",
        )

        observations = RuleBasedExtractor.default().extract_observations(item)

        self.assertTrue(observations)
        self.assertIn(
            AspectCategory.TRUST_RELIABILITY,
            {observation.aspect_category for observation in observations},
        )

    def test_uses_one_primary_task_and_aspect_per_evidence_span(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="14",
            title="",
            body=(
                "I told a Claude agent to update AGENTS.md and it attempted to grant itself permission, "
                "deleted its VM, and made the sandbox unsafe."
            ),
            community_id="hacker_news",
        )

        features = RuleBasedExtractor.default().extract_features(item)

        self.assertEqual(features.task_categories, [TaskCategory.AGENTS])
        self.assertEqual(features.aspect_categories, [AspectCategory.TRUST_RELIABILITY])

    def test_does_not_emit_writing_task_for_agentic_coding_span(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="15",
            title="",
            body="I use Claude agents for coding so I am not manually writing code anymore.",
            community_id="hacker_news",
        )

        features = RuleBasedExtractor.default().extract_features(item)

        self.assertEqual(len(features.task_categories), 1)
        self.assertNotEqual(features.task_categories, [TaskCategory.WRITING])


if __name__ == "__main__":
    unittest.main()
