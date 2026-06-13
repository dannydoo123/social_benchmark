import unittest

from social_benchmark.pipeline.clients.github import github_comment_to_raw_item, github_issue_to_raw_item
from social_benchmark.pipeline.clients.hackernews import HackerNewsClient, hn_item_to_raw_item
from social_benchmark.pipeline.extractors.rules import RuleBasedExtractor
from social_benchmark.pipeline.models import RawItem, SourcePlatform


class ClientMappingTest(unittest.TestCase):
    def test_hackernews_mapping_normalizes_story(self):
        raw = hn_item_to_raw_item(
            {
                "id": 123,
                "type": "story",
                "by": "alice",
                "time": 1710000000,
                "title": "Ask HN: GPT-4o for coding?",
                "text": "<p>I used it</p>",
                "score": 42,
                "descendants": 5,
            }
        )

        self.assertEqual(raw.platform, SourcePlatform.HACKER_NEWS)
        self.assertEqual(raw.source_id, "123")
        self.assertEqual(raw.thread_id, "123")
        self.assertEqual(raw.body, "I used it")
        self.assertEqual(raw.engagement["score"], 42)

    def test_hyphenated_model_aliases_normalize(self):
        item = RawItem(
            platform=SourcePlatform.HACKER_NEWS,
            source_id="124",
            title="I'm Claude Opus (claude-opus-4-8), running in Claude Code.",
            body="I used it in production and it worked well enough for me.",
            community_id="hacker_news",
        )

        observations = RuleBasedExtractor.default().extract_observations(item)

        self.assertIn("claude-opus-4.8", {observation.model_id for observation in observations})

    def test_github_issue_and_comment_mapping(self):
        issue = {
            "id": 10,
            "number": 7,
            "title": "Claude API regression",
            "body": "Requests timeout",
            "html_url": "https://github.com/acme/repo/issues/7",
            "user": {"id": 5, "login": "dev"},
            "created_at": "2026-05-01T00:00:00Z",
            "comments": 1,
            "reactions": {"total_count": 2},
            "labels": [{"name": "bug"}],
            "state": "open",
        }
        comment = {
            "id": 11,
            "body": "I see this in production too.",
            "html_url": "https://github.com/acme/repo/issues/7#issuecomment-11",
            "user": {"id": 6, "login": "ops"},
            "created_at": "2026-05-01T01:00:00Z",
            "reactions": {"total_count": 1},
        }

        raw_issue = github_issue_to_raw_item(issue, "acme/repo")
        raw_comment = github_comment_to_raw_item(comment, issue, "acme/repo")

        self.assertEqual(raw_issue.community_id, "acme/repo")
        self.assertEqual(raw_issue.thread_id, "7")
        self.assertIsNone(raw_issue.parent_id)
        self.assertEqual(raw_comment.parent_id, "10")
        self.assertEqual(raw_comment.thread_id, "7")

    def test_hackernews_comment_fetch_respects_depth_and_metadata(self):
        items = {
            100: {
                "id": 100,
                "type": "story",
                "by": "alice",
                "time": 1710000000,
                "title": "Ask HN: Claude?",
                "text": "Story body",
                "kids": [101],
                "score": 10,
                "descendants": 2,
            },
            101: {
                "id": 101,
                "type": "comment",
                "by": "bob",
                "time": 1710000100,
                "parent": 100,
                "text": "I used it and it was good.",
                "kids": [102],
                "score": 3,
            },
            102: {
                "id": 102,
                "type": "comment",
                "by": "carol",
                "time": 1710000200,
                "parent": 101,
                "text": "Deep reply should be cut off.",
                "kids": [],
                "score": 1,
            },
        }

        class FakeHackerNewsClient(HackerNewsClient):
            def get_item(self, item_id):  # type: ignore[override]
                return items.get(int(item_id))

        client = FakeHackerNewsClient()
        raw_items = client.fetch_story_with_comments(100, max_comments=10, max_depth=1)

        self.assertEqual([item.source_id for item in raw_items], ["100", "101"])
        self.assertEqual(raw_items[0].metadata["depth"], 0)
        self.assertEqual(raw_items[0].metadata["root_story_id"], "100")
        self.assertEqual(raw_items[1].metadata["depth"], 1)
        self.assertEqual(raw_items[1].metadata["comment_index"], 1)

    def test_false_context_mentions_are_dropped(self):
        from social_benchmark.pipeline.catalog import ModelCatalog

        catalog = ModelCatalog()
        self.assertEqual(
            [m.model_id for m in catalog.find_mentions("Head to gemini://gemi.dev in your Gemini browser")],
            [],
        )
        self.assertEqual(
            [m.model_id for m in catalog.find_mentions("depends who is hosting the model (together, openrouter, grok)")],
            [],
        )
        self.assertEqual(
            [m.model_id for m in catalog.find_mentions("Gemini 2.5 Pro handles large context well")],
            ["gemini-2.5"],
        )
        self.assertEqual(
            [m.model_id for m in catalog.find_mentions("Grok had an existential crisis")],
            ["grok"],
        )

    def test_hackernews_search_filters_and_paginates(self):
        pages = [
            {
                "hits": [
                    {"story_id": 200, "num_comments": 50},
                    {"objectID": "201", "num_comments": 2},
                    {"story_id": 202, "num_comments": 30},
                ],
                "nbPages": 2,
            },
            {
                "hits": [{"story_id": 203, "num_comments": 25}],
                "nbPages": 2,
            },
        ]
        requested_params = []

        class FakeSearchHttp:
            def get_json(self, path, params=None, headers=None):
                requested_params.append((path, params))
                return pages[params["page"]]

        client = HackerNewsClient(search_http=FakeSearchHttp())
        story_ids = client.search_story_ids("Claude", max_stories=3, min_comments=10)

        self.assertEqual(story_ids, [200, 202, 203])
        self.assertEqual(requested_params[0][0], "search")
        self.assertEqual(requested_params[0][1]["tags"], "story")


if __name__ == "__main__":
    unittest.main()
