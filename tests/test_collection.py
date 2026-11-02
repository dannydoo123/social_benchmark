import time
import unittest
from unittest.mock import patch

from social_benchmark.pipeline.collection import _collect_hn, collect_from_config
from social_benchmark.pipeline.models import RawItem, SourcePlatform


class HackerNewsCollectionTest(unittest.TestCase):
    def test_parallel_story_collection_preserves_feed_order(self):
        class FakeHackerNewsClient:
            def top_story_ids(self):
                return [100, 200]

            new_story_ids = top_story_ids
            best_story_ids = top_story_ids
            ask_story_ids = top_story_ids

            def fetch_story_with_comments(self, story_id, max_comments, max_depth):
                if story_id == 100:
                    time.sleep(0.02)
                return [
                    RawItem(
                        platform=SourcePlatform.HACKER_NEWS,
                        source_id=str(story_id),
                        thread_id=str(story_id),
                    )
                ]

        rows = _collect_hn(
            FakeHackerNewsClient(),
            {"kind": "top", "limit": 2, "comments": 10, "max_depth": 4, "workers": 2},
        )

        self.assertEqual([row.source_id for row in rows], ["100", "200"])

    def test_config_collects_reddit_subreddits_and_searches(self):
        class FakeRedditClient:
            def fetch_subreddit_posts(
                self,
                subreddit,
                sort,
                time_filter,
                limit,
                include_comments,
                max_comments,
                max_depth,
            ):
                return [
                    RawItem(
                        platform=SourcePlatform.REDDIT,
                        source_id=f"t3_{subreddit.lower()}",
                        community_id=f"r/{subreddit}",
                    )
                ]

            def search_subreddit_posts(
                self,
                subreddit,
                query,
                sort,
                time_filter,
                limit,
                include_comments,
                max_comments,
                max_depth,
            ):
                return [
                    RawItem(
                        platform=SourcePlatform.REDDIT,
                        source_id=f"t3_{subreddit.lower()}_search",
                        community_id=f"r/{subreddit}",
                    )
                ]

        config = {
            "reddit_subreddits": [
                {"name": "ClaudeAI", "sort": "top", "time": "month", "limit": 10, "comments": 20}
            ],
            "reddit_searches": [
                {"subreddits": ["LocalLLaMA", "ChatGPT"], "query": "Claude vs GPT", "limit": 5}
            ],
        }

        with patch("social_benchmark.pipeline.collection.RedditClient", FakeRedditClient):
            rows = collect_from_config(config)

        self.assertEqual(
            [(row.source_id, row.community_id) for row in rows],
            [
                ("t3_claudeai", "r/ClaudeAI"),
                ("t3_localllama_search", "r/LocalLLaMA"),
                ("t3_chatgpt_search", "r/ChatGPT"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
