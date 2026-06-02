import time
import unittest

from social_benchmark.pipeline.collection import _collect_hn
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


if __name__ == "__main__":
    unittest.main()
