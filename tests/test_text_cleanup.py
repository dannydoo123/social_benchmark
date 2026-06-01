import unittest

from social_benchmark.pipeline.models import RawItem, SourcePlatform
from social_benchmark.pipeline.text_cleanup import clean_item_text


class TextCleanupTest(unittest.TestCase):
    def test_github_cleanup_removes_template_without_model_signal(self):
        item = RawItem(
            platform=SourcePlatform.GITHUB,
            source_id="1",
            title="[Task] TASK-1",
            body="""### Expected behavior
- [ ] Fill checklist
### Actual behavior
GPT-5 API timeouts are slower than before.
```log
irrelevant stacktrace
```
""",
        )

        _, body = clean_item_text(item)

        self.assertIn("GPT-5 API timeouts", body)
        self.assertNotIn("Fill checklist", body)
        self.assertNotIn("irrelevant stacktrace", body)


if __name__ == "__main__":
    unittest.main()

