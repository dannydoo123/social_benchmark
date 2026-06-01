import unittest

from social_benchmark.pipeline.firsthand import is_firsthand_text


class FirsthandDetectionTest(unittest.TestCase):
    def test_detects_direct_usage(self):
        self.assertTrue(is_firsthand_text("I used Claude Code in production last week."))
        self.assertTrue(is_firsthand_text("We see GPT-5 timeouts in our app."))
        self.assertTrue(is_firsthand_text("My experience with Gemini was bad."))

    def test_rejects_secondhand_or_release_language(self):
        self.assertFalse(is_firsthand_text("People say Claude is better now."))
        self.assertFalse(is_firsthand_text("The release notes claim GPT-5 improved coding."))


if __name__ == "__main__":
    unittest.main()

