import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

spec = importlib.util.spec_from_file_location('archive', Path(__file__).resolve().parents[1] / 'scripts/collect_reddit_archive.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class ArchiveTests(unittest.TestCase):
    def test_comment_retains_context_and_original_date(self):
        row = archive.normalize({'id': 'abc', 'link_id': 't3_post', 'parent_id': 't1_parent',
            'body': 'Great for coding', 'subreddit': 'ChatGPT', 'created_utc': 1700000000,
            'author': 'example'}, 'comment', {'ChatGPT': 'vendor'})
        self.assertEqual(row['source_id'], 't1_abc')
        self.assertEqual(row['parent_id'], 't1_parent')
        self.assertEqual(row['thread_id'], 't3_post')
        self.assertTrue(row['published_at'].startswith('2023-11-14'))
        self.assertEqual(row['metadata']['cohort'], 'vendor')
        self.assertNotEqual(row['author_id_hash'], 'example')
        self.assertNotIn('model_id', row)

    def test_removed_text_is_not_imported(self):
        self.assertIsNone(archive.normalize({'body': '[removed]'}, 'comment', {}))

    def test_access_denial_stops_and_cools_down(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            with patch.object(archive, 'STATE', root / 'state.json'), patch.object(archive, 'HEALTH', root / 'health.json'), \
                 patch.object(archive.time, 'sleep'), patch.object(archive.sqlite3, 'connect') as db, \
                 patch.object(archive, 'urlopen', side_effect=HTTPError('url', 403, 'Forbidden', {}, None)) as fetch:
                archive.collect(30)
                archive.collect(30)
                self.assertEqual(fetch.call_count, 1)
                health = json.loads((root / 'health.json').read_text())['sources'][0]
                self.assertEqual(health['status'], 'failed')
                self.assertEqual(health['new_unique_items'], 0)
                self.assertEqual(health['requests'], 1)

    def test_budget_cannot_be_unbounded(self):
        with self.assertRaises(ValueError):
            archive.collect(0)
