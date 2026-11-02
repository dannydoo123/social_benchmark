import unittest
from datetime import datetime, timezone
from social_benchmark.pipeline.live_evidence import categories, summarize

class LiveEvidenceTests(unittest.TestCase):
    def row(self, **changes):
        row=dict(source_platform='hacker_news',source_item_id='1',model_id='gpt-6-astra',aspect_category='value',evidence_text='I found this model useful and cheap',published_at='2026-10-29T00:00:00+00:00',categories=['cost_efficiency'],review_status='machine_agreement',firsthand_flag=True,author_id_hash='author',thread_id='thread',community_id='hn',polarity_score=1,url='https://news.ycombinator.com/item?id=1')
        return dict(row,**changes)
    def test_deduplication_and_no_premature_score(self):
        rows=summarize([self.row(),self.row(source_item_id='2')],datetime(2026,10,30,tzinfo=timezone.utc))
        self.assertEqual(rows[0]['observations'],1)
        self.assertIsNone(rows[0]['score'])
    def test_old_missing_and_future_dates_excluded(self):
        self.assertEqual(summarize([self.row(published_at='2020-01-01T00:00:00Z'),self.row(published_at=None),self.row(published_at='2030-01-01T00:00:00Z')],datetime(2026,10,30,tzinfo=timezone.utc)),[])
    def test_training_not_implicitly_accepted(self):
        rows=summarize([self.row(review_status='pending')],datetime(2026,10,30,tzinfo=timezone.utc))
        self.assertEqual(rows[0]['accepted'],0)
    def test_categories(self):
        self.assertIn('design',categories('The UI layout looks good'))
        self.assertIn('vibe_coding',categories('vibe coding prototype'))
        self.assertNotIn('design',categories('build a quick guide'))

    def test_exact_new_model_aliases(self):
        from social_benchmark.pipeline.catalog import ModelCatalog
        catalog = ModelCatalog()
        self.assertIn('gpt-6-astra', {m.model_id for m in catalog.find_mentions('I used GPT-6 Astra')})
        self.assertIn('gpt-6.1-sol', {m.model_id for m in catalog.find_mentions('I used GPT-6.1 Sol')})
        self.assertNotIn('gpt-6-astra', {m.model_id for m in catalog.find_mentions('I used ChatGPT')})
