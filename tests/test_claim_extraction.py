import unittest
from social_benchmark.pipeline.claim_extraction import split_claims
from social_benchmark.pipeline.clients.reddit_feed import parse_reddit_feed
class ClaimTests(unittest.TestCase):
 def test_mixed_aspects(self):
  rows=split_claims('GPT-6-Astra is the best model for vibe coding, but it is way too expensive')
  self.assertIn(('vibe_coding',1),[(r['category'],r['polarity_score']) for r in rows])
  self.assertIn(('cost_efficiency',-1),[(r['category'],r['polarity_score']) for r in rows])
  self.assertNotIn(('cost_efficiency',1),[(r['category'],r['polarity_score']) for r in rows])
 def test_negation(self):
  rows=split_claims('GPT-6.1-Sol is not good for coding')
  self.assertEqual(rows[0]['polarity_score'],-1)
 def test_explicit_comparison_keeps_direction(self):
  rows=split_claims('GPT-6-Astra is better than GPT-6.1-Sol for coding')
  self.assertEqual([(r['model_id'],r['polarity_score'],r['evidence_kind']) for r in rows],[('gpt-6-astra',1,'relative_preference'),('gpt-6.1-sol',-1,'relative_preference')])
 def test_adjacent_sentence_pronoun(self):
  self.assertEqual(len(split_claims('GPT-6-Astra is great for coding. It is expensive')),2)
 def test_spans(self):
  text='GPT-6-Astra is great for design, but it is expensive'
  for row in split_claims(text):self.assertEqual(text[row['span_start']:row['span_end']],row['evidence_text'])
 def test_rss_metadata(self):
  xml='''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>t3_abc</id><title>GPT-6-Astra</title><link href="https://www.reddit.com/r/OpenAI/comments/abc/test/"/><author><name>/u/example</name></author><published>2026-10-28T12:00:00Z</published><content>&lt;p&gt;Great for coding&lt;/p&gt;</content></entry></feed>'''
  row=parse_reddit_feed(xml)[0]
  self.assertEqual(row.community_id,'OpenAI');self.assertEqual(row.thread_id,'t3_abc');self.assertIsNone(row.parent_id);self.assertEqual(row.body,'Great for coding')

 def test_versioned_mentions_preserved(self):
  rows=split_claims('Claude Opus 5.5 is great for coding')
  self.assertEqual(rows[0]['model_id'],'claude-opus-5.5')
  rows=split_claims('Gemini 3.8 Flash is expensive')
  self.assertEqual(rows[0]['model_id'],'gemini-3.8-flash')

 def test_unnamed_competitor_not_praise_for_target(self):
  self.assertEqual(split_claims('A 4B model is better than GPT-6 Astra'),[])
 def test_question_not_positive_evidence(self):
  self.assertEqual(split_claims('Is GPT-6 Astra good for coding?'),[])
