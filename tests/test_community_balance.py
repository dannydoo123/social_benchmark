import unittest
from social_benchmark.pipeline.community_balance import balanced_candidate_score
class CommunityBalanceTests(unittest.TestCase):
 def row(self,cohort,author,polarity):return dict(cohort=cohort,platform='reddit',community=cohort,author_id_hash=author,thread_id='thread',polarity_score=polarity)
 def test_vendor_cannot_dominate(self):
  rows=[self.row('general','a',-1)]+[self.row('vendor',str(i),1) for i in range(1000)]
  result=balanced_candidate_score(rows)
  self.assertEqual(result['vendor_weight'],.2);self.assertEqual(result['score'],20)
 def test_only_vendor_has_no_aggregate(self):self.assertIsNone(balanced_candidate_score([self.row('vendor','a',1)])['score'])
 def test_duplicate_claims_do_not_gain_influence(self):
  rows=[self.row('general','a',1),self.row('general','b',-1)]
  self.assertEqual(balanced_candidate_score(rows),balanced_candidate_score(rows+[rows[0]]*100))
