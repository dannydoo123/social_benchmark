import importlib.util
import unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('backfill',Path(__file__).resolve().parents[1]/'scripts/backfill_model_conversations.py')
backfill=importlib.util.module_from_spec(spec);spec.loader.exec_module(backfill)
class BackfillTests(unittest.TestCase):
 def test_nested_replies_keep_thread_and_parent(self):
  root={'id':1,'author':'a','title':'GPT-6 Astra','type':'story','children':[{'id':2,'author':'b','text':'Good for design','type':'comment','parent_id':1,'children':[{'id':3,'author':'c','text':'Too expensive','type':'comment','parent_id':2}]}]}
  rows=list(backfill.flatten_hn(root,1,root['title']))
  self.assertEqual(len(rows),3);self.assertEqual(rows[-1]['thread_id'],'1');self.assertEqual(rows[-1]['parent_id'],'2');self.assertEqual(rows[-1]['body'],'Too expensive')
  self.assertNotIn('model_id',rows[-1])
 def test_deleted_parent_does_not_delete_visible_descendants(self):
  root={'id':1,'children':[{'id':2,'author':'b','text':'Visible reply','type':'comment','parent_id':1}]}
  rows=list(backfill.flatten_hn(root,1,'Topic'))
  self.assertEqual(len(rows),1);self.assertEqual(rows[0]['parent_id'],'1')
 def test_full_model_boundary(self):
  self.assertTrue(backfill.PATTERNS['gpt-6.1-sol'].search('GPT 6.1 Sol'))
  self.assertFalse(backfill.PATTERNS['gpt-6.1-sol'].search('GPT-6 Sol'))
  self.assertFalse(backfill.PATTERNS['gpt-6-astra'].search('AstraZeneca'))
