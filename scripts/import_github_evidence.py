"""Import public GitHub connector results as quarantined issue-tracker evidence."""
import json,sqlite3
from pathlib import Path
from datetime import datetime,timezone
from social_benchmark.pipeline.normalization import normalize_text,hash_identifier

def run():
 db=sqlite3.connect('data/live/evidence.sqlite',timeout=30);count=0;communities=set()
 for path in sorted(Path('data/live').glob('github-*.json')):
  if not (path.name.startswith('github-issues-') or path.name.startswith('github-comments-')):continue
  comments=path.name.startswith('github-comments-')
  for item in json.loads(path.read_text(encoding='utf-8')):
   title=item.get('title','');body=item.get('body') or ''
   if not body and not title:continue
   repo=item['repository'];communities.add(repo)
   record={'platform':'github','source_id':('comment:' if comments else 'issue:')+str(item['id']),'title':title,'body':normalize_text(body),'url':item['url'],'community_id':repo,'thread_id':item['thread_id'] if comments else str(item['id']),'parent_id':None,'author_id_hash':hash_identifier(item['user']),'published_at':item['created_at'],'engagement':{},'metadata':{'type':'comment' if comments else 'issue','transport':'github_connector','story_title':item.get('thread_title',title),'discovery_target':item['discovery_target'],'cohort':'issue_tracker','exclude_from_public_scoring':True,'review_reason':'Issue tracker selection bias; model versus application attribution and generated text require review'}}
   db.execute('INSERT OR REPLACE INTO raw VALUES (?,?)',('github:'+record['source_id'],json.dumps(record)));count+=1
  db.commit()
 db.close();report={'attempted_at':datetime.now(timezone.utc).isoformat(),'sources':[{'source':'GitHub public issues and comments','status':'partial','items':count,'transport':'github_connector','cohort':'issue_tracker','error':'Bounded search and first 100 comments of selected threads; quarantined pending relevance review','communities':len(communities)}]};Path('data/live/github-health.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':run()
