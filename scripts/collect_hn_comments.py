"""Search recent HN comments directly, retaining conversation IDs and exact timestamps."""
import json,sqlite3
from datetime import datetime,timezone,timedelta
from pathlib import Path
from social_benchmark.pipeline.clients.base import HttpJsonClient
from social_benchmark.pipeline.normalization import normalize_evidence_text,normalize_text,hash_identifier

def collect():
    config=json.loads(Path('config/live_sources.json').read_text(encoding='utf-8-sig'))
    db=sqlite3.connect('data/live/evidence.sqlite',timeout=30)
    http=HttpJsonClient('https://hn.algolia.com/api/v1',min_delay_seconds=.5,retries=1)
    cutoff=int((datetime.now(timezone.utc)-timedelta(days=config['window_days'])).timestamp())
    health=[]
    for query in ['GPT-6 Astra','GPT-6.1 Sol','GPT 6 Astra','GPT 6.1 Sol','Opus','Sonnet','Gemini']:
        count=0;error=''
        try:
            for page in range(10):
                data=http.get_json('/search_by_date',{'query':query,'tags':'comment','hitsPerPage':100,'page':page,'numericFilters':f'created_at_i>{cutoff}'})
                for hit in data.get('hits',[]):
                    raw={'platform':'hacker_news','source_id':str(hit['objectID']),'title':'','body':normalize_evidence_text(hit.get('comment_text','')),'url':'https://news.ycombinator.com/item?id='+str(hit['objectID']),'community_id':'hacker_news','thread_id':str(hit['story_id']) if hit.get('story_id') else None,'parent_id':str(hit['parent_id']) if hit.get('parent_id') else None,'author_id_hash':hash_identifier(hit.get('author')),'published_at':hit.get('created_at'),'engagement':{},'metadata':{'transport':'algolia_api','type':'comment','story_title':hit.get('story_title'),'search_query':query,'coverage':'search_sample'}}
                    db.execute('INSERT OR REPLACE INTO raw VALUES (?,?)',('hacker_news:'+raw['source_id'],json.dumps(raw)));count+=1
                db.commit()
                if page+1>=data.get('nbPages',0) or not data.get('hits'):break
        except Exception as exc:error=type(exc).__name__
        status={'source':'HN comments / '+query,'status':'partial' if count and error else 'failed' if error else 'ok','items':count,'transport':'algolia_api','error':error};health.append(status);print(json.dumps(status),flush=True)
    Path('data/live/hn-search-health.json').write_text(json.dumps({'attempted_at':datetime.now(timezone.utc).isoformat(),'sources':health},indent=2),encoding='utf-8');db.close()
if __name__=='__main__':collect()
