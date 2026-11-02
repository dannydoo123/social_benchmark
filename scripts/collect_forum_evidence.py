"""Bounded backfill from a public Discourse community; explicitly vendor-focused."""
import json,sqlite3,time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlencode
from social_benchmark.pipeline.clients.base import HttpJsonClient
from social_benchmark.pipeline.normalization import normalize_evidence_text,normalize_text,hash_identifier


def collect():
    db=sqlite3.connect('data/live/evidence.sqlite',timeout=30)
    client=HttpJsonClient('https://community.openai.com',min_delay_seconds=1.5,retries=1)
    visited=set();total=0;errors=[]
    for query in ['GPT-6 Astra','GPT-6.1 Sol']:
        for page in range(1,5):
            try:
                search=client.get_json('/search.json',{'q':query,'page':page})
                topics=search.get('topics',[])
                if not topics:break
                for topic in topics[:15]:
                    tid=topic['id']
                    if tid in visited:continue
                    visited.add(tid)
                    try:
                        data=client.get_json(f'/t/{tid}.json')
                        posts=data.get('post_stream',{}).get('posts',[])
                        ids={p['post_number']:p['id'] for p in posts}
                        for post in posts:
                            body=normalize_evidence_text(post.get('cooked',''))
                            if not body:continue
                            raw={'platform':'community_forum','source_id':str(post['id']),'title':data.get('title','') if post.get('post_number')==1 else '', 'body':body,'url':f'https://community.openai.com/t/{tid}/{post["post_number"]}','community_id':'OpenAI Developer Community','thread_id':str(tid),'parent_id':str(ids[post['reply_to_post_number']]) if post.get('reply_to_post_number') in ids else None,'author_id_hash':hash_identifier(post.get('username')),'published_at':post.get('created_at'),'engagement':{'likes':sum(a.get('count',0) for a in post.get('actions_summary',[]) if a.get('id')==2)},'metadata':{'transport':'discourse_api','type':'post' if post.get('post_number')==1 else 'comment','coverage':'partial_topic','cohort':'vendor'}}
                            db.execute('INSERT OR REPLACE INTO raw VALUES (?,?)',('community_forum:'+raw['source_id'],json.dumps(raw)));total+=1
                        db.commit()
                    except Exception as exc: errors.append(type(exc).__name__)
                print(json.dumps({'query':query,'page':page,'topics':len(visited),'records':total}),flush=True)
                if not search.get('grouped_search_result',{}).get('more_full_page_results'):break
            except Exception as exc: errors.append(type(exc).__name__);break
    db.close()
    status={'attempted_at':datetime.now(timezone.utc).isoformat(),'sources':[{'source':'OpenAI Developer Community','status':'partial' if errors else 'ok','items':total,'transport':'discourse_api','cohort':'vendor','error':', '.join(errors)}]}
    Path('data/live/forum-health.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
if __name__=='__main__':collect()
