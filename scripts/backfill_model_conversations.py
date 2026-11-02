"""Broad public-discussion backfill with explicit relevance accounting.
Search matches and thread replies are collected separately; neither implies a scored opinion.
"""
import json,re,sqlite3,hashlib
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
from pathlib import Path
from urllib.parse import urlparse
from social_benchmark.pipeline.clients.base import HttpJsonClient
from social_benchmark.pipeline.normalization import normalize_evidence_text,normalize_text,hash_identifier

QUERIES={'gpt-6-astra':['GPT-6 Astra','GPT 6 Astra'],'gpt-6.1-sol':['GPT-6.1 Sol','GPT 6.1 Sol']}
PATTERNS={'gpt-6-astra':re.compile(r'\bgpt[-\s]*6[-\s]*astra\b',re.I),'gpt-6.1-sol':re.compile(r'\bgpt[-\s]*6\.1[-\s]*sol\b',re.I)}

def flatten_hn(node,root_id,root_title):
    stack=[node]
    while stack:
        item=stack.pop();stack.extend(reversed(item.get('children') or []))
        if not item.get('author') or not (item.get('text') or item.get('title')):continue
        yield {'platform':'hacker_news','source_id':str(item['id']),'title':normalize_text(item.get('title')),'body':normalize_evidence_text(item.get('text')),'url':f'https://news.ycombinator.com/item?id={item["id"]}','community_id':'hacker_news','thread_id':str(root_id),'parent_id':str(item['parent_id']) if item.get('parent_id') else None,'author_id_hash':hash_identifier(item.get('author')),'published_at':item.get('created_at'),'engagement':{'points':item.get('points')},'metadata':{'transport':'algolia_api','type':item.get('type'),'story_title':root_title,'coverage':'indexed_thread_tree'}}

class Backfill:
    def __init__(self):
        self.db=sqlite3.connect('data/live/evidence.sqlite',timeout=30)
        self.health=[]
        self.cutoff=datetime.now(timezone.utc)-timedelta(days=90)
        self.hn=HttpJsonClient('https://hn.algolia.com/api/v1',min_delay_seconds=.4,retries=0)
        self.forum=HttpJsonClient('https://community.openai.com',min_delay_seconds=1,retries=0)
        self.lemmy=HttpJsonClient('https://lemmy.world',min_delay_seconds=1,retries=0)
    def save(self,rows):
        count=0
        for row in rows:
            key=row['platform']+':'+row['source_id']
            previous=self.db.execute('SELECT payload FROM raw WHERE key=?',(key,)).fetchone()
            if previous:
                old=json.loads(previous[0]);row['metadata']={**old.get('metadata',{}),**row.get('metadata',{})}
            self.db.execute('INSERT OR REPLACE INTO raw VALUES (?,?)',(key,json.dumps(row)));count+=1
        self.db.commit();return count
    def record(self,source,count,error='',**extra):
        item={'source':source,'status':'partial' if count and error else 'failed' if error else 'ok','items':count,'error':error,**extra}
        self.health.append(item);print(json.dumps(item),flush=True)
        Path('data/live/broad-collection-health.json').write_text(json.dumps({'attempted_at':datetime.now(timezone.utc).isoformat(),'sources':self.health},indent=2),encoding='utf-8')
    def collect_hn(self):
        threads=set()
        for target,queries in QUERIES.items():
            count=0;error=''
            try:
                for query in queries:
                    for page in range(10):
                        data=self.hn.get_json('/search_by_date',{'query':query,'tags':'comment','hitsPerPage':100,'page':page,'numericFilters':f'created_at_i>{int(self.cutoff.timestamp())}'})
                        rows=[]
                        for hit in data.get('hits',[]):
                            if hit.get('story_id'):threads.add(int(hit['story_id']))
                            rows.append({'platform':'hacker_news','source_id':str(hit['objectID']),'title':'','body':normalize_evidence_text(hit.get('comment_text')),'url':f'https://news.ycombinator.com/item?id={hit["objectID"]}','community_id':'hacker_news','thread_id':str(hit.get('story_id')) if hit.get('story_id') else None,'parent_id':str(hit['parent_id']) if hit.get('parent_id') else None,'author_id_hash':hash_identifier(hit.get('author')),'published_at':hit.get('created_at'),'engagement':{},'metadata':{'transport':'algolia_api','type':'comment','story_title':hit.get('story_title'),'discovery_target':target,'coverage':'search_match'}})
                        count+=self.save(rows)
                        if page+1>=data.get('nbPages',0) or not data.get('hits'):break
            except Exception as exc:error=str(exc)
            self.record('HN 90-day search / '+target,count,error)
        count=0;failures=0
        for index,tid in enumerate(sorted(threads)):
            try:
                data=self.hn.get_json('/items/'+str(tid));count+=self.save(flatten_hn(data,tid,data.get('title','')))
            except Exception:failures+=1
            if (index+1)%25==0:print(json.dumps({'hn_threads':index+1,'total_threads':len(threads),'records_processed':count}),flush=True)
        self.record('HN full indexed conversations',count,f'{failures} failed threads' if failures else '',threads=len(threads))
    def collect_forum(self):
        topics=set()
        for queries in QUERIES.values():
            for page in range(1,11):
                try:
                    data=self.forum.get_json('/search.json',{'q':queries[0],'page':page})
                    topics.update(t['id'] for t in data.get('topics',[]))
                    if not data.get('grouped_search_result',{}).get('more_full_page_results'):break
                except Exception as exc:self.record('Forum search',0,str(exc));break
        count=0;errors=0
        for index,tid in enumerate(sorted(topics)):
            try:
                topic=self.forum.get_json(f'/t/{tid}.json');stream=topic.get('post_stream',{});posts=stream.get('posts',[]);seen={p['id'] for p in posts}
                missing=[pid for pid in stream.get('stream',[]) if pid not in seen]
                for offset in range(0,len(missing),20):
                    more=self.forum.get_json(f'/t/{tid}/posts.json',{'post_ids[]':missing[offset:offset+20]})
                    posts.extend(more.get('post_stream',{}).get('posts',[]))
                ids={p['post_number']:p['id'] for p in posts};rows=[]
                for post in posts:
                    body=normalize_evidence_text(post.get('cooked'))
                    if not body:continue
                    rows.append({'platform':'community_forum','source_id':str(post['id']),'title':topic.get('title','') if post.get('post_number')==1 else '', 'body':body,'url':f'https://community.openai.com/t/{tid}/{post["post_number"]}','community_id':'OpenAI Developer Community','thread_id':str(tid),'parent_id':str(ids[post['reply_to_post_number']]) if post.get('reply_to_post_number') in ids else None,'author_id_hash':hash_identifier(post.get('username')),'published_at':post.get('created_at'),'engagement':{},'metadata':{'transport':'discourse_api','type':'post' if post.get('post_number')==1 else 'comment','story_title':topic.get('title',''),'coverage':'full_returned_stream','cohort':'vendor'}})
                count+=self.save(rows)
            except Exception:errors+=1
            if (index+1)%20==0:print(json.dumps({'forum_topics':index+1,'total_topics':len(topics),'records_processed':count}),flush=True)
        self.record('OpenAI forum full conversations',count,f'{errors} failed topics' if errors else '',threads=len(topics),cohort='vendor')
    def collect_lemmy(self):
        for target,queries in QUERIES.items():
            count=0;error='';seen=set()
            try:
                for query in queries:
                    for page in range(1,21):
                        data=self.lemmy.get_json('/api/v3/search',{'q':query,'type_':'Comments','sort':'New','limit':50,'page':page})
                        comments=data.get('comments',[]);rows=[];new=0
                        for view in comments:
                            item=view['comment'];uri=item.get('ap_id') or f'https://lemmy.world/comment/{item["id"]}'
                            if uri in seen:continue
                            seen.add(uri);new+=1
                            if item.get('deleted') or item.get('removed'):continue
                            community=view.get('community',{});post=view.get('post',{});creator=view.get('creator',{})
                            rows.append({'platform':'lemmy','source_id':hashlib.sha256(uri.encode()).hexdigest()[:24],'title':'','body':normalize_evidence_text(item.get('content')),'url':uri,'community_id':community.get('actor_id') or community.get('name','unknown'),'thread_id':post.get('ap_id') or 'lemmy.world:'+str(post.get('id')),'parent_id':None,'author_id_hash':hash_identifier(creator.get('actor_id')),'published_at':item.get('published'),'engagement':view.get('counts',{}),'metadata':{'transport':'lemmy_api','type':'comment','story_title':post.get('name'),'discovery_target':target,'coverage':'federated_search','canonical_uri':uri,'cohort':'unclassified'}})
                        count+=self.save(rows)
                        if len(comments)<50 or not new:break
            except Exception as exc:error=str(exc)
            self.record('Lemmy federated comments / '+target,count,error)
    def report(self):
        rows=[json.loads(x[0]) for x in self.db.execute('SELECT payload FROM raw')];result={}
        for model,pattern in PATTERNS.items():
            matches=[r for r in rows if pattern.search((r.get('title') or '')+' '+(r.get('body') or ''))]
            match_threads={(r['platform'],r.get('thread_id')) for r in matches if r.get('thread_id')}
            replies=[r for r in rows if (r['platform'],r.get('thread_id')) in match_threads]
            comments=[r for r in matches if r.get('metadata',{}).get('type')=='comment']
            result[model]={'exact_mention_records':len(matches),'exact_mention_comments':len(comments),'exact_mentions_by_platform':dict(Counter(r['platform'] for r in matches)),'threads':len(match_threads),'related_thread_records_not_all_relevant':len(replies),'distinct_exact_texts':len({hashlib.sha256(normalize_text(r.get('body')).lower().encode()).hexdigest() for r in matches})}
        report={'generated_at':datetime.now(timezone.utc).isoformat(),'stored_records':len(rows),'platform_counts':dict(Counter(r['platform'] for r in rows)),'models':result,'note':'Historical backfill. Exact mention is not an opinion label. Related thread records must not be counted as model opinions.'}
        Path('data/live/collection-coverage.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
        self.db.close()
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',choices=['hn','forum','lemmy','report','all'],default='all');args=p.parse_args();job=Backfill()
    if args.source in ('lemmy','all'):job.collect_lemmy()
    if args.source in ('hn','all'):job.collect_hn()
    if args.source in ('forum','all'):job.collect_forum()
    job.report()
