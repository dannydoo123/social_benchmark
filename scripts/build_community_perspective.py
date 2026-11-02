"""Build category-specific evidence candidates from stored posts, never from training rows."""
import json,sqlite3,hashlib
from collections import Counter, defaultdict
from social_benchmark.pipeline.community_balance import balanced_candidate_score
from datetime import datetime,timezone,timedelta
from pathlib import Path
from social_benchmark.pipeline.claim_extraction import split_claims
from social_benchmark.pipeline.catalog import ModelCatalog
from social_benchmark.pipeline.live_evidence import timestamp

def build():
    config=json.loads(Path('config/live_sources.json').read_text(encoding='utf-8-sig'))
    db=sqlite3.connect('data/live/evidence.sqlite',timeout=30)
    now=datetime.now(timezone.utc);cutoff=now-timedelta(days=config['window_days'])
    claims=[];seen=set();raw_count=0
    stages=Counter()
    # Release SQLite's read lock before CPU-heavy extraction so collectors can commit.
    raw_rows=db.execute('SELECT payload FROM raw').fetchall()
    records=[json.loads(payload) for payload, in raw_rows]
    del raw_rows
    index={(row['platform'],row['source_id']):row for row in records}
    catalog=ModelCatalog()
    for row in records:
        date=timestamp(row.get('published_at'))
        if not date or not cutoff<=date<=now:continue
        raw_count+=1
        if row.get('metadata',{}).get('exclude_from_public_scoring'):
            stages['quarantined_records']+=1
            continue
        text='\n'.join(filter(None,[row.get('title'),row.get('body')]))
        parent=index.get((row['platform'],row.get('parent_id')))
        if parent and (parent.get('thread_id')!=row.get('thread_id') or parent.get('metadata',{}).get('exclude_from_public_scoring')):
            parent=None
        context='\n'.join(filter(None,[parent.get('title'),parent.get('body')])) if parent else None
        extracted=split_claims(text,catalog,context_text=context)
        stages['records_with_candidate' if extracted else 'records_without_candidate']+=1
        for claim in extracted:
            fingerprint=(claim['model_id'],claim['category'],hashlib.sha256(claim['evidence_text'].lower().encode()).hexdigest())
            if fingerprint in seen:
                stages['duplicate_claims_removed']+=1
                continue
            seen.add(fingerprint)
            community=row.get('community_id') or 'unknown'
            cohort=row.get('metadata',{}).get('cohort') or config['community_groups'].get(community,'general' if row['platform']=='hacker_news' else 'unclassified')
            url='https://news.ycombinator.com/item?id='+row['source_id'] if row['platform']=='hacker_news' else row['url']
            claims.append({**claim,'id':row['platform']+':'+row['source_id']+':'+claim['claim_id'],'platform':row['platform'],'community':community,'cohort':cohort,'source_item_id':row['source_id'],'thread_id':row.get('thread_id'),'author_id_hash':row.get('author_id_hash'),'published_at':row['published_at'],'url':url,'transport':row.get('metadata',{}).get('transport','api'),'evidence_text':claim['evidence_text'][:320]})
            if claim['attribution']=='direct_parent_pronoun' and parent:
                claims[-1]['context_source_id']=parent['source_id']
                claims[-1]['context_url']=parent['url']
                claims[-1]['context_evidence']=context[:500]
    claims.sort(key=lambda x:x['published_at'],reverse=True)
    counts=Counter(c['model_id'] for c in claims)
    sources=[]
    for file in ['web/public/live-perspective.json','data/live/reddit-feed-health.json','data/live/reddit-archive-health.json','data/live/forum-health.json','data/live/hn-search-health.json','data/live/broad-collection-health.json','data/live/github-health.json']:
        p=Path(file)
        if p.exists():
            data=json.loads(p.read_text(encoding='utf-8'));sources.extend([{**s,'attempted_at':data.get('attempted_at',data.get('generated_at'))} for s in data.get('sources',[]) if file!='web/public/live-perspective.json' or not s['source'].startswith('Reddit')])
    output={'generated_at':now.isoformat(),'window_days':config['window_days'],'refresh_hours':config['refresh_hours'],'raw_items':raw_count,'claims':claims,'sources':sources,'targets':[{'model':m,'target':config['target_chunks_per_model'],'candidate_chunks':counts[m],'validated_chunks':0} for m in config['target_models']], 'methodology_version':'claims-v1','validation_status':'pending','scope_note':'Exploratory clause labels, not validated scores or a representative survey. Feed coverage is partial. Chunks from one conversation are not independent votes.'}
    output['methodology_version']='aspect-rules-v2'
    output['extraction_summary']={'records_in_window':raw_count,**dict(stages),'candidate_claims':len(claims),'attribution_counts':dict(Counter(c['attribution'] for c in claims)),'note':'Records without candidates include neutral, unrelated, ambiguous, quoted, and unsupported wording. This is not an accuracy estimate.'}
    grouped=defaultdict(list)
    for claim in claims: grouped[(claim['model_id'],claim['category'])].append(claim)
    diagnostics=[{'model':key[0],'category':key[1],**balanced_candidate_score(rows)} for key,rows in grouped.items()]
    Path('data/live/balance-diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    coverage=Path('data/live/collection-coverage.json')
    if coverage.exists(): output['collection_coverage']=json.loads(coverage.read_text(encoding='utf-8'))
    destination=Path('web/public/community-perspective.json');tmp=destination.with_suffix('.tmp');tmp.write_text(json.dumps(output,indent=2),encoding='utf-8');tmp.replace(destination)
    db.close();print(json.dumps({'raw_in_window':raw_count,'claims':len(claims),'models':len(counts),'targets':output['targets']}))
if __name__=='__main__':build()
