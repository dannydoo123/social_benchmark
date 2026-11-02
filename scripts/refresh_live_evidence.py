"""Run once or continuously every 1/4 hours; atomic public status and durable raw store."""
import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from social_benchmark.pipeline.clients.hackernews import HackerNewsClient
from social_benchmark.pipeline.collection import collect_from_config
from social_benchmark.pipeline.extractors.rules import RuleBasedExtractor
from social_benchmark.pipeline.models import to_jsonable
from social_benchmark.pipeline.live_evidence import categories, summarize
from social_benchmark.pipeline.auto_review import _review_row
from social_benchmark.pipeline.local_classifier import load_classifier


def run(config_path='config/live_sources.json', cached=False):
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    now = datetime.now(timezone.utc)
    Path('data/live').mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect('data/live/evidence.sqlite', timeout=5)
    db.execute('CREATE TABLE IF NOT EXISTS raw (key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS observations (key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    # Serializes runs, including collection, to avoid overlapping publishers.
    db.execute('BEGIN IMMEDIATE')
    health, items = [], []
    client = HackerNewsClient()
    for query in ([] if cached else config['hn_queries']):
        try:
            batch = []
            for sid in client.search_story_ids(query, max_stories=config['stories_per_query'], sort_by_date=True):
                batch.extend(client.fetch_story_with_comments(sid, max_comments=config['comments_per_story']))
            items.extend(batch)
            health.append({'source': 'HN / '+query, 'status': 'ok', 'items': len(batch)})
        except Exception as exc:
            health.append({'source': 'HN / '+query, 'status': 'failed', 'error': type(exc).__name__})
    for community in ([] if cached or not os.environ.get('REDDIT_CLIENT_ID') else config['reddit_communities']):
        try:
            batch = collect_from_config({'reddit_subreddits':[{'name':community,'sort':'new','limit':10,'comments':30}]})
            items.extend(batch)
            health.append({'source':'Reddit / '+community,'status':'ok','items':len(batch)})
        except Exception as exc:
            health.append({'source':'Reddit / '+community,'status':'failed','error':type(exc).__name__})
    if cached:
        from social_benchmark.pipeline.cli import _raw_item_from_record
        items = [_raw_item_from_record(json.loads(r[0])) for r in db.execute('SELECT payload FROM raw')]
        previous = Path('web/public/live-perspective.json')
        health = json.loads(previous.read_text(encoding='utf-8'))['sources'] if previous.exists() else []
    for item in items:
        db.execute('INSERT OR REPLACE INTO raw VALUES (?,?)',(item.platform.value+':'+item.source_id,json.dumps(to_jsonable(item))))
    extractor = RuleBasedExtractor.default()
    model_error = None
    try:
        classifier = load_classifier(config['classifier'])
    except Exception as exc:
        classifier = None
        model_error = type(exc).__name__
    for item in items:
        if item.metadata.get('exclude_from_public_scoring'): continue
        parent = db.execute('SELECT payload FROM raw WHERE key=?',(item.platform.value+':'+str(item.parent_id),)).fetchone()
        context = json.loads(parent[0]) if parent else {}
        # Refreshing an item replaces its observations; deleted/edited content cannot linger.
        prefix = item.platform.value+':'+item.source_id+':'
        db.execute('DELETE FROM observations WHERE substr(key,1,?)=?',(len(prefix),prefix))
        for obs in extractor.extract_observations(item):
            row = to_jsonable(obs)
            if item.platform.value == 'hacker_news':
                row['url'] = 'https://news.ycombinator.com/item?id=' + item.source_id
            row.update(parent_id=item.parent_id, context_text=context.get('body',''), engagement=item.engagement, source_metadata=item.metadata, collected_at=now.isoformat(), categories=categories(obs.evidence_text), human_labeled_flag=False)
            row['review_status'] = 'pending'
            if classifier:
                try:
                    verdict = _review_row(row,classifier,{'context_text':row['context_text']},source_platform=item.platform.value,min_extractor_confidence=.72,min_model_confidence=.8,accept_neutral=True)
                    row['review_status'] = 'machine_agreement' if verdict.get('reviewed_flag') == 'true' else 'rejected'
                    row['review_reason'] = verdict.get('human_exclusion_reason','')
                except Exception as exc:
                    row['review_reason'] = type(exc).__name__
            if not any(c.isdigit() for c in row['model_id']):
                row['review_status'] = 'ambiguous_model'
            db.execute('INSERT OR REPLACE INTO observations VALUES (?,?)',(prefix+row['model_id']+':'+row['aspect_category'],json.dumps(row)))
    rows = [json.loads(x[0]) for x in db.execute('SELECT payload FROM observations')]
    output = {'generated_at':now.isoformat(), 'reprocessed_only':cached,'refresh_hours':config['refresh_hours'],'next_refresh_at':(now+timedelta(hours=config['refresh_hours'])).isoformat(),'window_days':config['window_days'],'sources':health,'classifier_error':model_error,'raw_items':db.execute('SELECT count(*) FROM raw').fetchone()[0], 'models':summarize(rows,now,config['window_days']), 'methodology':'Training corpus excluded. Equal community and thread weighting; author/thread and exact-text deduplication. Keyword category tags are exploratory. Scores withheld pending independent fresh-domain validation.'}
    if cached and previous.exists():
        prior = json.loads(previous.read_text(encoding='utf-8'))
        output['generated_at'] = prior['generated_at']
        output['next_refresh_at'] = prior['next_refresh_at']
    db.commit()
    db.close()
    destination = Path('web/public/live-perspective.json')
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(output,indent=2),encoding='utf-8')
    os.replace(temporary,destination)
    print(json.dumps({'raw_items':output['raw_items'],'model_categories':len(output['models']),'sources':health,'classifier_error':model_error}),flush=True)
    return config['refresh_hours']

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--loop',action='store_true')
    parser.add_argument('--cached',action='store_true',help='Re-evaluate stored raw records without collection')
    args=parser.parse_args()
    while True:
        hours=run(cached=args.cached)
        if not args.loop: break
        time.sleep(hours*3600)
