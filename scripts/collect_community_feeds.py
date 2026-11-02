"""Backfill public feeds with bounded, rate-limited requests and per-source checkpoints."""
import json
import sqlite3
from datetime import datetime,timezone
from pathlib import Path
from social_benchmark.pipeline.clients.reddit_feed import RedditFeedClient
from social_benchmark.pipeline.catalog import ModelCatalog
from social_benchmark.pipeline.models import to_jsonable


def collect():
    config=json.loads(Path('config/live_sources.json').read_text(encoding='utf-8-sig'))
    db=sqlite3.connect('data/live/evidence.sqlite',timeout=30)
    client=RedditFeedClient(delay=6)
    catalog=ModelCatalog()
    results=[]
    cursor_path=Path('data/live/reddit-cursor.json')
    cursor=json.loads(cursor_path.read_text()).get('next',0) if cursor_path.exists() else 0
    communities=config['reddit_communities']
    ordered=communities[cursor:]+communities[:cursor]
    for community in ordered:
        count=0
        errors=[]
        for sort in ['new','top']:
            try:
                items=client.listing(community,sort)
                for item in items:
                    db.execute('INSERT OR REPLACE INTO raw VALUES (?,?)',('reddit:'+item.source_id,json.dumps(to_jsonable(item))))
                db.commit();count+=len(items)
                # Follow relevant discussion threads rather than collecting all unrelated replies.
                for post in [x for x in items if catalog.find_mentions(x.text)][:6]:
                    try:
                        comments=client.thread(post.url)
                        for item in comments:
                            db.execute('INSERT OR REPLACE INTO raw VALUES (?,?)',('reddit:'+item.source_id,json.dumps(to_jsonable(item))))
                        db.commit();count+=len(comments)
                    except Exception as exc:
                        errors.append(str(exc));break
            except Exception as exc:
                errors.append(str(exc));break
            if any('429' in error for error in errors): break
        status={'source':'Reddit / '+community,'status':'partial' if count and errors else 'ok' if count else 'failed','items':count,'transport':'rss','cohort':config['community_groups'].get(community,'general'),'error':'; '.join(errors)}
        results.append(status);print(json.dumps(status),flush=True)
        Path('data/live/reddit-feed-health.json').write_text(json.dumps({'attempted_at':datetime.now(timezone.utc).isoformat(),'sources':results},indent=2),encoding='utf-8')
        cursor_path.write_text(json.dumps({'next':(communities.index(community)+1)%len(communities)}))
        if any('429' in error for error in errors):
            print('Rate limit reached; remaining communities deferred until next scheduled run.',flush=True)
            break
    db.close()

if __name__=='__main__': collect()
