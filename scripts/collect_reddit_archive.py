"""Bounded, free Arctic Shift collection; archived timestamps never imply freshness."""
import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from social_benchmark.pipeline.normalization import hash_identifier, normalize_evidence_text, normalize_text

BASE = 'https://arctic-shift.photon-reddit.com'
STATE = Path('data/live/reddit-archive-state.json')
HEALTH = Path('data/live/reddit-archive-health.json')


def fullname(value, prefix):
    return value if value.startswith(('t1_', 't3_')) else prefix + value


def normalize(item, kind, groups):
    body = item.get('body' if kind == 'comment' else 'selftext') or ''
    if body.strip() in ('[deleted]', '[removed]'):
        return None
    title = item.get('title', '') if kind == 'post' else ''
    if not body.strip() and not title.strip():
        return None
    source_id = fullname(str(item['id']), 't1_' if kind == 'comment' else 't3_')
    thread = fullname(str(item['link_id']), 't3_') if kind == 'comment' else source_id
    community = item['subreddit']
    parent = item.get('parent_id')
    # Unprefixed archive parent IDs denote comments; an absent parent is top-level.
    parent = fullname(str(parent), 't1_') if parent else (thread if kind == 'comment' else None)
    author = item.get('author')
    return {
        'platform': 'reddit', 'source_id': source_id, 'title': normalize_text(title),
        'body': normalize_evidence_text(body),
        'url': f'https://www.reddit.com/r/{community}/comments/{thread[3:]}/' + (f'_/{source_id[3:]}/' if kind == 'comment' else ''),
        'community_id': community, 'thread_id': thread, 'parent_id': parent,
        'author_id_hash': hash_identifier(author) if author and author != '[deleted]' else None,
        'published_at': datetime.fromtimestamp(item['created_utc'], timezone.utc).isoformat(),
        'engagement': {'score': item.get('score')},
        'metadata': {'transport': 'arctic_shift', 'type': kind, 'coverage': 'partial_archive',
                     'cohort': groups.get(community, 'unclassified'),
                     'archive_retrieved_on': item.get('retrieved_on'),
                     'collected_at': datetime.now(timezone.utc).isoformat()},
    }


def collect(max_requests=30):
    if not 1 <= max_requests <= 100:
        raise ValueError('Request budget must be between 1 and 100')
    STATE.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if time.time() < state.get('retry_after', 0):
        print('Archive collector cooling down; previous failure remains visible.')
        return
    config = json.loads(Path('config/live_sources.json').read_text(encoding='utf-8-sig'))
    jobs = [(sub, query) for sub in config['reddit_communities'] for query in ('Astra', 'Sol')]
    cursor = state.get('cursor', 0) % len(jobs)
    db = sqlite3.connect('data/live/evidence.sqlite', timeout=30)
    db.execute('CREATE TABLE IF NOT EXISTS raw (key TEXT PRIMARY KEY, payload TEXT)')
    requests = inserted = fetched = 0
    latest = None
    error = ''
    status = 'ok'
    seen_threads = set()

    def get(path, params):
        nonlocal requests
        requests += 1
        time.sleep(1)
        req = Request(BASE + path + '?' + urlencode(params), headers={'User-Agent': 'social-benchmark/0.3 public model discussion analysis'})
        with urlopen(req, timeout=20) as response:
            return json.load(response)['data']

    def save(items, kind):
        nonlocal inserted, fetched, latest
        for item in items:
            row = normalize(item, kind, config['community_groups'])
            if row is None:
                continue
            fetched += 1
            latest = max(latest or row['published_at'], row['published_at'])
            # Never overwrite a current API/feed record with potentially stale archive text.
            inserted += db.execute('INSERT OR IGNORE INTO raw VALUES (?, ?)',
                                   ('reddit:' + row['source_id'], json.dumps(row))).rowcount
        db.commit()

    try:
        for offset in range(len(jobs)):
            if requests >= max_requests:
                break
            index = (cursor + offset) % len(jobs)
            sub, query = jobs[index]
            posts = get('/api/posts/search', {'subreddit': sub, 'query': query, 'after': '90d', 'sort': 'desc', 'limit': 25})
            save(posts, 'post')
            state['cursor'] = (index + 1) % len(jobs)
            # Reserve requests for other communities instead of spending the run on one.
            for post in posts[:5]:
                if requests >= max_requests:
                    break
                tid = fullname(str(post['id']), 't3_')
                if tid in seen_threads:
                    continue
                seen_threads.add(tid)
                comments = get('/api/comments/search', {'link_id': tid, 'limit': 100, 'sort': 'desc'})
                save(comments, 'comment')
    except HTTPError as exc:
        error = f'Archive HTTP {exc.code}: ' + exc.read(500).decode('utf-8', errors='replace')
        status = 'partial' if fetched else 'failed'
        # Stop on access denial or rate limiting, with no endpoint/proxy rotation.
        state['retry_after'] = time.time() + (86400 if exc.code in (401, 403) else 14400)
    except Exception as exc:
        error = str(exc)
        status = 'partial' if fetched else 'failed'
        state['retry_after'] = time.time() + 14400
    finally:
        db.close()
        STATE.write_text(json.dumps(state), encoding='utf-8')
        health = {'attempted_at': datetime.now(timezone.utc).isoformat(), 'sources': [{
            'source': 'Reddit / Arctic Shift archive', 'status': status,
            'items': fetched, 'new_unique_items': inserted, 'requests': requests,
            'transport': 'arctic_shift', 'error': error, 'latest_published_at': latest,
            'coverage': 'bounded archive sample; not complete or guaranteed current',
            'provider_spend_usd': 0,
        }]}
        HEALTH.write_text(json.dumps(health, indent=2), encoding='utf-8')
        print(json.dumps(health), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-requests', type=int, default=30)
    collect(parser.parse_args().max_requests)
