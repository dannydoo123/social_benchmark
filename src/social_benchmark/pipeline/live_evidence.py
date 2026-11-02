"""Rolling evidence store. Training data never enters this store automatically."""
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import math
import re

CATEGORY_TERMS = {
    'cost_efficiency': r'\b(cost|price|pricing|cheap|expensive|value|quota|tokens? per|subscription)\b',
    'vibe_coding': r'\b(vibe.?cod\w*|prototype|no.code|build an app|built an app)\b',
    'design': r'\b(design|ui|ux|layout|typography|figma|frontend|front.end|visual)\b',
    'coding': r'\b(code|coding|debug|programming|refactor|repository)\b',
    'research': r'\b(research|citation|sources|paper|fact.check)\b',
    'writing': r'\b(writing|write|essay|copywriting|prose)\b',
    'agents': r'\b(agent|tool use|autonomous|workflow)\b',
}

def categories(text):
    return [k for k, pattern in CATEGORY_TERMS.items() if re.search(pattern, text, re.I)] or ['general']

def timestamp(value):
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return dt if dt.tzinfo else None
    except (ValueError, TypeError):
        return None

def summarize(rows, now=None, days=30):
    now = now or datetime.now(timezone.utc)
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        date = timestamp(row.get('published_at'))
        if not date or not now - timedelta(days=days) <= date <= now:
            continue
        key = (row['source_platform'], row['source_item_id'], row['model_id'], row['aspect_category'])
        fingerprint = (row['model_id'], row['aspect_category'], hashlib.sha256(re.sub(r'\s+', ' ', row['evidence_text'].lower()).encode()).hexdigest())
        if key in seen or fingerprint in seen:
            continue
        seen.update((key, fingerprint))
        for category in ['all', *row['categories']]:
            groups[(row['model_id'], category)].append(row)
    result = []
    for (model, category), group in sorted(groups.items()):
        accepted = [r for r in group if r.get('review_status') == 'machine_agreement' and r.get('firsthand_flag') and r.get('author_id_hash') and r.get('thread_id') and r.get('community_id')]
        # One contribution per author per thread, then equal thread weight within each community.
        authors = defaultdict(list)
        for row in accepted:
            authors[(row['source_platform'], row['community_id'], row['thread_id'], row['author_id_hash'])].append((float(row['polarity_score']) + 2) * 25)
        threads = defaultdict(list)
        for key, values in authors.items():
            threads[key[:3]].append(sum(values) / len(values))
        communities = defaultdict(list)
        for key, values in threads.items():
            communities[key[:2]].append(sum(values) / len(values))
        community_scores = [{'community': ':'.join(k), 'threads': len(v), 'score': round(sum(v)/len(v), 1)} for k,v in communities.items()]
        n_authors = len({(k[0],k[3]) for k in authors})
        reasons = []
        if len(threads) < 30: reasons.append('fewer than 30 independent threads')
        if n_authors < 30: reasons.append('fewer than 30 independent authors')
        if len(communities) < 3: reasons.append('fewer than 3 communities')
        if len({k[0] for k in communities}) < 2: reasons.append('fewer than 2 platforms')
        if communities and min(map(len, communities.values())) < 5: reasons.append('fewer than 5 threads in a community')
        # Agreement is a screening mechanism, not evidence of calibrated accuracy on a new domain.
        reasons.append('fresh-domain classifier validation pending')
        result.append({'model': model, 'category': category, 'observations': len(group), 'accepted': len(accepted), 'threads': len(threads), 'authors': n_authors, 'communities': community_scores, 'score': None, 'status': 'insufficient_evidence', 'reasons': reasons, 'latest_evidence_at': max(r['published_at'] for r in group), 'evidence': [{'url': r['url'], 'community': r['community_id'], 'published_at': r['published_at'], 'review_status': r['review_status'], 'aspect': r['aspect_category']} for r in group[:8]]})
    return result
