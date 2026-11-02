"""Create a reproducible, unlabelled review queue; never auto-approve candidates."""
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path


def build():
    feed=json.loads(Path('web/public/community-perspective.json').read_text(encoding='utf-8'))
    groups=defaultdict(list)
    for c in feed['claims']:
        groups[(c['platform'],c['attribution'],c['category'],c['polarity_score'])].append(c)
    rng=random.Random(615)
    for group in groups.values():rng.shuffle(group)
    selected={}
    # Round-robin strata prevent the largest platform/category filling the queue.
    while groups and len(selected)<150:
        for key in list(groups):
            c=groups[key].pop()
            selected[(c['platform'],c['source_item_id'])]=True
            if not groups[key]:del groups[key]
            if len(selected)>=150:break
    records={}
    with sqlite3.connect('data/live/evidence.sqlite') as db:
        for payload, in db.execute('SELECT payload FROM raw'):
            row=json.loads(payload)
            records[(row['platform'],row['source_id'])]=row
    claimed={(c['platform'],c['source_item_id']) for c in feed['claims']}
    # Negative sampling is essential for detecting missed opinions (recall).
    others=[key for key,r in records.items() if key not in claimed and not r.get('metadata',{}).get('exclude_from_public_scoring') and r.get('published_at') and len(r.get('body',''))>30]
    rng.shuffle(others)
    for key in others[:50]:selected[key]=False
    output=[]
    for key,has_candidate in selected.items():
        row=records[key]
        proposed=[c for c in feed['claims'] if (c['platform'],c['source_item_id'])==key]
        parent=records.get((row['platform'],row.get('parent_id')))
        output.append({'id':':'.join(key),'url':row['url'],'thread_id':row.get('thread_id'),
            'text':'\n'.join(filter(None,[row.get('title'),row.get('body')])),
            'parent_text':'\n'.join(filter(None,[parent.get('title'),parent.get('body')])) if parent else None,
            'sampling_stratum':'candidate' if has_candidate else 'no_candidate',
            'proposed_claims':proposed,'review_status':'pending','reviewer':None,'expected_claims':None})
    path=Path('data/live/extraction-review-queue.json')
    path.write_text(json.dumps({'instructions':'Label model/version, category, polarity, exact evidence span, and attribution for every opinion, including missed claims. Use uncertain for ambiguous cases. Proposed labels are not ground truth. Split evaluation by thread; do not tune on the held-out portion.','items':output},indent=2),encoding='utf-8')
    print(json.dumps({'review_documents':len(output),'candidate_documents':sum(x['sampling_stratum']=='candidate' for x in output),'unlabelled_negative_sample':sum(x['sampling_stratum']=='no_candidate' for x in output)}))

if __name__=='__main__':build()
