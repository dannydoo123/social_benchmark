"""Sampling diagnostics only. No candidate-label aggregate is a published benchmark."""
from collections import defaultdict

def balanced_candidate_score(claims):
    communities=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for row in claims:
        if not row.get('author_id_hash') or not row.get('thread_id'):continue
        communities[(row['cohort'],row['platform'],row['community'])][row['author_id_hash']][row['thread_id']].append(row['polarity_score'])
    cohorts=defaultdict(list)
    for key,authors in communities.items():
        # Duplicate chunks never give a person extra influence within a thread.
        author_means=[]
        for threads in authors.values():
            thread_means=[sum(values)/len(values) for values in threads.values()]
            author_means.append(sum(thread_means)/len(thread_means))
        cohorts[key[0]].append(sum(author_means)/len(author_means))
    ordinary=[sum(values)/len(values) for cohort,values in cohorts.items() if cohort in ('general','workflow')]
    if not ordinary:return {'score':None,'vendor_weight':0,'reason':'No non-vendor cohort'}
    base=sum(ordinary)/len(ordinary)
    vendor=cohorts.get('vendor')
    vendor_weight=.2 if vendor else 0
    mean=base*(1-vendor_weight)+(sum(vendor)/len(vendor)*vendor_weight if vendor else 0)
    return {'score':round((mean+1)*50,2),'vendor_weight':vendor_weight,'reason':'Unvalidated candidate-label diagnostic; equal cohort/community/author weighting'}
