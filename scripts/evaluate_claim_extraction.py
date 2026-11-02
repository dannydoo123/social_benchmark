"""Frozen, assistant-labeled diagnostic. Not independent validation of public scores."""
import argparse
import json
import re
from pathlib import Path
from social_benchmark.pipeline.claim_extraction import split_claims


def evaluate(split):
    data=json.loads(Path('datasets/evaluation/claim_extraction_real.json').read_text(encoding='utf-8'))
    tp=fp=fn=exact=total=skipped=0
    failures=[]
    for case in data['examples']:
        if case['split']!=split: continue
        if case['review_note']:
            skipped+=1;continue
        total+=1
        expected={tuple(x) for x in case['expected']}
        actual={(r['model_id'],r['category'],r['polarity_score']) for r in split_claims(case['text']) if re.search(r'\d',r['model_id'])}
        tp+=len(expected&actual);fp+=len(actual-expected);fn+=len(expected-actual)
        exact+=expected==actual
        if expected!=actual: failures.append({'key':case['key'],'missing':sorted(expected-actual),'extra':sorted(actual-expected)})
    return {'split':split,'documents':total,'ambiguous_excluded':skipped,'true_positive':tp,'false_positive':fp,'false_negative':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'exact_document_match':exact/total if total else None,'failures':failures,'validation':'assistant-labeled diagnostic only; not independently validated'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['development','holdout'],default='development');parser.add_argument('--output')
    args=parser.parse_args();result=evaluate(args.split)
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
