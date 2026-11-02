"""Evidence-preserving aspect extraction. Candidates require independent review."""
import hashlib
import re
from social_benchmark.pipeline.catalog import ModelCatalog

CATEGORY_PATTERNS = {
 'cost_efficiency': r'\b(costs?|prices?|pricing|cheap|expensive|overpriced|affordable|inexpensive|costly|value|quota|allowance|credits?|subscription|usage limits?|rate limits?|tokens? per|savings|budget)\b',
 'vibe_coding': r'\b(vibe[ -]?cod\w*|prototyp\w*|no[ -]code|build an app|built an app)\b',
 'design': r'\b(design\w*|ui|ux|layouts?|typography|figma|frontend|front.end|visual\w*|aesthetics?)\b',
 'coding': r'\b(cod(?:e|es|ing|ebase)|debug\w*|programming|refactor\w*|repositor\w*|bugs?|unit tests?|integration tests?|terraform|typescript|python|rust|javascript)\b',
 'research': r'\b(research\w*|citations?|sources|papers?|fact.check\w*)\b',
 'writing': r'\b(writ(?:ing|e|es)|essays?|copywriting|prose|stories|storytelling)\b',
 'agents': r'\b(agents?|agentic|tool (?:use|calling|calls)|autonomous|workflows?)\b',
}
POSITIVE = r'\b(best|great|excellent|good|love|amazing|useful|cheap|affordable|inexpensive|efficient|effective|reliable|impressive|beautiful|polished|accurate|helpful|fast|capable|solid|savings)\b|\bworks? (?:well|flawlessly)\b|\bsav(?:es?|ed) (?:me )?(?:time|money)\b'
NEGATIVE = r'\b(expensive|overpriced|bad|poor|worst|hate|unreliable|slow|broken|terrible|disappointing|costly|unhappy|unusable|ugly|inaccurate|fails?|failed|hallucinat\w*|mistakes?|outages?|lies?|lying|forgets?|forgetting)\b|\blow efficiency\b|\b(?:produces?|introduces?|creates?) bugs\b'
BOUNDARY = re.compile(r'\n+|[;!?]|(?<!\d)\.|\.(?!\d)|\b(?:but|however|yet|although|whereas)\b|,?\s+and\s+(?=(?:it|its|this model|the model|is|was|too|very|really|great|good|bad|terrible|cheap|expensive|excellent|poor)\b)', re.I)
PRONOUN = re.compile(r'^(?:it|its|this model|the model)\b', re.I)
ELLIPSIS = re.compile(r'^(?:is|was|too|very|really|great|good|bad|terrible|cheap|expensive|excellent|poor|love|hate)\b', re.I)


def categories(text):
    found=[key for key,pattern in CATEGORY_PATTERNS.items() if re.search(pattern,text,re.I)]
    if 'vibe_coding' in found and 'coding' in found: found.remove('coding')
    return found or ['general']


def sentiment(text):
    signs=[]
    # Quoted output and sarcastic cues are not safe literal sentiment evidence.
    if re.search(r'/s\b|yeah[ ,]+right|as if|thanks for nothing',text,re.I): return None
    if re.search(r'\b(?:burns? through|drains?|consum\w*).{0,55}\b(?:quota|allowance|budget|credits?)\b',text,re.I):
        return -1
    if re.search(r'\b(?:not enough improvement|takes a lot of steering|needs multiple prompts)\b',text,re.I):
        return None
    for pattern,sign in ((POSITIVE,1),(NEGATIVE,-1)):
        for match in re.finditer(pattern,text,re.I):
            prefix=text[max(0,match.start()-45):match.start()].replace('\u2019', "'")
            if re.search(r'\bnot only\s*$',prefix,re.I): negated=False
            else: negated=bool(re.search(r"\b(?:not|never|no|isn't|wasn't|aren't|can't|don't|doesn't|didn't)\s+(?:\w+\s+){0,3}$",prefix,re.I))
            signs.append(-sign if negated else sign)
    return signs[0] if signs and len(set(signs))==1 else None


def masked_text(text):
    masked=text
    # Preserve offsets exactly while hiding code, URLs, and explicit quoted lines.
    patterns=[r'```[\s\S]*?(?:```|$)',r'`[^`\n]+`',r'https?://\S+',r'(?m)^\s*>[^\n]*',r'"[^"\n]+"',r'\u201c[^\u201d\n]+\u201d']
    for pattern in patterns:
        masked=re.sub(pattern,lambda m:' '*len(m.group()),masked)
    return masked


def explicit_models(text, catalog):
    patterns=[
        (r"\b(?:claude[- ]+)?(opus|sonnet|haiku)[- ]+(\d+(?:\.\d+)?)\b", 'anthropic', lambda m:'claude-'+m[1].lower()+'-'+m[2]),
        (r"\bgemini[- ]+(\d+(?:\.\d+)?)[- ]+(pro|flash(?:[- ]lite)?)\b", 'google', lambda m:'gemini-'+m[1]+'-'+m[2].lower().replace(' ','-')),
        (r"\bgpt[- ]?(\d+(?:\.\d+)?)[- ]+(astra|sol|terra|luna)\b", 'openai', lambda m:'gpt-'+m[1]+'-'+m[2].lower()),
    ]
    spans=[];found={}
    for pattern,provider,canonical in patterns:
        for match in re.finditer(pattern,text,re.I):
            spans.append(match.span());found[canonical(match)]=provider
    for mention in catalog.find_mentions(text):
        if not any(mention.start < end and mention.end > start for start,end in spans):found[mention.model_id]=mention.provider_id
    return found

def split_claims(text, catalog=None, context_text=None):
    catalog=catalog or ModelCatalog()
    masked=masked_text(text)
    previous=None
    previous_attribution=None
    output=[]
    if context_text:
        context_models=explicit_models(masked_text(context_text),catalog)
        if len(context_models)==1:
            previous=next(iter(context_models.items()));previous_attribution='direct_parent_pronoun'
    offset=0

    def emit(start,end,model,provider,polarity,cats,attribution,kind='absolute'):
        fragment=text[start:end]
        for category in cats:
            output.append({'claim_id':hashlib.sha256(f'{start}:{end}:{model}:{category}:{polarity}:{fragment}'.encode()).hexdigest()[:20],
                'model_id':model,'provider_id':provider,'category':category,'polarity_score':polarity,
                'evidence_text':fragment,'span_start':start,'span_end':end,'attribution':attribution,
                'method':'aspect_rules_v2','review_status':'candidate','evidence_kind':kind})

    for match in list(BOUNDARY.finditer(masked))+[None]:
        if match and re.search(r'\band\b',match.group(),re.I):
            left_categories=categories(masked[offset:match.start()])
            next_boundary=BOUNDARY.search(masked,match.end())
            right_categories=categories(masked[match.end():next_boundary.start() if next_boundary else len(masked)])
            # "Good and bad at coding" is mixed sentiment, not two independent aspects.
            if left_categories==['general'] or set(left_categories)&set(right_categories):
                continue
        end=match.start() if match else len(text)
        segment=masked[offset:end]
        fragment=segment.strip(' ,\n')
        start=offset+len(segment)-len(segment.lstrip(' ,\n'))
        stop=start+len(fragment)
        boundary=match.group() if match else ''
        mentions=explicit_models(fragment,catalog)
        blocked=(not fragment or '?' in boundary or fragment.count('|')>=3 or bool(re.search(r'^(?:if|suppose|imagine|would|could|should|can|does|is it)\b',fragment,re.I)) or bool(re.search(r'\b(?:would|might|will probably|could be|metadata|routing.hint|api.key authentication|TUI startup|good harness)\b|/v1/|model=',fragment,re.I)))
        # Never reinterpret a named comparison target as the praised subject.
        comparison=re.search(r'\b(better than|worse than|outperforms?|beats?)\b',fragment,re.I)
        comparative_cue=re.search(r'\b(compared to|versus|vs|faster than|slower than|closer)\b|\bas\s+\w+\s+as\b',fragment,re.I)
        if not blocked and comparison:
            left=explicit_models(fragment[:comparison.start()],catalog)
            right=explicit_models(fragment[comparison.end():],catalog)
            if len(left)==len(right)==1 and set(left).isdisjoint(right) and not re.search(r"\b(?:not|never|isn't|isn\u2019t)\b",fragment[:comparison.start()],re.I):
                sign=-1 if comparison[1].lower()=='worse than' else 1
                for mapping,pol in ((left,sign),(right,-sign)):
                    model,provider=next(iter(mapping.items()))
                    emit(start,stop,model,provider,pol,categories(fragment),'explicit_comparison','relative_preference')
            previous=None
        elif not blocked and not comparative_cue:
            attribution='explicit'
            if len(mentions)==1:
                model,provider=next(iter(mentions.items()))
                previous=(model,provider);previous_attribution='same_post_pronoun'
            elif not mentions and previous and (PRONOUN.match(fragment) or (previous_attribution!='direct_parent_pronoun' and ELLIPSIS.match(fragment))):
                model,provider=previous;attribution=previous_attribution
            else:
                model=provider=None
                previous=None
            if model:
                polarity=sentiment(fragment)
                if polarity is not None:
                    emit(start,stop,model,provider,polarity,categories(fragment),attribution)
        else:
            previous=None
        # Paragraph boundaries end local reference scope. A single title newline may carry scope.
        if '\n\n' in boundary: previous=None
        offset=match.end() if match else len(text)
    return output
