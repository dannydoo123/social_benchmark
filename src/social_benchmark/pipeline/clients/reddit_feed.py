"""Public Atom feed ingestion. RSS is partial coverage, never a full Reddit archive."""
from datetime import datetime
import re
import time
import xml.etree.ElementTree as ET
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from social_benchmark.pipeline.models import RawItem, SourcePlatform
from social_benchmark.pipeline.normalization import normalize_evidence_text,normalize_text, hash_identifier

NS={'a':'http://www.w3.org/2005/Atom'}

def parse_reddit_feed(payload):
    root=ET.fromstring(payload)
    rows=[]
    for entry in root.findall('a:entry',NS):
        identifier=entry.findtext('a:id','',NS)
        link=entry.find('a:link',NS)
        url=link.get('href','') if link is not None else ''
        match=re.search(r'/r/([^/]+)/comments/([^/]+)',url)
        category=entry.find('a:category',NS)
        community=match[1] if match else (category.get('term','') if category is not None else '')
        thread='t3_'+match[2] if match else None
        published=entry.findtext('a:published',None,NS) or entry.findtext('a:updated',None,NS)
        author=entry.findtext('a:author/a:name',None,NS)
        body=entry.findtext('a:content','',NS)
        # Remove RSS transport attribution, not the author's prose.
        body=re.split(r'submitted by\s',body,flags=re.I)[0]
        body=normalize_evidence_text(body)
        if not identifier or not url or not body: continue
        is_post=identifier.startswith('t3_')
        rows.append(RawItem(platform=SourcePlatform.REDDIT,source_id=identifier,title=normalize_text(entry.findtext('a:title','',NS)) if is_post else '',body=body,url=url,community_id=community,thread_id=thread or (identifier if is_post else None),parent_id=None,author_id_hash=hash_identifier(author) if author and author not in ('/u/[deleted]','[deleted]') else None,published_at=datetime.fromisoformat(published.replace('Z','+00:00')) if published else None,metadata={'transport':'rss','type':'post' if is_post else 'comment','coverage':'partial_feed','parent_unavailable':True,'engagement_unavailable':True}))
    return rows

class RedditFeedClient:
    def __init__(self, delay=5):
        self.delay=delay
        self.last_request=0
    def fetch(self,url):
        time.sleep(max(0,self.delay-(time.monotonic()-self.last_request)))
        self.last_request=time.monotonic()
        request=Request(url,headers={'User-Agent':'social-benchmark/0.2 public community research','Accept':'application/atom+xml'})
        try:
            with urlopen(request,timeout=20) as response: return parse_reddit_feed(response.read())
        except HTTPError as exc:
            # Do not evade 403 or hammer 429. Let the next scheduled run retry.
            raise RuntimeError(f'Reddit RSS HTTP {exc.code}; retry next run') from exc
    def listing(self,community,sort='new'):
        return self.fetch(f'https://www.reddit.com/r/{community}/{sort}/.rss')
    def thread(self,url):
        match=re.search(r'(https://www.reddit.com/r/[^/]+/comments/[^/]+)',url)
        return self.fetch(match[1]+'/.rss') if match else []
