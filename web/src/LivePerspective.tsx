import { useEffect, useState } from 'react';
type Model = { model: string; category: string; observations: number; accepted: number; threads: number; authors: number; score: number | null; reasons: string[]; communities: {community:string;threads:number;score:number}[]; evidence:{url:string;community:string;published_at:string;review_status:string;aspect:string}[] };
type Feed = { generated_at:string; refresh_hours:number; window_days:number; raw_items:number; classifier_error:string|null; sources:{source:string;status:string;items?:number;error?:string}[]; models:Model[]; methodology:string };
export function LivePerspective() {
  const [feed,setFeed]=useState<Feed|null>(null);
  const [error,setError]=useState('');
  const [category,setCategory]=useState('all');
  const [query,setQuery]=useState('');
  const [community,setCommunity]=useState('all');
  const [clock,setClock]=useState(Date.now());
  useEffect(()=>{const load=()=>fetch('/live-perspective.json',{cache:'no-store'}).then(r=>{if(!r.ok) throw new Error('Fresh evidence feed has not been generated.');return r.json();}).then(v=>{setFeed(v);setError('');}).catch(e=>setError(String(e)));void load();const timer=setInterval(()=>{setClock(Date.now());void load();},60000);return()=>clearInterval(timer);},[]);
  if(!feed) return <p role="status">{error || 'Loading fresh evidence…'}</p>;
  const stale=clock-Date.parse(feed.generated_at)>feed.refresh_hours*3600000;
  const communities=[...new Set(feed.models.flatMap(m=>m.evidence.map(e=>e.community)))].sort();
  const rows=feed.models.filter(m=>m.category===category && m.model.includes(query.toLowerCase()) && (community==='all'||m.evidence.some(e=>e.community===community)));
  return <section className="methodology" aria-label="Fresh community perspective">
    <h2>How people experience models</h2><p>Fresh conversations, model-specific evidence, and visible uncertainty.</p>
    <div className="data-health-band"><strong>{stale?'Refresh overdue':feed.sources.every(s=>s.status!=='ok')?'Collection unavailable':'Collection attempted recently'}</strong> · {new Date(feed.generated_at).toLocaleString()} · target every {feed.refresh_hours} hours · rolling {feed.window_days} days · {feed.raw_items} stored posts/comments</div>
    {error && <p role="alert">{error}</p>}
    <p>Training examples are excluded. Collection freshness does not imply recent evidence or reliable scores.</p>
    <div style={{display:'flex',gap:16,flexWrap:'wrap',margin:'24px 0'}}>
      <label>Category <select value={category} onChange={e=>setCategory(e.target.value)}>{['all','cost_efficiency','vibe_coding','design','coding','research','writing','agents','general'].map(c=><option key={c} value={c}>{c.replaceAll('_',' ')}</option>)}</select></label>
      <label>Model <input placeholder="Search exact model" value={query} onChange={e=>setQuery(e.target.value)}/></label>
      <label>Community <select value={community} onChange={e=>setCommunity(e.target.value)}><option value="all">All communities</option>{communities.map(c=><option key={c}>{c}</option>)}</select></label>
    </div>
    {!rows.length && <p>No qualifying fresh evidence for these filters. No score is assigned.</p>}
    <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(280px,1fr))',gap:20}}>{rows.map(m=><article key={m.model} style={{border:'1px solid #ccc',borderRadius:12,padding:20}}><h3>{m.model}</h3><strong>{m.score===null?'Not enough validated evidence':`${m.score}/100`}</strong><p>{m.observations} candidate observations · {m.accepted} screened firsthand rows · {m.threads} independent threads · {m.authors} authors</p><details><summary>Why no score?</summary><ul>{m.reasons.map(r=><li key={r}>{r}</li>)}</ul></details><details><summary>Conversation sources</summary><ul>{m.evidence.map((e,i)=><li key={i}><a href={e.url} target="_blank" rel="noreferrer">{e.community} · {e.aspect}</a> — {new Date(e.published_at).toLocaleDateString()} · {e.review_status}</li>)}</ul></details></article>)}</div>
    <h3>Collection health</h3><ul>{feed.sources.map(s=><li key={s.source}>{s.source}: {s.status==='ok'?`${s.items} items fetched`:`unavailable (${s.error})`}</li>)}</ul>
    {feed.classifier_error && <p>Classifier unavailable: {feed.classifier_error}. Candidates remain unscored.</p>}
    <p>{feed.methodology}</p>
  </section>;
}
