import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  BadgeCheck,
  BarChart3,
  BookOpen,
  Building2,
  ChevronDown,
  ChevronUp,
  Database,
  ExternalLink,
  FileText,
  FlaskConical,
  Info,
  Layers,
  Search,
  ShieldCheck,
  Users,
} from 'lucide-react';
import {
  getObservations,
  getSnapshot,
  type EvidenceSample,
  type LeaderboardRow,
  type ObservationRow,
  type ProviderRow,
  type Snapshot,
  type TaskRow,
  type Tier,
} from './dataSource';

const providerNames: Record<string, string> = {
  anthropic: 'Anthropic',
  openai: 'OpenAI',
  google: 'Google',
  meta: 'Meta',
  mistral: 'Mistral',
  deepseek: 'DeepSeek',
  alibaba: 'Alibaba (Qwen)',
  xai: 'xAI',
  moonshot: 'Moonshot (Kimi)',
  unknown: 'Unattributed',
};

function providerName(id: string) {
  return providerNames[id] ?? id;
}

const aspectLabels: Record<string, string> = {
  overall: 'Overall',
  satisfaction: 'Satisfaction',
  trust_reliability: 'Trust & Reliability',
  task_fit: 'Task Fit',
  regression_stability: 'Regression Stability',
  hallucination_safety: 'Hallucination Safety',
  refusal_acceptance: 'Refusal Acceptance',
  value: 'Value',
  developer_ergonomics: 'Developer Ergonomics',
};

const taskLabels: Record<string, string> = {
  all: 'All tasks',
  coding: 'Coding',
  writing: 'Writing',
  research: 'Research',
  agents: 'Agents',
  roleplay: 'Roleplay',
  data_analysis: 'Data Analysis',
  long_context: 'Long Context',
  multimodal: 'Multimodal',
  api_developer_workflow: 'API & Dev Workflow',
  general: 'General',
};

const tierLabels: Record<Tier, string> = {
  insufficient: 'insufficient data',
  provisional: 'provisional',
  ranked: 'ranked',
};

type Page = 'providers' | 'leaderboard' | 'fields' | 'evidence' | 'methodology';

export function Dashboard() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [source, setSource] = useState<'supabase' | 'static' | ''>('');
  const [error, setError] = useState('');
  const [page, setPage] = useState<Page>('providers');

  useEffect(() => {
    getSnapshot()
      .then((result) => {
        setSnapshot(result.snapshot);
        setSource(result.source);
      })
      .catch(() =>
        setError(
          'No snapshot found. Load Supabase with load-supabase, or build the static file with build-score-snapshot --web-out web/public/snapshot.json.',
        ),
      );
  }, []);

  if (error) return <section className="empty-state">{error}</section>;
  if (!snapshot) return <section className="empty-state">Loading benchmark…</section>;

  return (
    <section className="benchmark-pane">
      <header className="topbar">
        <div>
          <p className="eyebrow">
            Snapshot {snapshot.snapshot_id} · {snapshot.corpus.models} models · {snapshot.corpus.threads} threads ·
            single source (Hacker News)
          </p>
          <h2>Community Model Benchmark</h2>
        </div>
        <div className="topbar-right">
          <span className={`source-pill ${source}`} title="Where this view's data was loaded from">
            <Database size={13} aria-hidden="true" />
            {source === 'supabase' ? 'live · Supabase' : 'static snapshot'}
          </span>
          <nav className="view-switch wide" aria-label="Dashboard page">
            <button className={page === 'providers' ? 'selected' : ''} onClick={() => setPage('providers')}>
              <Building2 size={15} aria-hidden="true" /> Providers
            </button>
            <button className={page === 'leaderboard' ? 'selected' : ''} onClick={() => setPage('leaderboard')}>
              <BarChart3 size={15} aria-hidden="true" /> Models
            </button>
            <button className={page === 'fields' ? 'selected' : ''} onClick={() => setPage('fields')}>
              <Layers size={15} aria-hidden="true" /> Fields
            </button>
            <button className={page === 'evidence' ? 'selected' : ''} onClick={() => setPage('evidence')}>
              <Search size={15} aria-hidden="true" /> Evidence
            </button>
            <button className={page === 'methodology' ? 'selected' : ''} onClick={() => setPage('methodology')}>
              <BookOpen size={15} aria-hidden="true" /> Methodology
            </button>
          </nav>
        </div>
      </header>

      <DataHealthBand snapshot={snapshot} />

      {page === 'providers' ? <ProvidersPage snapshot={snapshot} /> : null}
      {page === 'leaderboard' ? <LeaderboardPage snapshot={snapshot} /> : null}
      {page === 'fields' ? <FieldsPage snapshot={snapshot} /> : null}
      {page === 'evidence' ? <EvidencePage /> : null}
      {page === 'methodology' ? <Methodology snapshot={snapshot} /> : null}
    </section>
  );
}

function DataHealthBand({ snapshot }: { snapshot: Snapshot }) {
  const reviewed = snapshot.corpus.human_reviewed_included;
  const machine = snapshot.corpus.machine_included;
  return (
    <section className="health-band" aria-label="Data health">
      <div className="health-metric">
        <strong>{reviewed + machine}</strong>
        <span>scored observations</span>
      </div>
      <div className="health-metric">
        <strong>{snapshot.corpus.providers ?? snapshot.providers.length}</strong>
        <span>providers</span>
      </div>
      <div className="health-metric">
        <strong>{snapshot.corpus.models}</strong>
        <span>versioned models</span>
      </div>
      <div className="health-metric">
        <strong>{snapshot.corpus.threads}</strong>
        <span>threads</span>
      </div>
      <p className="health-note">
        <AlertTriangle size={14} aria-hidden="true" />
        Single-source (Hacker News) corpus. {snapshot.corpus.unspecified_observations ?? 0} of {reviewed} mentions name
        only a provider (e.g. “Claude”) with no version — those roll up to the <strong>provider</strong> board, not a
        specific model. Per-model evidence is thin, so models cap at <em>provisional</em>; treat low-n cells as
        directional.
      </p>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Providers (the rankable board)                                      */
/* ------------------------------------------------------------------ */

function ProvidersPage({ snapshot }: { snapshot: Snapshot }) {
  const [expanded, setExpanded] = useState('');
  const providers = useMemo(
    () =>
      [...snapshot.providers].sort((a, b) => {
        const order = { ranked: 0, provisional: 1, insufficient: 2 } as const;
        if (order[a.tier] !== order[b.tier]) return order[a.tier] - order[b.tier];
        return b.score - a.score;
      }),
    [snapshot.providers],
  );

  return (
    <>
      <section className="notice-band subtle">
        <Building2 size={15} aria-hidden="true" />
        Provider scores roll up every observation about a provider — versioned models <em>and</em> unversioned mentions
        like “Claude”. Because they pool far more evidence, providers reach <strong>ranked</strong> confidence where
        individual model versions can’t yet.
      </section>

      <section className="rank-table" aria-label="Provider ranking">
        <div className="rank-head provider-head">
          <span>#</span>
          <span>Provider</span>
          <span>Overall</span>
          <span>Confidence</span>
          <span>Evidence</span>
          <span>Status</span>
        </div>
        {providers.map((provider, index) => (
          <ProviderRowView
            key={provider.provider_id}
            provider={provider}
            rank={index + 1}
            expanded={expanded === provider.provider_id}
            onToggle={() => setExpanded(expanded === provider.provider_id ? '' : provider.provider_id)}
            snapshot={snapshot}
          />
        ))}
      </section>
    </>
  );
}

function ProviderRowView({
  provider,
  rank,
  expanded,
  onToggle,
  snapshot,
}: {
  provider: ProviderRow;
  rank: number;
  expanded: boolean;
  onToggle: () => void;
  snapshot: Snapshot;
}) {
  const unranked = provider.tier === 'insufficient';
  const unspecifiedShare = provider.n_observations
    ? Math.round((provider.unspecified_observations / provider.n_observations) * 100)
    : 0;
  return (
    <div className={`rank-row ${unranked ? 'unranked' : ''}`}>
      <button className="rank-line provider-line" onClick={onToggle} aria-expanded={expanded}>
        <span className="rank-number">{unranked ? '—' : rank}</span>
        <span className="rank-model">
          <strong>{providerName(provider.provider_id)}</strong>
          <small>{provider.n_models} versioned models</small>
        </span>
        <span className="rank-score">
          <strong>{provider.score.toFixed(1)}</strong>
          <ScoreBar score={provider.score} muted={unranked} />
        </span>
        <span className="rank-ci">ESS {provider.ess.toFixed(0)}</span>
        <span className="rank-evidence">
          {provider.n_observations} obs · {provider.n_threads} threads
          <small>{unspecifiedShare}% unversioned</small>
        </span>
        <span className={`tier-chip tier-${provider.tier}`}>
          {provider.tier === 'ranked' ? <BadgeCheck size={13} aria-hidden="true" /> : null}
          {tierLabels[provider.tier]}
        </span>
        {expanded ? <ChevronUp size={16} aria-hidden="true" /> : <ChevronDown size={16} aria-hidden="true" />}
      </button>
      {expanded ? (
        <div className="model-detail">
          <div className="aspect-strip">
            {provider.aspects.map((cell) => (
              <div key={cell.aspect} className="aspect-cell">
                <span>{aspectLabels[cell.aspect] ?? cell.aspect}</span>
                <strong>{cell.score.toFixed(0)}</strong>
                <ScoreBar score={cell.score} ci={cell.ci} muted={cell.tier === 'insufficient'} />
                <small>
                  ESS {cell.ess.toFixed(0)} · {cell.n_observations} obs
                </small>
              </div>
            ))}
          </div>
          <EvidenceList samples={snapshot.evidence[`${provider.provider_id}|satisfaction`] ?? []} title="Evidence receipts" />
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Leaderboard                                                         */
/* ------------------------------------------------------------------ */

type DisplayRow = {
  model_id: string;
  provider_id: string;
  aspect?: string;
  score: number;
  ci?: [number, number];
  ess?: number;
  n_observations?: number;
  n_threads?: number;
  n_authors?: number;
  firsthand_ratio?: number;
  warnings?: string[];
  tier: Tier;
};

function LeaderboardPage({ snapshot }: { snapshot: Snapshot }) {
  const [aspect, setAspect] = useState('overall');
  const [task, setTask] = useState('all');
  const [expanded, setExpanded] = useState('');
  const rows = useMemo(() => buildRows(snapshot, aspect, task), [snapshot, aspect, task]);

  return (
    <>
      <section className="dash-controls">
        <div className="control-group" role="group" aria-label="Aspect">
          {Object.keys(aspectLabels).map((key) => (
            <button key={key} className={aspect === key ? 'selected' : ''} onClick={() => setAspect(key)}>
              {aspectLabels[key]}
            </button>
          ))}
        </div>
        <label className="task-select">
          <span>Task</span>
          <select value={task} onChange={(event) => setTask(event.target.value)} disabled={aspect === 'overall'}>
            {Object.keys(taskLabels).map((key) => (
              <option key={key} value={key} disabled={key !== 'all' && !snapshot.tasks[key]}>
                {taskLabels[key]}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section className="notice-band subtle">
        <Info size={15} aria-hidden="true" />
        Specific <strong>versioned</strong> models only (e.g. claude-opus-4.8). Comments that name just a provider roll
        up to the <strong>Providers</strong> board instead. Per-model evidence is still thin, so models top out at
        provisional until the corpus grows.
      </section>

      <section className="rank-table" aria-label="Model ranking">
        <div className="rank-head">
          <span>#</span>
          <span>Model</span>
          <span>Score</span>
          <span>Confidence</span>
          <span>Evidence</span>
          <span>Sources</span>
          <span>Status</span>
        </div>
        {rows.map((row, index) => (
          <RankRow
            key={`${row.model_id}-${row.aspect ?? 'overall'}`}
            row={row}
            rank={index + 1}
            expanded={expanded === row.model_id}
            onToggle={() => setExpanded(expanded === row.model_id ? '' : row.model_id)}
            snapshot={snapshot}
            aspect={aspect}
          />
        ))}
        {rows.length === 0 ? (
          <div className="empty-cell">No models meet the minimum evidence for this view yet.</div>
        ) : null}
      </section>
    </>
  );
}

function buildRows(snapshot: Snapshot, aspect: string, task: string): DisplayRow[] {
  if (aspect === 'overall') {
    // Overall tier comes from the model's TOTAL evidence (computed in the
    // snapshot), not its weakest aspect — so a well-covered model is no longer
    // dragged to "insufficient" by one thin dimension.
    const providerOf = new Map(snapshot.leaderboard.map((row) => [row.model_id, row.provider_id]));
    const authorsOf = new Map<string, number>();
    for (const row of snapshot.leaderboard) {
      authorsOf.set(row.model_id, Math.max(authorsOf.get(row.model_id) ?? 0, row.n_authors));
    }
    return snapshot.overall
      .map((entry) => ({
        model_id: entry.model_id,
        provider_id: entry.provider_id ?? providerOf.get(entry.model_id) ?? '',
        score: entry.score,
        ess: entry.ess,
        n_observations: entry.n_observations,
        n_threads: entry.n_threads,
        n_authors: authorsOf.get(entry.model_id) ?? 0,
        tier: entry.tier,
      }))
      .filter((row) => row.tier !== 'insufficient' || row.n_observations >= 5)
      .sort(rankOrder);
  }
  const source: (LeaderboardRow | TaskRow)[] = task === 'all' ? snapshot.leaderboard : snapshot.tasks[task] ?? [];
  return source
    .filter((row) => row.aspect === aspect)
    .map((row) => ({ ...(row as LeaderboardRow), provider_id: (row as LeaderboardRow).provider_id ?? '' }))
    .sort(rankOrder);
}

function rankOrder(a: DisplayRow, b: DisplayRow) {
  const tiers = { ranked: 0, provisional: 1, insufficient: 2 };
  if (tiers[a.tier] !== tiers[b.tier]) return tiers[a.tier] - tiers[b.tier];
  return b.score - a.score;
}

function tierFromEss(ess: number): Tier {
  if (ess < 30) return 'insufficient';
  if (ess < 150) return 'provisional';
  return 'ranked';
}

function RankRow({
  row,
  rank,
  expanded,
  onToggle,
  snapshot,
  aspect,
}: {
  row: DisplayRow;
  rank: number;
  expanded: boolean;
  onToggle: () => void;
  snapshot: Snapshot;
  aspect: string;
}) {
  const ciWidth = row.ci ? row.ci[1] - row.ci[0] : 0;
  const unranked = row.tier === 'insufficient';
  return (
    <div className={`rank-row ${unranked ? 'unranked' : ''}`}>
      <button className="rank-line" onClick={onToggle} aria-expanded={expanded}>
        <span className="rank-number">{unranked ? '—' : rank}</span>
        <span className="rank-model">
          <strong>{row.model_id}</strong>
          <small>{row.provider_id}</small>
        </span>
        <span className="rank-score">
          <strong>{unranked ? '·' : row.score.toFixed(1)}</strong>
          <ScoreBar score={row.score} ci={row.ci} muted={unranked} />
        </span>
        <span className="rank-ci">
          {row.ci ? `${row.ci[0].toFixed(0)}–${row.ci[1].toFixed(0)}` : '—'}
          {ciWidth > 30 ? <AlertTriangle size={14} aria-label="Wide confidence interval" /> : null}
        </span>
        <span className="rank-evidence">
          {row.n_observations ?? '—'} obs · ESS {row.ess?.toFixed(0) ?? '—'}
          {row.firsthand_ratio !== undefined ? <small>{Math.round(row.firsthand_ratio * 100)}% firsthand</small> : null}
        </span>
        <span className="rank-sources">
          <Users size={14} aria-hidden="true" />
          {row.n_threads ?? 0} threads · {row.n_authors ?? 0} authors
        </span>
        <span className={`tier-chip tier-${row.tier}`}>
          {row.tier === 'ranked' ? <BadgeCheck size={13} aria-hidden="true" /> : null}
          {tierLabels[row.tier]}
        </span>
        {expanded ? <ChevronUp size={16} aria-hidden="true" /> : <ChevronDown size={16} aria-hidden="true" />}
      </button>
      {expanded ? <ModelDetail model={row.model_id} snapshot={snapshot} aspect={aspect} /> : null}
    </div>
  );
}

function ScoreBar({ score, ci, muted }: { score: number; ci?: [number, number]; muted?: boolean }) {
  return (
    <span className={`score-bar ${muted ? 'muted' : ''}`} aria-hidden="true">
      {ci ? <span className="score-bar-ci" style={{ left: `${ci[0]}%`, width: `${Math.max(1, ci[1] - ci[0])}%` }} /> : null}
      <span className="score-bar-dot" style={{ left: `${score}%` }} />
    </span>
  );
}

function ModelDetail({ model, snapshot, aspect }: { model: string; snapshot: Snapshot; aspect: string }) {
  const aspects = snapshot.leaderboard.filter((row) => row.model_id === model);
  const evidenceKey = `${model}|${aspect === 'overall' ? 'satisfaction' : aspect}`;
  const samples = snapshot.evidence[evidenceKey] ?? [];
  return (
    <div className="model-detail">
      <div className="aspect-strip">
        {aspects.map((row) => (
          <div key={row.aspect} className="aspect-cell">
            <span>{aspectLabels[row.aspect] ?? row.aspect}</span>
            <strong>{row.score.toFixed(0)}</strong>
            <ScoreBar score={row.score} ci={row.ci} muted={row.tier === 'insufficient'} />
            <small>
              ESS {row.ess.toFixed(0)} · {tierLabels[row.tier]}
            </small>
          </div>
        ))}
      </div>
      <EvidenceList samples={samples} title="Evidence receipts" />
    </div>
  );
}

function EvidenceList({ samples, title }: { samples: EvidenceSample[]; title: string }) {
  if (samples.length === 0) return null;
  return (
    <div className="evidence-list">
      <h4>{title}</h4>
      {samples.map((sample, index) => (
        <blockquote key={index} className="evidence-item">
          <p>“{sample.span}”</p>
          <footer>
            <span className={sample.polarity >= 0 ? 'pol-pos' : 'pol-neg'}>
              {sample.polarity > 0 ? '+' : ''}
              {sample.polarity}
            </span>
            <span>{sample.evidence_type.replaceAll('_', ' ')}</span>
            <span>{sample.firsthand ? 'firsthand' : 'secondhand'}</span>
            <span className={sample.human_labeled ? 'label-human' : 'label-auto'}>
              {sample.human_labeled ? 'human reviewed' : 'auto (gated)'}
            </span>
            <a href={sample.url} target="_blank" rel="noreferrer">
              source <ExternalLink size={12} aria-hidden="true" />
            </a>
          </footer>
        </blockquote>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Fields (benchmarks by task)                                         */
/* ------------------------------------------------------------------ */

function FieldsPage({ snapshot }: { snapshot: Snapshot }) {
  const taskKeys = useMemo(
    () => Object.keys(snapshot.tasks).sort((a, b) => (taskLabels[a] ?? a).localeCompare(taskLabels[b] ?? b)),
    [snapshot.tasks],
  );
  const [task, setTask] = useState(taskKeys[0] ?? 'coding');
  const activeTask = snapshot.tasks[task] ?? [];

  const byAspect = useMemo(() => {
    const grouped: Record<string, TaskRow[]> = {};
    for (const row of activeTask) {
      (grouped[row.aspect] ??= []).push(row);
    }
    for (const list of Object.values(grouped)) {
      list.sort((a, b) => {
        const order = { ranked: 0, provisional: 1, insufficient: 2 } as const;
        if (order[a.tier] !== order[b.tier]) return order[a.tier] - order[b.tier];
        return b.score - a.score;
      });
    }
    return grouped;
  }, [activeTask]);

  const aspectOrder = Object.keys(aspectLabels).filter((key) => key !== 'overall' && byAspect[key]);

  return (
    <>
      <section className="dash-controls">
        <div className="control-group" role="group" aria-label="Task">
          {taskKeys.map((key) => (
            <button key={key} className={task === key ? 'selected' : ''} onClick={() => setTask(key)}>
              {taskLabels[key] ?? key}
            </button>
          ))}
        </div>
      </section>

      <section className="notice-band subtle">
        <Layers size={15} aria-hidden="true" />
        How models compare on <strong>{taskLabels[task] ?? task}</strong> work across each quality dimension. Only
        observations whose task label cleared the calibrated gate feed this view.
      </section>

      {aspectOrder.length === 0 ? (
        <div className="empty-cell">No task-gated observations for this field yet.</div>
      ) : (
        <div className="fields-grid">
          {aspectOrder.map((key) => (
            <article key={key} className="field-card">
              <header>
                <h3>{aspectLabels[key]}</h3>
                <span>{byAspect[key].length} models</span>
              </header>
              {byAspect[key].slice(0, 6).map((row, index) => (
                <FieldRow key={`${row.model_id}-${index}`} row={row} rank={index + 1} snapshot={snapshot} aspect={key} />
              ))}
            </article>
          ))}
        </div>
      )}
    </>
  );
}

function FieldRow({
  row,
  rank,
  snapshot,
  aspect,
}: {
  row: TaskRow;
  rank: number;
  snapshot: Snapshot;
  aspect: string;
}) {
  const [open, setOpen] = useState(false);
  const samples = snapshot.evidence[`${row.model_id}|${aspect}`] ?? [];
  const unranked = row.tier === 'insufficient';
  return (
    <div className={`field-row-line ${unranked ? 'unranked' : ''}`}>
      <button onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="field-rank">{unranked ? '—' : rank}</span>
        <span className="field-model">{row.model_id}</span>
        <ScoreBar score={row.score} ci={row.ci} muted={unranked} />
        <span className="field-score">{row.score.toFixed(0)}</span>
        <span className={`tier-dot tier-${row.tier}`} title={tierLabels[row.tier]} />
        <small>ESS {row.ess.toFixed(0)}</small>
      </button>
      {open ? <EvidenceList samples={samples} title="Why" /> : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Evidence explorer                                                   */
/* ------------------------------------------------------------------ */

const PAGE_SIZE = 40;

function EvidencePage() {
  const [rows, setRows] = useState<ObservationRow[] | null>(null);
  const [source, setSource] = useState<'supabase' | 'static' | ''>('');
  const [model, setModel] = useState('all');
  const [aspect, setAspect] = useState('all');
  const [task, setTask] = useState('all');
  const [polarity, setPolarity] = useState('all');
  const [firsthand, setFirsthand] = useState('all');
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(PAGE_SIZE);

  useEffect(() => {
    getObservations().then((result) => {
      setRows(result.rows);
      setSource(result.source);
    });
  }, []);

  const options = useMemo(() => {
    const models = new Set<string>();
    const aspects = new Set<string>();
    const tasks = new Set<string>();
    for (const row of rows ?? []) {
      if (row.model_id) models.add(row.model_id);
      if (row.aspect_category) aspects.add(row.aspect_category);
      if (row.task_category) tasks.add(row.task_category);
    }
    return {
      models: [...models].sort(),
      aspects: [...aspects].sort(),
      tasks: [...tasks].sort(),
    };
  }, [rows]);

  const filtered = useMemo(() => {
    const text = query.trim().toLowerCase();
    return (rows ?? []).filter((row) => {
      if (model !== 'all' && row.model_id !== model) return false;
      if (aspect !== 'all' && row.aspect_category !== aspect) return false;
      if (task !== 'all' && row.task_category !== task) return false;
      if (polarity === 'pos' && row.polarity_score <= 0) return false;
      if (polarity === 'neg' && row.polarity_score >= 0) return false;
      if (polarity === 'neu' && row.polarity_score !== 0) return false;
      if (firsthand === 'yes' && !row.firsthand_flag) return false;
      if (firsthand === 'no' && row.firsthand_flag) return false;
      if (text && !row.evidence_text.toLowerCase().includes(text)) return false;
      return true;
    });
  }, [rows, model, aspect, task, polarity, firsthand, query]);

  useEffect(() => setLimit(PAGE_SIZE), [model, aspect, task, polarity, firsthand, query]);

  if (!rows) return <div className="empty-cell">Loading evidence…</div>;

  return (
    <>
      <section className="notice-band subtle">
        <Search size={15} aria-hidden="true" />
        Every reviewed observation behind the scores, with full provenance. {filtered.length} of {rows.length} shown ·
        source: {source === 'supabase' ? 'Supabase' : 'static file'}.
      </section>

      <section className="evidence-filters">
        <FilterSelect label="Model" value={model} onChange={setModel} options={options.models} />
        <FilterSelect label="Aspect" value={aspect} onChange={setAspect} options={options.aspects} labels={aspectLabels} />
        <FilterSelect label="Task" value={task} onChange={setTask} options={options.tasks} labels={taskLabels} />
        <FilterSelect
          label="Polarity"
          value={polarity}
          onChange={setPolarity}
          options={['pos', 'neu', 'neg']}
          labels={{ pos: 'Positive', neu: 'Neutral', neg: 'Negative' }}
        />
        <FilterSelect
          label="Firsthand"
          value={firsthand}
          onChange={setFirsthand}
          options={['yes', 'no']}
          labels={{ yes: 'Firsthand', no: 'Secondhand' }}
        />
        <label className="evidence-search">
          <Search size={14} aria-hidden="true" />
          <input value={query} placeholder="Search text…" onChange={(event) => setQuery(event.target.value)} />
        </label>
      </section>

      <section className="evidence-table" aria-label="Observations">
        {filtered.slice(0, limit).map((row, index) => (
          <article key={`${row.source_item_id}-${row.model_id}-${index}`} className="obs-row">
            <div className="obs-head">
              <strong>{row.model_id}</strong>
              <span className="obs-tag">{row.provider_id || 'unknown'}</span>
              <span className="obs-tag">{aspectLabels[row.aspect_category] ?? row.aspect_category}</span>
              <span className="obs-tag">{taskLabels[row.task_category] ?? row.task_category}</span>
              <span className={`obs-pol ${row.polarity_score >= 0 ? 'pol-pos' : 'pol-neg'}`}>
                {row.polarity_score > 0 ? '+' : ''}
                {row.polarity_score}
              </span>
              <span className="obs-tag subtle">{row.evidence_type.replaceAll('_', ' ')}</span>
              <span className="obs-tag subtle">{row.firsthand_flag ? 'firsthand' : 'secondhand'}</span>
              {row.url ? (
                <a href={row.url} target="_blank" rel="noreferrer">
                  source <ExternalLink size={12} aria-hidden="true" />
                </a>
              ) : null}
            </div>
            <p>{row.evidence_text}</p>
          </article>
        ))}
        {filtered.length === 0 ? <div className="empty-cell">No observations match these filters.</div> : null}
        {filtered.length > limit ? (
          <button className="load-more" onClick={() => setLimit(limit + PAGE_SIZE)}>
            Show more ({filtered.length - limit} remaining)
          </button>
        ) : null}
      </section>
    </>
  );
}

function FilterSelect({
  label,
  value,
  onChange,
  options,
  labels = {},
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: string[];
  labels?: Record<string, string>;
}) {
  return (
    <label className="filter-select">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="all">All</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {labels[option] ?? option}
          </option>
        ))}
      </select>
    </label>
  );
}

/* ------------------------------------------------------------------ */
/* Methodology / technological efforts                                 */
/* ------------------------------------------------------------------ */

const fieldLabels: Record<string, string> = {
  firsthand_flag: 'Firsthand',
  aspect_category: 'Aspect',
  evidence_type: 'Evidence type',
  task_category: 'Task',
  polarity_score: 'Polarity (sign)',
};

const timeline: { title: string; detail: string; metric: string }[] = [
  {
    title: 'Rule + Naive Bayes baseline',
    detail: 'Dependency-free extraction and a bag-of-words classifier on the first ~550 reviewed spans.',
    metric: 'bootstrap',
  },
  {
    title: 'Routed rubric classifier',
    detail: 'Per-field frozen encoders (BGE / GTE / E5) with logistic heads, selective soft-chaining and zero-shot NLI polarity features.',
    metric: 'macro F1 0.42 → 0.46',
  },
  {
    title: 'Multi-encoder stacking',
    detail: 'Four frozen encoders concatenated (BGE-small/-base, GTE-base, E5-base, MPNet) feeding per-field heads.',
    metric: 'macro F1 0.4700',
  },
  {
    title: 'Flat sign-level polarity head',
    detail: 'Collapsed −2/+2 magnitude into negative / neutral / positive; the flat head beat the ordinal one.',
    metric: '3-class precision @ gate 0.81',
  },
  {
    title: 'Isotonic confidence calibration',
    detail: 'Leakage-safe isotonic regression on inner out-of-fold predictions maps raw scores to interpretable gates.',
    metric: 'all 5 fields ≥80% precision points',
  },
  {
    title: 'Fine-tuned shared encoder',
    detail: 'BGE-base fine-tuned with multi-task per-field heads and partial layer unfreezing — one model, all fields.',
    metric: 'macro F1 0.4702 · firsthand F1 0.80',
  },
];

function Methodology({ snapshot }: { snapshot: Snapshot }) {
  const corpus = snapshot.corpus;
  const artifactKey = Object.keys(snapshot.methodology.gated_precision_artifacts)[0];
  const artifact = artifactKey ? snapshot.methodology.gated_precision_artifacts[artifactKey] : undefined;

  return (
    <section className="methodology" aria-label="Methodology">
      <section className="notice-band subtle">
        <ShieldCheck size={15} aria-hidden="true" />
        Every number on this dashboard is either human-reviewed or produced by a classifier above a calibrated
        confidence gate with measured ≥80% out-of-fold precision. Below-gate predictions are excluded, never guessed.
      </section>

      {snapshot.coverage ? <CoverageBand snapshot={snapshot} /> : null}

      <div className="method-grid">
        <article>
          <h3>
            <FlaskConical size={16} aria-hidden="true" /> How the data flows
          </h3>
          <ol>
            {snapshot.methodology.pipeline.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
        </article>
        <article>
          <h3>This snapshot</h3>
          <ul>
            <li>{corpus.training_examples} human-reviewed training observations</li>
            <li>{corpus.human_reviewed_included} human-reviewed observations in scoring</li>
            <li>{corpus.machine_included} machine-labeled observations admitted by the gates</li>
            <li>{corpus.machine_rejected_by_gates} machine predictions rejected for low confidence</li>
            <li>{corpus.excluded_by_review} spans excluded by reviewers as not model-quality claims</li>
          </ul>
        </article>
        <article>
          <h3>Confidence gates</h3>
          <ul>
            {Object.entries(snapshot.methodology.gates).map(([field, threshold]) => (
              <li key={field}>
                <code>{fieldLabels[field] ?? field.replaceAll('_', ' ')}</code>: calibrated confidence ≥{' '}
                {threshold.toFixed(2)}
              </li>
            ))}
          </ul>
          <p>
            Polarity is published at sign level only (negative / neutral / positive); magnitude always requires human
            review.
          </p>
        </article>
        <article>
          <h3>Ranking rules</h3>
          <ul>
            <li>Effective sample size (ESS) below 30 → unranked, shown as insufficient data.</li>
            <li>ESS 30–150 → provisional badge with a visible confidence interval.</li>
            <li>ESS 150+ → fully ranked. Thread, author, and community caps stop any single discussion from dominating.</li>
            <li>Hacker News is currently the only source, so all scores carry a single-source caveat until more platforms are added.</li>
          </ul>
        </article>
      </div>

      <section className="method-section">
        <h3>
          <Layers size={16} aria-hidden="true" /> Classifier evolution
        </h3>
        <p className="method-lede">
          The extraction model is the heart of the project: turning messy community comments into structured,
          provenance-backed observations without paid per-post LLM calls. Each iteration was evaluated with
          thread-grouped holdouts so threads never leak between train and test.
        </p>
        <ol className="timeline">
          {timeline.map((item) => (
            <li key={item.title}>
              <div className="timeline-dot" aria-hidden="true" />
              <div>
                <strong>{item.title}</strong>
                <span className="timeline-metric">{item.metric}</span>
                <p>{item.detail}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      {artifact ? (
        <section className="method-section">
          <h3>
            <BarChart3 size={16} aria-hidden="true" /> Calibrated gated precision
          </h3>
          <p className="method-lede">
            For each field we sweep the calibrated-confidence threshold and measure out-of-fold precision and the share
            of observations that clear it (coverage). The dashboard publishes at the lowest threshold that reaches ≥80%
            precision.
          </p>
          <div className="gp-table">
            <div className="gp-head">
              <span>Field</span>
              <span>Best ≥80% point</span>
              <span>Precision</span>
              <span>Coverage</span>
            </div>
            {Object.entries(artifact).map(([field, data]) => {
              const best = bestThreshold(data.thresholds);
              return (
                <div className="gp-row" key={field}>
                  <span>{fieldLabels[field] ?? field.replaceAll('_', ' ')}</span>
                  <span>{best ? `conf ≥ ${best.threshold}` : '— not reached —'}</span>
                  <span>{best ? `${(best.precision * 100).toFixed(0)}%` : '—'}</span>
                  <span>{best ? `${(best.coverage * 100).toFixed(0)}%` : '—'}</span>
                </div>
              );
            })}
          </div>
          <p className="method-footnote">
            <FileText size={13} aria-hidden="true" /> Source artifact: <code>{artifactKey}</code>
          </p>
        </section>
      ) : null}
    </section>
  );
}

function CoverageBand({ snapshot }: { snapshot: Snapshot }) {
  const coverage = snapshot.coverage!;
  const status = coverage.by_status ?? {};
  const total = Object.values(status).reduce((sum, value) => sum + value, 0) || 1;
  const ranked = snapshot.tier_thresholds?.ranked_ess ?? 50;
  const provisional = snapshot.tier_thresholds?.provisional_ess ?? 15;
  const statusOrder: [string, string][] = [
    ['registry', 'In registry'],
    ['versioned_unregistered', 'Versioned, needs registry entry'],
    ['unversioned', 'Unversioned → provider bucket'],
    ['unknown', 'Unattributed'],
  ];
  const unregistered = Object.entries(coverage.unregistered_versioned ?? {}).slice(0, 10);
  return (
    <section className="method-section">
      <h3>
        <Building2 size={16} aria-hidden="true" /> Model identity &amp; registry coverage
      </h3>
      <p className="method-lede">
        Raw mentions are resolved against <code>config/model_registry.json</code>. Unversioned mentions roll up to the
        provider; versioned mentions not yet in the registry are still scored and flagged here so the registry can be
        refreshed from provider docs — that is how new models enter the board. Tiers for this corpus: ranked at ESS ≥{' '}
        {ranked}, provisional at ESS ≥ {provisional}.
      </p>
      <div className="coverage-bars">
        {statusOrder.map(([key, label]) => {
          const value = status[key] ?? 0;
          return (
            <div className="coverage-bar" key={key}>
              <span className="coverage-label">{label}</span>
              <span className="coverage-track">
                <span className={`coverage-fill cov-${key}`} style={{ width: `${(value / total) * 100}%` }} />
              </span>
              <span className="coverage-count">{value}</span>
            </div>
          );
        })}
      </div>
      {unregistered.length > 0 ? (
        <p className="method-footnote">
          <AlertTriangle size={13} aria-hidden="true" /> Top versioned models awaiting a registry entry:{' '}
          {unregistered.map(([model, count]) => `${model} (${count})`).join(', ')}.
        </p>
      ) : null}
    </section>
  );
}

function bestThreshold(thresholds: Record<string, { precision: number; coverage: number }>) {
  const entries = Object.entries(thresholds)
    .map(([threshold, value]) => ({ threshold, ...value }))
    .sort((a, b) => Number(a.threshold) - Number(b.threshold));
  return entries.find((entry) => entry.precision >= 0.8) ?? null;
}
