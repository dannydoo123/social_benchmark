import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  BarChart3,
  BookOpen,
  Check,
  ClipboardList,
  Download,
  ExternalLink,
  FileText,
  RotateCcw,
  Upload,
} from 'lucide-react';
import { benchmarkModels, sampleAspectScores } from './benchmarkData';
import { sampleRows } from './sampleRows';
import type { LabelRow } from './types';

const taskOptions = [
  'coding',
  'writing',
  'research',
  'agents',
  'roleplay',
  'data_analysis',
  'long_context',
  'multimodal',
  'api_developer_workflow',
  'general',
];

const aspectOptions = [
  'satisfaction',
  'trust_reliability',
  'task_fit',
  'regression_stability',
  'hallucination_safety',
  'refusal_acceptance',
  'value',
  'developer_ergonomics',
];

const evidenceOptions = [
  'firsthand_usage',
  'comparative_evaluation',
  'bug_regression_report',
  'integration_failure',
  'benchmark_anecdote',
  'hearsay',
  'release_update_reaction',
  'pricing_value_comment',
];

const polarityOptions = [
  { value: -2, label: '-2 strong complaint' },
  { value: -1, label: '-1 complaint' },
  { value: 0, label: '0 neutral' },
  { value: 1, label: '1 praise' },
  { value: 2, label: '2 strong praise' },
];

const exclusionReasonOptions = [
  'not_about_model_quality',
  'factual_release_or_adoption',
  'too_vague_or_speculative',
  'duplicate_or_low_value',
  'off_topic',
  'other',
];

const aspectGuide = [
  ['satisfaction', 'General like/dislike: "nailed it", "best so far", "terrible output".'],
  ['trust_reliability', 'Dependability: stable, flaky, production-safe, breaks often.'],
  ['task_fit', 'Whether it fits the exact job: coding, writing, research, agents, etc.'],
  ['regression_stability', 'The model got worse, changed behavior, or broke an old workflow.'],
  ['hallucination_safety', 'Made-up facts, fake citations, invented APIs, wrong claims.'],
  ['refusal_acceptance', 'Refuses too much, is over-cautious, or handles safety boundaries well.'],
  ['value', 'Worth the cost, quota, subscription, latency, or usage limits.'],
  ['developer_ergonomics', 'API, SDK, docs, latency, rate limits, integration bugs.'],
] as const;

const firsthandGuide = [
  ['True', '"I used Opus 4.8", "we deployed GPT-5", "my tests show..."'],
  ['False', '"People say...", release notes, rumors, speculation, summaries of community sentiment.'],
] as const;

const identityGuide = [
  ['Provider', 'Company/lab: anthropic, openai, google.'],
  ['Model', 'Actual base model: claude-opus-4.8, gpt-5.5, gemini-3.5-flash.'],
  ['Product', 'Access surface: claude-code, chatgpt, openai-api, cursor.'],
  ['Profile', 'Run mode if stated: ultracode, high_effort, thinking. Blank if not stated.'],
] as const;

const evidenceGuide = [
  ['firsthand_usage', 'The author directly used, tested, or deployed it.'],
  ['comparative_evaluation', 'Direct comparison: "better than", "worse than", "vs".'],
  ['bug_regression_report', 'The model got worse or broke a previous workflow.'],
  ['integration_failure', 'API, SDK, latency, docs, deployment, or rate-limit issue.'],
  ['benchmark_anecdote', 'A prompt/test result without enough repeatable benchmark structure.'],
  ['hearsay', 'Secondhand, vague, speculative, or no direct use.'],
  ['release_update_reaction', 'Reaction to a release, update, or provider announcement.'],
  ['pricing_value_comment', 'Price, subscription, quota, cost, or value claim.'],
] as const;

const exclusionGuide = [
  ['not_about_model_quality', 'Factual metadata, adoption notes, or comments that do not judge quality.'],
  ['factual_release_or_adoption', 'Release or adoption announcements without a user-quality claim.'],
  ['too_vague_or_speculative', 'No direct signal, only speculation, or no usable evidence.'],
  ['duplicate_or_low_value', 'Repeated, low-information, or redundant content.'],
  ['off_topic', 'Not about a model, product, or usage outcome.'],
  ['other', 'Use for an edge case and explain it in notes.'],
] as const;

export function App() {
  const [rows, setRows] = useState<LabelRow[]>(() => sampleRows.map(withReviewDefaults));
  const [activeIndex, setActiveIndex] = useState(0);
  const [view, setView] = useState<'review' | 'benchmark'>('review');
  const active = rows[activeIndex];

  const stats = useMemo(() => {
    const reviewed = rows.filter(isReviewed).length;
    const changed = rows.filter(hasAnyCorrection).length;
    const excluded = rows.filter((row) => Boolean(row.human_excluded_from_scoring)).length;
    return { reviewed, changed, excluded, remaining: rows.length - reviewed };
  }, [rows]);

  function updateActive(patch: Partial<LabelRow>) {
    setRows((current) =>
      current.map((row, index) => (index === activeIndex ? { ...row, ...patch } : row)),
    );
  }

  function acceptMachineLabel() {
    updateActive({
      reviewed_flag: true,
      human_excluded_from_scoring: false,
      human_exclusion_reason: '',
      human_provider_id: active.provider_id ?? providerFromModel(active.model_id),
      human_model_id: active.model_id,
      human_product_id: active.product_id ?? '',
      human_inference_profile: active.inference_profile ?? '',
      human_task_category: active.task_category,
      human_aspect_category: active.aspect_category,
      human_evidence_type: active.evidence_type,
      human_polarity_score: active.polarity_score,
      human_firsthand_flag: active.firsthand_flag,
    });
    goNext();
  }

  function acceptClassifierSuggestion() {
    updateActive({
      reviewed_flag: true,
      human_excluded_from_scoring: false,
      human_exclusion_reason: '',
      human_provider_id: active.provider_id ?? providerFromModel(active.model_id),
      human_model_id: active.model_id,
      human_product_id: active.product_id ?? '',
      human_inference_profile: active.inference_profile ?? '',
      human_task_category: active.classifier_task_category || active.task_category,
      human_aspect_category: active.classifier_aspect_category || active.aspect_category,
      human_evidence_type: active.classifier_evidence_type || active.evidence_type,
      human_polarity_score:
        active.classifier_polarity_score === undefined || active.classifier_polarity_score === ''
          ? active.polarity_score
          : active.classifier_polarity_score,
      human_firsthand_flag:
        active.classifier_firsthand_flag === undefined || active.classifier_firsthand_flag === ''
          ? active.firsthand_flag
          : active.classifier_firsthand_flag,
    });
    goNext();
  }

  function clearHumanLabel() {
    updateActive({
      ...reviewDefaults(active),
      reviewed_flag: false,
      human_excluded_from_scoring: false,
      human_exclusion_reason: '',
      human_notes: '',
    });
  }

  function excludeRow() {
    updateActive({
      reviewed_flag: true,
      human_excluded_from_scoring: true,
      human_exclusion_reason:
        active.human_exclusion_reason ||
        (active.evidence_type === 'release_update_reaction'
          ? 'factual_release_or_adoption'
          : 'not_about_model_quality'),
      human_provider_id: active.provider_id ?? providerFromModel(active.model_id),
      human_model_id: active.model_id,
      human_product_id: active.product_id ?? '',
      human_inference_profile: active.inference_profile ?? '',
      human_task_category: active.task_category,
      human_aspect_category: active.aspect_category,
      human_evidence_type: active.evidence_type,
      human_polarity_score: 0,
      human_firsthand_flag: false,
    });
    goNext();
  }

  function setExclusionReason(value: string) {
    updateActive({
      human_exclusion_reason: value,
      reviewed_flag: true,
    });
  }

  function goNext() {
    setActiveIndex((index) => Math.min(rows.length - 1, index + 1));
  }

  function goPrevious() {
    setActiveIndex((index) => Math.max(0, index - 1));
  }

  async function handleFile(file: File) {
    const text = await file.text();
    const parsed = parseRows(text);
    if (parsed.length > 0) {
      setRows(parsed.map(withReviewDefaults));
      setActiveIndex(0);
    }
  }

  function exportCsv() {
    downloadFile('label_queue_reviewed.csv', rowsToCsv(rows), 'text/csv;charset=utf-8');
  }

  function exportJson() {
    downloadFile('label_queue_reviewed.json', JSON.stringify(rows, null, 2), 'application/json');
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <ClipboardList size={24} aria-hidden="true" />
          <div>
            <h1>Social Benchmark</h1>
            <p>{view === 'review' ? 'Label review workspace' : 'Benchmark dashboard'}</p>
          </div>
        </div>

        <nav className="view-switch" aria-label="Workspace view">
          <button className={view === 'review' ? 'selected' : ''} onClick={() => setView('review')}>
            <ClipboardList size={16} aria-hidden="true" />
            Review
          </button>
          <button className={view === 'benchmark' ? 'selected' : ''} onClick={() => setView('benchmark')}>
            <BarChart3 size={16} aria-hidden="true" />
            Benchmark
          </button>
        </nav>

        {view === 'review' ? (
          <label className="file-drop">
            <Upload size={18} aria-hidden="true" />
            <span>Load CSV or JSON</span>
            <input
              type="file"
              accept=".csv,.json,.jsonl,text/csv,application/json"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void handleFile(file);
              }}
            />
          </label>
        ) : (
          <section className="sidebar-note">
            <strong>Internal preview</strong>
            <span>Scores are blocked from public display until sample size and source mix improve.</span>
          </section>
        )}

        <section className="stats-grid" aria-label="Review progress">
          <Stat label="Rows" value={rows.length} />
          <Stat label="Reviewed" value={stats.reviewed} />
          <Stat label="Excluded" value={stats.excluded} />
          <Stat label="Changed" value={stats.changed} />
          <Stat label="Remaining" value={stats.remaining} />
        </section>

        {view === 'review' ? (
          <div className="queue-list" aria-label="Rows">
            {rows.map((row, index) => (
              <button
                key={`${row.source_platform}-${row.source_item_id}-${row.model_id}-${index}`}
                className={`queue-item ${index === activeIndex ? 'active' : ''} ${
                  isReviewed(row) ? 'reviewed' : ''
                } ${row.human_excluded_from_scoring ? 'excluded' : ''}`}
                onClick={() => setActiveIndex(index)}
              >
                <span>{index + 1}</span>
                <strong>{row.model_id}</strong>
                <small>
                  {row.human_excluded_from_scoring
                    ? `excluded: ${labelForExclusionReason(row.human_exclusion_reason || 'not_about_model_quality')}`
                    : row.aspect_category}
                </small>
              </button>
            ))}
          </div>
        ) : (
          <ProviderList />
        )}
      </aside>

      {view === 'benchmark' ? (
        <BenchmarkDashboard rows={rows} />
      ) : active ? (
        <ReviewPane
          active={active}
          activeIndex={activeIndex}
          rowCount={rows.length}
          updateActive={updateActive}
          acceptMachineLabel={acceptMachineLabel}
          acceptClassifierSuggestion={acceptClassifierSuggestion}
          clearHumanLabel={clearHumanLabel}
          excludeRow={excludeRow}
          setExclusionReason={setExclusionReason}
          goPrevious={goPrevious}
          goNext={goNext}
          exportCsv={exportCsv}
          exportJson={exportJson}
        />
      ) : (
        <section className="empty-state">No rows loaded.</section>
      )}
    </main>
  );
}

function ReviewPane({
  active,
  activeIndex,
  rowCount,
  updateActive,
  acceptMachineLabel,
  acceptClassifierSuggestion,
  clearHumanLabel,
  excludeRow,
  setExclusionReason,
  goPrevious,
  goNext,
  exportCsv,
  exportJson,
}: {
  active: LabelRow;
  activeIndex: number;
  rowCount: number;
  updateActive: (patch: Partial<LabelRow>) => void;
  acceptMachineLabel: () => void;
  acceptClassifierSuggestion: () => void;
  clearHumanLabel: () => void;
  excludeRow: () => void;
  setExclusionReason: (value: string) => void;
  goPrevious: () => void;
  goNext: () => void;
  exportCsv: () => void;
  exportJson: () => void;
}) {
  return (
    <section className="review-pane">
      <header className="topbar">
        <div>
          <p className="eyebrow">
            Row {activeIndex + 1} of {rowCount}
          </p>
          <h2>{active.model_id}</h2>
        </div>
        <div className="toolbar">
          <IconButton label="Previous" onClick={goPrevious} disabled={activeIndex === 0}>
            <ArrowLeft size={18} />
          </IconButton>
          <IconButton label="Next" onClick={goNext} disabled={activeIndex === rowCount - 1}>
            <ArrowRight size={18} />
          </IconButton>
          <IconButton label="Export CSV" onClick={exportCsv}>
            <Download size={18} />
          </IconButton>
          <IconButton label="Export JSON" onClick={exportJson}>
            <FileText size={18} />
          </IconButton>
        </div>
      </header>

      <div className="review-grid">
        <section className="evidence-panel">
          <div className="source-line">
            <span>{active.source_platform}</span>
            <span>{active.community_id}</span>
            <a href={active.url} target="_blank" rel="noreferrer">
              Source <ExternalLink size={14} aria-hidden="true" />
            </a>
          </div>
          <blockquote>{active.evidence_text}</blockquote>
        </section>

        <section className="machine-panel">
          <h3>Machine Label</h3>
          <Field label="Model" value={active.model_id} />
          <Field label="Provider" value={active.provider_id || '-'} />
          <Field label="Product" value={active.product_id || '-'} />
          <Field label="Profile" value={active.inference_profile || '-'} />
          <Field label="Task" value={active.task_category} />
          <Field label="Aspect" value={active.aspect_category} />
          <Field label="Evidence" value={active.evidence_type} />
          <Field label="Polarity" value={String(active.polarity_score)} />
          <Field label="Firsthand" value={active.firsthand_flag ? 'true' : 'false'} />
          <Field label="Confidence" value={active.extractor_confidence.toFixed(2)} />
        </section>

        <section className="machine-panel">
          <h3>Classifier Suggestion</h3>
          <Field label="Task" value={classifierLabel(active.classifier_task_category, active.classifier_task_confidence)} />
          <Field label="Aspect" value={classifierLabel(active.classifier_aspect_category, active.classifier_aspect_confidence)} />
          <Field label="Evidence" value={classifierLabel(active.classifier_evidence_type, active.classifier_evidence_confidence)} />
          <Field
            label="Polarity"
            value={classifierLabel(
              active.classifier_polarity_score === undefined || active.classifier_polarity_score === ''
                ? ''
                : String(active.classifier_polarity_score),
              active.classifier_polarity_confidence,
            )}
          />
          <Field
            label="Firsthand"
            value={classifierLabel(
              active.classifier_firsthand_flag === undefined || active.classifier_firsthand_flag === ''
                ? ''
                : String(active.classifier_firsthand_flag),
              active.classifier_firsthand_confidence,
            )}
          />
          <Field
            label="Disagreements"
            value={
              active.classifier_disagreement_count === undefined || active.classifier_disagreement_count === ''
                ? '-'
                : String(active.classifier_disagreement_count)
            }
          />
        </section>

        <section className="human-panel">
          <h3>Human Review</h3>
          <SegmentedBoolean
            label="Exclude from scoring"
            value={active.human_excluded_from_scoring ?? ''}
            placeholder={Boolean(active.human_excluded_from_scoring)}
            onChange={(value) => {
              updateActive({ human_excluded_from_scoring: value });
              if (value === true) {
                if (!active.human_exclusion_reason) {
                  updateActive({ human_exclusion_reason: active.evidence_type === 'release_update_reaction' ? 'factual_release_or_adoption' : 'not_about_model_quality' });
                }
              }
              if (value === false) {
                updateActive({ human_exclusion_reason: '' });
              }
            }}
          />
          {active.human_excluded_from_scoring ? (
            <SelectInput
              label="Exclusion reason"
              value={active.human_exclusion_reason ?? ''}
              placeholder={active.human_exclusion_reason || 'select a reason'}
              options={exclusionReasonOptions}
              labels={Object.fromEntries(exclusionReasonOptions.map((item) => [item, labelForExclusionReason(item)]))}
              onChange={(value) => {
                setExclusionReason(value);
                if (value) {
                  goNext();
                }
              }}
            />
          ) : null}
          <TextInput
            label="Provider"
            value={active.human_provider_id ?? ''}
            placeholder={active.provider_id ?? providerFromModel(active.model_id)}
            onChange={(value) => updateActive({ human_provider_id: value })}
          />
          <TextInput
            label="Model"
            value={active.human_model_id ?? ''}
            placeholder={active.model_id}
            onChange={(value) => updateActive({ human_model_id: value })}
          />
          <TextInput
            label="Product"
            value={active.human_product_id ?? ''}
            placeholder={active.product_id ?? ''}
            onChange={(value) => updateActive({ human_product_id: value })}
          />
          <TextInput
            label="Profile"
            value={active.human_inference_profile ?? ''}
            placeholder={active.inference_profile ?? ''}
            onChange={(value) => updateActive({ human_inference_profile: value })}
          />
          <SelectInput
            label="Task"
            value={active.human_task_category ?? ''}
            placeholder={active.task_category}
            options={taskOptions}
            onChange={(value) => updateActive({ human_task_category: value })}
          />
          <SelectInput
            label="Aspect"
            value={active.human_aspect_category ?? ''}
            placeholder={active.aspect_category}
            options={aspectOptions}
            onChange={(value) => updateActive({ human_aspect_category: value })}
          />
          <SelectInput
            label="Evidence"
            value={active.human_evidence_type ?? ''}
            placeholder={active.evidence_type}
            options={evidenceOptions}
            onChange={(value) => updateActive({ human_evidence_type: value })}
          />
          <SelectInput
            label="Polarity"
            value={active.human_polarity_score === '' ? '' : String(active.human_polarity_score ?? '')}
            placeholder={String(active.polarity_score)}
            options={polarityOptions.map((item) => String(item.value))}
            labels={Object.fromEntries(polarityOptions.map((item) => [String(item.value), item.label]))}
            onChange={(value) =>
              updateActive({ human_polarity_score: value === '' ? '' : Number(value) })
            }
          />
          <SegmentedBoolean
            label="Firsthand"
            value={active.human_firsthand_flag ?? ''}
            placeholder={active.firsthand_flag}
            onChange={(value) => updateActive({ human_firsthand_flag: value })}
          />
          <label className="form-field full">
            <span>Notes</span>
            <textarea
              value={active.human_notes ?? ''}
              onChange={(event) => updateActive({ human_notes: event.target.value })}
              rows={3}
            />
          </label>

          <div className="action-row">
            <button className="primary-button" onClick={acceptMachineLabel}>
              <Check size={17} aria-hidden="true" />
              Accept
            </button>
            <button
              className="secondary-button"
              onClick={acceptClassifierSuggestion}
              disabled={!hasClassifierSuggestion(active)}
            >
              <Check size={17} aria-hidden="true" />
              Use Classifier
            </button>
            <button className="secondary-button" onClick={excludeRow}>
              <Check size={17} aria-hidden="true" />
              Exclude & Next
            </button>
            <button className="secondary-button" onClick={clearHumanLabel}>
              <RotateCcw size={17} aria-hidden="true" />
              Reset
            </button>
          </div>
        </section>
      </div>
      <ReviewGuide />
    </section>
  );
}

function BenchmarkDashboard({ rows }: { rows: LabelRow[] }) {
  const rowStats = useMemo(() => {
    const providers = new Set(rows.map((row) => row.provider_id || providerFromModel(row.model_id)).filter(Boolean));
    const products = new Set(rows.map((row) => row.product_id).filter(Boolean));
    const profiles = new Set(rows.map((row) => row.inference_profile).filter(Boolean));
    return { providers: providers.size, products: products.size, profiles: profiles.size };
  }, [rows]);

  const modelCards = useMemo(() => buildModelCards(), []);

  return (
    <section className="benchmark-pane">
      <header className="topbar">
        <div>
          <p className="eyebrow">Internal benchmark preview</p>
          <h2>Provider-Grouped Models</h2>
        </div>
      </header>

      <section className="benchmark-summary">
        <Stat label="Tracked Models" value={benchmarkModels.length} />
        <Stat label="Providers" value={rowStats.providers || groupModelsByProvider().length} />
        <Stat label="Products" value={rowStats.products} />
        <Stat label="Profiles" value={rowStats.profiles} />
      </section>

      <section className="notice-band">
        These scores are internal only. Public ranking is blocked until each model has enough effective sample size,
        healthier source mix, and reviewed labels.
      </section>

      <section className="model-table" aria-label="Model score preview">
        <div className="model-table-head">
          <span>Model</span>
          <span>Provider</span>
          <span>Best Internal Aspect</span>
          <span>Evidence</span>
          <span>Status</span>
        </div>
        {modelCards.map((card) => (
          <div className="model-table-row" key={card.id}>
            <div>
              <strong>{card.displayName}</strong>
              <small>{card.family}</small>
            </div>
            <span>{card.providerName}</span>
            <span>{card.bestAspect}</span>
            <span>{card.evidence}</span>
            <span className={card.publishable ? 'status-good' : 'status-blocked'}>
              {card.publishable ? 'publishable' : 'needs review'}
            </span>
          </div>
        ))}
      </section>
    </section>
  );
}

function ProviderList() {
  const providers = groupModelsByProvider();
  return (
    <div className="provider-list" aria-label="Model providers">
      {providers.map((provider) => (
        <section key={provider.providerId}>
          <h3>{provider.providerName}</h3>
          {provider.models.map((model) => (
            <div className="provider-model" key={model.id}>
              <strong>{model.displayName}</strong>
              <small>{model.family}</small>
            </div>
          ))}
        </section>
      ))}
    </div>
  );
}

function ReviewGuide() {
  return (
    <section className="guide-panel" aria-label="Labeling guide">
      <div className="guide-title">
        <BookOpen size={18} aria-hidden="true" />
        <h3>Review Guide</h3>
      </div>
      <div className="guide-grid">
        <GuideSection title="Identity" rows={identityGuide} />
        <GuideSection title="Firsthand" rows={firsthandGuide} />
        <GuideSection title="Exclusion" rows={exclusionGuide} />
        <GuideSection title="Aspects" rows={aspectGuide} />
        <GuideSection title="Evidence" rows={evidenceGuide} />
        <section className="guide-section">
          <h4>Polarity</h4>
          <p>
            Use <strong>2</strong> for strong praise, <strong>1</strong> for mild praise,{' '}
            <strong>0</strong> for neutral or unclear, <strong>-1</strong> for mild complaint, and{' '}
            <strong>-2</strong> for severe failure or strong complaint.
          </p>
        </section>
      </div>
    </section>
  );
}

function GuideSection({
  title,
  rows,
}: {
  title: string;
  rows: readonly (readonly [string, string])[];
}) {
  return (
    <section className="guide-section">
      <h4>{title}</h4>
      <dl>
        {rows.map(([term, description]) => (
          <div key={term}>
            <dt>{term}</dt>
            <dd>{description}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function labelForExclusionReason(value: string) {
  switch (value) {
    case 'not_about_model_quality':
      return 'Not about model quality';
    case 'factual_release_or_adoption':
      return 'Factual release or adoption';
    case 'too_vague_or_speculative':
      return 'Too vague or speculative';
    case 'duplicate_or_low_value':
      return 'Duplicate or low value';
    case 'off_topic':
      return 'Off topic';
    case 'other':
      return 'Other';
    default:
      return value;
  }
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="stat">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="field-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TextInput({
  label,
  value,
  placeholder,
  onChange,
}: {
  label: string;
  value: string;
  placeholder: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="form-field">
      <span>{label}</span>
      <input value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function SelectInput({
  label,
  value,
  placeholder,
  options,
  labels = {},
  onChange,
}: {
  label: string;
  value: string;
  placeholder: string;
  options: string[];
  labels?: Record<string, string>;
  onChange: (value: string) => void;
}) {
  return (
    <label className="form-field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">Keep: {placeholder}</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {labels[option] ?? option}
          </option>
        ))}
      </select>
    </label>
  );
}

function SegmentedBoolean({
  label,
  value,
  placeholder,
  onChange,
}: {
  label: string;
  value: boolean | '';
  placeholder: boolean;
  onChange: (value: boolean | '') => void;
}) {
  return (
    <div className="form-field">
      <span>{label}</span>
      <div className="segmented" role="group" aria-label={label}>
        <button className={value === '' ? 'selected' : ''} onClick={() => onChange('')}>
          Keep {String(placeholder)}
        </button>
        <button className={value === true ? 'selected' : ''} onClick={() => onChange(true)}>
          True
        </button>
        <button className={value === false ? 'selected' : ''} onClick={() => onChange(false)}>
          False
        </button>
      </div>
    </div>
  );
}

function IconButton({
  label,
  disabled,
  onClick,
  children,
}: {
  label: string;
  disabled?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button className="icon-button" type="button" aria-label={label} title={label} disabled={disabled} onClick={onClick}>
      {children}
    </button>
  );
}

function groupModelsByProvider() {
  const grouped = new Map<string, { providerId: string; providerName: string; models: typeof benchmarkModels }>();
  for (const model of benchmarkModels) {
    if (!grouped.has(model.providerId)) {
      grouped.set(model.providerId, {
        providerId: model.providerId,
        providerName: model.providerName,
        models: [],
      });
    }
    grouped.get(model.providerId)?.models.push(model);
  }
  return [...grouped.values()];
}

function buildModelCards() {
  return benchmarkModels.map((model) => {
    const scores = sampleAspectScores.filter((score) => score.model_id === model.id);
    const best = [...scores].sort((a, b) => b.score - a.score)[0];
    return {
      ...model,
      bestAspect: best ? `${best.aspect_category} ${Math.round(best.score)}` : 'no evidence yet',
      evidence: best ? `n_eff ${best.effective_n.toFixed(1)} / w ${best.weighted_n.toFixed(1)}` : 'waiting for observations',
      publishable: scores.some((score) => score.publishable),
    };
  });
}

function isReviewed(row: LabelRow) {
  return Boolean(row.reviewed_flag || hasAnyCorrection(row) || row.human_notes);
}

function hasAnyCorrection(row: LabelRow) {
  return Boolean(
    (row.human_provider_id && row.human_provider_id !== (row.provider_id ?? providerFromModel(row.model_id))) ||
      (row.human_model_id && row.human_model_id !== row.model_id) ||
      (row.human_product_id && row.human_product_id !== (row.product_id ?? '')) ||
      (row.human_inference_profile && row.human_inference_profile !== (row.inference_profile ?? '')) ||
      (row.human_task_category && row.human_task_category !== row.task_category) ||
      (row.human_aspect_category && row.human_aspect_category !== row.aspect_category) ||
      (row.human_evidence_type && row.human_evidence_type !== row.evidence_type) ||
      (row.human_polarity_score !== undefined &&
        row.human_polarity_score !== '' &&
        row.human_polarity_score !== row.polarity_score) ||
      (row.human_firsthand_flag !== undefined &&
        row.human_firsthand_flag !== '' &&
        row.human_firsthand_flag !== row.firsthand_flag),
  );
}

function parseRows(text: string): LabelRow[] {
  const trimmed = text.trim();
  if (!trimmed) return [];
  if (trimmed.startsWith('[')) return (JSON.parse(trimmed) as LabelRow[]).map(coerceRow);
  if (trimmed.startsWith('{')) return trimmed.split(/\r?\n/).map((line) => coerceRow(JSON.parse(line)));
  return parseCsv(trimmed).map(coerceRow);
}

function parseCsv(text: string) {
  const rows: string[][] = [];
  let current = '';
  let row: string[] = [];
  let inQuotes = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];
    if (char === '"' && next === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      row.push(current);
      current = '';
    } else if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') index += 1;
      row.push(current);
      rows.push(row);
      row = [];
      current = '';
    } else {
      current += char;
    }
  }
  row.push(current);
  rows.push(row);
  const [headers, ...body] = rows.filter((item) => item.some((cell) => cell.length > 0));
  return body.map((cells) =>
    Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ''])),
  );
}

function coerceRow(row: Record<string, unknown>): LabelRow {
  const coerced: LabelRow = {
    review_id: stringValue(row.review_id),
    source_platform: stringValue(row.source_platform),
    community_id: stringValue(row.community_id),
    thread_id: stringValue(row.thread_id),
    source_item_id: stringValue(row.source_item_id),
    url: stringValue(row.url),
    model_id: stringValue(row.model_id),
    provider_id: stringValue(row.provider_id),
    product_id: stringValue(row.product_id),
    inference_profile: stringValue(row.inference_profile),
    task_category: stringValue(row.task_category) as LabelRow['task_category'],
    aspect_category: stringValue(row.aspect_category) as LabelRow['aspect_category'],
    evidence_type: stringValue(row.evidence_type) as LabelRow['evidence_type'],
    claim_type: stringValue(row.claim_type),
    polarity_score: numberValue(row.polarity_score),
    extractor_confidence: numberValue(row.extractor_confidence),
    firsthand_flag: boolValue(row.firsthand_flag),
    regression_flag: boolValue(row.regression_flag),
    hallucination_flag: boolValue(row.hallucination_flag),
    refusal_flag: boolValue(row.refusal_flag),
    value_flag: boolValue(row.value_flag),
    evidence_text: stringValue(row.evidence_text),
    classifier_task_category: stringValue(row.classifier_task_category),
    classifier_task_confidence:
      row.classifier_task_confidence === undefined || row.classifier_task_confidence === ''
        ? ''
        : numberValue(row.classifier_task_confidence),
    classifier_aspect_category: stringValue(row.classifier_aspect_category),
    classifier_aspect_confidence:
      row.classifier_aspect_confidence === undefined || row.classifier_aspect_confidence === ''
        ? ''
        : numberValue(row.classifier_aspect_confidence),
    classifier_evidence_type: stringValue(row.classifier_evidence_type),
    classifier_evidence_confidence:
      row.classifier_evidence_confidence === undefined || row.classifier_evidence_confidence === ''
        ? ''
        : numberValue(row.classifier_evidence_confidence),
    classifier_polarity_score:
      row.classifier_polarity_score === undefined || row.classifier_polarity_score === ''
        ? ''
        : numberValue(row.classifier_polarity_score),
    classifier_polarity_confidence:
      row.classifier_polarity_confidence === undefined || row.classifier_polarity_confidence === ''
        ? ''
        : numberValue(row.classifier_polarity_confidence),
    classifier_firsthand_flag:
      row.classifier_firsthand_flag === undefined || row.classifier_firsthand_flag === ''
        ? ''
        : boolValue(row.classifier_firsthand_flag),
    classifier_firsthand_confidence:
      row.classifier_firsthand_confidence === undefined || row.classifier_firsthand_confidence === ''
        ? ''
        : numberValue(row.classifier_firsthand_confidence),
    classifier_disagreement_count:
      row.classifier_disagreement_count === undefined || row.classifier_disagreement_count === ''
        ? ''
        : numberValue(row.classifier_disagreement_count),
    reviewed_flag: boolValue(row.reviewed_flag),
    human_excluded_from_scoring:
      row.human_excluded_from_scoring === undefined || row.human_excluded_from_scoring === ''
        ? ''
        : boolValue(row.human_excluded_from_scoring),
    human_exclusion_reason: stringValue(row.human_exclusion_reason),
    human_provider_id: stringValue(row.human_provider_id),
    human_model_id: stringValue(row.human_model_id),
    human_product_id: stringValue(row.human_product_id),
    human_inference_profile: stringValue(row.human_inference_profile),
    human_task_category: stringValue(row.human_task_category),
    human_aspect_category: stringValue(row.human_aspect_category),
    human_evidence_type: stringValue(row.human_evidence_type),
    human_polarity_score:
      row.human_polarity_score === undefined || row.human_polarity_score === ''
        ? ''
        : numberValue(row.human_polarity_score),
    human_firsthand_flag:
      row.human_firsthand_flag === undefined || row.human_firsthand_flag === ''
        ? ''
        : boolValue(row.human_firsthand_flag),
    human_notes: stringValue(row.human_notes),
  };
  return withReviewDefaults(coerced);
}

function rowsToCsv(rows: LabelRow[]) {
  const headers = [
    'review_id',
    'source_platform',
    'community_id',
    'thread_id',
    'source_item_id',
    'url',
    'model_id',
    'provider_id',
    'product_id',
    'inference_profile',
    'task_category',
    'aspect_category',
    'evidence_type',
    'claim_type',
    'polarity_score',
    'extractor_confidence',
    'firsthand_flag',
    'regression_flag',
    'hallucination_flag',
    'refusal_flag',
    'value_flag',
    'evidence_text',
    'classifier_task_category',
    'classifier_task_confidence',
    'classifier_aspect_category',
    'classifier_aspect_confidence',
    'classifier_evidence_type',
    'classifier_evidence_confidence',
    'classifier_polarity_score',
    'classifier_polarity_confidence',
    'classifier_firsthand_flag',
    'classifier_firsthand_confidence',
    'classifier_disagreement_count',
    'reviewed_flag',
    'human_excluded_from_scoring',
    'human_exclusion_reason',
    'human_provider_id',
    'human_model_id',
    'human_product_id',
    'human_inference_profile',
    'human_task_category',
    'human_aspect_category',
    'human_evidence_type',
    'human_polarity_score',
    'human_firsthand_flag',
    'human_notes',
  ];
  return [headers.join(','), ...rows.map((row) => headers.map((header) => csvEscape(row[header as keyof LabelRow])).join(','))].join(
    '\n',
  );
}

function csvEscape(value: unknown) {
  const text = value === undefined ? '' : String(value);
  if (/[",\n\r]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

function stringValue(value: unknown) {
  return value === undefined || value === null ? '' : String(value);
}

function numberValue(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function boolValue(value: unknown) {
  if (typeof value === 'boolean') return value;
  return String(value).toLowerCase() === 'true';
}

function withReviewDefaults(row: LabelRow): LabelRow {
  return {
    ...row,
    ...reviewDefaults(row),
  };
}

function hasClassifierSuggestion(row: LabelRow) {
  return Boolean(
    row.classifier_task_category ||
      row.classifier_aspect_category ||
      row.classifier_evidence_type ||
      (row.classifier_polarity_score !== undefined && row.classifier_polarity_score !== '') ||
      (row.classifier_firsthand_flag !== undefined && row.classifier_firsthand_flag !== ''),
  );
}

function classifierLabel(value: string | number | boolean | '' | undefined, confidence: number | '' | undefined) {
  if (value === undefined || value === '') return '-';
  if (confidence === undefined || confidence === '') return String(value);
  return `${String(value)} (${confidence.toFixed(2)})`;
}

function reviewDefaults(row: LabelRow): Partial<LabelRow> {
  return {
    human_provider_id: row.human_provider_id || row.provider_id || providerFromModel(row.model_id),
    human_model_id: row.human_model_id || row.model_id,
    human_product_id: row.human_product_id || row.product_id || '',
    human_inference_profile: row.human_inference_profile || row.inference_profile || '',
    human_excluded_from_scoring:
      row.human_excluded_from_scoring === undefined || row.human_excluded_from_scoring === ''
        ? false
        : row.human_excluded_from_scoring,
    human_exclusion_reason: row.human_exclusion_reason || '',
    human_task_category: row.human_task_category || row.task_category,
    human_aspect_category: row.human_aspect_category || row.aspect_category,
    human_evidence_type: row.human_evidence_type || row.evidence_type,
    human_polarity_score:
      row.human_polarity_score === undefined || row.human_polarity_score === ''
        ? row.polarity_score
        : row.human_polarity_score,
    human_firsthand_flag:
      row.human_firsthand_flag === undefined || row.human_firsthand_flag === ''
        ? row.firsthand_flag
        : row.human_firsthand_flag,
  };
}

function providerFromModel(modelId: string) {
  if (modelId.startsWith('claude-')) return 'anthropic';
  if (modelId.startsWith('gpt-') || modelId.startsWith('o3') || modelId.startsWith('o4-')) return 'openai';
  if (modelId.startsWith('gemini-')) return 'google';
  if (modelId.startsWith('llama-')) return 'meta';
  if (modelId.startsWith('mistral-')) return 'mistral';
  if (modelId.startsWith('deepseek-')) return 'deepseek';
  return '';
}

function downloadFile(filename: string, body: string, type: string) {
  const blob = new Blob([body], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
