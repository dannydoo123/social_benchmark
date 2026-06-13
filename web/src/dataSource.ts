import { getSupabase } from './supabaseClient';

export type Tier = 'insufficient' | 'provisional' | 'ranked';

export type LeaderboardRow = {
  model_id: string;
  provider_id: string;
  aspect: string;
  score: number;
  ci: [number, number];
  ess: number;
  weighted_n: number;
  n_observations: number;
  n_threads: number;
  n_authors: number;
  firsthand_ratio: number;
  human_share: number;
  warnings: string[];
  publishable: boolean;
  tier: Tier;
};

export type TaskRow = {
  model_id: string;
  aspect: string;
  score: number;
  ci: [number, number];
  ess: number;
  tier: Tier;
};

export type EvidenceSample = {
  span: string;
  url: string;
  polarity: number;
  evidence_type: string;
  firsthand: boolean;
  human_labeled: boolean;
};

export type OverallRow = {
  model_id: string;
  provider_id?: string;
  score: number;
  ess: number;
  n_observations: number;
  n_threads: number;
  tier: Tier;
};

export type ProviderAspect = {
  aspect: string;
  score: number;
  ci: [number, number];
  ess: number;
  n_observations: number;
  tier: Tier;
};

export type ProviderRow = {
  provider_id: string;
  score: number;
  ess: number;
  n_observations: number;
  n_threads: number;
  n_models: number;
  unspecified_observations: number;
  tier: Tier;
  aspects: ProviderAspect[];
};

export type Coverage = {
  by_status: Record<string, number>;
  unspecified_by_provider: Record<string, number>;
  unregistered_versioned: Record<string, number>;
};

export type Snapshot = {
  snapshot_id: string;
  generated_at: string;
  gates: Record<string, number>;
  tier_thresholds?: { ranked_ess: number; provisional_ess: number };
  corpus: {
    human_reviewed_included: number;
    machine_included: number;
    machine_rejected_by_gates: number;
    excluded_by_review: number;
    training_examples: number;
    models: number;
    providers?: number;
    unspecified_observations?: number;
    threads: number;
  };
  overall: OverallRow[];
  providers: ProviderRow[];
  coverage?: Coverage;
  leaderboard: LeaderboardRow[];
  tasks: Record<string, TaskRow[]>;
  evidence: Record<string, EvidenceSample[]>;
  methodology: {
    gates: Record<string, number>;
    training_examples: number;
    gated_precision_artifacts: Record<
      string,
      Record<string, { evaluated?: number; thresholds: Record<string, { precision: number; coverage: number }> }>
    >;
    pipeline: string[];
  };
};

export type ObservationRow = {
  source_platform: string;
  source_item_id: string;
  thread_id: string;
  url: string;
  model_id: string;
  provider_id: string;
  product_id: string;
  inference_profile: string;
  task_category: string;
  aspect_category: string;
  evidence_type: string;
  polarity_score: number;
  firsthand_flag: boolean;
  evidence_text: string;
};

export type SnapshotResult = { snapshot: Snapshot; source: 'supabase' | 'static' };
export type ObservationResult = { rows: ObservationRow[]; source: 'supabase' | 'static' };

let staticSnapshotPromise: Promise<Snapshot> | null = null;

function loadStaticSnapshot(): Promise<Snapshot> {
  if (!staticSnapshotPromise) {
    staticSnapshotPromise = fetch('/snapshot.json').then((response) => {
      if (!response.ok) throw new Error(`snapshot.json ${response.status}`);
      return response.json() as Promise<Snapshot>;
    });
  }
  return staticSnapshotPromise;
}

/**
 * Load the current snapshot. Prefers Supabase (live store); if the project is
 * unreachable or has no current snapshot, falls back to the prebuilt static file.
 */
export async function getSnapshot(): Promise<SnapshotResult> {
  const supabase = getSupabase();
  if (supabase) {
    try {
      const reconstructed = await snapshotFromSupabase(supabase);
      if (reconstructed) return { snapshot: reconstructed, source: 'supabase' };
    } catch (error) {
      console.warn('Supabase snapshot unavailable, using static fallback:', error);
    }
  }
  return { snapshot: await loadStaticSnapshot(), source: 'static' };
}

async function snapshotFromSupabase(
  supabase: NonNullable<ReturnType<typeof getSupabase>>,
): Promise<Snapshot | null> {
  const { data: meta, error: metaError } = await supabase
    .from('score_snapshots')
    .select(
      'snapshot_id, generated_at, gates, tier_thresholds, corpus, overall, providers, coverage, methodology',
    )
    .eq('is_current', true)
    .limit(1)
    .maybeSingle();
  if (metaError) throw metaError;
  if (!meta) return null;

  const snapshotId = meta.snapshot_id as string;
  const [leaderboard, taskRows, evidenceRows] = await Promise.all([
    supabase
      .from('leaderboard_rows')
      .select('*')
      .eq('snapshot_id', snapshotId)
      .then(unwrap),
    supabase.from('task_rows').select('*').eq('snapshot_id', snapshotId).then(unwrap),
    supabase.from('evidence_samples').select('*').eq('snapshot_id', snapshotId).order('ord').then(unwrap),
  ]);

  const tasks: Record<string, TaskRow[]> = {};
  for (const row of taskRows as any[]) {
    (tasks[row.task] ??= []).push({
      model_id: row.model_id,
      aspect: row.aspect,
      score: row.score,
      ci: [row.ci_low ?? 0, row.ci_high ?? 0],
      ess: row.ess ?? 0,
      tier: row.tier,
    });
  }

  let evidence: Record<string, EvidenceSample[]> = {};
  for (const row of evidenceRows as any[]) {
    (evidence[`${row.model_id}|${row.aspect}`] ??= []).push({
      span: row.span,
      url: row.url,
      polarity: row.polarity,
      evidence_type: row.evidence_type,
      firsthand: row.firsthand,
      human_labeled: row.human_labeled,
    });
  }
  // Evidence receipts may be loaded only by the service-role loader; fall back to
  // the static snapshot so provenance still renders for this snapshot.
  if (Object.keys(evidence).length === 0) {
    try {
      evidence = (await loadStaticSnapshot()).evidence ?? {};
    } catch {
      evidence = {};
    }
  }

  return {
    snapshot_id: snapshotId,
    generated_at: meta.generated_at as string,
    gates: (meta.gates as Record<string, number>) ?? {},
    tier_thresholds: meta.tier_thresholds as Snapshot['tier_thresholds'],
    corpus: meta.corpus as Snapshot['corpus'],
    overall: (meta.overall as Snapshot['overall']) ?? [],
    providers: (meta.providers as Snapshot['providers']) ?? [],
    coverage: meta.coverage as Snapshot['coverage'],
    leaderboard: (leaderboard as any[]).map((row) => ({
      model_id: row.model_id,
      provider_id: row.provider_id ?? '',
      aspect: row.aspect,
      score: row.score,
      ci: [row.ci_low ?? 0, row.ci_high ?? 0],
      ess: row.ess ?? 0,
      weighted_n: row.weighted_n ?? 0,
      n_observations: row.n_observations ?? 0,
      n_threads: row.n_threads ?? 0,
      n_authors: row.n_authors ?? 0,
      firsthand_ratio: row.firsthand_ratio ?? 0,
      human_share: row.human_share ?? 0,
      warnings: row.warnings ?? [],
      publishable: row.publishable ?? false,
      tier: row.tier,
    })),
    tasks,
    evidence,
    methodology: meta.methodology as Snapshot['methodology'],
  };
}

let staticObservationsPromise: Promise<ObservationRow[]> | null = null;

function loadStaticObservations(): Promise<ObservationRow[]> {
  if (!staticObservationsPromise) {
    staticObservationsPromise = fetch('/observations.json')
      .then((response) => (response.ok ? (response.json() as Promise<ObservationRow[]>) : []))
      .catch(() => []);
  }
  return staticObservationsPromise;
}

/** Load reviewed observations for the evidence explorer (Supabase, else static). */
export async function getObservations(): Promise<ObservationResult> {
  const supabase = getSupabase();
  if (supabase) {
    try {
      const { data, error } = await supabase
        .from('observations')
        .select(
          'source_platform, source_item_id, thread_id, url, model_id, provider_id, product_id, inference_profile, task_category, aspect_category, evidence_type, polarity_score, firsthand_flag, evidence_text',
        )
        .limit(5000);
      if (error) throw error;
      if (data && data.length > 0) return { rows: data as ObservationRow[], source: 'supabase' };
    } catch (error) {
      console.warn('Supabase observations unavailable, using static fallback:', error);
    }
  }
  return { rows: await loadStaticObservations(), source: 'static' };
}

function unwrap<T>({ data, error }: { data: T | null; error: unknown }): T {
  if (error) throw error;
  return (data ?? ([] as unknown)) as T;
}
