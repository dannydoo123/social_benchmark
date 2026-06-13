-- Score snapshot tables for the public dashboard.
-- Every displayed number traces to a snapshot row; snapshots are immutable.

create table if not exists score_snapshots (
  id text primary key, -- e.g. 20260612T120000Z
  generated_at timestamptz not null,
  training_examples integer not null,
  human_reviewed_included integer not null,
  machine_included integer not null,
  machine_rejected_by_gates integer not null,
  gates jsonb not null,
  methodology jsonb not null,
  created_at timestamptz not null default now()
);

create table if not exists snapshot_aspect_scores (
  id bigserial primary key,
  snapshot_id text not null references score_snapshots(id) on delete cascade,
  model_id text not null,
  provider_id text not null default '',
  task_category text, -- null = all tasks
  aspect_category text not null,
  score numeric(6, 2) not null,
  confidence_low numeric(6, 2),
  confidence_high numeric(6, 2),
  effective_n numeric(10, 1) not null,
  weighted_n numeric(10, 1) not null,
  n_observations integer not null default 0,
  n_threads integer not null default 0,
  n_authors integer not null default 0,
  firsthand_ratio numeric(5, 3) not null default 0,
  human_share numeric(5, 3) not null default 0,
  warnings jsonb not null default '[]'::jsonb,
  publishable boolean not null default false,
  trust_tier text not null default 'insufficient', -- insufficient | provisional | ranked
  unique (snapshot_id, model_id, task_category, aspect_category)
);

create index if not exists snapshot_aspect_scores_lookup
  on snapshot_aspect_scores (snapshot_id, aspect_category, task_category);

create table if not exists snapshot_overall_scores (
  id bigserial primary key,
  snapshot_id text not null references score_snapshots(id) on delete cascade,
  model_id text not null,
  score numeric(6, 2) not null,
  unique (snapshot_id, model_id)
);

create table if not exists snapshot_evidence_samples (
  id bigserial primary key,
  snapshot_id text not null references score_snapshots(id) on delete cascade,
  model_id text not null,
  aspect_category text not null,
  span text not null, -- truncated quote; full text stays at the source
  url text not null,
  polarity smallint not null,
  evidence_type text not null,
  firsthand boolean not null,
  human_labeled boolean not null
);

create index if not exists snapshot_evidence_lookup
  on snapshot_evidence_samples (snapshot_id, model_id, aspect_category);

-- The dashboard reads only the latest snapshot by default.
create or replace view latest_snapshot as
  select * from score_snapshots order by generated_at desc limit 1;
