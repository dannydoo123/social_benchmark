-- Score snapshot tables for the public dashboard.
-- Every displayed number traces to one immutable snapshot row.

-- Migration 001 originally created pipeline-oriented `score_snapshots` and
-- `observations` tables. The dashboard uses snapshot-wide rows instead, so
-- replace those legacy shapes when this migration is applied to a fresh DB.
do $$
begin
  if exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'score_snapshots'
      and column_name = 'model_id'
  ) or (
    exists (
      select 1
      from information_schema.tables
      where table_schema = 'public'
        and table_name = 'score_snapshots'
    )
    and not exists (
      select 1
      from information_schema.columns
      where table_schema = 'public'
        and table_name = 'score_snapshots'
        and column_name = 'snapshot_id'
    )
  ) then
    drop table if exists score_snapshots cascade;
  end if;

  if exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'observations'
      and column_name = 'source_item_id'
      and data_type = 'bigint'
  ) then
    drop table if exists observations cascade;
  end if;
end $$;

drop table if exists snapshot_evidence_samples cascade;
drop table if exists snapshot_overall_scores cascade;
drop table if exists snapshot_aspect_scores cascade;

create table if not exists score_snapshots (
  id bigserial primary key,
  snapshot_id text not null unique, -- e.g. 20260612T120000Z
  generated_at timestamptz not null,
  is_current boolean not null default false,
  gates jsonb not null default '{}'::jsonb,
  tier_thresholds jsonb not null default '{}'::jsonb,
  corpus jsonb not null default '{}'::jsonb,
  overall jsonb not null default '[]'::jsonb,
  providers jsonb not null default '[]'::jsonb,
  coverage jsonb not null default '{}'::jsonb,
  methodology jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

alter table score_snapshots add column if not exists snapshot_id text;
alter table score_snapshots add column if not exists is_current boolean not null default false;
alter table score_snapshots add column if not exists tier_thresholds jsonb not null default '{}'::jsonb;
alter table score_snapshots add column if not exists corpus jsonb not null default '{}'::jsonb;
alter table score_snapshots add column if not exists overall jsonb not null default '[]'::jsonb;
alter table score_snapshots add column if not exists providers jsonb not null default '[]'::jsonb;
alter table score_snapshots add column if not exists coverage jsonb not null default '{}'::jsonb;

create unique index if not exists score_snapshots_snapshot_id_key
  on score_snapshots(snapshot_id);

create unique index if not exists score_snapshots_one_current
  on score_snapshots(is_current)
  where is_current;

create table if not exists leaderboard_rows (
  id bigserial primary key,
  snapshot_id text not null references score_snapshots(snapshot_id) on delete cascade,
  model_id text not null,
  provider_id text not null default '',
  aspect text not null,
  score numeric(6, 2) not null,
  ci_low numeric(6, 2),
  ci_high numeric(6, 2),
  ess numeric(10, 1) not null default 0,
  weighted_n numeric(10, 1) not null default 0,
  n_observations integer not null default 0,
  n_threads integer not null default 0,
  n_authors integer not null default 0,
  firsthand_ratio numeric(5, 3) not null default 0,
  human_share numeric(5, 3) not null default 0,
  warnings jsonb not null default '[]'::jsonb,
  publishable boolean not null default false,
  tier text not null default 'insufficient',
  unique (snapshot_id, model_id, aspect)
);

create index if not exists leaderboard_rows_lookup
  on leaderboard_rows(snapshot_id, aspect, model_id);

create table if not exists task_rows (
  id bigserial primary key,
  snapshot_id text not null references score_snapshots(snapshot_id) on delete cascade,
  task text not null,
  model_id text not null,
  aspect text not null,
  score numeric(6, 2) not null,
  ci_low numeric(6, 2),
  ci_high numeric(6, 2),
  ess numeric(10, 1) not null default 0,
  tier text not null default 'insufficient',
  unique (snapshot_id, task, model_id, aspect)
);

create index if not exists task_rows_lookup
  on task_rows(snapshot_id, task, aspect);

create table if not exists evidence_samples (
  id bigserial primary key,
  snapshot_id text not null references score_snapshots(snapshot_id) on delete cascade,
  model_id text not null,
  aspect text not null,
  ord integer not null default 0,
  span text not null,
  url text not null,
  polarity smallint not null,
  evidence_type text not null,
  firsthand boolean not null,
  human_labeled boolean not null
);

create index if not exists evidence_samples_lookup
  on evidence_samples(snapshot_id, model_id, aspect, ord);

create table if not exists observations (
  id bigserial primary key,
  snapshot_id text not null references score_snapshots(snapshot_id) on delete cascade,
  source_platform text not null,
  source_item_id text not null,
  thread_id text not null default '',
  url text not null default '',
  model_id text not null,
  provider_id text not null default '',
  product_id text not null default '',
  inference_profile text not null default '',
  task_category text not null,
  aspect_category text not null,
  evidence_type text not null,
  polarity_score smallint not null,
  firsthand_flag boolean not null default false,
  evidence_text text not null default ''
);

create index if not exists observations_snapshot_lookup
  on observations(snapshot_id, source_platform, model_id, aspect_category);

-- The dashboard reads only the current snapshot by default.
create or replace view latest_snapshot as
  select *
  from score_snapshots
  where is_current
  order by generated_at desc
  limit 1;
