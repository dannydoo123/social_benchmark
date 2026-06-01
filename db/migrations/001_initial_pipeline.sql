-- Initial Social Benchmark data pipeline schema.
-- Designed for PostgreSQL/Supabase.

create table if not exists source_platforms (
  id text primary key,
  display_name text not null,
  source_quality_weight numeric(6, 3) not null default 1.000,
  created_at timestamptz not null default now()
);

create table if not exists communities (
  id bigserial primary key,
  platform_id text not null references source_platforms(id),
  external_id text not null,
  display_name text not null,
  url text,
  source_quality_weight numeric(6, 3) not null default 1.000,
  created_at timestamptz not null default now(),
  unique (platform_id, external_id)
);

create table if not exists providers (
  id text primary key,
  display_name text not null,
  created_at timestamptz not null default now()
);

create table if not exists models (
  id text primary key,
  provider_id text not null references providers(id),
  display_name text not null,
  family text,
  first_seen_at timestamptz,
  created_at timestamptz not null default now()
);

create table if not exists model_aliases (
  id bigserial primary key,
  model_id text not null references models(id),
  alias text not null,
  normalized_alias text not null,
  created_at timestamptz not null default now(),
  unique (normalized_alias)
);

create table if not exists products (
  id text primary key,
  provider_id text references providers(id),
  display_name text not null,
  product_type text not null default 'interface',
  created_at timestamptz not null default now()
);

create table if not exists source_items (
  id bigserial primary key,
  platform_id text not null references source_platforms(id),
  community_id bigint references communities(id),
  external_id text not null,
  thread_external_id text,
  parent_external_id text,
  author_id_hash text,
  author_handle text,
  title text not null default '',
  body text not null default '',
  url text not null default '',
  published_at timestamptz,
  engagement jsonb not null default '{}'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  content_hash text,
  fetched_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  unique (platform_id, external_id)
);

create index if not exists idx_source_items_platform_published
  on source_items(platform_id, published_at desc);

create index if not exists idx_source_items_thread
  on source_items(platform_id, thread_external_id);

create index if not exists idx_source_items_content_hash
  on source_items(content_hash);

create table if not exists duplicate_clusters (
  id bigserial primary key,
  cluster_key text not null unique,
  method text not null,
  representative_source_item_id bigint references source_items(id),
  created_at timestamptz not null default now()
);

create table if not exists extraction_runs (
  id bigserial primary key,
  extractor_model_name text not null,
  extractor_model_version text not null,
  config jsonb not null default '{}'::jsonb,
  started_at timestamptz not null default now(),
  completed_at timestamptz,
  created_at timestamptz not null default now()
);

create table if not exists candidate_features (
  id bigserial primary key,
  source_item_id bigint not null references source_items(id) on delete cascade,
  extraction_run_id bigint references extraction_runs(id),
  relevant boolean not null default false,
  model_mentions jsonb not null default '[]'::jsonb,
  product_id text references products(id),
  inference_profile text,
  evidence_text text not null default '',
  task_categories text[] not null default '{}',
  aspect_categories text[] not null default '{}',
  evidence_types text[] not null default '{}',
  polarity_score smallint not null default 0 check (polarity_score between -2 and 2),
  severity_score numeric(6, 4) not null default 0,
  extractor_confidence numeric(6, 4) not null default 0,
  flags jsonb not null default '{}'::jsonb,
  matched_terms text[] not null default '{}',
  created_at timestamptz not null default now()
);

create index if not exists idx_candidate_features_source_item
  on candidate_features(source_item_id);

create table if not exists observations (
  id bigserial primary key,
  source_item_id bigint not null references source_items(id) on delete cascade,
  extraction_run_id bigint references extraction_runs(id),
  duplicate_cluster_id bigint references duplicate_clusters(id),
  evidence_text text not null default '',
  model_id text not null references models(id),
  provider_id text not null references providers(id),
  model_version_or_alias text not null,
  product_id text references products(id),
  inference_profile text,
  task_category text not null,
  aspect_category text not null,
  evidence_type text not null,
  claim_type text not null,
  polarity_score smallint not null check (polarity_score between -2 and 2),
  severity_score numeric(6, 4) not null,
  extractor_confidence numeric(6, 4) not null,
  firsthand_flag boolean not null default false,
  comparative_flag boolean not null default false,
  regression_flag boolean not null default false,
  hallucination_flag boolean not null default false,
  refusal_flag boolean not null default false,
  value_flag boolean not null default false,
  source_quality_weight numeric(8, 4) not null default 1.0000,
  firsthand_weight numeric(8, 4) not null default 1.0000,
  author_credibility_weight numeric(8, 4) not null default 1.0000,
  corroboration_weight numeric(8, 4) not null default 1.0000,
  recency_weight numeric(8, 4) not null default 1.0000,
  engagement_weight numeric(8, 4) not null default 1.0000,
  duplicate_penalty numeric(8, 4) not null default 1.0000,
  final_weight numeric(10, 5) generated always as (
    source_quality_weight
    * firsthand_weight
    * author_credibility_weight
    * corroboration_weight
    * recency_weight
    * engagement_weight
    * duplicate_penalty
  ) stored,
  human_labeled_flag boolean not null default false,
  created_at timestamptz not null default now()
);

create index if not exists idx_observations_model_aspect
  on observations(model_id, aspect_category);

create index if not exists idx_observations_task
  on observations(task_category);

create index if not exists idx_observations_flags
  on observations(regression_flag, hallucination_flag, refusal_flag, value_flag);

create table if not exists score_snapshots (
  id bigserial primary key,
  model_id text not null references models(id),
  aspect_category text not null,
  window_start timestamptz,
  window_end timestamptz,
  score numeric(7, 3) not null,
  weighted_n numeric(12, 4) not null,
  capped_weighted_n numeric(12, 4),
  effective_n numeric(12, 4) not null,
  confidence_low numeric(7, 3),
  confidence_high numeric(7, 3),
  warnings text[] not null default '{}',
  publishable boolean not null default false,
  publication_blockers text[] not null default '{}',
  source_mix jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_score_snapshots_model_window
  on score_snapshots(model_id, window_end desc);

create table if not exists release_updates (
  id bigserial primary key,
  provider_id text not null references providers(id),
  model_id text references models(id),
  release_or_update_name text not null,
  announced_at timestamptz,
  effective_at timestamptz,
  source_url text not null,
  capability_claims text[] not null default '{}',
  pricing_or_limit_changes text[] not null default '{}',
  deprecation_or_routing_changes text[] not null default '{}',
  expected_affected_categories text[] not null default '{}',
  notes text not null default '',
  created_at timestamptz not null default now()
);

insert into source_platforms (id, display_name, source_quality_weight)
values
  ('hacker_news', 'Hacker News', 1.150),
  ('github', 'GitHub', 1.250),
  ('stack_exchange', 'Stack Exchange', 1.200),
  ('reddit', 'Reddit', 1.000),
  ('hugging_face', 'Hugging Face', 1.150)
on conflict (id) do nothing;

insert into providers (id, display_name)
values
  ('openai', 'OpenAI'),
  ('anthropic', 'Anthropic'),
  ('google', 'Google'),
  ('meta', 'Meta'),
  ('mistral', 'Mistral'),
  ('deepseek', 'DeepSeek'),
  ('xai', 'xAI'),
  ('alibaba', 'Alibaba'),
  ('moonshot', 'Moonshot AI'),
  ('cohere', 'Cohere'),
  ('github', 'GitHub'),
  ('cursor', 'Cursor')
on conflict (id) do nothing;

insert into products (id, provider_id, display_name, product_type)
values
  ('claude-code', 'anthropic', 'Claude Code', 'coding_agent'),
  ('chatgpt', 'openai', 'ChatGPT', 'chat_app'),
  ('openai-api', 'openai', 'OpenAI API', 'api'),
  ('anthropic-api', 'anthropic', 'Anthropic API', 'api'),
  ('gemini-api', 'google', 'Gemini API', 'api'),
  ('google-ai-studio', 'google', 'Google AI Studio', 'developer_tool'),
  ('github-copilot', 'github', 'GitHub Copilot', 'coding_assistant'),
  ('cursor', 'cursor', 'Cursor', 'ide')
on conflict (id) do nothing;
