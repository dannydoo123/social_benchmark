export type BenchmarkAspectScore = {
  model_id: string;
  aspect_category: string;
  score: number;
  effective_n: number;
  weighted_n: number;
  publishable: boolean;
  publication_blockers: string[];
  source_mix: Record<string, number>;
};

export type BenchmarkModel = {
  id: string;
  displayName: string;
  providerId: string;
  providerName: string;
  family: string;
};

export const benchmarkModels: BenchmarkModel[] = [
  { id: 'claude-opus-4.8', displayName: 'Claude Opus 4.8', providerId: 'anthropic', providerName: 'Anthropic', family: 'claude-opus' },
  { id: 'claude-opus-4.5', displayName: 'Claude Opus 4.5', providerId: 'anthropic', providerName: 'Anthropic', family: 'claude-opus' },
  { id: 'claude-sonnet-4.6', displayName: 'Claude Sonnet 4.6', providerId: 'anthropic', providerName: 'Anthropic', family: 'claude-sonnet' },
  { id: 'claude-sonnet-4.5', displayName: 'Claude Sonnet 4.5', providerId: 'anthropic', providerName: 'Anthropic', family: 'claude-sonnet' },
  { id: 'claude-haiku-4.5', displayName: 'Claude Haiku 4.5', providerId: 'anthropic', providerName: 'Anthropic', family: 'claude-haiku' },
  { id: 'gpt-5.5', displayName: 'GPT-5.5', providerId: 'openai', providerName: 'OpenAI', family: 'gpt-5' },
  { id: 'gpt-5.4', displayName: 'GPT-5.4', providerId: 'openai', providerName: 'OpenAI', family: 'gpt-5' },
  { id: 'gpt-5.4-mini', displayName: 'GPT-5.4 mini', providerId: 'openai', providerName: 'OpenAI', family: 'gpt-5' },
  { id: 'gpt-5', displayName: 'GPT-5', providerId: 'openai', providerName: 'OpenAI', family: 'gpt-5' },
  { id: 'gpt-4.1', displayName: 'GPT-4.1', providerId: 'openai', providerName: 'OpenAI', family: 'gpt-4' },
  { id: 'gpt-4o', displayName: 'GPT-4o', providerId: 'openai', providerName: 'OpenAI', family: 'gpt-4' },
  { id: 'o4-mini', displayName: 'o4-mini', providerId: 'openai', providerName: 'OpenAI', family: 'reasoning' },
  { id: 'gemini-3.5-flash', displayName: 'Gemini 3.5 Flash', providerId: 'google', providerName: 'Google', family: 'gemini' },
  { id: 'gemini-3.1-pro', displayName: 'Gemini 3.1 Pro', providerId: 'google', providerName: 'Google', family: 'gemini' },
  { id: 'gemini-3-flash', displayName: 'Gemini 3 Flash', providerId: 'google', providerName: 'Google', family: 'gemini' },
  { id: 'gemini-2.5', displayName: 'Gemini 2.5', providerId: 'google', providerName: 'Google', family: 'gemini' },
];

export const sampleAspectScores: BenchmarkAspectScore[] = [
  {
    model_id: 'gpt-5',
    aspect_category: 'satisfaction',
    score: 44.0,
    effective_n: 6.0,
    weighted_n: 7.9,
    publishable: false,
    publication_blockers: ['effective_sample_size_below_30', 'thread_overconcentrated'],
    source_mix: { github: 0.59, hacker_news: 0.41 },
  },
  {
    model_id: 'gpt-5',
    aspect_category: 'developer_ergonomics',
    score: 41.7,
    effective_n: 6.0,
    weighted_n: 5.4,
    publishable: false,
    publication_blockers: ['effective_sample_size_below_30'],
    source_mix: { github: 0.5, hacker_news: 0.5 },
  },
  {
    model_id: 'claude-sonnet-4.5',
    aspect_category: 'satisfaction',
    score: 48.1,
    effective_n: 10.4,
    weighted_n: 13.5,
    publishable: false,
    publication_blockers: ['effective_sample_size_below_30', 'thread_overconcentrated'],
    source_mix: { github: 0.45, hacker_news: 0.55 },
  },
  {
    model_id: 'claude-sonnet-4.5',
    aspect_category: 'developer_ergonomics',
    score: 50.0,
    effective_n: 4.0,
    weighted_n: 3.8,
    publishable: false,
    publication_blockers: ['effective_sample_size_below_30', 'platform_overconcentrated'],
    source_mix: { github: 1.0 },
  },
  {
    model_id: 'gemini-3.5-flash',
    aspect_category: 'satisfaction',
    score: 45.0,
    effective_n: 3.5,
    weighted_n: 5.0,
    publishable: false,
    publication_blockers: ['effective_sample_size_below_30', 'thread_overconcentrated'],
    source_mix: { github: 0.45, hacker_news: 0.55 },
  },
];
