export type TaskCategory =
  | 'coding'
  | 'writing'
  | 'research'
  | 'agents'
  | 'roleplay'
  | 'data_analysis'
  | 'long_context'
  | 'multimodal'
  | 'api_developer_workflow'
  | 'general';

export type AspectCategory =
  | 'satisfaction'
  | 'trust_reliability'
  | 'task_fit'
  | 'regression_stability'
  | 'hallucination_safety'
  | 'refusal_acceptance'
  | 'value'
  | 'developer_ergonomics';

export type EvidenceType =
  | 'firsthand_usage'
  | 'comparative_evaluation'
  | 'bug_regression_report'
  | 'integration_failure'
  | 'benchmark_anecdote'
  | 'hearsay'
  | 'release_update_reaction'
  | 'pricing_value_comment';

export type LabelRow = {
  review_id?: string;
  source_platform: string;
  community_id: string;
  thread_id: string;
  source_item_id: string;
  url: string;
  model_id: string;
  provider_id?: string;
  product_id?: string;
  inference_profile?: string;
  task_category: TaskCategory;
  aspect_category: AspectCategory;
  evidence_type: EvidenceType;
  claim_type: string;
  polarity_score: number;
  extractor_confidence: number;
  firsthand_flag: boolean;
  regression_flag: boolean;
  hallucination_flag: boolean;
  refusal_flag: boolean;
  value_flag: boolean;
  evidence_text: string;
  classifier_task_category?: string;
  classifier_task_confidence?: number | '';
  classifier_aspect_category?: string;
  classifier_aspect_confidence?: number | '';
  classifier_evidence_type?: string;
  classifier_evidence_confidence?: number | '';
  classifier_polarity_score?: number | '';
  classifier_polarity_confidence?: number | '';
  classifier_firsthand_flag?: boolean | '';
  classifier_firsthand_confidence?: number | '';
  classifier_disagreement_count?: number | '';
  reviewed_flag?: boolean;
  human_excluded_from_scoring?: boolean | '';
  human_exclusion_reason?: string;
  human_provider_id?: string;
  human_model_id?: string;
  human_product_id?: string;
  human_inference_profile?: string;
  human_task_category?: string;
  human_aspect_category?: string;
  human_evidence_type?: string;
  human_polarity_score?: number | '';
  human_firsthand_flag?: boolean | '';
  human_notes?: string;
};
