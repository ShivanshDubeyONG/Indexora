export interface SystemStatus {
  engine_version: string;
  python_version: string;
  numpy_version: string;
  dataset: {
    n_base: number;
    n_queries: number;
    d: number;
    n_clusters: number;
    seed: number;
    id_range: [number, number];
    dataset_id: string;
  };
  session: {
    index_name: string;
    active_count: number;
    d: number;
    added_count: number;
    deleted_count: number;
  };
  semantic_available: boolean;
}

export interface DatasetCurrent {
  n_base: number;
  n_queries: number;
  d: number;
  n_clusters: number;
  seed: number;
  base_samples: Array<{
    id: number;
    cluster: number;
    vector: number[];
  }>;
  query_samples: Array<{
    id: number;
    vector: number[];
  }>;
}

export interface SearchResultItem {
  rank: number;
  id: number;
  squared_l2: number;
  is_exact_match: boolean;
}

export interface SearchResponse {
  index_type: string;
  query_label: string;
  k: number;
  latency_ms: number;
  distance_computations: number;
  recall_at_k: number;
  num_eligible_ground_truth: number;
  was_cached: boolean;
  build_time_ms: number;
  results: SearchResultItem[];
}

export interface BenchmarkMetric {
  index_name: string;
  build_time_ms: number;
  median_latency_ms: number;
  p95_latency_ms: number;
  qps: number;
  recall_at_k: number;
  distance_computations_per_query: number;
  memory_kb: number;
  memory_method: string;
}

export interface BenchmarkSweepResponse {
  k: number;
  n_queries: number;
  metrics: BenchmarkMetric[];
}

export interface SiftEvidence {
  provenance: {
    platform: string;
    cpu: string;
    python: string;
    numpy: string;
    git_commit: string;
    dataset_sift100k_sha256: string;
    queries_sift100k_sha256: string;
  };
  sift100k_baseline: any[];
  sift100k_ivfpq: any[];
  sift10k_hnsw: any[];
}

export interface PcaPoint {
  id: number;
  x: number;
  y: number;
  cluster: number;
  is_retrieved: boolean;
}

export interface PcaCentroid {
  cluster: number;
  x: number;
  y: number;
}

export interface PcaProjectionResponse {
  points: PcaPoint[];
  centroids: PcaCentroid[];
  query: {
    query_index: number;
    id: number;
    x: number;
    y: number;
  };
  d: number;
  n_clusters: number;
}

export interface SemanticStatus {
  available: boolean;
  model_name: string;
  document_count: number;
  categories: string[];
  corpus_sample: Array<{
    doc_id: number;
    category: string;
    title: string;
    text: string;
    tags: string[];
  }>;
}

export interface RankedDocument {
  rank: number;
  doc_id: number;
  category: string;
  title: string;
  text: string;
  tags: string[];
  cosine_similarity: number;
  squared_l2: number;
  is_oracle_hit: boolean;
}

export interface SemanticSearchResponse {
  query: string;
  k: number;
  category_filter: string | null;
  total_latency_ms: number;
  text_embed_time_ms: number;
  vector_search_time_ms: number;
  distance_computations: number;
  oracle_agreement_pct: number;
  ranked_documents: RankedDocument[];
}
