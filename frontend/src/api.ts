import {
  SystemStatus,
  DatasetCurrent,
  SearchResponse,
  BenchmarkSweepResponse,
  SiftEvidence,
  PcaProjectionResponse,
  SemanticStatus,
  SemanticSearchResponse,
} from './types';

const BASE_URL = '/api';

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const res = await fetch(`${BASE_URL}/status`);
  if (!res.ok) throw new Error(`Failed to fetch status: ${res.statusText}`);
  return res.json();
}

export async function generateDataset(params: {
  n_base: number;
  n_queries: number;
  d: number;
  n_clusters: number;
  cluster_std: number;
  seed: number;
}) {
  const res = await fetch(`${BASE_URL}/dataset/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Dataset generation failed: ${res.statusText}`);
  return res.json();
}

export async function fetchCurrentDataset(): Promise<DatasetCurrent> {
  const res = await fetch(`${BASE_URL}/dataset/current`);
  if (!res.ok) throw new Error(`Failed to fetch dataset: ${res.statusText}`);
  return res.json();
}

export async function executeSearch(params: {
  index_type: string;
  query_id_index?: number;
  custom_vector?: number[];
  k: number;
  nprobe?: number;
  ef_search?: number;
  filter_mode: string;
  custom_filter_ids?: number[];
}): Promise<SearchResponse> {
  const res = await fetch(`${BASE_URL}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export async function runBenchmarkSweep(params: {
  k: number;
  repetitions: number;
  ivf_nlist: number;
  ivf_nprobe: number;
  pq_m: number;
  pq_ksub: number;
  hnsw_m: number;
  hnsw_ef_construction: number;
  hnsw_ef_search: number;
}): Promise<BenchmarkSweepResponse> {
  const res = await fetch(`${BASE_URL}/benchmark/sweep`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Benchmark sweep failed: ${res.statusText}`);
  return res.json();
}

export async function fetchSiftEvidence(): Promise<SiftEvidence> {
  const res = await fetch(`${BASE_URL}/benchmark/sift`);
  if (!res.ok) throw new Error(`Failed to fetch SIFT evidence: ${res.statusText}`);
  return res.json();
}

export async function fetchPcaProjection(params: {
  query_id_index: number;
  retrieved_ids?: number[];
}): Promise<PcaProjectionResponse> {
  const queryParam = `query_id_index=${params.query_id_index}`;
  const idsParam = params.retrieved_ids?.length ? `&retrieved_ids=${params.retrieved_ids.join(',')}` : '';
  const res = await fetch(`${BASE_URL}/explorer/pca?${queryParam}${idsParam}`);
  if (!res.ok) throw new Error(`Failed to compute PCA projection: ${res.statusText}`);
  return res.json();
}

export async function addVectorMutation(params: {
  ext_id: number;
  constant_val?: number;
  vector?: number[];
}) {
  const res = await fetch(`${BASE_URL}/mutations/add`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export async function deleteVectorMutation(ext_id: number) {
  const res = await fetch(`${BASE_URL}/mutations/delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ext_id }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export async function listSavedIndexes(): Promise<{ saved_indexes: string[] }> {
  const res = await fetch(`${BASE_URL}/persistence/list`);
  if (!res.ok) throw new Error(`Failed to list saved indexes: ${res.statusText}`);
  return res.json();
}

export async function saveNamedIndex(name: string, overwrite = true) {
  const res = await fetch(`${BASE_URL}/persistence/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, overwrite }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export async function loadNamedIndex(name: string, verify_checksums = true) {
  const res = await fetch(`${BASE_URL}/persistence/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, verify_checksums }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export async function fetchSemanticStatus(): Promise<SemanticStatus> {
  const res = await fetch(`${BASE_URL}/semantic/status`);
  if (!res.ok) throw new Error(`Failed to fetch semantic status: ${res.statusText}`);
  return res.json();
}

export async function searchSemantic(params: {
  query: string;
  k: number;
  category?: string;
  index_type: string;
  nprobe: number;
  ef_search: number;
}): Promise<SemanticSearchResponse> {
  const res = await fetch(`${BASE_URL}/semantic/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

