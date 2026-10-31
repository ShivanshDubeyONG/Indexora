import React, { useState, useEffect } from 'react';
import { executeSearch } from '../api';
import { SearchResponse, DatasetCurrent } from '../types';
import { Search, Zap, CheckCircle, AlertTriangle, Filter, Clock, Hash } from 'lucide-react';

interface SearchLabViewProps {
  dataset: DatasetCurrent | null;
}

export const SearchLabView: React.FC<SearchLabViewProps> = ({ dataset }) => {
  const [indexType, setIndexType] = useState('Exact BLAS');
  const [queryIndex, setQueryIndex] = useState(0);
  const [useCustomVector, setUseCustomVector] = useState(false);
  const [customVectorStr, setCustomVectorStr] = useState('');
  const [k, setK] = useState(5);
  const [nprobe, setNprobe] = useState(2);
  const [efSearch, setEfSearch] = useState(32);
  const [filterMode, setFilterMode] = useState('none');
  const [customFilterStr, setCustomFilterStr] = useState('1000, 1001, 1002');
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runSearch = async () => {
    setLoading(true);
    setError(null);
    try {
      let customVec: number[] | undefined = undefined;
      if (useCustomVector && customVectorStr.trim()) {
        const parts = customVectorStr.split(',').map((s) => parseFloat(s.trim()));
        if (parts.some(isNaN)) throw new Error('Custom vector must be comma-separated numbers');
        customVec = parts;
      }

      let customIds: number[] | undefined = undefined;
      if (filterMode === 'custom' && customFilterStr.trim()) {
        customIds = customFilterStr.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n));
      }

      const res = await executeSearch({
        index_type: indexType,
        query_id_index: useCustomVector ? undefined : queryIndex,
        custom_vector: customVec,
        k: Number(k),
        nprobe: Number(nprobe),
        ef_search: Number(efSearch),
        filter_mode: filterMode,
        custom_filter_ids: customIds,
      });
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Search execution failed');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (dataset) {
      runSearch();
    }
  }, [indexType, queryIndex, k, nprobe, efSearch, filterMode]);

  return (
    <div className="space-y-8">
      {/* View Header */}
      <div className="border-b border-border pb-4">
        <h1 className="font-heading font-bold text-xl sm:text-2xl text-graphite-dark tracking-tight">
          Interactive Query Retrieval Lab
        </h1>
        <p className="text-xs sm:text-sm text-graphite-muted mt-1">
          Execute nearest-neighbor queries against live in-memory index structures with real-time ground-truth verification.
        </p>
      </div>

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Control Column (5 Cols) */}
        <div className="lg:col-span-5 bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-5">
          <div className="flex items-center justify-between border-b border-border pb-3">
            <span className="font-heading font-semibold text-sm text-graphite-dark">Query Configuration</span>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-surface-secondary text-graphite">
              d={dataset?.d || 8} float32
            </span>
          </div>

          {/* Architecture Picker */}
          <div>
            <label className="block text-xs font-medium text-graphite-muted mb-1.5">
              Index Architecture
            </label>
            <select
              value={indexType}
              onChange={(e) => setIndexType(e.target.value)}
              className="w-full px-3 py-2 text-xs rounded-lg border border-border bg-white text-graphite font-medium focus:border-accent focus:outline-none"
            >
              <option value="Exact BLAS">Exact BLAS (Level-3 Matrix Kernel)</option>
              <option value="Exact Flat (Ref)">Exact Flat (Float64 Reference Oracle)</option>
              <option value="IVF-Flat">IVF-Flat (Inverted Voronoi Partitioning)</option>
              <option value="IVF-PQ">IVF-PQ (Product Quantization + ADC)</option>
              <option value="HNSW">HNSW (Hierarchical Navigable Small World)</option>
            </select>
          </div>

          {/* Query Vector Selection */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-graphite-muted">Target Query Vector</label>
              <button
                type="button"
                onClick={() => setUseCustomVector(!useCustomVector)}
                className="text-[11px] text-accent hover:underline cursor-pointer"
              >
                {useCustomVector ? 'Use Precomputed Queries' : 'Enter Custom Vector'}
              </button>
            </div>

            {useCustomVector ? (
              <input
                type="text"
                placeholder="e.g. 0.12, -0.45, 0.88, ..."
                value={customVectorStr}
                onChange={(e) => setCustomVectorStr(e.target.value)}
                className="w-full px-3 py-2 text-xs font-mono rounded-lg border border-border bg-white text-graphite focus:border-accent focus:outline-none"
              />
            ) : (
              <select
                value={queryIndex}
                onChange={(e) => setQueryIndex(Number(e.target.value))}
                className="w-full px-3 py-2 text-xs rounded-lg border border-border bg-white text-graphite font-mono focus:border-accent focus:outline-none"
              >
                {dataset?.query_samples?.map((q, idx) => (
                  <option key={q.id} value={idx}>
                    Query #{idx} (Entity ID {q.id})
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Top-k & Dynamic Hyperparameters */}
          <div className="space-y-4 pt-2 border-t border-border">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="font-medium text-graphite-muted">Top-k Results</span>
                <span className="font-mono font-bold text-accent">{k}</span>
              </div>
              <input
                type="range"
                min={1}
                max={Math.min(dataset?.n_base || 100, 25)}
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="w-full accent-accent"
              />
            </div>

            {/* IVF Hyperparameter */}
            {indexType.includes('IVF') && (
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-medium text-graphite-muted">Centroid Probe Depth (nprobe)</span>
                  <span className="font-mono font-bold text-accent">{nprobe} / {dataset?.n_clusters || 4}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={Math.max(2, dataset?.n_clusters || 4)}
                  value={nprobe}
                  onChange={(e) => setNprobe(Number(e.target.value))}
                  className="w-full accent-accent"
                />
                <span className="text-[11px] text-graphite-muted mt-1 block">
                  Probing {nprobe} Voronoi cells per query.
                </span>
              </div>
            )}

            {/* HNSW Hyperparameter */}
            {indexType === 'HNSW' && (
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-medium text-graphite-muted">Beam Exploration Depth (ef_search)</span>
                  <span className="font-mono font-bold text-accent">{efSearch}</span>
                </div>
                <input
                  type="range"
                  min={8}
                  max={96}
                  step={8}
                  value={efSearch}
                  onChange={(e) => setEfSearch(Number(e.target.value))}
                  className="w-full accent-accent"
                />
                <span className="text-[11px] text-graphite-muted mt-1 block">
                  Layer 0 priority queue candidate exploration beam.
                </span>
              </div>
            )}
          </div>

          {/* Filter Eligibility (filter_ids) */}
          <div className="space-y-2 pt-2 border-t border-border">
            <label className="block text-xs font-medium text-graphite-muted flex items-center gap-1.5">
              <Filter className="w-3.5 h-3.5 text-accent" />
              <span>Metadata Eligibility Filter (filter_ids)</span>
            </label>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <button
                type="button"
                onClick={() => setFilterMode('none')}
                className={`py-1.5 px-2 rounded border text-left cursor-pointer transition ${
                  filterMode === 'none' ? 'border-accent bg-surface-secondary text-accent font-semibold' : 'border-border bg-white text-graphite'
                }`}
              >
                All Eligible
              </button>
              <button
                type="button"
                onClick={() => setFilterMode('even_ids')}
                className={`py-1.5 px-2 rounded border text-left cursor-pointer transition ${
                  filterMode === 'even_ids' ? 'border-accent bg-surface-secondary text-accent font-semibold' : 'border-border bg-white text-graphite'
                }`}
              >
                Even IDs Only
              </button>
              <button
                type="button"
                onClick={() => setFilterMode('exclude_top1')}
                className={`py-1.5 px-2 rounded border text-left cursor-pointer transition ${
                  filterMode === 'exclude_top1' ? 'border-accent bg-surface-secondary text-accent font-semibold' : 'border-border bg-white text-graphite'
                }`}
              >
                Exclude Top-1
              </button>
              <button
                type="button"
                onClick={() => setFilterMode('custom')}
                className={`py-1.5 px-2 rounded border text-left cursor-pointer transition ${
                  filterMode === 'custom' ? 'border-accent bg-surface-secondary text-accent font-semibold' : 'border-border bg-white text-graphite'
                }`}
              >
                Custom List
              </button>
            </div>

            {filterMode === 'custom' && (
              <input
                type="text"
                value={customFilterStr}
                onChange={(e) => setCustomFilterStr(e.target.value)}
                placeholder="1000, 1002, 1005"
                className="w-full px-2.5 py-1.5 text-xs font-mono rounded border border-border bg-white text-graphite mt-2"
              />
            )}
          </div>

          <button
            onClick={runSearch}
            disabled={loading}
            className="w-full py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white text-xs font-semibold shadow-sm transition flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
          >
            <Search className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span>{loading ? 'Evaluating Query...' : 'Execute Vector Search'}</span>
          </button>
        </div>

        {/* Right Output Column (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          {error && (
            <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 flex-shrink-0 text-rose-600" />
              <span>{error}</span>
            </div>
          )}

          {/* Metric Cards Row */}
          {result && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-surface-card border border-border rounded-lg p-3.5 shadow-subtle">
                <span className="text-[11px] font-mono text-graphite-muted block">Query Latency</span>
                <span className="text-lg font-mono font-bold text-graphite-dark mt-0.5 block">
                  {result.latency_ms.toFixed(3)} ms
                </span>
                <span className="text-[10px] text-graphite-muted">
                  {result.was_cached ? '⚡ Cached Index' : `Build: ${result.build_time_ms.toFixed(1)}ms`}
                </span>
              </div>

              <div className="bg-surface-card border border-border rounded-lg p-3.5 shadow-subtle">
                <span className="text-[11px] font-mono text-graphite-muted block">Distance Ops</span>
                <span className="text-lg font-mono font-bold text-accent mt-0.5 block">
                  {result.distance_computations} / {dataset?.n_base || 120}
                </span>
                <span className="text-[10px] text-graphite-muted">
                  {(((dataset?.n_base || 120) - result.distance_computations) / (dataset?.n_base || 120) * 100).toFixed(0)}% Work Saved
                </span>
              </div>

              <div className="bg-surface-card border border-border rounded-lg p-3.5 shadow-subtle">
                <span className="text-[11px] font-mono text-graphite-muted block">Oracle Recall@k</span>
                <span className="text-lg font-mono font-bold text-emerald-700 mt-0.5 block">
                  {(result.recall_at_k * 100).toFixed(1)}%
                </span>
                <span className="text-[10px] text-graphite-muted">
                  vs. Exhaustive BLAS
                </span>
              </div>

              <div className="bg-surface-card border border-border rounded-lg p-3.5 shadow-subtle">
                <span className="text-[11px] font-mono text-graphite-muted block">Eligible Pool</span>
                <span className="text-lg font-mono font-bold text-graphite-dark mt-0.5 block">
                  {result.num_eligible_ground_truth}
                </span>
                <span className="text-[10px] text-graphite-muted">
                  Filter: {filterMode}
                </span>
              </div>
            </div>
          )}

          {/* Results Table */}
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-4">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div>
                <h3 className="font-heading font-semibold text-sm text-graphite-dark">
                  Retrieved Nearest Neighbors
                </h3>
                <span className="text-xs text-graphite-muted">
                  Sorted deterministically by <code>(distance, external_id)</code>
                </span>
              </div>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-surface-secondary text-graphite">
                Top-{k}
              </span>
            </div>

            {result && result.results.length > 0 ? (
              <div className="overflow-x-auto rounded-lg border border-border">
                <table className="w-full text-xs font-mono text-left">
                  <thead className="bg-surface-secondary text-graphite-muted border-b border-border">
                    <tr>
                      <th className="py-2.5 px-3 font-semibold">Rank</th>
                      <th className="py-2.5 px-3 font-semibold">Entity ID</th>
                      <th className="py-2.5 px-3 font-semibold text-right">Squared L2 (||q-x||²)</th>
                      <th className="py-2.5 px-3 font-semibold text-center">Oracle Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border bg-white text-graphite">
                    {result.results.map((item) => (
                      <tr key={item.id} className="hover:bg-surface-subtle">
                        <td className="py-2 px-3 font-bold text-graphite">#{item.rank}</td>
                        <td className="py-2 px-3 font-bold text-accent">{item.id}</td>
                        <td className="py-2 px-3 text-right text-graphite-muted font-mono">
                          {item.squared_l2.toFixed(5)}
                        </td>
                        <td className="py-2 px-3 text-center">
                          {item.is_exact_match ? (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-700 font-semibold text-[11px]">
                              <CheckCircle className="w-3 h-3" /> Oracle Hit
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-blue-50 border border-blue-200 text-accent font-semibold text-[11px]">
                              ANN Approximation
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="p-8 text-center text-xs text-graphite-muted border border-dashed border-border rounded-lg">
                No eligible candidates retrieved matching the query and filter criteria.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
