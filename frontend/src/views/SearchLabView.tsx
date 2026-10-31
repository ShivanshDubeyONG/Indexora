import React, { useState, useEffect } from 'react';
import { executeSearch } from '../api';
import { SearchResponse, DatasetCurrent } from '../types';
import {
  Search,
  Zap,
  CheckCircle,
  AlertTriangle,
  Filter,
  Clock,
  CheckCircle2,
  Sparkles,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Activity,
  Sliders
} from 'lucide-react';

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

  // Execution Feedback State
  const [evalTimestamp, setEvalTimestamp] = useState<string | null>(null);
  const [evalFlash, setEvalFlash] = useState(false);
  const [flashKey, setFlashKey] = useState(0);

  // Vector Coordinate Inspector State
  const [inspectedId, setInspectedId] = useState<number | null>(null);

  const runSearch = async (isManual = false) => {
    setLoading(true);
    setError(null);
    const startTime = performance.now();
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

      // Provide human-perceptible tactile feedback on manual click
      if (isManual) {
        const elapsed = performance.now() - startTime;
        if (elapsed < 260) {
          await new Promise((r) => setTimeout(r, 260 - elapsed));
        }
      }

      setResult(res);
      setEvalTimestamp(new Date().toLocaleTimeString());
      setEvalFlash(true);
      setFlashKey((prev) => prev + 1);
      setTimeout(() => setEvalFlash(false), 800);
    } catch (err: any) {
      setError(err.message || 'Search execution failed');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (dataset) {
      runSearch(false);
    }
  }, [indexType, queryIndex, k, nprobe, efSearch, filterMode]);

  // Current query vector for vector diff
  const currentQueryVec = dataset?.query_samples?.[queryIndex]?.vector || [];

  return (
    <div className="space-y-8 font-sans">
      {/* View Header */}
      <div className="border-b border-border pb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="font-heading font-bold text-xl sm:text-2xl text-graphite-dark tracking-tight">
            Interactive Query Retrieval Lab
          </h1>
          <p className="text-xs sm:text-sm text-graphite-muted mt-1">
            Execute nearest-neighbor queries against live in-memory index structures with real-time ground-truth verification.
          </p>
        </div>

        {/* Live Execution Stamp */}
        {evalTimestamp && (
          <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-mono transition-all duration-300 ${
            evalFlash
              ? 'bg-accent/15 border-accent text-accent shadow-glow scale-[1.02]'
              : 'bg-surface-secondary border-border text-graphite-muted'
          }`}>
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>Last compute: {evalTimestamp}</span>
            <span className="text-graphite-subtle">·</span>
            <span className="text-emerald-600 dark:text-emerald-400 font-semibold">
              {result?.latency_ms.toFixed(3)} ms
            </span>
          </div>
        )}
      </div>

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Control Column (5 Cols) */}
        <div className="lg:col-span-5 bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-5 transition-colors">
          <div className="flex items-center justify-between border-b border-border pb-3">
            <span className="font-heading font-semibold text-sm text-graphite-dark">Query Configuration</span>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-surface-secondary text-graphite border border-border">
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
              className="w-full px-3 py-2 text-xs rounded-lg border border-border bg-surface-subtle text-graphite font-medium focus:border-accent focus:outline-none transition-colors"
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
                className="w-full px-3 py-2 text-xs font-mono rounded-lg border border-border bg-surface-subtle text-graphite focus:border-accent focus:outline-none transition-colors"
              />
            ) : (
              <select
                value={queryIndex}
                onChange={(e) => setQueryIndex(Number(e.target.value))}
                className="w-full px-3 py-2 text-xs rounded-lg border border-border bg-surface-subtle text-graphite font-mono focus:border-accent focus:outline-none transition-colors"
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
                className="w-full accent-accent cursor-pointer"
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
                  className="w-full accent-accent cursor-pointer"
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
                  className="w-full accent-accent cursor-pointer"
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
                className={`py-1.5 px-2 rounded-lg border text-left cursor-pointer transition ${
                  filterMode === 'none'
                    ? 'border-accent bg-accent/10 text-accent font-semibold'
                    : 'border-border bg-surface-subtle text-graphite hover:bg-surface-secondary'
                }`}
              >
                All Eligible
              </button>
              <button
                type="button"
                onClick={() => setFilterMode('even_ids')}
                className={`py-1.5 px-2 rounded-lg border text-left cursor-pointer transition ${
                  filterMode === 'even_ids'
                    ? 'border-accent bg-accent/10 text-accent font-semibold'
                    : 'border-border bg-surface-subtle text-graphite hover:bg-surface-secondary'
                }`}
              >
                Even IDs Only
              </button>
              <button
                type="button"
                onClick={() => setFilterMode('exclude_top1')}
                className={`py-1.5 px-2 rounded-lg border text-left cursor-pointer transition ${
                  filterMode === 'exclude_top1'
                    ? 'border-accent bg-accent/10 text-accent font-semibold'
                    : 'border-border bg-surface-subtle text-graphite hover:bg-surface-secondary'
                }`}
              >
                Exclude Top-1
              </button>
              <button
                type="button"
                onClick={() => setFilterMode('custom')}
                className={`py-1.5 px-2 rounded-lg border text-left cursor-pointer transition ${
                  filterMode === 'custom'
                    ? 'border-accent bg-accent/10 text-accent font-semibold'
                    : 'border-border bg-surface-subtle text-graphite hover:bg-surface-secondary'
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
                className="w-full px-2.5 py-1.5 text-xs font-mono rounded-lg border border-border bg-surface-subtle text-graphite mt-2 focus:border-accent focus:outline-none"
              />
            )}
          </div>

          {/* Action Button with Tactile Feedback */}
          <button
            onClick={() => runSearch(true)}
            disabled={loading}
            className="w-full py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white text-xs font-semibold shadow-sm hover:shadow-glow transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-60 active:scale-[0.98]"
          >
            {loading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Evaluating Matrix Kernel...</span>
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                <span>Execute Vector Search</span>
              </>
            )}
          </button>
        </div>

        {/* Right Output Column (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          {error && (
            <div className="p-4 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900 text-red-700 dark:text-red-300 text-xs flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 flex-shrink-0 text-red-500" />
              <span>{error}</span>
            </div>
          )}

          {/* Metric Cards Row with Flash Animation */}
          {result && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className={`bg-surface-card border rounded-xl p-3.5 shadow-subtle transition-all duration-300 ${
                evalFlash ? 'border-accent shadow-glow scale-[1.02]' : 'border-border'
              }`}>
                <span className="text-[11px] font-mono text-graphite-muted block">Query Latency</span>
                <span className="text-lg font-mono font-bold text-graphite-dark mt-0.5 block">
                  {result.latency_ms.toFixed(3)} ms
                </span>
                <span className="text-[10px] text-graphite-muted">
                  {result.was_cached ? '⚡ Cached Index' : `Build: ${result.build_time_ms.toFixed(1)}ms`}
                </span>
              </div>

              <div className={`bg-surface-card border rounded-xl p-3.5 shadow-subtle transition-all duration-300 ${
                evalFlash ? 'border-accent shadow-glow scale-[1.02]' : 'border-border'
              }`}>
                <span className="text-[11px] font-mono text-graphite-muted block">Distance Ops</span>
                <span className="text-lg font-mono font-bold text-accent mt-0.5 block">
                  {result.distance_computations} / {dataset?.n_base || 120}
                </span>
                <span className="text-[10px] text-graphite-muted">
                  {(((dataset?.n_base || 120) - result.distance_computations) / (dataset?.n_base || 120) * 100).toFixed(0)}% Work Saved
                </span>
              </div>

              <div className={`bg-surface-card border rounded-xl p-3.5 shadow-subtle transition-all duration-300 ${
                evalFlash ? 'border-accent shadow-glow scale-[1.02]' : 'border-border'
              }`}>
                <span className="text-[11px] font-mono text-graphite-muted block">Oracle Recall@k</span>
                <span className="text-lg font-mono font-bold text-emerald-600 dark:text-emerald-400 mt-0.5 block">
                  {(result.recall_at_k * 100).toFixed(1)}%
                </span>
                <span className="text-[10px] text-graphite-muted">
                  vs. Exhaustive BLAS
                </span>
              </div>

              <div className={`bg-surface-card border rounded-xl p-3.5 shadow-subtle transition-all duration-300 ${
                evalFlash ? 'border-accent shadow-glow scale-[1.02]' : 'border-border'
              }`}>
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

          {/* Results Table with Row Inspection & Heatstrips */}
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-4 transition-colors">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div>
                <h3 className="font-heading font-semibold text-sm text-graphite-dark">
                  Retrieved Nearest Neighbors
                </h3>
                <span className="text-xs text-graphite-muted">
                  Click any entity to inspect its float32 vector coordinates & deviation
                </span>
              </div>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-surface-secondary text-graphite border border-border">
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
                      <th className="py-2.5 px-3 font-semibold text-center">Coordinate Heatstrip</th>
                      <th className="py-2.5 px-3 font-semibold text-right">Squared L2 (||q-x||²)</th>
                      <th className="py-2.5 px-3 font-semibold text-center">Oracle Status</th>
                    </tr>
                  </thead>
                  <tbody key={flashKey} className="divide-y divide-border bg-surface-card text-graphite">
                    {result.results.map((item, idx) => {
                      const sample = dataset?.base_samples?.find((s) => s.id === item.id);
                      const isInspected = inspectedId === item.id;
                      const vec = sample?.vector || [];

                      return (
                        <React.Fragment key={item.id}>
                          <tr
                            onClick={() => setInspectedId(isInspected ? null : item.id)}
                            className={`hover:bg-surface-subtle transition-colors cursor-pointer animate-fade-in ${
                              isInspected ? 'bg-surface-subtle font-semibold' : ''
                            }`}
                            style={{ animationDelay: `${idx * 20}ms` }}
                          >
                            <td className="py-2.5 px-3 font-bold text-graphite">#{item.rank}</td>
                            <td className="py-2.5 px-3 font-bold text-accent flex items-center gap-1.5">
                              <span>{item.id}</span>
                              {isInspected ? <ChevronUp className="w-3 h-3 text-graphite-subtle" /> : <ChevronDown className="w-3 h-3 text-graphite-subtle opacity-40" />}
                            </td>

                            {/* Dimensionality Spectrum Sparkline */}
                            <td className="py-2.5 px-3 text-center">
                              <div className="inline-flex items-center gap-0.5 bg-surface-subtle p-1 rounded border border-border/60">
                                {vec.slice(0, 8).map((val, dIdx) => {
                                  const normalized = Math.min(Math.max((val + 2) / 4, 0.1), 1.0);
                                  return (
                                    <span
                                      key={dIdx}
                                      title={`d[${dIdx}] = ${val.toFixed(3)}`}
                                      className="w-1.5 rounded-sm transition-all"
                                      style={{
                                        height: `${Math.round(normalized * 12) + 3}px`,
                                        backgroundColor: val >= 0 ? '#3B82F6' : '#94A3B8',
                                      }}
                                    />
                                  );
                                })}
                              </div>
                            </td>

                            <td className="py-2.5 px-3 text-right text-graphite-muted font-mono">
                              {item.squared_l2.toFixed(5)}
                            </td>

                            <td className="py-2.5 px-3 text-center">
                              {item.is_exact_match ? (
                                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 font-semibold text-[11px]">
                                  <CheckCircle className="w-3 h-3" /> Oracle Hit
                                </span>
                              ) : (
                                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/20 text-accent font-semibold text-[11px]">
                                  ANN Approx
                                </span>
                              )}
                            </td>
                          </tr>

                          {/* Expanded Coordinate Telemetry Drawer */}
                          {isInspected && (
                            <tr className="bg-surface-secondary/40 border-b border-border">
                              <td colSpan={5} className="p-4 space-y-3 font-mono text-xs">
                                <div className="flex items-center justify-between border-b border-border pb-2 text-[11px]">
                                  <span className="font-bold text-graphite flex items-center gap-1.5">
                                    <Activity className="w-3.5 h-3.5 text-accent" />
                                    Vector Telemetry Inspection (Entity #{item.id})
                                  </span>
                                  <span className="text-graphite-muted">
                                    Cluster: {sample ? `Cluster ${sample.cluster}` : 'Live Index'}
                                  </span>
                                </div>

                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-[11px]">
                                  <div className="p-2.5 rounded bg-surface-card border border-border space-y-1">
                                    <span className="text-graphite-subtle text-[10px] block">Candidate Vector Coordinates x ∈ ℝᵈ:</span>
                                    <span className="text-graphite font-mono block break-all leading-relaxed">
                                      [{vec.map((v) => v.toFixed(3)).join(', ')}]
                                    </span>
                                  </div>

                                  <div className="p-2.5 rounded bg-surface-card border border-border space-y-1">
                                    <span className="text-graphite-subtle text-[10px] block">Coordinate Absolute Error |q - x|:</span>
                                    <span className="text-accent font-mono block break-all leading-relaxed">
                                      [{vec.map((v, i) => Math.abs((currentQueryVec[i] || 0) - v).toFixed(3)).join(', ')}]
                                    </span>
                                  </div>
                                </div>
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      );
                    })}
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
