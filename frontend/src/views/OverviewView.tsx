import React, { useState } from 'react';
import { SystemStatus, DatasetCurrent } from '../types';
import { generateDataset } from '../api';
import { ArchitectureShowpiece } from '../components/ArchitectureShowpiece';
import { 
  ArrowRight, 
  Layers, 
  Cpu, 
  SlidersHorizontal, 
  CheckCircle2, 
  Database, 
  RefreshCw 
} from 'lucide-react';

interface OverviewViewProps {
  status: SystemStatus | null;
  dataset: DatasetCurrent | null;
  onDatasetUpdated: () => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({ status, dataset, onDatasetUpdated }) => {
  const [showConfig, setShowConfig] = useState(false);
  const [nBase, setNBase] = useState(status?.dataset.n_base || 120);
  const [dDim, setDDim] = useState(status?.dataset.d || 8);
  const [clusters, setClusters] = useState(status?.dataset.n_clusters || 4);
  const [std, setStd] = useState(0.25);
  const [seed, setSeed] = useState(42);
  const [generating, setGenerating] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const handleRegenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    setGenerating(true);
    setMsg(null);
    try {
      await generateDataset({
        n_base: Number(nBase),
        n_queries: 8,
        d: Number(dDim),
        n_clusters: Number(clusters),
        cluster_std: Number(std),
        seed: Number(seed),
      });
      setMsg('Dataset regenerated successfully!');
      onDatasetUpdated();
      setTimeout(() => setMsg(null), 3500);
    } catch (err: any) {
      setMsg(`Error: ${err.message}`);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="space-y-10">
      {/* 1. Hero Statement & Architecture Vision */}
      <section className="bg-surface-card border border-border rounded-xl p-6 sm:p-8 shadow-card relative overflow-hidden">
        <div className="max-w-3xl space-y-3 relative z-10">
          <h1 className="font-heading font-bold text-2xl sm:text-3xl text-graphite-dark tracking-tight">
            Vector Index Architectures from First Principles
          </h1>
          <p className="text-graphite-muted text-sm sm:text-base leading-relaxed">
            Indexora is an engineering-grade nearest-neighbor retrieval system implemented directly in Python and NumPy. It eliminates black-box vector database abstractions to demonstrate the exact data structures, partitioning kernels, cache dynamics, and mechanical trade-offs that govern high-dimensional similarity search.
          </p>

          <div className="pt-2 flex flex-wrap items-center gap-3 text-xs font-mono">
            <span className="px-3 py-1 rounded-md bg-surface-secondary border border-border text-graphite">
              Zero External C++/FAISS Dependencies
            </span>
            <span className="px-3 py-1 rounded-md bg-surface-secondary border border-border text-graphite">
              Level-3 BLAS GEMM Optimization
            </span>
            <span className="px-3 py-1 rounded-md bg-surface-secondary border border-border text-graphite">
              Deterministic (Distance, ID) Tie-Breaking
            </span>
          </div>
        </div>
      </section>

      {/* Interactive Architecture Visualizer Showpiece */}
      <ArchitectureShowpiece />

      {/* 2. Multi-Stage Retrieval Pipeline Map */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="font-heading font-bold text-lg text-graphite-dark">
            Retrieval Execution Pipeline
          </h2>
          <span className="text-xs font-mono text-graphite-muted">5 Deterministic Stages</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
          <div className="bg-surface-card border border-border rounded-lg p-4 shadow-subtle flex flex-col justify-between">
            <div>
              <div className="text-[11px] font-mono font-bold text-accent mb-1 tracking-wider">STAGE 01</div>
              <div className="font-heading font-semibold text-sm text-graphite-dark">Query Ingestion</div>
              <div className="text-xs text-graphite-muted mt-1 font-mono">q ∈ ℝᵈ (float32)</div>
            </div>
            <div className="mt-4 text-xs text-graphite-muted border-t border-border pt-2">
              Validates dimensionality and floating-point alignment.
            </div>
          </div>

          <div className="bg-surface-card border border-border rounded-lg p-4 shadow-subtle flex flex-col justify-between">
            <div>
              <div className="text-[11px] font-mono font-bold text-accent mb-1 tracking-wider">STAGE 02</div>
              <div className="font-heading font-semibold text-sm text-graphite-dark">Partitioning</div>
              <div className="text-xs text-graphite-muted mt-1 font-mono">Voronoi / Skip-Graph</div>
            </div>
            <div className="mt-4 text-xs text-graphite-muted border-t border-border pt-2">
              Prunes search space via centroid assignment or graph routing.
            </div>
          </div>

          <div className="bg-surface-card border border-border rounded-lg p-4 shadow-subtle flex flex-col justify-between">
            <div>
              <div className="text-[11px] font-mono font-bold text-accent mb-1 tracking-wider">STAGE 03</div>
              <div className="font-heading font-semibold text-sm text-graphite-dark">Candidate Scoring</div>
              <div className="text-xs text-graphite-muted mt-1 font-mono">BLAS GEMM / ADC LUT</div>
            </div>
            <div className="mt-4 text-xs text-graphite-muted border-t border-border pt-2">
              Squared L2 computation via Level-3 matrix products or lookup tables.
            </div>
          </div>

          <div className="bg-surface-card border border-border rounded-lg p-4 shadow-subtle flex flex-col justify-between">
            <div>
              <div className="text-[11px] font-mono font-bold text-accent mb-1 tracking-wider">STAGE 04</div>
              <div className="font-heading font-semibold text-sm text-graphite-dark">Top-k Selection</div>
              <div className="text-xs text-graphite-muted mt-1 font-mono">argpartition + lexsort</div>
            </div>
            <div className="mt-4 text-xs text-graphite-muted border-t border-border pt-2">
              O(N) partial selection with deterministic (distance, ID) ordering.
            </div>
          </div>

          <div className="bg-surface-secondary border border-accent/40 rounded-lg p-4 shadow-subtle flex flex-col justify-between">
            <div>
              <div className="text-[11px] font-mono font-bold text-accent mb-1 tracking-wider">STAGE 05</div>
              <div className="font-heading font-semibold text-sm text-graphite-dark">Ranked Output</div>
              <div className="text-xs text-accent mt-1 font-mono font-semibold">SearchResult(ids, dists)</div>
            </div>
            <div className="mt-4 text-xs text-graphite border-t border-accent/20 pt-2">
              Delivers immutable top-k entity IDs and exact squared distances.
            </div>
          </div>
        </div>

        {/* Technical Callout */}
        <div className="bg-surface-secondary border-l-4 border-accent p-4 rounded-r-lg text-xs leading-relaxed text-graphite">
          <strong className="font-semibold text-graphite-dark">Selection Architecture Notice:</strong> Indexora strictly avoids naive O(N log N) full sorting. It performs O(N) average partial selection via <code>np.argpartition</code> for the top-k candidates, followed by <code>np.lexsort</code> on <code>(distance, external_id)</code> pairs to guarantee deterministic tie-breaking across multi-chunk evaluations.
        </div>
      </section>

      {/* 3. Five Supported Index Families */}
      <section className="space-y-4">
        <h2 className="font-heading font-bold text-lg text-graphite-dark">
          Core Engine Implementations
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Exact BLAS */}
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card hover:border-accent/60 transition space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-heading font-bold text-sm text-graphite-dark">1. Exact BLAS</h3>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-700 font-semibold">
                100% Recall Oracle
              </span>
            </div>
            <p className="text-xs text-graphite-muted leading-relaxed">
              Exhaustive brute-force evaluation reformulating Euclidean distance into matrix multiplication: <code>||q - x||² = ||q||² + ||x||² - 2⟨q, x⟩</code>. Employs precomputed database norms and blocked GEMM kernel execution.
            </p>
            <div className="pt-2 border-t border-border flex items-center justify-between text-xs font-mono text-graphite-muted">
              <span>Time: O(N · d)</span>
              <span>Kernel: GEMM</span>
            </div>
          </div>

          {/* IVF-Flat */}
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card hover:border-accent/60 transition space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-heading font-bold text-sm text-graphite-dark">2. IVF-Flat</h3>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-blue-50 border border-blue-200 text-accent font-semibold">
                Voronoi Partitioning
              </span>
            </div>
            <p className="text-xs text-graphite-muted leading-relaxed">
              Partitions the vector space into <code>nlist</code> centroids via K-Means++ clustering. During query execution, evaluates only the posting lists of the <code>nprobe</code> nearest centroids, reducing candidate distance calculations by 10x–80x.
            </p>
            <div className="pt-2 border-t border-border flex items-center justify-between text-xs font-mono text-graphite-muted">
              <span>Time: O(nprobe · N/nlist · d)</span>
              <span>Storage: Uncompressed</span>
            </div>
          </div>

          {/* IVF-PQ */}
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card hover:border-accent/60 transition space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-heading font-bold text-sm text-graphite-dark">3. IVF-PQ</h3>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-purple-50 border border-purple-200 text-purple-700 font-semibold">
                20.8x Memory Compression
              </span>
            </div>
            <p className="text-xs text-graphite-muted leading-relaxed">
              Decomposes d-dimensional residual vectors into M orthogonal subvectors. Quantizes subvectors into 8-bit codes (k_sub=256) and computes distances via Asymmetric Distance Computation (ADC) lookup tables in M byte additions.
            </p>
            <div className="pt-2 border-t border-border flex items-center justify-between text-xs font-mono text-graphite-muted">
              <span>Time: O(N_cand · M) LUT</span>
              <span>Storage: M bytes/vec</span>
            </div>
          </div>

          {/* HNSW */}
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card hover:border-accent/60 transition space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-heading font-bold text-sm text-graphite-dark">4. HNSW Proximity Graph</h3>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-blue-50 border border-blue-200 text-accent font-semibold">
                Sublinear Graph ANN
              </span>
            </div>
            <p className="text-xs text-graphite-muted leading-relaxed">
              Multi-layer hierarchical proximity graph with exponential level decay and Algorithm 4 diversity neighbor selection. Upper layers act as high-speed expressways; layer 0 executes greedy beam search with exploration depth <code>ef_search</code>.
            </p>
            <div className="pt-2 border-t border-border flex items-center justify-between text-xs font-mono text-graphite-muted">
              <span>Traversal: Empirical O(log N)</span>
              <span>Params: M, ef</span>
            </div>
          </div>

          {/* Exact Flat (Reference) */}
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card hover:border-accent/60 transition space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-heading font-bold text-sm text-graphite-dark">5. Exact Reference Kernel</h3>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-amber-50 border border-amber-200 text-amber-700 font-semibold">
                Float64 Precision Oracle
              </span>
            </div>
            <p className="text-xs text-graphite-muted leading-relaxed">
              Direct coordinate differencing with double-precision float64 accumulator. Eliminates matrix arithmetic rounding and catastrophic cancellation to serve as the benchmark validation baseline.
            </p>
            <div className="pt-2 border-t border-border flex items-center justify-between text-xs font-mono text-graphite-muted">
              <span>Precision: float64 accumulator</span>
              <span>Role: Verification</span>
            </div>
          </div>

          {/* Atomic Persistence */}
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card hover:border-accent/60 transition space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-heading font-bold text-sm text-graphite-dark">6. Sandboxed Persistence</h3>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-700 font-semibold">
                Pickle-Free SHA-256
              </span>
            </div>
            <p className="text-xs text-graphite-muted leading-relaxed">
              Serializes numerical arrays to raw <code>.npy</code> buffers and generates a versioned <code>manifest.json</code> with cryptographic SHA-256 digests. Strictly sandboxed to prevent arbitrary directory traversal.
            </p>
            <div className="pt-2 border-t border-border flex items-center justify-between text-xs font-mono text-graphite-muted">
              <span>Integrity: Streaming SHA-256</span>
              <span>Format: .npy + JSON</span>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Live Dataset Configuration & Sample Preview */}
      <section className="bg-surface-card border border-border rounded-xl p-6 shadow-card space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h2 className="font-heading font-bold text-lg text-graphite-dark">
              Active Evaluation Dataset
            </h2>
            <p className="text-xs text-graphite-muted mt-0.5">
              Deterministic Gaussian cluster distribution in ℝᵈ for algorithm verification.
            </p>
          </div>

          <button
            onClick={() => setShowConfig(!showConfig)}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border bg-surface-subtle hover:bg-surface-secondary text-graphite text-xs font-medium transition cursor-pointer"
          >
            <SlidersHorizontal className="w-3.5 h-3.5 text-accent" />
            <span>{showConfig ? 'Hide Dataset Parameters' : 'Adjust Dataset Parameters'}</span>
          </button>
        </div>

        {/* Dataset Generator Drawer */}
        {showConfig && (
          <form onSubmit={handleRegenerate} className="bg-surface-subtle border border-border rounded-lg p-4 space-y-4 text-xs">
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              <div>
                <label className="block text-graphite-muted mb-1 font-medium">Base Vectors (N)</label>
                <input
                  type="number"
                  value={nBase}
                  onChange={(e) => setNBase(Number(e.target.value))}
                  min={30}
                  max={500}
                  className="w-full px-2.5 py-1.5 rounded border border-border bg-white text-graphite font-mono"
                />
              </div>
              <div>
                <label className="block text-graphite-muted mb-1 font-medium">Dimension (d)</label>
                <input
                  type="number"
                  value={dDim}
                  onChange={(e) => setDDim(Number(e.target.value))}
                  min={4}
                  max={32}
                  step={4}
                  className="w-full px-2.5 py-1.5 rounded border border-border bg-white text-graphite font-mono"
                />
              </div>
              <div>
                <label className="block text-graphite-muted mb-1 font-medium">Clusters</label>
                <input
                  type="number"
                  value={clusters}
                  onChange={(e) => setClusters(Number(e.target.value))}
                  min={2}
                  max={8}
                  className="w-full px-2.5 py-1.5 rounded border border-border bg-white text-graphite font-mono"
                />
              </div>
              <div>
                <label className="block text-graphite-muted mb-1 font-medium">Cluster Std</label>
                <input
                  type="number"
                  value={std}
                  onChange={(e) => setStd(Number(e.target.value))}
                  min={0.1}
                  max={0.8}
                  step={0.05}
                  className="w-full px-2.5 py-1.5 rounded border border-border bg-white text-graphite font-mono"
                />
              </div>
              <div>
                <label className="block text-graphite-muted mb-1 font-medium">RNG Seed</label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  className="w-full px-2.5 py-1.5 rounded border border-border bg-white text-graphite font-mono"
                />
              </div>
            </div>

            <div className="flex items-center justify-between pt-2">
              <span className="text-xs text-graphite-muted">
                Generates stable cluster centroids and SVD PCA projection matrix.
              </span>
              <button
                type="submit"
                disabled={generating}
                className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-accent hover:bg-accent-hover text-white text-xs font-semibold shadow-sm transition disabled:opacity-50 cursor-pointer"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${generating ? 'animate-spin' : ''}`} />
                <span>{generating ? 'Regenerating...' : 'Regenerate Dataset'}</span>
              </button>
            </div>

            {msg && (
              <div className="p-2.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs">
                {msg}
              </div>
            )}
          </form>
        )}

        {/* Live Vector Sample Table */}
        <div className="space-y-2">
          <div className="text-xs font-mono font-semibold text-graphite flex items-center justify-between">
            <span>Database Vector Coordinate Sample (First 8 Vectors)</span>
            <span className="text-graphite-muted font-normal">dtype: float32</span>
          </div>

          {dataset && dataset.base_samples ? (
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-xs font-mono text-left">
                <thead className="bg-surface-secondary text-graphite-muted border-b border-border">
                  <tr>
                    <th className="py-2 px-3 font-semibold">Entity ID</th>
                    <th className="py-2 px-3 font-semibold">Cluster</th>
                    {Array.from({ length: dataset.d }).map((_, i) => (
                      <th key={i} className="py-2 px-3 font-semibold text-right">d{i}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-border bg-white text-graphite">
                  {dataset.base_samples.slice(0, 8).map((sample) => (
                    <tr key={sample.id} className="hover:bg-surface-subtle">
                      <td className="py-1.5 px-3 font-bold text-accent">{sample.id}</td>
                      <td className="py-1.5 px-3 text-graphite-muted">Cluster {sample.cluster}</td>
                      {sample.vector.map((val, idx) => (
                        <td key={idx} className="py-1.5 px-3 text-right text-graphite-muted">
                          {val.toFixed(3)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="p-4 text-center text-xs text-graphite-muted">Loading vector table...</div>
          )}
        </div>
      </section>
    </div>
  );
};

