import React, { useState } from 'react';
import { Layers, Network, Grid3X3, Cpu, Sparkles, ArrowRight, Eye, ShieldCheck } from 'lucide-react';

export const ArchitectureShowpiece: React.FC = () => {
  const [activeMode, setActiveMode] = useState<'hnsw' | 'ivf' | 'pq' | 'blas'>('hnsw');
  const [ivfProbe, setIvfProbe] = useState<number>(2);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  return (
    <div className="bg-surface-card border border-border rounded-xl p-5 sm:p-7 shadow-card space-y-6 transition-colors">
      {/* Showpiece Title Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border pb-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-accent/10 border border-accent/20 text-accent font-mono text-[11px] font-semibold uppercase tracking-wider mb-1.5">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Interactive Algorithmic Showpiece</span>
          </div>
          <h2 className="font-heading font-bold text-lg sm:text-xl text-graphite-dark">
            Vector Search Geometry & Kernel Visualizer
          </h2>
          <p className="text-xs text-graphite-muted mt-0.5">
            Explore how Indexora partitions vector space, routes graph queries, and compresses high-dimensional floating-point tensors.
          </p>
        </div>

        {/* Algorithm Switcher Tabs */}
        <div className="inline-flex p-1 rounded-lg bg-surface-subtle border border-border self-start sm:self-auto overflow-x-auto text-xs font-mono scrollbar-none">
          <button
            onClick={() => setActiveMode('hnsw')}
            className={`px-3 py-1.5 rounded-md font-medium transition-all cursor-pointer whitespace-nowrap flex items-center gap-1.5 ${
              activeMode === 'hnsw'
                ? 'bg-surface-card text-accent font-semibold shadow-subtle border border-border/80'
                : 'text-graphite-muted hover:text-graphite'
            }`}
          >
            <Network className="w-3.5 h-3.5" />
            <span>HNSW Graph</span>
          </button>

          <button
            onClick={() => setActiveMode('ivf')}
            className={`px-3 py-1.5 rounded-md font-medium transition-all cursor-pointer whitespace-nowrap flex items-center gap-1.5 ${
              activeMode === 'ivf'
                ? 'bg-surface-card text-accent font-semibold shadow-subtle border border-border/80'
                : 'text-graphite-muted hover:text-graphite'
            }`}
          >
            <Grid3X3 className="w-3.5 h-3.5" />
            <span>IVF Voronoi</span>
          </button>

          <button
            onClick={() => setActiveMode('pq')}
            className={`px-3 py-1.5 rounded-md font-medium transition-all cursor-pointer whitespace-nowrap flex items-center gap-1.5 ${
              activeMode === 'pq'
                ? 'bg-surface-card text-accent font-semibold shadow-subtle border border-border/80'
                : 'text-graphite-muted hover:text-graphite'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>IVF-PQ Quantization</span>
          </button>

          <button
            onClick={() => setActiveMode('blas')}
            className={`px-3 py-1.5 rounded-md font-medium transition-all cursor-pointer whitespace-nowrap flex items-center gap-1.5 ${
              activeMode === 'blas'
                ? 'bg-surface-card text-accent font-semibold shadow-subtle border border-border/80'
                : 'text-graphite-muted hover:text-graphite'
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>BLAS GEMM</span>
          </button>
        </div>
      </div>

      {/* Interactive Visual Canvas Area */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        {/* SVG Diagram Canvas (8 cols) */}
        <div className="lg:col-span-8 bg-surface-subtle border border-border rounded-xl p-4 sm:p-6 flex flex-col items-center justify-center relative overflow-hidden min-h-[340px]">
          {/* MODE 1: HNSW HIERARCHICAL GRAPH */}
          {activeMode === 'hnsw' && (
            <svg viewBox="0 0 620 280" className="w-full h-auto max-w-[580px]">
              {/* Layer 2: Sparse Highway */}
              <g className="transition-opacity">
                <rect x="30" y="20" width="560" height="55" rx="8" fill="currentColor" className="text-surface-secondary/60 dark:text-surface-secondary/40" />
                <text x="45" y="42" fill="var(--text-graphite-muted)" fontSize="11" fontFamily="JetBrains Mono" fontWeight="bold">Layer 2 (Express Highway, M=2)</text>
                <line x1="120" y1="52" x2="380" y2="52" stroke="var(--accent-primary)" strokeWidth="2.5" strokeDasharray="4 4" className="animate-pulse" />
                <circle cx="120" cy="52" r="7" fill="var(--accent-primary)" stroke="#FFFFFF" strokeWidth="2" />
                <text x="105" y="73" fill="var(--text-graphite)" fontSize="9" fontFamily="JetBrains Mono">Entry Ep</text>
                <circle cx="380" cy="52" r="7" fill="var(--accent-primary)" stroke="#FFFFFF" strokeWidth="2" />
                <text x="370" y="73" fill="var(--text-graphite)" fontSize="9" fontFamily="JetBrains Mono">Hop 1</text>
                {/* Descent arrow */}
                <path d="M380 59 L380 95" stroke="var(--accent-primary)" strokeWidth="2" strokeDasharray="3 3" markerEnd="url(#arrow)" />
              </g>

              {/* Layer 1: Intermediate Routing */}
              <g className="transition-opacity">
                <rect x="30" y="105" width="560" height="65" rx="8" fill="currentColor" className="text-surface-secondary/60 dark:text-surface-secondary/40" />
                <text x="45" y="127" fill="var(--text-graphite-muted)" fontSize="11" fontFamily="JetBrains Mono" fontWeight="bold">Layer 1 (Density Routing, M=8)</text>
                <line x1="180" y1="145" x2="380" y2="145" stroke="#94A3B8" strokeWidth="1.5" />
                <line x1="380" y1="145" x2="480" y2="145" stroke="var(--accent-primary)" strokeWidth="2.5" strokeDasharray="4 4" />
                <circle cx="180" cy="145" r="5" fill="#94A3B8" />
                <circle cx="380" cy="145" r="6" fill="var(--accent-primary)" stroke="#FFFFFF" strokeWidth="1.5" />
                <circle cx="480" cy="145" r="6" fill="var(--accent-primary)" stroke="#FFFFFF" strokeWidth="1.5" />
                <text x="468" y="165" fill="var(--text-graphite)" fontSize="9" fontFamily="JetBrains Mono">Hop 2</text>
                {/* Descent arrow */}
                <path d="M480 151 L480 188" stroke="var(--accent-primary)" strokeWidth="2" strokeDasharray="3 3" />
              </g>

              {/* Layer 0: Ground Dense Network */}
              <g className="transition-opacity">
                <rect x="30" y="195" width="560" height="75" rx="8" fill="currentColor" className="text-surface-secondary/60 dark:text-surface-secondary/40" />
                <text x="45" y="215" fill="var(--text-graphite-muted)" fontSize="11" fontFamily="JetBrains Mono" fontWeight="bold">Layer 0 (Ground Truth Graph, M=16, efSearch=32)</text>

                {/* Connectors */}
                <line x1="140" y1="245" x2="240" y2="245" stroke="#94A3B8" strokeWidth="1" />
                <line x1="240" y1="245" x2="360" y2="245" stroke="#94A3B8" strokeWidth="1" />
                <line x1="360" y1="245" x2="480" y2="245" stroke="#94A3B8" strokeWidth="1" />
                <line x1="480" y1="245" x2="540" y2="238" stroke="var(--accent-primary)" strokeWidth="2.5" />

                <circle cx="140" cy="245" r="4.5" fill="#94A3B8" />
                <circle cx="240" cy="245" r="4.5" fill="#94A3B8" />
                <circle cx="360" cy="245" r="4.5" fill="#94A3B8" />
                <circle cx="480" cy="245" r="5.5" fill="var(--accent-primary)" stroke="#FFFFFF" strokeWidth="1.5" />

                {/* Nearest Neighbor Target */}
                <circle cx="540" cy="238" r="8" fill="#10B981" stroke="#FFFFFF" strokeWidth="2" className="animate-pulse" />
                <text x="505" y="260" fill="#10B981" fontSize="10" fontFamily="JetBrains Mono" fontWeight="bold">Nearest Top-1</text>
              </g>
            </svg>
          )}

          {/* MODE 2: IVF VORONOI CELLS */}
          {activeMode === 'ivf' && (
            <div className="w-full flex flex-col items-center space-y-3">
              <svg viewBox="0 0 540 240" className="w-full h-auto max-w-[500px]">
                {/* 4 Voronoi Cells */}
                {/* Cell 0 (Top Left) */}
                <path
                  d="M 40,30 L 260,30 L 230,120 L 40,110 Z"
                  fill={ivfProbe >= 1 ? 'rgba(59, 130, 246, 0.15)' : 'currentColor'}
                  stroke="var(--border-color)"
                  strokeWidth="1.5"
                  className={ivfProbe >= 1 ? 'text-blue-500' : 'text-surface-secondary/40'}
                />
                <circle cx="140" cy="70" r="5" fill="var(--accent-primary)" />
                <text x="125" y="90" fill="var(--text-graphite)" fontSize="10" fontFamily="JetBrains Mono">C₀ (Active)</text>

                {/* Cell 1 (Top Right) */}
                <path
                  d="M 260,30 L 500,30 L 500,120 L 230,120 Z"
                  fill={ivfProbe >= 2 ? 'rgba(59, 130, 246, 0.12)' : 'currentColor'}
                  stroke="var(--border-color)"
                  strokeWidth="1.5"
                  className={ivfProbe >= 2 ? 'text-blue-500' : 'text-surface-secondary/40'}
                />
                <circle cx="360" cy="70" r="5" fill={ivfProbe >= 2 ? 'var(--accent-primary)' : '#94A3B8'} />
                <text x="340" y="90" fill="var(--text-graphite)" fontSize="10" fontFamily="JetBrains Mono">
                  C₁ {ivfProbe >= 2 ? '(Probed)' : '(Pruned)'}
                </text>

                {/* Cell 2 (Bottom Left) */}
                <path
                  d="M 40,110 L 230,120 L 270,220 L 40,220 Z"
                  fill={ivfProbe >= 3 ? 'rgba(59, 130, 246, 0.12)' : 'currentColor'}
                  stroke="var(--border-color)"
                  strokeWidth="1.5"
                  className={ivfProbe >= 3 ? 'text-blue-500' : 'text-surface-secondary/40'}
                />
                <circle cx="140" cy="170" r="5" fill={ivfProbe >= 3 ? 'var(--accent-primary)' : '#94A3B8'} />
                <text x="125" y="190" fill="var(--text-graphite)" fontSize="10" fontFamily="JetBrains Mono">
                  C₂ {ivfProbe >= 3 ? '(Probed)' : '(Pruned)'}
                </text>

                {/* Cell 3 (Bottom Right) */}
                <path
                  d="M 230,120 L 500,120 L 500,220 L 270,220 Z"
                  fill="currentColor"
                  stroke="var(--border-color)"
                  strokeWidth="1.5"
                  className="text-surface-secondary/40"
                />
                <circle cx="380" cy="170" r="5" fill="#94A3B8" />
                <text x="360" y="190" fill="var(--text-graphite-muted)" fontSize="10" fontFamily="JetBrains Mono">C₃ (Pruned)</text>

                {/* Query Vector Star */}
                <circle cx="190" cy="95" r="7" fill="#F59E0B" stroke="#FFFFFF" strokeWidth="2" className="animate-pulse" />
                <text x="202" y="99" fill="#F59E0B" fontSize="11" fontFamily="JetBrains Mono" fontWeight="bold">q (Query)</text>
              </svg>

              {/* Probe Slider inside canvas */}
              <div className="flex items-center gap-3 bg-surface-card px-4 py-2 rounded-lg border border-border text-xs font-mono">
                <span className="text-graphite-muted">nprobe:</span>
                <input
                  type="range"
                  min="1"
                  max="3"
                  value={ivfProbe}
                  onChange={(e) => setIvfProbe(Number(e.target.value))}
                  className="w-32 accent-accent cursor-pointer"
                />
                <span className="text-accent font-bold">{ivfProbe} / 4 cells</span>
                <span className="text-emerald-500 font-semibold">({((4 - ivfProbe) / 4 * 100).toFixed(0)}% Work Saved)</span>
              </div>
            </div>
          )}

          {/* MODE 3: PRODUCT QUANTIZATION (ADC) */}
          {activeMode === 'pq' && (
            <div className="w-full flex flex-col items-center space-y-4">
              <svg viewBox="0 0 540 200" className="w-full h-auto max-w-[500px]">
                {/* 128-d Vector Decomposition */}
                <text x="30" y="25" fill="var(--text-graphite)" fontSize="11" fontFamily="JetBrains Mono" fontWeight="bold">
                  Raw Vector x ∈ ℝ¹²⁸ (512 bytes float32)
                </text>
                <rect x="30" y="35" width="480" height="25" rx="5" fill="currentColor" stroke="var(--border-color)" className="text-surface-secondary" />
                <line x1="150" y1="35" x2="150" y2="60" stroke="var(--border-color)" strokeWidth="1.5" />
                <line x1="270" y1="35" x2="270" y2="60" stroke="var(--border-color)" strokeWidth="1.5" />
                <line x1="390" y1="35" x2="390" y2="60" stroke="var(--border-color)" strokeWidth="1.5" />

                <text x="75" y="52" fill="var(--text-graphite-muted)" fontSize="9" fontFamily="JetBrains Mono">Subspace 1 (32d)</text>
                <text x="195" y="52" fill="var(--text-graphite-muted)" fontSize="9" fontFamily="JetBrains Mono">Subspace 2 (32d)</text>
                <text x="315" y="52" fill="var(--text-graphite-muted)" fontSize="9" fontFamily="JetBrains Mono">Subspace 3 (32d)</text>
                <text x="435" y="52" fill="var(--text-graphite-muted)" fontSize="9" fontFamily="JetBrains Mono">Subspace 4 (32d)</text>

                {/* Arrow down to 8-bit quantized codes */}
                <path d="M 270 70 L 270 95" stroke="var(--accent-primary)" strokeWidth="2" strokeDasharray="3 3" />
                <text x="285" y="87" fill="var(--accent-primary)" fontSize="10" fontFamily="JetBrains Mono">K-Means Residual Vector Quantization</text>

                {/* Quantized 4-byte byte code */}
                <text x="30" y="115" fill="var(--text-graphite)" fontSize="11" fontFamily="JetBrains Mono" fontWeight="bold">
                  Compact 8-bit Byte Array (Only 4 bytes total — 128x Compression)
                </text>
                <rect x="30" y="125" width="115" height="30" rx="4" fill="rgba(59, 130, 246, 0.15)" stroke="var(--accent-primary)" />
                <text x="65" y="145" fill="var(--accent-primary)" fontSize="11" fontFamily="JetBrains Mono" fontWeight="bold">[ 42 ]</text>

                <rect x="150" y="125" width="115" height="30" rx="4" fill="rgba(59, 130, 246, 0.15)" stroke="var(--accent-primary)" />
                <text x="185" y="145" fill="var(--accent-primary)" fontSize="11" fontFamily="JetBrains Mono" fontWeight="bold">[ 189 ]</text>

                <rect x="270" y="125" width="115" height="30" rx="4" fill="rgba(59, 130, 246, 0.15)" stroke="var(--accent-primary)" />
                <text x="305" y="145" fill="var(--accent-primary)" fontSize="11" fontFamily="JetBrains Mono" fontWeight="bold">[ 77 ]</text>

                <rect x="390" y="125" width="120" height="30" rx="4" fill="rgba(59, 130, 246, 0.15)" stroke="var(--accent-primary)" />
                <text x="425" y="145" fill="var(--accent-primary)" fontSize="11" fontFamily="JetBrains Mono" fontWeight="bold">[ 251 ]</text>

                <text x="30" y="185" fill="var(--text-graphite-muted)" fontSize="10" fontFamily="JetBrains Mono">
                  Distance: LUT[0, 42] + LUT[1, 189] + LUT[2, 77] + LUT[3, 251] (O(M) Cache-line lookups)
                </text>
              </svg>
            </div>
          )}

          {/* MODE 4: BLAS GEMM VECTORIZATION */}
          {activeMode === 'blas' && (
            <div className="w-full flex flex-col items-center space-y-4 font-mono text-xs">
              <div className="p-3.5 rounded-lg bg-surface-card border border-border text-center max-w-md w-full">
                <span className="text-graphite-muted text-[10px] block mb-1">Euclidean Expansion Formulation</span>
                <span className="text-accent font-bold text-sm block">
                  ||q - x||² = ||q||² + ||x||² - 2 · ⟨q, x⟩
                </span>
              </div>

              <div className="grid grid-cols-3 gap-3 w-full max-w-lg text-center">
                <div className="p-3 rounded-lg bg-surface-card border border-border">
                  <span className="text-[10px] text-graphite-muted block">Query Norm</span>
                  <span className="text-graphite font-bold">||q||²</span>
                  <span className="text-[10px] text-emerald-500 block mt-1">O(d) once</span>
                </div>
                <div className="p-3 rounded-lg bg-surface-card border border-border">
                  <span className="text-[10px] text-graphite-muted block">Database Norms</span>
                  <span className="text-graphite font-bold">||x||²</span>
                  <span className="text-[10px] text-emerald-500 block mt-1">Precomputed</span>
                </div>
                <div className="p-3 rounded-lg bg-surface-card border border-accent/40 bg-accent/5">
                  <span className="text-[10px] text-accent block font-semibold">Matrix Kernel</span>
                  <span className="text-accent font-bold">q · Xᵀ</span>
                  <span className="text-[10px] text-accent block mt-1">Level-3 GEMM</span>
                </div>
              </div>

              <span className="text-[11px] text-graphite-muted text-center max-w-md">
                Transforms N individual vector distance evaluations into single multi-threaded SIMD matrix multiplication with L1/L2 cache blocking.
              </span>
            </div>
          )}
        </div>

        {/* Dynamic Detail Card on the Right (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-surface-subtle p-5 rounded-xl border border-border space-y-3">
            <span className="text-xs font-mono font-bold text-accent uppercase tracking-wider block">
              {activeMode === 'hnsw' && 'Hierarchical Proximity Graphs'}
              {activeMode === 'ivf' && 'Inverted Voronoi Partitioning'}
              {activeMode === 'pq' && 'Subspace Vector Quantization'}
              {activeMode === 'blas' && 'BLAS Level-3 GEMM Kernel'}
            </span>

            <h3 className="font-heading font-bold text-sm text-graphite-dark">
              {activeMode === 'hnsw' && 'O(log N) Greedy Hop Traversal'}
              {activeMode === 'ivf' && 'Voronoi Pruning & Centroid Assignment'}
              {activeMode === 'pq' && '20.8x Memory Reduction via ADC'}
              {activeMode === 'blas' && 'Hardware-Accelerated Exact Oracle'}
            </h3>

            <p className="text-xs text-graphite leading-relaxed">
              {activeMode === 'hnsw' &&
                'Constructs multi-layer proximity graphs with exponentially declining density. Traversal begins at top layer expressways and greedily drops down layers toward nearest neighbors.'}
              {activeMode === 'ivf' &&
                'Clusters base vectors into K centroids using K-Means++. Queries evaluate distances only against vectors in the nprobe closest Voronoi posting lists.'}
              {activeMode === 'pq' &&
                'Splits d-dimensional space into M orthogonal sub-vectors. Compresses 32-bit floats into 8-bit codebook indices and computes distances using precalculated tables.'}
              {activeMode === 'blas' &&
                'Avoids slow nested loops by algebraically mapping distance calculations to hardware-optimized Level-3 GEMM matrix kernels with deterministic tie-breaking.'}
            </p>

            <div className="pt-2 border-t border-border flex items-center justify-between text-xs font-mono">
              <span className="text-graphite-muted">Complexity:</span>
              <span className="font-bold text-graphite">
                {activeMode === 'hnsw' && 'O(log N)'}
                {activeMode === 'ivf' && 'O(nprobe · N / nlist)'}
                {activeMode === 'pq' && 'O(M · 256 + N_candidates · M)'}
                {activeMode === 'blas' && 'O(N · d)'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
