import React, { useState } from 'react';
import {
  Cpu,
  HardDrive,
  Zap,
  Target,
  Check,
  AlertCircle,
  Sparkles,
  Sliders
} from 'lucide-react';

export const HardwareMatrix: React.FC = () => {
  const [activeConstraint, setActiveConstraint] = useState<'latency' | 'memory' | 'recall' | 'mutations'>('latency');

  const algorithms = [
    {
      name: 'Exact BLAS',
      type: 'Oracle',
      flops: 'O(N · d)',
      memory: '51.2 MB (Raw)',
      locality: 'Sequential SIMD Stream',
      mutations: 'Array Compaction',
      optimalFor: ['recall'],
      score: { latency: 72, memory: 55, recall: 100, mutations: 60 },
      pros: '100% mathematical ground-truth guarantee, zero build latency',
      cons: 'Exhaustive linear scaling with N; memory-bandwidth bound at scale',
    },
    {
      name: 'IVF-Flat',
      type: 'Partitioned',
      flops: 'O(nprobe · N / nlist · d)',
      memory: '51.4 MB (Raw + Inverted Lists)',
      locality: 'Segmented Inverted Posts',
      mutations: 'Posting List Append / Prune',
      optimalFor: ['mutations'],
      score: { latency: 85, memory: 54, recall: 96, mutations: 90 },
      pros: 'Prunes 75–90% of candidate comparisons, simple posting list maintenance',
      cons: 'Suffers from Voronoi boundary effects; requires K-Means re-clustering',
    },
    {
      name: 'IVF-PQ (Product Quant.)',
      type: 'Compressed',
      flops: 'O(M · 256 + N_cand · M)',
      memory: '2.5 MB (~20.8x Compression)',
      locality: 'L1/L2 Lookup Table Cache',
      mutations: 'Codebook Alignment',
      optimalFor: ['memory'],
      score: { latency: 88, memory: 98, recall: 82, mutations: 70 },
      pros: '95.2% RAM savings; turns expensive dot products into CPU cache table additions',
      cons: 'Lossy vector reconstruction introduces quantization noise',
    },
    {
      name: 'HNSW Graph',
      type: 'Graph',
      flops: 'O(log N · M · d)',
      memory: '64.0 MB (Adjacency Lists)',
      locality: 'Pointer-Chase Graph Hops',
      mutations: 'Soft Tombstone Flags',
      optimalFor: ['latency'],
      score: { latency: 98, memory: 40, recall: 98, mutations: 50 },
      pros: 'Sub-millisecond latency at >98% recall; logarithmic skip-list traversal',
      cons: 'Pointer chasing degrades L1/L2 cache hit rate; higher memory overhead',
    },
  ];

  return (
    <div className="bg-surface-card border border-border rounded-xl p-5 sm:p-7 shadow-card space-y-6 transition-colors font-sans">
      {/* Title & Constraint Switcher */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border pb-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-accent/10 border border-accent/20 text-accent font-mono text-[11px] font-semibold uppercase tracking-wider mb-1.5">
            <Sliders className="w-3.5 h-3.5" />
            <span>Engineering Systems Decision Matrix</span>
          </div>
          <h2 className="font-heading font-bold text-lg sm:text-xl text-graphite-dark">
            Hardware & Cache Telemetry Matrix
          </h2>
          <p className="text-xs text-graphite-muted mt-0.5">
            Compare CPU cache locality, memory bandwidth, SIMD utilization, and mechanical trade-offs.
          </p>
        </div>

        {/* Constraint Filter Pills */}
        <div className="flex items-center gap-1.5 bg-surface-subtle p-1 rounded-lg border border-border overflow-x-auto text-xs font-mono scrollbar-none">
          <button
            onClick={() => setActiveConstraint('latency')}
            className={`px-3 py-1.5 rounded-md transition-all cursor-pointer whitespace-nowrap flex items-center gap-1.5 ${
              activeConstraint === 'latency'
                ? 'bg-accent text-white font-semibold shadow-sm'
                : 'text-graphite-muted hover:text-graphite'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            <span>Latency Critical</span>
          </button>

          <button
            onClick={() => setActiveConstraint('memory')}
            className={`px-3 py-1.5 rounded-md transition-all cursor-pointer whitespace-nowrap flex items-center gap-1.5 ${
              activeConstraint === 'memory'
                ? 'bg-accent text-white font-semibold shadow-sm'
                : 'text-graphite-muted hover:text-graphite'
            }`}
          >
            <HardDrive className="w-3.5 h-3.5" />
            <span>RAM Constrained</span>
          </button>

          <button
            onClick={() => setActiveConstraint('recall')}
            className={`px-3 py-1.5 rounded-md transition-all cursor-pointer whitespace-nowrap flex items-center gap-1.5 ${
              activeConstraint === 'recall'
                ? 'bg-accent text-white font-semibold shadow-sm'
                : 'text-graphite-muted hover:text-graphite'
            }`}
          >
            <Target className="w-3.5 h-3.5" />
            <span>100% Precision Oracle</span>
          </button>
        </div>
      </div>

      {/* Matrix Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {algorithms.map((algo) => {
          const isWinner = algo.optimalFor.includes(activeConstraint);
          const score = algo.score[activeConstraint];

          return (
            <div
              key={algo.name}
              className={`p-4 rounded-xl border transition-all duration-200 flex flex-col justify-between space-y-4 ${
                isWinner
                  ? 'bg-surface-secondary/70 border-accent shadow-glow ring-1 ring-accent/30'
                  : 'bg-surface-subtle border-border opacity-90 hover:opacity-100'
              }`}
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-heading font-bold text-sm text-graphite-dark">
                    {algo.name}
                  </span>
                  {isWinner ? (
                    <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 font-mono text-[10px] font-bold flex items-center gap-1">
                      <Sparkles className="w-3 h-3" /> Recommended
                    </span>
                  ) : (
                    <span className="px-1.5 py-0.5 rounded bg-surface-card border border-border font-mono text-[10px] text-graphite-muted">
                      {algo.type}
                    </span>
                  )}
                </div>

                {/* Constraint Match Meter */}
                <div className="space-y-1">
                  <div className="flex justify-between text-[11px] font-mono">
                    <span className="text-graphite-muted">Suitability Score</span>
                    <span className={`font-bold ${isWinner ? 'text-accent' : 'text-graphite'}`}>
                      {score}%
                    </span>
                  </div>
                  <div className="w-full h-1.5 rounded-full bg-surface-card border border-border overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        isWinner ? 'bg-accent' : 'bg-graphite-subtle'
                      }`}
                      style={{ width: `${score}%` }}
                    />
                  </div>
                </div>

                {/* Specs List */}
                <div className="space-y-2 pt-2 border-t border-border/70 text-xs font-mono">
                  <div className="flex items-center justify-between">
                    <span className="text-graphite-subtle text-[10px]">FLOPs:</span>
                    <span className="text-graphite text-[11px] truncate max-w-[140px] text-right font-medium">
                      {algo.flops}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-graphite-subtle text-[10px]">RAM Footprint:</span>
                    <span className="text-graphite text-[11px] truncate max-w-[140px] text-right font-medium">
                      {algo.memory}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-graphite-subtle text-[10px]">Cache Access:</span>
                    <span className="text-graphite text-[11px] truncate max-w-[140px] text-right font-medium">
                      {algo.locality}
                    </span>
                  </div>
                </div>
              </div>

              {/* Bottom Trade-Off Note */}
              <div className="pt-3 border-t border-border/70 text-[11px] text-graphite-muted leading-relaxed">
                <span className="text-graphite font-semibold block mb-0.5">Architecture Takeaway:</span>
                {isWinner ? algo.pros : algo.cons}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
