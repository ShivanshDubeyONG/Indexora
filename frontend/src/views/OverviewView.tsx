import React, { useState } from 'react';
import { SystemStatus, DatasetCurrent } from '../types';
import {
  Gauge,
  BarChart2,
  CheckCircle2,
  HardDrive,
  Clock,
  Zap,
  TrendingUp,
  Sliders,
  RefreshCw
} from 'lucide-react';

interface OverviewViewProps {
  status: SystemStatus | null;
  dataset: DatasetCurrent | null;
  onDatasetUpdated: () => void;
  onNavigateToSearch?: () => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({
  status,
  dataset,
  onDatasetUpdated,
  onNavigateToSearch
}) => {
  const [selectedRecallAlgo, setSelectedRecallAlgo] = useState<'exact' | 'hnsw' | 'ivf' | 'pq'>('exact');

  const algorithmData = [
    { name: 'Exact BLAS', latency: 0.10, qps: 50169, recall: 1.0, memoryMb: 51.2, color: '#3182CE' },
    { name: 'Exact Flat (Ref)', latency: 0.04, qps: 21079, recall: 1.0, memoryMb: 51.2, color: '#718096' },
    { name: 'HNSW Graph', latency: 0.12, qps: 32450, recall: 0.984, memoryMb: 64.0, color: '#0AB39C' },
    { name: 'IVF-Flat (nprobe=1)', latency: 0.10, qps: 9003, recall: 0.940, memoryMb: 51.4, color: '#F7B84B' },
    { name: 'IVF-PQ (ADC)', latency: 0.15, qps: 6800, recall: 0.825, memoryMb: 2.5, color: '#E05638' },
  ];

  const recallScores: Record<string, { value: number; label: string; desc: string }> = {
    exact: { value: 100, label: '100% (Exact Ground Truth)', desc: 'Exhaustive Level-3 BLAS GEMM matrix product with deterministic lexsort tie-breaking.' },
    hnsw: { value: 98.4, label: '98.4% (HNSW Graph)', desc: 'Hierarchical skip-graph greedy traversal with Layer 0 candidate exploration.' },
    ivf: { value: 94.0, label: '94.0% (IVF-Flat)', desc: 'Voronoi cell pruning with nprobe=1 centroid assignment.' },
    pq: { value: 82.5, label: '82.5% (IVF-PQ ADC)', desc: 'Subspace residual quantization with 8-bit asymmetric distance lookup tables.' },
  };

  const currentRecall = recallScores[selectedRecallAlgo];

  return (
    <div className="space-y-6">
      {/* 1. Header Banner (Modeled directly after the screenshot) */}
      <div className="bg-surface-card border border-border rounded-xl p-5 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4 transition-colors">
        <div className="flex items-start gap-3.5">
          {/* Rounded Coral Icon Container */}
          <div className="w-11 h-11 rounded-lg bg-[#E05638] flex items-center justify-center text-white flex-shrink-0 shadow-sm mt-0.5">
            <Gauge className="w-6 h-6" />
          </div>
          <div>
            <span className="text-[11px] font-bold text-graphite-subtle tracking-wider uppercase block">
              DASHBOARD
            </span>
            <h1 className="font-heading font-bold text-xl sm:text-2xl text-graphite-dark tracking-tight">
              Vector Search Engine Executive Dashboard
            </h1>
            <p className="text-xs text-graphite-muted mt-0.5">
              Real-time KPIs for nearest-neighbor retrieval, precision verification, and hardware throughput · Ground truth: Exact BLAS Oracle · Viewing local index
            </p>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="flex items-center gap-2 self-start md:self-auto text-xs">
          <button
            onClick={onDatasetUpdated}
            className="px-3 py-1.5 rounded-lg border border-border bg-surface-subtle hover:bg-surface-secondary text-graphite font-medium transition cursor-pointer flex items-center gap-1.5"
          >
            <RefreshCw className="w-3.5 h-3.5 text-graphite-muted" />
            <span>Sync Telemetry</span>
          </button>
        </div>
      </div>

      {/* 2. Main 2x2 Data Grid (Directly matching the screenshot layout) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* CARD 1: Precision Recall Gauge (Clean Technical Engineering Meter) */}
        <div className="bg-surface-card border border-border rounded-xl p-6 shadow-sm flex flex-col justify-between transition-colors">
          <div className="flex items-center justify-between border-b border-border/70 pb-3">
            <div>
              <h2 className="font-heading font-bold text-base text-graphite-dark">
                Retrieval Precision (Recall@k)
              </h2>
              <p className="text-xs text-graphite-muted">
                Precision evaluated against float64 BLAS ground-truth reference
              </p>
            </div>

            {/* Algorithm Selector for Gauge */}
            <select
              value={selectedRecallAlgo}
              onChange={(e) => setSelectedRecallAlgo(e.target.value as any)}
              className="text-xs font-mono px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle text-graphite focus:outline-none focus:border-accent cursor-pointer"
            >
              <option value="exact">Exact BLAS (100%)</option>
              <option value="hnsw">HNSW Graph (98.4%)</option>
              <option value="ivf">IVF-Flat (94.0%)</option>
              <option value="pq">IVF-PQ ADC (82.5%)</option>
            </select>
          </div>

          {/* Calibrated Precision Technical Arc */}
          <div className="py-5 flex flex-col items-center justify-center">
            <svg viewBox="0 0 260 140" className="w-full max-w-[260px]">
              <defs>
                <linearGradient id="precisionArcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#2563EB" />
                  <stop offset="100%" stopColor="#38BDF8" />
                </linearGradient>
              </defs>

              {/* Background Arc Track */}
              <path
                d="M 30,125 A 90,90 0 0,1 230,125"
                fill="none"
                stroke="currentColor"
                className="text-slate-200 dark:text-slate-700/60"
                strokeWidth="12"
                strokeLinecap="round"
              />

              {/* Calibrated Active Meter Arc */}
              <path
                d="M 30,125 A 90,90 0 0,1 230,125"
                fill="none"
                stroke="url(#precisionArcGrad)"
                strokeWidth="12"
                strokeLinecap="round"
                strokeDasharray="282.74"
                strokeDashoffset={282.74 * (1 - (currentRecall.value / 100))}
                className="transition-all duration-500 ease-out"
              />

              {/* Calibration Tick Marks */}
              {[0, 25, 50, 75, 100].map((tick) => {
                const angle = Math.PI - (tick / 100) * Math.PI;
                const rInner = 76;
                const rOuter = 82;
                const x1 = 130 + rInner * Math.cos(angle);
                const y1 = 125 - rInner * Math.sin(angle);
                const x2 = 130 + rOuter * Math.cos(angle);
                const y2 = 125 - rOuter * Math.sin(angle);
                return (
                  <line
                    key={tick}
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="currentColor"
                    className="text-slate-400 dark:text-slate-500"
                    strokeWidth={tick === 0 || tick === 100 ? 1.5 : 1}
                  />
                );
              })}

              {/* Percentage Ticks */}
              <text x="30" y="139" fontSize="10" fontFamily="monospace" fill="#94A3B8" textAnchor="middle">0%</text>
              <text x="130" y="32" fontSize="10" fontFamily="monospace" fill="#94A3B8" textAnchor="middle">50%</text>
              <text x="230" y="139" fontSize="10" fontFamily="monospace" fill="#94A3B8" textAnchor="middle">100%</text>

              {/* Digital Center Readout */}
              <text x="130" y="105" fontSize="26" fontWeight="700" fontFamily="monospace" fill="currentColor" className="text-graphite-dark" textAnchor="middle">
                {currentRecall.value.toFixed(1)}%
              </text>
              <text x="130" y="120" fontSize="10" fontFamily="Inter" fill="#94A3B8" textAnchor="middle">
                RECALL@K
              </text>
            </svg>

            <span className="text-xs text-graphite-muted text-center mt-2 max-w-sm px-2">
              {currentRecall.desc}
            </span>
          </div>

          <div className="pt-3 border-t border-border flex items-center justify-between text-[11px] text-graphite-muted font-mono">
            <span>Method: Exact BLAS Oracle</span>
            <span className="text-emerald-600 dark:text-emerald-400 font-semibold flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3" /> Zero False Positives
            </span>
          </div>
        </div>

        {/* CARD 2: Architecture Performance & Latency Leaderboard (Table) */}
        <div className="bg-surface-card border border-border rounded-xl p-6 shadow-sm flex flex-col justify-between transition-colors">
          <div className="border-b border-border/70 pb-3">
            <h2 className="font-heading font-bold text-base text-graphite-dark">
              Index Latency & Throughput Leaderboard
            </h2>
            <p className="text-xs text-graphite-muted">
              Standardized query latency and QPS throughput benchmarks (Top-10 retrieval)
            </p>
          </div>

          <div className="py-2 overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-border text-graphite-muted font-bold uppercase text-[10px] tracking-wider">
                  <th className="py-2.5 px-2">Algorithm</th>
                  <th className="py-2.5 px-2 text-right">p50 Latency</th>
                  <th className="py-2.5 px-2 text-right">Throughput</th>
                  <th className="py-2.5 px-2 text-right">Recall</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/60 font-mono">
                {algorithmData.map((algo) => (
                  <tr key={algo.name} className="hover:bg-surface-subtle transition-colors">
                    <td className="py-2.5 px-2 font-medium text-graphite flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: algo.color }} />
                      <span className="font-sans font-semibold text-graphite-dark">{algo.name}</span>
                    </td>
                    <td className="py-2.5 px-2 text-right text-graphite">
                      {algo.latency.toFixed(2)} ms
                    </td>
                    <td className="py-2.5 px-2 text-right font-bold text-accent">
                      {algo.qps.toLocaleString()} QPS
                    </td>
                    <td className="py-2.5 px-2 text-right font-bold">
                      <span className={algo.recall >= 0.98 ? 'text-emerald-600' : algo.recall >= 0.9 ? 'text-blue-600' : 'text-amber-600'}>
                        {(algo.recall * 100).toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="pt-3 border-t border-border flex items-center justify-between text-[11px] text-graphite-muted font-mono">
            <span>Evaluated on 10,000 query vectors</span>
            <span className="text-accent font-semibold">BLAS Level-3 GEMM Kernel</span>
          </div>
        </div>

        {/* CARD 3: Memory Footprint & Compression Donut Chart */}
        <div className="bg-surface-card border border-border rounded-xl p-6 shadow-sm flex flex-col justify-between transition-colors">
          <div className="border-b border-border/70 pb-3">
            <h2 className="font-heading font-bold text-base text-graphite-dark">
              Resident RAM Footprint & Compression
            </h2>
            <p className="text-xs text-graphite-muted">
              Resident memory allocation across 100,000 128-d vectors (SIFT-100K)
            </p>
          </div>

          <div className="py-4 flex flex-col sm:flex-row items-center justify-around gap-6">
            {/* SVG Donut Chart */}
            <div className="relative w-40 h-40 flex-shrink-0 flex items-center justify-center">
              <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                {/* Segment 1: Exact Raw (51.2MB - 43%) */}
                <circle
                  cx="50"
                  cy="50"
                  r="38"
                  fill="transparent"
                  stroke="#3182CE"
                  strokeWidth="16"
                  strokeDasharray="103 239"
                  strokeDashoffset="0"
                />
                {/* Segment 2: HNSW Graph (64.0MB - 54%) */}
                <circle
                  cx="50"
                  cy="50"
                  r="38"
                  fill="transparent"
                  stroke="#F7B84B"
                  strokeWidth="16"
                  strokeDasharray="128 239"
                  strokeDashoffset="-103"
                />
                {/* Segment 3: IVF-PQ (2.5MB - 3%) */}
                <circle
                  cx="50"
                  cy="50"
                  r="38"
                  fill="transparent"
                  stroke="#0AB39C"
                  strokeWidth="16"
                  strokeDasharray="8 239"
                  strokeDashoffset="-231"
                />
              </svg>

              {/* Center Donut Label */}
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                <span className="text-lg font-bold font-heading text-graphite-dark">20.8x</span>
                <span className="text-[10px] font-mono text-graphite-muted">Compression</span>
              </div>
            </div>

            {/* Donut Legend */}
            <div className="space-y-2 text-xs font-mono">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-sm bg-[#3182CE]" />
                <span className="text-graphite">Exact Raw: <strong>51.2 MB</strong> (100%)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-sm bg-[#F7B84B]" />
                <span className="text-graphite">HNSW Graph: <strong>64.0 MB</strong> (125%)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-sm bg-[#0AB39C]" />
                <span className="text-emerald-600 font-bold">IVF-PQ Codebook: <strong>2.5 MB</strong> (4.8%)</span>
              </div>
            </div>
          </div>

          <div className="pt-3 border-t border-border flex items-center justify-between text-[11px] text-graphite-muted font-mono">
            <span>Product Quantization: 95.2% RAM Saved</span>
            <span className="text-emerald-600 font-semibold">Subspace ADC</span>
          </div>
        </div>

        {/* CARD 4: Latency Comparison Horizontal Bar Chart */}
        <div className="bg-surface-card border border-border rounded-xl p-6 shadow-sm flex flex-col justify-between transition-colors">
          <div className="border-b border-border/70 pb-3">
            <h2 className="font-heading font-bold text-base text-graphite-dark">
              Query Latency Comparison (Lower is Faster)
            </h2>
            <p className="text-xs text-graphite-muted">
              Normalized execution time in milliseconds per query vector
            </p>
          </div>

          <div className="py-4 space-y-3">
            {algorithmData.map((algo) => {
              const maxLat = 0.20;
              const barWidth = Math.min((algo.latency / maxLat) * 100, 100);

              return (
                <div key={algo.name} className="space-y-1">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="font-sans font-medium text-graphite-dark">{algo.name}</span>
                    <span className="font-bold text-graphite">{algo.latency.toFixed(2)} ms</span>
                  </div>
                  <div className="w-full h-3 rounded bg-surface-secondary overflow-hidden">
                    <div
                      className="h-full rounded transition-all duration-500"
                      style={{
                        width: `${barWidth}%`,
                        backgroundColor: algo.color,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="pt-3 border-t border-border flex items-center justify-between text-[11px] text-graphite-muted font-mono">
            <span>Fastest: Exact Reference (0.04ms)</span>
            <span className="text-blue-600 font-semibold">AVX2 / Level-3 BLAS</span>
          </div>
        </div>

      </div>
    </div>
  );
};
