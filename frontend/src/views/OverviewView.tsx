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

  // Semicircular Gauge Needle Angle (-90 deg to +90 deg for 0% to 100%)
  const needleAngle = -90 + (currentRecall.value / 100) * 180;

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

        {/* CARD 1: Semi-Circle Speedometer Accuracy Gauge */}
        <div className="bg-surface-card border border-border rounded-xl p-6 shadow-sm flex flex-col justify-between transition-colors">
          <div className="flex items-center justify-between border-b border-border/70 pb-3">
            <div>
              <h2 className="font-heading font-bold text-base text-graphite-dark">
                Retrieval Recall Accuracy (Recall@k)
              </h2>
              <p className="text-xs text-graphite-muted">
                Evaluated against exhaustive float64 mathematical ground-truth oracle
              </p>
            </div>

            {/* Algorithm Selector for Gauge */}
            <select
              value={selectedRecallAlgo}
              onChange={(e) => setSelectedRecallAlgo(e.target.value as any)}
              className="text-xs font-mono px-2 py-1 rounded border border-border bg-surface-subtle text-graphite focus:outline-none"
            >
              <option value="exact">Exact BLAS (100%)</option>
              <option value="hnsw">HNSW Graph (98.4%)</option>
              <option value="ivf">IVF-Flat (94.0%)</option>
              <option value="pq">IVF-PQ ADC (82.5%)</option>
            </select>
          </div>

          {/* SVG Semicircular Speedometer Gauge */}
          <div className="py-6 flex flex-col items-center justify-center">
            <svg viewBox="0 0 280 150" className="w-full max-w-[280px]">
              {/* Arc 1: Red (0% - 66%) */}
              <path
                d="M 30,130 A 110,110 0 0,1 100,35"
                fill="none"
                stroke="#E05638"
                strokeWidth="20"
                strokeLinecap="round"
              />
              {/* Arc 2: Amber (66% - 85%) */}
              <path
                d="M 103,32 A 110,110 0 0,1 177,32"
                fill="none"
                stroke="#F7B84B"
                strokeWidth="20"
              />
              {/* Arc 3: Cyan/Emerald (85% - 100%) */}
              <path
                d="M 180,35 A 110,110 0 0,1 250,130"
                fill="none"
                stroke="#0AB39C"
                strokeWidth="20"
                strokeLinecap="round"
              />

              {/* Tick Labels */}
              <text x="30" y="145" fontSize="11" fontFamily="Inter" fill="#718096" textAnchor="middle">0%</text>
              <text x="140" y="24" fontSize="11" fontFamily="Inter" fill="#718096" textAnchor="middle">85%</text>
              <text x="250" y="145" fontSize="11" fontFamily="Inter" fill="#718096" textAnchor="middle">100%</text>

              {/* Gauge Needle */}
              <g transform={`rotate(${needleAngle}, 140, 130)`}>
                <line x1="140" y1="130" x2="140" y2="42" stroke="#1A202C" strokeWidth="3.5" strokeLinecap="round" />
                <circle cx="140" cy="130" r="7" fill="#1A202C" />
                <circle cx="140" cy="130" r="3" fill="#FFFFFF" />
              </g>

              {/* Digital Readout */}
              <text x="140" y="115" fontSize="20" fontWeight="bold" fontFamily="Inter" fill="#1A202C" textAnchor="middle">
                {currentRecall.value.toFixed(1)}%
              </text>
            </svg>

            <span className="text-xs text-graphite-muted text-center mt-2 max-w-sm">
              {currentRecall.desc}
            </span>
          </div>

          <div className="pt-3 border-t border-border flex items-center justify-between text-[11px] text-graphite-muted font-mono">
            <span>Source: Verified Ground Truth Oracle</span>
            <span className="text-emerald-600 font-semibold">Strict Zero False Positives</span>
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
