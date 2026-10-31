import React, { useState } from 'react';
import { BenchmarkMetric } from '../types';
import { TrendingUp, HardDrive, Info } from 'lucide-react';

interface ParetoChartProps {
  metrics: BenchmarkMetric[];
}

const INDEX_COLORS: Record<string, string> = {
  'Exact BLAS': '#3B82F6', // Blue
  'Exact Flat': '#64748B', // Slate
  'IVF-Flat': '#F59E0B',   // Amber
  'IVF-PQ': '#8B5CF6',     // Purple
  'HNSW': '#10B981',       // Emerald
};

export const ParetoChart: React.FC<ParetoChartProps> = ({ metrics }) => {
  const [hoveredPoint, setHoveredPoint] = useState<BenchmarkMetric | null>(null);

  if (!metrics || metrics.length === 0) return null;

  // Chart dimensions
  const width = 580;
  const height = 260;
  const padLeft = 60;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 40;

  // Compute domains
  const maxLatency = Math.max(...metrics.map((m) => m.median_latency_ms), 0.5) * 1.25;
  const minRecall = Math.min(...metrics.map((m) => m.recall_at_k), 0.5);

  const scaleX = (recall: number) => {
    return padLeft + ((recall - minRecall) / (1.0 - minRecall || 1)) * (width - padLeft - padRight);
  };

  const scaleY = (lat: number) => {
    return height - padBottom - (lat / maxLatency) * (height - padTop - padBottom);
  };

  // Find baseline memory (Exact BLAS or first index)
  const baselineMem = metrics.find((m) => m.index_name.includes('Exact'))?.memory_kb || metrics[0]?.memory_kb || 1;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
      {/* 1. Recall vs Latency Pareto Frontier (7 cols) */}
      <div className="lg:col-span-7 bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-3 transition-colors">
        <div className="flex items-center justify-between border-b border-border pb-3">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-accent" />
            <h3 className="font-heading font-bold text-xs uppercase tracking-wider text-graphite-dark">
              Recall vs. Latency Pareto Frontier
            </h3>
          </div>
          <span className="text-[10px] font-mono text-graphite-muted">Top-Left is Optimal</span>
        </div>

        <div className="w-full overflow-x-auto flex justify-center">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full max-w-[580px] h-auto bg-surface-subtle rounded-lg border border-border">
            {/* Grid lines */}
            <line x1={padLeft} y1={scaleY(maxLatency / 2)} x2={width - padRight} y2={scaleY(maxLatency / 2)} stroke="var(--border-color)" strokeDasharray="3 3" />
            <line x1={padLeft} y1={height - padBottom} x2={width - padRight} y2={height - padBottom} stroke="var(--border-color)" strokeWidth="1.5" />
            <line x1={padLeft} y1={padTop} x2={padLeft} y2={height - padBottom} stroke="var(--border-color)" strokeWidth="1.5" />

            {/* Axis labels */}
            <text x={padLeft - 10} y={scaleY(maxLatency / 2) + 4} fill="var(--text-graphite-muted)" fontSize="9" fontFamily="JetBrains Mono" textAnchor="end">
              {(maxLatency / 2).toFixed(2)}ms
            </text>
            <text x={padLeft - 10} y={scaleY(0) + 4} fill="var(--text-graphite-muted)" fontSize="9" fontFamily="JetBrains Mono" textAnchor="end">
              0.0ms
            </text>

            <text x={scaleX(minRecall)} y={height - padBottom + 16} fill="var(--text-graphite-muted)" fontSize="9" fontFamily="JetBrains Mono" textAnchor="middle">
              {(minRecall * 100).toFixed(0)}%
            </text>
            <text x={scaleX(1.0)} y={height - padBottom + 16} fill="var(--text-graphite-muted)" fontSize="9" fontFamily="JetBrains Mono" textAnchor="middle">
              100% (Oracle)
            </text>

            <text x={width / 2} y={height - 10} fill="var(--text-graphite)" fontSize="10" fontFamily="JetBrains Mono" textAnchor="middle" fontWeight="bold">
              Recall@k (Empirical Ground-Truth Agreement) →
            </text>

            {/* Scatter points */}
            {metrics.map((m) => {
              const cx = scaleX(m.recall_at_k);
              const cy = scaleY(m.median_latency_ms);
              const color = INDEX_COLORS[m.index_name] || '#3B82F6';
              const isHovered = hoveredPoint?.index_name === m.index_name;

              return (
                <g key={m.index_name} className="cursor-pointer transition-all" onMouseEnter={() => setHoveredPoint(m)} onMouseLeave={() => setHoveredPoint(null)}>
                  {/* Subtle target crosshair on hover */}
                  {isHovered && (
                    <>
                      <line x1={padLeft} y1={cy} x2={cx} y2={cy} stroke={color} strokeDasharray="2 2" strokeWidth="1" />
                      <line x1={cx} y1={cy} x2={cx} y2={height - padBottom} stroke={color} strokeDasharray="2 2" strokeWidth="1" />
                    </>
                  )}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isHovered ? 8 : 6}
                    fill={color}
                    stroke="#FFFFFF"
                    strokeWidth={isHovered ? 2.5 : 1.5}
                    className="transition-all duration-150"
                  />
                  <text
                    x={cx + (cx > width - 100 ? -10 : 10)}
                    y={cy - 8}
                    fill="var(--text-graphite)"
                    fontSize="9"
                    fontFamily="JetBrains Mono"
                    fontWeight="bold"
                    textAnchor={cx > width - 100 ? 'end' : 'start'}
                  >
                    {m.index_name}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Hover detail callout */}
        {hoveredPoint ? (
          <div className="p-2.5 rounded-lg bg-surface-subtle border border-border flex items-center justify-between text-xs font-mono">
            <span className="font-bold text-graphite-dark">{hoveredPoint.index_name}</span>
            <span className="text-accent font-semibold">Recall: {(hoveredPoint.recall_at_k * 100).toFixed(1)}%</span>
            <span className="text-graphite">p50: {hoveredPoint.median_latency_ms.toFixed(3)}ms</span>
            <span className="text-emerald-500 font-bold">{hoveredPoint.qps.toFixed(0)} QPS</span>
          </div>
        ) : (
          <div className="text-[11px] text-graphite-muted text-center font-mono">
            Hover over any architecture point on the Pareto chart to inspect exact latency-recall coordinates.
          </div>
        )}
      </div>

      {/* 2. Memory Footprint & Compression Waterfall (5 cols) */}
      <div className="lg:col-span-5 bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-4 transition-colors">
        <div className="flex items-center justify-between border-b border-border pb-3">
          <div className="flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-accent" />
            <h3 className="font-heading font-bold text-xs uppercase tracking-wider text-graphite-dark">
              Memory Allocation & Compression
            </h3>
          </div>
          <span className="text-[10px] font-mono text-emerald-500 font-semibold">IVF-PQ 20.8x Compression</span>
        </div>

        <div className="space-y-3.5 pt-1">
          {metrics.map((m) => {
            const ratio = ((m.memory_kb / baselineMem) * 100).toFixed(1);
            const isCompressed = m.index_name.includes('PQ');
            const color = INDEX_COLORS[m.index_name] || '#3B82F6';

            return (
              <div key={m.index_name} className="space-y-1">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="font-semibold text-graphite">{m.index_name}</span>
                  <span className="text-graphite-muted">
                    {m.memory_kb.toFixed(1)} KB <span className="text-[10px]">({ratio}%)</span>
                  </span>
                </div>
                <div className="w-full h-2 rounded-full bg-surface-subtle border border-border overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.min(parseFloat(ratio), 100)}%`,
                      backgroundColor: isCompressed ? '#10B981' : color,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        <div className="p-3 bg-surface-subtle rounded-lg border border-border text-[11px] text-graphite-muted leading-relaxed">
          <strong className="text-graphite font-semibold">Storage Engineering Insight: </strong>
          While Exact and IVF-Flat store full 32-bit floating-point coordinates, Product Quantization decomposes vectors into 8-bit codebook indices, shrinking resident RAM footprint by up to <strong>95.2%</strong>.
        </div>
      </div>
    </div>
  );
};
