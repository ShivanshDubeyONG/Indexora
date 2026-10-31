import React, { useState, useEffect } from 'react';
import { fetchPcaProjection } from '../api';
import { PcaProjectionResponse, DatasetCurrent } from '../types';
import { Layers, Search, RefreshCw, Eye, Info } from 'lucide-react';

interface VectorExplorerViewProps {
  dataset: DatasetCurrent | null;
}

const CLUSTER_COLORS = [
  '#3986D9', // primary accent blue
  '#10B981', // emerald
  '#F59E0B', // amber
  '#8B5CF6', // purple
  '#EC4899', // pink
  '#06B6D4', // cyan
  '#64748B', // slate
  '#F97316', // orange
];

export const VectorExplorerView: React.FC<VectorExplorerViewProps> = ({ dataset }) => {
  const [queryIdIndex, setQueryIdIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pcaData, setPcaData] = useState<PcaProjectionResponse | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<any | null>(null);

  const loadProjection = async (qIndex: number) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPcaProjection({ query_id_index: qIndex });
      setPcaData(data);
    } catch (err: any) {
      setError(err.message || 'Failed to compute PCA projection');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProjection(queryIdIndex);
  }, [queryIdIndex]);

  // Compute SVG bounding box mapping
  let minX = -3, maxX = 3, minY = -3, maxY = 3;
  if (pcaData && pcaData.points.length > 0) {
    const allX = [...pcaData.points.map((p) => p.x), pcaData.query.x, ...pcaData.centroids.map((c) => c.x)];
    const allY = [...pcaData.points.map((p) => p.y), pcaData.query.y, ...pcaData.centroids.map((c) => c.y)];
    minX = Math.min(...allX);
    maxX = Math.max(...allX);
    minY = Math.min(...allY);
    maxY = Math.max(...allY);
    const padX = (maxX - minX) * 0.12 || 1;
    const padY = (maxY - minY) * 0.12 || 1;
    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;
  }

  const width = 800;
  const height = 520;
  const mapX = (x: number) => ((x - minX) / (maxX - minX)) * (width - 80) + 40;
  const mapY = (y: number) => height - (((y - minY) / (maxY - minY)) * (height - 80) + 40);

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="bg-surface-card border border-border rounded-xl p-6 sm:p-8 shadow-card">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-accent/10 text-accent font-mono text-xs font-semibold uppercase tracking-wider mb-2">
              Geometric Dimensionality Reduction
            </div>
            <h1 className="font-heading font-bold text-2xl text-graphite-dark">
              Vector Space PCA Projection Explorer
            </h1>
            <p className="text-sm text-graphite-muted mt-1 max-w-2xl">
              2D Principal Component Analysis (PCA) projection of the {dataset?.d || 8}-dimensional vector manifold.
              Examine spatial clustering, Voronoi partitions, cluster centroids, and geometric query proximity.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-xs">
              <label className="font-mono text-graphite-muted">Target Query:</label>
              <select
                value={queryIdIndex}
                onChange={(e) => setQueryIdIndex(parseInt(e.target.value))}
                className="px-3 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-xs text-graphite focus:outline-none focus:ring-1 focus:ring-accent"
              >
                {Array.from({ length: dataset?.n_queries || 8 }).map((_, idx) => (
                  <option key={idx} value={idx}>
                    Query #{idx} (q{idx})
                  </option>
                ))}
              </select>
            </div>
            <button
              onClick={() => loadProjection(queryIdIndex)}
              disabled={loading}
              className="p-2 rounded-lg border border-border bg-surface-subtle hover:bg-surface-secondary text-graphite cursor-pointer transition-all"
              title="Refresh PCA projection"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-accent' : ''}`} />
            </button>
          </div>
        </div>
      </section>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs">
          {error}
        </div>
      )}

      {/* Main Canvas Area */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Interactive SVG Projection */}
        <div className="lg:col-span-3 bg-surface-card border border-border rounded-xl p-5 shadow-card relative flex flex-col items-center">
          <div className="w-full flex items-center justify-between border-b border-border pb-3 mb-4 text-xs">
            <span className="font-heading font-bold text-graphite-dark flex items-center gap-1.5">
              <Eye className="w-4 h-4 text-accent" />
              Interactive PCA 2D Manifold Canvas
            </span>
            <div className="flex items-center gap-3 font-mono text-graphite-muted">
              <span>N={pcaData?.points.length || 0} Points</span>
              <span>·</span>
              <span>K={pcaData?.n_clusters || 0} Centroids</span>
            </div>
          </div>

          <div className="w-full overflow-x-auto relative flex justify-center">
            <svg
              viewBox={`0 0 ${width} ${height}`}
              className="w-full max-w-[800px] h-auto bg-surface-subtle rounded-lg border border-border/80"
              style={{ minHeight: '380px' }}
            >
              {/* Grid Lines */}
              <line x1={40} y1={height / 2} x2={width - 40} y2={height / 2} stroke="#D5E2EF" strokeDasharray="3 3" />
              <line x1={width / 2} y1={40} x2={width / 2} y2={height - 40} stroke="#D5E2EF" strokeDasharray="3 3" />

              {/* Centroids */}
              {pcaData?.centroids.map((c) => {
                const cx = mapX(c.x);
                const cy = mapY(c.y);
                const color = CLUSTER_COLORS[c.cluster % CLUSTER_COLORS.length];
                return (
                  <g key={`c-${c.cluster}`}>
                    <circle cx={cx} cy={cy} r={9} fill="none" stroke={color} strokeWidth={2} strokeDasharray="2 2" />
                    <line x1={cx - 5} y1={cy} x2={cx + 5} y2={cy} stroke={color} strokeWidth={2} />
                    <line x1={cx} y1={cy - 5} x2={cx} y2={cy + 5} stroke={color} strokeWidth={2} />
                    <text x={cx + 10} y={cy + 3} fill="#687583" fontSize={9} fontFamily="monospace">
                      C{c.cluster}
                    </text>
                  </g>
                );
              })}

              {/* Base Vectors */}
              {pcaData?.points.map((p) => {
                const px = mapX(p.x);
                const py = mapY(p.y);
                const color = CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length];
                const isHovered = hoveredPoint?.id === p.id;
                return (
                  <circle
                    key={p.id}
                    cx={px}
                    cy={py}
                    r={isHovered ? 7 : 4.5}
                    fill={color}
                    fillOpacity={isHovered ? 1.0 : 0.75}
                    stroke={isHovered ? '#303943' : '#FFFFFF'}
                    strokeWidth={isHovered ? 2 : 1}
                    className="cursor-pointer transition-all duration-150"
                    onMouseEnter={() => setHoveredPoint(p)}
                    onMouseLeave={() => setHoveredPoint(null)}
                  />
                );
              })}

              {/* Query Vector */}
              {pcaData?.query && (
                <g>
                  {/* Outer pulse ring */}
                  <circle
                    cx={mapX(pcaData.query.x)}
                    cy={mapY(pcaData.query.y)}
                    r={14}
                    fill="#3986D9"
                    fillOpacity={0.2}
                    className="animate-pulse"
                  />
                  {/* Query center star */}
                  <circle
                    cx={mapX(pcaData.query.x)}
                    cy={mapY(pcaData.query.y)}
                    r={6.5}
                    fill="#3986D9"
                    stroke="#FFFFFF"
                    strokeWidth={2}
                  />
                  <text
                    x={mapX(pcaData.query.x) + 12}
                    y={mapY(pcaData.query.y) + 4}
                    fill="#303943"
                    fontSize={11}
                    fontWeight="bold"
                    fontFamily="monospace"
                  >
                    Query #{pcaData.query.query_index}
                  </text>
                </g>
              )}
            </svg>
          </div>

          <div className="w-full mt-3 flex items-center justify-between text-[11px] text-graphite-muted">
            <span className="flex items-center gap-2">
              <span className="inline-block w-3 h-3 rounded-full bg-accent border border-white" />
              <span>Query Target Vector (q)</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span>Centroids:</span>
              <span className="font-mono text-graphite">+ with dashed ring</span>
            </span>
            <span>Hover any vector for coordinate inspection</span>
          </div>
        </div>

        {/* Inspector Sidebar */}
        <div className="space-y-4">
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card">
            <h3 className="font-heading font-bold text-xs uppercase tracking-wider text-graphite mb-3 flex items-center gap-2">
              <Info className="w-3.5 h-3.5 text-accent" />
              Inspector & Point Detail
            </h3>

            {hoveredPoint ? (
              <div className="space-y-2.5 text-xs font-mono">
                <div className="p-2.5 rounded-lg bg-surface-subtle border border-border">
                  <span className="text-graphite-subtle text-[10px] block">External ID</span>
                  <span className="text-graphite-dark font-bold text-sm">#{hoveredPoint.id}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-surface-subtle border border-border">
                  <span className="text-graphite-subtle text-[10px] block">Assigned Cluster</span>
                  <span className="text-accent font-semibold">Cluster {hoveredPoint.cluster}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-surface-subtle border border-border">
                  <span className="text-graphite-subtle text-[10px] block">PCA Projection Coordinates</span>
                  <span className="text-graphite">
                    X: {hoveredPoint.x.toFixed(4)}, Y: {hoveredPoint.y.toFixed(4)}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-xs text-graphite-muted py-6 text-center border border-dashed border-border rounded-lg bg-surface-subtle/50">
                Hover over any vector dot on the canvas to inspect its projected coordinates and partition cluster.
              </div>
            )}
          </div>

          {/* Cluster Legend */}
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-3">
            <h3 className="font-heading font-bold text-xs uppercase tracking-wider text-graphite">
              Partition Clusters
            </h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {Array.from({ length: dataset?.n_clusters || 4 }).map((_, idx) => (
                <div key={idx} className="flex items-center gap-2 p-1.5 rounded bg-surface-subtle border border-border">
                  <span
                    className="w-3 h-3 rounded-full flex-shrink-0"
                    style={{ backgroundColor: CLUSTER_COLORS[idx % CLUSTER_COLORS.length] }}
                  />
                  <span className="font-mono text-graphite">Cluster {idx}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
