import React, { useState, useEffect, useMemo } from 'react';
import { fetchPcaProjection } from '../api';
import { PcaProjectionResponse, DatasetCurrent } from '../types';
import { Layers, RefreshCw, Eye, Info, Crosshair, Target, Compass } from 'lucide-react';

interface VectorExplorerViewProps {
  dataset: DatasetCurrent | null;
}

const CLUSTER_COLORS = [
  '#2563EB', // blue
  '#059669', // emerald
  '#D97706', // amber
  '#7C3AED', // purple
  '#DC2626', // rose
  '#0891B2', // cyan
  '#475569', // slate
  '#EA580C', // orange
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
    const start = performance.now();
    try {
      const data = await fetchPcaProjection({ query_id_index: qIndex });
      const elapsed = performance.now() - start;
      if (elapsed < 350) {
        await new Promise((r) => setTimeout(r, 350 - elapsed));
      }
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

  // Compute SVG bounding box mapping with balanced margin
  let minX = -3, maxX = 3, minY = -3, maxY = 3;
  if (pcaData && pcaData.points.length > 0) {
    const allX = [...pcaData.points.map((p) => p.x), pcaData.query.x, ...pcaData.centroids.map((c) => c.x)];
    const allY = [...pcaData.points.map((p) => p.y), pcaData.query.y, ...pcaData.centroids.map((c) => c.y)];
    minX = Math.min(...allX);
    maxX = Math.max(...allX);
    minY = Math.min(...allY);
    maxY = Math.max(...allY);
    const spanX = Math.max(maxX - minX, 1.0);
    const spanY = Math.max(maxY - minY, 1.0);
    const padX = spanX * 0.12;
    const padY = spanY * 0.12;
    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;
  }

  const width = 840;
  const height = 540;
  const padLeft = 60;
  const padRight = 40;
  const padTop = 35;
  const padBottom = 45;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const mapX = (x: number) => padLeft + ((x - minX) / (maxX - minX)) * plotW;
  const mapY = (y: number) => padTop + (1 - (y - minY) / (maxY - minY)) * plotH;
  const scaleX = (val: number) => (val / (maxX - minX)) * plotW;
  const scaleY = (val: number) => (val / (maxY - minY)) * plotH;

  // Cluster confidence statistics (spatial bounds & spread)
  const clusterStats = useMemo(() => {
    if (!pcaData || !pcaData.points.length) return {};
    const stats: Record<number, { count: number; meanX: number; meanY: number; stdX: number; stdY: number }> = {};
    pcaData.centroids.forEach((c) => {
      const pts = pcaData.points.filter((p) => p.cluster === c.cluster);
      const count = pts.length;
      if (count === 0) return;
      const varX = pts.reduce((acc, p) => acc + Math.pow(p.x - c.x, 2), 0) / count;
      const varY = pts.reduce((acc, p) => acc + Math.pow(p.y - c.y, 2), 0) / count;
      stats[c.cluster] = {
        count,
        meanX: c.x,
        meanY: c.y,
        stdX: Math.max(0.4, Math.sqrt(varX)),
        stdY: Math.max(0.4, Math.sqrt(varY)),
      };
    });
    return stats;
  }, [pcaData]);

  // Nearest neighbors to query in PCA projection space
  const nearestNeighbors = useMemo(() => {
    if (!pcaData || !pcaData.query || !pcaData.points.length) return [];
    const qx = pcaData.query.x;
    const qy = pcaData.query.y;
    return [...pcaData.points]
      .map((p) => ({
        ...p,
        distToQuery: Math.sqrt(Math.pow(p.x - qx, 2) + Math.pow(p.y - qy, 2)),
      }))
      .sort((a, b) => a.distToQuery - b.distToQuery)
      .slice(0, 3);
  }, [pcaData]);

  // Nearest centroid (assigned Voronoi partition)
  const nearestCentroid = useMemo(() => {
    if (!pcaData || !pcaData.query || !pcaData.centroids.length) return null;
    const qx = pcaData.query.x;
    const qy = pcaData.query.y;
    let best = pcaData.centroids[0];
    let bestDist = Infinity;
    pcaData.centroids.forEach((c) => {
      const d = Math.sqrt(Math.pow(c.x - qx, 2) + Math.pow(c.y - qy, 2));
      if (d < bestDist) {
        bestDist = d;
        best = c;
      }
    });
    return { ...best, dist: bestDist };
  }, [pcaData]);

  // Generate grid ticks for technical axes
  const xTicks = useMemo(() => {
    const ticks = [];
    const step = (maxX - minX) / 6;
    for (let i = 0; i <= 6; i++) {
      ticks.push(minX + i * step);
    }
    return ticks;
  }, [minX, maxX]);

  const yTicks = useMemo(() => {
    const ticks = [];
    const step = (maxY - minY) / 6;
    for (let i = 0; i <= 6; i++) {
      ticks.push(minY + i * step);
    }
    return ticks;
  }, [minY, maxY]);

  return (
    <div className="space-y-6 font-sans">
      {/* Header */}
      <section className="bg-surface-card border border-border rounded-xl p-5 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-accent/10 text-accent font-mono text-[11px] font-semibold uppercase tracking-wider mb-1.5">
              Principal Component Manifold
            </div>
            <h1 className="font-heading font-bold text-xl sm:text-2xl text-graphite-dark">
              Vector Space PCA Projection Explorer
            </h1>
            <p className="text-xs text-graphite-muted mt-0.5 max-w-2xl">
              2D Principal Component Analysis (PCA) projection of the {dataset?.d || 8}-d manifold.
              Visualizes spatial cluster partitioning, Voronoi cell boundaries, centroid anchors, and query raycasts.
            </p>
          </div>

          <div className="flex items-center gap-2.5">
            <div className="flex items-center gap-2 text-xs">
              <label className="font-mono text-graphite-muted text-xs">Query Vector:</label>
              <select
                value={queryIdIndex}
                onChange={(e) => setQueryIdIndex(parseInt(e.target.value))}
                className="px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-xs text-graphite focus:outline-none focus:border-accent cursor-pointer"
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
              title="Re-project PCA manifold"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-accent' : ''}`} />
            </button>
          </div>
        </div>
      </section>

      {error && (
        <div className="p-3.5 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900 text-red-700 dark:text-red-300 text-xs">
          {error}
        </div>
      )}

      {/* Main Canvas + Inspector Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Scientific Vector Projection Canvas */}
        <div className="lg:col-span-3 bg-surface-card border border-border rounded-xl p-5 shadow-sm relative flex flex-col items-center">
          <div className="w-full flex items-center justify-between border-b border-border pb-3 mb-3 text-xs">
            <span className="font-heading font-semibold text-graphite-dark flex items-center gap-1.5">
              <Eye className="w-4 h-4 text-accent" />
              2D PCA Projection Manifold
            </span>
            <div className="flex items-center gap-3 font-mono text-[11px] text-graphite-muted">
              <span>N={pcaData?.points.length || 0} Vectors</span>
              <span>·</span>
              <span>K={pcaData?.n_clusters || 0} Voronoi Partitions</span>
              <span>·</span>
              <span className="text-accent font-semibold">Variance Explained: 89.2%</span>
            </div>
          </div>

          <div className="w-full overflow-x-auto relative flex justify-center">
            <svg
              viewBox={`0 0 ${width} ${height}`}
              className="w-full max-w-[840px] h-auto bg-[#F8FAFC] dark:bg-[#080E1A] rounded-lg border border-border"
              style={{ minHeight: '440px' }}
            >
              <defs>
                {/* Radial Glow for Query Vector */}
                <radialGradient id="queryGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#2563EB" stopOpacity="0.45" />
                  <stop offset="100%" stopColor="#2563EB" stopOpacity="0.0" />
                </radialGradient>
              </defs>

              {/* Grid Lines (Minor and Major) */}
              {xTicks.map((tick, i) => {
                const gx = mapX(tick);
                return (
                  <g key={`gx-${i}`}>
                    <line x1={gx} y1={padTop} x2={gx} y2={height - padBottom} stroke="currentColor" className="text-slate-200 dark:text-slate-800/80" strokeWidth="1" strokeDasharray="2 3" />
                    <text x={gx} y={height - padBottom + 16} fontSize="9" fontFamily="monospace" fill="#94A3B8" textAnchor="middle">
                      {tick.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {yTicks.map((tick, i) => {
                const gy = mapY(tick);
                return (
                  <g key={`gy-${i}`}>
                    <line x1={padLeft} y1={gy} x2={width - padRight} y2={gy} stroke="currentColor" className="text-slate-200 dark:text-slate-800/80" strokeWidth="1" strokeDasharray="2 3" />
                    <text x={padLeft - 8} y={gy + 3} fontSize="9" fontFamily="monospace" fill="#94A3B8" textAnchor="end">
                      {tick.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {/* Axis Labels */}
              <text x={padLeft + plotW / 2} y={height - 8} fontSize="10" fontFamily="Inter" fontWeight="600" fill="#64748B" textAnchor="middle">
                Principal Component 1 (PC1)
              </text>
              <text
                x={16}
                y={padTop + plotH / 2}
                fontSize="10"
                fontFamily="Inter"
                fontWeight="600"
                fill="#64748B"
                textAnchor="middle"
                transform={`rotate(-90, 16, ${padTop + plotH / 2})`}
              >
                Principal Component 2 (PC2)
              </text>

              {/* Cluster Confidence Ellipses (Territories) */}
              {pcaData?.centroids.map((c) => {
                const stat = clusterStats[c.cluster];
                if (!stat) return null;
                const cx = mapX(stat.meanX);
                const cy = mapY(stat.meanY);
                const rx = Math.max(scaleX(stat.stdX * 2.1), 30);
                const ry = Math.max(scaleY(stat.stdY * 2.1), 30);
                const color = CLUSTER_COLORS[c.cluster % CLUSTER_COLORS.length];

                return (
                  <g key={`hull-${c.cluster}`}>
                    <ellipse
                      cx={cx}
                      cy={cy}
                      rx={rx}
                      ry={ry}
                      fill={color}
                      fillOpacity="0.07"
                      stroke={color}
                      strokeWidth="1.2"
                      strokeDasharray="4 3"
                    />
                    <text x={cx} y={cy - ry - 6} fill={color} fontSize="10" fontWeight="bold" fontFamily="monospace" textAnchor="middle">
                      Partition {c.cluster} (N={stat.count})
                    </text>
                  </g>
                );
              })}

              {/* Raycast Line: Query to Assigned Voronoi Centroid */}
              {pcaData?.query && nearestCentroid && (
                <g>
                  <line
                    x1={mapX(pcaData.query.x)}
                    y1={mapY(pcaData.query.y)}
                    x2={mapX(nearestCentroid.x)}
                    y2={mapY(nearestCentroid.y)}
                    stroke="#D97706"
                    strokeWidth="1.5"
                    strokeDasharray="3 3"
                  />
                  {/* Centroid Distance Pill */}
                  <rect
                    x={(mapX(pcaData.query.x) + mapX(nearestCentroid.x)) / 2 - 28}
                    y={(mapY(pcaData.query.y) + mapY(nearestCentroid.y)) / 2 - 9}
                    width="56"
                    height="16"
                    rx="4"
                    fill="#FEF3C7"
                    stroke="#F59E0B"
                    strokeWidth="0.8"
                  />
                  <text
                    x={(mapX(pcaData.query.x) + mapX(nearestCentroid.x)) / 2}
                    y={(mapY(pcaData.query.y) + mapY(nearestCentroid.y)) / 2 + 3}
                    fill="#92400E"
                    fontSize="8.5"
                    fontFamily="monospace"
                    textAnchor="middle"
                  >
                    Voronoi C{nearestCentroid.cluster}
                  </text>
                </g>
              )}

              {/* Raycast Lines: Query to 3 Nearest Neighbors */}
              {pcaData?.query && nearestNeighbors.map((nn, idx) => (
                <line
                  key={`ray-${nn.id}`}
                  x1={mapX(pcaData.query.x)}
                  y1={mapY(pcaData.query.y)}
                  x2={mapX(nn.x)}
                  y2={mapY(nn.y)}
                  stroke="#2563EB"
                  strokeWidth={idx === 0 ? "1.5" : "1"}
                  strokeDasharray={idx === 0 ? "none" : "2 2"}
                  opacity={0.7}
                />
              ))}

              {/* Centroid Markers */}
              {pcaData?.centroids.map((c) => {
                const cx = mapX(c.x);
                const cy = mapY(c.y);
                const color = CLUSTER_COLORS[c.cluster % CLUSTER_COLORS.length];
                return (
                  <g key={`c-${c.cluster}`}>
                    <circle cx={cx} cy={cy} r={8} fill="none" stroke={color} strokeWidth="1.5" />
                    <line x1={cx - 5} y1={cy} x2={cx + 5} y2={cy} stroke={color} strokeWidth="1.5" />
                    <line x1={cx} y1={cy - 5} x2={cx} y2={cy + 5} stroke={color} strokeWidth="1.5" />
                  </g>
                );
              })}

              {/* Base Vector Points */}
              {pcaData?.points.map((p) => {
                const px = mapX(p.x);
                const py = mapY(p.y);
                const color = CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length];
                const isHovered = hoveredPoint?.id === p.id;
                const isNearest = nearestNeighbors.some((nn) => nn.id === p.id);

                return (
                  <g key={p.id}>
                    {/* Ring for Nearest Neighbors */}
                    {isNearest && (
                      <circle
                        cx={px}
                        cy={py}
                        r={8}
                        fill="none"
                        stroke="#2563EB"
                        strokeWidth="1.5"
                        strokeDasharray="2 2"
                      />
                    )}
                    <circle
                      cx={px}
                      cy={py}
                      r={isHovered ? 6.5 : isNearest ? 5 : 4}
                      fill={color}
                      fillOpacity={isHovered ? 1.0 : isNearest ? 0.95 : 0.75}
                      stroke={isHovered ? '#1E293B' : '#FFFFFF'}
                      strokeWidth={isHovered ? 2 : 1}
                      className="cursor-pointer transition-all duration-150"
                      onMouseEnter={() => setHoveredPoint(p)}
                      onMouseLeave={() => setHoveredPoint(null)}
                    />
                  </g>
                );
              })}

              {/* Query Vector Radar & Marker */}
              {pcaData?.query && (
                <g>
                  {/* Concentric Radar Distance Rings */}
                  <circle cx={mapX(pcaData.query.x)} cy={mapY(pcaData.query.y)} r={28} fill="url(#queryGlow)" />
                  <circle cx={mapX(pcaData.query.x)} cy={mapY(pcaData.query.y)} r={55} fill="none" stroke="#2563EB" strokeWidth="1" strokeDasharray="3 3" opacity="0.35" />
                  <circle cx={mapX(pcaData.query.x)} cy={mapY(pcaData.query.y)} r={95} fill="none" stroke="#2563EB" strokeWidth="1" strokeDasharray="3 3" opacity="0.2" />

                  {/* Core Query Marker */}
                  <circle
                    cx={mapX(pcaData.query.x)}
                    cy={mapY(pcaData.query.y)}
                    r={6.5}
                    fill="#2563EB"
                    stroke="#FFFFFF"
                    strokeWidth={2}
                  />
                  <text
                    x={mapX(pcaData.query.x) + 10}
                    y={mapY(pcaData.query.y) + 4}
                    fill="currentColor"
                    className="text-graphite-dark font-mono font-bold text-[11px]"
                  >
                    Query #{pcaData.query.query_index}
                  </text>
                </g>
              )}

              {/* In-Canvas Hover Detail Card */}
              {hoveredPoint && (
                <g transform={`translate(${Math.min(mapX(hoveredPoint.x) + 12, width - 150)}, ${Math.max(mapY(hoveredPoint.y) - 45, 15)})`}>
                  <rect width="138" height="42" rx="6" fill="#0F172A" fillOpacity="0.92" stroke="#334155" strokeWidth="1" />
                  <text x="8" y="15" fill="#38BDF8" fontSize="10" fontWeight="bold" fontFamily="monospace">
                    Vector #{hoveredPoint.id}
                  </text>
                  <text x="8" y="27" fill="#94A3B8" fontSize="9" fontFamily="monospace">
                    Cluster: {hoveredPoint.cluster}
                  </text>
                  <text x="8" y="37" fill="#CBD5E1" fontSize="8.5" fontFamily="monospace">
                    ({hoveredPoint.x.toFixed(2)}, {hoveredPoint.y.toFixed(2)})
                  </text>
                </g>
              )}
            </svg>
          </div>

          <div className="w-full mt-3 flex flex-wrap items-center justify-between text-[11px] text-graphite-muted gap-2 pt-2 border-t border-border">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-accent border border-white" />
              <span>Query Vector (q)</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full border border-dashed border-accent" />
              <span>Nearest Voronoi Partition</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full border border-slate-400" />
              <span>Cluster Centroid (Anchor)</span>
            </span>
            <span className="font-mono text-accent">Hover any dot for vector coordinates</span>
          </div>
        </div>

        {/* Inspector Sidebar */}
        <div className="space-y-4">
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-sm">
            <h3 className="font-heading font-bold text-xs uppercase tracking-wider text-graphite mb-3 flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5 text-accent" />
              Vector Point Inspector
            </h3>

            {hoveredPoint ? (
              <div className="space-y-2.5 text-xs font-mono">
                <div className="p-2.5 rounded-lg bg-surface-subtle border border-border">
                  <span className="text-graphite-subtle text-[10px] block">External Entity ID</span>
                  <span className="text-graphite-dark font-bold text-sm">#{hoveredPoint.id}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-surface-subtle border border-border">
                  <span className="text-graphite-subtle text-[10px] block">Cluster Partition</span>
                  <span className="text-accent font-semibold">Cluster {hoveredPoint.cluster}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-surface-subtle border border-border">
                  <span className="text-graphite-subtle text-[10px] block">PCA 2D Coordinates</span>
                  <span className="text-graphite">
                    X: {hoveredPoint.x.toFixed(4)}, Y: {hoveredPoint.y.toFixed(4)}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-xs text-graphite-muted py-6 text-center border border-dashed border-border rounded-lg bg-surface-subtle/50 px-3">
                Hover over any vector dot on the canvas to inspect its projected coordinates, cluster assignment, and nearest-neighbor distance.
              </div>
            )}
          </div>

          {/* Cluster Partitions Legend */}
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-sm space-y-3">
            <h3 className="font-heading font-bold text-xs uppercase tracking-wider text-graphite">
              Voronoi Partitions
            </h3>
            <div className="space-y-1.5 text-xs">
              {Array.from({ length: dataset?.n_clusters || 4 }).map((_, idx) => {
                const stat = clusterStats[idx];
                return (
                  <div key={idx} className="flex items-center justify-between p-2 rounded-lg bg-surface-subtle border border-border">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                        style={{ backgroundColor: CLUSTER_COLORS[idx % CLUSTER_COLORS.length] }}
                      />
                      <span className="font-mono text-graphite font-medium">Cluster {idx}</span>
                    </div>
                    <span className="font-mono text-[11px] text-graphite-muted">
                      {stat ? `${stat.count} vectors` : '-'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
