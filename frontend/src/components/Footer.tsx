import React from 'react';
import { Terminal, ShieldCheck, Mail, GitBranch, Cpu, Zap, Database } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-border mt-16 bg-surface-card text-xs text-graphite-muted transition-colors font-sans">
      {/* Engineering Telemetry Dock */}
      <div className="border-b border-border/80 bg-surface-subtle/60 py-2.5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-wrap items-center justify-between gap-3 text-[11px] font-mono text-graphite-muted">
          <div className="flex items-center gap-4 overflow-x-auto scrollbar-none">
            <span className="flex items-center gap-1.5 text-graphite font-semibold">
              <Cpu className="w-3.5 h-3.5 text-accent" />
              <span>SIMD Kernel: Level-3 BLAS (dgemm)</span>
            </span>
            <span className="text-graphite-subtle">·</span>
            <span>Cache Alignment: 64-byte blocked</span>
            <span className="text-graphite-subtle">·</span>
            <span>Tie-Breaking: np.lexsort (distance, external_id)</span>
            <span className="text-graphite-subtle">·</span>
            <span>Quantization: ADC 8-bit Codebooks</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-emerald-600 dark:text-emerald-400 font-semibold">
              389 Unit & Property Tests Passing
            </span>
          </div>
        </div>
      </div>

      {/* Main Footer Info */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        {/* Left: Engine Statement */}
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-graphite font-semibold">
            <Terminal className="w-4 h-4 text-accent" />
            <span>Indexora Vector Architecture</span>
            <span className="text-graphite-subtle font-normal">| First-Principles Python & NumPy Engine</span>
          </div>
          <p className="text-graphite-muted max-w-xl text-xs leading-relaxed">
            Exhaustive BLAS GEMM, Voronoi IVF-Flat, Asymmetric Product Quantization (IVF-PQ), and Hierarchical Proximity Graphs (HNSW) implemented directly with zero external vector database dependencies.
          </p>
        </div>

        {/* Right: Author & Contact */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 text-graphite">
          <a
            href="mailto:shivansh1.mitblr2024@learner.manipal.edu"
            className="inline-flex items-center gap-1.5 text-accent hover:underline text-xs"
          >
            <Mail className="w-3.5 h-3.5" />
            <span>shivansh1.mitblr2024@learner.manipal.edu</span>
          </a>

          <a
            href="https://github.com/ShivanshDubeyONG/Indexora"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-graphite-muted hover:text-graphite transition text-xs"
          >
            <GitBranch className="w-4 h-4 text-accent" />
            <span>GitHub Repository</span>
          </a>
        </div>
      </div>
    </footer>
  );
};
