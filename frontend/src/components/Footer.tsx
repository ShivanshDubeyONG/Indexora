import React from 'react';
import { Terminal, ShieldCheck, Mail, GitBranch } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-border mt-16 bg-surface-card text-xs text-graphite-muted py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        {/* Left: Engine Statement */}
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-graphite font-semibold">
            <Terminal className="w-4 h-4 text-accent" />
            <span>Indexora Vector Architecture</span>
            <span className="text-graphite-subtle font-normal">| First-Principles Python & NumPy Engine</span>
          </div>
          <p className="text-graphite-muted max-w-xl">
            Exhaustive BLAS GEMM, Voronoi IVF-Flat, Asymmetric Product Quantization (IVF-PQ), and Hierarchical Proximity Graphs (HNSW) implemented directly with zero external vector database dependencies.
          </p>
        </div>

        {/* Right: Author & Contact */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 text-graphite">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-surface-secondary border border-border text-xs font-mono">
            <ShieldCheck className="w-3.5 h-3.5 text-accent-emerald" />
            <span>381 Tests Passing</span>
          </div>

          <a
            href="mailto:shivansh1.mitblr2024@learner.manipal.edu"
            className="inline-flex items-center gap-1.5 text-accent hover:underline"
          >
            <Mail className="w-3.5 h-3.5" />
            <span>shivansh1.mitblr2024@learner.manipal.edu</span>
          </a>

          <a
            href="https://github.com/ShivanshDubeyONG/Indexora"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-graphite-muted hover:text-graphite transition"
          >
            <GitBranch className="w-4 h-4 text-accent" />
            <span>GitHub Repository</span>
          </a>
        </div>
      </div>
    </footer>
  );
};
