import React from 'react';
import { SystemStatus } from '../types';
import { Search, Database, Cpu, CheckCircle2, ShieldCheck } from 'lucide-react';

interface HeaderProps {
  status: SystemStatus | null;
  loading: boolean;
  onRefresh: () => void;
  onOpenPalette?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  status,
  loading,
  onRefresh,
  onOpenPalette
}) => {
  return (
    <header className="h-14 border-b border-border bg-surface-card sticky top-0 z-30 flex items-center justify-between px-6 transition-colors shadow-sm">
      {/* Left: Brand / Cloud Icon */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center text-white font-bold text-sm shadow-sm">
          IX
        </div>
        <div className="flex items-baseline gap-2">
          <span className="font-heading font-extrabold text-base tracking-tight text-graphite-dark">
            INDEXORA
          </span>
          <span className="text-[11px] font-mono px-1.5 py-0.2 rounded bg-surface-secondary text-graphite-muted border border-border">
            v1.0.0
          </span>
        </div>
      </div>

      {/* Center: Search Bar (Like in the screenshot) */}
      <div className="flex-1 max-w-md mx-6 hidden sm:block">
        <div
          onClick={onOpenPalette}
          className="relative flex items-center cursor-pointer group"
        >
          <Search className="w-4 h-4 text-graphite-subtle absolute left-3 group-hover:text-accent transition-colors" />
          <input
            type="text"
            readOnly
            placeholder="Search Indexora vectors, queries, benchmarks... (⌘K)"
            className="w-full pl-9 pr-12 py-1.5 text-xs rounded-md border border-border bg-surface-subtle text-graphite placeholder:text-graphite-subtle group-hover:border-accent/50 focus:outline-none transition-colors cursor-pointer"
          />
          <kbd className="absolute right-2.5 px-1.5 py-0.5 rounded bg-surface-secondary border border-border text-[10px] font-mono text-graphite-subtle">
            ⌘K
          </kbd>
        </div>
      </div>

      {/* Right: Telemetry & Status Badges */}
      <div className="flex items-center gap-3 text-xs font-mono">
        {status ? (
          <>
            <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded bg-surface-subtle border border-border text-graphite">
              <Database className="w-3.5 h-3.5 text-accent" />
              <span>N={status.dataset.n_base}</span>
              <span className="text-graphite-subtle">·</span>
              <span>d={status.dataset.d}</span>
            </div>

            <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded bg-surface-subtle border border-border text-graphite">
              <Cpu className="w-3.5 h-3.5 text-accent" />
              <span>{status.session.index_name}</span>
            </div>

            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="font-sans font-medium text-[11px]">Engine Ready</span>
            </div>
          </>
        ) : (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-600">
            <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
            <span className="font-sans font-medium text-[11px]">Connecting...</span>
          </div>
        )}

        <div className="hidden sm:flex items-center gap-1 px-2.5 py-1 rounded bg-surface-subtle border border-border text-graphite text-[11px]">
          <ShieldCheck className="w-3.5 h-3.5 text-blue-500" />
          <span>389 Tests</span>
        </div>
      </div>
    </header>
  );
};
