import React from 'react';
import { SystemStatus } from '../types';
import { Cpu, Database, Activity, RefreshCw } from 'lucide-react';

interface HeaderProps {
  status: SystemStatus | null;
  loading: boolean;
  onRefresh: () => void;
}

export const Header: React.FC<HeaderProps> = ({ status, loading, onRefresh }) => {
  return (
    <header className="border-b border-border bg-surface-card sticky top-0 z-40 backdrop-blur-md bg-opacity-95">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        {/* Brand identity */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-accent to-accent-hover flex items-center justify-center font-bold text-white text-lg shadow-sm shadow-accent/20 flex-shrink-0">
            ⚡
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-heading font-bold text-lg tracking-tight text-graphite-dark">
                INDEXORA
              </span>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-surface-secondary text-graphite border border-border">
                v1.0.0
              </span>
            </div>
            <p className="text-xs text-graphite-muted">
              First-Principles Vector Search Engine & Architecture Lab
            </p>
          </div>
        </div>

        {/* Live system state bar */}
        <div className="flex items-center gap-2 sm:gap-4 overflow-x-auto py-1 text-xs font-mono">
          {status ? (
            <>
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-surface-subtle border border-border text-graphite">
                <Database className="w-3.5 h-3.5 text-accent" />
                <span>N={status.dataset.n_base}</span>
                <span className="text-graphite-subtle">·</span>
                <span>d={status.dataset.d}</span>
              </div>

              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-surface-subtle border border-border text-graphite">
                <Cpu className="w-3.5 h-3.5 text-accent" />
                <span>{status.session.index_name}</span>
                <span className="text-graphite-subtle">·</span>
                <span>{status.session.active_count} active</span>
              </div>

              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="font-sans font-medium text-xs">Engine Online</span>
              </div>
            </>
          ) : (
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-700">
              <Activity className="w-3.5 h-3.5 animate-spin" />
              <span className="font-sans text-xs">Connecting...</span>
            </div>
          )}

          <button
            onClick={onRefresh}
            disabled={loading}
            title="Refresh engine status"
            className="p-1.5 rounded-md hover:bg-surface-secondary border border-transparent hover:border-border text-graphite-muted hover:text-graphite transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>
    </header>
  );
};
