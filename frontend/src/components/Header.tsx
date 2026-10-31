import React, { useState, useEffect } from 'react';
import { SystemStatus } from '../types';
import { Logo } from './Logo';
import { Cpu, Database, Activity, RefreshCw, Sun, Moon } from 'lucide-react';

interface HeaderProps {
  status: SystemStatus | null;
  loading: boolean;
  onRefresh: () => void;
}

export const Header: React.FC<HeaderProps> = ({ status, loading, onRefresh }) => {
  const [isDark, setIsDark] = useState<boolean>(() => {
    if (typeof window !== 'undefined') {
      return (
        localStorage.getItem('indexora_theme') === 'dark' ||
        (!('indexora_theme' in localStorage) &&
          window.matchMedia('(prefers-color-scheme: dark)').matches)
      );
    }
    return false;
  });

  useEffect(() => {
    const root = document.documentElement;
    if (isDark) {
      root.classList.add('dark');
      localStorage.setItem('indexora_theme', 'dark');
    } else {
      root.classList.remove('dark');
      localStorage.setItem('indexora_theme', 'light');
    }
  }, [isDark]);

  const toggleTheme = () => setIsDark((prev) => !prev);

  return (
    <header className="border-b border-border bg-surface-card sticky top-0 z-40 backdrop-blur-md bg-opacity-95 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        {/* Brand identity with bespoke vector search logo */}
        <div className="flex items-center gap-3">
          <Logo size={36} className="shadow-subtle hover:scale-105 transition-transform" />
          <div>
            <div className="flex items-center gap-2">
              <span className="font-heading font-bold text-lg tracking-tight text-graphite-dark">
                INDEXORA
              </span>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-surface-secondary text-graphite border border-border">
                v1.0.0
              </span>
            </div>
            <p className="text-xs text-graphite-muted">
              First-Principles Vector Search Engine & Architecture Lab
            </p>
          </div>
        </div>

        {/* Live system state bar & Controls */}
        <div className="flex items-center gap-2 sm:gap-3 overflow-x-auto py-1 text-xs font-mono">
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

              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="font-sans font-medium text-xs">Engine Online</span>
              </div>
            </>
          ) : (
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400">
              <Activity className="w-3.5 h-3.5 animate-spin" />
              <span className="font-sans text-xs">Connecting...</span>
            </div>
          )}

          {/* Theme Toggle (Dark/Light) */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle theme"
            title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            className="p-1.5 rounded-lg bg-surface-subtle hover:bg-surface-secondary border border-border text-graphite-muted hover:text-graphite transition-all cursor-pointer flex items-center justify-center"
          >
            {isDark ? (
              <Sun className="w-3.5 h-3.5 text-amber-400" />
            ) : (
              <Moon className="w-3.5 h-3.5 text-graphite" />
            )}
          </button>

          {/* Status Refresh */}
          <button
            onClick={onRefresh}
            disabled={loading}
            title="Refresh engine status"
            className="p-1.5 rounded-lg bg-surface-subtle hover:bg-surface-secondary border border-border text-graphite-muted hover:text-graphite transition-all cursor-pointer flex items-center justify-center"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-accent' : ''}`} />
          </button>
        </div>
      </div>
    </header>
  );
};
