import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { CommandPalette } from './components/CommandPalette';
import { NavTabId } from './components/Navigation';
import { OverviewView } from './views/OverviewView';
import { SearchLabView } from './views/SearchLabView';
import { BenchmarkStudioView } from './views/BenchmarkStudioView';
import { VectorExplorerView } from './views/VectorExplorerView';
import { IndexManagementView } from './views/IndexManagementView';
import { SemanticSearchView } from './views/SemanticSearchView';
import { fetchSystemStatus, fetchCurrentDataset } from './api';
import { SystemStatus, DatasetCurrent } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTabId>('overview');
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [dataset, setDataset] = useState<DatasetCurrent | null>(null);
  const [loading, setLoading] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);

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

  const toggleTheme = () => {
    setIsDark((prev) => {
      const next = !prev;
      const root = document.documentElement;
      if (next) {
        root.classList.add('dark');
        localStorage.setItem('indexora_theme', 'dark');
      } else {
        root.classList.remove('dark');
        localStorage.setItem('indexora_theme', 'light');
      }
      return next;
    });
  };

  const loadState = async () => {
    setLoading(true);
    const start = performance.now();
    try {
      const [s, d] = await Promise.all([
        fetchSystemStatus(),
        fetchCurrentDataset(),
      ]);
      const elapsed = performance.now() - start;
      if (elapsed < 350) {
        await new Promise((r) => setTimeout(r, 350 - elapsed));
      }
      setStatus(s);
      setDataset(d);
    } catch (err) {
      console.error('Failed to load system state:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadState();
  }, []);

  // Global Keyboard Shortcuts (Ctrl+K, 1-6 tabs, D theme, R refresh)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setPaletteOpen((prev) => !prev);
        return;
      }

      const activeEl = document.activeElement;
      const isInput = activeEl && (
        activeEl.tagName === 'INPUT' ||
        activeEl.tagName === 'TEXTAREA' ||
        activeEl.tagName === 'SELECT'
      );
      if (isInput) return;

      if (e.key === '1') setActiveTab('overview');
      else if (e.key === '2') setActiveTab('search');
      else if (e.key === '3') setActiveTab('benchmarks');
      else if (e.key === '4') setActiveTab('explorer');
      else if (e.key === '5') setActiveTab('management');
      else if (e.key === '6') setActiveTab('semantic');
      else if (e.key.toLowerCase() === 'd') toggleTheme();
      else if (e.key.toLowerCase() === 'r') loadState();
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="min-h-screen flex bg-surface text-graphite antialiased font-sans selection:bg-accent/20 selection:text-accent transition-colors">
      {/* Slim Vertical Icon Rail (Modeled after the Salesforce/Enterprise dashboard screenshot) */}
      <Sidebar
        activeTab={activeTab}
        onChangeTab={setActiveTab}
        isDark={isDark}
        onToggleTheme={toggleTheme}
        onRefresh={loadState}
        loading={loading}
      />

      {/* Main Content Column */}
      <div className="flex-1 flex flex-col pl-16 min-w-0 transition-all">
        <Header
          status={status}
          loading={loading}
          onRefresh={loadState}
          onOpenPalette={() => setPaletteOpen(true)}
        />

        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {activeTab === 'overview' && (
            <OverviewView
              status={status}
              dataset={dataset}
              onDatasetUpdated={loadState}
              onNavigateToSearch={() => setActiveTab('search')}
            />
          )}
          {activeTab === 'search' && (
            <SearchLabView
              dataset={dataset}
            />
          )}
          {activeTab === 'benchmarks' && (
            <BenchmarkStudioView
              dataset={dataset}
            />
          )}
          {activeTab === 'explorer' && (
            <VectorExplorerView
              dataset={dataset}
            />
          )}
          {activeTab === 'management' && (
            <IndexManagementView
              status={status}
              onStateChanged={loadState}
            />
          )}
          {activeTab === 'semantic' && (
            <SemanticSearchView />
          )}
        </main>

        <Footer />
      </div>

      {/* Global Command Palette */}
      <CommandPalette
        isOpen={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        onSelectTab={(tab) => {
          setActiveTab(tab);
          setPaletteOpen(false);
        }}
        onToggleTheme={toggleTheme}
        onRefresh={loadState}
        isDark={isDark}
      />
    </div>
  );
};
