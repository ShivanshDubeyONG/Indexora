import React from 'react';
import {
  Compass,
  Search,
  BarChart3,
  Layers,
  Sliders,
  BookOpen
} from 'lucide-react';

export type NavTabId =
  | 'overview'
  | 'search'
  | 'benchmarks'
  | 'explorer'
  | 'management'
  | 'semantic';

interface NavProps {
  activeTab: NavTabId;
  onChangeTab: (tab: NavTabId) => void;
}

export const Navigation: React.FC<NavProps> = ({ activeTab, onChangeTab }) => {
  const tabs: Array<{ id: NavTabId; label: string; num: string; icon: React.ReactNode }> = [
    { id: 'overview', label: 'Overview', num: '1', icon: <Compass className="w-4 h-4" /> },
    { id: 'search', label: 'Search Lab', num: '2', icon: <Search className="w-4 h-4" /> },
    { id: 'benchmarks', label: 'Benchmark Studio', num: '3', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'explorer', label: 'Vector Explorer', num: '4', icon: <Layers className="w-4 h-4" /> },
    { id: 'management', label: 'Index Management', num: '5', icon: <Sliders className="w-4 h-4" /> },
    { id: 'semantic', label: 'Semantic Search', num: '6', icon: <BookOpen className="w-4 h-4" /> },
  ];

  return (
    <nav className="border-b border-border bg-surface-subtle/50 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-1.5 overflow-x-auto py-2 scrollbar-none">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onChangeTab(tab.id)}
                className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap cursor-pointer ${
                  isActive
                    ? 'bg-surface-card text-accent font-semibold border border-border shadow-subtle'
                    : 'text-graphite-muted hover:text-graphite hover:bg-surface-secondary/70 border border-transparent'
                }`}
              >
                <span className={isActive ? 'text-accent' : 'text-graphite-subtle'}>
                  {tab.icon}
                </span>
                <span>{tab.label}</span>
                <kbd className={`text-[10px] font-mono px-1 py-0.2 rounded border ${
                  isActive
                    ? 'bg-accent/10 border-accent/20 text-accent font-bold'
                    : 'bg-surface-secondary border-border/80 text-graphite-subtle'
                }`}>
                  {tab.num}
                </kbd>
              </button>
            );
          })}
        </div>
      </div>
    </nav>
  );
};
