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
  const tabs: Array<{ id: NavTabId; label: string; icon: React.ReactNode }> = [
    { id: 'overview', label: '1. Overview', icon: <Compass className="w-4 h-4" /> },
    { id: 'search', label: '2. Search Lab', icon: <Search className="w-4 h-4" /> },
    { id: 'benchmarks', label: '3. Benchmark Studio', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'explorer', label: '4. Vector Explorer', icon: <Layers className="w-4 h-4" /> },
    { id: 'management', label: '5. Index Management', icon: <Sliders className="w-4 h-4" /> },
    { id: 'semantic', label: '6. Semantic Search', icon: <BookOpen className="w-4 h-4" /> },
  ];

  return (
    <nav className="border-b border-border bg-surface-subtle/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-1.5 overflow-x-auto py-2.5 scrollbar-none">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onChangeTab(tab.id)}
                className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap cursor-pointer ${
                  isActive
                    ? 'bg-surface-card text-accent font-semibold border border-border shadow-subtle'
                    : 'text-graphite-muted hover:text-graphite hover:bg-surface-secondary/70 border border-transparent'
                }`}
              >
                <span className={isActive ? 'text-accent' : 'text-graphite-subtle'}>
                  {tab.icon}
                </span>
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </nav>
  );
};

