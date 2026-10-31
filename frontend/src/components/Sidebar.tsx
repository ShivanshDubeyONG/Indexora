import React from 'react';
import { NavTabId } from './Navigation';
import {
  Menu,
  Gauge,
  FlaskConical,
  BarChart2,
  Layers,
  Database,
  BookOpen,
  Sun,
  Moon,
  RefreshCw,
  GitBranch
} from 'lucide-react';

interface SidebarProps {
  activeTab: NavTabId;
  onChangeTab: (tab: NavTabId) => void;
  isDark: boolean;
  onToggleTheme: () => void;
  onRefresh: () => void;
  loading: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onChangeTab,
  isDark,
  onToggleTheme,
  onRefresh,
  loading,
}) => {
  const navItems: Array<{ id: NavTabId; label: string; icon: React.ReactNode }> = [
    { id: 'overview', label: 'Dashboard Overview', icon: <Gauge className="w-5 h-5" /> },
    { id: 'search', label: 'Search Lab', icon: <FlaskConical className="w-5 h-5" /> },
    { id: 'benchmarks', label: 'Benchmark Studio', icon: <BarChart2 className="w-5 h-5" /> },
    { id: 'explorer', label: 'Vector Explorer', icon: <Layers className="w-5 h-5" /> },
    { id: 'management', label: 'Storage & Mutations', icon: <Database className="w-5 h-5" /> },
    { id: 'semantic', label: 'Semantic Corpus', icon: <BookOpen className="w-5 h-5" /> },
  ];

  return (
    <aside className="w-16 flex-shrink-0 bg-[#0E1E38] text-slate-400 flex flex-col items-center justify-between py-4 select-none z-40 fixed top-0 bottom-0 left-0">
      {/* Top Section: Menu Toggle & Brand Icon */}
      <div className="flex flex-col items-center space-y-6 w-full">
        <button
          title="Indexora Navigation"
          className="text-slate-300 hover:text-white p-2 rounded-lg hover:bg-white/10 transition"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Navigation Items */}
        <nav className="flex flex-col items-center space-y-3 w-full px-2">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onChangeTab(item.id)}
                title={item.label}
                className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all cursor-pointer relative group ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-900/30'
                    : 'text-slate-400 hover:text-white hover:bg-white/10'
                }`}
              >
                {item.icon}

                {/* Hover Tooltip */}
                <div className="absolute left-14 px-2.5 py-1 rounded bg-slate-900 text-white text-xs font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity shadow-lg z-50">
                  {item.label}
                </div>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Section: Theme, Refresh, GitHub */}
      <div className="flex flex-col items-center space-y-3 w-full px-2 pt-4 border-t border-slate-700/60">
        <button
          onClick={onToggleTheme}
          title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          className="w-10 h-10 rounded-lg flex items-center justify-center text-slate-400 hover:text-white hover:bg-white/10 transition cursor-pointer"
        >
          {isDark ? <Sun className="w-5 h-5 text-amber-400" /> : <Moon className="w-5 h-5" />}
        </button>

        <button
          onClick={onRefresh}
          disabled={loading}
          title="Refresh Engine State"
          className="w-10 h-10 rounded-lg flex items-center justify-center text-slate-400 hover:text-white hover:bg-white/10 transition cursor-pointer"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-blue-400' : ''}`} />
        </button>

        <a
          href="https://github.com/ShivanshDubeyONG/Indexora"
          target="_blank"
          rel="noopener noreferrer"
          title="View GitHub Repository"
          className="w-10 h-10 rounded-lg flex items-center justify-center text-slate-400 hover:text-white hover:bg-white/10 transition"
        >
          <GitBranch className="w-4 h-4" />
        </a>
      </div>
    </aside>
  );
};
