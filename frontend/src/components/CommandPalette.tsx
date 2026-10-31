import React, { useState, useEffect } from 'react';
import { NavTabId } from './Navigation';
import {
  Search,
  Compass,
  BarChart3,
  Layers,
  Sliders,
  BookOpen,
  Moon,
  Sun,
  RefreshCw,
  ArrowRight,
  Zap,
  HardDrive
} from 'lucide-react';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectTab: (tab: NavTabId) => void;
  onToggleTheme: () => void;
  onRefresh: () => void;
  isDark: boolean;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  onSelectTab,
  onToggleTheme,
  onRefresh,
  isDark,
}) => {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);

  const actions = [
    {
      id: 'tab-overview',
      category: 'Navigation',
      title: 'Go to Overview & Architecture Pipeline',
      shortcut: '1',
      icon: <Compass className="w-4 h-4 text-accent" />,
      run: () => onSelectTab('overview'),
    },
    {
      id: 'tab-search',
      category: 'Navigation',
      title: 'Go to Interactive Search Lab',
      shortcut: '2',
      icon: <Search className="w-4 h-4 text-accent" />,
      run: () => onSelectTab('search'),
    },
    {
      id: 'tab-benchmarks',
      category: 'Navigation',
      title: 'Go to Benchmark Studio & SIFT Evidence',
      shortcut: '3',
      icon: <BarChart3 className="w-4 h-4 text-accent" />,
      run: () => onSelectTab('benchmarks'),
    },
    {
      id: 'tab-explorer',
      category: 'Navigation',
      title: 'Go to Vector Explorer (2D PCA Manifold)',
      shortcut: '4',
      icon: <Layers className="w-4 h-4 text-accent" />,
      run: () => onSelectTab('explorer'),
    },
    {
      id: 'tab-management',
      category: 'Navigation',
      title: 'Go to Index Management & Snapshots',
      shortcut: '5',
      icon: <Sliders className="w-4 h-4 text-accent" />,
      run: () => onSelectTab('management'),
    },
    {
      id: 'tab-semantic',
      category: 'Navigation',
      title: 'Go to Natural Language Semantic Search',
      shortcut: '6',
      icon: <BookOpen className="w-4 h-4 text-accent" />,
      run: () => onSelectTab('semantic'),
    },
    {
      id: 'action-theme',
      category: 'Preferences',
      title: isDark ? 'Switch to Light Mode' : 'Switch to Obsidian Dark Mode',
      shortcut: 'D',
      icon: isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-graphite" />,
      run: onToggleTheme,
    },
    {
      id: 'action-refresh',
      category: 'System',
      title: 'Refresh Engine State & Active Index',
      shortcut: 'R',
      icon: <RefreshCw className="w-4 h-4 text-emerald-500" />,
      run: onRefresh,
    },
  ];

  const filtered = actions.filter((act) =>
    act.title.toLowerCase().includes(query.toLowerCase()) ||
    act.category.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => (prev + 1) % (filtered.length || 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) => (prev - 1 + filtered.length) % (filtered.length || 1));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filtered[selectedIndex]) {
          filtered[selectedIndex].run();
          onClose();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filtered, selectedIndex, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Palette Card */}
      <div className="relative w-full max-w-xl bg-surface-card border border-border rounded-xl shadow-2xl overflow-hidden z-10 animate-fade-in font-sans">
        {/* Search Input Bar */}
        <div className="flex items-center px-4 py-3.5 border-b border-border bg-surface-subtle/50">
          <Search className="w-4 h-4 text-accent mr-3 flex-shrink-0" />
          <input
            autoFocus
            type="text"
            placeholder="Type a command or jump to tool... (Esc to exit)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full bg-transparent border-none text-sm text-graphite placeholder:text-graphite-subtle focus:outline-none"
          />
          <kbd className="px-2 py-0.5 rounded bg-surface-secondary border border-border text-[10px] font-mono text-graphite-muted">
            ESC
          </kbd>
        </div>

        {/* Command Items List */}
        <div className="max-h-80 overflow-y-auto p-2 space-y-1">
          {filtered.length > 0 ? (
            filtered.map((item, idx) => {
              const isSelected = idx === selectedIndex;
              return (
                <div
                  key={item.id}
                  onClick={() => {
                    item.run();
                    onClose();
                  }}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={`flex items-center justify-between px-3 py-2.5 rounded-lg text-xs cursor-pointer transition-colors ${
                    isSelected
                      ? 'bg-accent text-white font-medium shadow-subtle'
                      : 'text-graphite hover:bg-surface-secondary'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <span className={isSelected ? 'text-white' : ''}>{item.icon}</span>
                    <span className="font-sans">{item.title}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-[10px] uppercase font-mono px-1.5 py-0.5 rounded ${
                      isSelected ? 'bg-white/20 text-white' : 'bg-surface-subtle text-graphite-muted'
                    }`}>
                      {item.category}
                    </span>
                    <kbd className={`text-[11px] font-mono px-1.5 py-0.5 rounded border ${
                      isSelected ? 'border-white/30 text-white' : 'border-border text-graphite-subtle'
                    }`}>
                      {item.shortcut}
                    </kbd>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="p-6 text-center text-xs text-graphite-muted font-mono">
              No matching commands or tools found for "{query}".
            </div>
          )}
        </div>

        {/* Footer Navigation Tip */}
        <div className="px-4 py-2 border-t border-border bg-surface-subtle flex items-center justify-between text-[11px] font-mono text-graphite-muted">
          <span>Navigate with <kbd className="text-graphite">↑</kbd> <kbd className="text-graphite">↓</kbd></span>
          <span>Press <kbd className="text-graphite">Enter</kbd> to execute</span>
        </div>
      </div>
    </div>
  );
};
