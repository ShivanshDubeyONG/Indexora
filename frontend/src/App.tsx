import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Navigation, NavTabId } from './components/Navigation';
import { Footer } from './components/Footer';
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

  const loadState = async () => {
    setLoading(true);
    try {
      const [s, d] = await Promise.all([
        fetchSystemStatus(),
        fetchCurrentDataset(),
      ]);
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

  return (
    <div className="min-h-screen flex flex-col bg-surface text-graphite antialiased font-sans selection:bg-accent/20 selection:text-accent">
      <Header status={status} loading={loading} onRefresh={loadState} />
      <Navigation activeTab={activeTab} onChangeTab={setActiveTab} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'overview' && (
          <OverviewView 
            status={status} 
            dataset={dataset} 
            onDatasetUpdated={loadState} 
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
  );
};
