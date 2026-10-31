import React, { useState, useEffect } from 'react';
import { fetchSemanticStatus, searchSemantic } from '../api';
import { SemanticStatus, SemanticSearchResponse } from '../types';
import {
  BookOpen,
  Search,
  AlertTriangle,
  Tag,
  CheckCircle2,
  Terminal,
  ChevronDown,
  ChevronUp,
  Sparkles,
  ArrowUpRight,
  Filter,
  X
} from 'lucide-react';

const CATEGORY_STYLES: Record<string, { bg: string; text: string; border: string }> = {
  'Machine Learning': { bg: 'bg-blue-500/10', text: 'text-blue-600 dark:text-blue-400', border: 'border-blue-500/20' },
  'Databases': { bg: 'bg-emerald-500/10', text: 'text-emerald-600 dark:text-emerald-400', border: 'border-emerald-500/20' },
  'Distributed Systems': { bg: 'bg-purple-500/10', text: 'text-purple-600 dark:text-purple-400', border: 'border-purple-500/20' },
  'Algorithms': { bg: 'bg-amber-500/10', text: 'text-amber-600 dark:text-amber-400', border: 'border-amber-500/20' },
  'Security': { bg: 'bg-rose-500/10', text: 'text-rose-600 dark:text-rose-400', border: 'border-rose-500/20' },
};

export const SemanticSearchView: React.FC = () => {
  const [status, setStatus] = useState<SemanticStatus | null>(null);
  const [loadingStatus, setLoadingStatus] = useState(true);

  // Search form
  const [query, setQuery] = useState('vector indexing and hierarchical proximity graphs');
  const [category, setCategory] = useState<string>('all');
  const [indexType, setIndexType] = useState('Exact BLAS');
  const [k, setK] = useState(5);
  const [nprobe, setNprobe] = useState(2);
  const [efSearch, setEfSearch] = useState(32);

  const [searching, setSearching] = useState(false);
  const [result, setResult] = useState<SemanticSearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Corpus Reader Magic State
  const [activeCorpusTab, setActiveCorpusTab] = useState<string>('all');
  const [corpusSearchTerm, setCorpusSearchTerm] = useState<string>('');
  const [expandedDocs, setExpandedDocs] = useState<Record<number, boolean>>({});

  useEffect(() => {
    fetchSemanticStatus()
      .then((data) => setStatus(data))
      .catch((err) => console.error(err))
      .finally(() => setLoadingStatus(false));
  }, []);

  const toggleExpand = (docId: number) => {
    setExpandedDocs((prev) => ({ ...prev, [docId]: !prev[docId] }));
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    setError(null);
    try {
      const res = await searchSemantic({
        query: query.trim(),
        k: Number(k),
        category: category === 'all' ? undefined : category,
        index_type: indexType,
        nprobe: Number(nprobe),
        ef_search: Number(efSearch),
      });
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Semantic search failed');
    } finally {
      setSearching(false);
    }
  };

  const selectDocForQuery = (docTitle: string) => {
    setQuery(docTitle);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Filter corpus documents
  const filteredCorpus = status?.corpus_sample?.filter((doc) => {
    const matchesTab = activeCorpusTab === 'all' || doc.category.toLowerCase() === activeCorpusTab.toLowerCase();
    const matchesSearch = !corpusSearchTerm.trim() ||
      doc.title.toLowerCase().includes(corpusSearchTerm.toLowerCase()) ||
      doc.text.toLowerCase().includes(corpusSearchTerm.toLowerCase()) ||
      doc.tags.some((t) => t.toLowerCase().includes(corpusSearchTerm.toLowerCase()));
    return matchesTab && matchesSearch;
  }) || [];

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="bg-surface-card border border-border rounded-xl p-6 sm:p-8 shadow-card transition-colors">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-accent/10 text-accent font-mono text-xs font-semibold uppercase tracking-wider mb-2">
            Dense Neural Representation Layer
          </div>
          <h1 className="font-heading font-bold text-2xl text-graphite-dark">
            Natural Language Semantic Retrieval
          </h1>
          <p className="text-sm text-graphite-muted mt-1 max-w-2xl">
            Search across a curated technical corpus of 25 engineering documents using dense sentence embeddings
            and Indexora's native vector engines (Exact BLAS, IVF-Flat, and HNSW).
          </p>
        </div>
      </section>

      {/* Semantic Dependency Notice if not installed */}
      {!loadingStatus && (!status || !status.available) && (
        <div className="bg-surface-card border border-amber-500/30 rounded-xl p-6 shadow-card space-y-4 transition-colors">
          <div className="flex items-start gap-3">
            <div className="p-2 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div className="space-y-1.5">
              <h3 className="font-heading font-bold text-base text-graphite-dark">
                Semantic Text Search Unavailable in this Environment
              </h3>
              <p className="text-xs text-graphite leading-relaxed">
                Real semantic retrieval requires neural sentence embeddings (<code className="bg-surface-secondary px-1.5 py-0.5 rounded font-mono text-accent">sentence-transformers</code> and <code className="bg-surface-secondary px-1.5 py-0.5 rounded font-mono text-accent">PyTorch</code>).
                These optional deep-learning packages are not installed in the current cloud execution environment.
              </p>
            </div>
          </div>

          <div className="pl-11 space-y-3 text-xs">
            <div className="p-3.5 rounded-lg bg-surface-subtle border border-border space-y-1.5">
              <p className="font-semibold text-graphite flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-accent" />
                Why pseudo-random mock search is disabled:
              </p>
              <p className="text-graphite-muted leading-relaxed">
                Indexora strictly avoids substituting pseudo-random or hash-derived vectors for semantic retrieval.
                Random vectors lack semantic manifold topology, and ranking them generates misleading similarity scores and false ground-truth claims.
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-surface-subtle border border-border space-y-2">
              <p className="font-semibold text-graphite flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5 text-accent" />
                To enable real semantic search locally:
              </p>
              <div className="font-mono text-xs bg-surface-card p-2.5 rounded border border-border text-graphite">
                pip install -e ".[semantic]"
              </div>
            </div>

            <div className="p-3.5 rounded-lg bg-surface-secondary/50 border border-border text-graphite-muted">
              <strong className="text-graphite">Core Engine Status: </strong>
              Tabs 1–5 (Overview, Search Lab, Benchmarks, Vector Explorer, Index Management) remain 100% operational using pure Python and NumPy with zero deep-learning dependencies.
            </div>
          </div>
        </div>
      )}

      {/* Real Semantic Search Form (When Available) */}
      {status?.available && (
        <form onSubmit={handleSearch} className="bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-4 transition-colors">
          <div className="flex items-center justify-between border-b border-border pb-3">
            <span className="font-heading font-bold text-xs uppercase tracking-wider text-graphite flex items-center gap-2">
              <Search className="w-4 h-4 text-accent" />
              Dense Semantic Query Formulation
            </span>
            <span className="text-xs font-mono text-accent bg-accent/10 px-2 py-0.5 rounded border border-accent/20">
              Model: {status.model_name}
            </span>
          </div>

          <div className="space-y-3">
            <div>
              <label className="block text-xs font-mono text-graphite-muted mb-1">Natural Language Query</label>
              <div className="relative">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask about vector indexing, graph search, quantization..."
                  className="w-full pl-3.5 pr-10 py-2.5 rounded-lg border border-border bg-surface-subtle text-sm text-graphite focus:outline-none focus:ring-1 focus:ring-accent transition-colors"
                  required
                />
                {query && (
                  <button
                    type="button"
                    onClick={() => setQuery('')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-graphite-subtle hover:text-graphite cursor-pointer"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div>
                <label className="block text-graphite-muted mb-1 font-mono">Category Filter</label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent transition-colors"
                >
                  <option value="all">All Domains ({status.document_count})</option>
                  {status.categories.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-graphite-muted mb-1 font-mono">Index Engine</label>
                <select
                  value={indexType}
                  onChange={(e) => setIndexType(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent transition-colors"
                >
                  <option value="Exact BLAS">Exact BLAS (Oracle)</option>
                  <option value="IVF-Flat">IVF-Flat</option>
                  <option value="HNSW">HNSW</option>
                </select>
              </div>

              <div>
                <label className="block text-graphite-muted mb-1 font-mono">Top-k Results</label>
                <input
                  type="number"
                  min={1}
                  max={25}
                  value={k}
                  onChange={(e) => setK(parseInt(e.target.value) || 1)}
                  className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent transition-colors"
                />
              </div>

              {indexType === 'IVF-Flat' && (
                <div>
                  <label className="block text-graphite-muted mb-1 font-mono">nprobe</label>
                  <input
                    type="number"
                    min={1}
                    max={8}
                    value={nprobe}
                    onChange={(e) => setNprobe(parseInt(e.target.value) || 1)}
                    className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent transition-colors"
                  />
                </div>
              )}

              {indexType === 'HNSW' && (
                <div>
                  <label className="block text-graphite-muted mb-1 font-mono">efSearch</label>
                  <input
                    type="number"
                    min={4}
                    max={128}
                    value={efSearch}
                    onChange={(e) => setEfSearch(parseInt(e.target.value) || 4)}
                    className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent transition-colors"
                  />
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={searching || !query.trim()}
              className="px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white font-medium text-xs transition-all shadow-sm cursor-pointer disabled:opacity-50 flex items-center gap-2 active:scale-[0.98]"
            >
              <Search className="w-3.5 h-3.5" />
              <span>{searching ? 'Embedding & Searching...' : 'Execute Semantic Search'}</span>
            </button>
          </div>
        </form>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900 text-red-700 dark:text-red-300 text-xs">
          {error}
        </div>
      )}

      {/* Ranked Search Results (When Available) */}
      {result && (
        <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-4 transition-colors">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border pb-3">
            <h3 className="font-heading font-bold text-sm text-graphite-dark">
              Retrieved Documents ({result.ranked_documents.length} matches for "{result.query}")
            </h3>
            <div className="flex items-center gap-3 text-xs font-mono">
              <span className="text-graphite">Total: {result.total_latency_ms.toFixed(2)} ms</span>
              <span className="text-graphite-subtle">·</span>
              <span className="text-accent">Embed: {result.text_embed_time_ms.toFixed(2)} ms</span>
              <span className="text-graphite-subtle">·</span>
              <span className="text-emerald-600 dark:text-emerald-400 font-bold">Search: {result.vector_search_time_ms.toFixed(2)} ms</span>
            </div>
          </div>

          <div className="space-y-3">
            {result.ranked_documents.map((doc) => (
              <div key={doc.doc_id} className="p-4 rounded-xl bg-surface-subtle border border-border space-y-2 transition-colors">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded-full bg-accent/15 text-accent font-mono font-bold text-xs flex items-center justify-center">
                      {doc.rank}
                    </span>
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-surface-card border border-border text-graphite">
                      {doc.category}
                    </span>
                    <h4 className="font-heading font-bold text-sm text-graphite-dark">
                      {doc.title}
                    </h4>
                  </div>
                  <div className="flex items-center gap-2 text-xs font-mono">
                    <span className="text-accent font-bold">
                      {(doc.cosine_similarity * 100).toFixed(1)}% Sim
                    </span>
                    <span className="text-graphite-muted">
                      L2: {doc.squared_l2.toFixed(3)}
                    </span>
                  </div>
                </div>
                <p className="text-xs text-graphite leading-relaxed">
                  {doc.text}
                </p>
                <div className="flex items-center gap-1.5 pt-1">
                  {doc.tags.map((tag) => (
                    <span key={tag} className="text-[10px] font-mono px-2 py-0.5 rounded bg-surface-card text-graphite-muted border border-border">
                      #{tag}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* CURATED TECHNICAL DOCUMENT CORPUS (Redesigned with Magic & High Readability) */}
      <div className="bg-surface-card border border-border rounded-xl p-5 sm:p-6 shadow-card space-y-5 transition-colors">
        {/* Section Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border pb-4">
          <div>
            <div className="flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-accent" />
              <h2 className="font-heading font-bold text-sm sm:text-base text-graphite-dark">
                Curated Technical Document Corpus
              </h2>
              <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-surface-secondary text-graphite border border-border">
                25 Docs
              </span>
            </div>
            <p className="text-xs text-graphite-muted mt-0.5">
              5 Engineering Domains · 384-dimensional dense sentence embedding representations.
            </p>
          </div>

          {/* Quick Corpus Search Filter */}
          <div className="relative w-full sm:w-64">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-graphite-subtle" />
            <input
              type="text"
              placeholder="Filter topics or tags..."
              value={corpusSearchTerm}
              onChange={(e) => setCorpusSearchTerm(e.target.value)}
              className="w-full pl-8 pr-7 py-1.5 text-xs rounded-lg border border-border bg-surface-subtle text-graphite placeholder:text-graphite-subtle focus:outline-none focus:ring-1 focus:ring-accent transition-colors"
            />
            {corpusSearchTerm && (
              <button
                onClick={() => setCorpusSearchTerm('')}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-graphite-subtle hover:text-graphite cursor-pointer"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>

        {/* Domain Category Filter Tabs */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none text-xs font-mono">
          <button
            onClick={() => setActiveCorpusTab('all')}
            className={`px-3 py-1.5 rounded-lg border transition-all cursor-pointer whitespace-nowrap ${
              activeCorpusTab === 'all'
                ? 'bg-accent text-white border-accent font-semibold shadow-sm'
                : 'bg-surface-subtle hover:bg-surface-secondary text-graphite-muted hover:text-graphite border-border'
            }`}
          >
            All Categories ({status?.corpus_sample?.length || 25})
          </button>
          {status?.categories?.map((cat) => {
            const count = status.corpus_sample?.filter((d) => d.category === cat).length || 5;
            const isActive = activeCorpusTab.toLowerCase() === cat.toLowerCase();
            return (
              <button
                key={cat}
                onClick={() => setActiveCorpusTab(cat)}
                className={`px-3 py-1.5 rounded-lg border transition-all cursor-pointer whitespace-nowrap ${
                  isActive
                    ? 'bg-accent text-white border-accent font-semibold shadow-sm'
                    : 'bg-surface-subtle hover:bg-surface-secondary text-graphite-muted hover:text-graphite border-border'
                }`}
              >
                {cat} ({count})
              </button>
            );
          })}
        </div>

        {/* Interactive 2-Column Responsive Card Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filteredCorpus.length > 0 ? (
            filteredCorpus.map((doc) => {
              const isExpanded = !!expandedDocs[doc.doc_id];
              const style = CATEGORY_STYLES[doc.category] || {
                bg: 'bg-accent/10',
                text: 'text-accent',
                border: 'border-accent/20',
              };

              return (
                <div
                  key={doc.doc_id}
                  className="p-4 rounded-xl bg-surface-subtle border border-border hover:border-accent/40 hover:shadow-subtle transition-all space-y-2.5 group flex flex-col justify-between"
                >
                  <div className="space-y-2">
                    {/* Header with Category Pill, ID, and Search Shortcut */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className={`text-[11px] font-mono px-2 py-0.5 rounded-md border font-semibold ${style.bg} ${style.text} ${style.border}`}>
                          {doc.category}
                        </span>
                        <span className="text-[10px] font-mono text-graphite-subtle">
                          ID: #{doc.doc_id}
                        </span>
                      </div>

                      {/* Shortcut to query this document */}
                      <button
                        onClick={() => selectDocForQuery(doc.title)}
                        title="Load title into semantic query input"
                        className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-surface-card text-accent text-xs font-mono flex items-center gap-1 cursor-pointer"
                      >
                        <span>Query this</span>
                        <ArrowUpRight className="w-3 h-3" />
                      </button>
                    </div>

                    {/* Document Title */}
                    <h3 className="font-heading font-bold text-xs sm:text-sm text-graphite-dark leading-snug group-hover:text-accent transition-colors">
                      {doc.title}
                    </h3>

                    {/* Content Preview with Expand Toggle */}
                    <p className={`text-xs text-graphite leading-relaxed transition-all ${
                      isExpanded ? '' : 'line-clamp-2'
                    }`}>
                      {doc.text}
                    </p>
                  </div>

                  <div className="pt-1 border-t border-border/60 flex items-center justify-between">
                    {/* Tags */}
                    <div className="flex items-center gap-1 overflow-x-auto scrollbar-none py-0.5">
                      {doc.tags.slice(0, 3).map((tag) => (
                        <button
                          key={tag}
                          type="button"
                          onClick={() => setCorpusSearchTerm(tag)}
                          className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-surface-card text-graphite-muted border border-border hover:border-accent hover:text-accent transition-colors cursor-pointer whitespace-nowrap"
                        >
                          #{tag}
                        </button>
                      ))}
                      {doc.tags.length > 3 && (
                        <span className="text-[10px] font-mono text-graphite-subtle">
                          +{doc.tags.length - 3}
                        </span>
                      )}
                    </div>

                    {/* Expand / Collapse Button */}
                    <button
                      type="button"
                      onClick={() => toggleExpand(doc.doc_id)}
                      className="text-[11px] font-mono text-accent hover:underline flex items-center gap-0.5 cursor-pointer flex-shrink-0 ml-2"
                    >
                      <span>{isExpanded ? 'Less' : 'Abstract'}</span>
                      {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                    </button>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="col-span-full p-8 text-center text-xs text-graphite-muted border border-dashed border-border rounded-xl">
              No technical documents match "{corpusSearchTerm}" in {activeCorpusTab}.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
