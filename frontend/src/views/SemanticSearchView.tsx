import React, { useState, useEffect } from 'react';
import { fetchSemanticStatus, searchSemantic } from '../api';
import { SemanticStatus, SemanticSearchResponse } from '../types';
import { 
  BookOpen, 
  Search, 
  AlertTriangle, 
  Clock, 
  Layers, 
  Tag, 
  CheckCircle2, 
  SlidersHorizontal,
  Terminal,
  ExternalLink
} from 'lucide-react';

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

  useEffect(() => {
    fetchSemanticStatus()
      .then((data) => setStatus(data))
      .catch((err) => console.error(err))
      .finally(() => setLoadingStatus(false));
  }, []);

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

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="bg-surface-card border border-border rounded-xl p-6 sm:p-8 shadow-card">
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
        <div className="bg-surface-card border border-amber-200/80 rounded-xl p-6 shadow-card space-y-4">
          <div className="flex items-start gap-3">
            <div className="p-2 rounded-lg bg-amber-50 text-amber-700 flex-shrink-0 mt-0.5">
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
        <form onSubmit={handleSearch} className="bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-4">
          <div className="flex items-center justify-between border-b border-border pb-3">
            <span className="font-heading font-bold text-xs uppercase tracking-wider text-graphite flex items-center gap-2">
              <Search className="w-4 h-4 text-accent" />
              Dense Semantic Query Formulation
            </span>
            <span className="text-xs font-mono text-accent bg-accent/10 px-2 py-0.5 rounded">
              Model: {status.model_name}
            </span>
          </div>

          <div className="space-y-3">
            <div>
              <label className="block text-xs font-mono text-graphite-muted mb-1">Natural Language Query</label>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask about vector indexing, graph search, quantization..."
                className="w-full px-3.5 py-2 rounded-lg border border-border bg-surface-subtle text-sm text-graphite focus:outline-none focus:ring-1 focus:ring-accent"
                required
              />
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div>
                <label className="block text-graphite-muted mb-1 font-mono">Category Filter</label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent"
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
                  className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent"
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
                  className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent"
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
                    className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent"
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
                    className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent"
                  />
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={searching || !query.trim()}
              className="px-5 py-2 rounded-lg bg-accent hover:bg-accent-hover text-white font-medium text-xs transition-all shadow-sm cursor-pointer disabled:opacity-50 flex items-center gap-2"
            >
              <Search className="w-3.5 h-3.5" />
              <span>{searching ? 'Embedding & Searching...' : 'Execute Semantic Search'}</span>
            </button>
          </div>
        </form>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs">
          {error}
        </div>
      )}

      {/* Ranked Search Results (When Available) */}
      {result && (
        <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border pb-3">
            <h3 className="font-heading font-bold text-sm text-graphite-dark">
              Retrieved Documents ({result.ranked_documents.length} matches for "{result.query}")
            </h3>
            <div className="flex items-center gap-3 text-xs font-mono">
              <span className="text-graphite">Total: {result.total_latency_ms.toFixed(2)} ms</span>
              <span className="text-graphite-subtle">·</span>
              <span className="text-accent">Embed: {result.text_embed_time_ms.toFixed(2)} ms</span>
              <span className="text-graphite-subtle">·</span>
              <span className="text-emerald-700 font-bold">Search: {result.vector_search_time_ms.toFixed(2)} ms</span>
            </div>
          </div>

          <div className="space-y-3">
            {result.ranked_documents.map((doc) => (
              <div key={doc.doc_id} className="p-4 rounded-lg bg-surface-subtle border border-border space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded-full bg-accent/10 text-accent font-mono font-bold text-xs flex items-center justify-center">
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

      {/* Curated Technical Corpus Preview */}
      <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-4">
        <div className="flex items-center justify-between border-b border-border pb-3">
          <span className="font-heading font-bold text-xs uppercase tracking-wider text-graphite flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-accent" />
            Curated Technical Document Corpus (25 Engineering Documents)
          </span>
          <span className="text-xs font-mono text-graphite-muted">
            5 Domains · 384-d Dense Embedding Space
          </span>
        </div>

        <div className="space-y-3">
          {status?.corpus_sample ? (
            status.corpus_sample.map((doc) => (
              <div key={doc.doc_id} className="p-3.5 rounded-lg bg-surface-subtle border border-border space-y-1.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-surface-card border border-border text-accent font-semibold">
                      {doc.category}
                    </span>
                    <span className="font-heading font-bold text-xs text-graphite-dark">
                      {doc.title}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-graphite-muted">ID: #{doc.doc_id}</span>
                </div>
                <p className="text-xs text-graphite leading-relaxed">
                  {doc.text}
                </p>
                <div className="flex items-center gap-1.5 pt-1">
                  {doc.tags.map((t) => (
                    <span key={t} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-surface-card text-graphite-muted border border-border">
                      #{t}
                    </span>
                  ))}
                </div>
              </div>
            ))
          ) : (
            <div className="text-xs text-graphite-muted font-mono py-4 text-center">
              Loading corpus catalog...
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
