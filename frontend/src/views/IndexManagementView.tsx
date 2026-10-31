import React, { useState, useEffect } from 'react';
import {
  addVectorMutation,
  deleteVectorMutation,
  listSavedIndexes,
  saveNamedIndex,
  loadNamedIndex
} from '../api';
import { SystemStatus } from '../types';
import {
  Sliders,
  PlusCircle,
  Trash2,
  Save,
  FolderOpen,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  HardDrive,
  FileCode
} from 'lucide-react';

interface IndexManagementViewProps {
  status: SystemStatus | null;
  onStateChanged: () => void;
}

export const IndexManagementView: React.FC<IndexManagementViewProps> = ({ status, onStateChanged }) => {
  // Mutation state
  const [addId, setAddId] = useState(2001);
  const [constantVal, setConstantVal] = useState(0.5);
  const [customVectorStr, setCustomVectorStr] = useState('');
  const [adding, setAdding] = useState(false);
  const [addMsg, setAddMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const [delId, setDelId] = useState(1000);
  const [deleting, setDeleting] = useState(false);
  const [delMsg, setDelMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Persistence state
  const [savedIndexes, setSavedIndexes] = useState<string[]>([]);
  const [loadingList, setLoadingList] = useState(false);
  const [saveName, setSaveName] = useState('demo_snapshot_v1');
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const [selectedLoadName, setSelectedLoadName] = useState('');
  const [verifyChecksums, setVerifyChecksums] = useState(true);
  const [loadingIndex, setLoadingIndex] = useState(false);
  const [loadMsg, setLoadMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const refreshSavedList = async () => {
    setLoadingList(true);
    try {
      const res = await listSavedIndexes();
      setSavedIndexes(res.saved_indexes);
      if (res.saved_indexes.length > 0 && !selectedLoadName) {
        setSelectedLoadName(res.saved_indexes[0]);
      }
    } catch (err: any) {
      console.error(err);
    } finally {
      setLoadingList(false);
    }
  };

  useEffect(() => {
    refreshSavedList();
  }, []);

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    setAdding(true);
    setAddMsg(null);
    try {
      let vec: number[] | undefined = undefined;
      if (customVectorStr.trim()) {
        const parts = customVectorStr.split(',').map((s) => parseFloat(s.trim()));
        if (parts.some(isNaN)) throw new Error('Vector elements must be valid numbers');
        vec = parts;
      }
      const res = await addVectorMutation({
        ext_id: Number(addId),
        constant_val: vec ? undefined : Number(constantVal),
        vector: vec,
      });
      setAddMsg({ type: 'success', text: `Vector #${addId} added. Total active: ${res.active_count}` });
      setAddId((prev) => prev + 1);
      onStateChanged();
    } catch (err: any) {
      setAddMsg({ type: 'error', text: err.message || 'Failed to add vector' });
    } finally {
      setAdding(false);
    }
  };

  const handleDelete = async (e: React.FormEvent) => {
    e.preventDefault();
    setDeleting(true);
    setDelMsg(null);
    try {
      const res = await deleteVectorMutation(Number(delId));
      setDelMsg({ type: 'success', text: `Vector #${delId} deleted. Active count: ${res.active_count}` });
      onStateChanged();
    } catch (err: any) {
      setDelMsg({ type: 'error', text: err.message || 'Failed to delete vector' });
    } finally {
      setDeleting(false);
    }
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!saveName.trim()) return;
    setSaving(true);
    setSaveMsg(null);
    try {
      const res = await saveNamedIndex(saveName.trim(), true);
      setSaveMsg({ type: 'success', text: `Index snapshot "${saveName}" saved with SHA-256 manifest!` });
      refreshSavedList();
    } catch (err: any) {
      setSaveMsg({ type: 'error', text: err.message || 'Failed to save index' });
    } finally {
      setSaving(false);
    }
  };

  const handleLoad = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedLoadName) return;
    setLoadingIndex(true);
    setLoadMsg(null);
    try {
      const res = await loadNamedIndex(selectedLoadName, verifyChecksums);
      setLoadMsg({
        type: 'success',
        text: `Loaded "${res.name}" successfully (${res.active_count} active vectors, ${res.index_type}). Checksum verified!`,
      });
      onStateChanged();
    } catch (err: any) {
      setLoadMsg({ type: 'error', text: err.message || 'Failed to load index' });
    } finally {
      setLoadingIndex(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="bg-surface-card border border-border rounded-xl p-6 sm:p-8 shadow-card">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-accent/10 text-accent font-mono text-xs font-semibold uppercase tracking-wider mb-2">
            Dynamic State & Persistence Layer
          </div>
          <h1 className="font-heading font-bold text-2xl text-graphite-dark">
            Index Mutations & Sandboxed Persistence
          </h1>
          <p className="text-sm text-graphite-muted mt-1 max-w-2xl">
            Execute real-time vector insertions and deletions against live indexing structures.
            Save full state manifests to isolated storage (<code className="text-accent font-mono">storage/lab_indexes/</code>)
            with strict cryptographic SHA-256 data integrity validation.
          </p>
        </div>
      </section>

      {/* Grid: Mutations Left, Persistence Right */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* LEFT COLUMN: LIVE MUTATIONS */}
        <div className="space-y-6">
          {/* Add Vector Panel */}
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-4">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <span className="font-heading font-bold text-xs uppercase tracking-wider text-graphite flex items-center gap-2">
                <PlusCircle className="w-4 h-4 text-emerald-600" />
                Incremental Vector Insertion
              </span>
              <span className="text-xs font-mono text-graphite-muted">
                d={status?.dataset.d || 8} Dim
              </span>
            </div>

            <form onSubmit={handleAdd} className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-graphite-muted mb-1 font-mono">External ID</label>
                  <input
                    type="number"
                    value={addId}
                    onChange={(e) => setAddId(parseInt(e.target.value) || 0)}
                    className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent"
                    required
                  />
                </div>
                <div>
                  <label className="block text-graphite-muted mb-1 font-mono">Constant Fill Value</label>
                  <input
                    type="number"
                    step="0.05"
                    value={constantVal}
                    onChange={(e) => setConstantVal(parseFloat(e.target.value) || 0)}
                    disabled={!!customVectorStr.trim()}
                    className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent disabled:opacity-50"
                  />
                </div>
              </div>

              <div>
                <label className="block text-graphite-muted mb-1 font-mono">
                  Custom Coordinate Vector (optional, comma-separated)
                </label>
                <input
                  type="text"
                  placeholder="e.g. 0.12, -0.45, 0.88, 0.23..."
                  value={customVectorStr}
                  onChange={(e) => setCustomVectorStr(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent"
                />
              </div>

              <button
                type="submit"
                disabled={adding}
                className="w-full py-2 px-4 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-medium text-xs transition-all shadow-sm cursor-pointer disabled:opacity-50 flex items-center justify-center gap-2"
              >
                <PlusCircle className="w-3.5 h-3.5" />
                <span>{adding ? 'Inserting...' : 'Insert Vector Into Active Index'}</span>
              </button>

              {addMsg && (
                <div className={`p-2.5 rounded-lg text-xs font-mono ${
                  addMsg.type === 'success' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-red-50 text-red-700 border border-red-200'
                }`}>
                  {addMsg.text}
                </div>
              )}
            </form>
          </div>

          {/* Delete Vector Panel */}
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-4">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <span className="font-heading font-bold text-xs uppercase tracking-wider text-graphite flex items-center gap-2">
                <Trash2 className="w-4 h-4 text-rose-600" />
                Vector Deletion & Compaction
              </span>
              <span className="text-xs font-mono text-graphite-muted">
                Tombstone / Posting Pruning
              </span>
            </div>

            <form onSubmit={handleDelete} className="space-y-3 text-xs">
              <div>
                <label className="block text-graphite-muted mb-1 font-mono">External ID to Delete</label>
                <input
                  type="number"
                  value={delId}
                  onChange={(e) => setDelId(parseInt(e.target.value) || 0)}
                  className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={deleting}
                className="w-full py-2 px-4 rounded-lg bg-rose-600 hover:bg-rose-700 text-white font-medium text-xs transition-all shadow-sm cursor-pointer disabled:opacity-50 flex items-center justify-center gap-2"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>{deleting ? 'Deleting...' : 'Delete Vector'}</span>
              </button>

              {delMsg && (
                <div className={`p-2.5 rounded-lg text-xs font-mono ${
                  delMsg.type === 'success' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-red-50 text-red-700 border border-red-200'
                }`}>
                  {delMsg.text}
                </div>
              )}
            </form>

            <div className="p-3 bg-surface-subtle rounded-lg border border-border text-[11px] text-graphite-muted space-y-1">
              <p className="font-semibold text-graphite">Architectural Deletion Mechanics:</p>
              <ul className="list-disc list-inside space-y-0.5 font-mono text-[10px]">
                <li><strong className="text-graphite">Exact Flat / BLAS:</strong> Immediate contiguous array slicing and memory compaction.</li>
                <li><strong className="text-graphite">IVF-Flat / IVF-PQ:</strong> Voronoi posting list pruning and codebook ID alignment.</li>
                <li><strong className="text-graphite">HNSW:</strong> Soft tombstone flagging to preserve proximity graph navigation invariants.</li>
              </ul>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: PERSISTENCE & INTEGRITY */}
        <div className="space-y-6">
          {/* Save Snapshot */}
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-4">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <span className="font-heading font-bold text-xs uppercase tracking-wider text-graphite flex items-center gap-2">
                <Save className="w-4 h-4 text-accent" />
                Save Index Snapshot
              </span>
              <span className="text-xs font-mono text-accent bg-accent/10 px-2 py-0.5 rounded">
                SHA-256 Manifest
              </span>
            </div>

            <form onSubmit={handleSave} className="space-y-3 text-xs">
              <div>
                <label className="block text-graphite-muted mb-1 font-mono">Snapshot Name</label>
                <input
                  type="text"
                  value={saveName}
                  onChange={(e) => setSaveName(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={saving || !saveName.trim()}
                className="w-full py-2 px-4 rounded-lg bg-accent hover:bg-accent-hover text-white font-medium text-xs transition-all shadow-sm cursor-pointer disabled:opacity-50 flex items-center justify-center gap-2"
              >
                <Save className="w-3.5 h-3.5" />
                <span>{saving ? 'Writing Snapshot...' : 'Write Atomic Snapshot'}</span>
              </button>

              {saveMsg && (
                <div className={`p-2.5 rounded-lg text-xs font-mono ${
                  saveMsg.type === 'success' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-red-50 text-red-700 border border-red-200'
                }`}>
                  {saveMsg.text}
                </div>
              )}
            </form>
          </div>

          {/* Load Snapshot & Verify */}
          <div className="bg-surface-card border border-border rounded-xl p-5 shadow-card space-y-4">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <span className="font-heading font-bold text-xs uppercase tracking-wider text-graphite flex items-center gap-2">
                <FolderOpen className="w-4 h-4 text-accent" />
                Restore Snapshot & Verify
              </span>
              <span className="text-xs font-mono text-graphite-muted">
                {savedIndexes.length} Found
              </span>
            </div>

            <form onSubmit={handleLoad} className="space-y-3 text-xs">
              <div>
                <label className="block text-graphite-muted mb-1 font-mono">Select Saved Index</label>
                <select
                  value={selectedLoadName}
                  onChange={(e) => setSelectedLoadName(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-surface-subtle font-mono text-graphite focus:outline-none focus:ring-1 focus:ring-accent"
                >
                  {savedIndexes.length === 0 && <option value="">No snapshots found in storage</option>}
                  {savedIndexes.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex items-center gap-2 py-1">
                <input
                  type="checkbox"
                  id="chkVerify"
                  checked={verifyChecksums}
                  onChange={(e) => setVerifyChecksums(e.target.checked)}
                  className="rounded border-border text-accent focus:ring-accent cursor-pointer"
                />
                <label htmlFor="chkVerify" className="text-graphite font-mono cursor-pointer flex items-center gap-1.5">
                  <ShieldCheck className="w-3.5 h-3.5 text-accent" />
                  <span>Verify SHA-256 Checksums on Load</span>
                </label>
              </div>

              <button
                type="submit"
                disabled={loadingIndex || !selectedLoadName}
                className="w-full py-2 px-4 rounded-lg bg-surface-secondary hover:bg-surface-subtle text-graphite font-medium text-xs border border-border transition-all cursor-pointer disabled:opacity-50 flex items-center justify-center gap-2"
              >
                <FolderOpen className="w-3.5 h-3.5 text-accent" />
                <span>{loadingIndex ? 'Verifying & Loading...' : 'Load & Verify Snapshot'}</span>
              </button>

              {loadMsg && (
                <div className={`p-2.5 rounded-lg text-xs font-mono ${
                  loadMsg.type === 'success' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-red-50 text-red-700 border border-red-200'
                }`}>
                  {loadMsg.text}
                </div>
              )}
            </form>

            <div className="p-3 bg-surface-subtle rounded-lg border border-border text-[11px] text-graphite-muted space-y-1">
              <p className="font-semibold text-graphite flex items-center gap-1.5">
                <FileCode className="w-3.5 h-3.5 text-accent" />
                On-Disk Persistence Layout:
              </p>
              <pre className="font-mono text-[10px] text-graphite-muted overflow-x-auto bg-surface-card p-2 rounded border border-border">
{`storage/lab_indexes/<snapshot_name>/
├── index_metadata.json    (Hyperparameters, active count, d, index_type)
├── manifest.sha256        (Cryptographic SHA-256 hash digest of all binaries)
├── vectors.npy            (Memory-mapped dense continuous array)
└── ids.npy                (64-bit external ID array)`}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
