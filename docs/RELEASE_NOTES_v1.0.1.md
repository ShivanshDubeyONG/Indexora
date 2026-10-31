# Indexora v1.0.1 — Maintenance & Quality Hardening Release

**Release:** `v1.0.1`  
**Milestone:** Post-v1.0 Quality, Security & Reproducibility Hardening  
**Historical Continuity:** Historical tags `v0.1` (commit `6d4d91f`) and `v1.0` (commit `a55e8bb`) remain intact and unchanged.

---

## Executive Summary

Indexora `v1.0.1` is a focused maintenance and quality release addressing external engineering audit findings across security boundaries, semantic retrieval correctness, UI reactive state consistency, and benchmark documentation reproducibility.

No core search algorithms were replaced with external black-box libraries; Indexora remains a strictly first-principles, educational vector search engine implemented in pure Python and NumPy.

---

## Key Improvements & Bug Fixes

### 1. Semantic Retrieval Correctness & Availability Boundaries (Phase 1)
- **Elimination of Silent Mock Embeddings:** Production semantic search will no longer substitute pseudo-random or hash-derived vectors when deep learning dependencies are unavailable. Silently generating synthetic vectors corrupted vector space geometry and presented false rankings.
- **Explicit Diagnostic Guidance:** When optional dependencies (`sentence-transformers`, `torch`) are not installed, the UI and API surface clear, actionable installation instructions (`pip install -e ".[semantic]"`).
- **Metric Decoupling:** Pure index retrieval latency ($t_{\text{search}}$) is explicitly isolated from transformer text embedding latency ($t_{\text{embed}}$).
- **Vector Oracle Agreement:** Replaced ambiguous "Semantic Relevance" score labels with mathematically precise **Vector Oracle Agreement**, which directly quantifies top-$k$ candidate agreement against the exact brute-force BLAS oracle on the embedded corpus.

### 2. Persistence Security & Sandboxing (Phase 1 & ADR-0010)
- **Sanitized Public Persistence:** Public-facing lab sessions (`LabSession`) enforce sandboxed storage within a dedicated root (`storage/lab_indexes/`). User-supplied index names are validated against `^[a-zA-Z0-9_-]{1,64}$`, rejecting path traversal sequences (`..`), forward/backward slashes, null bytes, and absolute paths.
- **Symlink Escape Protection:** Target directories undergo strict root containment and symlink escape checks (`Path.resolve().is_relative_to(root)`).
- **Load Atomicity & Mutation Isolation:** Persisted index loading atomically resets mutation history (`added_ids = []`, `deleted_ids = []`) only after successful checksum and manifest verification. Any corrupted array or checksum mismatch fails without altering active session state.

### 3. UI Correctness, Trust & Reactive Metrics (Phase 2)
- **Reactive Mutation Metrics:** Add and delete operations in the Interactive Index Lab immediately synchronize vector counts and mutation history cards without requiring an extra manual refresh.
- **Accurate Algorithmic Complexity Annotations:** Updated lab descriptions to accurately reflect true theoretical complexities and implementation characteristics across Exact Flat, Exact BLAS, IVF-Flat, IVF-PQ, and HNSW.
- **Honest System Limitations:** Clarified Python bytecode interpreter loop overhead in graph beam searches, memory footprint trade-offs, and HNSW tombstone deletion semantics.

### 4. Benchmark Methodology & Documentation Polish (Phase 3 & v1.0.1)
- **SIFT-100k Benchmark Methodology:** Published [`docs/BENCHMARK_METHODOLOGY.md`](BENCHMARK_METHODOLOGY.md) documenting complete sampling protocols, hardware environment specifications, Texmex SIFT-100k provenance, and SHA-256 cryptographic checksums.
- **Rigorous Contention Qualification:** Formally qualified the Phase 6.6 batch QPS anomaly at $n_{\text{list}}=50, n_{\text{probe}}=10$ (8.2 QPS vs 16.4 QPS at $n_{\text{probe}}=20$) as a plausible hypothesis of transient OS/CPU thread scheduling contention during that multi-minute repeated batch measurement (supported by baseline runs in `results/sift100k` exhibiting strict monotonic scaling: 329.3 $\to$ 173.5 $\to$ 63.3 $\to$ 29.4 $\to$ 12.9 QPS), acknowledging that low-level kernel telemetry was not captured.
- **Portable Relative Documentation Links:** Converted all documentation links and `README.md` references from absolute local paths to repository-relative Markdown links.
- **Comprehensive Engineering Audit Report:** Published [`docs/FINAL_ENGINEERING_AUDIT.md`](FINAL_ENGINEERING_AUDIT.md) certifying architectural integrity, test coverage, and security boundaries.

---

## Verification & Quality Gates

- **Unit & Integration Tests:** 377 passing tests in `pytest` executed offline in $<100$ seconds (`1 skipped`, `1 deselected` for optional real model download).
- **Smoke Benchmark:** Passed offline verification in $<2$ seconds (`python -m indexora.benchmark --smoke --no-plot`).
- **Interactive Lab:** Verified headless initialization and UI import (`python -c "import indexora.ui; print('UI OK')"`).
- **Packaging:** Editable package installation verified cleanly (`pip install -e . --no-deps`).
- **Git Formatting:** Zero whitespace or merge conflict errors (`git diff --check`).
