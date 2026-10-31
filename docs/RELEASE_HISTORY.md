# Indexora Release History

This document chronicles the release milestones, architectural additions, and quality hardening passes across Indexora releases.

---

## Release v1.0.1 — Maintenance & Quality Hardening Release

**Tag / Commit:** `v1.0.1`
**Milestone:** Post-v1.0 Quality, Security & Reproducibility Hardening
**Historical Continuity:** Historical tags `v0.1` (commit `6d4d91f`) and `v1.0` (commit `a55e8bb`) remain intact and immutable.

### Executive Summary
Indexora `v1.0.1` is a focused maintenance and quality release addressing external engineering audit findings across security boundaries, semantic retrieval correctness, UI reactive state consistency, and benchmark documentation reproducibility.

No core search algorithms were replaced with external black-box libraries; Indexora remains a strictly first-principles, educational vector search engine implemented in pure Python and NumPy.

### Key Improvements & Bug Fixes

#### 1. Semantic Retrieval Correctness & Availability Boundaries
- **Elimination of Silent Mock Embeddings:** Production semantic search will no longer substitute pseudo-random or hash-derived vectors when deep learning dependencies are unavailable. Silently generating synthetic vectors corrupted vector space geometry and presented false rankings.
- **Explicit Diagnostic Guidance:** When optional dependencies (`sentence-transformers`, `torch`) are not installed, the UI and API surface clear, actionable installation instructions (`pip install -e ".[semantic]"`).
- **Metric Decoupling:** Pure index retrieval latency ($t_{\text{search}}$) is explicitly isolated from transformer text embedding latency ($t_{\text{embed}}$).
- **Vector Oracle Agreement:** Replaced ambiguous "Semantic Relevance" score labels with mathematically precise **Vector Oracle Agreement**, which directly quantifies top-$k$ candidate agreement against the exact brute-force BLAS oracle on the embedded corpus.

#### 2. Persistence Security & Sandboxing (ADR-0010)
- **Sanitized Public Persistence:** Public-facing lab sessions (`LabSession`) enforce sandboxed storage within a dedicated root (`storage/lab_indexes/`). User-supplied index names are validated against `^[a-zA-Z0-9_-]{1,64}$`, rejecting path traversal sequences (`..`), forward/backward slashes, null bytes, and absolute paths.
- **Symlink Escape Protection:** Target directories undergo strict root containment and symlink escape checks (`Path.resolve().is_relative_to(root)`).
- **Load Atomicity & Mutation Isolation:** Persisted index loading atomically resets mutation history (`added_ids = []`, `deleted_ids = []`) only after successful checksum and manifest verification. Any corrupted array or checksum mismatch fails without altering active session state.

#### 3. UI Correctness, Trust & Reactive Metrics
- **Reactive Mutation Metrics:** Add and delete operations in the Interactive Index Lab immediately synchronize vector counts and mutation history cards without requiring an extra manual refresh.
- **Accurate Algorithmic Complexity Annotations:** Updated lab descriptions to accurately reflect true theoretical complexities and implementation characteristics across Exact Flat, Exact BLAS, IVF-Flat, IVF-PQ, and HNSW.
- **Honest System Limitations:** Clarified Python bytecode interpreter loop overhead in graph beam searches, memory footprint trade-offs, and HNSW tombstone deletion semantics.

#### 4. Benchmark Methodology & Documentation Polish
- **SIFT-100k Benchmark Methodology:** Published [`docs/BENCHMARK_METHODOLOGY.md`](BENCHMARK_METHODOLOGY.md) documenting complete sampling protocols, hardware environment specifications, Texmex SIFT-100k provenance, and SHA-256 cryptographic checksums.
- **Rigorous Contention Qualification:** Formally qualified the Phase 6.6 batch QPS anomaly at $n_{\text{list}}=50, n_{\text{probe}}=10$ (8.2 QPS vs 16.4 QPS at $n_{\text{probe}}=20$) as a plausible hypothesis of transient OS/CPU thread scheduling contention during that multi-minute repeated batch measurement (supported by baseline runs in `results/sift100k` exhibiting strict monotonic scaling: 329.3 $\to$ 173.5 $\to$ 63.3 $\to$ 29.4 $\to$ 12.9 QPS), acknowledging that low-level kernel telemetry was not captured.
- **Portable Relative Documentation Links:** Converted all documentation links and `README.md` references from absolute local paths to repository-relative Markdown links.

---

## Release v1.0 — Complete Portfolio Release

**Tag / Commit:** `v1.0` (commit `a55e8bb`)
**Milestone:** Complete Learning-First Vector Search Engine
**Historical Note:** The `v0.1` release tag (commit `6d4d91f`, Oct 29, 2026) remains preserved as the historical milestone for the Phase 6.5 core (Exact Flat, IVF-Flat, and initial benchmark harness). The `v1.0` release encompassed the full 13-phase roadmap, delivering advanced quantization, proximity graphs, index lifecycle mutations, interactive exploration, and semantic retrieval.

### Executive Summary
Indexora is an educational, learning-first vector search engine built from scratch in Python and NumPy. It demystifies the internal mechanics of modern vector databases—from high-performance BLAS distance kernels to Inverted Files (IVF), Product Quantization (PQ), and Hierarchical Navigable Small Worlds (HNSW).

The core educational philosophy is **learning ownership**: foundational algorithmic logic is implemented from first principles without relying on black-box vector search libraries like FAISS or hnswlib for the core engine.

### What's Included in v1.0

#### 1. Mathematical Distance Kernels & Exact Search
- **Numerical Correctness Oracle:** Direct coordinate-difference squared Euclidean distance ($\sum (q_j - x_j)^2$) accumulated in 64-bit float precision before casting back to `float32`, eliminating catastrophic floating-point cancellation.
- **BLAS Acceleration Path:** Binomial norm expansion ($\|q - x\|^2 = \|q\|^2 + \|x\|^2 - 2(q \cdot x)$) powered by Level-3 BLAS GEMM (`np.dot(Q, X.T)`), delivering substantial speedups over the reference loop while clamping negative epsilon artifacts via `np.maximum(dist, 0.0)`.
- **Deterministic Top-k Selection:** Partial sorting via `np.argpartition` followed by ascending `(distance, external_id)` lexicographical tie-breaking, ensuring 100% reproducible result ordering across runs.

#### 2. Approximate Nearest Neighbor (ANN) Architectures
- **IVF-Flat (`IVFFlatIndex`):** Coarse Voronoi space partitioning using vectorized $K$-means++ clustering. Probes the nearest $n_{\text{probe}}$ centroids and evaluates candidate postings lists with exact in-path distance accounting.
- **Product Quantization & IVF-PQ (`IVFPQIndex`):** Subspace decomposition ($d = M \times d_{\text{sub}}$) with 8-bit quantization codebooks (`uint8`). Features coarse-centroid residual encoding and Asymmetric Distance Computation (ADC) lookup tables (LUTs), achieving up to $15.2\times$ persistent memory compression.
- **Hierarchical Navigable Small World (`HNSWIndex`):** Multi-layer proximity graph implemented purely in Python and NumPy. Features exponential level assignment ($m_L = 1 / \ln(M)$), bounded greedy routing, and Malkov & Yashunin's Algorithm 4 heuristic spatial diversity pruning with strict $M_0 = 2M$ degree bounds.

#### 3. Full Vector Lifecycle & Persistence
- **Incremental Mutations:** Dynamic `add()` for runtime ingestion and `delete()` supporting tombstone markers in HNSW to preserve graph routing reachability without immediate structural degradation.
- **Filtered Candidate Search:** Integrated `filter_ids` parameter accepting `set[int]`, NumPy arrays, or callable predicates to filter candidates prior to distance evaluation and top-$k$ insertion.
- **Safe Atomic Persistence:** Pickle-free `.npy` numerical array serialization coupled with `manifest.json` metadata, streaming 64 KiB SHA-256 integrity checksums, and strict directory path traversal verification.

#### 4. Interactive Lab & Semantic Retrieval
- **Interactive Index Lab (Streamlit):** Web playground with dark developer-tool aesthetics, 2D PCA spatial maps, real-time index mutation testing, and live recall vs. latency tradeoff sweeps across all 5 index architectures.
- **Semantic Text Search:** End-to-end natural language queries across a 25-document technical demonstration corpus using sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and unit-norm squared L2 distance to cosine similarity mapping ($\|q-x\|^2 = 2 - 2\cos(q, x)$).

#### 5. Benchmark Harness & Verified SIFT-100k Results
- **Harness Metrics:** Accurately isolates per-query latency distributions (p50, p95, p99) from full-batch throughput (QPS), tracks exact candidate distance evaluations, monitors persistent array storage vs peak process RSS, and provides distance-aware Recall@k for boundary ties.

---

## Release v0.1 — Core Engine Proof of Concept

**Tag / Commit:** `v0.1` (commit `6d4d91f`, Oct 29, 2026)
**Milestone:** Initial Core Release (Phase 6.5)

### Summary
The initial baseline release establishing:
- Coordinate-difference reference distance kernel (`pairwise_squared_l2_ref`)
- Vectorized BLAS distance kernel (`pairwise_squared_l2_blas`)
- Exact exhaustive scan indexes (`ExactFlatIndex`, `ExactBlasIndex`)
- Inverted File Flat index (`IVFFlatIndex`) with $K$-means++ clustering
- Baseline SIFT-100K benchmark harness and initial automated test suite.
