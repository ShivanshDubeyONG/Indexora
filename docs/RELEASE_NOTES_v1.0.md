# Indexora v1.0 — Complete Portfolio Release

**Release:** `v1.0`
**Milestone:** Complete Learning-First Vector Search Engine
**Historical Note:** The `v0.1` release tag (commit `6d4d91f`, Oct 29, 2026) remains preserved as the historical milestone for the Phase 6.5 core (Exact Flat, IVF-Flat, and initial benchmark harness). This `v1.0` release encompasses the full 13-phase roadmap, delivering advanced quantization, proximity graphs, index lifecycle mutations, interactive exploration, and semantic retrieval.

---

## Executive Summary

Indexora is an educational, learning-first vector search engine built from scratch in Python and NumPy. It demystifies the internal mechanics of modern vector databases—from high-performance BLAS distance kernels to Inverted Files (IVF), Product Quantization (PQ), and Hierarchical Navigable Small Worlds (HNSW).

The core educational philosophy is **learning ownership**: foundational algorithmic logic is implemented from first principles without relying on black-box vector search libraries like FAISS or hnswlib for the core engine.

---

## What's Included in v1.0

### 1. Mathematical Distance Kernels & Exact Search (Phases 1, 2, 6.6)
- **Numerical Correctness Oracle:** Direct coordinate-difference squared Euclidean distance ($\sum (q_j - x_j)^2$) accumulated in 64-bit float precision before casting back to `float32`, eliminating catastrophic floating-point cancellation.
- **BLAS Acceleration Path:** Binomial norm expansion ($\|q - x\|^2 = \|q\|^2 + \|x\|^2 - 2(q \cdot x)$) powered by Level-3 BLAS GEMM (`np.dot(Q, X.T)`), delivering a $3.15\times$ speedup over the reference loop while clamping negative epsilon artifacts via `np.maximum(dist, 0.0)`.
- **Deterministic Top-k Selection:** Partial sorting via `np.argpartition` followed by ascending `(distance, external_id)` lexicographical tie-breaking, ensuring 100% reproducible result ordering across runs.

### 2. Approximate Nearest Neighbor (ANN) Architectures (Phases 4, 5, 9, 10)
- **IVF-Flat (`IVFFlatIndex`):** Coarse Voronoi space partitioning using vectorized $K$-means++ clustering. Probes the nearest $n_{\text{probe}}$ centroids and evaluates candidate postings lists with exact in-path distance accounting.
- **Product Quantization & IVF-PQ (`IVFPQIndex`):** Subspace decomposition ($d = M \times d_{\text{sub}}$) with 8-bit quantization codebooks (`uint8`). Features coarse-centroid residual encoding and Asymmetric Distance Computation (ADC) lookup tables (LUTs), achieving up to $15.2\times$ persistent memory compression.
- **Hierarchical Navigable Small World (`HNSWIndex`):** Multi-layer proximity graph implemented purely in Python and NumPy. Features exponential level assignment ($m_L = 1 / \ln(M)$), bounded greedy routing, and Malkov & Yashunin's Algorithm 4 heuristic spatial diversity pruning with strict $M_0 = 2M$ degree bounds.

### 3. Full Vector Lifecycle & Security Hardening (Phase 11)
- **Incremental Mutations:** Dynamic `add()` for runtime ingestion and `delete()` supporting tombstone markers in HNSW to preserve graph routing reachability without immediate structural degradation.
- **Filtered Candidate Search:** Integrated `filter_ids` parameter accepting `set[int]`, NumPy arrays, or callable predicates to filter candidates prior to distance evaluation and top-$k$ insertion.
- **Safe Atomic Persistence:** Pickle-free `.npy` numerical array serialization coupled with `manifest.json` metadata, streaming 64 KiB SHA-256 integrity checksums, and strict directory path traversal verification (`is_relative_to(path_resolved)`).

### 4. Interactive Lab & Semantic Retrieval (Phases 12A, 12B)
- **Interactive Index Lab (Streamlit):** Web playground with dark developer-tool aesthetics, 2D PCA spatial maps, real-time index mutation testing, and live recall vs. latency tradeoff sweeps across all 5 index architectures.
- **Semantic Text Search:** End-to-end natural language queries across a 25-document technical demonstration corpus. Employs sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and leverages the mathematical equivalence between unit-norm squared L2 distance and cosine similarity ($\|q-x\|^2 = 2 - 2\cos(q, x)$). Fully decoupled embedding time from index retrieval time.

### 5. Benchmark Harness & Verified SIFT-100k Results (Phases 6, 6.5, 6.6)
- **Harness Metrics:** Accurately isolates per-query latency distributions (p50, p95, p99) from full-batch throughput (QPS), tracks exact candidate distance evaluations, monitors persistent array storage vs peak process RSS, and provides distance-aware Recall@k for boundary ties.
- **Verified SIFT-100k Measurements:** Evaluated on 100,000 vectors ($d=128$, $k=10$):
  - Exact Reference: p50 = 150.69 ms, 6.2 QPS (100,000 distance ops)
  - Exact BLAS: p50 = 99.74 ms, 19.5 QPS (100,000 distance ops)
  - IVF-Flat ($n_{\text{list}}=100, n_{\text{probe}}=1$): p50 = 0.72 ms, 1,534.6 QPS, Recall@10 = 0.5287 (1,213 distance ops)
  - IVF-Flat ($n_{\text{list}}=100, n_{\text{probe}}=10$): p50 = 8.32 ms, 116.8 QPS, Recall@10 = 0.9791 (10,928 distance ops)
  - IVF-Flat ($n_{\text{list}}=100, n_{\text{probe}}=20$): p50 = 17.75 ms, 55.9 QPS, Recall@10 = 0.9980 (21,481 distance ops)

### 6. Test Suite & Verification Gates
- **329 passing unit and integration tests** in `pytest` executed completely offline in $<28$ seconds (1 opt-in live model download test deselected).
- Clean `git diff --check` with zero trailing whitespace or formatting warnings.
- CI pipeline running across Python 3.10, 3.11, and 3.12.

---

## Known Disclosures & Engineering Trade-Offs

1. **Educational & Learning-First Scope:** Indexora is engineered for transparency, mathematical rigor, and architectural learning. It is not intended as a drop-in replacement for production C++/Rust engines like FAISS or hnswlib on multi-million vector workloads.
2. **Interpreter Overhead on Small $N$:** On small collections ($N \le 1,000$), optimized C-level BLAS matrix multiplication often outperforms Python-level graph traversals or IVF list lookups due to Python interpreter loop overhead.
3. **Empirical HNSW Complexity:** HNSW query latency scales logarithmically ($O(\log N)$) empirically under fixed exploration parameters ($ef, M$); it is an approximate heuristic without universal theoretical worst-case guarantees.
4. **Hardware Specificity:** Benchmark latencies and throughput metrics reflect single-process execution on a standard x86_64 CPU workstation; numbers vary across different CPU microarchitectures, BLAS implementations (MKL vs OpenBLAS), and threading configurations.
