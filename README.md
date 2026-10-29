# Indexora

A learning-first vector search engine built from scratch in Python and NumPy.

## Purpose

Indexora explores the internals of vector indexing and approximate nearest-neighbor (ANN) search from first principles. It begins with a trustworthy exact-search baseline and reproducible evaluation, progressively adding algorithms such as k-means, IVF-Flat, Product Quantization (PQ), IVF-PQ, and HNSW to benchmark recall, latency, memory footprint, and build cost.

The core educational philosophy is **learning ownership**: the core algorithm bodies are implemented and understood directly by the human owner, while supporting engineering (packaging, benchmarks, adapters, test harnesses) is agent-assisted.

## Current Status

- **Phase 0 (Setup):** Complete. Minimal package layout, dependency configuration, smoke tests, conventions, and CI workflow established.
- **Phase 1 (Distance Kernels & Top-k):** Complete. Reference squared-L2 kernel with float64 accumulation, deterministic tie-breaking, and frozen `SearchResult` container.
- **Phase 2 (Exact Flat & Ground Truth):** Complete. Exhaustive exact search baseline (`exact_search`, `ExactFlatIndex`) and ground-truth generation (`generate_ground_truth`).
- **Phase 3 (Data & Dataset Utilities):** Complete. `VectorDataset` split abstraction, stable IDs, reproducible synthetic & clustered generators, L2 normalization with zero-vector policy, and checksummed offline cache.
- **Phase 4 (Clustering):** Complete. $K$-means++ centroid initialization, vectorized Lloyd iteration loop, configurable empty-cluster policies (`reseed_furthest`, `keep`, `error`), and frozen `KMeansResult` container.
- **Phase 5 (IVF-Flat):** Complete. Inverted File Flat index with coarse k-means quantization, durable int64 IDs, immutable posting lists, multi-probe candidate gathering, and bit-for-bit all-probe equivalence with exact search.
- **Phase 6 (Benchmark Harness v0.1):** Complete. Ground-truth baseline evaluation, multi-probe IVF sweeps across nlist/nprobe, Recall@k with sentinel exclusion, distance computation accounting, machine-readable JSON/CSV reporting, automated plot generation, and CLI runner.
- **Phase 6.5 (Benchmark Hardening & Real Data):** Complete. Disentangled ID `-1` from array padding via explicit `SearchResult.mask`, separated individual per-query latency percentiles from full-batch throughput (QPS), added persistent index storage memory metrics (`index_memory_bytes`, `index_memory_mib`, `bytes_per_vector`), instrumented actual search work directly in search paths, added binary `.fvecs` SIFT1M subset loader (`read_fvecs`, `load_sift_dataset`) with streaming SHA-256 verification, and expanded CI with an offline smoke-benchmark gate.
- **Phase 6.6 (Performance Baseline & Benchmark Credibility):** Complete. Added fast BLAS-accelerated exact search (`ExactBlasIndex`, `exact_search_blas`, `pairwise_squared_l2_blas`) leveraging $\|x-y\|^2 = \|x\|^2 + \|y\|^2 - 2(x \cdot y)$ while preserving the coordinate-difference reference kernel as the correctness oracle; accelerated K-means centroid assignment; bounded query batch memory via `max_query_chunk=1024`; added distance-aware Recall@k (`compute_distance_aware_recall_at_k`) to resolve boundary equidistant ties; exposed K-means convergence diagnostics (`kmeans_n_iter`, `kmeans_converged`, `kmeans_inertia`); tracked peak process RSS memory; and updated benchmark manifests, CSVs, and dual-baseline plots.

## Project Conventions

- **Vectors:** Contiguous NumPy `float32` arrays of shape `(N, d)`.
- **Identifiers:** Stable `int64` arrays of length `N` (unique among live items; not tied to row position). Any `int64` ID (including `-1`) is fully supported.
- **Distance Metric:** Squared Euclidean distance ($d^2(q, x) = \sum_{j=1}^d (q_j - x_j)^2$) as canonical reference.
- **Result Ordering:** Deterministic ascending order by composite key `(distance, ID)` to guarantee reproducible tie-breaking.
- **Result Validity:** `SearchResult` exposes `.mask` (`bool`) distinguishing legitimate nearest-neighbor matches from batched IVF padding.

## Getting Started

### Prerequisites

- Python 3.10 or newer

### Environment Setup

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Installation

Install Indexora in editable mode with development dependencies:

```bash
pip install -e ".[dev,benchmark]"
```

### Running Tests

Execute the full test suite using pytest:

```bash
pytest
```

### Running Benchmarks

#### 1. Synthetic Smoke Test (Quick Offline Verification)

Run the fast synthetic clustered smoke benchmark (< 1 second):

```bash
python -m indexora.benchmark --smoke --output-dir benchmark_output
```

This creates:
- `benchmark_output/benchmark_results.json`: Complete machine-readable manifest with environment, git commit, and run metrics.
- `benchmark_output/benchmark_summary.csv`: Tabular run metrics.
- `benchmark_output/recall_vs_latency.png`: Headline plot of Recall@k vs. p50 per-query latency.
- `benchmark_output/recall_vs_distance_computations.png`: Recall@k vs. distance computations per query.

#### 2. Synthetic Parameter Sweeps

Run a sweep over synthetic clustered data:

```bash
python -m indexora.benchmark --n-base 10000 --n-queries 100 --k 10 --nlist 10,50,100 --nprobe 1,2,5,10,20
```

#### 3. Real SIFT1M Dataset Benchmark (SIFT-100k Subset)

To benchmark against the canonical SIFT dataset:
1. Download and extract SIFT1M from the Texmex benchmark:
   ```bash
   curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
   tar -xzf sift.tar.gz
   ```
2. Run the benchmark on the 100,000-vector subset:
   ```bash
   python -m indexora.benchmark --dataset sift_fvecs --base-path sift/sift_base.fvecs --query-path sift/sift_query.fvecs --n-base 100000 --n-queries 1000
   ```

### Verified SIFT-100k Benchmark Findings (Phase 6.6)

Evaluation on the first 100,000 base vectors and 1,000 queries ($d=128$, $k=10$, squared L2):

| Index | $n_{\text{list}}$ | $n_{\text{probe}}$ | Recall@10 | DistRec | p50 (ms) | p95 (ms) | Batch QPS | Dist / Q | Build Time | K-means $n_{\text{iter}}$ / Conv |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact (Ref)** | — | — | **1.0000** | 1.0000 | 150.69 | 198.22 | 6.2 | 100,000 | 0.01s | — |
| **Exact (BLAS)**| — | — | **1.0000** | 1.0000 | **99.74** | 146.96 | **19.5** | 100,000 | 0.03s | — |
| **IVF-Flat** | 10 | 1 | 0.7882 | 0.7882 | 15.41 | 23.45 | 64.0 | 10,542 | 16.38s | 83 / True |
| **IVF-Flat** | 10 | 10 | **1.0000** | 1.0000 | 175.20 | 214.04 | 5.9 | 100,010 | 16.38s | 83 / True |
| **IVF-Flat** | 50 | 1 | 0.5976 | 0.5976 | 3.83 | 5.45 | 263.0 | 2,276 | 36.71s | 100 / False |
| **IVF-Flat** | 50 | 20 | **1.0000** | 1.0000 | 67.90 | 92.15 | 16.4 | 40,973 | 36.71s | 100 / False |
| **IVF-Flat** | 100 | 1 | 0.5287 | 0.5287 | **0.72** | 1.22 | **1,534.6** | 1,213 | 22.77s | 100 / False |
| **IVF-Flat** | 100 | 10 | 0.9791 | 0.9791 | 8.32 | 11.14 | 116.8 | 10,928 | 22.77s | 100 / False |
| **IVF-Flat** | 100 | 20 | **0.9980** | 0.9980 | 17.75 | 21.18 | 55.9 | 21,481 | 22.77s | 100 / False |

**Key Takeaways & Limitations:**
- **Exact Scan Baseline:** BLAS Exact achieves $3.15\times$ higher throughput (19.5 vs. 6.2 QPS) and $1.51\times$ lower p50 latency than the reference oracle.
- **IVF-Flat Speedup:** Compared to the realistic BLAS exact baseline, IVF-Flat (`nlist=100, nprobe=1`) provides a $78.7\times$ throughput speedup (1,534.6 vs. 19.5 QPS) at Recall@10 = 0.5287. Probing $n_{\text{probe}}=10$ lists achieves Recall@10 = 0.9791 with a $6.0\times$ throughput speedup (116.8 vs. 19.5 QPS).
- **Build-Time Acceleration:** Vectorized BLAS cluster assignments reduced K-means index build time for $n_{\text{list}}=100$ from $\approx 18$ minutes to 22.8 seconds ($\approx 47\times$ faster).
- **Limitations:** Measurements reflect single-process NumPy execution on CPU with $N=100,000$ and $Q=1,000$ vectors; results do not extrapolate to different vector distributions, hardware, or multi-threaded C++/GPU implementations.

### Metric Definitions

- **Recall@k:** Fraction of true top-$k$ nearest neighbors retrieved relative to exact search ground truth. Evaluated strictly using `.mask` so that padding slots are excluded while real IDs (including `-1`) count toward hits.
- **Distance-Aware Recall@k:** Evaluates recall by considering retrieved candidates whose distance matches or beats the distance of the $k$-th ground-truth neighbor ($d \le d_k^{\text{GT}} + \epsilon$). This prevents penalizing approximate search when multiple database vectors share identical distances at the top-$k$ boundary.
- **Per-Query Latency (ms):** Individual query execution times sampled across independent calls (reporting p50, p95, p99, mean, std, min, and max percentiles).
- **Batch Throughput (QPS):** Queries per second measured over repeated full-batch searches ($Q / t_{\text{sec}}$).
- **Distance Computations (DC):** Vector distance calculations executed inside the search path per query ($N$ for exact scan, $n_{\text{list}} + \text{mean}(N_{\text{cand}})$ for IVF-Flat).
- **Index Storage Memory (MiB):** Exact memory of stored index arrays (centroids, vectors, and durable IDs) in binary mebibytes ($1024^2$ bytes), reported alongside `bytes_per_vector`.
- **Peak Process RSS (MiB):** Peak resident set size of the process across execution, captured directly from OS memory counters.
- **Exact Baselines:** Benchmarks track both `exact_reference` (coordinate-difference direct kernel, correctness oracle) and `exact_blas` (norm expansion via BLAS matrix multiplication).

## Reference Adapters (Phase 7)

Indexora provides reference adapters for industrial vector search libraries to validate correctness, compare recall/latency trade-offs, and ground educational benchmarks:
- **`FaissFlatIndex`:** Exact flat L2 scan using `faiss.IndexFlatL2` + `IndexIDMap2`.
- **`FaissIVFFlatIndex`:** Approximate inverted-file scan using `faiss.IndexIVFFlat` + `IndexIDMap2` with dynamic runtime `nprobe` tuning.
- **`HnswlibIndex`:** Graph-based approximate nearest neighbor search using `hnswlib.Index` (L2 space).

### Installation

FAISS and hnswlib are purely optional dependencies. Indexora installs and runs without them:

```bash
# Install with adapter support
pip install ".[adapters]"

# Or install manually
pip install faiss-cpu hnswlib
```

### Adapter Contracts & Parity

All adapters implement Indexora's standard index interface:
- **Normalized Output:** Return immutable `SearchResult(distances, ids, mask)` containers.
- **Distance Metric:** Squared Euclidean distance $\|q - x\|_2^2$ consistent with Indexora's distance contract.
- **ID Support:** Arbitrary `int64` IDs including negative values (`-1`, `-42`) without sentinel collision.
- **Memory Tracking:** Report `.index_memory_bytes`, `.index_memory_mib`, and `.bytes_per_vector`.
- **Graceful Degradation:** Raise informative `ImportError` with installation instructions when optional libraries are not installed.

### Comparative Benchmarking

```bash
# Run benchmark sweep including FAISS baselines
python -m indexora.benchmark --compare-faiss

# Run SIFT-100k benchmark with external reference comparisons
python -m indexora.benchmark --dataset sift_fvecs --base-path sift/sift_base.fvecs --query-path sift/sift_query.fvecs --compare-faiss
```

## Documentation

- `plan/project srs.md`: Project Brief & Software Requirements Specification (SRS v3.0)
- `plan/tech design.md`: Architecture, Data Model, Kernel Contracts & Experiments (v3.0)
- `plan/implementation.md`: 14-phase acceptance-gated roadmap and ownership policy (v3.0)
- `docs/decisions.md`: Architecture Decision Records (ADRs 0001 through 0008)
- `docs/phase1_learning_guide.md`: Phase 1 concepts, hand-worked examples, and exercises
- `docs/phase1_test_spec.md`: Phase 1 test matrix and edge-case contracts
