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

### Metric Definitions

- **Recall@k:** Fraction of true top-$k$ nearest neighbors retrieved relative to exact search ground truth. Evaluated strictly using `.mask` so that padding slots are excluded while real IDs (including `-1`) count toward hits.
- **Per-Query Latency (ms):** Individual query execution times sampled across independent calls, reporting p50, p95, p99, mean, min, and max percentiles.
- **Batch Throughput (QPS):** Queries per second measured over repeated full-batch searches ($Q / t_{\text{sec}}$).
- **Distance Computations (DC):** Vector distance calculations executed inside the search path per query ($N$ for exact scan, $n_{\text{list}} + \text{mean}(N_{\text{cand}})$ for IVF-Flat).
- **Index Storage Memory (MiB):** Exact memory of stored index arrays (centroids, vectors, and durable IDs) in binary mebibytes ($1024^2$ bytes), reported alongside `bytes_per_vector`.

## Documentation

- `plan/project srs.md`: Project Brief & Software Requirements Specification (SRS v3.0)
- `plan/tech design.md`: Architecture, Data Model, Kernel Contracts & Experiments (v3.0)
- `plan/implementation.md`: 14-phase acceptance-gated roadmap and ownership policy (v3.0)
- `docs/decisions.md`: Architecture Decision Records (ADRs 0001 through 0008)
- `docs/phase1_learning_guide.md`: Phase 1 concepts, hand-worked examples, and exercises
- `docs/phase1_test_spec.md`: Phase 1 test matrix and edge-case contracts
