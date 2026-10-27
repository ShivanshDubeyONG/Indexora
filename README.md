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

## Project Conventions

- **Vectors:** Contiguous NumPy `float32` arrays of shape `(N, d)`.
- **Identifiers:** Stable `int64` arrays of length `N` (unique among live items; not tied to row position).
- **Distance Metric:** Squared Euclidean distance ($d^2(q, x) = \sum_{j=1}^d (q_j - x_j)^2$) as canonical reference.
- **Result Ordering:** Deterministic ascending order by composite key `(distance, ID)` to guarantee reproducible tie-breaking.

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
pip install -e ".[dev]"
```

### Running Tests

Execute the test suite using pytest:

```bash
pytest
```

## Documentation

- `plan/project srs.md`: Project Brief & Software Requirements Specification (SRS v3.0)
- `plan/tech design.md`: Architecture, Data Model, Kernel Contracts & Experiments (v3.0)
- `plan/implementation.md`: 14-phase acceptance-gated roadmap and ownership policy (v3.0)
- `docs/decisions.md`: Architecture Decision Records (ADRs) and design conventions
- `docs/phase1_learning_guide.md`: Phase 1 concepts, hand-worked examples, and exercises
- `docs/phase1_test_spec.md`: Phase 1 test matrix and edge-case contracts
