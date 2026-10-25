# Indexora --- Project Brief and Software Requirements Specification

**Version:** 3.0 \| **Status:** Source of truth \| **Core stack:**
Python + NumPy

## 1. Project summary

Indexora is a learning-first vector search engine implemented from
scratch in Python and NumPy. It begins with a trustworthy exact-search
baseline and reproducible evaluation, then adds approximate
nearest-neighbor (ANN) algorithms so their recall, latency, memory use,
and build cost can be compared fairly. Custom algorithms are the
educational core; FAISS and hnswlib are optional reference baselines,
not substitutes for the implementations being learned.

The project should be defensible in a code review or interview: clear
contracts, tests, repeatable experiments, measured trade-offs, and no
performance claims unsupported by actual runs.

## 2. Goals

1.  Implement exact vector search and use it as the correctness oracle.
2.  Implement and understand k-means, IVF-Flat, Product Quantization
    (PQ), IVF-PQ, and HNSW.
3.  Build reproducible benchmarks for recall, latency, throughput,
    distance computations, memory, and build time.
4.  Compare custom implementations with reference libraries under
    matched conditions.
5.  Demonstrate the engine on a standard vector dataset and a small
    semantic-search application.
6.  Keep the core algorithms understandable and explainable by the
    project owner.

## 3. Non-goals

This is not a production vector database. Out of scope: distributed
indexing, sharding/replication, GPU/CUDA, a C++ core, billion-vector
claims, a large frontend, BM25/hybrid retrieval, a full RAG platform,
OPQ, and production multi-tenant security/SLAs. These can be future
work, not reasons to delay the core.

## 4. Design principles

-   **Correctness before speed.** Establish exact behavior and tests
    before optimization.
-   **Explicit reference truth.** Exact search defines ground truth; ANN
    methods are evaluated against it.
-   **Reproducibility.** Record dataset identity, splits, seeds,
    parameters, versions, environment, and raw measurements.
-   **No hidden preprocessing.** Normalization, dtype conversion,
    filtering, and ID semantics must be explicit.
-   **Fair comparisons.** Compare latency and memory at matched recall
    where possible.
-   **No invented results.** Populate charts and resume metrics only
    after real runs.
-   **Learning ownership.** The owner writes and understands core
    algorithm bodies; agents accelerate supporting engineering.

## 5. Release ladder

### v0.1 --- Trustworthy core

-   Vector/data abstractions with stable integer IDs.
-   Reference squared-L2 distance, deterministic top-k, exact Flat
    search, and ground truth.
-   k-means++ and Lloyd iterations.
-   IVF-Flat.
-   Unit/integration tests and reproducible benchmark harness.
-   Initial recall/latency and recall/distance-computation plots.

**Gate:** hand-checkable exact cases pass; probing every IVF list agrees
with exact search under the same tie policy; benchmark output includes
metadata and raw results.

### v0.2 --- Quantization and reference baselines

-   Optional adapters for reference libraries.
-   Profiling and at most one justified Numba hot path.
-   PQ, ADC scoring, IVF-PQ residual encoding, and optional exact
    reranking.
-   Memory accounting and matched-recall comparisons.

**Gate:** codebooks, encoding, ADC, residual semantics, and memory
estimates are tested; reference results are clearly labeled.

### v0.3 --- Custom HNSW

-   Custom graph construction/search with `M`, `efConstruction`, and
    `efSearch`.
-   Graph invariant tests and reference comparison.

**Gate:** graph invariants pass and recall/latency are measured. Large
`efSearch` is not assumed to make graph search exhaustive. Exactness, if
required, uses a separately named exhaustive fallback.

### v0.4 --- Index operations

-   Insert/delete with tombstones, update and ID-reuse policy.
-   Metadata filtering, compaction/rebuild, diagnostics, safe
    persistence.

**Gate:** lifecycle, filtering, corruption, and round-trip tests pass.

### v1.0 --- Portfolio release

-   Small semantic-search CLI or web demo with cached text embeddings.
-   Standard vector-dataset and text-embedding experiments.
-   Architecture diagram, benchmark methodology, limitations,
    reproducible commands, and interview notes.

**Gate:** a new user can install, test, reproduce a benchmark, and run
the demo.

## 6. Functional requirements

### Data and ranking

-   Vectors are contiguous `float32` arrays initially; IDs are stable
    `int64`.
-   Validate shapes, dimensions, finiteness, and ID uniqueness.
-   Do not silently normalize vectors. Cosine ranking requires an
    explicit normalization path and zero-vector policy.
-   Start with squared Euclidean distance.
-   Final result ordering is deterministic, e.g. ascending
    `(distance, ID)`.
-   `argpartition` alone does not define tie behavior at the k-boundary;
    handle boundary ties explicitly.
-   Document behavior for empty indexes, invalid `k`, `k > N`, and
    filters with no matches.

### Search algorithms

-   Exact search is the benchmark oracle.
-   IVF-Flat probing every populated list must agree with exact search.
-   PQ/IVF-PQ must document approximate scoring and optional reranking.
-   HNSW must report measured approximation quality; it must not imply
    guarantees standard graph search does not provide.

### Evaluation

Record, where applicable: dataset/split, shape/dtype, query count,
metric, seed, index parameters, commit/version, environment, warm-up
policy, repeats, recall@k, p50/p95 latency, QPS, distance computations
per query, build time, and memory. Keep raw runs and aggregate
summaries. Separate JIT compilation from steady-state timings.

## 7. Ownership policy

**Human-owned core learning tasks:** distance kernels and top-k;
exact-search loop and ground truth; k-means++; IVF assignment/search; PQ
training/encoding/ADC; IVF-PQ residual logic; HNSW level assignment,
insertion, neighbor selection, and search-layer logic.

**Agent-assisted tasks:** repository scaffolding, packaging/CI, test
scaffolds, dataset cache utilities, benchmark orchestration/plots,
reference adapters, demo/API plumbing, persistence plumbing,
documentation drafts, and reviews.

An agent may help with interfaces, tests, pseudocode, hints, and
debugging for human-owned phases, but must not silently implement the
finished core body. If it does, the owner must re-derive, explain, and
rewrite or materially validate it before claiming ownership.

## 8. Definition of done

A feature needs a contract, tests, appropriate experiment evidence, and
documentation that matches actual behavior. A release is ready when a
clean checkout can install, run tests, run a documented benchmark, and
launch the demo. Clearly label custom implementations, reference
baselines, experimental features, and planned work.
