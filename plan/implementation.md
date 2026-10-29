# Indexora --- Phased Implementation Plan

**Version:** 3.0 \| **Planning model:** Acceptance gates, not calendar
promises

## 1. How to work

Work on one phase at a time. A phase ends when its acceptance gate
passes, not when an estimated date arrives. After each phase, record:
changed files, design decisions, tests and actual outputs, what was
learned, limitations, and next step.

### Owner/agent split

The owner implements and understands the core algorithm bodies. For
human-owned phases, the agent may explain, propose interfaces, write
tests, give pseudocode/hints, and review the owner's attempt, but must
not silently deliver the finished core algorithm. If agent-written core
code is used as a reference, the owner must re-derive, explain, rewrite,
or materially validate it before claiming ownership.

Agent-assisted tasks include packaging/CI, dataset cache utilities, test
scaffolds, benchmark orchestration/plots, reference adapters, demo
plumbing, persistence plumbing, and documentation drafts.

### Rules for every agent task

1.  Inspect the repository before editing.
2.  Work only on the named phase.
3.  Preserve unrelated work and avoid scope creep.
4.  Add/update tests with meaningful changes.
5.  Run tests and report actual commands and outputs.
6.  Never invent benchmark results, test output, or completion claims.
7.  Keep optional dependencies optional.
8.  End with a concise handoff: files changed, tests, results,
    limitations, next step.

## 2. Phase overview

  -------------------------------------------------------------------------------------------
  Phase             Deliverable                     Ownership         Acceptance gate
  ----------------- ------------------------------- ----------------- -----------------------
  0                 Repository setup                Agent-assisted    Clean install and tests
                                                                      work

  1                 Reference distance +            Human core        Numerical/tie tests
                    deterministic top-k                               pass

  2                 Exact search + ground truth     Human core        Hand-computed cases
                                                                      pass

  3                 Data/dataset utilities          Agent-assisted    Stable IDs and offline
                                                                      tests

  4                 k-means++/Lloyd                 Human core        Clustering and
                                                                      empty-cluster tests
                                                                      pass

  5                 IVF-Flat                        Human core        All-probe results equal
                                                                      exact

  6                 Benchmark harness v0.1          Agent-assisted    Reproducible raw
                                                                      results/plots

  7                 Reference adapters              Agent-assisted    Optional, labeled
                                                                      comparisons

  8                 Profile/one acceleration        Shared            Measured benefit or
                    experiment                                        documented no-go

  9                 PQ + IVF-PQ                     Human core        Encoding/ADC/residual
                                                                      tests pass

  10                Custom HNSW                     Human core        Invariants pass; recall
                                                                      measured

  11                Mutations/filters/persistence   Shared            Lifecycle tests pass

  12                Semantic-search demo            Agent-assisted    Documented run uses
                                                                      Indexora API

  13                Release/interview prep          Shared            Reproducible release
                                                                      and evidence-based
                                                                      claims
  -------------------------------------------------------------------------------------------

## 3. Detailed phases

### Phase 0 --- Repository setup (agent-assisted)

Create a minimal package layout, Python/NumPy dependency, test runner,
optional formatting/linting, basic CI, `.gitignore`, license choice, and
short README. Do not implement search algorithms.

**Prompt:** "Inspect the current repository first. Set up only the
minimal packaging, test, and documentation foundation for Indexora. Do
not implement vector-search algorithms. Keep optional dependencies
optional. Run checks and report actual outputs."

**Gate:** clean install works, `import indexora` works, test command is
documented.

### Phase 1 --- Distance kernels and top-k (HUMAN CORE)

Implement the direct-difference squared-L2 reference kernel and
deterministic top-k. Test duplicate vectors, equal-distance boundary
ties, dimensions, empty arrays, non-finite values, and `k` edge cases.
Add a BLAS-style fast kernel only after the reference is correct.

**Prompt:** "Do not write the final bodies of the human-owned distance
or top-k functions. Refine edge-case tests, explain expected semantics,
provide hints/pseudocode when asked, and review my implementation for
numerical and tie-handling bugs."

**Gate:** hand-computable results pass; final ordering is deterministic
by the agreed tie policy; fast-path disagreement is measured and it is
not called exact without proof.

### Phase 2 --- Exact Flat and ground truth (HUMAN CORE)

Implement scan-based exact search using reference semantics, stable IDs,
and deterministic ranking. Generate ground truth through this path. Test
duplicate vectors, empty data, invalid `k`, non-row-position IDs, and
malformed input.

**Prompt:** "Help define the exact-search contract and tests, but do not
implement the core search loop for me. Ground truth must use the
reference implementation, not an ANN index."

**Gate:** exact outputs match hand-computed cases and ground truth is
reproducible.

### Phase 3 --- Data and dataset utilities (AGENT-ASSISTED)

Add dataset validation, splits, stable IDs, download/cache helpers,
provenance, and checksums where practical. Keep unit tests offline.

**Prompt:** "Implement only dataset/cache utilities. Do not change
kernel or exact-search semantics. Add offline tests with generated
arrays and document dataset acquisition."

**Gate:** invalid shapes/IDs fail clearly; unit tests need no network.

### Phase 4 --- k-means (HUMAN CORE)

Implement k-means++ and Lloyd updates. Define empty-cluster handling,
convergence criteria, seeds, and inertia reporting. Test synthetic
clusters and fixed-seed behavior.

**Prompt:** "Do not implement the core k-means++ or Lloyd update bodies.
Help define tests and pseudocode, then review my attempt, especially
empty clusters, seeds, and convergence."

**Gate:** synthetic tests pass and empty clusters follow an explicit
policy.

### Phase 5 --- IVF-Flat (HUMAN CORE)

Train coarse centroids, assign each vector to one list, preserve stable
IDs, and implement centroid ranking plus `nprobe` search. Test
unbalanced/empty lists and invalid parameters.

**Prompt:** "Help with interfaces and tests, but do not implement core
IVF assignment/search. Add an invariant test: probing every populated
list must match exact search under the same metric and tie policy."

**Gate:** each live vector belongs to one list; all-probe IVF equals
exact. Partial probing is evaluated, not assumed to have a particular
recall.

### Phase 6 --- Benchmark harness v0.1 (AGENT-ASSISTED)

Add a command to run exact and IVF sweeps. Save raw CSV/JSON and
metadata. Measure recall@k, p50/p95 latency, QPS, distance computations,
build time, and memory where available. Record dataset/split,
parameters, seed, commit, environment, warm-up, repeats, and timing
policy. Plot recall vs latency and recall vs distance computations.

**Prompt:** "Build a reproducible harness for existing exact and IVF
APIs. Adapt to current code instead of rewriting algorithms. Save raw
measurements and metadata. Do not use placeholders as results. Add a
tiny synthetic smoke test."

**Gate:** documented command creates results and plots; smoke test
works. Tag v0.1 only after the gate passes.

*Status: Complete & Released.* Phase 6 and 6.5 acceptance gates passed.
Verified with 146 passing tests and full SIFT-100k real-data evaluation (N=100,000, Q=1,000, d=128).
Annotated release tag `v0.1` published with attached benchmark manifest, CSV summary, and trade-off plots.

### Phase 6.6 --- Performance baseline and benchmark credibility (AGENT-ASSISTED)

Audit and stabilize benchmark methodology before Phase 7 reference adapters:
1. Fast exact search baseline (`exact_search_blas`, `ExactBlasIndex`, `pairwise_squared_l2_blas`) implementing $\|x-y\|^2 = \|x\|^2 + \|y\|^2 - 2(x \cdot y)$ with floating-point negative round-off clipping to zero. The direct coordinate-difference kernel remains the canonical reference oracle.
2. Vectorized BLAS cluster assignment in K-means training and coarse quantization while preserving K-means++ initialization, Lloyd iteration, and deterministic tie policies.
3. Bounded batch query memory via `max_query_chunk=1024` to prevent unbounded $(Q, N)$ temporary allocations.
4. Distance-aware Recall@k metric (`compute_distance_aware_recall_at_k`) resolving equidistant neighbor boundary ties without penalizing approximate search.
5. Surface K-means convergence statistics (`kmeans_n_iter`, `kmeans_converged`, `kmeans_inertia`).
6. Benchmarking dual exact baselines (`exact_reference` and `exact_blas`), reporting query latency standard deviation (`per_query_std_ms`), seeded latency query sampling, and tracking OS peak process RSS.

*Status: Complete & Verified.* 163 tests passing. Full SIFT-100k evaluation completed: BLAS Exact achieves 19.5 QPS (vs 6.2 QPS for reference); K-means build time reduced from ~18 min to 22.8s for nlist=100; IVF-Flat sweeps demonstrate 78.7x speedup over BLAS Exact at nlist=100, nprobe=1 (Recall@10=0.5287) and 6.0x speedup at nprobe=10 (Recall@10=0.9791). Artifacts preserved in `results/sift100k_p66/`.

### Phase 7 --- Reference adapters (AGENT-ASSISTED)

Add optional adapters for suitable libraries such as FAISS or hnswlib.
Keep imports lazy, skip tests cleanly when unavailable, and record
library version and parameters.

**Gate:** core tests pass without optional libraries; comparisons share
dataset, metric, query set, and recall definition.

*Status: Complete & Verified.* 173 tests passing.
- Implemented `FaissFlatIndex`, `FaissIVFFlatIndex`, and `HnswlibIndex` in `src/indexora/adapters.py`.
- Enforced strict output container conformance returning frozen `SearchResult(distances, ids, mask)`.
- Maintained squared Euclidean distance metric bit-for-bit parity and arbitrary `int64` ID support (including negative IDs).
- Enforced optional dependency isolation via `is_faiss_available()` and `is_hnswlib_available()`, raising clear `ImportError` on missing packages.
- Configured `[project.optional-dependencies]` with `adapters = ["faiss-cpu>=1.7.4", "hnswlib>=0.7.0"]`.
- Added unit tests in `tests/test_adapters.py` testing oracle parity, edge cases, negative IDs, all-probe IVF parity, storage properties, and mock behavior for missing dependencies.
- Integrated adapter comparisons into `src/indexora/benchmark.py` via `--compare-faiss` and `--compare-hnswlib`. Verified via synthetic smoke runs.

### Phase 8 --- Profiling/acceleration (SHARED)

Profile current code. Select one hot path only if profiling justifies
it; try Numba only where suitable. Separate compilation from
steady-state timings and compare against reference behavior.

**Gate:** same-workload before/after evidence exists. A measured no-go
is valid; do not retain acceleration just for appearance.

*Status: Complete & Verified.* 175 tests passing.
- **Profiling:** cProfile and granular component timing identified distance calculations as the dominant bottleneck in IVF-Flat search, accounting for 83.0% of total search time (70.2% candidate distances, 12.8% coarse centroid distances).
- **Optimization:** Implemented batched coarse quantization via BLAS GEMM (`pairwise_squared_l2_blas`), configurable `distance_method` in `IVFFlatIndex` (defaulting to `"blas"` with optional `"reference"` oracle mode), and zero-copy candidate handling when a single inverted list is probed.
- **Measured Results (N=10,000, Q=100, d=128, repeats=25):**
  - `nprobe=1`: p50 latency reduced from 32.98 ms to 21.20 ms (**1.56x speedup**, 4,716.8 QPS), 100.00% ID match.
  - `nprobe=5`: p50 latency reduced from 90.70 ms to 39.79 ms (**2.28x speedup**, 2,513.5 QPS), 100.00% ID match.
  - `nprobe=10`: p50 latency reduced from 167.76 ms to 64.05 ms (**2.62x speedup**, 1,561.4 QPS), 100.00% ID match, latency std reduced from 16.50 ms to 5.98 ms.
- **Outcome:** The optimization is retained. Reference implementation remains accessible via `distance_method="reference"` as an uncompromised correctness oracle.

### Phase 9 --- PQ and IVF-PQ (HUMAN CORE)

Implement subspace splitting, PQ codebooks, encoding, ADC
lookup/scoring, IVF-PQ residual encoding, and optional exact reranking.
Measure memory, approximation error, recall, and latency.

**Prompt:** "Do not implement core PQ training, encoding, ADC, or IVF-PQ
residual logic for me. Help with interfaces, tests, and pseudocode. Test
shape/code constraints, fixed seeds, and raw PQ vs residual semantics;
review my attempt."

**Gate:** tiny ADC cases agree with direct calculations; code
shapes/ranges are valid; residual behavior is tested; memory accounting
states what is included. Tag v0.2 only when advertised scope is actually
complete.

*Status: Complete & Verified.* 190 tests passing.
- **ProductQuantizer ([`src/indexora/pq.py`](file:///c:/Users/Shivansh/Desktop/Indexora/src/indexora/pq.py)):** Subspace decomposition into $M$ orthogonal subvectors ($d_{\text{sub}} = d // M$), per-subspace codebook training via K-means, compact discrete encoding (`uint8` for $k_{\text{sub}} \le 256$, `uint16` for $k_{\text{sub}} \le 65536$), and vector reconstruction with MSE evaluation.
- **Asymmetric Distance Computation (ADC):** Exact algebraic parity verified against direct distances to reconstructed vectors ($4.77 \times 10^{-7}$ floating-point tolerance). Vectorized table lookup scores candidates in $\approx 50$ microseconds per 1,000 vectors.
- **IVFPQIndex ([`src/indexora/ivfpq.py`](file:///c:/Users/Shivansh/Desktop/Indexora/src/indexora/ivfpq.py)):** Inverted-file coarse clustering combined with PQ on coarse-centroid residuals ($x - c$) or raw vectors. Posting lists store only compact PQ codes and stable `int64` IDs (including negative IDs). Conforms to Indexora's frozen `SearchResult` contract.
- **Memory Compression:** Verified $15.2\times$ memory compression on $N=10,000, d=128$ (0.33 MB vs 5.01 MB for IVF-Flat, reducing storage from 525.1 to 34.2 bytes/vector) with precise persistent array accounting.
- **Benchmark Integration:** Added `--ivf-pq`, `--pq-m`, and `--pq-ksub` to `src/indexora/benchmark.py` and integrated trade-off curves into visualizer.

### Phase 10 --- Custom HNSW (HUMAN CORE)

Implement level sampling, graph storage, greedy descent, search-layer
heaps, insertion, neighbor selection, and pruning. Test graph invariants
and seeded builds. Sweep `M`, `efConstruction`, and `efSearch`; compare
with hnswlib where available.

**Prompt:** "Do not implement core HNSW level assignment, insertion,
neighbor selection, or search-layer algorithms. Help define invariants,
tests, and pseudocode, then review my attempt. Do not assume ordinary
HNSW is exhaustive when `efSearch >= N`; use a separately named
exhaustive fallback if exact equality is needed."

**Gate:** valid links/degrees/entry point; search returns valid live
IDs; recall and latency are measured. Tag v0.3 only after tests and
evidence.

*Status: Complete & Verified.* 203 tests passing.
- **HNSWIndex ([`src/indexora/hnsw.py`](file:///c:/Users/Shivansh/Desktop/Indexora/src/indexora/hnsw.py)):** Built hierarchical multi-layer proximity graph from scratch in Python and NumPy. Implemented exponential level sampling ($m_L = 1 / \ln(M)$), greedy layer descent, and bounded best-first beam search (`_search_layer_internal`) with candidate min-heaps and evaluation pool max-heaps.
- **Algorithm 4 Neighbor Selection Heuristic:** Genuine heuristic selection enforcing spatial diversity and preventing clustering along lines-of-sight. Shrinkage checks discard candidates closer to existing neighbors than to base query, with fallback for pruned connections.
- **Degree Bounds & Graph Invariants:** Enforced strict degree bounds ($M_0 = 2M$ at level 0, $M$ at levels $>0$). Graph integrity tests verify no self-loops, valid node indices across all levels, and correct entry-point tracking.
- **ID & Search Result Contracts:** Full compatibility with Indexora's frozen `SearchResult` container. Supports arbitrary `int64` IDs (including negative IDs and `-1`), deterministic ascending `(distance, external_id)` tie-breaking, and exact distance evaluation tracking.
- **Exact Memory Accounting:** Reports `.vector_memory_bytes`, `.id_memory_bytes`, `.graph_edge_count`, `.graph_memory_bytes`, `.index_memory_bytes`, `.index_memory_mib`, and `.bytes_per_vector`.
- **Benchmark Integration:** Added `--hnsw`, `--hnsw-m`, `--hnsw-ef-construction`, and `--hnsw-ef-search` CLI flags to `src/indexora/benchmark.py`. Validated on synthetic clustered benchmarks ($N=1,000, d=32$), achieving **1.0000 Recall@10** evaluating only 144 to 152 distances per query (~15% of database).

### Phase 11 --- Mutations, filters, persistence (SHARED) [COMPLETED]

Implemented reliable index mutations (`add`, `delete`), metadata/ID filtering (`filter_ids`), and atomic disk persistence (`save_index`, `load_index`) across all core Indexora index implementations.

- **Capability Contract:** Established unambiguous support matrix across all five index architectures (`ExactFlatIndex`, `ExactBlasIndex`, `IVFFlatIndex`, `IVFPQIndex`, `HNSWIndex`).
- **Mutation Semantics:**
  - `add(vectors, ids=None)`: Appends vectors to flat stores, coarse inverted lists, or multi-layer graph structures. Handles automatic sequential ID assignment, enforces uniqueness, rejects duplicate ID conflicts, and strictly enforces float32 dtypes.
  - `delete(ids)`: Inverted lists and flat indexes purge removed vectors and rebuild internal contiguous arrays. HNSW applies non-destructive tombstone markers (`_deleted_nodes`) preserving graph routing connectivity while completely excluding tombstoned vectors from search candidate pools.
- **Filtered Search:** Added caller-supplied `filter_ids` parameter to `search()` supporting Sequences, Sets, NumPy `int64` arrays, or arbitrary Callables `(int) -> bool`. Excludes non-eligible vectors at level 0/candidate evaluation, preserving deterministic `(distance, external_id)` tie-breaking and maintaining standard `SearchResult` contract.
- **Safe Persistence:**
  - Zero arbitrary Python pickle: Pure `.npy` array storage using `np.save(allow_pickle=False)`.
  - Atomic directory writes: Writes to temporary staging folder (`.tmp_<uuid>`) before atomic replacement.
  - Integrity and provenance manifests: `manifest.json` tracks `schema_version="1.0"`, hyperparameters, array shapes, dtypes, and file SHA-256 checksums. Verified upon load with clear error reporting on corruption or version mismatch.
  - Exact state restoration: Reconstructs HNSW multi-layer CSR graphs without graph rebuilds; reconstructs IVF-PQ codebooks and posting lists without retraining coarse k-means or subspace quantizers.
- **Validation:** 88 dedicated unit tests across mutations (`test_mutations.py`), filtering (`test_filtering.py`), and persistence (`test_persistence.py`), achieving 291/291 passing tests in the full test suite.

### Phase 12A --- Interactive Index Lab (UI Playground) [COMPLETED]

Implemented a local-first web playground (`src/indexora/ui.py` and `src/indexora/lab.py`) using Streamlit that provides an interactive, visual interface directly backed by Indexora's actual Python/NumPy implementations.

- **Non-UI Data & Lab Layer (`src/indexora/lab.py`):**
  - Deterministic clustered dataset generation with configurable dispersion, cluster counts, and stable `int64` IDs.
  - Deterministic 2D PCA projection computed via pure NumPy SVD without external dependencies.
  - Exact reference ground truth computation and Recall@k evaluation.
  - Multi-index benchmark execution measuring build time, latency percentiles (p50, p95), QPS, distance evaluations, and memory footprint.
  - `LabSession` state manager tracking incremental additions, tombstone deletions, and atomic directory persistence.
- **Interactive UI Playground (`src/indexora/ui.py`):**
  - **Overview & Mechanics:** Architecture comparison cards for Exact Flat/BLAS, IVF-Flat, IVF-PQ, and HNSW.
  - **Search Playground:** Single-query exploration, index selection with index-specific parameter tuning (`nprobe`, `ef_search`), optional `filter_ids`, and interactive 2D PCA map highlighting true nearest neighbors vs. approximations.
  - **Accuracy & Performance Lab:** Live multi-index benchmark sweep with Recall vs. Latency scatter plots, distance computation bar charts, and memory accounting.
  - **Mutations & Persistence:** Real-time demonstration of vector additions, tombstone deletions, and atomic save/load round-trips with SHA-256 manifest verification.
- **Validation:** 11 dedicated unit tests in `tests/test_lab.py`, bringing the total test suite to 302/302 passing tests.

### Phase 12B --- Semantic Text Search Integration [COMPLETED]

Integrated end-to-end semantic text search (`src/indexora/semantic.py` and `src/indexora/ui.py`) backed by Indexora's core retrieval engines (Exact BLAS, Exact Flat, IVF-Flat, and HNSW) and pretrained sentence embeddings.

- **Corpus & Data Modeling (`src/indexora/semantic.py`):**
  - Immutable `Document` container with durable integer ID, title, text, category, and metadata tags.
  - Curated 25-document technical corpus spanning 5 engineering domains: Machine Learning, Databases, Distributed Systems, Cybersecurity, and Software Engineering.
- **Embedding & Distance Contract:**
  - Pretrained `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) as an optional extra (`pip install -e ".[semantic]"`).
  - L2 normalization guaranteeing exact mathematical equivalence:
    $$\|q - x\|^2 = \|q\|^2 + \|x\|^2 - 2(q \cdot x) = 2 - 2 \cos(q, x)$$
    Minimizing squared Euclidean distance directly maximizes cosine similarity without altering Indexora's native distance kernels.
  - Deterministic offline mock embedder (`mock_embed_texts`) enabling fully offline test execution and zero-dependency simulation.
- **Search Engine Pipeline (`SemanticSearchEngine`):**
  - Clean separation between embedding generation time ($t_{\text{embed}}$) and vector index retrieval time ($t_{\text{search}}$).
  - Exact reference ground-truth comparison on every query to evaluate empirical approximate index recall (`is_exact_match`).
  - Native category and ID filtering integrated into index search paths.
- **Interactive UI Integration (Tab 5 in `src/indexora/ui.py`):**
  - Dedicated "📚 Semantic Text Search" tab in the Streamlit lab.
  - Quick-preset technical queries, category filter dropdown, top-$k$ controls, and index parameter sliders (`nprobe`, `ef_search`).
  - Latency breakdown cards, distance computation accounting, and formatted document result cards with cosine similarity and exact match badges.
- **Validation:** 14 dedicated offline unit tests in `tests/test_semantic.py`, bringing the total test suite to 325 passing tests.

### Phase 13 --- Release Readiness & Interview Preparation [COMPLETED]

Audited the repository, validated the complete user experience, created comprehensive technical documentation, and finalized presentation readiness.

- **Technical Architecture (`docs/architecture.md`):**
  - Modular layered architecture with Mermaid diagrams.
  - Mathematical foundations of squared L2 distance, Level-3 BLAS GEMM norm expansion, and unit vector cosine equivalence.
  - In-depth descriptions of all 5 index architectures: Exact Reference, Exact BLAS, IVF-Flat, IVF-PQ, and custom HNSW.
  - Lifecycle mechanisms: single-stage filtering, incremental additions, HNSW routing tombstones, and pickle-free atomic persistence with streaming SHA-256 manifests.
  - Transparent disclosures regarding Python interpreter overhead on small datasets ($N \le 10^3$) vs vectorized BLAS.
- **Interview Preparation Guide (`docs/interview_prep.md`):**
  - 16 rigorous technical interview questions and code-backed answers spanning computational geometry, systems trade-offs, and vector database internals.
  - 2-minute live walkthrough script for technical demos.
- **Presentation-Ready README (`README.md`):**
  - 30-second elevator pitch and core problem statement.
  - Mermaid architecture diagram and index selection guide table ("When to use which index").
  - 10-second quickstart installation and Python usage snippets.
  - Documented empirical SIFT-100k benchmarks ($N=100k, d=128, Q=1k, k=10$) with full environment context.
- **Validation:** 325 passing tests in the automated suite (`pytest -q`), clean benchmark smoke runs, live neural semantic retrieval verified on CPU with `all-MiniLM-L6-v2`, and tag `v0.1` preserved at commit `6d4d91f`.

## 4. Release gates

-   **v0.1:** exact search, ground truth, k-means, IVF-Flat, tests,
    benchmark harness, initial plots.
-   **v0.2:** reference adapters, one justified acceleration experiment,
    PQ and IVF-PQ.
-   **v0.3:** custom HNSW, invariant tests, sweeps, reference
    comparison.
-   **v0.4:** mutations, filters, compaction, persistence, diagnostics.
-   **v1.0:** semantic demo, reproducible commands, report,
    evidence-based claims.

Do not tag a release because a target date arrived. Defer or revise
scope transparently.

## 5. If time gets tight

1.  Correct reference distances, deterministic top-k, exact search, and
    ground truth.
2.  k-means and IVF-Flat with all-probe equivalence.
3.  Reproducible benchmark harness and one honest plot.
4.  Choose **one** next advanced algorithm---PQ/IVF-PQ or HNSW---and
    finish it before starting the other.
5.  Add the second algorithm only if the first is stable.
6.  Add mutations/filtering/persistence after core search is reliable.
7.  Finish the small demo and documentation.

This is a prioritization strategy, not a universal claim that one
algorithm is more valuable. A correct narrow project beats a broad
project with weak tests and untrustworthy numbers.

## 6. Weekly review checklist

-   What did I implement myself versus with agent help?
-   Can I explain the algorithm without reading the code?
-   Which tests demonstrate correctness?
-   What limitations or edge cases remain?
-   What did the benchmark actually measure?
-   Are README/resume claims stronger than the evidence?
-   What is the smallest next step to clear the current gate?

**Final rule:** optimize for algorithms you can explain, code you can
trust, and experiments another person can reproduce---not for the number
of algorithms in the README.
