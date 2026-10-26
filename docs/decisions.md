# Indexora — Architecture Decision Records (ADR)

This document records the architectural and design decisions for Indexora, along with the core technical conventions established across all phases.

---

## Architecture Decision Record Template

```markdown
### ADR-XXXX: [Short Title]

- **Status:** [Proposed | Accepted | Superseded | Deprecated]
- **Date:** YYYY-MM-DD
- **Decision Owner:** [Human Core | Agent-assisted | Shared]

#### Context
What problem are we solving? What constraints or requirements apply?

#### Decision
What is the chosen approach, contract, or architecture?

#### Alternatives Considered
What other options were evaluated, and why were they not chosen?

#### Consequences & Invariants
What invariants must be preserved by this decision? What are the trade-offs?
```

---

## ADR-0001: Core Data Conventions and Result Ordering

- **Status:** Accepted
- **Date:** 2026-10-25
- **Decision Owner:** Shared (Phase 0 Foundation)

### 1. Vector Representation: Contiguous `float32`

- **Format:** Vectors must be represented as 2D NumPy arrays of shape `(N, d)` with `dtype=np.float32`.
- **Memory Layout:** Arrays must be contiguous in memory (C-contiguous, e.g., via `np.ascontiguousarray`). Contiguity ensures predictable memory access patterns for SIMD vectorization and cache utilization.
- **Validation:** Dimensions, shape compatibility, and value finiteness (`np.isfinite`) must be validated upon entry. 
- **No Silent Preprocessing:** The engine must never silently normalize vectors, convert dtypes, truncate dimensions, or impute missing/NaN values. Any metric requiring unit-norm vectors (such as Cosine distance) must provide an explicit normalization utility with a documented zero-vector policy.

### 2. Identifier Representation: Stable `int64` IDs

- **Format:** Identifiers must be 1D NumPy arrays of length `N` with `dtype=np.int64`.
- **Durable Identity:** Row position (index 0 to $N-1$ in the vector matrix) is ephemeral and must **never** be used as durable identity. Real-world vector indices undergo deletions, updates, filtering, and compactions, altering row indices over time.
- **Uniqueness:** IDs must be strictly unique among live index records.
- **Metadata Alignment:** Metadata collections (columnar arrays or JSON payloads) are keyed exclusively by stable `int64` ID.

### 3. Canonical Distance Metric: Squared-L2 Distance

- **Definition:** The primary metric for vector comparison is squared Euclidean distance:
  $$d^2(q, x) = \sum_{j=1}^{d} (q_j - x_j)^2$$
- **Rationale:** 
  1. For any set of vectors, ranking by $d^2(q, x)$ is strictly monotonically equivalent to ranking by true Euclidean distance $\|q - x\|_2$.
  2. Eliminating the square root operation ($\sqrt{\cdot}$) saves significant CPU cycles across millions of pairwise comparisons.
- **Correctness Reference:** The direct coordinate difference $\sum (q_j - x_j)^2$ (with higher-precision accumulation where appropriate) serves as the system's correctness oracle. Fast BLAS matrix formulations ($\|q\|^2 - 2q^Tx + \|x\|^2$) are treated as optimizations that must be benchmarked against this oracle.

### 4. Result Ordering: Deterministic `(distance, ID)` Composite Key

- **Ordering Contract:** Search results must always be returned in strictly deterministic order:
  $$\text{Sort key: } (d^2(q, x), \text{ID})$$
  specifically: ascending by distance, and tie-breaking by ascending `int64` ID.
- **Tie-Breaking Rule:** When two vectors have identical distances to a query (e.g., identical vectors, symmetric coordinates, or numerical collision), the candidate with the smaller numerical ID must be ranked higher.
- **Boundary Invariant:** Naive partial-selection functions such as NumPy's `np.argpartition` do not guarantee deterministic selection at the $k$-th rank boundary when ties exist. Indexora implementations must explicitly handle boundary ties so that exact search, IVF lists, and evaluation scripts yield bit-for-bit identical top-$k$ results across runs and platforms.

---

## ADR-0002: Phase 1 Top-k Query Contracts and SearchResult Type

- **Status:** Accepted (API Contract Specification)
- **Date:** 2026-10-25
- **Decision Owner:** Human Owner (Approved Phase 1 Contracts)

### 1. Context
During the preparation of Phase 1 (Distance Kernels and Top-$k$ Selection), four operational behaviors required explicit project policies prior to human core implementation:
1. Retrieval behavior when requested neighbor count $k$ exceeds database size $N$.
2. Input validation policy for non-positive query limits ($k \le 0$).
3. Type safety policy when vector inputs have dtypes other than `float32`.
4. Return container structure for top-$k$ selection and nearest-neighbor search.

These decisions define design-time API contracts and test specifications. They do not represent already-implemented code.

### 2. Decisions

1. **Oversized Neighbor Queries ($k > N$):**
   - When $k > N$, the search operation returns all $N$ available records sorted by ascending `(distance, ID)`.
   - The engine must not raise an error, crash, or pad the output with sentinel/dummy IDs.
   - *Rationale:* Aligns with standard information retrieval semantics and industry references (FAISS, scikit-learn).

2. **Non-Positive Neighbor Count ($k \le 0$):**
   - If $k \le 0$, `select_top_k` must raise a `ValueError("k must be positive")`.
   - Searching for zero or negative nearest neighbors is an invalid query operation.
   - *Open Decision Note:* The project documents do not specify behavior for non-integer $k$ values (e.g. `float`, `str`, `None`); this remains an unresolved owner decision.

3. **Strict Vector Dtype Enforcement (No Silent Casting):**
   - Distance kernels and search functions must require `np.float32` inputs.
   - Non-`float32` arrays (such as `float64`, `int32`, `int64`) must be rejected by raising `TypeError` or `ValueError`.
   - *Rationale:* Preserves SRS design principle "No hidden preprocessing". Silent promotion to `float64` doubles memory bandwidth and cache pressure; silent downcasting masks precision loss. Callers must explicitly cast their data if necessary.

4. **Structured Return Object (`SearchResult` for Top-$k$ Selection):**
   - Top-$k$ selection (`select_top_k`) and subsequent search functions return a `SearchResult` container exposing named attributes:
     - `.distances`: 1D (or 2D if batched) `np.ndarray` of `float32`.
     - `.ids`: 1D (or 2D if batched) `np.ndarray` of `int64`.
   - *API Separation:* Raw distance kernels (such as `pairwise_squared_l2_ref`) compute pairwise distances and return a NumPy array of `float32` distances. They do not return `SearchResult` and do not process IDs. `SearchResult` is exclusively produced by top-$k$ selection and search layers.
   - *Rationale:* Plain tuples like `(distances, ids)` are error-prone during caller unpacking (risk of swapping indices vs distances). Named attributes provide type safety, self-documenting code, and clean indexing across all engine layers.

### 3. Consequences & Invariants
- These contracts define the acceptance interface for the upcoming Phase 1 implementation (to be implemented by the human owner).
- Test specifications in `docs/phase1_test_spec.md` will verify adherence to these contracts once implemented.
- The engine does not yet contain these implementations; when implemented, the distance kernel must return a NumPy `float32` array, and top-$k$ selection must explicitly validate `k >= 1`, validate `arr.dtype == np.float32` upon entry, and return a `SearchResult`.

---

## ADR-0003: Phase 2 Exact Flat Search and Ground Truth API Contracts

- **Status:** Accepted
- **Date:** 2026-10-26
- **Decision Owner:** Shared (Phase 2 Foundation)

### 1. Context
Phase 2 establishes the exact search baseline and ground-truth generation mechanism for Indexora. In vector search engines, approximate nearest-neighbor (ANN) indexes (such as IVF-Flat, PQ, and HNSW) cannot be evaluated without an infallible ground-truth oracle. This decision records the public API contracts, shape conventions, identifier handling, and evaluation invariants for exact search.

### 2. Decisions

1. **Exact Search Execution Path:**
   - Exact search scans all eligible database vectors using the reference distance kernel (`pairwise_squared_l2_ref`) and deterministic top-$k$ selection (`select_top_k`).
   - The functional API `exact_search(queries, database, ids=None, k=10)` and the object-oriented wrapper `ExactFlatIndex(database, ids=None)` provide access to exhaustive search.
   - No acceleration shortcuts, BLAS candidate slack pruning, or ANN approximations are permitted in the exact search baseline.

2. **Durable Identifier Contract:**
   - If `ids` is not provided (`ids=None`), it defaults to contiguous `np.arange(N, dtype=np.int64)`.
   - When custom `ids` are provided, they must be a 1D `int64` NumPy array with length equal to database size $N$, and all values must be strictly unique among live records.
   - Non-contiguous and unsorted IDs are supported and correctly mapped to their respective database vectors.

3. **Query Shape Conventions & Batched Search:**
   - **Single Query:** 1D array of shape `(d,)` returns `SearchResult` with 1D `.distances` of shape `(k',)` and 1D `.ids` of shape `(k',)` where $k' = \min(k, N)$.
   - **Batched Queries:** 2D array of shape `(Q, d)` returns `SearchResult` with 2D `.distances` of shape `(Q, k')` and 2D `.ids` of shape `(Q, k')`.
   - Empty batches ($Q=0$) return 2D arrays of shape `(0, k')`. Empty databases ($N=0$) return shape `(0,)` for 1D queries or `(Q, 0)` for 2D queries.

4. **Ground-Truth Invariant:**
   - Ground truth is generated via `generate_ground_truth(queries, database, ids=None, k=10)`.
   - Ground truth is strictly identical to `exact_search`. Ground truth must never be generated by or approximated with an ANN index.
   - Future Recall@k measurements compare ANN candidate ID sets against this ground truth.

5. **Data Ownership and Buffer Semantics (`ExactFlatIndex`):**
   - `ExactFlatIndex` borrows caller-provided `database` and `ids` arrays by reference without defensive copying, avoiding memory duplication for large benchmark datasets.
   - Properties `.database` and `.ids` expose internal array references directly.
   - *Mutation Implications:* External in-place modification of caller-owned arrays after index construction bypasses initial validation and directly affects subsequent search results. Callers requiring buffer isolation must pass copies explicitly.

### 3. Consequences & Invariants
- All subsequent ANN algorithms (IVF-Flat, PQ, HNSW) must benchmark Recall@k directly against `generate_ground_truth`.
- When IVF-Flat probes every populated inverted list, its output must agree bit-for-bit with `exact_search` under identical composite key ordering.
- Buffer borrowing in `ExactFlatIndex` preserves single-allocation memory efficiency across benchmarks while making mutation caller-responsible.

---

## ADR-0004: Phase 3 Dataset Representation, Normalization, and Caching Contracts

- **Status:** Accepted
- **Date:** 2026-10-26
- **Decision Owner:** Agent-assisted (Phase 3 Utilities)

### 1. Context
Phase 3 introduces the vector data layer needed to supply benchmark vectors, evaluation splits, and training data to upcoming clustering (Phase 4) and indexing (Phase 5) algorithms. This ADR formalizes dataset split representations, strict C-contiguity enforcement, explicit cosine/unit normalization with a zero-vector policy, and offline cache integrity.

### 2. Decisions

1. **Structured Dataset Representation (`VectorDataset`):**
   - Datasets are represented by the immutable `VectorDataset` container managing:
     - `base`: 2D `(N, d)` database vectors to index and search.
     - `queries`: 2D `(Q, d)` evaluation queries.
     - `train`: optional 2D `(T, d)` training vectors (e.g. for fitting k-means centroids or codebooks).
     - `base_ids`, `query_ids`, `train_ids`: 1D `int64` durable identifiers matching split lengths with strictly unique values within each split.
   - All vector arrays must strictly be C-contiguous `float32` arrays with finite values and identical dimensionality $d$.
   - Empty splits ($N=0, Q=0, T=0$) with matching column dimensions are fully supported.

2. **Explicit Cosine / L2 Unit-Normalization (`normalize_l2`):**
   - Implements the required ADR-0001 explicit normalization utility:
     $$\hat{v} = \frac{v}{\|v\|_2}$$
   - **Zero-Vector Policy:**
     - `zero_vector_policy="zero"` (default): Vectors with zero (or near-zero $\le 10^{-12}$) norm remain all zeros (`0.0`), preventing NaN poisoning or division-by-zero runtime warnings while maintaining valid float32 values.
     - `zero_vector_policy="error"`: Raises `ValueError("Zero-norm vector encountered...")`.
   - The engine performs no automatic or silent normalization during indexing or search.

3. **Reproducible Clustered & Synthetic Generators:**
   - `generate_synthetic_dataset`: Generates uniform/normal random splits with explicit seeds.
   - `generate_clustered_dataset`: Generates vectors sampled around well-separated centroids with Gaussian noise, returning `ClusteredDataset` with ground-truth cluster labels (`base_labels`) and `centroids`.

4. **Offline Cache & Checksum Integrity:**
   - `save_dataset` and `load_dataset` serialize splits to standard NumPy `.npy` files with a `metadata.json` manifest.
   - Every file's SHA-256 hexadecimal digest is computed and verified on load.
   - Corrupted or tampered files are rejected immediately with descriptive `ValueError`.
   - All dataset utilities and unit tests operate strictly offline without network calls.

### 3. Consequences & Invariants
- Clustering algorithms (k-means) and IVF-Flat can directly ingest `dataset.train` or `dataset.base` with guaranteed C-contiguity and type safety.
- Ground-truth evaluation scripts can verify recall against `dataset.queries` and `dataset.base_ids`.
