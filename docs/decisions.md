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

---

## ADR-0005: Phase 4 K-means Clustering and Centroid Initialization Contracts

- **Status:** Proposed (Contract Specification for Human Core)
- **Date:** 2026-10-26
- **Decision Owner:** Human Core (Phase 4 Algorithm Ownership)

### 1. Context
Phase 4 introduces unsupervised vector partitioning via $k$-means clustering, which serves as the coarse codebook generator for IVF-Flat (Phase 5) and product quantization (Phase 8). Per `plan/implementation.md`, the core initialization ($k$-means++) and iteration loop (Lloyd's algorithm) are owned by the human author. This decision documents the public API contracts, array shapes, hyperparameters, empty-cluster policies, and mathematical invariants prior to human implementation.

### 2. Decisions

1. **Public API Signatures & Return Types:**
   - Centroid initialization:
     `kmeans_plus_plus(X: np.ndarray, k: int, seed: int | np.random.Generator | None = None) -> np.ndarray`
     Returns `(k, d)` C-contiguous `float32` array of initial centroids.
   - Lloyd clustering:
     `kmeans(X: np.ndarray, k: int, max_iter: int = 100, tol: float = 1e-4, seed: int | np.random.Generator | None = None, empty_cluster_policy: str = "reseed_furthest") -> KMeansResult`
     Returns an immutable `KMeansResult` dataclass with attributes:
     - `.centroids`: `(k, d)` array (`np.float32`)
     - `.labels`: `(N,)` array (`np.int64`), where each label $0 \le \text{label}[i] < k$
     - `.inertia`: Python `float`, computed with `float64` accumulation
     - `.n_iter`: `int`, count of executed Lloyd iterations
     - `.converged`: `bool`, whether centroid movement stopped within `tol`
     - `.inertia_history`: `tuple[float, ...]`, recorded inertia per completed iteration

2. **Strict Vector Validation & Array Ownership:**
   - Input matrix `X` must be a 2D NumPy array of shape `(N, d)` with strict `dtype=np.float32`, C-contiguous, and finite (`np.isfinite`). Silent type promotion or downcasting is prohibited.
   - Matrix `X` is borrowed read-only; neither `kmeans_plus_plus` nor `kmeans` may mutate caller-owned buffers.
   - Boundaries on $k$:
     - $k < 1$ or non-integer $k$: raises `ValueError` / `TypeError`.
     - $k > N$: raises `ValueError("k cannot exceed number of samples N")`.
     - $k = N$: valid boundary where each sample is its own centroid with inertia $0.0$.

3. **K-means++ Degenerate-Distance Fallback Policy:**
   - For step $j \in \{1, \dots, k-1\}$, minimum squared distances $D(x_i)^2 = \min_{m < j} \|x_i - c_m\|_2^2$ are computed across all $N$ rows.
   - **Standard Path ($\sum D(x_i)^2 > 0$):** Sample candidate index $i$ with probability $P(i) = D(x_i)^2 / \sum D(x)^2$. Any row identical to an already selected centroid has $D(x_i)^2 = 0$ and is never chosen.
   - **Degenerate Path ($\sum D(x_i)^2 == 0$):** Occurs when all remaining points in $X$ are exact duplicates of already-selected centroids (or when all $N$ rows in $X$ are identical).
     - The algorithm tracks the set of already-chosen row indices (`selected_indices`).
     - If $\sum D^2 == 0$, candidate pool is restricted to unselected row indices:
       $$\text{unselected} = \{i \in \{0, \dots, N-1\} \setminus \text{selected\_indices}\}$$
     - Since $k \le N$, $|\text{unselected}| \ge 1$. The next centroid row index is sampled uniformly from $\text{unselected}$ using `rng.choice(unselected)`.
     - *Invariant:* Guarantees that $k$ distinct row indices are selected from $X$ without duplicate index reuse when unselected rows remain, avoiding NaNs and infinite loops.

4. **Deterministic Assignment & Tie-Breaking:**
   - Pairwise distance between samples $X$ and candidate centroids $C$ is evaluated via squared Euclidean distance $\|x_i - c_j\|_2^2$ (compatible with `pairwise_squared_l2_ref`).
   - Tie-breaking: If sample $x_i$ is equidistant from multiple centroids $j_1, j_2$ ($j_1 < j_2$), `np.argmin` deterministically assigns $x_i$ to the lower centroid index $j_1$.

5. **Empty-Cluster Handling Policy (`reseed_furthest`):**
   - Pathological data distributions or high $k$ relative to $N$ can produce empty clusters during Lloyd iterations ($|S_j| = 0$).
   - Supported policies via `empty_cluster_policy`:
     - `"reseed_furthest"` (default):
       1. Identify candidate points: Find the vector $x_{i^*}$ currently assigned to a multi-point cluster ($|S| > 1$) with the maximum squared distance to its assigned centroid:
          $$i^* = \arg\max_{i} \|x_i - c_{\text{labels}[i]}\|_2^2$$
          (Ties broken by smallest index $i^*$; if multiple clusters are empty, candidates are assigned greedily by descending distance).
       2. Reseed the empty centroid: Set $c_j = X[i^*]$.
       3. Non-empty clusters update to the arithmetic mean of their assigned points.
       4. Centroid shift evaluation includes the reseeded coordinate, driving $\|\Delta C\|_F^2 > 0$ to prevent premature false convergence.
     - `"keep"`: Retains the centroid's coordinate from the previous iteration.
     - `"error"`: Raises `RuntimeError` immediately upon detecting an empty cluster.

6. **Convergence Criteria & Final Return Synchronization:**
   - Convergence is achieved when the squared Frobenius norm of centroid shift across consecutive iterations satisfies:
     $$\Delta C = \sum_{j=0}^{k-1} \|c_j^{(t)} - c_j^{(t-1)}\|_2^2 \le \text{tol}$$
   - Inertia is defined as the total sum of squared distances to assigned centroids:
     $$\text{Inertia} = \sum_{i=0}^{N-1} \|x_i - c_{\text{labels}[i]}\|_2^2$$
   - **Final Synchronization Invariant:**
     Upon loop termination (convergence or reaching `max_iter`), a final assignment pass over $X$ is executed using the final `centroids`:
     $$\text{labels} = \arg\min_j \|x_i - c_j\|_2^2, \quad \text{inertia} = \sum_{i=0}^{N-1} \min_j \|x_i - c_j\|_2^2$$
     This guarantees that `KMeansResult.labels`, `centroids`, and `inertia` are 100% mutually consistent, with `labels` strictly representing the nearest-centroid partition.

### 3. Consequences & Invariants
- Phase 5 (IVF-Flat) can train its coarse centroids directly using `kmeans(dataset.train or dataset.base, k=nlist, seed=seed)`.
- Synthetic multi-cluster datasets (`generate_clustered_dataset`) can be used to mathematically verify that $k$-means converges cleanly on well-separated data.

---

## ADR-0006: Phase 5 IVF-Flat Architecture, Inverted Lists, and Multi-Probe Search

- **Status:** Proposed (Design & Contract Specification for Human Core)
- **Date:** 2026-10-27
- **Decision Owner:** Human Core (Phase 5 Algorithm Ownership)

### 1. Context
Phase 5 introduces Indexora's first approximate nearest neighbor (ANN) index: **IVF-Flat (Inverted File with Flat Vectors)**. Inverted file indexing trades search recall for latency by partitioning the vector space into $W = \text{nlist}$ Voronoi cells using coarse centroids trained via $k$-means. At query time, rather than performing exhaustive brute-force distance comparisons against all $N$ database vectors, the engine finds the $P = \text{nprobe}$ nearest coarse centroids and restricts exact distance calculations to candidate vectors stored in those $P$ inverted lists.

Per `plan/implementation.md`, Phase 5 is designated **HUMAN CORE**. The core indexing assignment loop and multi-probe query loop belong to the human author. This record formalizes the data structures, indexing invariants, candidate gathering semantics, and decisive acceptance criteria.

### 2. Decisions

1. **Inverted List Container (`InvertedList`):**
   - Each inverted list partition is represented by an immutable dataclass `InvertedList`:
     - `ids: np.ndarray`: 1D array of durable vector identifiers with strict `dtype=np.int64`.
     - `vectors: np.ndarray`: 2D array of shape `(L, d)` with strict `dtype=np.float32`, guaranteed C-contiguous in memory.
   - **Empty List Representation:** If a coarse cluster is empty ($L = 0$), the list contains `ids = np.empty((0,), dtype=np.int64)` and `vectors = np.empty((0, d), dtype=np.float32)`. Empty lists are fully valid and must not cause division-by-zero, indexing errors, or NaNs.

2. **Index Class Structure (`IVFFlatIndex`):**
   - The index exposes:
     - `.d`: Feature dimension ($d \ge 1$).
     - `.ntotal`: Total number of indexed vectors ($N \ge 1$).
     - `.nlist`: Number of coarse inverted lists ($1 \le \text{nlist} \le N$).
     - `.nprobe`: Number of inverted lists probed per query ($1 \le \text{nprobe} \le \text{nlist}$). Configurable dynamically via setter property or query-time parameter.
     - `.centroids`: Array of shape `(nlist, d)` with `dtype=np.float32`.
     - `.invlists`: Tuple of length `nlist` containing the individual `InvertedList` instances.
     - `list_sizes()`: 1D `int64` array of shape `(nlist,)` returning vector occupancy counts across all lists.
     - `get_list(list_id: int)`: Direct lookup for list partition $j \in [0, \text{nlist}-1]$.

3. **Core Partition Invariants:**
   - **Exhaustive Single-List Assignment:** Every database vector $x_i$ must belong to exactly one inverted list:
     $$\sum_{j=0}^{\text{nlist}-1} |S_j| = N$$
   - **Identifier Preservation:** The union of all IDs stored across all inverted lists must form an exact permutation of the original database IDs:
     $$\bigcup_{j=0}^{\text{nlist}-1} \text{invlists}[j].ids \equiv \text{database\_ids}$$
     No database vector is dropped, duplicated, or reassigned to a synthetic ID.

4. **Multi-Probe Candidate Gathering & Search Semantics:**
   - For each query vector $q \in \mathbb{R}^d$:
     1. **Coarse Quantization:** Compute squared Euclidean distances between $q$ and all $\text{nlist}$ coarse centroids using `pairwise_squared_l2_ref(q[None, :], centroids)`.
     2. **Centroid Ranking:** Select the nearest $P = \text{nprobe}$ centroid indices. Centroid ties are broken deterministically by lower centroid index.
     3. **Candidate Gathering:** Pool all vectors and IDs from the selected $P$ inverted lists:
        $$X_{\text{cand}} = \text{concatenate}([S_{j_1}.\text{vectors}, \dots, S_{j_P}.\text{vectors}])$$
        $$\text{IDs}_{\text{cand}} = \text{concatenate}([S_{j_1}.\text{ids}, \dots, S_{j_P}.\text{ids}])$$
        If all probed lists are empty ($N_{\text{cand}} = 0$), return empty results or fall back according to standard top-$k$ contracts.
     4. **Exact Candidate Scoring & Top-$k$ Selection:**
        - Compute exact squared Euclidean distances between $q$ and $X_{\text{cand}}$ using `pairwise_squared_l2_ref`.
        - Rank candidate vectors and select top-$k$ using `select_top_k(distances, IDs_{\text{cand}}, k)`.
        - Result sorting and tie-breaking strictly adhere to ADR-0001: ascending by distance, breaking equal-distance ties by ascending `int64` ID.

5. **Decisive Acceptance Gate (All-Probe Equivalence):**
   - When $\text{nprobe} == \text{nlist}$, every inverted list is inspected.
   - The candidate set $X_{\text{cand}}$ contains all $N$ database vectors, and $\text{IDs}_{\text{cand}}$ contains all $N$ database IDs.
   - **Invariant:** With $\text{nprobe} == \text{nlist}$, IVF-Flat search must produce top-$k$ results bit-for-bit identical to `exact_search` in both `.ids` and `.distances`.
   - Partial probing ($\text{nprobe} < \text{nlist}$) must be evaluated empirically against exact ground truth; recall values are never hard-coded or assumed.

### 3. Consequences & Invariants
- Zero vector loss during partitioning; index validation guarantees memory safety and strict dtype enforcement (`float32` vectors, `int64` IDs).
- Complete determinism across all queries and runs.
- Clean foundation for Phase 6 benchmarking sweeps (`nprobe` vs. recall@k and latency).
