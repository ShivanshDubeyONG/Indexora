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
