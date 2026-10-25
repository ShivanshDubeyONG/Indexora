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
