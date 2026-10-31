# Indexora — First-Principles Vector Search Practice Guide

Comprehensive engineering reference and interview practice specification for core vector search algorithms implemented from scratch in Python and NumPy.

---

## Table of Contents
1. [Core Philosophy & Theoretical Framework](#core-philosophy--theoretical-framework)
2. [Problem 1: Exact Vector Search (BLAS GEMM & Deterministic Top-k)](#problem-1-exact-vector-search)
   - [Problem Statement & Preconditions](#11-problem-statement--preconditions)
   - [Mathematical Derivations & Precision Analysis](#12-mathematical-derivations--precision-analysis)
   - [Algorithmic Breakdown & Matrix Algebra](#13-algorithmic-breakdown--matrix-algebra)
   - [Complexity Analysis (FLOPs & Memory Footprint)](#14-complexity-analysis)
   - [Edge Cases & Boundary Conditions](#15-edge-cases--boundary-conditions)
   - [System Invariants](#16-system-invariants)
   - [Technical Interview Q&A](#17-technical-interview-qa)
   - [Clean Whiteboard Implementation from Memory](#18-clean-whiteboard-implementation-from-memory)
   - [Self-Contained Verification Fixtures](#19-self-contained-verification-fixtures)
3. [Problem 2: Hierarchical Navigable Small World (HNSW)](#problem-2-hierarchical-navigable-small-world-hnsw)
   - [Problem Statement & Architectural Invariants](#21-problem-statement--architectural-invariants)
   - [Mathematical Foundations & Skip-List Generalization](#22-mathematical-foundations)
   - [Algorithmic Walkthrough & Heuristics](#23-algorithmic-walkthrough--heuristics)
   - [Complexity Analysis (Construction, Query, Memory)](#24-complexity-analysis)
   - [Edge Cases & Graph Topology Degradation](#25-edge-cases--graph-topology-degradation)
   - [System Invariants](#26-system-invariants)
   - [Technical Interview Q&A](#27-technical-interview-qa)
   - [Clean Whiteboard Implementation from Memory](#28-clean-whiteboard-implementation-from-memory)
   - [Self-Contained Verification Fixtures](#29-self-contained-verification-fixtures)
4. [Algorithmic Verification Checklist](#algorithmic-verification-checklist)

---

## Core Philosophy & Theoretical Framework

Modern vector search engines balance three core constraints:
1. **Search Accuracy (Recall@k):** The fraction of true nearest neighbors retrieved relative to an exhaustive ground truth.
2. **Query Latency & Throughput (QPS):** Response time distribution ($p_{50}, p_{95}, p_{99}$) and system capacity under concurrent batch evaluation.
3. **Index Footprint (Bytes/Vector):** RAM requirements for graph topologies, inverted lists, and quantized codebooks.

In learning-first engineering, understanding the mechanical boundaries between CPU hardware instructions (Level-3 BLAS GEMM), memory bandwidth (cache hierarchies, contiguous row-major layouts), and Python interpreter overhead is essential for defensible systems design.

---

## Problem 1: Exact Vector Search

### 1.1 Problem Statement & Preconditions

Given a query matrix $Q \in \mathbb{R}^{B \times d}$ and a database matrix $X \in \mathbb{R}^{N \times d}$, retrieve for each query vector $q_i$ the $k$ database vectors with the smallest squared Euclidean ($L_2$) distances:

$$d^2(q, x) = \sum_{j=1}^d (q_j - x_j)^2$$

**Preconditions:**
- Query matrix $Q$ has shape $(B, d)$ and database matrix $X$ has shape $(N, d)$, with dtype `float32`.
- Both matrices are contiguous row-major (`C_CONTIGUOUS`).
- Query count $B \ge 1$, database size $N \ge 1$, dimension $d \ge 1$, target neighbor count $k \ge 1$.
- External integer IDs are non-negative `int64` identifiers uniquely mapped to rows of $X$.

**Output Contract:**
- `ids`: Integer array of shape $(B, k)$ of dtype `int64` containing the top-$k$ IDs for each query.
- `distances`: Float array of shape $(B, k)$ of dtype `float32` containing corresponding squared $L_2$ distances.
- Result rows are sorted in non-decreasing distance order. Equidistant candidates must break ties deterministically by ascending external ID.

---

### 1.2 Mathematical Derivations & Precision Analysis

#### Direct Difference vs. Binomial Expansion
The naive textbook approach evaluates coordinate differences directly:
$$d^2(q, x) = \sum_{j=1}^d (q_j - x_j)^2$$

This requires allocating intermediate tensors of size $(B, N, d)$ or running elementwise loops. In Python, an unvectorized elementwise loop suffers severe interpreter overhead ($O(B \cdot N \cdot d)$ bytecode dispatches).

Expanding the binomial form:
$$\|q - x\|^2 = \|q\|^2 + \|x\|^2 - 2 (q \cdot x)$$

This decomposes the all-pairs computation into three distinct operations:
1. Query squared norms: $\|q_i\|^2 = \sum_{j=1}^d q_{i,j}^2$ (shape: $B \times 1$)
2. Database squared norms: $\|x_m\|^2 = \sum_{j=1}^d x_{m,j}^2$ (shape: $1 \times N$)
3. Cross inner products: $G = Q X^T$ (matrix multiplication of shape $(B, d) \times (d, N) \to (B, N)$)

#### Floating-Point Cancellation & Clamping
While binomial expansion enables Level-3 BLAS acceleration, it introduces potential floating-point rounding hazards:
When $x \approx q$, $\|q\|^2 + \|x\|^2 \approx 2(q \cdot x)$. In single-precision IEEE 754 arithmetic (24 bits of mantissa precision, $\approx 7.22$ decimal digits):
$$\|q\|^2 + \|x\|^2 - 2(q \cdot x)$$
can produce small negative numbers (e.g., $-1.42 \times 10^{-7}$) due to subtractive cancellation.

**Mitigation:**
1. All dot products are computed using optimized BLAS GEMM (`np.dot` / `gemm`), which maintains 64-bit accumulators internally on modern SIMD hardware before down-casting.
2. Explicit numerical clamping via `np.maximum(dist_matrix, 0.0, out=dist_matrix)` guarantees non-negativity without altering candidate ordering.

---

### 1.3 Algorithmic Breakdown & Matrix Algebra

```
Given Q (B x d), X (N x d):

1. Compute Query Norms:
   q_norms = sum(Q * Q, axis=1, keepdims=True)         # (B, 1)

2. Compute Database Norms:
   x_norms = sum(X * X, axis=1, keepdims=True).T       # (1, N)

3. Compute Cross Gram Matrix via Level-3 BLAS GEMM:
   G = dot(Q, X.T)                                     # (B, N)

4. Compute All-Pairs Squared Distance Matrix:
   D = q_norms + x_norms - 2.0 * G                     # Broadcasted (B, N)
   np.maximum(D, 0.0, out=D)

5. Partition & Deterministic Sorting (per row):
   partition_idx = argpartition(D, kth=k-1, axis=1)[:, :k]
   # Sort top-k partition by (distance, id) lexicographical order
```

#### Chunking for Bounded Memory Footprint
When $N$ or $B$ is large (e.g. $B=10,000, N=1,000,000$), allocating the full $(B \times N)$ distance matrix in memory requires:
$$10,000 \times 1,000,000 \times 4\text{ bytes} \approx 40\text{ GB RAM}$$

To guarantee bounded memory allocations, database vectors are processed in blocks of `max_db_chunk` (e.g. 32,768 vectors $\approx 16.7$ MB per chunk for $d=128$), maintaining a running top-$k$ heap or partial-sorted pool across blocks.

---

### 1.4 Complexity Analysis

| Stage | Operation | FLOPs / Work | Auxiliary Memory |
| :--- | :--- | :--- | :--- |
| **Query Norms** | $\|q_i\|^2$ | $2 \cdot B \cdot d$ | $O(B)$ |
| **Database Norms** | $\|x_m\|^2$ | $2 \cdot N \cdot d$ | $O(N)$ (Precomputed) |
| **Gram Matrix** | $Q X^T$ (GEMM) | $2 \cdot B \cdot N \cdot d$ | $O(B \cdot N)$ |
| **Distance Matrix** | Broadcasted Sum | $3 \cdot B \cdot N$ | In-place on Gram matrix |
| **Top-$k$ Selection**| `argpartition` + Sort | $O(B \cdot N + B \cdot k \log k)$ | $O(B \cdot k)$ |
| **Total** | | **$2BNd + O(BN)$** | **$O(B \cdot \min(N, \text{chunk\_size}))$** |

---

### 1.5 Edge Cases & Boundary Conditions

1. **$k \ge N$:** When the requested neighbor count exceeds the dataset size, clamp effective $k = \min(k, N)$. Return all $N$ vectors sorted by distance.
2. **Duplicate Vectors:** When multiple database vectors have identical coordinates ($d^2 = 0$), sort by ascending external integer ID.
3. **Zero Vectors ($x = \mathbf{0}$):** $\|x\|^2 = 0$, $q \cdot x = 0$, distance correctly reduces to $\|q\|^2$.
4. **NaN or Inf Coordinates:** Input validation must verify `np.isfinite(Q).all()` and `np.isfinite(X).all()`. If present, raise `ValueError` before computing norms.
5. **Dimension Mismatch:** $Q.\text{shape}[1] \ne X.\text{shape}[1]$ must raise an explicit dimension error.

---

### 1.6 System Invariants

1. **Exact Monotonicity:** $\forall i, \forall j \in [0, k-2]: \text{distances}[i, j] \le \text{distances}[i, j+1]$.
2. **Strict Determinism:** Identical inputs $(Q, X, \text{ids}, k)$ produce bit-for-bit identical $(ids, distances)$ across all runs, regardless of BLAS thread scheduling.
3. **Non-Negativity:** $\forall i, j: \text{distances}[i, j] \ge 0.0$.

---

### 1.7 Technical Interview Q&A

**Q1: Why do vector search engines compute squared Euclidean distance ($d^2$) instead of Euclidean distance ($d$)?**
> Because the square root function $f(z) = \sqrt{z}$ is strictly monotonically increasing for $z \ge 0$:
> $$z_1 < z_2 \iff \sqrt{z_1} < \sqrt{z_2}$$
> Preserving the relative ranking of candidates does not require computing $\sqrt{\cdot}$. Avoiding $N$ square-root operations per query saves substantial compute cycles without altering top-$k$ results.

**Q2: What is the primary performance bottleneck of brute-force search on modern CPUs?**
> Brute-force search is memory-bandwidth bound when vectors cannot fit in CPU L1/L2 caches. For large $N$, the CPU cores spend clock cycles waiting for memory lines to be fetched from DRAM. Level-3 BLAS GEMM mitigates this by tiling matrices into blocks that fit within L1/L2/L3 caches, achieving arithmetic intensity near the CPU's theoretical peak FLOPS.

**Q3: How does `argpartition` improve top-$k$ selection over full sorting?**
> A full quicksort takes $O(N \log N)$ time. Introsort-based `argpartition` (based on Quickselect) partitions the array around the $k$-th smallest element in $O(N)$ average time. Only the top-$k$ elements are then sorted in $O(k \log k)$ time, reducing selection overhead from $O(N \log N)$ to $O(N + k \log k)$. When $N = 100,000$ and $k=10$, this is a $>10\times$ speedup for the top-$k$ selection phase.

---

### 1.8 Clean Whiteboard Implementation from Memory

```python
import numpy as np

def exact_search_blas(
    queries: np.ndarray,
    database: np.ndarray,
    ids: np.ndarray,
    k: int = 10,
    db_norms: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exhaustive exact vector search using Level-3 BLAS matrix multiplication.

    Parameters
    ----------
    queries : np.ndarray
        Query matrix of shape (B, d), dtype float32.
    database : np.ndarray
        Database matrix of shape (N, d), dtype float32.
    ids : np.ndarray
        External ID array of shape (N,), dtype int64.
    k : int
        Number of nearest neighbors to retrieve.
    db_norms : np.ndarray | None
        Optional precomputed squared norms of database vectors of shape (N, 1).

    Returns
    -------
    top_ids : np.ndarray
        Retrieved neighbor IDs of shape (B, k), dtype int64.
    top_dists : np.ndarray
        Squared L2 distances of shape (B, k), dtype float32.
    """
    Q = np.ascontiguousarray(queries, dtype=np.float32)
    X = np.ascontiguousarray(database, dtype=np.float32)
    B, d = Q.shape
    N, d_db = X.shape
    assert d == d_db, f"Dimension mismatch: {d} vs {d_db}"

    eff_k = min(k, N)

    # 1. Compute squared L2 norms
    q_norms = np.sum(Q * Q, axis=1, keepdims=True)  # (B, 1)
    if db_norms is None:
        db_norms = np.sum(X * X, axis=1, keepdims=True)  # (N, 1)
    x_norms_t = db_norms.reshape(1, N)  # (1, N)

    # 2. Level-3 BLAS GEMM for cross inner-products: (B, d) @ (d, N) -> (B, N)
    # ||q - x||^2 = ||q||^2 + ||x||^2 - 2 (q . x)
    gram = np.dot(Q, X.T)
    dists = q_norms + x_norms_t - 2.0 * gram
    np.maximum(dists, 0.0, out=dists)

    # 3. Top-k selection per query
    out_ids = np.empty((B, eff_k), dtype=np.int64)
    out_dists = np.empty((B, eff_k), dtype=np.float32)

    for i in range(B):
        row_dists = dists[i]
        # Partial sort to isolate smallest eff_k candidates in O(N)
        partition_idx = np.argpartition(row_dists, kth=eff_k - 1)[:eff_k]
        part_dists = row_dists[partition_idx]
        part_ids = ids[partition_idx]

        # Deterministic lexicographical tie-break: primary=(dist), secondary=(id)
        sort_order = np.lexsort((part_ids, part_dists))
        sorted_idx = partition_idx[sort_order]

        out_ids[i] = ids[sorted_idx]
        out_dists[i] = row_dists[sorted_idx]

    return out_ids, out_dists
```

---

### 1.9 Self-Contained Verification Fixtures

```python
def test_exact_search_verification():
    rng = np.random.default_rng(42)
    N, B, d, k = 1000, 10, 64, 5

    X = rng.standard_normal((N, d)).astype(np.float32)
    Q = rng.standard_normal((B, d)).astype(np.float32)
    ids = np.arange(100, 100 + N, dtype=np.int64)

    # Run BLAS implementation
    res_ids, res_dists = exact_search_blas(Q, X, ids, k=k)

    # Reference naive oracle computation
    for q_idx in range(B):
        diff = X - Q[q_idx]
        true_dists = np.sum(diff * diff, axis=1)
        # Lexicographical sort
        ref_order = np.lexsort((ids, true_dists))[:k]

        assert np.array_equal(res_ids[q_idx], ids[ref_order]), f"ID mismatch for query {q_idx}"
        np.testing.assert_allclose(res_dists[q_idx], true_dists[ref_order], rtol=1e-5, atol=1e-5)
    print("Exact search verification: PASSED (100% agreement with reference oracle)")

if __name__ == "__main__":
    test_exact_search_verification()
```

---

## Problem 2: Hierarchical Navigable Small World (HNSW)

### 2.1 Problem Statement & Architectural Invariants

An Approximate Nearest Neighbor (ANN) proximity graph structures database vectors as nodes in a multi-layer graph $G = (V, E)$, optimizing the fundamental trade-off between search recall and distance evaluation count.

**Key Invariants:**
1. **Layer Hierarchy:** The graph consists of $L+1$ layers numbered $0, 1, \dots, L$, where layer 0 contains all vectors in $V$, and upper layers contain exponentially sparser subsets.
2. **Degree Bounding:** Every node has a maximum out-degree bounded by $M$ on layers $l > 0$, and $M_0 = 2M$ on ground layer 0.
3. **Single Entry Point:** Graph traversal begins at a top-layer entry node $ep \in V_L$.

---

### 2.2 Mathematical Foundations

#### Skip-List Generalization & Layer Assignment
HNSW generalizes 1D probabilistic skip-lists to metric spaces. The maximum level $l$ for an inserted vector $v$ is sampled from an exponential distribution parameterized by normalization factor $m_L$:

$$l = \lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \rfloor \quad \text{where} \quad m_L = \frac{1}{\ln(M)}$$

This ensures that the probability of an element residing on layer $l$ decays exponentially:
$$P(\text{level} \ge l) = M^{-l}$$
Under this scaling:
- Expected number of layers: $O(\log N)$.
- Overlap between layers maintains graph connectivity without bottlenecks.

#### Small-World Clustering & Navigability
A network exhibits small-world properties if its clustering coefficient is significantly higher than a random graph, while average path length scales logarithmically:
$$\langle L \rangle \sim O(\log N)$$
Greedy routing on upper layers provides long-range "express lane" routing, navigating across coarse geometric neighborhoods before transitioning to dense ground layers for fine-grained local convergence.

---

### 2.3 Algorithmic Walkthrough & Heuristics

#### Algorithm 1: Greedy Top-Down Routing (Layers $L \to l_c + 1$)
On upper layers above the target insertion/search layer, routing is pure 1-greedy:
1. Initialize candidate node $c \leftarrow ep$.
2. For each neighbor $e \in \text{neighbors}(c, \text{layer})$:
   - If $d(q, e) < d(q, c)$, set $c \leftarrow e$ and repeat.
3. When no neighbor is closer to $q$ than $c$, step down to $\text{layer} - 1$ using $c$ as the entry point.

#### Algorithm 2: SearchLayer (Beam Search on Layer $l$)
On layers where thorough exploration is required:
1. Maintain two priority queues:
   - Candidates queue $C$ (min-heap ordered by distance to $q$).
   - Best-found results $W$ (max-heap of size $ef$ ordered by distance to $q$).
2. Maintain a `visited` set to prevent re-evaluating nodes.
3. While $C$ is not empty:
   - Extract closest candidate $c \in C$.
   - Extract furthest result $f = \max(W)$.
   - If $d(q, c) > d(q, f)$, terminate search (no remaining candidate in $C$ can improve $W$).
   - For each neighbor $e \in \text{neighbors}(c, \text{layer})$:
     - If $e \notin \text{visited}$:
       - Mark $e \in \text{visited}$.
       - If $d(q, e) < d(q, f)$ or $|W| < ef$:
         - Add $e$ to $C$ and $W$.
         - If $|W| > ef$, evict furthest node from $W$.
4. Return $W$.

#### Algorithm 3: Malkov & Yashunin's Heuristic Diverse Pruning (Algorithm 4)
Naive closest-neighbor selection creates dense clusters of redundant edges pointing to the same local neighborhood, destroying navigability. Algorithm 4 enforces **geometric spatial diversity**:

A candidate $e$ is connected to base node $u$ only if $e$ is closer to $u$ than to any neighbor already selected in the neighbor set $R$:
$$\forall r \in R: d(e, u) < d(e, r)$$

This forms a Voronoi-like directional coverage spanning diverse angles from node $u$.

---

### 2.4 Complexity Analysis

| Phase | Metric | Expected Complexity | Governing Parameters |
| :--- | :--- | :--- | :--- |
| **Insertion** | Time per vector | $O(ef_c \cdot M \cdot \log N)$ | $ef_{\text{construction}}, M$ |
| **Full Build** | Total Construction Time | $O(N \log N \cdot ef_c \cdot M)$ | $N, ef_c, M$ |
| **Search Query** | Time per query | $O(ef_s \cdot M \cdot \log N)$ | $ef_{\text{search}}, M$ |
| **Graph Memory** | Storage Overhead | $O(N \cdot M)$ integers | $M_0 = 2M, M$ |

---

### 2.5 Edge Cases & Graph Topology Degradation

1. **Empty Graph / Initial Node:** The first inserted vector becomes the root entry point at its sampled level without edge connections.
2. **Disconnected Components:** If $ef_{\text{construction}}$ is set too small, or if diverse pruning is overly aggressive, the graph can fracture into disconnected components. Mitigate by setting $ef_c \ge 2M$.
3. **High Dimensionality (Curse of Dimensionality):** In spaces with $d > 500$, distance concentration makes all pairwise distances roughly equal ($\frac{\sigma_d}{\mu_d} \to 0$). Hubness emerges where a few nodes appear as nearest neighbors to disproportionate numbers of points.
4. **Tombstone Deletion Semantics:** True node deletion in proximity graphs risks severing routing paths. Indexora employs logical **tombstoning**: deleted nodes remain in graph layers to preserve routing connectivity for other points, but are masked out from final search results.

---

### 2.6 System Invariants

1. **Degree Bound Invariant:** $\forall u \in V, \forall l > 0: |\text{neighbors}(u, l)| \le M$; for layer 0: $|\text{neighbors}(u, 0)| \le M_0 = 2M$.
2. **Entry Point Layer Reachability:** The global entry point $ep$ always resides on the maximum active graph level $L = \max_{u \in V} \text{level}(u)$.
3. **Graph Symmetry & Bidirectionality:** If node $u$ adds neighbor $v$, node $v$ attempts to add $u$ back (subject to degree pruning).

---

### 2.7 Technical Interview Q&A

**Q1: Why does HNSW use a multi-layer hierarchy instead of searching a single flat graph?**
> A single flat proximity graph suffers from the local minimum problem: greedy routing gets trapped in local geometric valleys. Upper layers act as long-range "expressways" that skip over valleys and reach the general vicinity of the target in $O(\log N)$ hops. Once near the destination, lower layers perform dense local refinement.

**Q2: What is the purpose of the diversity heuristic (Algorithm 4)?**
> The diversity heuristic prevents redundant clustering. If point $B$ is very close to point $A$, connecting both $A$ and $B$ to node $U$ provides almost no extra directional information. By rejecting candidate $B$ when $d(U, B) > d(A, B)$, the algorithm forces edges to point in divergent directions, maximizing the graph's angular coverage and preventing dead-ends.

**Q3: How does $ef_{\text{search}}$ navigate the recall-latency tradeoff?**
> $ef_{\text{search}}$ dictates the size of the dynamic candidate priority queue during layer-0 search. Larger $ef_s$ inspects more candidates, decreasing the chance of terminating in a false local minimum (higher recall) at the cost of computing more vector distances (lower QPS).

---

### 2.8 Clean Whiteboard Implementation from Memory

```python
import math
import heapq
import numpy as np

class MinimalHNSW:
    """Whiteboard-ready self-contained HNSW implementation."""

    def __init__(self, dim: int, M: int = 16, ef_c: int = 32, seed: int = 42):
        self.dim = dim
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_c
        self.mL = 1.0 / math.log(M)
        self.rng = np.random.default_rng(seed)

        self.vectors: list[np.ndarray] = []
        self.ids: list[int] = []
        self.levels: list[int] = []
        # graph[level][node_idx] = list of neighbor node_idx
        self.graph: list[dict[int, list[int]]] = []
        self.entry_point: int | None = None
        self.max_level: int = -1

    def _dist(self, u_vec: np.ndarray, v_vec: np.ndarray) -> float:
        diff = u_vec - v_vec
        return float(np.dot(diff, diff))

    def _sample_level(self) -> int:
        unif = max(self.rng.uniform(0.0, 1.0), 1e-10)
        return int(-math.log(unif) * self.mL)

    def _search_layer(self, q_vec: np.ndarray, ep: int, ef: int, level: int) -> list[tuple[float, int]]:
        visited = {ep}
        d_ep = self._dist(q_vec, self.vectors[ep])
        # candidates: min-heap (dist, node)
        candidates = [(d_ep, ep)]
        # results: max-heap (-dist, node)
        results = [(-d_ep, ep)]

        while candidates:
            c_dist, c_node = heapq.heappop(candidates)
            furthest_dist = -results[0][0]

            if c_dist > furthest_dist:
                break

            for neighbor in self.graph[level].get(c_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    n_dist = self._dist(q_vec, self.vectors[neighbor])
                    furthest_dist = -results[0][0]

                    if n_dist < furthest_dist or len(results) < ef:
                        heapq.heappush(candidates, (n_dist, neighbor))
                        heapq.heappush(results, (-n_dist, neighbor))
                        if len(results) > ef:
                            heapq.heappop(results)

        return sorted([(-d, n) for d, n in results])

    def insert(self, ext_id: int, vec: np.ndarray) -> None:
        u = len(self.vectors)
        self.vectors.append(vec.astype(np.float32))
        self.ids.append(ext_id)

        insert_level = self._sample_level()
        self.levels.append(insert_level)

        while len(self.graph) <= insert_level:
            self.graph.append({})
        for l in range(insert_level + 1):
            self.graph[l][u] = []

        if self.entry_point is None:
            self.entry_point = u
            self.max_level = insert_level
            return

        curr_ep = self.entry_point
        # 1. Greedy routing down to insert_level + 1
        for l in range(self.max_level, insert_level, -1):
            changed = True
            while changed:
                changed = False
                curr_dist = self._dist(vec, self.vectors[curr_ep])
                for neighbor in self.graph[l].get(curr_ep, []):
                    n_dist = self._dist(vec, self.vectors[neighbor])
                    if n_dist < curr_dist:
                        curr_dist = n_dist
                        curr_ep = neighbor
                        changed = True

        # 2. Search and connect on levels from min(max_level, insert_level) down to 0
        for l in range(min(self.max_level, insert_level), -1, -1):
            candidates = self._search_layer(vec, curr_ep, self.ef_construction, l)
            curr_ep = candidates[0][1]
            max_m = self.M0 if l == 0 else self.M
            neighbors = [node for _, node in candidates[:max_m]]
            self.graph[l][u] = neighbors

            # Bidirectional edge addition
            for n in neighbors:
                self.graph[l][n].append(u)
                if len(self.graph[l][n]) > max_m:
                    # Prune to max_m nearest
                    n_vec = self.vectors[n]
                    pruned = sorted(self.graph[l][n], key=lambda x: self._dist(n_vec, self.vectors[x]))[:max_m]
                    self.graph[l][n] = pruned

        if insert_level > self.max_level:
            self.max_level = insert_level
            self.entry_point = u

    def search(self, q_vec: np.ndarray, k: int = 5, ef_search: int = 16) -> list[tuple[int, float]]:
        if self.entry_point is None:
            return []

        curr_ep = self.entry_point
        for l in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                curr_dist = self._dist(q_vec, self.vectors[curr_ep])
                for neighbor in self.graph[l].get(curr_ep, []):
                    n_dist = self._dist(q_vec, self.vectors[neighbor])
                    if n_dist < curr_dist:
                        curr_dist = n_dist
                        curr_ep = neighbor
                        changed = True

        results = self._search_layer(q_vec, curr_ep, max(ef_search, k), 0)
        return [(self.ids[node], dist) for dist, node in results[:k]]
```

---

### 2.9 Self-Contained Verification Fixtures

```python
def test_hnsw_minimal_verification():
    rng = np.random.default_rng(42)
    N, d, k = 200, 16, 5
    X = rng.standard_normal((N, d)).astype(np.float32)
    Q = rng.standard_normal((5, d)).astype(np.float32)

    hnsw = MinimalHNSW(dim=d, M=8, ef_c=32, seed=42)
    for i in range(N):
        hnsw.insert(ext_id=i, vec=X[i])

    for q_idx in range(5):
        # Approximate search
        results = hnsw.search(Q[q_idx], k=k, ef_search=32)
        retrieved_ids = [res[0] for res in results]

        # Ground truth
        dists = np.sum((X - Q[q_idx]) ** 2, axis=1)
        exact_top_k = set(np.argsort(dists)[:k])

        overlap = len(exact_top_k.intersection(set(retrieved_ids)))
        recall = overlap / k
        assert recall >= 0.8, f"HNSW verification recall too low ({recall:.2f}) for query {q_idx}"

    print("HNSW minimal verification: PASSED (Recall >= 80% on test fixture)")

if __name__ == "__main__":
    test_hnsw_minimal_verification()
```

---

## Algorithmic Verification Checklist

Before publishing code or answering technical interview coding questions:

- [ ] **Coordinate Precision:** Matrix differences accumulated in 64-bit float precision or Level-3 BLAS GEMM to prevent catastrophic cancellation.
- [ ] **Distance Non-Negativity:** Squared L2 distances clamped to `0.0` via `np.maximum(dists, 0.0)`.
- [ ] **Contiguous Storage:** Arrays validated as `C_CONTIGUOUS` before passing to BLAS or indexing routines.
- [ ] **Deterministic Ordering:** Lexicographical sorting with secondary key on integer ID: `np.lexsort((ids, dists))`.
- [ ] **Degree Bounds Enforced:** Every HNSW layer adheres strictly to $M$ (upper layers) and $M_0$ (ground layer).
- [ ] **Safe Top-Down Entry:** HNSW search initializes strictly from global entry point at `max_level`.
- [ ] **Memory Bounded:** Matrix dimensions verified before allocating $B \times N$ distance arrays; batched chunking applied when $N > 32,768$.
- [ ] **Clean Decoupling:** Algorithmic routines do not import heavy deep-learning frameworks or third-party vector search black-boxes.
