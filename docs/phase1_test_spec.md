# Phase 1 Test Specification: Distance Kernels & Deterministic Top-$k$

**Status:** Draft for Review (Phase 1 Preparation)  
**Author:** Agent-assisted  
**Reviewer:** Human Owner  

This document specifies the exact test matrix for Phase 1. It defines the verification cases, inputs, expected outputs, and specific bugs targeted by each test. 

> [!NOTE]
> Tests are specified here as contracts before code implementation. Do not write test code against nonexistent APIs until the interface design is reviewed and approved.

---

## 1. Test Matrix

| ID | Test Category | Target Invariant | Setup | Expected Result | Bug Caught |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **T1.1** | **Identical Vectors** | Distance self-identity | $q = [1.5, -2.0, 3.0]$, $x = [1.5, -2.0, 3.0]$ ($d=3$) | $d^2(q, x) = 0.0$ exactly | Non-zero diagonal distances caused by precision loss or indexing errors |
| **T1.2** | **Hand-Calculated Distances** | Numerical correctness | Database of 4 known vectors from learning guide ($d=2$), $q = [1.0, 1.0]$ | Exactly $[1.0, 34.0, 1.0, 1.0]$ | Arithmetic errors, missing coordinate terms, or sign errors in difference |
| **T1.3** | **Single vs Batched Queries** | Shape flexibility | Single query $q \in \mathbb{R}^{d}$ and batched query $Q \in \mathbb{R}^{1 \times d}$ against same database $X \in \mathbb{R}^{N \times d}$ | Identical distance values; output shapes $(N,)$ and $(1, N)$ respectively | Shape broadcasting bugs or silent flattening of batch dimensions |
| **T1.4** | **Dimension Mismatch** | Input validation | Query with $d=4$, database with $d=3$ | Raises `ValueError` with informative message | Silent truncation, out-of-bounds memory access, or invalid dot products |
| **T1.5** | **Empty Inputs** | Boundary robustness | 1. $N = 0$, $Q \ge 1$<br>2. $Q = 0$, $N \ge 1$ | 1. Distance matrix shape $(Q, 0)$<br>2. Distance matrix shape $(0, N)$ | Crash on zero-length array strides or unhandled empty slices |
| **T1.6** | **Non-Finite Values** | Input sanitization | Vector containing `np.nan` or `np.inf` | Raises `ValueError` on validation | Silent NaN propagation poisoning rankings or causing undefined comparisons |
| **T1.7** | **Non-Float Dtype Handling** | Type safety | Vector passed as `int32` or `float64` | Converted cleanly or rejected per dtype policy | Unexpected dtype promotion or integer division truncation |
| **T1.8** | **Deterministic Equal Distances** | Tie-breaking rule | Two identical vectors with distinct IDs `[200, 100]` equidistant to $q$ | Sorted order must place ID `100` before ID `200` | Flaky results and unstable ordering when distances are equal |
| **T1.9** | **Stable-ID Tie-Breaking** | ID independence | Non-contiguous, non-sequential IDs (e.g. `[9999, 42, 500]`) with equidistant vectors | Top result must be ID `42`, followed by `500`, then `9999` | Code accidentally sorting by row index (0, 1, 2) instead of durable ID |
| **T1.10** | **Top-$k$ with $k=1$** | Minimal selection | $N = 1000$ synthetic random vectors, query $q$ | Returns 1D or $(1,)$ array containing absolute closest neighbor | Off-by-one errors in index slicing (e.g. returning 0 items or 2 items) |
| **T1.11** | **Top-$k$ with $k=N$** | Full set selection | $N = 50$, $k = 50$ | Returns all 50 items sorted strictly by $(d^2, \text{ID})$ | Partitioning crashing when $k = N$ or missing the boundary element |
| **T1.12** | **Top-$k$ with $k > N$** | Oversized query | $N = 5$, request $k = 10$ | Returns all 5 available items in sorted order without padding invalid IDs | `IndexError`, crash, or returning dummy placeholder IDs |
| **T1.13** | **Boundary Tie at Rank $k$** | Partition ambiguity | Distances `[1.0, 2.0, 2.0, 2.0, 5.0]` with IDs `[1, 40, 20, 30, 5]`, request $k = 3$ | Top 3 must be: `(1.0, 1)`, `(2.0, 20)`, `(2.0, 30)` | Naive `np.argpartition` picking arbitrary subset of tied elements at the $k$-th position |
| **T1.14** | **Fast vs Reference Agreement** | Optimization verification | 100 random vectors, evaluate both reference kernel and fast BLAS kernel | Disagreement rate measured; max absolute error within tolerance ($\le 10^{-5}$) | Silent floating-point cancellation errors or un-clamped negative distances |

---

## 2. Detailed Test Specifications

### T1.2: Hand-Calculated Squared-$L_2$ Distance
- **Target Component:** `pairwise_squared_l2_ref`
- **Setup:**
  ```python
  X = np.array([
      [1.0, 2.0],
      [4.0, 6.0],
      [2.0, 1.0],
      [1.0, 2.0]
  ], dtype=np.float32)
  q = np.array([[1.0, 1.0]], dtype=np.float32)
  ```
- **Expected Result:**
  ```python
  expected = np.array([[1.0, 34.0, 1.0, 1.0]], dtype=np.float32)
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
  ```
- **Bug Caught:** Incorrect coordinate differences or accumulator indexing.

---

### T1.8 & T1.9: Stable-ID Tie-Breaking
- **Target Component:** `select_top_k`
- **Setup:**
  ```python
  distances = np.array([10.0, 5.0, 5.0, 8.0, 5.0], dtype=np.float32)
  ids = np.array([1001, 5003, 2002, 3004, 1005], dtype=np.int64)
  k = 3
  ```
- **Analysis:**
  - Candidates with distance 5.0 have IDs: `5003`, `2002`, `1005`.
  - Under ascending `(distance, ID)` composite ordering:
    1. `(5.0, 1005)`
    2. `(5.0, 2002)`
    3. `(5.0, 5003)`
- **Expected Result:**
  - Selected IDs: `[1005, 2002, 5003]`
  - Selected Distances: `[5.0, 5.0, 5.0]`
- **Bug Caught:** Falling back to row index ordering (`[5003, 2002, 1005]`) or non-deterministic hash ordering.

---

### T1.13: Exact Tie at the $k$-th Rank Boundary
- **Target Component:** `select_top_k`
- **Setup:**
  ```python
  distances = np.array([1.0, 2.0, 2.0, 2.0, 5.0], dtype=np.float32)
  ids = np.array([1, 40, 20, 30, 5], dtype=np.int64)
  k = 3
  ```
- **Analysis:**
  - Distances are: `1.0` (ID 1), `2.0` (IDs 40, 20, 30), `5.0` (ID 5).
  - Rank 1 is `(1.0, 1)`.
  - Ranks 2, 3, and 4 are tied on distance `2.0`.
  - For $k = 3$, only two of the three tied items can be included.
  - Sorting the tied candidates by ID: `20 < 30 < 40`.
  - Thus, ID 20 and ID 30 are selected; ID 40 is excluded.
- **Expected Result:**
  - Returned IDs: `[1, 20, 30]`
  - Returned Distances: `[1.0, 2.0, 2.0]`
- **Bug Caught:** `np.argpartition` leaving arbitrary elements on the left side of the partition.

---

## 3. Decisions Requiring Owner Approval

The following behaviors are open design choices for the project owner to confirm before implementing Phase 1 code:

1. **Policy for $k > N$:**
   - *Option A (Recommended):* Return all $N$ available items sorted by `(distance, ID)`. This aligns with standard retrieval libraries (FAISS, scikit-learn).
   - *Option B:* Raise a `ValueError("k cannot exceed number of vectors in database")`.
2. **Policy for $k \le 0$:**
   - *Option A (Recommended):* Raise `ValueError("k must be a positive integer >= 1")`.
   - *Option B:* Allow $k = 0$ returning empty arrays, and raise `ValueError` only for $k < 0$.
3. **Dtype Flexibility for Reference Kernel:**
   - *Option A (Recommended):* Enforce strict `float32` inputs; reject other dtypes to catch unintended double-precision memory overhead early.
   - *Option B:* Automatically cast `float64` or integer arrays to `float32` with a warning.
4. **Return Structure for Top-$k$:**
   - *Option A (Recommended):* Named tuple or dataclass: `SearchResult(distances=..., ids=...)`.
   - *Option B:* Plain 2-tuple: `(distances, ids)`.
