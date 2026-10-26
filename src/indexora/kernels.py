"""Distance calculation kernels for Indexora.

This module provides pairwise distance computation kernels between query vectors
and database vectors. In Phase 1, the primary kernel is the exact reference
implementation (`pairwise_squared_l2_ref`), which acts as the numerical oracle
for all subsequent approximations and hardware optimizations.
"""

from __future__ import annotations

import numpy as np


def pairwise_squared_l2_ref(
    queries: np.ndarray,
    database: np.ndarray,
) -> np.ndarray:
    """Compute pairwise squared Euclidean (L2) distances between queries and database vectors.

    This is the reference implementation and correctness oracle for Indexora.
    It follows direct coordinate differences with float64 accumulation before
    storing results in float32 precision.

    Parameters
    ----------
    queries : np.ndarray
        Query vector(s) of shape `(d,)` for a single query, or `(Q, d)` for
        batched queries, with strict `dtype=np.float32`.
    database : np.ndarray
        Database vectors of shape `(N, d)` with strict `dtype=np.float32`.

    Returns
    -------
    np.ndarray
        Pairwise squared-L2 distances with strict `dtype=np.float32`:
        - Shape `(N,)` if `queries` is 1D of shape `(d,)`.
        - Shape `(Q, N)` if `queries` is 2D of shape `(Q, d)`.

    Raises
    ------
    TypeError or ValueError
        If `queries` or `database` are not `float32` arrays (no silent casting).
    ValueError
        If array dimensions are mismatched (e.g. `d_q != d_db`), if `database`
        is not a 2D array, if `queries` is not 1D or 2D, or if any input contains
        non-finite values (`np.nan` or `np.inf`).

    Notes
    -----
    - This function computes geometric distances only; it does not process
      entity IDs, sort results, or return `SearchResult` containers (per ADR-0002).
    - If `N = 0` (empty database), returns an empty array of shape `(Q, 0)` (or
      `(0,)` if `queries` is 1D) with `dtype=np.float32`.
    - If `Q = 0` (empty queries), returns an empty array of shape `(0, N)` with
      `dtype=np.float32`.
    """
    # 1. Validate strict float32 dtypes
    if queries.dtype != np.float32 or database.dtype != np.float32:
        raise TypeError(
            f"Expected float32 inputs, got queries.dtype={queries.dtype} "
            f"and database.dtype={database.dtype}. Silent casting is not permitted."
        )

    # 2. Validate input finiteness
    if not np.all(np.isfinite(queries)) or not np.all(np.isfinite(database)):
        raise ValueError("Inputs must contain only finite numbers (no NaN or Inf).")

    # 3. Validate input shapes and dimensionalities
    if database.ndim != 2:
        raise ValueError(
            f"database must be a 2D array of shape (N, d), got ndim={database.ndim} with shape {database.shape}"
        )

    is_single_query = queries.ndim == 1
    if is_single_query:
        q_2d = queries[np.newaxis, :]
    elif queries.ndim == 2:
        q_2d = queries
    else:
        raise ValueError(
            f"queries must be a 1D array of shape (d,) or 2D array of shape (Q, d), got ndim={queries.ndim}"
        )

    # 4. Check matching feature dimensions
    if q_2d.shape[1] != database.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query dimension {q_2d.shape[1]} does not match database dimension {database.shape[1]}"
        )

    # 5. Handle empty inputs
    num_queries = q_2d.shape[0]
    num_database = database.shape[0]
    if num_queries == 0 or num_database == 0:
        if is_single_query:
            return np.empty((0,), dtype=np.float32)
        return np.empty((num_queries, num_database), dtype=np.float32)

    # 6. Direct coordinate differences and careful accumulation in float64
    diff = q_2d[:, np.newaxis, :].astype(np.float64) - database[np.newaxis, :, :].astype(np.float64)
    dists_sq = np.sum(diff * diff, axis=-1, dtype=np.float64)
    out = dists_sq.astype(np.float32)

    # 7. Return with correct output shape
    if is_single_query:
        return out[0]
    return out
