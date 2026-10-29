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
    max_buffer_bytes: int = 32 * 1024 * 1024,
) -> np.ndarray:
    """Compute pairwise squared Euclidean (L2) distances between queries and database vectors.

    This is the reference implementation and correctness oracle for Indexora.
    It follows direct coordinate differences with float64 accumulation before
    storing results in float32 precision.

    Temporary memory allocations are strictly bounded by chunking query and/or
    database blocks to prevent allocating large (Q, N, d) difference tensors.

    Parameters
    ----------
    queries : np.ndarray
        Query vector(s) of shape `(d,)` for a single query, or `(Q, d)` for
        batched queries, with strict `dtype=np.float32`.
    database : np.ndarray
        Database vectors of shape `(N, d)` with strict `dtype=np.float32`.
    max_buffer_bytes : int, default=33554432 (32 MiB)
        Memory budget in bytes governing the primary intermediate float64
        difference tensor `(Q_b, N_b, d)`. Must be a positive integer.
        - Covers: the `(Q_b, N_b, d)` float64 difference block (`8 * Q_b * N_b * d` bytes).
        - Excludes: the final `(Q, N)` float32 output array (`4 * Q * N` bytes),
          and the float64 slice casts `q_chunk` and `db_chunk` (`8 * (Q_b + N_b) * d` bytes).
        - Minimum floor: If `max_buffer_bytes < 8 * d` (smaller than a single pair's
          coordinate difference vector), the allocation floors to `Q_b = 1, N_b = 1`
          (`8 * d` bytes).

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
        If `max_buffer_bytes` is not an integer or is <= 0.
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
    - Coordinate differences and float64 accumulations within each chunk are bit-for-bit
      identical to unchunked reference computation because summation order over feature
      dimension `d` is strictly preserved.
    """
    # 1. Validate max_buffer_bytes
    if not isinstance(max_buffer_bytes, (int, np.integer)) or isinstance(max_buffer_bytes, bool):
        raise TypeError(
            f"max_buffer_bytes must be an integer, got {type(max_buffer_bytes).__name__}"
        )
    if max_buffer_bytes <= 0:
        raise ValueError(
            f"max_buffer_bytes must be positive, got {max_buffer_bytes}"
        )

    # 2. Validate strict float32 dtypes
    if queries.dtype != np.float32 or database.dtype != np.float32:
        raise TypeError(
            f"Expected float32 inputs, got queries.dtype={queries.dtype} "
            f"and database.dtype={database.dtype}. Silent casting is not permitted."
        )

    # 3. Validate input finiteness
    if not np.all(np.isfinite(queries)) or not np.all(np.isfinite(database)):
        raise ValueError("Inputs must contain only finite numbers (no NaN or Inf).")

    # 4. Validate input shapes and dimensionalities
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

    # 5. Check matching feature dimensions
    if q_2d.shape[1] != database.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query dimension {q_2d.shape[1]} does not match database dimension {database.shape[1]}"
        )

    # 6. Handle empty inputs
    num_queries = q_2d.shape[0]
    num_database = database.shape[0]
    d = q_2d.shape[1]

    if num_queries == 0 or num_database == 0:
        if is_single_query:
            return np.empty((0,), dtype=np.float32)
        return np.empty((num_queries, num_database), dtype=np.float32)

    # 7. Compute bounded chunk sizes
    # Each float64 difference in (Q_b, N_b, d) takes 8 bytes
    bytes_per_pair = 8 * d
    max_pairs = max(1, max_buffer_bytes // bytes_per_pair)

    if num_database <= max_pairs:
        N_b = num_database
        Q_b = min(num_queries, max(1, max_pairs // num_database))
    else:
        Q_b = 1
        N_b = min(num_database, max_pairs)

    out = np.empty((num_queries, num_database), dtype=np.float32)

    # 8. Direct coordinate differences and careful accumulation in float64
    for q_start in range(0, num_queries, Q_b):
        q_end = min(q_start + Q_b, num_queries)
        q_chunk = q_2d[q_start:q_end]

        for db_start in range(0, num_database, N_b):
            db_end = min(db_start + N_b, num_database)
            db_chunk = database[db_start:db_end]

            # Primary bounded temporary: (q_len, db_len, d) in float64
            diff = q_chunk[:, np.newaxis, :].astype(np.float64) - db_chunk[np.newaxis, :, :].astype(np.float64)
            # In-place squaring avoids allocating a secondary (q_len, db_len, d) float64 tensor
            diff *= diff
            # Float64 summation accumulated directly into the output slice in float32
            np.sum(diff, axis=-1, dtype=np.float64, out=out[q_start:q_end, db_start:db_end])

    # 9. Return with correct output shape
    if is_single_query:
        return out[0]
    return out


def pairwise_squared_l2_blas(
    queries: np.ndarray,
    database: np.ndarray,
    max_buffer_bytes: int = 32 * 1024 * 1024,
) -> np.ndarray:
    """Compute pairwise squared Euclidean (L2) distances using BLAS matrix multiplication.

    This function implements the algebraic norm expansion:
        ||x - y||² = ||x||² + ||y||² - 2 (x · y)

    Where the inner product (x · y) is evaluated via high-performance BLAS matrix
    multiplication (`np.matmul` / `@`), which leverages multi-core SIMD vectorization.

    Numerical Trade-offs vs. Reference Kernel:
    -----------------------------------------
    1. Performance: The BLAS-backed gemm kernel is typically 10x-20x faster than
       the coordinate-difference reference kernel (`pairwise_squared_l2_ref`).
    2. Numerical Precision: The reference kernel computes (x_i - y_i) directly
       with float64 summation, preserving catastrophic cancellation safety.
       The norm expansion formula computes ||x||² + ||y||² - 2(x · y), which can
       suffer from loss of significance when ||x - y|| is very small relative to
       ||x|| and ||y||. Floating-point round-off can also occasionally yield small
       negative numbers for identical or near-identical vectors (e.g. -1e-7),
       which this implementation explicitly clips to 0.0.
    3. Intermediate Results: Floating-point distances may differ by small numerical
       tolerances (typically < 1e-4 absolute, < 1e-6 relative) from the reference
       kernel. They are not guaranteed to be bit-for-bit identical to the reference
       oracle.

    Parameters
    ----------
    queries : np.ndarray
        Query vector(s) of shape `(d,)` for a single query, or `(Q, d)` for
        batched queries, with strict `dtype=np.float32`.
    database : np.ndarray
        Database vectors of shape `(N, d)` with strict `dtype=np.float32`.
    max_buffer_bytes : int, default=33554432 (32 MiB)
        Memory budget in bytes governing intermediate query-database block
        multiplications. Must be a positive integer.

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
        If `max_buffer_bytes` is not an integer or is <= 0.
    ValueError
        If array dimensions are mismatched, if `database` is not 2D, if `queries`
        is not 1D or 2D, or if any input contains non-finite values (`np.nan` or `np.inf`).
    """
    # 1. Validate max_buffer_bytes
    if not isinstance(max_buffer_bytes, (int, np.integer)) or isinstance(max_buffer_bytes, bool):
        raise TypeError(
            f"max_buffer_bytes must be an integer, got {type(max_buffer_bytes).__name__}"
        )
    if max_buffer_bytes <= 0:
        raise ValueError(
            f"max_buffer_bytes must be positive, got {max_buffer_bytes}"
        )

    # 2. Validate strict float32 dtypes
    if queries.dtype != np.float32 or database.dtype != np.float32:
        raise TypeError(
            f"Expected float32 inputs, got queries.dtype={queries.dtype} "
            f"and database.dtype={database.dtype}. Silent casting is not permitted."
        )

    # 3. Validate input finiteness
    if not np.all(np.isfinite(queries)) or not np.all(np.isfinite(database)):
        raise ValueError("Inputs must contain only finite numbers (no NaN or Inf).")

    # 4. Validate input shapes and dimensionalities
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

    # 5. Check matching feature dimensions
    if q_2d.shape[1] != database.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query dimension {q_2d.shape[1]} does not match database dimension {database.shape[1]}"
        )

    # 6. Handle empty inputs
    num_queries = q_2d.shape[0]
    num_database = database.shape[0]
    d = q_2d.shape[1]

    if num_queries == 0 or num_database == 0:
        if is_single_query:
            return np.empty((0,), dtype=np.float32)
        return np.empty((num_queries, num_database), dtype=np.float32)

    # 7. Compute precalculated database squared norms: shape (num_database,)
    # Accumulation in float32 for speed, consistent with standard BLAS squared-L2 routines
    db_sq = np.sum(database * database, axis=1).astype(np.float32)

    # 8. Compute bounded chunk sizes
    # Each float32 pairwise distance in (Q_b, N_b) takes 4 bytes
    bytes_per_pair = 4
    max_pairs = max(1, max_buffer_bytes // bytes_per_pair)

    if num_database <= max_pairs:
        N_b = num_database
        Q_b = min(num_queries, max(1, max_pairs // num_database))
    else:
        Q_b = 1
        N_b = min(num_database, max_pairs)

    out = np.empty((num_queries, num_database), dtype=np.float32)

    # 9. Blocked evaluation via GEMM matrix multiplication
    for q_start in range(0, num_queries, Q_b):
        q_end = min(q_start + Q_b, num_queries)
        q_chunk = q_2d[q_start:q_end]
        q_sq_chunk = np.sum(q_chunk * q_chunk, axis=1, keepdims=True).astype(np.float32)

        for db_start in range(0, num_database, N_b):
            db_end = min(db_start + N_b, num_database)
            db_chunk = database[db_start:db_end]
            db_sq_chunk = db_sq[np.newaxis, db_start:db_end]

            # BLAS matrix product: (q_len, d) @ (d, db_len) -> (q_len, db_len)
            dot = q_chunk @ db_chunk.T

            # Norm expansion: ||q||^2 + ||db||^2 - 2 * (q @ db.T)
            chunk_dist = q_sq_chunk + db_sq_chunk
            chunk_dist -= 2.0 * dot

            # Numerical safety: clip negative floating-point roundoff to 0.0
            np.maximum(chunk_dist, 0.0, out=out[q_start:q_end, db_start:db_end])

    # 10. Return with correct output shape
    if is_single_query:
        return out[0]
    return out
