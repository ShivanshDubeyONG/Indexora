"""Exact Flat vector search and ground-truth generation for Indexora.

This module provides exhaustive exact nearest-neighbor search over vector databases
using Indexora's reference distance kernel (`pairwise_squared_l2_ref`) and
deterministic top-k selection (`select_top_k`).

In Indexora, exact search serves as the ground-truth oracle for all subsequent
approximate nearest neighbor (ANN) indexes, benchmarks, and recall evaluations.
"""

from __future__ import annotations

import numpy as np

from indexora.kernels import pairwise_squared_l2_blas, pairwise_squared_l2_ref
from indexora.topk import SearchResult, select_top_k


def _validate_ids(ids: np.ndarray, expected_n: int) -> np.ndarray:
    """Validate that IDs array is 1D int64, matches database length, and has unique values."""
    if not isinstance(ids, np.ndarray):
        raise TypeError(
            f"ids must be a numpy ndarray, got {type(ids).__name__}"
        )
    if ids.dtype != np.int64:
        raise TypeError(
            f"ids must have dtype int64, got {ids.dtype}. Silent casting is not permitted."
        )
    if ids.ndim != 1:
        raise ValueError(
            f"ids must be a 1D array of shape (N,), got ndim={ids.ndim} with shape {ids.shape}"
        )
    if ids.shape[0] != expected_n:
        raise ValueError(
            f"Length mismatch: database has {expected_n} vectors but ids has {ids.shape[0]} elements"
        )
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Database IDs must be strictly unique.")
    return ids


def _exact_search_single_incremental(
    query: np.ndarray,
    database: np.ndarray,
    ids: np.ndarray,
    k: int,
    max_db_chunk: int,
) -> SearchResult:
    """Perform exact search for a single query by consuming database blocks incrementally.

    Maintains running top-k results per query to avoid allocating or retaining
    large distance matrices in memory.
    """
    running_dists: np.ndarray | None = None
    running_ids: np.ndarray | None = None
    num_database = database.shape[0]

    for start in range(0, num_database, max_db_chunk):
        end = min(start + max_db_chunk, num_database)
        db_chunk = database[start:end]
        id_chunk = ids[start:end]
        chunk_dists = pairwise_squared_l2_ref(query, db_chunk)
        chunk_topk = select_top_k(chunk_dists, id_chunk, k=k)

        if running_dists is None:
            running_dists = chunk_topk.distances
            running_ids = chunk_topk.ids
        else:
            merged_dists = np.concatenate([running_dists, chunk_topk.distances])
            merged_ids = np.concatenate([running_ids, chunk_topk.ids])
            merged_topk = select_top_k(merged_dists, merged_ids, k=k)
            running_dists = merged_topk.distances
            running_ids = merged_topk.ids

    return SearchResult(
        distances=running_dists if running_dists is not None else np.empty((0,), dtype=np.float32),
        ids=running_ids if running_ids is not None else np.empty((0,), dtype=np.int64),
    )


def _exact_search_single_incremental_blas(
    query: np.ndarray,
    database: np.ndarray,
    ids: np.ndarray,
    k: int,
    max_db_chunk: int,
) -> SearchResult:
    """Perform BLAS exact search for a single query by consuming database blocks incrementally.

    Maintains running top-k results per query to avoid allocating or retaining
    large distance matrices in memory.
    """
    running_dists: np.ndarray | None = None
    running_ids: np.ndarray | None = None
    num_database = database.shape[0]

    for start in range(0, num_database, max_db_chunk):
        end = min(start + max_db_chunk, num_database)
        db_chunk = database[start:end]
        id_chunk = ids[start:end]
        chunk_dists = pairwise_squared_l2_blas(query, db_chunk)
        chunk_topk = select_top_k(chunk_dists, id_chunk, k=k)

        if running_dists is None:
            running_dists = chunk_topk.distances
            running_ids = chunk_topk.ids
        else:
            merged_dists = np.concatenate([running_dists, chunk_topk.distances])
            merged_ids = np.concatenate([running_ids, chunk_topk.ids])
            merged_topk = select_top_k(merged_dists, merged_ids, k=k)
            running_dists = merged_topk.distances
            running_ids = merged_topk.ids

    return SearchResult(
        distances=running_dists if running_dists is not None else np.empty((0,), dtype=np.float32),
        ids=running_ids if running_ids is not None else np.empty((0,), dtype=np.int64),
    )


def exact_search(
    queries: np.ndarray,
    database: np.ndarray,
    ids: np.ndarray | None = None,
    k: int = 10,
    max_db_chunk: int = 32768,
    max_query_chunk: int = 1024,
) -> SearchResult:
    """Perform exhaustive exact nearest-neighbor search using reference semantics.

    Computes pairwise squared-L2 distances against database vectors using
    `pairwise_squared_l2_ref`, and deterministically selects the top-k nearest
    neighbors using `select_top_k` with ascending `(distance, ID)` tie-breaking.

    For large workloads, search evaluates queries and database blocks
    in bounded chunks (`max_query_chunk` and `max_db_chunk`) without allocating
    an unbounded full `(Q, N)` distance matrix in memory.

    Parameters
    ----------
    queries : np.ndarray
        Query vector(s) of shape `(d,)` for a single query, or `(Q, d)` for
        batched queries, with strict `dtype=np.float32`.
    database : np.ndarray
        Database vectors of shape `(N, d)` with strict `dtype=np.float32`.
    ids : np.ndarray | None, default=None
        Durable identifiers of shape `(N,)` with strict `dtype=np.int64`.
        Must be strictly unique among live records. If None, defaults to
        `np.arange(N, dtype=np.int64)`.
    k : int, default=10
        Number of nearest neighbors to retrieve. Must be a positive integer (k >= 1).
    max_db_chunk : int, default=32768
        Maximum number of database vectors evaluated per block during distance scoring.
    max_query_chunk : int, default=1024
        Maximum number of query vectors evaluated per block during distance scoring.

    Returns
    -------
    SearchResult
        Container with `.distances` (float32) and `.ids` (int64):
        - For a single query of shape `(d,)`: 1D arrays of length `min(k, N)`.
        - For batched queries of shape `(Q, d)`: 2D arrays of shape `(Q, min(k, N))`.

    Raises
    ------
    ValueError
        If `k <= 0`, `max_db_chunk <= 0`, or `max_query_chunk <= 0`.
        If dimensions mismatch, if inputs contain non-finite numbers, or if IDs are not unique.
    TypeError
        If `queries` or `database` are not `float32` arrays, or if `ids` is not `int64`.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    if not isinstance(max_db_chunk, (int, np.integer)) or max_db_chunk <= 0:
        raise ValueError(f"max_db_chunk must be a positive integer, got {max_db_chunk}")
    if not isinstance(max_query_chunk, (int, np.integer)) or max_query_chunk <= 0:
        raise ValueError(f"max_query_chunk must be a positive integer, got {max_query_chunk}")

    if not isinstance(queries, np.ndarray):
        raise TypeError(
            f"queries must be a numpy ndarray, got {type(queries).__name__}"
        )
    if not isinstance(database, np.ndarray):
        raise TypeError(
            f"database must be a numpy ndarray, got {type(database).__name__}"
        )

    num_database = database.shape[0] if database.ndim >= 1 else 0

    if ids is None:
        ids_arr = np.arange(num_database, dtype=np.int64)
    else:
        ids_arr = _validate_ids(ids, expected_n=num_database)

    is_single_query = queries.ndim == 1
    if is_single_query:
        if num_database <= max_db_chunk:
            distances = pairwise_squared_l2_ref(queries, database)
            res = select_top_k(distances, ids_arr, k=k)
            return SearchResult(
                distances=res.distances,
                ids=res.ids,
                mask=res.mask,
                distance_computations=num_database,
                centroid_distance_computations=0,
                candidate_distance_computations=num_database,
            )
        res = _exact_search_single_incremental(
            queries, database, ids_arr, k=k, max_db_chunk=max_db_chunk
        )
        return SearchResult(
            distances=res.distances,
            ids=res.ids,
            mask=res.mask,
            distance_computations=num_database,
            centroid_distance_computations=0,
            candidate_distance_computations=num_database,
        )

    # Batched queries: shape (Q, d)
    if queries.ndim != 2:
        raise ValueError(
            f"queries must be a 1D array of shape (d,) or 2D array of shape (Q, d), got ndim={queries.ndim}"
        )

    num_queries = queries.shape[0]
    k_eff = min(k, num_database)

    if num_queries == 0:
        return SearchResult(
            distances=np.empty((0, k_eff), dtype=np.float32),
            ids=np.empty((0, k_eff), dtype=np.int64),
            distance_computations=0,
            centroid_distance_computations=0,
            candidate_distance_computations=0,
        )

    if num_database == 0:
        return SearchResult(
            distances=np.empty((num_queries, 0), dtype=np.float32),
            ids=np.empty((num_queries, 0), dtype=np.int64),
            distance_computations=0,
            centroid_distance_computations=0,
            candidate_distance_computations=0,
        )

    batch_distances = np.empty((num_queries, k_eff), dtype=np.float32)
    batch_ids = np.empty((num_queries, k_eff), dtype=np.int64)

    for q_start in range(0, num_queries, max_query_chunk):
        q_end = min(q_start + max_query_chunk, num_queries)
        q_chunk = queries[q_start:q_end]
        q_count = q_end - q_start

        if num_database <= max_db_chunk:
            chunk_distances = pairwise_squared_l2_ref(q_chunk, database)
            for i in range(q_count):
                res = select_top_k(chunk_distances[i], ids_arr, k=k)
                batch_distances[q_start + i] = res.distances
                batch_ids[q_start + i] = res.ids
        else:
            for i in range(q_count):
                res = _exact_search_single_incremental(
                    q_chunk[i], database, ids_arr, k=k, max_db_chunk=max_db_chunk
                )
                batch_distances[q_start + i] = res.distances
                batch_ids[q_start + i] = res.ids

    batch_mask = np.ones_like(batch_ids, dtype=bool)
    total_dc = num_queries * num_database

    return SearchResult(
        distances=batch_distances,
        ids=batch_ids,
        mask=batch_mask,
        distance_computations=total_dc,
        centroid_distance_computations=0,
        candidate_distance_computations=total_dc,
    )


def exact_search_blas(
    queries: np.ndarray,
    database: np.ndarray,
    ids: np.ndarray | None = None,
    k: int = 10,
    max_db_chunk: int = 32768,
    max_query_chunk: int = 1024,
) -> SearchResult:
    """Perform exhaustive exact nearest-neighbor search accelerated via BLAS matrix multiplication.

    Computes pairwise squared-L2 distances against database vectors using
    `pairwise_squared_l2_blas`, and deterministically selects the top-k nearest
    neighbors using `select_top_k` with ascending `(distance, ID)` tie-breaking.

    Memory allocations are strictly bounded across both queries and database
    records (`max_query_chunk` and `max_db_chunk`) to prevent allocating large
    intermediate distance arrays.

    Parameters
    ----------
    queries : np.ndarray
        Query vector(s) of shape `(d,)` for a single query, or `(Q, d)` for
        batched queries, with strict `dtype=np.float32`.
    database : np.ndarray
        Database vectors of shape `(N, d)` with strict `dtype=np.float32`.
    ids : np.ndarray | None, default=None
        Durable identifiers of shape `(N,)` with strict `dtype=np.int64`.
        Must be strictly unique among live records. If None, defaults to
        `np.arange(N, dtype=np.int64)`.
    k : int, default=10
        Number of nearest neighbors to retrieve. Must be a positive integer (k >= 1).
    max_db_chunk : int, default=32768
        Maximum number of database vectors evaluated per block during distance scoring.
    max_query_chunk : int, default=1024
        Maximum number of query vectors evaluated per block during distance scoring.

    Returns
    -------
    SearchResult
        Container with `.distances` (float32) and `.ids` (int64):
        - For a single query of shape `(d,)`: 1D arrays of length `min(k, N)`.
        - For batched queries of shape `(Q, d)`: 2D arrays of shape `(Q, min(k, N))`.

    Raises
    ------
    ValueError
        If `k <= 0`, `max_db_chunk <= 0`, or `max_query_chunk <= 0`.
        If dimensions mismatch, if inputs contain non-finite numbers, or if IDs are not unique.
    TypeError
        If `queries` or `database` are not `float32` arrays, or if `ids` is not `int64`.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    if not isinstance(max_db_chunk, (int, np.integer)) or max_db_chunk <= 0:
        raise ValueError(f"max_db_chunk must be a positive integer, got {max_db_chunk}")
    if not isinstance(max_query_chunk, (int, np.integer)) or max_query_chunk <= 0:
        raise ValueError(f"max_query_chunk must be a positive integer, got {max_query_chunk}")

    if not isinstance(queries, np.ndarray):
        raise TypeError(
            f"queries must be a numpy ndarray, got {type(queries).__name__}"
        )
    if not isinstance(database, np.ndarray):
        raise TypeError(
            f"database must be a numpy ndarray, got {type(database).__name__}"
        )

    num_database = database.shape[0] if database.ndim >= 1 else 0

    if ids is None:
        ids_arr = np.arange(num_database, dtype=np.int64)
    else:
        ids_arr = _validate_ids(ids, expected_n=num_database)

    is_single_query = queries.ndim == 1
    if is_single_query:
        if num_database <= max_db_chunk:
            distances = pairwise_squared_l2_blas(queries, database)
            res = select_top_k(distances, ids_arr, k=k)
            return SearchResult(
                distances=res.distances,
                ids=res.ids,
                mask=res.mask,
                distance_computations=num_database,
                centroid_distance_computations=0,
                candidate_distance_computations=num_database,
            )
        res = _exact_search_single_incremental_blas(
            queries, database, ids_arr, k=k, max_db_chunk=max_db_chunk
        )
        return SearchResult(
            distances=res.distances,
            ids=res.ids,
            mask=res.mask,
            distance_computations=num_database,
            centroid_distance_computations=0,
            candidate_distance_computations=num_database,
        )

    # Batched queries: shape (Q, d)
    if queries.ndim != 2:
        raise ValueError(
            f"queries must be a 1D array of shape (d,) or 2D array of shape (Q, d), got ndim={queries.ndim}"
        )

    num_queries = queries.shape[0]
    k_eff = min(k, num_database)

    if num_queries == 0:
        return SearchResult(
            distances=np.empty((0, k_eff), dtype=np.float32),
            ids=np.empty((0, k_eff), dtype=np.int64),
            distance_computations=0,
            centroid_distance_computations=0,
            candidate_distance_computations=0,
        )

    if num_database == 0:
        return SearchResult(
            distances=np.empty((num_queries, 0), dtype=np.float32),
            ids=np.empty((num_queries, 0), dtype=np.int64),
            distance_computations=0,
            centroid_distance_computations=0,
            candidate_distance_computations=0,
        )

    batch_distances = np.empty((num_queries, k_eff), dtype=np.float32)
    batch_ids = np.empty((num_queries, k_eff), dtype=np.int64)

    for q_start in range(0, num_queries, max_query_chunk):
        q_end = min(q_start + max_query_chunk, num_queries)
        q_chunk = queries[q_start:q_end]
        q_count = q_end - q_start

        if num_database <= max_db_chunk:
            chunk_distances = pairwise_squared_l2_blas(q_chunk, database)
            for i in range(q_count):
                res = select_top_k(chunk_distances[i], ids_arr, k=k)
                batch_distances[q_start + i] = res.distances
                batch_ids[q_start + i] = res.ids
        else:
            for i in range(q_count):
                res = _exact_search_single_incremental_blas(
                    q_chunk[i], database, ids_arr, k=k, max_db_chunk=max_db_chunk
                )
                batch_distances[q_start + i] = res.distances
                batch_ids[q_start + i] = res.ids

    batch_mask = np.ones_like(batch_ids, dtype=bool)
    total_dc = num_queries * num_database

    return SearchResult(
        distances=batch_distances,
        ids=batch_ids,
        mask=batch_mask,
        distance_computations=total_dc,
        centroid_distance_computations=0,
        candidate_distance_computations=total_dc,
    )


def generate_ground_truth(
    queries: np.ndarray,
    database: np.ndarray,
    ids: np.ndarray | None = None,
    k: int = 10,
    max_db_chunk: int = 32768,
) -> SearchResult:
    """Generate deterministic ground-truth nearest neighbors using exact reference search.

    This function produces the ground-truth baseline against which all approximate
    nearest neighbor (ANN) indexes (such as IVF-Flat, PQ, and HNSW) are evaluated
    for Recall@k. It strictly executes the reference distance kernel and deterministic
    top-k selection with ascending (distance, ID) tie-breaking.

    Parameters
    ----------
    queries : np.ndarray
        Query vector(s) of shape `(d,)` or `(Q, d)` with strict `dtype=np.float32`.
    database : np.ndarray
        Database vectors of shape `(N, d)` with strict `dtype=np.float32`.
    ids : np.ndarray | None, default=None
        Durable identifiers of shape `(N,)` with strict `dtype=np.int64`.
        If None, defaults to `np.arange(N, dtype=np.int64)`.
    k : int, default=10
        Number of nearest neighbors to retrieve. Must be a positive integer (k >= 1).
    max_db_chunk : int, default=32768
        Maximum number of database vectors evaluated per block during distance scoring.

    Returns
    -------
    SearchResult
        Ground-truth search results containing `.distances` (float32) and `.ids` (int64).
    """
    return exact_search(queries=queries, database=database, ids=ids, k=k, max_db_chunk=max_db_chunk)


class ExactFlatIndex:
    """Exact Flat vector index performing exhaustive scan with reference semantics.

    Parameters
    ----------
    database : np.ndarray
        Database vectors of shape `(N, d)` with strict `dtype=np.float32`.
    ids : np.ndarray | None, default=None
        Durable identifiers of shape `(N,)` with strict `dtype=np.int64`.
        Must be strictly unique among live records. If None, defaults to
        `np.arange(N, dtype=np.int64)`.

    Notes
    -----
    - ExactFlatIndex retains references to caller-owned database and IDs arrays
      without defensive copying to avoid doubling memory allocation for large
      datasets.
    - Properties `.database` and `.ids` return internal array references. External
      in-place mutation after construction bypasses initial validation and alters
      subsequent search results. Callers requiring buffer isolation must pass copies.
    """


    def __init__(
        self,
        database: np.ndarray,
        ids: np.ndarray | None = None,
    ) -> None:
        if not isinstance(database, np.ndarray):
            raise TypeError(
                f"database must be a numpy ndarray, got {type(database).__name__}"
            )
        if database.dtype != np.float32:
            raise TypeError(
                f"database must have dtype float32, got {database.dtype}. "
                "Silent casting is not permitted."
            )
        if database.ndim != 2:
            raise ValueError(
                f"database must be a 2D array of shape (N, d), got ndim={database.ndim}"
            )
        if not np.all(np.isfinite(database)):
            raise ValueError("database vectors must contain only finite numbers (no NaN or Inf).")

        if ids is None:
            self._ids = np.arange(database.shape[0], dtype=np.int64)
        else:
            self._ids = _validate_ids(ids, expected_n=database.shape[0])

        self._database = database

    @property
    def database(self) -> np.ndarray:
        """Database vectors of shape (N, d)."""
        return self._database

    @property
    def ids(self) -> np.ndarray:
        """Durable entity IDs of shape (N,)."""
        return self._ids

    @property
    def n(self) -> int:
        """Number of vectors in the database."""
        return self._database.shape[0]

    @property
    def d(self) -> int:
        """Dimensionality of the vector space."""
        return self._database.shape[1]

    def __len__(self) -> int:
        """Return the number of vectors stored in the index."""
        return self._database.shape[0]

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        max_db_chunk: int = 32768,
    ) -> SearchResult:
        """Search nearest neighbors for queries using reference semantics.

        Parameters
        ----------
        queries : np.ndarray
            Query vector(s) of shape `(d,)` or `(Q, d)` with strict `dtype=np.float32`.
        k : int, default=10
            Number of nearest neighbors to retrieve. Must be a positive integer (k >= 1).
        max_db_chunk : int, default=32768
            Maximum number of database vectors evaluated per block during distance scoring.

        Returns
        -------
        SearchResult
            SearchResult with `.distances` and `.ids`.
        """
        return exact_search(queries=queries, database=self._database, ids=self._ids, k=k, max_db_chunk=max_db_chunk)

    @property
    def index_memory_bytes(self) -> int:
        """Logical storage memory represented by indexed vectors and IDs in bytes."""
        return int(self._database.shape[0] * (self._database.shape[1] * 4 + 8))

    @property
    def index_memory_mib(self) -> float:
        """Logical storage memory in mebibytes (MiB)."""
        return float(self.index_memory_bytes / (1024.0 * 1024.0))

    @property
    def bytes_per_vector(self) -> float:
        """Average logical storage bytes per indexed vector."""
        if self._database.shape[0] == 0:
            return 0.0
        return float(self.index_memory_bytes / self._database.shape[0])


class ExactBlasIndex:
    """Exact Flat vector index accelerated via BLAS matrix multiplication.

    This index implements the algebraic norm-expansion formulation:
        ||x - y||² = ||x||² + ||y||² - 2 (x · y)
    using high-performance BLAS-backed matrix multiplication. It serves as
    Indexora's fast exact search baseline for fair comparison against approximate
    nearest neighbor methods (e.g. IVF-Flat).

    Parameters
    ----------
    database : np.ndarray
        Database vectors of shape `(N, d)` with strict `dtype=np.float32`.
    ids : np.ndarray | None, default=None
        Durable identifiers of shape `(N,)` with strict `dtype=np.int64`.
        Must be strictly unique among live records. If None, defaults to
        `np.arange(N, dtype=np.int64)`.
    """

    def __init__(
        self,
        database: np.ndarray,
        ids: np.ndarray | None = None,
    ) -> None:
        if not isinstance(database, np.ndarray):
            raise TypeError(
                f"database must be a numpy ndarray, got {type(database).__name__}"
            )
        if database.dtype != np.float32:
            raise TypeError(
                f"database must have dtype float32, got {database.dtype}. "
                "Silent casting is not permitted."
            )
        if database.ndim != 2:
            raise ValueError(
                f"database must be a 2D array of shape (N, d), got ndim={database.ndim}"
            )
        if not np.all(np.isfinite(database)):
            raise ValueError("database vectors must contain only finite numbers (no NaN or Inf).")

        if ids is None:
            self._ids = np.arange(database.shape[0], dtype=np.int64)
        else:
            self._ids = _validate_ids(ids, expected_n=database.shape[0])

        self._database = database

    @property
    def database(self) -> np.ndarray:
        """Database vectors of shape (N, d)."""
        return self._database

    @property
    def ids(self) -> np.ndarray:
        """Durable entity IDs of shape (N,)."""
        return self._ids

    @property
    def n(self) -> int:
        """Number of vectors in the database."""
        return self._database.shape[0]

    @property
    def d(self) -> int:
        """Dimensionality of the vector space."""
        return self._database.shape[1]

    def __len__(self) -> int:
        """Return the number of vectors stored in the index."""
        return self._database.shape[0]

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        max_db_chunk: int = 32768,
        max_query_chunk: int = 1024,
    ) -> SearchResult:
        """Search nearest neighbors for queries using BLAS-accelerated exact scan.

        Parameters
        ----------
        queries : np.ndarray
            Query vector(s) of shape `(d,)` or `(Q, d)` with strict `dtype=np.float32`.
        k : int, default=10
            Number of nearest neighbors to retrieve. Must be a positive integer (k >= 1).
        max_db_chunk : int, default=32768
            Maximum number of database vectors evaluated per block during distance scoring.
        max_query_chunk : int, default=1024
            Maximum number of query vectors evaluated per block during distance scoring.

        Returns
        -------
        SearchResult
            SearchResult with `.distances`, `.ids`, and `.mask`.
        """
        return exact_search_blas(
            queries=queries,
            database=self._database,
            ids=self._ids,
            k=k,
            max_db_chunk=max_db_chunk,
            max_query_chunk=max_query_chunk,
        )

    @property
    def index_memory_bytes(self) -> int:
        """Logical storage memory represented by indexed vectors and IDs in bytes."""
        return int(self._database.shape[0] * (self._database.shape[1] * 4 + 8))

    @property
    def index_memory_mib(self) -> float:
        """Logical storage memory in mebibytes (MiB)."""
        return float(self.index_memory_bytes / (1024.0 * 1024.0))

    @property
    def bytes_per_vector(self) -> float:
        """Average logical storage bytes per indexed vector."""
        if self._database.shape[0] == 0:
            return 0.0
        return float(self.index_memory_bytes / self._database.shape[0])
