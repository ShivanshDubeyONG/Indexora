"""IVF-Flat (Inverted File with Flat Vectors) index for Indexora.

This module provides Indexora's first approximate nearest neighbor (ANN) index:
- `InvertedList`: Container for vectors and IDs belonging to an inverted list.
- `IVFFlatIndex`: Inverted File index with coarse k-means quantization and multi-probe search.
- `ivf_flat_build` / `ivf_flat_search`: Functional interface mirroring exact search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from indexora.clustering import kmeans
from indexora.kernels import pairwise_squared_l2_ref
from indexora.topk import SearchResult, select_top_k


@dataclass(frozen=True)
class InvertedList:
    """Immutable representation of a single inverted list (posting list).

    Attributes
    ----------
    ids : np.ndarray
        1D int64 array of durable vector IDs assigned to this cluster.
    vectors : np.ndarray
        2D float32 array of shape `(L, d)` containing the raw vectors assigned to this cluster.
        Guaranteed to be C-contiguous and finite.
    """

    ids: np.ndarray
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.ids, np.ndarray) or self.ids.dtype != np.int64:
            raise TypeError("InvertedList.ids must be an int64 numpy ndarray.")
        if self.ids.ndim != 1:
            raise ValueError(f"InvertedList.ids must be 1D, got ndim={self.ids.ndim}")
        if not isinstance(self.vectors, np.ndarray) or self.vectors.dtype != np.float32:
            raise TypeError("InvertedList.vectors must be a float32 numpy ndarray.")
        if self.vectors.ndim != 2:
            raise ValueError(f"InvertedList.vectors must be 2D, got ndim={self.vectors.ndim}")
        if not self.vectors.flags.c_contiguous:
            raise ValueError("InvertedList.vectors must be C-contiguous.")
        if self.ids.shape[0] != self.vectors.shape[0]:
            raise ValueError(
                f"Length mismatch: {self.ids.shape[0]} IDs but {self.vectors.shape[0]} vectors."
            )
        self.ids.flags.writeable = False
        self.vectors.flags.writeable = False

    @property
    def size(self) -> int:
        """Number of vectors currently in this inverted list."""
        return self.ids.shape[0]


def _validate_ivf_build_inputs(
    database: np.ndarray,
    nlist: int,
    ids: np.ndarray | None,
    nprobe: int,
) -> tuple[int, int, np.ndarray]:
    """Validate database, nlist, ids, and nprobe for IVF-Flat index construction."""
    if not isinstance(database, np.ndarray):
        raise TypeError(f"database must be a numpy ndarray, got {type(database).__name__}")
    if database.dtype != np.float32:
        raise TypeError(
            f"database must have strict dtype float32, got {database.dtype}. Silent casting is not permitted."
        )
    if database.ndim != 2:
        raise ValueError(f"database must be a 2D array of shape (N, d), got ndim={database.ndim}")
    if not database.flags.c_contiguous:
        raise ValueError("database must be C-contiguous in memory.")
    if not np.all(np.isfinite(database)):
        raise ValueError("database vectors must contain only finite numbers (no NaN or Inf).")

    N, d = database.shape
    if N < 1:
        raise ValueError(f"database must contain at least 1 vector, got N={N}")
    if d < 1:
        raise ValueError(f"database feature dimension d must be positive, got d={d}")

    if not isinstance(nlist, (int, np.integer)):
        raise TypeError(f"nlist must be an integer, got {type(nlist).__name__}")
    if nlist < 1:
        raise ValueError(f"nlist must be at least 1, got nlist={nlist}")
    if nlist > N:
        raise ValueError(f"nlist cannot exceed database size N: nlist={nlist} > N={N}")

    if not isinstance(nprobe, (int, np.integer)):
        raise TypeError(f"nprobe must be an integer, got {type(nprobe).__name__}")
    if nprobe < 1:
        raise ValueError(f"nprobe must be at least 1, got nprobe={nprobe}")
    if nprobe > nlist:
        raise ValueError(f"nprobe cannot exceed nlist: nprobe={nprobe} > nlist={nlist}")

    if ids is None:
        valid_ids = np.arange(N, dtype=np.int64)
    else:
        if not isinstance(ids, np.ndarray):
            raise TypeError(f"ids must be a numpy ndarray, got {type(ids).__name__}")
        if ids.dtype != np.int64:
            raise TypeError(f"ids must have strict dtype int64, got {ids.dtype}.")
        if ids.ndim != 1:
            raise ValueError(f"ids must be a 1D array of shape (N,), got ndim={ids.ndim}")
        if ids.shape[0] != N:
            raise ValueError(f"Length mismatch: database has {N} vectors but ids has {ids.shape[0]} elements")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Database IDs must be strictly unique.")
        valid_ids = ids

    return N, d, valid_ids


def _validate_centroids(
    centroids: np.ndarray,
    expected_nlist: int,
    expected_d: int,
) -> np.ndarray:
    """Validate pre-supplied coarse centroids array."""
    if not isinstance(centroids, np.ndarray):
        raise TypeError(f"centroids must be a numpy ndarray, got {type(centroids).__name__}")
    if centroids.dtype != np.float32:
        raise TypeError(
            f"centroids must have strict dtype float32, got {centroids.dtype}. Silent casting is not permitted."
        )
    if centroids.ndim != 2:
        raise ValueError(f"centroids must be a 2D array of shape (nlist, d), got ndim={centroids.ndim}")
    if centroids.shape != (expected_nlist, expected_d):
        raise ValueError(
            f"centroids shape mismatch: expected ({expected_nlist}, {expected_d}), got {centroids.shape}"
        )
    if not centroids.flags.c_contiguous:
        raise ValueError("centroids must be C-contiguous in memory.")
    if not np.all(np.isfinite(centroids)):
        raise ValueError("centroids must contain only finite numbers (no NaN or Inf).")
    return centroids


class IVFFlatIndex:
    """Inverted File Flat (IVF-Flat) index for approximate nearest neighbor search.

    The index partitions the vector space into `nlist` Voronoi cells using coarse
    centroids trained via k-means. Vectors and their stable `int64` IDs are assigned
    to inverted lists. Queries probe the nearest `nprobe` lists and perform exact
    distance scoring on candidate vectors within those lists.

    Parameters
    ----------
    database : np.ndarray
        Database vectors of shape `(N, d)` with strict `dtype=np.float32`. Must be C-contiguous.
    nlist : int
        Number of inverted lists / coarse clusters (`1 <= nlist <= N`).
    ids : np.ndarray | None, default=None
        Durable identifiers of shape `(N,)` with strict `dtype=np.int64`. Must be unique.
        If None, defaults to `np.arange(N, dtype=np.int64)`.
    nprobe : int, default=1
        Default number of inverted lists to probe during search (`1 <= nprobe <= nlist`).
    centroids : np.ndarray | None, default=None
        Pre-trained coarse centroids of shape `(nlist, d)`. If None, centroids are trained
        automatically using Phase 4 `kmeans`.
    seed : int | np.random.Generator | None, default=42
        Random seed for k-means centroid training if centroids are not pre-supplied.
    max_iter : int, default=100
        Maximum k-means iterations for centroid training.
    tol : float, default=1e-4
        Convergence tolerance for k-means centroid training.
    """

    def __init__(
        self,
        database: np.ndarray,
        nlist: int,
        ids: np.ndarray | None = None,
        nprobe: int = 1,
        centroids: np.ndarray | None = None,
        seed: int | np.random.Generator | None = 42,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> None:
        N, d, valid_ids = _validate_ivf_build_inputs(database, nlist, ids, nprobe)

        self._d = d
        self._ntotal = N
        self._nlist = nlist
        self._nprobe = nprobe

        # 1. Coarse Centroids: train via Phase 4 kmeans or validate pre-supplied codebook
        if centroids is None:
            km_res = kmeans(
                database,
                k=nlist,
                max_iter=max_iter,
                tol=tol,
                seed=seed,
                empty_cluster_policy="reseed_furthest",
            )
            self._centroids = np.ascontiguousarray(km_res.centroids, dtype=np.float32)
        else:
            validated_c = _validate_centroids(centroids, expected_nlist=nlist, expected_d=d)
            # Defensive copy ensures caller cannot mutate internal centroid representation
            self._centroids = np.ascontiguousarray(validated_c.copy(), dtype=np.float32)

        # 2. Coarse Quantization: assign each database vector to nearest coarse centroid
        # Uses bounded database blocks to avoid allocating an unconstrained (N, nlist) matrix
        assignments = np.empty(N, dtype=np.int64)
        assign_block_size = 10000
        for b_start in range(0, N, assign_block_size):
            b_end = min(b_start + assign_block_size, N)
            block_dists = pairwise_squared_l2_ref(database[b_start:b_end], self._centroids)
            # Deterministic tie-breaking: np.argmin selects lowest centroid index on equal distance
            assignments[b_start:b_end] = np.argmin(block_dists, axis=1)

        # 3. Build inverted lists (one per coarse centroid)
        invlists = []
        for j in range(nlist):
            mask = (assignments == j)
            if np.any(mask):
                lst_ids = np.ascontiguousarray(valid_ids[mask], dtype=np.int64)
                lst_vecs = np.ascontiguousarray(database[mask], dtype=np.float32)
            else:
                # Valid empty list for unpopulated cluster
                lst_ids = np.empty((0,), dtype=np.int64)
                lst_vecs = np.empty((0, d), dtype=np.float32)
            invlists.append(InvertedList(ids=lst_ids, vectors=lst_vecs))

        self._invlists = tuple(invlists)

    @property
    def d(self) -> int:
        """Feature dimensionality."""
        return self._d

    @property
    def ntotal(self) -> int:
        """Total number of indexed vectors."""
        return self._ntotal

    @property
    def nlist(self) -> int:
        """Number of inverted lists."""
        return self._nlist

    @property
    def nprobe(self) -> int:
        """Default number of inverted lists probed per query."""
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        if not isinstance(value, (int, np.integer)) or value < 1 or value > self._nlist:
            raise ValueError(f"nprobe must be an integer between 1 and {self._nlist}, got {value}")
        self._nprobe = int(value)

    @property
    def centroids(self) -> np.ndarray:
        """Coarse centroids of shape `(nlist, d)` with `dtype=np.float32`."""
        return self._centroids.copy()

    @property
    def invlists(self) -> tuple[InvertedList, ...]:
        """Tuple of all inverted lists, length `nlist`."""
        return self._invlists

    def list_sizes(self) -> np.ndarray:
        """Return 1D int64 array of shape `(nlist,)` with the occupancy count of each list."""
        return np.array([lst.size for lst in self._invlists], dtype=np.int64)

    def get_list(self, list_id: int) -> InvertedList:
        """Return the inverted list for cluster index `list_id`."""
        if not isinstance(list_id, (int, np.integer)):
            raise TypeError(f"list_id must be an integer, got {type(list_id).__name__}")
        if not (0 <= list_id < self._nlist):
            raise IndexError(f"list_id {list_id} out of bounds for nlist={self._nlist}")
        return self._invlists[list_id]

    @property
    def index_memory_bytes(self) -> int:
        """Total memory in bytes of coarse centroids and stored inverted lists."""
        centroid_bytes = int(self._centroids.nbytes)
        invlists_bytes = sum(int(lst.vectors.nbytes + lst.ids.nbytes) for lst in self._invlists)
        return centroid_bytes + invlists_bytes

    @property
    def index_memory_mib(self) -> float:
        """Total index storage memory in mebibytes (MiB)."""
        return float(self.index_memory_bytes / (1024.0 * 1024.0))

    @property
    def bytes_per_vector(self) -> float:
        """Average storage bytes per indexed vector."""
        if self._ntotal == 0:
            return 0.0
        return float(self.index_memory_bytes / self._ntotal)

    def _search_single(
        self,
        query: np.ndarray,
        k: int,
        eff_nprobe: int,
    ) -> SearchResult:
        """Search nearest neighbors for a single query vector of shape `(d,)`."""
        # 1. Rank coarse centroids by distance to query
        # Shape: (nlist,)
        centroid_dists = pairwise_squared_l2_ref(query, self._centroids)

        # Stable sort guarantees that equidistant centroids break ties by lowest centroid index
        probed_centroids = np.argsort(centroid_dists, kind="stable")[:eff_nprobe]

        # 2. Candidate Gathering: accumulate vectors and IDs from probed lists
        cand_vec_list = []
        cand_id_list = []
        for c in probed_centroids:
            lst = self._invlists[c]
            if lst.size > 0:
                cand_vec_list.append(lst.vectors)
                cand_id_list.append(lst.ids)

        # 3. Handle case where all probed lists are empty
        if not cand_vec_list:
            return SearchResult(
                distances=np.empty((0,), dtype=np.float32),
                ids=np.empty((0,), dtype=np.int64),
                mask=np.empty((0,), dtype=bool),
                distance_computations=self._nlist,
                centroid_distance_computations=self._nlist,
                candidate_distance_computations=0,
            )

        # 4. Concatenate candidate buffers
        cand_vecs = np.concatenate(cand_vec_list, axis=0)
        cand_ids = np.concatenate(cand_id_list, axis=0)
        if not cand_vecs.flags.c_contiguous:
            cand_vecs = np.ascontiguousarray(cand_vecs)

        # 5. Exact distance calculation against candidate vectors
        cand_dists = pairwise_squared_l2_ref(query, cand_vecs)

        # 6. Top-k selection with deterministic (distance, ID) ordering
        res = select_top_k(cand_dists, cand_ids, k=k)
        total_dc = self._nlist + len(cand_ids)
        return SearchResult(
            distances=res.distances,
            ids=res.ids,
            mask=res.mask,
            distance_computations=total_dc,
            centroid_distance_computations=self._nlist,
            candidate_distance_computations=len(cand_ids),
        )

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        nprobe: int | None = None,
    ) -> SearchResult:
        """Search nearest neighbors for query vectors using multi-probe inverted list lookup.

        Parameters
        ----------
        queries : np.ndarray
            Query vector(s) of shape `(d,)` or `(Q, d)` with strict `dtype=np.float32`.
        k : int, default=10
            Number of nearest neighbors to retrieve per query (`k >= 1`).
        nprobe : int | None, default=None
            Number of inverted lists to probe (`1 <= nprobe <= nlist`). If None,
            defaults to `self.nprobe`.

        Returns
        -------
        SearchResult
            Named container with `.distances` (float32) and `.ids` (int64):
            - Single query `(d,)`: 1D arrays of exact length `min(k, N_cand)` without padding.
            - Batched queries `(Q, d)`: 2D arrays of shape `(Q, target_k)` where
              `target_k = max(retrieved_lengths)`. If queries retrieve variable numbers of
              candidates (< k), shorter result rows are padded with sentinel ID `-1` and
              sentinel distance `inf` to maintain rectangular 2D arrays per ADR-0006.
        """
        # Validate k
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"k must be an integer, got {type(k).__name__}")
        if k <= 0:
            raise ValueError("k must be positive")

        # Validate and resolve effective nprobe
        if nprobe is None:
            eff_nprobe = self._nprobe
        else:
            if not isinstance(nprobe, (int, np.integer)):
                raise TypeError(f"nprobe must be an integer, got {type(nprobe).__name__}")
            if nprobe < 1:
                raise ValueError(f"nprobe must be at least 1, got nprobe={nprobe}")
            if nprobe > self._nlist:
                raise ValueError(
                    f"nprobe cannot exceed nlist: nprobe={nprobe} > nlist={self._nlist}"
                )
            eff_nprobe = int(nprobe)

        # Validate queries array
        if not isinstance(queries, np.ndarray):
            raise TypeError(f"queries must be a numpy ndarray, got {type(queries).__name__}")
        if queries.dtype != np.float32:
            raise TypeError(
                f"queries must have strict dtype float32, got {queries.dtype}. Silent casting is not permitted."
            )
        if not np.all(np.isfinite(queries)):
            raise ValueError("queries must contain only finite numbers (no NaN or Inf).")

        is_single_query = queries.ndim == 1
        if is_single_query:
            if queries.shape[0] != self._d:
                raise ValueError(
                    f"Query feature dimension {queries.shape[0]} does not match index dimension {self._d}"
                )
            if not queries.flags.c_contiguous:
                raise ValueError("queries must be C-contiguous in memory.")
            return self._search_single(queries, k=k, eff_nprobe=eff_nprobe)

        if queries.ndim != 2:
            raise ValueError(
                f"queries must be 1D or 2D array, got ndim={queries.ndim}"
            )

        if queries.shape[1] != self._d:
            raise ValueError(
                f"Query feature dimension {queries.shape[1]} does not match index dimension {self._d}"
            )
        if not queries.flags.c_contiguous:
            raise ValueError("queries must be C-contiguous in memory.")

        num_queries = queries.shape[0]
        if num_queries == 0:
            return SearchResult(
                distances=np.empty((0, min(k, self._ntotal)), dtype=np.float32),
                ids=np.empty((0, min(k, self._ntotal)), dtype=np.int64),
                mask=np.empty((0, min(k, self._ntotal)), dtype=bool),
                distance_computations=0,
                centroid_distance_computations=0,
                candidate_distance_computations=0,
            )

        per_query_results = [
            self._search_single(queries[i], k=k, eff_nprobe=eff_nprobe)
            for i in range(num_queries)
        ]

        total_dc = sum(
            r.distance_computations for r in per_query_results if r.distance_computations is not None
        )
        total_centroid_dc = sum(
            r.centroid_distance_computations for r in per_query_results if r.centroid_distance_computations is not None
        )
        total_cand_dc = sum(
            r.candidate_distance_computations for r in per_query_results if r.candidate_distance_computations is not None
        )
        lengths = [len(r.ids) for r in per_query_results]
        if len(set(lengths)) == 1:
            batch_distances = np.stack([r.distances for r in per_query_results], axis=0)
            batch_ids = np.stack([r.ids for r in per_query_results], axis=0)
            batch_mask = np.stack([r.mask for r in per_query_results], axis=0)
        else:
            # Handle uneven candidate counts across queries: pad up to max retrieved with -1 and inf, mask=False
            target_k = max(lengths)
            batch_distances = np.full((num_queries, target_k), np.float32(np.inf), dtype=np.float32)
            batch_ids = np.full((num_queries, target_k), np.int64(-1), dtype=np.int64)
            batch_mask = np.zeros((num_queries, target_k), dtype=bool)
            for i, r in enumerate(per_query_results):
                L = len(r.ids)
                batch_distances[i, :L] = r.distances
                batch_ids[i, :L] = r.ids
                batch_mask[i, :L] = r.mask

        return SearchResult(
            distances=batch_distances,
            ids=batch_ids,
            mask=batch_mask,
            distance_computations=total_dc,
            centroid_distance_computations=total_centroid_dc,
            candidate_distance_computations=total_cand_dc,
        )


def ivf_flat_build(
    database: np.ndarray,
    nlist: int,
    ids: np.ndarray | None = None,
    nprobe: int = 1,
    centroids: np.ndarray | None = None,
    seed: int | np.random.Generator | None = 42,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> IVFFlatIndex:
    """Construct an IVFFlatIndex from database vectors."""
    return IVFFlatIndex(
        database=database,
        nlist=nlist,
        ids=ids,
        nprobe=nprobe,
        centroids=centroids,
        seed=seed,
        max_iter=max_iter,
        tol=tol,
    )


def ivf_flat_search(
    index: IVFFlatIndex,
    queries: np.ndarray,
    k: int = 10,
    nprobe: int | None = None,
) -> SearchResult:
    """Search an IVFFlatIndex using query vectors."""
    if not isinstance(index, IVFFlatIndex):
        raise TypeError(f"index must be an IVFFlatIndex instance, got {type(index).__name__}")
    return index.search(queries=queries, k=k, nprobe=nprobe)
