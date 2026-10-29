"""Inverted File with Product Quantization (IVF-PQ) index for Indexora.

This module provides the Phase 9 IVF-PQ approximate nearest neighbor index:
- `InvertedPQList`: Container for posting list IDs and compact quantized codes.
- `IVFPQIndex`: IVF coarse quantization combined with Product Quantization on
  residual or raw vectors, featuring Asymmetric Distance Computation (ADC) lookup,
  deterministic top-k selection, and exact memory accounting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from indexora.clustering import kmeans
from indexora.kernels import pairwise_squared_l2_blas, pairwise_squared_l2_ref
from indexora.pq import ProductQuantizer, _validate_pq_inputs
from indexora.topk import SearchResult, select_top_k


@dataclass(frozen=True)
class InvertedPQList:
    """Immutable inverted list holding stable entity IDs and compact PQ codes.

    Attributes
    ----------
    ids : np.ndarray
        1D array of shape `(size,)` with strict `dtype=np.int64`.
    codes : np.ndarray
        2D array of shape `(size, M)` with unsigned integer dtype (`uint8` or `uint16`).
    """

    ids: np.ndarray
    codes: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.ids, np.ndarray):
            raise TypeError(f"ids must be a numpy ndarray, got {type(self.ids).__name__}")
        if self.ids.dtype != np.int64:
            raise TypeError(f"ids must have dtype int64, got {self.ids.dtype}")
        if self.ids.ndim != 1:
            raise ValueError(f"ids must be a 1D array of shape (size,), got ndim={self.ids.ndim}")

        if not isinstance(self.codes, np.ndarray):
            raise TypeError(f"codes must be a numpy ndarray, got {type(self.codes).__name__}")
        if not np.issubdtype(self.codes.dtype, np.integer):
            raise TypeError(f"codes must have integer dtype, got {self.codes.dtype}")
        if self.codes.ndim != 2:
            raise ValueError(f"codes must be a 2D array of shape (size, M), got ndim={self.codes.ndim}")

        if self.ids.shape[0] != self.codes.shape[0]:
            raise ValueError(
                f"Length mismatch: ids has {self.ids.shape[0]} elements but "
                f"codes has {self.codes.shape[0]} rows"
            )
        if not self.ids.flags.c_contiguous:
            raise ValueError("ids must be C-contiguous in memory.")
        if not self.codes.flags.c_contiguous:
            raise ValueError("codes must be C-contiguous in memory.")

    @property
    def size(self) -> int:
        """Number of vectors currently indexed in this inverted list."""
        return int(self.ids.shape[0])


def _validate_ivfpq_build_inputs(
    database: np.ndarray,
    nlist: int,
    M: int,
    k_sub: int,
    ids: np.ndarray | None,
    nprobe: int,
) -> tuple[int, int, np.ndarray]:
    """Validate database, dimensions, IDs, and parameters for IVFPQIndex construction."""
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
        raise ValueError("database must contain only finite numbers (no NaN or Inf).")

    N, d = database.shape
    if not isinstance(nlist, (int, np.integer)):
        raise TypeError(f"nlist must be an integer, got {type(nlist).__name__}")
    if nlist < 1:
        raise ValueError(f"nlist must be at least 1, got {nlist}")
    if nlist > N and N > 0:
        raise ValueError(f"nlist cannot exceed database size N: nlist={nlist} > N={N}")

    _validate_pq_inputs(d, M, k_sub)

    if not isinstance(nprobe, (int, np.integer)):
        raise TypeError(f"nprobe must be an integer, got {type(nprobe).__name__}")
    if nprobe < 1:
        raise ValueError(f"nprobe must be at least 1, got {nprobe}")
    if nprobe > nlist:
        raise ValueError(f"nprobe cannot exceed nlist: nprobe={nprobe} > nlist={nlist}")

    if ids is not None:
        if not isinstance(ids, np.ndarray):
            raise TypeError(f"ids must be a numpy ndarray, got {type(ids).__name__}")
        if ids.dtype != np.int64:
            raise TypeError(
                f"ids must have strict dtype int64, got {ids.dtype}. Silent casting is not permitted."
            )
        if ids.ndim != 1 or ids.shape[0] != N:
            raise ValueError(f"ids must be 1D array of shape ({N},), got shape {ids.shape}")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("ids must be strictly unique among live database vectors.")
        valid_ids = np.ascontiguousarray(ids, dtype=np.int64)
    else:
        valid_ids = np.arange(N, dtype=np.int64)

    return N, d, valid_ids


class IVFPQIndex:
    """Inverted File with Product Quantization (IVF-PQ) vector search index.

    Combines inverted-file coarse quantization with subspace product quantization.
    Posting lists store only compact PQ centroid codes and external int64 IDs,
    achieving order-of-magnitude memory compression relative to raw float32 vectors.
    Search evaluates candidates using Asymmetric Distance Computation (ADC) lookup
    tables computed against query residuals.

    Parameters
    ----------
    database : np.ndarray
        Database vectors of shape `(N, d)` with strict `dtype=np.float32`.
    nlist : int
        Number of inverted lists / coarse clusters (`1 <= nlist <= N`).
    M : int
        Number of equal-sized Product Quantization subspaces (`d % M == 0`).
    k_sub : int, default=256
        Number of centroids per subspace (1 <= k_sub <= 65536).
    ids : np.ndarray | None, default=None
        Durable identifiers of shape `(N,)` with strict `dtype=np.int64`.
    nprobe : int, default=1
        Default number of inverted lists to probe during search.
    coarse_centroids : np.ndarray | None, default=None
        Pre-trained coarse centroids of shape `(nlist, d)`. If None, trained via k-means.
    pq : ProductQuantizer | None, default=None
        Pre-trained ProductQuantizer. If None, trained automatically.
    seed : int | None, default=42
        Random seed for k-means training of centroids and codebooks.
    max_iter : int, default=100
        Maximum k-means iterations for centroid and codebook training.
    tol : float, default=1e-4
        Convergence tolerance for k-means training.
    use_residuals : bool, default=True
        Whether to quantize coarse-centroid residual vectors ($x - c$) rather than
        raw vectors ($x$). Residual quantization significantly improves recall.
    """

    def __init__(
        self,
        database: np.ndarray,
        nlist: int,
        M: int,
        k_sub: int = 256,
        ids: np.ndarray | None = None,
        nprobe: int = 1,
        coarse_centroids: np.ndarray | None = None,
        pq: ProductQuantizer | None = None,
        seed: int | None = 42,
        max_iter: int = 100,
        tol: float = 1e-4,
        use_residuals: bool = True,
    ) -> None:
        N, d, valid_ids = _validate_ivfpq_build_inputs(database, nlist, M, k_sub, ids, nprobe)

        self._d = d
        self._ntotal = N
        self._nlist = nlist
        self._M = M
        self._k_sub = k_sub
        self._d_sub = d // M
        self._nprobe = nprobe
        self._use_residuals = bool(use_residuals)

        # 1. Coarse Centroids
        if coarse_centroids is None:
            km_res = kmeans(
                database,
                k=nlist,
                max_iter=max_iter,
                tol=tol,
                seed=seed,
                empty_cluster_policy="reseed_furthest",
                distance_method="blas",
            )
            self._centroids = np.ascontiguousarray(km_res.centroids, dtype=np.float32)
            self._kmeans_n_iter = km_res.n_iter
            self._kmeans_converged = km_res.converged
            self._kmeans_inertia = km_res.inertia
        else:
            if not isinstance(coarse_centroids, np.ndarray) or coarse_centroids.dtype != np.float32:
                raise TypeError("coarse_centroids must be float32 ndarray")
            if coarse_centroids.shape != (nlist, d):
                raise ValueError(
                    f"coarse_centroids shape mismatch: expected ({nlist}, {d}), got {coarse_centroids.shape}"
                )
            self._centroids = np.ascontiguousarray(coarse_centroids.copy(), dtype=np.float32)
            self._kmeans_n_iter = None
            self._kmeans_converged = None
            self._kmeans_inertia = None

        # 2. Coarse Quantization: assign database vectors to coarse centroids
        assignments = np.empty(N, dtype=np.int64)
        assign_block = 32768
        for b_start in range(0, N, assign_block):
            b_end = min(b_start + assign_block, N)
            dists = pairwise_squared_l2_blas(database[b_start:b_end], self._centroids)
            assignments[b_start:b_end] = np.argmin(dists, axis=1)

        # 3. Product Quantizer: train on residuals or raw vectors
        if pq is not None:
            if not isinstance(pq, ProductQuantizer):
                raise TypeError("pq must be a ProductQuantizer instance")
            if pq.d != d or pq.M != M or pq.k_sub != k_sub:
                raise ValueError("pq parameters do not match index configuration")
            if not pq.is_trained:
                raise RuntimeError("Provided pq must be pre-trained")
            self._pq = pq
        else:
            self._pq = ProductQuantizer(d=d, M=M, k_sub=k_sub)
            if self._use_residuals and N > 0:
                # Compute residual vectors: r_i = x_i - centroid[assignment_i]
                residuals = np.empty_like(database)
                for c in range(nlist):
                    mask = (assignments == c)
                    if np.any(mask):
                        residuals[mask] = database[mask] - self._centroids[c]
                # If N < k_sub, training requires at least k_sub samples
                self._pq.fit(residuals, max_iter=max_iter, tol=tol, seed=seed)
            elif N > 0:
                self._pq.fit(database, max_iter=max_iter, tol=tol, seed=seed)

        # 4. Construct inverted lists holding compact codes and IDs
        invlists = []
        for c in range(nlist):
            mask = (assignments == c)
            if np.any(mask):
                lst_ids = np.ascontiguousarray(valid_ids[mask], dtype=np.int64)
                if self._use_residuals:
                    sub_db = database[mask] - self._centroids[c]
                else:
                    sub_db = database[mask]
                lst_codes = self._pq.encode(sub_db)
            else:
                lst_ids = np.empty((0,), dtype=np.int64)
                lst_codes = np.empty((0, M), dtype=self._pq.code_dtype)
            invlists.append(InvertedPQList(ids=lst_ids, codes=lst_codes))

        self._invlists = list(invlists)

    @property
    def d(self) -> int:
        """Feature dimensionality."""
        return self._d

    @property
    def ntotal(self) -> int:
        """Total number of indexed vectors."""
        return self._ntotal

    def __len__(self) -> int:
        """Return the number of indexed vectors."""
        return self._ntotal

    @property
    def nlist(self) -> int:
        """Number of inverted lists."""
        return self._nlist

    @property
    def M(self) -> int:
        """Number of Product Quantization subspaces."""
        return self._M

    @property
    def d_sub(self) -> int:
        """Subspace dimensionality (`d // M`)."""
        return self._d_sub

    @property
    def k_sub(self) -> int:
        """Centroids per subspace."""
        return self._k_sub

    @property
    def nprobe(self) -> int:
        """Default number of inverted lists probed during search."""
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        if not isinstance(value, (int, np.integer)) or value < 1 or value > self._nlist:
            raise ValueError(f"nprobe must be an integer between 1 and {self._nlist}, got {value}")
        self._nprobe = int(value)

    @property
    def use_residuals(self) -> bool:
        """Whether vectors are encoded as coarse-centroid residuals."""
        return self._use_residuals

    @property
    def code_dtype(self) -> np.dtype:
        """Dtype of quantized codes (`uint8` or `uint16`)."""
        return self._pq.code_dtype

    @property
    def coarse_centroids(self) -> np.ndarray:
        """Coarse centroids of shape `(nlist, d)` with `dtype=np.float32`."""
        return self._centroids.copy()

    @property
    def pq(self) -> ProductQuantizer:
        """Trained ProductQuantizer model."""
        return self._pq

    @property
    def invlists(self) -> tuple[InvertedPQList, ...]:
        """Tuple of all inverted posting lists."""
        return tuple(self._invlists)

    @property
    def kmeans_n_iter(self) -> int | None:
        """Number of coarse k-means training iterations."""
        return self._kmeans_n_iter

    @property
    def kmeans_converged(self) -> bool | None:
        """Whether coarse k-means training converged."""
        return self._kmeans_converged

    @property
    def kmeans_inertia(self) -> float | None:
        """Final inertia of coarse k-means training."""
        return self._kmeans_inertia

    def list_sizes(self) -> np.ndarray:
        """Occupancy count of each inverted list as 1D int64 array."""
        return np.array([lst.size for lst in self._invlists], dtype=np.int64)

    def get_list(self, list_id: int) -> InvertedPQList:
        """Return the posting list for cluster index `list_id`."""
        if not isinstance(list_id, (int, np.integer)):
            raise TypeError(f"list_id must be an integer, got {type(list_id).__name__}")
        if not (0 <= list_id < self._nlist):
            raise IndexError(f"list_id {list_id} out of bounds for nlist={self._nlist}")
        return self._invlists[list_id]

    # ------------------------------------------------------------------------
    # Memory and Compression Accounting
    # ------------------------------------------------------------------------

    @property
    def coarse_centroid_memory_bytes(self) -> int:
        """Memory in bytes of coarse centroid arrays."""
        return int(self._centroids.nbytes)

    @property
    def codebook_memory_bytes(self) -> int:
        """Memory in bytes of PQ codebooks."""
        return self._pq.memory_bytes

    @property
    def codes_memory_bytes(self) -> int:
        """Memory in bytes of stored PQ codes across all inverted lists."""
        return sum(int(lst.codes.nbytes) for lst in self._invlists)

    @property
    def ids_memory_bytes(self) -> int:
        """Memory in bytes of stored IDs across all inverted lists."""
        return sum(int(lst.ids.nbytes) for lst in self._invlists)

    @property
    def index_memory_bytes(self) -> int:
        """Total logical index memory in bytes (centroids + codebooks + codes + IDs)."""
        return (
            self.coarse_centroid_memory_bytes
            + self.codebook_memory_bytes
            + self.codes_memory_bytes
            + self.ids_memory_bytes
        )

    @property
    def index_memory_mib(self) -> float:
        """Total logical index memory in mebibytes (MiB)."""
        return float(self.index_memory_bytes / (1024.0 * 1024.0))

    @property
    def bytes_per_vector(self) -> float:
        """Average logical storage bytes per indexed vector."""
        if self._ntotal == 0:
            return 0.0
        return float(self.index_memory_bytes / self._ntotal)

    @property
    def raw_uncompressed_bytes(self) -> int:
        """Theoretical storage of raw float32 database vectors and int64 IDs."""
        return int(self._ntotal * (self._d * 4 + 8))

    @property
    def compression_ratio(self) -> float:
        """Compression ratio against storing raw float32 vectors and int64 IDs."""
        if self.index_memory_bytes == 0:
            return 0.0
        return float(self.raw_uncompressed_bytes / self.index_memory_bytes)

    # ------------------------------------------------------------------------
    # Search Execution
    # ------------------------------------------------------------------------

    def _search_single(
        self,
        query: np.ndarray,
        k: int,
        eff_nprobe: int,
        filter_ids: Sequence[int] | set[int] | np.ndarray | Callable[[int], bool] | None = None,
    ) -> SearchResult:
        """Search nearest neighbors for a single query vector of shape `(d,)`."""
        # 1. Rank coarse centroids
        c_dists = pairwise_squared_l2_blas(query, self._centroids)
        probed_centroids = np.argsort(c_dists, kind="stable")[:eff_nprobe]

        # 2. Score candidates in probed lists using Asymmetric Distance Computation
        cand_dist_list: list[np.ndarray] = []
        cand_id_list: list[np.ndarray] = []

        if not self._use_residuals:
            # Non-residual mode: single LUT computed once against query
            lut = self._pq.compute_distance_table(query)
            for c in probed_centroids:
                lst = self._invlists[c]
                if lst.size > 0:
                    cand_dists = self._pq.compute_distances_from_table(lut, lst.codes)
                    cand_dist_list.append(cand_dists)
                    cand_id_list.append(lst.ids)
        else:
            # Residual mode: compute residual query (q - c) and LUT per probed cluster
            for c in probed_centroids:
                lst = self._invlists[c]
                if lst.size > 0:
                    q_res = query - self._centroids[c]
                    lut_c = self._pq.compute_distance_table(q_res)
                    cand_dists = self._pq.compute_distances_from_table(lut_c, lst.codes)
                    cand_dist_list.append(cand_dists)
                    cand_id_list.append(lst.ids)

        if not cand_dist_list:
            return SearchResult(
                distances=np.empty((0,), dtype=np.float32),
                ids=np.empty((0,), dtype=np.int64),
                mask=np.empty((0,), dtype=bool),
                distance_computations=self._nlist,
                centroid_distance_computations=self._nlist,
                candidate_distance_computations=0,
            )

        if len(cand_dist_list) == 1:
            all_cand_dists = cand_dist_list[0]
            all_cand_ids = cand_id_list[0]
        else:
            all_cand_dists = np.concatenate(cand_dist_list, axis=0)
            all_cand_ids = np.concatenate(cand_id_list, axis=0)

        if filter_ids is not None:
            if callable(filter_ids):
                f_mask = np.fromiter((bool(filter_ids(int(x))) for x in all_cand_ids), dtype=bool, count=len(all_cand_ids))
            elif isinstance(filter_ids, (set, frozenset)):
                f_mask = np.fromiter((int(x) in filter_ids for x in all_cand_ids), dtype=bool, count=len(all_cand_ids))
            else:
                f_set = set(int(x) for x in filter_ids)
                f_mask = np.fromiter((int(x) in f_set for x in all_cand_ids), dtype=bool, count=len(all_cand_ids))
            all_cand_dists = all_cand_dists[f_mask]
            all_cand_ids = all_cand_ids[f_mask]

        if len(all_cand_ids) == 0:
            return SearchResult(
                distances=np.empty((0,), dtype=np.float32),
                ids=np.empty((0,), dtype=np.int64),
                mask=np.empty((0,), dtype=bool),
                distance_computations=self._nlist,
                centroid_distance_computations=self._nlist,
                candidate_distance_computations=0,
            )

        # 3. Deterministic top-k selection
        res = select_top_k(all_cand_dists, all_cand_ids, k=k)
        total_cand_count = len(all_cand_ids)
        return SearchResult(
            distances=res.distances,
            ids=res.ids,
            mask=res.mask,
            distance_computations=self._nlist + total_cand_count,
            centroid_distance_computations=self._nlist,
            candidate_distance_computations=total_cand_count,
        )

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        nprobe: int | None = None,
        filter_ids: Sequence[int] | set[int] | np.ndarray | Callable[[int], bool] | None = None,
    ) -> SearchResult:
        """Search nearest neighbors for query vectors using multi-probe IVF-PQ.

        Parameters
        ----------
        queries : np.ndarray
            Query vector(s) of shape `(d,)` or `(Q, d)` with strict `dtype=np.float32`.
        k : int, default=10
            Number of nearest neighbors to retrieve per query (`k >= 1`).
        nprobe : int | None, default=None
            Number of inverted lists to probe (`1 <= nprobe <= nlist`). If None,
            defaults to `self.nprobe`.
        filter_ids : Sequence[int] | set[int] | np.ndarray | Callable[[int], bool] | None, default=None
            Eligibility filter restricting candidates to matching external IDs.

        Returns
        -------
        SearchResult
            SearchResult container conforming to Indexora's standard API.
        """
        # Validate k
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"k must be an integer, got {type(k).__name__}")
        if k <= 0:
            raise ValueError("k must be positive")

        # Validate nprobe
        if nprobe is None:
            eff_nprobe = self._nprobe
        else:
            if not isinstance(nprobe, (int, np.integer)):
                raise TypeError(f"nprobe must be an integer, got {type(nprobe).__name__}")
            if nprobe < 1:
                raise ValueError(f"nprobe must be at least 1, got nprobe={nprobe}")
            if nprobe > self._nlist:
                raise ValueError(f"nprobe cannot exceed nlist: {nprobe} > {self._nlist}")
            eff_nprobe = int(nprobe)

        # Validate queries
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
            return self._search_single(queries, k=k, eff_nprobe=eff_nprobe, filter_ids=filter_ids)

        if queries.ndim != 2:
            raise ValueError(f"queries must be 1D or 2D array, got ndim={queries.ndim}")
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

        # 1. Batched Coarse Quantization: compute coarse centroid distances for all queries
        all_c_dists = pairwise_squared_l2_blas(queries, self._centroids)
        all_probed = np.argsort(all_c_dists, axis=1, kind="stable")[:, :eff_nprobe]

        per_query_results: list[SearchResult] = []

        if not self._use_residuals:
            # Non-residual mode: precompute all query distance tables in one batched call
            all_luts = self._pq.compute_distance_table(queries)

        for i in range(num_queries):
            probed_centroids = all_probed[i]
            cand_dist_list: list[np.ndarray] = []
            cand_id_list: list[np.ndarray] = []

            if not self._use_residuals:
                lut_i = all_luts[i]
                for c in probed_centroids:
                    lst = self._invlists[c]
                    if lst.size > 0:
                        cand_dists = self._pq.compute_distances_from_table(lut_i, lst.codes)
                        cand_dist_list.append(cand_dists)
                        cand_id_list.append(lst.ids)
            else:
                for c in probed_centroids:
                    lst = self._invlists[c]
                    if lst.size > 0:
                        q_res = queries[i] - self._centroids[c]
                        lut_c = self._pq.compute_distance_table(q_res)
                        cand_dists = self._pq.compute_distances_from_table(lut_c, lst.codes)
                        cand_dist_list.append(cand_dists)
                        cand_id_list.append(lst.ids)

            if not cand_dist_list:
                per_query_results.append(
                    SearchResult(
                        distances=np.empty((0,), dtype=np.float32),
                        ids=np.empty((0,), dtype=np.int64),
                        mask=np.empty((0,), dtype=bool),
                        distance_computations=self._nlist,
                        centroid_distance_computations=self._nlist,
                        candidate_distance_computations=0,
                    )
                )
                continue

            if len(cand_dist_list) == 1:
                all_cand_dists = cand_dist_list[0]
                all_cand_ids = cand_id_list[0]
            else:
                all_cand_dists = np.concatenate(cand_dist_list, axis=0)
                all_cand_ids = np.concatenate(cand_id_list, axis=0)

            if filter_ids is not None:
                if callable(filter_ids):
                    f_mask = np.fromiter((bool(filter_ids(int(x))) for x in all_cand_ids), dtype=bool, count=len(all_cand_ids))
                elif isinstance(filter_ids, (set, frozenset)):
                    f_mask = np.fromiter((int(x) in filter_ids for x in all_cand_ids), dtype=bool, count=len(all_cand_ids))
                else:
                    f_set = set(int(x) for x in filter_ids)
                    f_mask = np.fromiter((int(x) in f_set for x in all_cand_ids), dtype=bool, count=len(all_cand_ids))
                all_cand_dists = all_cand_dists[f_mask]
                all_cand_ids = all_cand_ids[f_mask]

            if len(all_cand_ids) == 0:
                per_query_results.append(
                    SearchResult(
                        distances=np.empty((0,), dtype=np.float32),
                        ids=np.empty((0,), dtype=np.int64),
                        mask=np.empty((0,), dtype=bool),
                        distance_computations=self._nlist,
                        centroid_distance_computations=self._nlist,
                        candidate_distance_computations=0,
                    )
                )
                continue

            res = select_top_k(all_cand_dists, all_cand_ids, k=k)
            total_cands = len(all_cand_ids)
            per_query_results.append(
                SearchResult(
                    distances=res.distances,
                    ids=res.ids,
                    mask=res.mask,
                    distance_computations=self._nlist + total_cands,
                    centroid_distance_computations=self._nlist,
                    candidate_distance_computations=total_cands,
                )
            )

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

    def add(
        self,
        vectors: np.ndarray,
        ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Add new vectors to the index.

        Assigns vectors to nearest coarse centroids, computes residuals, and encodes
        them with existing Product Quantization codebooks without retraining.

        Parameters
        ----------
        vectors : np.ndarray
            Vectors array of shape `(N_new, d)` with strict `dtype=np.float32`.
        ids : np.ndarray | None, default=None
            Identifiers array of shape `(N_new,)` with strict `dtype=np.int64`. Must be unique
            and not exist in index. If None, assigns sequential IDs.

        Returns
        -------
        np.ndarray
            The array of added IDs.
        """
        if not isinstance(vectors, np.ndarray):
            raise TypeError(f"vectors must be a numpy ndarray, got {type(vectors).__name__}")
        if vectors.dtype != np.float32:
            raise TypeError(f"vectors must have dtype float32, got {vectors.dtype}")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be a 2D array, got ndim={vectors.ndim}")
        if vectors.shape[1] != self._d:
            raise ValueError(
                f"Dimension mismatch: vectors have d={vectors.shape[1]} but index expects d={self._d}"
            )
        if not vectors.flags.c_contiguous:
            vectors = np.ascontiguousarray(vectors)
        if not np.all(np.isfinite(vectors)):
            raise ValueError("vectors must contain only finite numbers (no NaN or Inf).")

        N_new = vectors.shape[0]
        if N_new == 0:
            return np.empty((0,), dtype=np.int64)

        existing_ids: set[int] = set()
        for lst in self._invlists:
            for i in lst.ids:
                existing_ids.add(int(i))

        if ids is None:
            start_id = max(existing_ids, default=-1) + 1
            new_ids = np.arange(start_id, start_id + N_new, dtype=np.int64)
        else:
            if not isinstance(ids, np.ndarray):
                raise TypeError(f"ids must be a numpy ndarray, got {type(ids).__name__}")
            if ids.dtype != np.int64:
                raise TypeError(f"ids must have dtype int64, got {ids.dtype}")
            if ids.ndim != 1 or ids.shape[0] != N_new:
                raise ValueError(f"ids must be 1D array of shape ({N_new},), got shape {ids.shape}")
            if len(np.unique(ids)) != len(ids):
                raise ValueError("Supplied IDs must be strictly unique.")
            conflicts = [int(i) for i in ids if int(i) in existing_ids]
            if conflicts:
                raise ValueError(f"ID conflict: ID {conflicts[0]} already exists in the index.")
            new_ids = np.ascontiguousarray(ids)

        c_dists = pairwise_squared_l2_blas(vectors, self._centroids)
        assigned_centroids = np.argmin(c_dists, axis=1)

        residuals = (
            vectors - self._centroids[assigned_centroids]
            if self._use_residuals
            else vectors
        )
        codes = self._pq.encode(residuals)

        for i in range(N_new):
            c = int(assigned_centroids[i])
            lst = self._invlists[c]
            new_code = codes[i : i + 1]
            new_id = new_ids[i : i + 1]
            if lst.size == 0:
                self._invlists[c] = InvertedPQList(ids=new_id, codes=new_code)
            else:
                merged_codes = np.concatenate([lst.codes, new_code], axis=0)
                merged_ids = np.concatenate([lst.ids, new_id], axis=0)
                self._invlists[c] = InvertedPQList(ids=merged_ids, codes=merged_codes)

        self._ntotal += N_new
        return new_ids

    def delete(self, ids: Sequence[int] | np.ndarray | int) -> int:
        """Delete vectors by external ID from inverted lists.

        Parameters
        ----------
        ids : Sequence[int] | np.ndarray | int
            External ID(s) to remove from the index.

        Returns
        -------
        int
            Number of vectors successfully deleted.
        """
        if isinstance(ids, (int, np.integer)):
            to_delete = {int(ids)}
        elif isinstance(ids, (set, frozenset)):
            to_delete = set(int(x) for x in ids)
        else:
            to_delete = set(int(x) for x in ids)

        if not to_delete or self._ntotal == 0:
            return 0

        total_deleted = 0
        code_dtype = self._invlists[0].codes.dtype if self._nlist > 0 else np.uint8
        for c in range(self._nlist):
            lst = self._invlists[c]
            if lst.size == 0:
                continue
            mask_keep = np.array([int(i) not in to_delete for i in lst.ids], dtype=bool)
            del_count = int(np.sum(~mask_keep))
            if del_count > 0:
                total_deleted += del_count
                new_codes = (
                    np.ascontiguousarray(lst.codes[mask_keep])
                    if np.any(mask_keep)
                    else np.empty((0, self._M), dtype=code_dtype)
                )
                new_ids = (
                    np.ascontiguousarray(lst.ids[mask_keep])
                    if np.any(mask_keep)
                    else np.empty((0,), dtype=np.int64)
                )
                self._invlists[c] = InvertedPQList(ids=new_ids, codes=new_codes)

        self._ntotal -= total_deleted
        return total_deleted

    def save(self, path: str | Path, overwrite: bool = True) -> dict[str, Any]:
        """Save this IVFPQIndex to disk atomically with manifest and checksums."""
        from indexora.persistence import save_index
        return save_index(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, path: str | Path, verify_checksums: bool = True) -> IVFPQIndex:
        """Load an IVFPQIndex from disk, verifying data integrity."""
        from indexora.persistence import load_index
        idx = load_index(path, verify_checksums=verify_checksums)
        if not isinstance(idx, cls):
            raise TypeError(f"Loaded index is of type {type(idx).__name__}, expected {cls.__name__}")
        return idx


def ivf_pq_build(
    database: np.ndarray,
    nlist: int,
    M: int,
    k_sub: int = 256,
    ids: np.ndarray | None = None,
    nprobe: int = 1,
    coarse_centroids: np.ndarray | None = None,
    pq: ProductQuantizer | None = None,
    seed: int | None = 42,
    max_iter: int = 100,
    tol: float = 1e-4,
    use_residuals: bool = True,
) -> IVFPQIndex:
    """Construct an IVFPQIndex from database vectors."""
    return IVFPQIndex(
        database=database,
        nlist=nlist,
        M=M,
        k_sub=k_sub,
        ids=ids,
        nprobe=nprobe,
        coarse_centroids=coarse_centroids,
        pq=pq,
        seed=seed,
        max_iter=max_iter,
        tol=tol,
        use_residuals=use_residuals,
    )


def ivf_pq_search(
    index: IVFPQIndex,
    queries: np.ndarray,
    k: int = 10,
    nprobe: int | None = None,
    filter_ids: Sequence[int] | set[int] | np.ndarray | Callable[[int], bool] | None = None,
) -> SearchResult:
    """Search an IVFPQIndex using query vectors."""
    if not isinstance(index, IVFPQIndex):
        raise TypeError(f"index must be an IVFPQIndex instance, got {type(index).__name__}")
    return index.search(queries=queries, k=k, nprobe=nprobe, filter_ids=filter_ids)
