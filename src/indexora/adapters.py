"""Optional reference adapters for established vector search libraries (Phase 7).

This module provides normalized adapter wrappers around third-party vector search
libraries to enable apples-to-apples comparisons using Indexora's standard evaluation
contracts:
- `FaissFlatIndex`: Exact brute-force scan via FAISS `IndexFlatL2`.
- `FaissIVFFlatIndex`: Approximate inverted-file scan via FAISS `IndexIVFFlat`.
- `HnswlibIndex`: Approximate graph search via `hnswlib.Index`.

Both libraries are completely optional:
- If a library is not installed, importing `indexora` still succeeds.
- Instantiating an adapter when its underlying dependency is missing raises an
  informative `ImportError` with clear installation instructions.
- All adapters return Indexora's standard frozen `SearchResult` container.
- All adapters respect Indexora's canonical squared-L2 distance metric and int64 IDs.
"""

from __future__ import annotations

import sys
from typing import Any, Sequence

import numpy as np

from indexora.topk import SearchResult


def is_faiss_available() -> bool:
    """Check if the FAISS library is installed and importable."""
    try:
        import faiss  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


def is_hnswlib_available() -> bool:
    """Check if the hnswlib library is installed and importable."""
    try:
        import hnswlib  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


def _require_faiss() -> Any:
    """Import and return the faiss module, or raise a clean ImportError."""
    try:
        import faiss

        return faiss
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "FAISS is not installed. Install it via 'pip install faiss-cpu' "
            "(or 'pip install indexora[adapters]')."
        ) from exc


def _require_hnswlib() -> Any:
    """Import and return the hnswlib module, or raise a clean ImportError."""
    try:
        import hnswlib

        return hnswlib
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "hnswlib is not installed. Install it via 'pip install hnswlib'. "
            "Note: On Windows, Microsoft C++ Build Tools may be required to build wheels."
        ) from exc


def _validate_adapter_inputs(
    database: np.ndarray,
    ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate database and identifiers according to Indexora contracts."""
    if not isinstance(database, np.ndarray):
        raise TypeError(f"database must be a numpy ndarray, got {type(database).__name__}")
    if database.dtype != np.float32:
        raise TypeError(f"database must have dtype float32, got {database.dtype}")
    if database.ndim != 2:
        raise ValueError(f"database must be 2D array of shape (N, d), got ndim={database.ndim}")
    if not np.all(np.isfinite(database)):
        raise ValueError("database vectors must contain only finite numbers (no NaN or Inf).")

    N, _ = database.shape
    if ids is None:
        ids_arr = np.arange(N, dtype=np.int64)
    else:
        if not isinstance(ids, np.ndarray):
            raise TypeError(f"ids must be a numpy ndarray, got {type(ids).__name__}")
        if ids.dtype != np.int64:
            raise TypeError(f"ids must have dtype int64, got {ids.dtype}")
        if ids.shape != (N,):
            raise ValueError(f"ids must be 1D array of length N={N}, got shape {ids.shape}")
        if len(np.unique(ids)) != N:
            raise ValueError("ids must be strictly unique among live records.")
        ids_arr = np.ascontiguousarray(ids, dtype=np.int64)

    return np.ascontiguousarray(database, dtype=np.float32), ids_arr


def _validate_query_input(queries: np.ndarray, expected_d: int) -> tuple[np.ndarray, bool]:
    """Validate query vector(s) shape and dtype."""
    if not isinstance(queries, np.ndarray):
        raise TypeError(f"queries must be a numpy ndarray, got {type(queries).__name__}")
    if queries.dtype != np.float32:
        raise TypeError(f"queries must have dtype float32, got {queries.dtype}")
    if not np.all(np.isfinite(queries)):
        raise ValueError("queries must contain only finite numbers (no NaN or Inf).")

    is_single = queries.ndim == 1
    if is_single:
        if queries.shape[0] != expected_d:
            raise ValueError(f"Query dimension {queries.shape[0]} does not match database dimension {expected_d}")
        q_2d = queries[np.newaxis, :]
    elif queries.ndim == 2:
        if queries.shape[1] != expected_d:
            raise ValueError(f"Query dimension {queries.shape[1]} does not match database dimension {expected_d}")
        q_2d = queries
    else:
        raise ValueError(f"queries must be 1D or 2D array, got ndim={queries.ndim}")

    return np.ascontiguousarray(q_2d, dtype=np.float32), is_single


class FaissFlatIndex:
    """FAISS exact flat search adapter wrapping `faiss.IndexFlatL2` and `IndexIDMap2`."""

    def __init__(
        self,
        database: np.ndarray,
        ids: np.ndarray | None = None,
    ) -> None:
        faiss = _require_faiss()
        self._database, self._ids = _validate_adapter_inputs(database, ids)
        self._N, self._d = self._database.shape

        raw_index = faiss.IndexFlatL2(self._d)
        self._index = faiss.IndexIDMap2(raw_index)
        self._index.add_with_ids(self._database, self._ids)

    @property
    def database(self) -> np.ndarray:
        """Contiguous database vectors of shape (N, d)."""
        return self._database

    @property
    def ids(self) -> np.ndarray:
        """Durable int64 identifiers of shape (N,)."""
        return self._ids

    @property
    def d(self) -> int:
        """Vector feature dimension."""
        return self._d

    @property
    def ntotal(self) -> int:
        """Total vectors indexed."""
        return self._N

    @property
    def index_memory_bytes(self) -> int:
        """Estimated index storage bytes: N * (d * 4 + 8)."""
        return self._N * (self._d * 4 + 8)

    @property
    def index_memory_mib(self) -> float:
        """Index storage footprint in binary MiB."""
        return float(self.index_memory_bytes / (1024 * 1024))

    @property
    def bytes_per_vector(self) -> float:
        """Index bytes per vector."""
        return float(self._d * 4 + 8)

    def search(self, queries: np.ndarray, k: int = 10) -> SearchResult:
        """Execute exhaustive exact flat search using FAISS.

        Parameters
        ----------
        queries : np.ndarray
            Query vector of shape `(d,)` or batch of shape `(Q, d)`.
        k : int, default=10
            Number of nearest neighbors to retrieve.

        Returns
        -------
        SearchResult
            Standard Indexora search container with distances, IDs, and validity mask.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        q_2d, is_single = _validate_query_input(queries, self._d)
        num_queries = q_2d.shape[0]

        if self._N == 0:
            empty_d = np.empty((0,) if is_single else (num_queries, 0), dtype=np.float32)
            empty_i = np.empty((0,) if is_single else (num_queries, 0), dtype=np.int64)
            empty_m = np.empty((0,) if is_single else (num_queries, 0), dtype=bool)
            return SearchResult(distances=empty_d, ids=empty_i, mask=empty_m, distance_computations=0)

        eff_k = min(k, self._N)
        raw_dists, raw_ids = self._index.search(q_2d, eff_k)

        # Convert to standard types and mask
        dists = raw_dists.astype(np.float32)
        ids = raw_ids.astype(np.int64)
        mask = dists < 1e30

        total_dc = num_queries * self._N

        if is_single:
            return SearchResult(
                distances=dists[0],
                ids=ids[0],
                mask=mask[0],
                distance_computations=total_dc,
                centroid_distance_computations=0,
                candidate_distance_computations=total_dc,
            )

        return SearchResult(
            distances=dists,
            ids=ids,
            mask=mask,
            distance_computations=total_dc,
            centroid_distance_computations=0,
            candidate_distance_computations=total_dc,
        )


class FaissIVFFlatIndex:
    """FAISS IVF-Flat approximate search adapter wrapping `faiss.IndexIVFFlat`."""

    def __init__(
        self,
        database: np.ndarray,
        nlist: int = 100,
        ids: np.ndarray | None = None,
        nprobe: int = 1,
        seed: int = 42,
    ) -> None:
        faiss = _require_faiss()
        self._database, self._ids = _validate_adapter_inputs(database, ids)
        self._N, self._d = self._database.shape
        self._nlist = int(nlist)
        self._nprobe = int(nprobe)

        if self._nlist <= 0:
            raise ValueError(f"nlist must be positive, got {nlist}")
        if self._nlist > self._N:
            raise ValueError(f"nlist={nlist} cannot exceed database size N={self._N}")
        if self._nprobe <= 0:
            raise ValueError(f"nprobe must be positive, got {nprobe}")

        quantizer = faiss.IndexFlatL2(self._d)
        ivf = faiss.IndexIVFFlat(quantizer, self._d, self._nlist, faiss.METRIC_L2)
        ivf.cp.seed = seed
        self._index = faiss.IndexIDMap2(ivf)
        self._index.train(self._database)
        self._index.add_with_ids(self._database, self._ids)
        self.set_nprobe(self._nprobe)

    @property
    def database(self) -> np.ndarray:
        """Contiguous database vectors of shape (N, d)."""
        return self._database

    @property
    def ids(self) -> np.ndarray:
        """Durable int64 identifiers of shape (N,)."""
        return self._ids

    @property
    def nlist(self) -> int:
        """Number of Voronoi coarse clusters."""
        return self._nlist

    @property
    def nprobe(self) -> int:
        """Default number of clusters probed during search."""
        return self._nprobe

    def set_nprobe(self, nprobe: int) -> None:
        """Update active nprobe parameter."""
        if nprobe <= 0:
            raise ValueError(f"nprobe must be positive, got {nprobe}")
        self._nprobe = int(nprobe)
        faiss = _require_faiss()
        faiss.ParameterSpace().set_index_parameter(self._index, "nprobe", self._nprobe)

    @property
    def index_memory_bytes(self) -> int:
        """Estimated index storage bytes: nlist * d * 4 + N * (d * 4 + 8)."""
        return self._nlist * self._d * 4 + self._N * (self._d * 4 + 8)

    @property
    def index_memory_mib(self) -> float:
        """Index storage footprint in binary MiB."""
        return float(self.index_memory_bytes / (1024 * 1024))

    @property
    def bytes_per_vector(self) -> float:
        """Index bytes per vector."""
        return float(self.index_memory_bytes / self._N) if self._N > 0 else 0.0

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        nprobe: int | None = None,
    ) -> SearchResult:
        """Execute IVF-Flat multi-probe approximate nearest neighbor search via FAISS.

        Parameters
        ----------
        queries : np.ndarray
            Query vector of shape `(d,)` or batch of shape `(Q, d)`.
        k : int, default=10
            Number of nearest neighbors to retrieve.
        nprobe : int | None, default=None
            Number of clusters to probe. If None, uses default index nprobe.

        Returns
        -------
        SearchResult
            Standard Indexora search container with distances, IDs, and validity mask.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        if nprobe is not None:
            self.set_nprobe(nprobe)

        q_2d, is_single = _validate_query_input(queries, self._d)
        num_queries = q_2d.shape[0]

        if self._N == 0:
            empty_d = np.empty((0,) if is_single else (num_queries, 0), dtype=np.float32)
            empty_i = np.empty((0,) if is_single else (num_queries, 0), dtype=np.int64)
            empty_m = np.empty((0,) if is_single else (num_queries, 0), dtype=bool)
            return SearchResult(distances=empty_d, ids=empty_i, mask=empty_m, distance_computations=0)

        eff_k = min(k, self._N)
        raw_dists, raw_ids = self._index.search(q_2d, eff_k)

        dists = raw_dists.astype(np.float32)
        ids = raw_ids.astype(np.int64)
        mask = dists < 1e30

        if is_single:
            return SearchResult(
                distances=dists[0],
                ids=ids[0],
                mask=mask[0],
            )

        return SearchResult(
            distances=dists,
            ids=ids,
            mask=mask,
        )


class HnswlibIndex:
    """hnswlib approximate nearest neighbor graph search adapter."""

    def __init__(
        self,
        database: np.ndarray,
        ids: np.ndarray | None = None,
        space: str = "l2",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        random_seed: int = 42,
    ) -> None:
        hnswlib = _require_hnswlib()
        self._database, self._ids = _validate_adapter_inputs(database, ids)
        self._N, self._d = self._database.shape
        self._space = space
        self._M = int(M)
        self._ef_construction = int(ef_construction)
        self._ef_search = int(ef_search)

        if self._M <= 0:
            raise ValueError(f"M must be positive, got {M}")
        if self._ef_construction <= 0:
            raise ValueError(f"ef_construction must be positive, got {ef_construction}")
        if self._ef_search <= 0:
            raise ValueError(f"ef_search must be positive, got {ef_search}")

        # hnswlib expects integer labels 0..N-1; we map them to arbitrary int64 IDs
        self._label_to_id = self._ids.copy()

        self._index = hnswlib.Index(space=self._space, dim=self._d)
        self._index.init_index(
            max_elements=max(self._N, 1),
            ef_construction=self._ef_construction,
            M=self._M,
            random_seed=random_seed,
        )
        if self._N > 0:
            internal_labels = np.arange(self._N, dtype=np.uint64)
            self._index.add_items(self._database, internal_labels)
        self.set_ef_search(self._ef_search)

    @property
    def database(self) -> np.ndarray:
        """Contiguous database vectors of shape (N, d)."""
        return self._database

    @property
    def ids(self) -> np.ndarray:
        """Durable int64 identifiers of shape (N,)."""
        return self._ids

    @property
    def M(self) -> int:
        """Maximum number of outgoing links per node in graph."""
        return self._M

    @property
    def ef_construction(self) -> int:
        """Size of candidate dynamic list during index construction."""
        return self._ef_construction

    @property
    def ef_search(self) -> int:
        """Size of candidate dynamic list during query search."""
        return self._ef_search

    def set_ef_search(self, ef_search: int) -> None:
        """Update active ef_search parameter."""
        if ef_search <= 0:
            raise ValueError(f"ef_search must be positive, got {ef_search}")
        self._ef_search = int(ef_search)
        self._index.set_ef(self._ef_search)

    @property
    def index_memory_bytes(self) -> int:
        """Estimated index memory footprint: vectors + links."""
        return self._N * (self._d * 4 + self._M * 8)

    @property
    def index_memory_mib(self) -> float:
        """Estimated index footprint in binary MiB."""
        return float(self.index_memory_bytes / (1024 * 1024))

    @property
    def bytes_per_vector(self) -> float:
        """Index bytes per vector."""
        return float(self.index_memory_bytes / self._N) if self._N > 0 else 0.0

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        ef_search: int | None = None,
    ) -> SearchResult:
        """Execute approximate nearest neighbor search via hnswlib.

        Parameters
        ----------
        queries : np.ndarray
            Query vector of shape `(d,)` or batch of shape `(Q, d)`.
        k : int, default=10
            Number of nearest neighbors to retrieve.
        ef_search : int | None, default=None
            Dynamic candidate list size during search.

        Returns
        -------
        SearchResult
            Standard Indexora search container with distances, mapped IDs, and validity mask.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        if ef_search is not None:
            self.set_ef_search(ef_search)

        q_2d, is_single = _validate_query_input(queries, self._d)
        num_queries = q_2d.shape[0]

        if self._N == 0:
            empty_d = np.empty((0,) if is_single else (num_queries, 0), dtype=np.float32)
            empty_i = np.empty((0,) if is_single else (num_queries, 0), dtype=np.int64)
            empty_m = np.empty((0,) if is_single else (num_queries, 0), dtype=bool)
            return SearchResult(distances=empty_d, ids=empty_i, mask=empty_m, distance_computations=0)

        eff_k = min(k, self._N)
        raw_labels, raw_dists = self._index.knn_query(q_2d, k=eff_k)

        # Map internal labels back to original durable int64 IDs
        mapped_ids = np.empty_like(raw_labels, dtype=np.int64)
        for i in range(num_queries):
            for j in range(eff_k):
                lbl = raw_labels[i, j]
                mapped_ids[i, j] = self._label_to_id[lbl]

        dists = raw_dists.astype(np.float32)
        mask = np.ones_like(mapped_ids, dtype=bool)

        if is_single:
            return SearchResult(
                distances=dists[0],
                ids=mapped_ids[0],
                mask=mask[0],
            )

        return SearchResult(
            distances=dists,
            ids=mapped_ids,
            mask=mask,
        )
