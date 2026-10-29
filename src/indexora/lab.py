"""Interactive Index Lab backend services and helper functions.

Provides deterministic synthetic clustered dataset generation, 2D PCA projection,
ground-truth evaluation, multi-index benchmarking, and stateful mutation/persistence
session management for the Indexora Interactive Index Lab UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np

from indexora.eval import compute_recall_at_k
from indexora.exact import ExactBlasIndex, ExactFlatIndex, generate_ground_truth
from indexora.hnsw import HNSWIndex
from indexora.ivf import IVFFlatIndex
from indexora.ivfpq import IVFPQIndex
from indexora.persistence import load_index, save_index
from indexora.topk import SearchResult


@dataclass(frozen=True)
class LabDataset:
    """Deterministic synthetic vector dataset container with precomputed PCA projection.

    Parameters
    ----------
    base : np.ndarray
        Database vectors of shape `(N, d)` with strict `dtype=np.float32`.
    base_ids : np.ndarray
        Stable entity IDs of shape `(N,)` with strict `dtype=np.int64`.
    queries : np.ndarray
        Evaluation query vectors of shape `(Q, d)` with strict `dtype=np.float32`.
    query_ids : np.ndarray
        Query entity IDs of shape `(Q,)` with strict `dtype=np.int64`.
    cluster_labels : np.ndarray
        Ground-truth cluster assignments of shape `(N,)` with `dtype=np.int64`.
    centroids : np.ndarray
        Cluster center coordinates of shape `(C, d)` with `dtype=np.float32`.
    pca_mean : np.ndarray
        Centering mean vector of shape `(d,)` used for 2D PCA projection.
    pca_components : np.ndarray
        Projection matrix of shape `(d, 2)` representing top 2 principal components.
    seed : int
        RNG seed used for deterministic generation.
    """

    base: np.ndarray
    base_ids: np.ndarray
    queries: np.ndarray
    query_ids: np.ndarray
    cluster_labels: np.ndarray
    centroids: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    seed: int

    @property
    def N(self) -> int:
        """Number of base vectors."""
        return int(self.base.shape[0])

    @property
    def Q(self) -> int:
        """Number of query vectors."""
        return int(self.queries.shape[0])

    @property
    def d(self) -> int:
        """Vector dimensionality."""
        return int(self.base.shape[1])

    @property
    def n_clusters(self) -> int:
        """Number of clusters in the synthetic distribution."""
        return int(self.centroids.shape[0])

    def project_2d(self, vectors: np.ndarray) -> np.ndarray:
        """Project vectors from d-dimensional space to 2D using deterministic PCA.

        Parameters
        ----------
        vectors : np.ndarray
            1D array of shape `(d,)` or 2D array of shape `(M, d)`.

        Returns
        -------
        np.ndarray
            2D projection coordinates of shape `(M, 2)` or `(2,)`.
        """
        is_1d = vectors.ndim == 1
        v2d = vectors.reshape(1, -1) if is_1d else vectors
        centered = v2d - self.pca_mean
        proj = centered @ self.pca_components
        return proj[0] if is_1d else proj

    def get_ground_truth(self, k: int = 10) -> SearchResult:
        """Compute exact reference ground truth for all queries against the base database.

        Parameters
        ----------
        k : int, default=10
            Number of nearest neighbors to retrieve.

        Returns
        -------
        SearchResult
            Exact ground-truth result container.
        """
        return generate_ground_truth(
            queries=self.queries,
            database=self.base,
            ids=self.base_ids,
            k=k,
        )


def generate_lab_dataset(
    n_base: int = 150,
    n_queries: int = 8,
    d: int = 8,
    n_clusters: int = 4,
    cluster_std: float = 0.25,
    seed: int = 42,
    id_start: int = 1000,
) -> LabDataset:
    """Generate a reproducible, structured clustered synthetic vector dataset.

    Note: This produces synthetic numeric coordinates in R^d for algorithmic
    demonstration and visualization; it does NOT generate semantic text embeddings.

    Parameters
    ----------
    n_base : int, default=150
        Number of database base vectors (must be >= 1).
    n_queries : int, default=8
        Number of query vectors (must be >= 1).
    d : int, default=8
        Dimensionality of the vector space (must be >= 2).
    n_clusters : int, default=4
        Number of distinct cluster centers (must be >= 1).
    cluster_std : float, default=0.25
        Standard deviation of Gaussian cluster dispersion around centers.
    seed : int, default=42
        RNG seed for bit-for-bit reproducible data generation.
    id_start : int, default=1000
        Initial scalar value for stable sequential base entity IDs.

    Returns
    -------
    LabDataset
        Structured dataset container with precomputed deterministic PCA projection.
    """
    if n_base < 1:
        raise ValueError(f"n_base must be >= 1, got {n_base}")
    if n_queries < 1:
        raise ValueError(f"n_queries must be >= 1, got {n_queries}")
    if d < 2:
        raise ValueError(f"Dimensionality d must be >= 2, got {d}")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    if cluster_std <= 0.0:
        raise ValueError(f"cluster_std must be positive, got {cluster_std}")

    rng = np.random.default_rng(seed=seed)

    # 1. Sample well-separated cluster centers in [-4.0, 4.0]
    centroids = rng.uniform(low=-4.0, high=4.0, size=(n_clusters, d)).astype(np.float32)

    # 2. Assign base points evenly across clusters
    labels = np.arange(n_base, dtype=np.int64) % n_clusters
    rng.shuffle(labels)

    base_noise = rng.normal(loc=0.0, scale=cluster_std, size=(n_base, d)).astype(np.float32)
    base = np.ascontiguousarray(centroids[labels] + base_noise, dtype=np.float32)
    base_ids = np.arange(id_start, id_start + n_base, dtype=np.int64)

    # 3. Sample queries from clusters with slight perturbation
    q_labels = rng.integers(low=0, high=n_clusters, size=n_queries)
    q_noise = rng.normal(loc=0.0, scale=cluster_std * 1.2, size=(n_queries, d)).astype(np.float32)
    queries = np.ascontiguousarray(centroids[q_labels] + q_noise, dtype=np.float32)
    query_ids = np.arange(id_start + 10000, id_start + 10000 + n_queries, dtype=np.int64)

    # 4. Compute deterministic 2D PCA projection matrix via SVD on centered base vectors
    pca_mean = np.mean(base, axis=0).astype(np.float32)
    centered = base - pca_mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pca_components = np.ascontiguousarray(vt[:2].T, dtype=np.float32)  # shape (d, 2)

    return LabDataset(
        base=base,
        base_ids=base_ids,
        queries=queries,
        query_ids=query_ids,
        cluster_labels=labels,
        centroids=centroids,
        pca_mean=pca_mean,
        pca_components=pca_components,
        seed=seed,
    )


@dataclass(frozen=True)
class IndexBenchmarkMetrics:
    """Benchmark performance metrics for an index evaluated on the Lab dataset."""

    index_name: str
    build_time_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    qps: float
    recall_at_k: float
    distance_computations_per_query: float | None
    memory_kb: float
    memory_method: str


def measure_index_memory(index: Any) -> tuple[float, str]:
    """Measure the storage footprint of an index in kilobytes with documented methodology.

    Parameters
    ----------
    index : Any
        Constructed Indexora index instance.

    Returns
    -------
    tuple[float, str]
        Memory footprint in KiB, and description of the accounting methodology.
    """
    if isinstance(index, (ExactFlatIndex, ExactBlasIndex)):
        mem_bytes = index.database.nbytes + index.ids.nbytes
        return mem_bytes / 1024.0, "Raw float32 database + int64 IDs arrays"

    if isinstance(index, IVFFlatIndex):
        c_bytes = index.centroids.nbytes
        lst_bytes = sum(int(lst.vectors.nbytes + lst.ids.nbytes) for lst in index.invlists)
        return (c_bytes + lst_bytes) / 1024.0, "Centroids array + inverted lists vectors/IDs"

    if isinstance(index, IVFPQIndex):
        c_bytes = index.coarse_centroid_memory_bytes
        cb_bytes = index.codebook_memory_bytes
        codes_bytes = index.codes_memory_bytes
        ids_bytes = index.ids_memory_bytes
        return (c_bytes + cb_bytes + codes_bytes + ids_bytes) / 1024.0, "Centroids + PQ codebooks + quantized codes + IDs"

    if isinstance(index, HNSWIndex):
        return index.index_memory_bytes / 1024.0, "Vectors + int64 IDs + graph adjacency lists"

    return 0.0, "Unknown"


def evaluate_index_search(
    index: Any,
    index_name: str,
    queries: np.ndarray,
    ground_truth: SearchResult,
    k: int,
    build_time_ms: float = 0.0,
    search_kwargs: dict[str, Any] | None = None,
    warmup: int = 1,
    repetitions: int = 5,
) -> IndexBenchmarkMetrics:
    """Benchmark an index on query vectors, measuring latency percentiles, throughput, and recall.

    Parameters
    ----------
    index : Any
        Constructed index instance.
    index_name : str
        Human-readable name of the index configuration.
    queries : np.ndarray
        Array of query vectors of shape `(Q, d)`.
    ground_truth : SearchResult
        Exact reference ground-truth search results for the same queries.
    k : int
        Top-k nearest neighbors retrieved per query.
    build_time_ms : float, default=0.0
        Elapsed index build time in milliseconds.
    search_kwargs : dict[str, Any] | None, default=None
        Optional search keyword arguments (e.g. `nprobe`, `ef_search`).
    warmup : int, default=1
        Number of warm-up iterations before timing.
    repetitions : int, default=5
        Number of timing repetitions.

    Returns
    -------
    IndexBenchmarkMetrics
        Comprehensive benchmark performance summary.
    """
    kwargs = search_kwargs or {}
    Q = queries.shape[0]

    # Warmup
    for _ in range(warmup):
        _ = index.search(queries, k=k, **kwargs)

    # Timing individual queries to obtain reliable p50 and p95 latency percentiles
    latencies_ms: list[float] = []
    total_batch_time_s = 0.0

    for _ in range(repetitions):
        t0 = time.perf_counter()
        res_batch = index.search(queries, k=k, **kwargs)
        t1 = time.perf_counter()
        total_batch_time_s += (t1 - t0)

        # Single query timings for latency distribution
        for q_idx in range(Q):
            q_vec = queries[q_idx]
            t_start = time.perf_counter()
            _ = index.search(q_vec, k=k, **kwargs)
            t_end = time.perf_counter()
            latencies_ms.append((t_end - t_start) * 1000.0)

    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    total_queries_run = repetitions * Q
    qps = float(total_queries_run / max(total_batch_time_s, 1e-9))

    # Evaluate Recall@k against exact ground truth
    final_res = index.search(queries, k=k, **kwargs)
    recall = compute_recall_at_k(retrieved=final_res, ground_truth=ground_truth, k=k)

    # Distance computations
    dc_per_q = None
    if final_res.distance_computations is not None and Q > 0:
        dc_per_q = float(final_res.distance_computations / Q)

    # Memory accounting
    mem_kb, mem_method = measure_index_memory(index)

    return IndexBenchmarkMetrics(
        index_name=index_name,
        build_time_ms=build_time_ms,
        median_latency_ms=median_lat,
        p95_latency_ms=p95_lat,
        qps=qps,
        recall_at_k=recall,
        distance_computations_per_query=dc_per_q,
        memory_kb=mem_kb,
        memory_method=mem_method,
    )


class LabSession:
    """Stateful management session for mutations, filtering, and persistence in the Lab.

    Tracks active index modifications, keeping a strict boundary between initial
    source dataset and mutated runtime state.
    """

    def __init__(self, index: Any, index_name: str, d: int) -> None:
        self.active_index = index
        self.index_name = index_name
        self.d = d
        self.added_ids: list[int] = []
        self.deleted_ids: list[int] = []

    @property
    def active_count(self) -> int:
        """Return the number of active, searchable items in the index."""
        if hasattr(self.active_index, "n_active"):
            return int(self.active_index.n_active)
        if hasattr(self.active_index, "ntotal"):
            return int(self.active_index.ntotal)
        return len(self.active_index)

    def add_vector(self, vector: np.ndarray, ext_id: int | None = None) -> int:
        """Incrementally add a vector to the active index.

        Parameters
        ----------
        vector : np.ndarray
            Vector coordinates of shape `(d,)` or `(1, d)`.
        ext_id : int | None, default=None
            Unique external ID. If None, auto-generated sequentially.

        Returns
        -------
        int
            Assigned external ID.
        """
        vec = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        if vec.shape[1] != self.d:
            raise ValueError(f"Vector dim {vec.shape[1]} does not match index dim {self.d}")

        id_arr = None if ext_id is None else np.array([ext_id], dtype=np.int64)
        assigned_ids = self.active_index.add(vec, ids=id_arr)
        assigned_id = int(assigned_ids[0])
        self.added_ids.append(assigned_id)
        if assigned_id in self.deleted_ids:
            self.deleted_ids.remove(assigned_id)
        return assigned_id

    def delete_vector(self, ext_id: int) -> int:
        """Delete an external ID from the active index.

        Parameters
        ----------
        ext_id : int
            External ID to delete.

        Returns
        -------
        int
            Count of vectors deleted (1 if present, 0 if not).
        """
        del_count = self.active_index.delete([ext_id])
        if del_count > 0:
            self.deleted_ids.append(ext_id)
            if ext_id in self.added_ids:
                self.added_ids.remove(ext_id)
        return del_count

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        filter_ids: Sequence[int] | set[int] | np.ndarray | Callable[[int], bool] | None = None,
        **kwargs,
    ) -> SearchResult:
        """Search nearest neighbors in the active index.

        Parameters
        ----------
        query : np.ndarray
            Query vector of shape `(d,)`.
        k : int, default=5
            Number of nearest neighbors to retrieve.
        filter_ids : Sequence[int] | set[int] | np.ndarray | Callable[[int], bool] | None, default=None
            Optional ID eligibility filter.

        Returns
        -------
        SearchResult
            Search result container.
        """
        return self.active_index.search(query, k=k, filter_ids=filter_ids, **kwargs)

    def save(self, target_dir: str | Path, overwrite: bool = True) -> dict[str, Any]:
        """Atomically persist the active index to a target directory.

        Parameters
        ----------
        target_dir : str | Path
            Directory path where index manifest and arrays will be written.
        overwrite : bool, default=True
            Whether to overwrite existing directory contents.

        Returns
        -------
        dict[str, Any]
            The saved manifest dictionary.
        """
        p = Path(target_dir)
        return save_index(self.active_index, p, overwrite=overwrite)

    def load(self, source_dir: str | Path, verify_checksums: bool = True) -> Any:
        """Load an index from disk into this session, verifying data integrity.

        Parameters
        ----------
        source_dir : str | Path
            Directory path containing manifest.json and array files.
        verify_checksums : bool, default=True
            Whether to verify SHA-256 checksums of all array files.

        Returns
        -------
        Any
            The restored index instance.
        """
        p = Path(source_dir)
        loaded = load_index(p, verify_checksums=verify_checksums)
        self.active_index = loaded
        self.index_name = type(loaded).__name__
        self.d = loaded.d
        return loaded
