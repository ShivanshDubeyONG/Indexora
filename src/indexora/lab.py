"""Interactive Index Lab backend services and helper functions.

Provides deterministic synthetic clustered dataset generation, 2D PCA projection,
ground-truth evaluation, multi-index benchmarking, and stateful mutation/persistence
session management for the Indexora Interactive Index Lab UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import re
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
    dataset_id : str, default=""
        Unique identifier for dataset instance used for cache invalidation.
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
    dataset_id: str = ""

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

    def get_ground_truth(
        self,
        k: int = 10,
        filter_ids: Sequence[int] | set[int] | np.ndarray | Callable[[int], bool] | None = None,
        query: np.ndarray | None = None,
    ) -> SearchResult:
        """Compute exact reference ground truth over eligible vectors.

        Parameters
        ----------
        k : int, default=10
            Number of nearest neighbors to retrieve.
        filter_ids : Sequence[int] | set[int] | np.ndarray | Callable[[int], bool] | None, default=None
            Optional ID eligibility filter restricting candidate vectors.
        query : np.ndarray | None, default=None
            Specific query vector of shape `(d,)` or batch `(Q, d)`. If None, evaluates `self.queries`.

        Returns
        -------
        SearchResult
            Exact ground-truth result container over eligible vectors.
        """
        q = self.queries if query is None else query
        return compute_exact_ground_truth(
            database=self.base,
            ids=self.base_ids,
            query=q,
            k=k,
            filter_ids=filter_ids,
        )


def compute_exact_ground_truth(
    database: np.ndarray,
    ids: np.ndarray,
    query: np.ndarray,
    k: int = 10,
    filter_ids: Sequence[int] | set[int] | np.ndarray | Callable[[int], bool] | None = None,
) -> SearchResult:
    """Compute exact reference ground truth for query vector(s) over eligible live vectors.

    Uses Indexora's Exact BLAS index to perform exhaustive squared-L2 scanning
    with deterministic ascending (distance, external_id) tie-breaking. Correctly
    evaluates callable, set, or sequence eligibility filters.

    Parameters
    ----------
    database : np.ndarray
        Database vectors array of shape `(N, d)` with strict `dtype=np.float32`.
    ids : np.ndarray
        Array of external entity IDs of shape `(N,)` with strict `dtype=np.int64`.
    query : np.ndarray
        Query vector of shape `(d,)` or batch of shape `(Q, d)` with strict `dtype=np.float32`.
    k : int, default=10
        Number of nearest neighbors to retrieve.
    filter_ids : Sequence[int] | set[int] | np.ndarray | Callable[[int], bool] | None, default=None
        Eligibility filter restricting candidates to matching external IDs.

    Returns
    -------
    SearchResult
        Ground-truth results containing exact top-k matches over eligible vectors.
    """
    exact_idx = ExactBlasIndex(database=database, ids=ids)
    return exact_idx.search(query, k=k, filter_ids=filter_ids)


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

    # Unique dataset instance identifier for cache invalidation
    dataset_id = f"ds_s{seed}_N{n_base}_d{d}_c{n_clusters}_std{cluster_std}_{time.time_ns()}"

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
        dataset_id=dataset_id,
    )


class IndexCache:
    """In-memory cache for built Indexora index instances.

    Prevents expensive index rebuilds (k-means clustering in IVF and graph construction
    in HNSW) during UI rerenders when the dataset and build parameters remain unchanged.
    Invalidates automatically when build parameters or the dataset change.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, tuple[tuple[str, Any], ...]], tuple[Any, float]] = {}

    def _make_key(
        self,
        index_type: str,
        dataset_id: str,
        build_params: dict[str, Any],
    ) -> tuple[str, str, tuple[tuple[str, Any], ...]]:
        sorted_params = tuple(sorted((k, v) for k, v in build_params.items()))
        return (index_type, dataset_id, sorted_params)

    def get(
        self,
        index_type: str,
        dataset_id: str,
        build_params: dict[str, Any],
    ) -> tuple[Any, float] | None:
        """Retrieve a cached (index_instance, build_time_ms) tuple if present."""
        key = self._make_key(index_type, dataset_id, build_params)
        return self._cache.get(key)

    def put(
        self,
        index_type: str,
        dataset_id: str,
        build_params: dict[str, Any],
        index: Any,
        build_time_ms: float,
    ) -> None:
        """Store an index instance and its build time in the cache."""
        key = self._make_key(index_type, dataset_id, build_params)
        self._cache[key] = (index, build_time_ms)

    def get_or_build(
        self,
        index_type: str,
        dataset: LabDataset,
        build_params: dict[str, Any],
        builder_fn: Callable[[], Any],
    ) -> tuple[Any, float, bool]:
        """Get an existing cached index or build a new one using builder_fn.

        Parameters
        ----------
        index_type : str
            Name of index architecture (e.g. 'IVF-Flat', 'HNSW').
        dataset : LabDataset
            Dataset instance containing dataset_id.
        build_params : dict[str, Any]
            Dictionary of build-time hyperparameters (excluding query-time parameters).
        builder_fn : Callable[[], Any]
            Zero-argument factory creating the index when not cached.

        Returns
        -------
        tuple[Any, float, bool]
            `(index_instance, build_time_ms, was_cached)`
        """
        cached = self.get(index_type, dataset.dataset_id, build_params)
        if cached is not None:
            return cached[0], cached[1], True

        t0 = time.perf_counter()
        idx = builder_fn()
        build_time_ms = (time.perf_counter() - t0) * 1000.0
        self.put(index_type, dataset.dataset_id, build_params, idx, build_time_ms)
        return idx, build_time_ms, False

    def clear(self) -> None:
        """Invalidate all cached indexes."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


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


DEFAULT_LAB_STORAGE_ROOT = Path("storage/lab_indexes")
SAFE_INDEX_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def sanitize_index_name(name: str) -> str:
    """Validate and sanitize an index name for public/sandboxed persistence.

    Parameters
    ----------
    name : str
        Identifier for the index.

    Returns
    -------
    str
        Validated index name.

    Raises
    ------
    ValueError
        If name is empty, contains path traversal characters, exceeds 64 characters,
        or contains characters outside alphanumeric, underscore, and hyphen.
    """
    if not isinstance(name, str):
        raise TypeError(f"Index name must be a string, got {type(name).__name__}")
    clean = name.strip()
    if not clean:
        raise ValueError("Index name cannot be empty or purely whitespace.")
    if len(clean) > 64:
        raise ValueError(f"Index name exceeds maximum length of 64 characters (got {len(clean)}).")
    if ".." in clean or "/" in clean or "\\" in clean or ":" in clean or "\x00" in clean:
        raise ValueError(f"Path traversal characters detected in index name: '{name}'")
    if not SAFE_INDEX_NAME_PATTERN.match(clean):
        raise ValueError(
            f"Invalid index name: '{name}'. Only alphanumeric characters, hyphens, "
            "and underscores are permitted (1-64 chars)."
        )
    return clean


def resolve_safe_storage_path(
    index_name: str,
    storage_root: str | Path = DEFAULT_LAB_STORAGE_ROOT,
) -> Path:
    """Resolve an index directory path within a restricted storage root.

    Enforces:
    1. Strict index name validation (no path traversal, alphanumeric/hyphen/underscore).
    2. Strict root containment: resolved path must be relative to resolved storage_root.
    3. Target directory cannot be the storage_root itself.
    4. Target directory cannot be an external symlink.

    Parameters
    ----------
    index_name : str
        Index identifier.
    storage_root : str | Path, default='storage/lab_indexes'
        Restricted base directory.

    Returns
    -------
    Path
        Resolved safe directory Path within storage_root.

    Raises
    ------
    ValueError
        If name is invalid, attempts traversal, or escapes storage_root.
    """
    clean_name = sanitize_index_name(index_name)
    root = Path(storage_root).resolve()
    target = (root / clean_name).resolve()

    if not target.is_relative_to(root):
        raise ValueError(f"Path traversal detected: '{index_name}' escapes storage root '{root}'")
    if target == root:
        raise ValueError("Index path cannot be the root storage directory itself.")
    if target.is_symlink():
        real_target = target.resolve()
        if not real_target.is_relative_to(root):
            raise ValueError(f"Symlink escape detected: '{index_name}' points outside storage root.")

    return target


class LabSession:
    """Stateful management session for mutations, filtering, and persistence in the Lab.

    Tracks active index modifications, keeping a strict boundary between initial
    source dataset and mutated runtime state. Enforces sandboxed persistence within
    a dedicated storage root for public deployments.
    """

    def __init__(
        self,
        index: Any,
        index_name: str,
        d: int,
        storage_root: str | Path = DEFAULT_LAB_STORAGE_ROOT,
        strict_sandbox: bool = False,
    ) -> None:
        self.active_index = index
        self.index_name = index_name
        self.d = d
        self.storage_root = Path(storage_root).resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.strict_sandbox = strict_sandbox
        self.added_ids: list[int] = []
        self.deleted_ids: list[int] = []

    def save_named(self, name: str, overwrite: bool = True) -> dict[str, Any]:
        """Safely persist active index inside storage_root using a validated index name."""
        safe_path = resolve_safe_storage_path(name, self.storage_root)
        return save_index(self.active_index, safe_path, overwrite=overwrite)

    def load_named(self, name: str, verify_checksums: bool = True) -> Any:
        """Safely load an index from storage_root using a validated index name."""
        safe_path = resolve_safe_storage_path(name, self.storage_root)
        loaded = load_index(safe_path, verify_checksums=verify_checksums)
        self.active_index = loaded
        self.index_name = type(loaded).__name__
        self.d = loaded.d
        self.added_ids = []
        self.deleted_ids = []
        return loaded

    def list_saved(self) -> list[str]:
        """List valid saved index names present in storage_root."""
        if not self.storage_root.exists():
            return []
        names: list[str] = []
        for item in self.storage_root.iterdir():
            if item.is_dir() and (item / "manifest.json").exists():
                try:
                    clean = sanitize_index_name(item.name)
                    names.append(clean)
                except ValueError:
                    continue
        return sorted(names)

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
        if assigned_id not in self.added_ids:
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
            if ext_id not in self.deleted_ids:
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
            Directory path or sanitized index identifier.
        overwrite : bool, default=True
            Whether to overwrite existing directory contents.

        Returns
        -------
        dict[str, Any]
            The saved manifest dictionary.
        """
        if self.strict_sandbox:
            p = resolve_safe_storage_path(str(target_dir), self.storage_root)
        else:
            p = Path(target_dir)
            if not p.is_absolute() and len(p.parts) == 1 and not (".." in str(target_dir)):
                try:
                    p = resolve_safe_storage_path(str(target_dir), self.storage_root)
                except ValueError:
                    p = Path(target_dir)
        return save_index(self.active_index, p, overwrite=overwrite)

    def load(self, source_dir: str | Path, verify_checksums: bool = True) -> Any:
        """Load an index from disk into this session, verifying data integrity.

        Parameters
        ----------
        source_dir : str | Path
            Directory path or sanitized index identifier.
        verify_checksums : bool, default=True
            Whether to verify SHA-256 checksums of all array files.

        Returns
        -------
        Any
            The restored index instance.
        """
        if self.strict_sandbox:
            p = resolve_safe_storage_path(str(source_dir), self.storage_root)
        else:
            p = Path(source_dir)
            if not p.is_absolute() and len(p.parts) == 1 and not (".." in str(source_dir)):
                try:
                    candidate = resolve_safe_storage_path(str(source_dir), self.storage_root)
                    if candidate.exists():
                        p = candidate
                except ValueError:
                    p = Path(source_dir)
        loaded = load_index(p, verify_checksums=verify_checksums)
        self.active_index = loaded
        self.index_name = type(loaded).__name__
        self.d = loaded.d
        self.added_ids = []
        self.deleted_ids = []
        return loaded
