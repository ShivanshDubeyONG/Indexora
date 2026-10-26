"""Dataset utilities, validation, synthetic generation, and caching for Indexora.

This module provides the core data layer for Indexora (Phase 3). It defines the
structured `VectorDataset` container for managing base, query, and training splits
with stable int64 IDs, reproducible clustered/synthetic dataset generators,
strict L2 normalization with a documented zero-vector policy, and offline
checksummed cache utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np


def _validate_vector_array(
    arr: np.ndarray,
    name: str,
    expected_dim: int | None = None,
) -> int:
    """Validate that a vector array is a 2D, C-contiguous, finite float32 array."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(arr).__name__}")
    if arr.dtype != np.float32:
        raise TypeError(
            f"{name} must have dtype float32, got {arr.dtype}. Silent casting is not permitted."
        )
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (N, d), got ndim={arr.ndim} with shape {arr.shape}")
    if not arr.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous in memory.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numbers (no NaN or Inf).")

    dim = arr.shape[1]
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(
            f"Dimension mismatch in {name}: expected dimensionality {expected_dim}, got {dim}"
        )
    return dim


def _validate_id_array(ids: np.ndarray, name: str, expected_n: int) -> None:
    """Validate that an ID array is a 1D int64 array with unique values matching vector count."""
    if not isinstance(ids, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(ids).__name__}")
    if ids.dtype != np.int64:
        raise TypeError(
            f"{name} must have dtype int64, got {ids.dtype}. Silent casting is not permitted."
        )
    if ids.ndim != 1:
        raise ValueError(f"{name} must be a 1D array of shape (N,), got ndim={ids.ndim} with shape {ids.shape}")
    if ids.shape[0] != expected_n:
        raise ValueError(
            f"Length mismatch: {name} length {ids.shape[0]} does not match vector count {expected_n}"
        )
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} must contain strictly unique values.")


@dataclass(frozen=True)
class VectorDataset:
    """Container for vector dataset splits and durable entity IDs.

    Represents database base vectors to index/search, evaluation query vectors,
    and optional training vectors (e.g. for k-means or quantization fitting).

    Parameters
    ----------
    base : np.ndarray
        Database vectors to index of shape `(N, d)`, strict C-contiguous `float32`.
    queries : np.ndarray
        Evaluation query vectors of shape `(Q, d)`, strict C-contiguous `float32`.
    base_ids : np.ndarray | None, default=None
        Durable identifiers of shape `(N,)`, strict `int64`. If None, defaults to `0..N-1`.
    query_ids : np.ndarray | None, default=None
        Durable identifiers of shape `(Q,)`, strict `int64`. If None, defaults to `0..Q-1`.
    train : np.ndarray | None, default=None
        Optional training vectors of shape `(T, d)`, strict C-contiguous `float32`.
    train_ids : np.ndarray | None, default=None
        Durable identifiers of shape `(T,)`, strict `int64`. If None, defaults to `0..T-1`.
    """

    base: np.ndarray
    queries: np.ndarray
    base_ids: np.ndarray
    query_ids: np.ndarray
    train: np.ndarray | None = None
    train_ids: np.ndarray | None = None

    def __init__(
        self,
        base: np.ndarray,
        queries: np.ndarray,
        base_ids: np.ndarray | None = None,
        query_ids: np.ndarray | None = None,
        train: np.ndarray | None = None,
        train_ids: np.ndarray | None = None,
    ) -> None:
        dim = _validate_vector_array(base, name="base")
        _validate_vector_array(queries, name="queries", expected_dim=dim)

        # Initialize base IDs
        if base_ids is None:
            resolved_base_ids = np.arange(base.shape[0], dtype=np.int64)
        else:
            _validate_id_array(base_ids, name="base_ids", expected_n=base.shape[0])
            resolved_base_ids = base_ids

        # Initialize query IDs
        if query_ids is None:
            resolved_query_ids = np.arange(queries.shape[0], dtype=np.int64)
        else:
            _validate_id_array(query_ids, name="query_ids", expected_n=queries.shape[0])
            resolved_query_ids = query_ids

        # Initialize optional train split
        resolved_train: np.ndarray | None = None
        resolved_train_ids: np.ndarray | None = None

        if train is not None:
            _validate_vector_array(train, name="train", expected_dim=dim)
            resolved_train = train
            if train_ids is None:
                resolved_train_ids = np.arange(train.shape[0], dtype=np.int64)
            else:
                _validate_id_array(train_ids, name="train_ids", expected_n=train.shape[0])
                resolved_train_ids = train_ids
        elif train_ids is not None:
            raise ValueError("train_ids provided without corresponding train vectors.")

        object.__setattr__(self, "base", base)
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "base_ids", resolved_base_ids)
        object.__setattr__(self, "query_ids", resolved_query_ids)
        object.__setattr__(self, "train", resolved_train)
        object.__setattr__(self, "train_ids", resolved_train_ids)

    @property
    def d(self) -> int:
        """Feature dimensionality of the dataset vectors."""
        return self.base.shape[1]

    @property
    def n_base(self) -> int:
        """Number of base vectors in the database."""
        return self.base.shape[0]

    @property
    def n_queries(self) -> int:
        """Number of query vectors."""
        return self.queries.shape[0]

    @property
    def n_train(self) -> int:
        """Number of training vectors (0 if train split is None)."""
        return 0 if self.train is None else self.train.shape[0]


@dataclass(frozen=True)
class ClusteredDataset:
    """Clustered dataset output containing vector splits and ground-truth cluster assignments."""

    dataset: VectorDataset
    base_labels: np.ndarray
    centroids: np.ndarray

    def __iter__(self):
        """Allow tuple unpacking: dataset, labels, centroids = clustered_dataset."""
        yield self.dataset
        yield self.base_labels
        yield self.centroids


def generate_synthetic_dataset(
    n_base: int,
    n_queries: int,
    d: int,
    n_train: int = 0,
    seed: int = 42,
) -> VectorDataset:
    """Generate a reproducible synthetic dataset with uniform random vectors.

    Parameters
    ----------
    n_base : int
        Number of database base vectors to generate (n_base >= 0).
    n_queries : int
        Number of evaluation query vectors to generate (n_queries >= 0).
    d : int
        Feature dimensionality (d >= 1).
    n_train : int, default=0
        Number of optional training vectors (n_train >= 0).
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    VectorDataset
        Deterministic dataset with C-contiguous float32 arrays and unique int64 IDs.
    """
    if n_base < 0 or n_queries < 0 or n_train < 0:
        raise ValueError("Vector counts (n_base, n_queries, n_train) cannot be negative.")
    if d < 1:
        raise ValueError("Dimensionality d must be positive (>= 1).")

    rng = np.random.default_rng(seed=seed)

    base = np.ascontiguousarray(rng.standard_normal((n_base, d)), dtype=np.float32)
    queries = np.ascontiguousarray(rng.standard_normal((n_queries, d)), dtype=np.float32)

    train = None
    if n_train > 0:
        train = np.ascontiguousarray(rng.standard_normal((n_train, d)), dtype=np.float32)

    return VectorDataset(base=base, queries=queries, train=train)


def generate_clustered_dataset(
    n_base: int,
    n_queries: int,
    d: int,
    n_clusters: int,
    n_train: int = 0,
    cluster_std: float = 0.1,
    seed: int = 42,
) -> ClusteredDataset:
    """Generate reproducible clustered vectors with ground-truth centroids and labels.

    Useful for validating k-means clustering (Phase 4) and IVF partitioning (Phase 5).

    Parameters
    ----------
    n_base : int
        Number of database base vectors (n_base >= 1).
    n_queries : int
        Number of query vectors (n_queries >= 0).
    d : int
        Feature dimensionality (d >= 1).
    n_clusters : int
        Number of clusters to generate (n_clusters >= 1).
    n_train : int, default=0
        Number of optional training vectors.
    cluster_std : float, default=0.1
        Standard deviation of Gaussian noise around cluster centroids (must be > 0).
    seed : int, default=42
        Random seed for bit-for-bit reproducibility.

    Returns
    -------
    ClusteredDataset
        Container with `.dataset`, `.base_labels` (1D int64 array), and `.centroids` (float32).
    """
    if n_base < 1:
        raise ValueError("n_base must be at least 1.")
    if n_queries < 0 or n_train < 0:
        raise ValueError("n_queries and n_train cannot be negative.")
    if d < 1:
        raise ValueError("Dimensionality d must be positive (>= 1).")
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive (>= 1).")
    if cluster_std <= 0.0:
        raise ValueError("cluster_std must be positive (> 0.0).")

    rng = np.random.default_rng(seed=seed)

    # 1. Sample well-separated cluster centroids uniformly in [-5.0, 5.0]
    centroids = np.ascontiguousarray(
        rng.uniform(low=-5.0, high=5.0, size=(n_clusters, d)), dtype=np.float32
    )

    # 2. Assign base vectors evenly across clusters
    base_labels = np.arange(n_base, dtype=np.int64) % n_clusters
    rng.shuffle(base_labels)

    base_noise = rng.normal(loc=0.0, scale=cluster_std, size=(n_base, d))
    base = np.ascontiguousarray(centroids[base_labels] + base_noise, dtype=np.float32)

    # 3. Generate query vectors sampled from random centroids
    if n_queries > 0:
        q_labels = rng.integers(low=0, high=n_clusters, size=n_queries)
        q_noise = rng.normal(loc=0.0, scale=cluster_std, size=(n_queries, d))
        queries = np.ascontiguousarray(centroids[q_labels] + q_noise, dtype=np.float32)
    else:
        queries = np.empty((0, d), dtype=np.float32)

    # 4. Generate optional training vectors
    train = None
    if n_train > 0:
        t_labels = rng.integers(low=0, high=n_clusters, size=n_train)
        t_noise = rng.normal(loc=0.0, scale=cluster_std, size=(n_train, d))
        train = np.ascontiguousarray(centroids[t_labels] + t_noise, dtype=np.float32)

    dataset = VectorDataset(base=base, queries=queries, train=train)
    return ClusteredDataset(dataset=dataset, base_labels=base_labels, centroids=centroids)


def normalize_l2(
    vectors: np.ndarray,
    zero_vector_policy: str = "zero",
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize vectors to unit L2 length with an explicit zero-vector policy.

    Parameters
    ----------
    vectors : np.ndarray
        Array of shape `(d,)` or `(N, d)` with strict `dtype=np.float32`.
    zero_vector_policy : {"zero", "error"}, default="zero"
        Documented policy for handling zero-norm (or near-zero) vectors:
        - "zero": Vectors with norm <= eps are returned as all zeros (finite, avoids NaN).
        - "error": Raises `ValueError` when a vector with norm <= eps is encountered.
    eps : float, default=1e-12
        Tolerance threshold below which a vector's L2 norm is considered zero.

    Returns
    -------
    np.ndarray
        Unit-normalized C-contiguous array with strict `dtype=np.float32` matching input shape.

    Raises
    ------
    TypeError
        If vectors is not a float32 numpy ndarray (no silent casting).
    ValueError
        If vectors is not 1D or 2D, contains NaN/Inf, or violates zero_vector_policy="error".
    """
    if not isinstance(vectors, np.ndarray):
        raise TypeError(f"vectors must be a numpy ndarray, got {type(vectors).__name__}")
    if vectors.dtype != np.float32:
        raise TypeError(
            f"vectors must have dtype float32, got {vectors.dtype}. Silent casting is not permitted."
        )
    if vectors.ndim not in (1, 2):
        raise ValueError(f"vectors must be 1D or 2D, got ndim={vectors.ndim} with shape {vectors.shape}")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("vectors must contain only finite numbers (no NaN or Inf).")
    if zero_vector_policy not in ("zero", "error"):
        raise ValueError(
            f"Unknown zero_vector_policy: {zero_vector_policy!r}. Must be 'zero' or 'error'."
        )

    is_1d = vectors.ndim == 1
    v_2d = vectors[np.newaxis, :] if is_1d else vectors

    # Compute Euclidean norms in float64 for numerical precision
    norms = np.linalg.norm(v_2d.astype(np.float64), axis=-1, keepdims=True)
    is_zero = norms <= eps

    if np.any(is_zero):
        if zero_vector_policy == "error":
            raise ValueError("Zero-norm vector encountered under zero_vector_policy='error'.")

    # Safe division: replace zero norms with 1.0 to avoid division by zero warnings
    safe_norms = np.where(is_zero, 1.0, norms)
    normalized = (v_2d.astype(np.float64) / safe_norms)

    # Force exact 0.0 for zero-norm vectors
    normalized = np.where(is_zero, 0.0, normalized)
    out = np.ascontiguousarray(normalized.astype(np.float32))

    return out[0] if is_1d else out


def _compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hexadecimal digest of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_dataset(
    dataset: VectorDataset,
    directory: str | Path,
) -> dict[str, Any]:
    """Save a VectorDataset to disk with SHA-256 checksums and provenance metadata.

    Parameters
    ----------
    dataset : VectorDataset
        The dataset to persist.
    directory : str or Path
        Target directory to write numpy binary files and metadata.json.

    Returns
    -------
    dict
        Provenance and checksum manifest written to metadata.json.
    """
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    files_to_save = {
        "base.npy": dataset.base,
        "base_ids.npy": dataset.base_ids,
        "queries.npy": dataset.queries,
        "query_ids.npy": dataset.query_ids,
    }

    if dataset.train is not None and dataset.train_ids is not None:
        files_to_save["train.npy"] = dataset.train
        files_to_save["train_ids.npy"] = dataset.train_ids

    checksums: dict[str, str] = {}
    for filename, arr in files_to_save.items():
        filepath = target_dir / filename
        np.save(filepath, arr)
        checksums[filename] = _compute_sha256(filepath)

    manifest = {
        "dimension": dataset.d,
        "n_base": dataset.n_base,
        "n_queries": dataset.n_queries,
        "n_train": dataset.n_train,
        "checksums": checksums,
    }

    with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_dataset(
    directory: str | Path,
    verify_checksums: bool = True,
) -> VectorDataset:
    """Load a VectorDataset from disk, verifying SHA-256 integrity.

    Parameters
    ----------
    directory : str or Path
        Directory containing the saved dataset and metadata.json.
    verify_checksums : bool, default=True
        Whether to compute and verify SHA-256 checksums before loading.

    Returns
    -------
    VectorDataset
        Reconstructed dataset.

    Raises
    ------
    FileNotFoundError
        If metadata.json or required numpy files are missing.
    ValueError
        If checksum mismatch occurs (indicating cache corruption or tampering).
    """
    target_dir = Path(directory)
    metadata_path = target_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {target_dir}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    checksums = manifest.get("checksums", {})

    if verify_checksums:
        for filename, expected_hash in checksums.items():
            filepath = target_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Missing expected dataset file: {filepath}")
            actual_hash = _compute_sha256(filepath)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Checksum mismatch for {filename} (expected {expected_hash}, got {actual_hash}). "
                    "Cache file is corrupted."
                )

    base = np.load(target_dir / "base.npy")
    base_ids = np.load(target_dir / "base_ids.npy")
    queries = np.load(target_dir / "queries.npy")
    query_ids = np.load(target_dir / "query_ids.npy")

    train = None
    train_ids = None
    train_path = target_dir / "train.npy"
    train_ids_path = target_dir / "train_ids.npy"

    if train_path.exists() and train_ids_path.exists():
        train = np.load(train_path)
        train_ids = np.load(train_ids_path)

    return VectorDataset(
        base=base,
        queries=queries,
        base_ids=base_ids,
        query_ids=query_ids,
        train=train,
        train_ids=train_ids,
    )
