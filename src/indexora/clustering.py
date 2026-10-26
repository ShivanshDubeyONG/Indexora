"""K-means clustering and k-means++ centroid initialization for Indexora.

This module provides the Phase 4 clustering foundation for Indexora:
- `KMeansResult`: Container for centroids, labels, inertia, iteration count, and convergence.
- `kmeans_plus_plus`: D^2 probabilistic centroid initialization.
- `kmeans`: Lloyd's iterative algorithm with configurable convergence, seed, and empty-cluster policy.

NOTE ON ALGORITHM OWNERSHIP:
Per project architecture rules (plan/implementation.md), k-means++ and Lloyd's updates
are human-core algorithms. The scaffolding, validation, result structures, and contracts
are defined here to enable test-driven implementation by the project owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from indexora.kernels import pairwise_squared_l2_ref


@dataclass(frozen=True)
class KMeansResult:
    """Immutable result container for k-means clustering.

    Attributes
    ----------
    centroids : np.ndarray
        Learned cluster centroids of shape `(k, d)` with strict `dtype=np.float32`.
        Guaranteed to be C-contiguous and finite.
    labels : np.ndarray
        Cluster assignment index for each input vector of shape `(N,)` with
        strict `dtype=np.int64`, where `0 <= labels[i] < k`.
    inertia : float
        Sum of squared Euclidean distances from each point to its assigned centroid.
        Computed in float64 accumulation for numerical precision.
    n_iter : int
        Number of iterations executed until convergence or reaching `max_iter`.
    converged : bool
        True if the algorithm converged according to `tol` before reaching `max_iter`.
    inertia_history : tuple[float, ...]
        History of inertia values computed at each completed iteration.
    """

    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int
    converged: bool
    inertia_history: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.centroids, np.ndarray) or self.centroids.dtype != np.float32:
            raise TypeError("centroids must be a float32 numpy ndarray.")
        if self.centroids.ndim != 2:
            raise ValueError("centroids must be a 2D array of shape (k, d).")
        if not isinstance(self.labels, np.ndarray) or self.labels.dtype != np.int64:
            raise TypeError("labels must be an int64 numpy ndarray.")
        if self.labels.ndim != 1:
            raise ValueError("labels must be a 1D array of shape (N,).")
        if self.labels.shape[0] == 0 and self.centroids.shape[0] > 0:
            raise ValueError("Cannot have non-empty centroids with 0 labels.")
        if not isinstance(self.inertia, (float, int)):
            raise TypeError("inertia must be a floating-point number.")
        if self.inertia < 0.0:
            raise ValueError(f"inertia cannot be negative, got {self.inertia}")


def _validate_kmeans_inputs(
    X: np.ndarray,
    k: int,
    max_iter: int | None = None,
    tol: float | None = None,
    empty_cluster_policy: str | None = None,
) -> tuple[int, int]:
    """Validate input matrix and hyperparameters for k-means.

    Returns
    -------
    tuple[int, int]
        (N, d) dimensions of the input matrix X.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be a numpy ndarray, got {type(X).__name__}")
    if X.dtype != np.float32:
        raise TypeError(
            f"X must have strict dtype float32, got {X.dtype}. Silent casting is not permitted."
        )
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array of shape (N, d), got ndim={X.ndim} with shape {X.shape}")
    if not X.flags.c_contiguous:
        raise ValueError("X must be C-contiguous in memory.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite numbers (no NaN or Inf).")

    N, d = X.shape
    if N < 1:
        raise ValueError(f"X must contain at least 1 sample, got N={N}")
    if d < 1:
        raise ValueError(f"X must have feature dimension d >= 1, got d={d}")

    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got k={k}")
    if k > N:
        raise ValueError(f"k cannot exceed number of samples N: k={k} > N={N}")

    if max_iter is not None:
        if not isinstance(max_iter, (int, np.integer)):
            raise TypeError(f"max_iter must be an integer, got {type(max_iter).__name__}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got max_iter={max_iter}")

    if tol is not None:
        if not isinstance(tol, (float, int, np.floating)):
            raise TypeError(f"tol must be a float, got {type(tol).__name__}")
        if tol < 0.0:
            raise ValueError(f"tol must be non-negative (>= 0.0), got tol={tol}")

    if empty_cluster_policy is not None:
        valid_policies = {"reseed_furthest", "keep", "error"}
        if empty_cluster_policy not in valid_policies:
            raise ValueError(
                f"Unknown empty_cluster_policy: {empty_cluster_policy!r}. "
                f"Must be one of {sorted(valid_policies)}"
            )

    return N, d


def kmeans_plus_plus(
    X: np.ndarray,
    k: int,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Initialize cluster centroids using the k-means++ D^2 sampling scheme.

    Parameters
    ----------
    X : np.ndarray
        Input matrix of shape `(N, d)` with strict `dtype=np.float32`. Must be C-contiguous
        and contain finite values.
    k : int
        Number of centroids to initialize (`1 <= k <= N`).
    seed : int, np.random.Generator, or None, default=None
        Random seed or NumPy Generator for reproducible centroid selection.

    Returns
    -------
    np.ndarray
        Array of shape `(k, d)` with strict `dtype=np.float32` containing initial centroids.
        Each centroid is guaranteed to be an exact row chosen from `X`.

    Raises
    ------
    TypeError
        If `X` is not a float32 ndarray or `k` is not an integer.
    ValueError
        If `X` is not 2D/C-contiguous/finite, or if `k < 1`, or if `k > N`.

    Notes
    -----
    Algorithm:
    1. Uniformly sample the first centroid c_0 from rows of X using `rng`.
       Track selected row indices in `selected_indices`.
    2. For each remaining centroid j in 1..k-1:
       a. Compute squared L2 distance D(x_i)^2 from each point x_i to its nearest
          already-chosen centroid: D(x_i)^2 = min_{m < j} ||x_i - c_m||^2.
       b. If sum(D(x)^2) > 0:
          Sample candidate index with probability P(x_i) = D(x_i)^2 / sum(D(x)^2).
       c. Else (sum(D(x)^2) == 0, e.g. duplicate or identical points):
          Sample uniformly at random from remaining unselected row indices:
          `unselected = [i for i in range(N) if i not in selected_indices]`.
       d. Record chosen index in `selected_indices` and set c_j = X[chosen_index].
    3. Return centroids array of shape `(k, d)` with `dtype=np.float32`.
    """
    N, d = _validate_kmeans_inputs(X, k)

    if seed is not None and not isinstance(seed, (int, np.integer, np.random.Generator)):
        raise TypeError(
            f"seed must be an integer, np.random.Generator, or None, got {type(seed).__name__}"
        )

    rng = np.random.default_rng(seed)

    centroids = np.empty((k, d), dtype=np.float32)
    selected_mask = np.zeros(N, dtype=bool)

    # 1. Uniformly sample the first centroid index from available rows
    first_idx = int(rng.integers(0, N))
    centroids[0] = X[first_idx]
    selected_mask[first_idx] = True

    if k == 1:
        return np.ascontiguousarray(centroids, dtype=np.float32)

    # Convert X to float64 for numerically stable intermediate distance arithmetic
    X_f64 = X.astype(np.float64)

    # 2. Compute squared distances from all rows to the first centroid
    diff0 = X_f64 - centroids[0].astype(np.float64)
    min_dists_sq = np.sum(diff0 * diff0, axis=1)

    # 3. Iteratively sample the remaining k-1 centroids
    for j in range(1, k):
        total_dist = float(np.sum(min_dists_sq))

        if total_dist > 0.0 and np.isfinite(total_dist):
            probs = min_dists_sq / total_dist
            # Enforce exact normalization to avoid floating-point drift in choice
            probs_sum = float(np.sum(probs))
            if probs_sum > 0.0 and np.isfinite(probs_sum):
                probs = probs / probs_sum
                chosen_idx = int(rng.choice(N, p=probs))
            else:
                unselected = np.flatnonzero(~selected_mask)
                chosen_idx = int(rng.choice(unselected))
        else:
            # Degenerate: sum(D^2) == 0.0 or non-finite (e.g. all remaining points are duplicates)
            unselected = np.flatnonzero(~selected_mask)
            chosen_idx = int(rng.choice(unselected))

        centroids[j] = X[chosen_idx]
        selected_mask[chosen_idx] = True

        # 4. Update minimum squared distances with the newly chosen centroid
        diff_j = X_f64 - centroids[j].astype(np.float64)
        new_dists_sq = np.sum(diff_j * diff_j, axis=1)
        min_dists_sq = np.minimum(min_dists_sq, new_dists_sq)

    return np.ascontiguousarray(centroids, dtype=np.float32)


def kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int | np.random.Generator | None = None,
    empty_cluster_policy: str = "reseed_furthest",
) -> KMeansResult:
    """Fit k-means clustering to input data using Lloyd's algorithm.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape `(N, d)` with strict `dtype=np.float32`. Must be C-contiguous
        and finite. Input array is never mutated.
    k : int
        Number of clusters to partition data into (`1 <= k <= N`).
    max_iter : int, default=100
        Maximum number of Lloyd iterations (`max_iter >= 1`).
    tol : float, default=1e-4
        Convergence tolerance (`tol >= 0.0`). The algorithm stops early when
        the squared Frobenius norm of centroid shift ||C^(t) - C^(t-1)||_F^2 <= tol.
    seed : int, np.random.Generator, or None, default=None
        Random seed or NumPy Generator for reproducible k-means++ initialization.
    empty_cluster_policy : {"reseed_furthest", "keep", "error"}, default="reseed_furthest"
        Strategy for handling an empty cluster during centroid update:
        - "reseed_furthest": Reseed the empty centroid to the vector in X that currently
          has the largest squared distance to its assigned centroid.
        - "keep": Retain the centroid's position from the previous iteration.
        - "error": Raise `RuntimeError` immediately when an empty cluster occurs.

    Returns
    -------
    KMeansResult
        Immutable dataclass containing:
        - `centroids`: `(k, d)` float32 array
        - `labels`: `(N,)` int64 array
        - `inertia`: float
        - `n_iter`: int
        - `converged`: bool
        - `inertia_history`: tuple of floats across iterations

    Raises
    ------
    TypeError
        If input types or dtypes are invalid (no silent casting).
    ValueError
        If inputs are malformed, non-finite, k < 1, k > N, or hyperparameters out of bounds.
    RuntimeError
        If an empty cluster occurs under `empty_cluster_policy="error"`.
    """
    N, d = _validate_kmeans_inputs(
        X,
        k,
        max_iter=max_iter,
        tol=tol,
        empty_cluster_policy=empty_cluster_policy,
    )

    if seed is not None and not isinstance(seed, (int, np.integer, np.random.Generator)):
        raise TypeError(
            f"seed must be an integer, np.random.Generator, or None, got {type(seed).__name__}"
        )

    # 1. Initialize centroids via k-means++
    centroids = kmeans_plus_plus(X, k, seed=seed)

    inertia_history: list[float] = []
    converged = False
    n_iter = 0

    # 2. Lloyd iteration loop
    for iteration in range(1, max_iter + 1):
        n_iter = iteration

        # Pairwise squared distances to current centroids: shape (N, k)
        dists = pairwise_squared_l2_ref(X, centroids)

        # Assignment step: argmin deterministically breaks ties by smallest index
        labels = np.argmin(dists, axis=1).astype(np.int64)

        # Record inertia from current assignment before update
        min_sq_dists = dists[np.arange(N), labels]
        iter_inertia = float(np.sum(min_sq_dists.astype(np.float64)))
        inertia_history.append(iter_inertia)

        # Cluster occupancy
        counts = np.bincount(labels, minlength=k)
        empty_clusters = [j for j in range(k) if counts[j] == 0]

        new_centroids = np.empty_like(centroids)
        reseeded_clusters = set()

        if empty_clusters:
            if empty_cluster_policy == "error":
                raise RuntimeError(
                    f"Empty cluster encountered at index {empty_clusters[0]} under policy='error'."
                )
            elif empty_cluster_policy == "keep":
                for j in empty_clusters:
                    new_centroids[j] = centroids[j]
                    reseeded_clusters.add(j)
            elif empty_cluster_policy == "reseed_furthest":
                donor_counts = counts.copy()
                used_donors = set()

                for j in empty_clusters:
                    # Eligible candidate points must belong to clusters with > 1 point
                    eligible_mask = np.array(
                        [(donor_counts[labels[i]] > 1 and i not in used_donors) for i in range(N)],
                        dtype=bool,
                    )

                    if np.any(eligible_mask):
                        eligible_indices = np.flatnonzero(eligible_mask)
                        furthest_subidx = np.argmax(min_sq_dists[eligible_indices])
                        best_donor_idx = eligible_indices[furthest_subidx]

                        new_centroids[j] = X[best_donor_idx]
                        reseeded_clusters.add(j)
                        used_donors.add(best_donor_idx)
                        donor_counts[labels[best_donor_idx]] -= 1
                    else:
                        # Fallback if no cluster has > 1 point
                        unused_candidates = [i for i in range(N) if i not in used_donors]
                        if unused_candidates:
                            sub_idx = np.argmax(min_sq_dists[unused_candidates])
                            i_star = unused_candidates[sub_idx]
                            new_centroids[j] = X[i_star]
                            used_donors.add(i_star)
                        else:
                            new_centroids[j] = centroids[j]
                        reseeded_clusters.add(j)

        # Recompute non-empty (and non-reseeded) centroids as arithmetic means using float64
        for j in range(k):
            if j not in reseeded_clusters:
                pts = X[labels == j].astype(np.float64)
                new_centroids[j] = np.mean(pts, axis=0).astype(np.float32)

        new_centroids = np.ascontiguousarray(new_centroids, dtype=np.float32)

        # Compute squared Frobenius centroid shift in float64
        diff_c = new_centroids.astype(np.float64) - centroids.astype(np.float64)
        centroid_shift_sq = float(np.sum(diff_c * diff_c))

        centroids = new_centroids

        if centroid_shift_sq <= tol:
            converged = True
            break

    # 3. Final synchronization pass
    final_dists = pairwise_squared_l2_ref(X, centroids)
    final_labels = np.argmin(final_dists, axis=1).astype(np.int64)
    final_inertia = float(np.sum(np.min(final_dists, axis=1).astype(np.float64)))

    return KMeansResult(
        centroids=centroids,
        labels=final_labels,
        inertia=final_inertia,
        n_iter=n_iter,
        converged=converged,
        inertia_history=tuple(inertia_history),
    )
