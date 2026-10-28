"""Deterministic top-k neighbor selection for Indexora.

This module provides data structures and selection algorithms for retrieving the
top-k nearest neighbors from distance arrays with strict deterministic tie-breaking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class SearchResult:
    """Structured container for nearest-neighbor query results.

    Exposes named attributes `.distances` (float32), `.ids` (int64),
    `.mask` (bool), and optional `.distance_computations` (int).

    Parameters
    ----------
    distances : np.ndarray
        Array of squared distances with strict `dtype=np.float32`.
    ids : np.ndarray
        Array of durable entity identifiers with strict `dtype=np.int64`.
    mask : np.ndarray | None, default=None
        Boolean array of the same shape indicating valid neighbor slots (`True`)
        versus padding slots (`False`). If None, defaults to all `True`.
    distance_computations : int | None, default=None
        Total count of vector distance computations performed for this search.
    """

    distances: np.ndarray
    ids: np.ndarray
    mask: np.ndarray = field(default=None)  # type: ignore[assignment]
    distance_computations: int | None = field(default=None)
    centroid_distance_computations: int | None = field(default=None)
    candidate_distance_computations: int | None = field(default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.distances, np.ndarray):
            raise TypeError(
                f"distances must be a numpy ndarray, got {type(self.distances).__name__}"
            )
        if not isinstance(self.ids, np.ndarray):
            raise TypeError(
                f"ids must be a numpy ndarray, got {type(self.ids).__name__}"
            )
        if self.distances.dtype != np.float32:
            raise TypeError(
                f"distances must have dtype float32, got {self.distances.dtype}"
            )
        if self.ids.dtype != np.int64:
            raise TypeError(
                f"ids must have dtype int64, got {self.ids.dtype}"
            )
        if self.distances.shape != self.ids.shape:
            raise ValueError(
                f"Shape mismatch between distances {self.distances.shape} and ids {self.ids.shape}"
            )

        if self.mask is None:
            object.__setattr__(self, "mask", np.ones(self.ids.shape, dtype=bool))
        else:
            if not isinstance(self.mask, np.ndarray):
                raise TypeError(
                    f"mask must be a numpy ndarray, got {type(self.mask).__name__}"
                )
            if self.mask.dtype != np.bool_ and self.mask.dtype != bool:
                raise TypeError(
                    f"mask must have dtype bool, got {self.mask.dtype}"
                )
            if self.mask.shape != self.ids.shape:
                raise ValueError(
                    f"Shape mismatch between mask {self.mask.shape} and ids {self.ids.shape}"
                )

        if self.distance_computations is not None:
            if not isinstance(self.distance_computations, (int, np.integer)):
                raise TypeError(
                    f"distance_computations must be an integer, got {type(self.distance_computations).__name__}"
                )
            if self.distance_computations < 0:
                raise ValueError("distance_computations must be non-negative")

        if self.centroid_distance_computations is not None:
            if not isinstance(self.centroid_distance_computations, (int, np.integer)):
                raise TypeError(
                    f"centroid_distance_computations must be an integer, got {type(self.centroid_distance_computations).__name__}"
                )
            if self.centroid_distance_computations < 0:
                raise ValueError("centroid_distance_computations must be non-negative")

        if self.candidate_distance_computations is not None:
            if not isinstance(self.candidate_distance_computations, (int, np.integer)):
                raise TypeError(
                    f"candidate_distance_computations must be an integer, got {type(self.candidate_distance_computations).__name__}"
                )
            if self.candidate_distance_computations < 0:
                raise ValueError("candidate_distance_computations must be non-negative")


def select_top_k(
    distances: np.ndarray,
    ids: np.ndarray,
    k: int,
) -> SearchResult:
    """Select top-k nearest neighbors sorted deterministically by (distance, ID).

    Parameters
    ----------
    distances : np.ndarray
        1D array of squared Euclidean distances with strict `dtype=np.float32`.
    ids : np.ndarray
        1D array of durable entity identifiers with strict `dtype=np.int64`.
    k : int
        Number of nearest neighbors to retrieve. Must be a positive integer (k >= 1).

    Returns
    -------
    SearchResult
        Container with `.distances` (float32) and `.ids` (int64) of length min(k, N),
        sorted in strictly ascending composite `(distance, ID)` order.

    Raises
    ------
    ValueError
        If `k <= 0`, raises `ValueError("k must be positive")`.
        If `len(distances) != len(ids)`.
        If `distances` or `ids` are not 1D arrays.
        If `distances` contains non-finite values (NaN or Inf).
    TypeError or ValueError
        If `distances.dtype != np.float32` or `ids.dtype != np.int64` (no silent casting).

    Notes
    -----
    - Oversized queries (k > N) return all N available items sorted by (distance, ID)
      without raising an error or padding with dummy IDs.
    - Boundary ties at rank k are broken deterministically by smaller int64 ID.
    - If N = 0 (empty input) and k > 0, returns an empty SearchResult with shape (0,).
    - Behavior for non-integer k is an open/unresolved owner decision and is not decided here.
    """
    # 1. Validate k parameter
    if k <= 0:
        raise ValueError("k must be positive")

    # 2. Validate strict dtypes (no silent casting per ADR-0001/0002)
    if distances.dtype != np.float32:
        raise TypeError(
            f"distances must have dtype float32, got {distances.dtype}. Silent casting is not permitted."
        )
    if ids.dtype != np.int64:
        raise TypeError(
            f"ids must have dtype int64, got {ids.dtype}. Silent casting is not permitted."
        )

    # 3. Validate 1D shapes and length alignment
    if distances.ndim != 1:
        raise ValueError(
            f"distances must be a 1D array of shape (N,), got ndim={distances.ndim} with shape {distances.shape}"
        )
    if ids.ndim != 1:
        raise ValueError(
            f"ids must be a 1D array of shape (N,), got ndim={ids.ndim} with shape {ids.shape}"
        )
    if distances.shape[0] != ids.shape[0]:
        raise ValueError(
            f"Length mismatch: distances length {distances.shape[0]} != ids length {ids.shape[0]}"
        )

    # 4. Validate finiteness
    if not np.all(np.isfinite(distances)):
        raise ValueError("distances must contain only finite numbers (no NaN or Inf).")

    # 5. Handle empty input
    n_items = distances.shape[0]
    if n_items == 0:
        return SearchResult(
            distances=np.empty((0,), dtype=np.float32),
            ids=np.empty((0,), dtype=np.int64),
        )

    # 6. Top-k candidate selection with O(N + k log k) partial selection
    # If k >= n_items, all candidates are included; we sort all n_items by (distance, ID)
    if k >= n_items:
        sorted_indices = np.lexsort((ids, distances))
        return SearchResult(
            distances=distances[sorted_indices],
            ids=ids[sorted_indices],
        )

    # 7. Partial selection for k < n_items:
    # Step 7a: Find boundary distance at rank k-1 in O(N) using np.partition
    boundary_dist = np.partition(distances, k - 1)[k - 1]

    # Step 7b: Partition candidates into strictly closer and boundary-tied sets
    strictly_closer = np.flatnonzero(distances < boundary_dist)
    n_closer = len(strictly_closer)
    n_needed = k - n_closer

    tied = np.flatnonzero(distances == boundary_dist)

    # Step 7c: Resolve candidates tied at the boundary by selecting smallest IDs
    if len(tied) == n_needed:
        selected_indices = np.concatenate([strictly_closer, tied])
    else:
        tied_ids = ids[tied]
        # Partial selection on tied IDs in O(|tied|) to choose the n_needed smallest IDs
        chosen_tied_local = np.argpartition(tied_ids, n_needed - 1)[:n_needed]
        selected_indices = np.concatenate([strictly_closer, tied[chosen_tied_local]])

    # Step 7d: Sort the final k chosen results by ascending (distance, ID)
    sel_dists = distances[selected_indices]
    sel_ids = ids[selected_indices]
    final_order = np.lexsort((sel_ids, sel_dists))

    return SearchResult(
        distances=sel_dists[final_order],
        ids=sel_ids[final_order],
    )
