"""Deterministic top-k neighbor selection for Indexora.

This module provides data structures and selection algorithms for retrieving the
top-k nearest neighbors from distance arrays with strict deterministic tie-breaking.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SearchResult:
    """Structured container for nearest-neighbor query results.

    Exposes named attributes `.distances` and `.ids` conforming to ADR-0002.

    Parameters
    ----------
    distances : np.ndarray
        Array of squared distances with strict `dtype=np.float32`.
    ids : np.ndarray
        Array of durable entity identifiers with strict `dtype=np.int64`.
    """

    distances: np.ndarray
    ids: np.ndarray

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

    # 6. Deterministic ordering: ascending by distance, tie-breaking by ascending ID
    # np.lexsort sorts by the last key first (distances), broken by the first key (ids)
    sorted_indices = np.lexsort((ids, distances))

    # 7. Select top min(k, N) candidates
    num_to_take = min(k, n_items)
    selected_indices = sorted_indices[:num_to_take]

    return SearchResult(
        distances=distances[selected_indices],
        ids=ids[selected_indices],
    )
