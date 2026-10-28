"""Evaluation metrics and ground-truth verification for Indexora.

This module provides standard evaluation metrics for nearest-neighbor search:
- `compute_recall_at_k`: Empirical Recall@k evaluation against exact ground truth.
- `count_distance_computations`: Distance computation accounting for exact and ANN indexes.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from indexora.exact import ExactFlatIndex
from indexora.ivf import IVFFlatIndex
from indexora.topk import SearchResult


def compute_recall_at_k(
    retrieved: SearchResult | np.ndarray,
    ground_truth: SearchResult | np.ndarray,
    k: int | None = None,
    sentinel_id: int | None = None,
    retrieved_mask: np.ndarray | None = None,
    ground_truth_mask: np.ndarray | None = None,
) -> float:
    """Compute empirical Recall@k of retrieved nearest neighbors against ground truth.

    Recall is defined as the fraction of true top-k nearest neighbors retrieved:

        Recall@k = (1 / Q) * sum_{i=1}^Q (|GT_i ∩ R_{i, valid}| / |GT_i|)

    Validity of neighbor slots is determined via the `.mask` attribute of `SearchResult`,
    or via explicit boolean mask arrays (`retrieved_mask` / `ground_truth_mask`).
    Legitimate entity IDs (including -1) marked valid in the mask are preserved and
    count toward recall hits. Only padded/invalid slots (`mask == False`) are excluded.

    Parameters
    ----------
    retrieved : SearchResult | np.ndarray
        Retrieved results: `SearchResult` or `int64` array of shape `(L,)` or `(Q, L)`.
    ground_truth : SearchResult | np.ndarray
        Ground-truth results: `SearchResult` or `int64` array of shape `(K_gt,)` or `(Q, K_gt)`.
    k : int | None, default=None
        Evaluation rank threshold. If None, defaults to the number of columns in ground truth.
    sentinel_id : int | None, default=None
        Deprecated sentinel value fallback for raw arrays without masks. If provided,
        slots matching `sentinel_id` are treated as invalid.
    retrieved_mask : np.ndarray | None, default=None
        Optional boolean mask for raw `retrieved` array (`True` = valid, `False` = padding).
    ground_truth_mask : np.ndarray | None, default=None
        Optional boolean mask for raw `ground_truth` array.

    Notes
    -----
    - **Fewer than k Candidates:** When the database or ground-truth has fewer than
      k candidates, only the valid ground-truth items are counted in the denominator.
      Padding/sentinel slots marked `False` in the mask never inflate or deflate the denominator.
    - **Empty Ground Truth:** If ground truth contains zero valid candidates for a query
      (`denominator == 0`), recall evaluates to 1.0 if the retrieved set is also empty
      (no spurious neighbors), and 0.0 otherwise.
    - **Exceeding Column Count:** If an explicit `k` is passed that exceeds the column
      count of the ground-truth array, raises `ValueError`. Pass `k=None` to evaluate
      against all available columns.

    Returns
    -------
    float
        Empirical Recall@k score between 0.0 and 1.0 (1.0 = 100% ground-truth recovery).
        Returns 1.0 if both ground truth and retrieved sets are empty.
    """
    # 1. Resolve retrieved IDs and mask
    if isinstance(retrieved, SearchResult):
        r_ids = retrieved.ids
        r_mask = retrieved.mask
    elif isinstance(retrieved, np.ndarray):
        r_ids = retrieved
        if retrieved_mask is not None:
            if not isinstance(retrieved_mask, np.ndarray) or retrieved_mask.dtype != bool:
                raise TypeError("retrieved_mask must be a boolean numpy ndarray.")
            if retrieved_mask.shape != r_ids.shape:
                raise ValueError("retrieved_mask shape mismatch with retrieved array.")
            r_mask = retrieved_mask
        elif sentinel_id is not None:
            r_mask = (r_ids != sentinel_id)
        else:
            r_mask = np.ones(r_ids.shape, dtype=bool)
    else:
        raise TypeError(f"retrieved must be a SearchResult or numpy ndarray, got {type(retrieved).__name__}")

    # 2. Resolve ground-truth IDs and mask
    if isinstance(ground_truth, SearchResult):
        gt_ids = ground_truth.ids
        gt_mask = ground_truth.mask
    elif isinstance(ground_truth, np.ndarray):
        gt_ids = ground_truth
        if ground_truth_mask is not None:
            if not isinstance(ground_truth_mask, np.ndarray) or ground_truth_mask.dtype != bool:
                raise TypeError("ground_truth_mask must be a boolean numpy ndarray.")
            if ground_truth_mask.shape != gt_ids.shape:
                raise ValueError("ground_truth_mask shape mismatch with ground_truth array.")
            gt_mask = ground_truth_mask
        elif sentinel_id is not None:
            gt_mask = (gt_ids != sentinel_id)
        else:
            gt_mask = np.ones(gt_ids.shape, dtype=bool)
    else:
        raise TypeError(f"ground_truth must be a SearchResult or numpy ndarray, got {type(ground_truth).__name__}")

    # 3. Validate dtypes and shapes
    if r_ids.dtype != np.int64:
        raise TypeError(f"retrieved must have dtype int64, got {r_ids.dtype}")
    if gt_ids.dtype != np.int64:
        raise TypeError(f"ground_truth must have dtype int64, got {gt_ids.dtype}")

    if r_ids.ndim != gt_ids.ndim:
        raise ValueError(
            f"Dimension mismatch between retrieved ({r_ids.ndim}D) "
            f"and ground_truth ({gt_ids.ndim}D)"
        )

    if r_ids.ndim not in (1, 2):
        raise ValueError(f"Arrays must be 1D or 2D, got ndim={r_ids.ndim}")

    is_single_query = r_ids.ndim == 1
    if is_single_query:
        r_2d = r_ids[np.newaxis, :]
        r_m_2d = r_mask[np.newaxis, :]
        gt_2d = gt_ids[np.newaxis, :]
        gt_m_2d = gt_mask[np.newaxis, :]
    else:
        r_2d = r_ids
        r_m_2d = r_mask
        gt_2d = gt_ids
        gt_m_2d = gt_mask

    num_queries = r_2d.shape[0]
    if num_queries != gt_2d.shape[0]:
        raise ValueError(
            f"Query count mismatch: retrieved has {num_queries} queries "
            f"but ground truth has {gt_2d.shape[0]} queries."
        )

    if num_queries == 0:
        return 1.0

    eff_k = gt_2d.shape[1] if k is None else int(k)
    if eff_k <= 0:
        raise ValueError(f"k must be a positive integer, got {eff_k}")
    if k is not None and k > gt_2d.shape[1]:
        raise ValueError(
            f"Requested k={k} exceeds ground-truth column count ({gt_2d.shape[1]})."
        )

    total_recall = 0.0
    for i in range(num_queries):
        # Extract valid ground-truth IDs up to eff_k
        gt_row_ids = gt_2d[i, :eff_k]
        gt_row_mask = gt_m_2d[i, :eff_k]
        gt_valid = [int(val) for val, valid in zip(gt_row_ids, gt_row_mask) if valid]
        gt_set = set(gt_valid)
        denominator = len(gt_set)

        if denominator == 0:
            # If ground truth has no valid candidates for this query
            ret_valid = [int(val) for val, valid in zip(r_2d[i, :eff_k], r_m_2d[i, :eff_k]) if valid]
            total_recall += 1.0 if len(ret_valid) == 0 else 0.0
            continue

        # Extract valid retrieved IDs up to eff_k
        ret_row_ids = r_2d[i]
        ret_row_mask = r_m_2d[i]
        ret_valid = [int(val) for val, valid in zip(ret_row_ids, ret_row_mask) if valid][:eff_k]
        ret_set = set(ret_valid)

        numerator = len(gt_set.intersection(ret_set))
        total_recall += numerator / denominator

    return float(total_recall / num_queries)


def count_distance_computations(
    index: ExactFlatIndex | IVFFlatIndex,
    queries: np.ndarray,
    nprobe: int | None = None,
    k: int = 10,
) -> float:
    """Calculate the average number of vector distance computations performed per query.

    Definitions:
    - `ExactFlatIndex`: Each query computes exact distances against all `N` database vectors.
      Distance computations per query = `N`.
    - `IVFFlatIndex`: Actual distance computations executed during search, including coarse
      centroid comparisons and candidates evaluated in probed inverted lists.
      Distance computations per query = `result.distance_computations / num_queries`.

    Parameters
    ----------
    index : ExactFlatIndex | IVFFlatIndex
        The search index being evaluated.
    queries : np.ndarray
        Query vector(s) of shape `(d,)` or `(Q, d)` with strict `dtype=np.float32`.
    nprobe : int | None, default=None
        Number of inverted lists probed for IVF-Flat. If None, uses `index.nprobe`.
    k : int, default=10
        Number of nearest neighbors to retrieve.

    Returns
    -------
    float
        Average number of distance computations per query.

    Raises
    ------
    TypeError
        If `index` is not a supported index type or `queries` is not a float32 ndarray.
    """
    if not isinstance(queries, np.ndarray) or queries.dtype != np.float32:
        raise TypeError("queries must be a float32 numpy ndarray.")

    is_single_query = queries.ndim == 1
    num_queries = 1 if is_single_query else queries.shape[0]

    if num_queries == 0:
        return 0.0

    if isinstance(index, ExactFlatIndex):
        return float(index.n)

    if isinstance(index, IVFFlatIndex):
        eff_nprobe = index.nprobe if nprobe is None else int(nprobe)
        res = index.search(queries, k=k, nprobe=eff_nprobe)
        if res.distance_computations is None:
            return 0.0
        return float(res.distance_computations / num_queries)

    raise TypeError(f"Unsupported index type: {type(index).__name__}")
