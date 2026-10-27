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


def compute_recall_at_k(
    retrieved_ids: np.ndarray,
    ground_truth_ids: np.ndarray,
    k: int | None = None,
    sentinel_id: int = -1,
) -> float:
    """Compute empirical Recall@k of retrieved nearest neighbors against ground truth.

    Recall is defined as the fraction of true top-k nearest neighbors retrieved:

        Recall@k = (1 / Q) * sum_{i=1}^Q (|GT_i ∩ R_{i, valid}| / |GT_i|)

    Sentinel IDs (such as -1 from batched IVF padding) are strictly excluded from
    the retrieved set and cannot count as true neighbor hits.

    Parameters
    ----------
    retrieved_ids : np.ndarray
        Array of retrieved entity IDs:
        - 1D array of shape `(L,)` for a single query.
        - 2D array of shape `(Q, L)` for batched queries.
        Strict `dtype=np.int64`.
    ground_truth_ids : np.ndarray
        Array of exact ground-truth entity IDs:
        - 1D array of shape `(K_gt,)` for a single query.
        - 2D array of shape `(Q, K_gt)` for batched queries.
        Strict `dtype=np.int64`.
    k : int | None, default=None
        Evaluation rank threshold. If None, defaults to `K_gt` (the number of
        columns in `ground_truth_ids`). Must be a positive integer if specified.
    sentinel_id : int, default=-1
        Sentinel ID value indicating padded missing slots. Padded slots are
        strictly excluded before computing set intersections.

    Returns
    -------
    float
        Empirical Recall@k score between 0.0 and 1.0 (1.0 = 100% ground-truth recovery).
        Returns 1.0 if both ground truth and retrieved sets are empty.

    Raises
    ------
    TypeError
        If `retrieved_ids` or `ground_truth_ids` are not NumPy ndarrays or not int64.
        If `k` is not an integer when provided.
    ValueError
        If `k <= 0`, if dimensionalities mismatch (e.g. 1D vs 2D), or if query counts
        `Q` do not match between retrieved and ground truth arrays.
    """
    if not isinstance(retrieved_ids, np.ndarray):
        raise TypeError(f"retrieved_ids must be a numpy ndarray, got {type(retrieved_ids).__name__}")
    if not isinstance(ground_truth_ids, np.ndarray):
        raise TypeError(f"ground_truth_ids must be a numpy ndarray, got {type(ground_truth_ids).__name__}")

    if retrieved_ids.dtype != np.int64:
        raise TypeError(f"retrieved_ids must have dtype int64, got {retrieved_ids.dtype}")
    if ground_truth_ids.dtype != np.int64:
        raise TypeError(f"ground_truth_ids must have dtype int64, got {ground_truth_ids.dtype}")

    if retrieved_ids.ndim != ground_truth_ids.ndim:
        raise ValueError(
            f"Dimension mismatch between retrieved_ids ({retrieved_ids.ndim}D) "
            f"and ground_truth_ids ({ground_truth_ids.ndim}D)"
        )

    if retrieved_ids.ndim not in (1, 2):
        raise ValueError(f"Arrays must be 1D or 2D, got ndim={retrieved_ids.ndim}")

    is_single_query = retrieved_ids.ndim == 1
    if is_single_query:
        r_2d = retrieved_ids[np.newaxis, :]
        gt_2d = ground_truth_ids[np.newaxis, :]
    else:
        r_2d = retrieved_ids
        gt_2d = ground_truth_ids

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

    total_recall = 0.0
    for i in range(num_queries):
        # Truncate ground truth to eff_k and filter any invalid sentinels
        gt_row = gt_2d[i, :eff_k]
        gt_valid = [x for x in gt_row if x != sentinel_id]
        gt_set = set(gt_valid)
        denominator = len(gt_set)

        if denominator == 0:
            # If ground truth is empty for this query, perfect recall if retrieved is also empty of valid items
            ret_valid = [x for x in r_2d[i, :eff_k] if x != sentinel_id]
            total_recall += 1.0 if len(ret_valid) == 0 else 0.0
            continue

        # Extract retrieved items up to eff_k, strictly excluding sentinel padding
        ret_row = r_2d[i]
        ret_valid = [x for x in ret_row if x != sentinel_id][:eff_k]
        ret_set = set(ret_valid)

        intersection_count = len(gt_set.intersection(ret_set))
        total_recall += intersection_count / denominator

    return float(total_recall / num_queries)


def count_distance_computations(
    index: ExactFlatIndex | IVFFlatIndex,
    queries: np.ndarray,
    nprobe: int | None = None,
) -> float:
    """Calculate the average number of vector distance computations performed per query.

    Definitions:
    - `ExactFlatIndex`: Each query computes exact distances against all `N` database vectors.
      Distance computations per query = `N`.
    - `IVFFlatIndex`: Each query computes distances against all `nlist` coarse centroids,
      plus exact distances against candidate vectors pooled from the nearest `nprobe` lists.
      Distance computations per query = `nlist + mean(candidate_counts)`.

    Parameters
    ----------
    index : ExactFlatIndex | IVFFlatIndex
        The search index being evaluated.
    queries : np.ndarray
        Query vector(s) of shape `(d,)` or `(Q, d)` with strict `dtype=np.float32`.
    nprobe : int | None, default=None
        Number of inverted lists probed for IVF-Flat. If None, uses `index.nprobe`.

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
        # Exact scan evaluates all N database vectors per query
        return float(index.n)

    if isinstance(index, IVFFlatIndex):
        eff_nprobe = index.nprobe if nprobe is None else int(nprobe)
        if eff_nprobe < 1 or eff_nprobe > index.nlist:
            raise ValueError(f"nprobe must be between 1 and {index.nlist}, got {eff_nprobe}")

        # Coarse stage: compares against all nlist centroids
        coarse_dc = index.nlist

        # Fine candidate stage: query finds eff_nprobe nearest centroids
        # and compares against vectors in those lists
        if is_single_query:
            q_2d = queries[np.newaxis, :]
        else:
            q_2d = queries

        # Coarse distance calculation to find probed list sizes
        # Centroid distance shape: (Q, nlist)
        from indexora.kernels import pairwise_squared_l2_ref

        centroid_dists = pairwise_squared_l2_ref(q_2d, index.centroids)
        # Sort centroids per query to find probed lists
        probed_centroids = np.argsort(centroid_dists, axis=1, kind="stable")[:, :eff_nprobe]

        list_sizes = index.list_sizes()
        # Sum sizes of probed lists for each query
        cand_counts = np.sum(list_sizes[probed_centroids], axis=1)

        mean_cand_dc = float(np.mean(cand_counts))
        return float(coarse_dc + mean_cand_dc)

    raise TypeError(f"Unsupported index type: {type(index).__name__}")
