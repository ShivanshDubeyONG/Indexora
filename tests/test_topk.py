"""Tests for deterministic top-k selection and SearchResult (Phase 1).

These tests verify `select_top_k` and `SearchResult` against the contracts in
`docs/decisions.md` (ADR-0001, ADR-0002) and `docs/phase1_test_spec.md`.
"""

import numpy as np
import pytest

from indexora.topk import SearchResult, select_top_k


def test_search_result_container_structure():
    """Verify SearchResult structure, attribute names, and strict dtype enforcement."""
    distances = np.array([0.5, 1.2], dtype=np.float32)
    ids = np.array([101, 102], dtype=np.int64)

    result = SearchResult(distances=distances, ids=ids)
    assert isinstance(result, SearchResult)
    assert hasattr(result, "distances")
    assert hasattr(result, "ids")
    np.testing.assert_array_equal(result.distances, distances)
    np.testing.assert_array_equal(result.ids, ids)
    assert result.distances.dtype == np.float32
    assert result.ids.dtype == np.int64

    # Reject non-float32 distances
    with pytest.raises(TypeError):
        SearchResult(distances=np.array([0.5, 1.2], dtype=np.float64), ids=ids)

    # Reject non-int64 ids
    with pytest.raises(TypeError):
        SearchResult(distances=distances, ids=np.array([101, 102], dtype=np.int32))

    # Reject shape mismatch
    with pytest.raises(ValueError):
        SearchResult(distances=distances, ids=np.array([101], dtype=np.int64))


def test_t1_10_minimal_selection_k1():
    """T1.10: Minimal selection with k=1 returns a SearchResult containing the nearest neighbor."""
    distances = np.array([12.5, 3.2, 0.45, 8.9], dtype=np.float32)
    ids = np.array([101, 102, 103, 104], dtype=np.int64)
    k = 1

    result = select_top_k(distances, ids, k=k)
    assert isinstance(result, SearchResult)
    assert result.ids.shape == (1,)
    assert result.distances.shape == (1,)
    assert result.ids.dtype == np.int64
    assert result.distances.dtype == np.float32
    assert result.ids[0] == 103
    np.testing.assert_allclose(result.distances[0], np.float32(0.45))


def test_t1_8_deterministic_equal_distances():
    """T1.8: Deterministic tie-breaking on identical distances by smaller int64 ID."""
    distances = np.array([3.0, 3.0], dtype=np.float32)
    ids = np.array([200, 100], dtype=np.int64)
    k = 2

    result = select_top_k(distances, ids, k=k)
    assert isinstance(result, SearchResult)
    np.testing.assert_array_equal(result.ids, np.array([100, 200], dtype=np.int64))
    np.testing.assert_allclose(result.distances, np.array([3.0, 3.0], dtype=np.float32))


def test_t1_9_stable_id_tie_breaking():
    """T1.9: Stable-ID tie-breaking with non-contiguous IDs and equal distances."""
    distances = np.array([10.0, 5.0, 5.0, 8.0, 5.0], dtype=np.float32)
    ids = np.array([1001, 5003, 2002, 3004, 1005], dtype=np.int64)
    k = 3

    result = select_top_k(distances, ids, k=k)
    assert isinstance(result, SearchResult)
    expected_ids = np.array([1005, 2002, 5003], dtype=np.int64)
    expected_distances = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    np.testing.assert_array_equal(result.ids, expected_ids)
    np.testing.assert_allclose(result.distances, expected_distances)


def test_t1_11_full_set_selection_k_equals_n():
    """T1.11: Top-k with k=N returns all N items sorted strictly by (distance, ID)."""
    distances = np.array([4.0, 1.0, 3.0, 2.0, 1.0], dtype=np.float32)
    ids = np.array([50, 20, 40, 30, 10], dtype=np.int64)
    k = 5

    result = select_top_k(distances, ids, k=k)
    assert isinstance(result, SearchResult)
    assert len(result.ids) == 5
    assert len(result.distances) == 5
    expected_ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    expected_distances = np.array([1.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_array_equal(result.ids, expected_ids)
    np.testing.assert_allclose(result.distances, expected_distances)


def test_t1_12_oversized_query_k_greater_than_n():
    """T1.12: When k > N, return all N available items sorted by (distance, ID) with no padding."""
    distances = np.array([3.0, 1.0, 2.0, 5.0, 4.0], dtype=np.float32)
    ids = np.array([30, 10, 20, 50, 40], dtype=np.int64)
    k = 10

    result = select_top_k(distances, ids, k=k)
    assert isinstance(result, SearchResult)
    assert len(result.ids) == 5
    assert len(result.distances) == 5
    expected_ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    expected_distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    np.testing.assert_array_equal(result.ids, expected_ids)
    np.testing.assert_allclose(result.distances, expected_distances)


def test_t1_13_boundary_tie_at_rank_k():
    """T1.13: Exact tie at the k-th rank boundary resolved deterministically by ID."""
    distances = np.array([1.0, 2.0, 2.0, 2.0, 5.0], dtype=np.float32)
    ids = np.array([1, 40, 20, 30, 5], dtype=np.int64)
    k = 3

    result = select_top_k(distances, ids, k=k)
    assert isinstance(result, SearchResult)
    expected_ids = np.array([1, 20, 30], dtype=np.int64)
    expected_distances = np.array([1.0, 2.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(result.ids, expected_ids)
    np.testing.assert_allclose(result.distances, expected_distances)


def test_t1_15_invalid_k_non_positive():
    """T1.15: Query with k <= 0 must raise ValueError('k must be positive')."""
    distances = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ids = np.array([10, 20, 30], dtype=np.int64)

    with pytest.raises(ValueError, match=r"^k must be positive$"):
        select_top_k(distances, ids, k=0)

    with pytest.raises(ValueError, match=r"^k must be positive$"):
        select_top_k(distances, ids, k=-5)


def test_empty_input_with_positive_k():
    """Empty database input with valid positive k must return empty SearchResult."""
    distances = np.empty((0,), dtype=np.float32)
    ids = np.empty((0,), dtype=np.int64)

    result = select_top_k(distances, ids, k=5)
    assert isinstance(result, SearchResult)
    assert result.distances.shape == (0,)
    assert result.ids.shape == (0,)
    assert result.distances.dtype == np.float32
    assert result.ids.dtype == np.int64


def test_input_shape_and_dtype_validation():
    """Validate rejection of mismatched dtypes, lengths, shapes, or non-finite numbers."""
    valid_dists = np.array([1.0, 2.0], dtype=np.float32)
    valid_ids = np.array([10, 20], dtype=np.int64)

    # Distances float64
    with pytest.raises((TypeError, ValueError)):
        select_top_k(np.array([1.0, 2.0], dtype=np.float64), valid_ids, k=1)

    # Distances int32
    with pytest.raises((TypeError, ValueError)):
        select_top_k(np.array([1, 2], dtype=np.int32), valid_ids, k=1)

    # IDs int32
    with pytest.raises((TypeError, ValueError)):
        select_top_k(valid_dists, np.array([10, 20], dtype=np.int32), k=1)

    # Length mismatch
    with pytest.raises(ValueError):
        select_top_k(valid_dists, np.array([10], dtype=np.int64), k=1)

    # 2D distances
    with pytest.raises(ValueError):
        select_top_k(np.ones((2, 2), dtype=np.float32), valid_ids, k=1)

    # 2D IDs
    with pytest.raises(ValueError):
        select_top_k(valid_dists, np.ones((2, 1), dtype=np.int64), k=1)

    # Non-finite distance
    nan_dists = np.array([1.0, np.nan], dtype=np.float32)
    with pytest.raises(ValueError):
        select_top_k(nan_dists, valid_ids, k=1)


def test_search_result_immutability():
    """Verify that SearchResult is frozen and attributes cannot be modified."""
    from dataclasses import FrozenInstanceError

    res = SearchResult(
        distances=np.array([1.0], dtype=np.float32),
        ids=np.array([42], dtype=np.int64),
    )
    with pytest.raises((FrozenInstanceError, AttributeError)):
        res.distances = np.array([2.0], dtype=np.float32)

    with pytest.raises((FrozenInstanceError, AttributeError)):
        res.ids = np.array([99], dtype=np.int64)


def test_deterministic_tie_breaking_negative_and_large_int64_ids():
    """Verify tie-breaking works deterministically with negative and large int64 IDs."""
    distances = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    ids = np.array([100, -50, np.int64(2**62)], dtype=np.int64)
    k = 3

    result = select_top_k(distances, ids, k=k)
    expected_ids = np.array([-50, 100, np.int64(2**62)], dtype=np.int64)
    np.testing.assert_array_equal(result.ids, expected_ids)


def test_single_element_database():
    """Verify N=1 database with k=1 and k > N."""
    distances = np.array([4.2], dtype=np.float32)
    ids = np.array([777], dtype=np.int64)

    # k = 1
    res1 = select_top_k(distances, ids, k=1)
    assert res1.ids.shape == (1,)
    assert res1.ids[0] == 777
    assert res1.distances[0] == np.float32(4.2)

    # k = 5 (k > N)
    res5 = select_top_k(distances, ids, k=5)
    assert res5.ids.shape == (1,)
    assert res5.ids[0] == 777
    assert res5.distances[0] == np.float32(4.2)
