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
    assert hasattr(result, "mask")
    assert hasattr(result, "distance_computations")
    np.testing.assert_array_equal(result.distances, distances)
    np.testing.assert_array_equal(result.ids, ids)
    np.testing.assert_array_equal(result.mask, np.ones_like(ids, dtype=bool))
    assert result.distances.dtype == np.float32
    assert result.ids.dtype == np.int64
    assert result.mask.dtype == bool
    assert result.distance_computations is None

    # Custom mask and distance_computations
    res_custom = SearchResult(
        distances=distances,
        ids=ids,
        mask=np.array([True, False], dtype=bool),
        distance_computations=42,
        centroid_distance_computations=10,
        candidate_distance_computations=32,
    )
    np.testing.assert_array_equal(res_custom.mask, np.array([True, False], dtype=bool))
    assert res_custom.distance_computations == 42
    assert res_custom.centroid_distance_computations == 10
    assert res_custom.candidate_distance_computations == 32

    # Reject non-float32 distances
    with pytest.raises(TypeError):
        SearchResult(distances=np.array([0.5, 1.2], dtype=np.float64), ids=ids)

    # Reject non-int64 ids
    with pytest.raises(TypeError):
        SearchResult(distances=distances, ids=np.array([101, 102], dtype=np.int32))

    # Reject shape mismatch
    with pytest.raises(ValueError):
        SearchResult(distances=distances, ids=np.array([101], dtype=np.int64))

    # Reject non-bool mask
    with pytest.raises(TypeError):
        SearchResult(distances=distances, ids=ids, mask=np.array([1, 0]))

    # Reject mask shape mismatch
    with pytest.raises(ValueError):
        SearchResult(distances=distances, ids=ids, mask=np.array([True], dtype=bool))

    # Reject negative distance_computations
    with pytest.raises(ValueError):
        SearchResult(distances=distances, ids=ids, distance_computations=-1)

    # Reject non-integer distance_computations
    with pytest.raises(TypeError):
        SearchResult(distances=distances, ids=ids, distance_computations=3.14)  # type: ignore[arg-type]

    # Reject negative centroid_distance_computations
    with pytest.raises(ValueError):
        SearchResult(distances=distances, ids=ids, centroid_distance_computations=-1)

    # Reject non-integer centroid_distance_computations
    with pytest.raises(TypeError):
        SearchResult(distances=distances, ids=ids, centroid_distance_computations="bad")  # type: ignore[arg-type]

    # Reject negative candidate_distance_computations
    with pytest.raises(ValueError):
        SearchResult(distances=distances, ids=ids, candidate_distance_computations=-1)

    # Reject non-integer candidate_distance_computations
    with pytest.raises(TypeError):
        SearchResult(distances=distances, ids=ids, candidate_distance_computations="bad")  # type: ignore[arg-type]


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

    with pytest.raises((FrozenInstanceError, AttributeError)):
        res.mask = np.array([False], dtype=bool)

    with pytest.raises((FrozenInstanceError, AttributeError)):
        res.distance_computations = 100

    with pytest.raises((FrozenInstanceError, AttributeError)):
        res.centroid_distance_computations = 50

    with pytest.raises((FrozenInstanceError, AttributeError)):
        res.candidate_distance_computations = 50


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


def test_partial_selection_equivalence_to_full_lexsort_randomized():
    """Verify select_top_k matches full np.lexsort oracle across 200 randomized inputs."""
    def full_lexsort_oracle(distances: np.ndarray, ids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        order = np.lexsort((ids, distances))
        take = min(k, len(distances))
        chosen = order[:take]
        return distances[chosen], ids[chosen]

    rng = np.random.default_rng(42)
    for _ in range(200):
        n = int(rng.integers(1, 300))
        k = int(rng.integers(1, n + 50))

        # Include frequent duplicate distances and negative IDs
        unique_dists = rng.uniform(0.0, 10.0, size=min(10, n)).astype(np.float32)
        dists = rng.choice(unique_dists, size=n)
        ids = rng.integers(-5000, 5000, size=n, dtype=np.int64)

        result = select_top_k(dists, ids, k=k)
        oracle_dists, oracle_ids = full_lexsort_oracle(dists, ids, k=k)

        np.testing.assert_array_equal(result.distances, oracle_dists)
        np.testing.assert_array_equal(result.ids, oracle_ids)


def test_boundary_tie_breaking_adversarial():
    """Verify boundary tie resolution when multiple candidates tie exactly at rank k."""
    # Construct: 3 items strictly closer (< 2.0)
    # 5 items tied at boundary (= 2.0) with scrambled IDs
    # 3 items strictly farther (> 2.0)
    # k = 5 -> should take 3 closer + 2 lowest IDs among the 5 tied
    distances = np.array([1.0, 0.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 5.0, 4.0, 6.0], dtype=np.float32)
    ids = np.array([10, 20, 30, 900, -50, 120, 0, 45, 999, 888, 777], dtype=np.int64)
    # Tied IDs are: 900, -50, 120, 0, 45
    # Two smallest are: -50, 0
    k = 5

    res = select_top_k(distances, ids, k=k)
    expected_ids = np.array([20, 10, 30, -50, 0], dtype=np.int64)
    expected_dists = np.array([0.5, 1.0, 1.5, 2.0, 2.0], dtype=np.float32)

    np.testing.assert_array_equal(res.ids, expected_ids)
    np.testing.assert_array_equal(res.distances, expected_dists)


def test_boundary_all_tied_distances():
    """Verify selection when every candidate has identical distance."""
    n = 50
    distances = np.full(n, 3.14, dtype=np.float32)
    rng = np.random.default_rng(505)
    ids = rng.permutation(np.arange(100, 100 + n, dtype=np.int64))

    for k in [1, 5, 25, 50, 75]:
        res = select_top_k(distances, ids, k=k)
        expected_len = min(k, n)
        assert len(res.ids) == expected_len
        # Since all distances are tied, result should be the expected_len smallest IDs in ascending order
        expected_ids = np.sort(ids)[:expected_len]
        np.testing.assert_array_equal(res.ids, expected_ids)
        np.testing.assert_allclose(res.distances, np.full(expected_len, 3.14, dtype=np.float32))


def test_topk_more_than_k_tied_at_boundary_negative_ids():
    """Verify tie-breaking when more than k candidates are tied exactly at the boundary with negative IDs."""
    # 2 closer candidates (dists 0.1, 0.2)
    # 10 tied candidates (all dist 1.5) with arbitrary positive and negative IDs
    # 2 farther candidates (dists 3.0, 4.0)
    # k = 5: requires 2 closer + 3 smallest IDs among the 10 tied
    dists = np.array([0.1, 0.2] + [1.5] * 10 + [3.0, 4.0], dtype=np.float32)
    ids = np.array([100, 200, -500, 800, -100, 42, -50, 999, -999, 12, -1, 7, 300, 400], dtype=np.int64)

    # 10 tied IDs: [-500, 800, -100, 42, -50, 999, -999, 12, -1, 7]
    # Smallest 3: -999, -500, -100
    res = select_top_k(dists, ids, k=5)
    expected_ids = np.array([100, 200, -999, -500, -100], dtype=np.int64)
    expected_dists = np.array([0.1, 0.2, 1.5, 1.5, 1.5], dtype=np.float32)

    np.testing.assert_array_equal(res.ids, expected_ids)
    np.testing.assert_allclose(res.distances, expected_dists)
