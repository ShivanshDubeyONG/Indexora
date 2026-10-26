"""Tests for Phase 2: Exact Flat Search and Ground Truth.

These tests verify `exact_search`, `ExactFlatIndex`, and `generate_ground_truth`
against the contracts in `plan/project srs.md`, `plan/tech design.md`, and
`docs/decisions.md`.
"""

import numpy as np
import pytest

from indexora import (
    ExactFlatIndex,
    SearchResult,
    exact_search,
    generate_ground_truth,
    pairwise_squared_l2_ref,
    select_top_k,
)


def test_hand_calculated_exact_search():
    """1. Hand-calculated dataset with known nearest neighbors."""
    # 4 known 2D points from Phase 1 learning guide
    database = np.array(
        [
            [1.0, 2.0],  # ID 101, dist^2 = 1.0
            [4.0, 6.0],  # ID 102, dist^2 = 34.0
            [2.0, 1.0],  # ID 103, dist^2 = 1.0
            [1.0, 2.0],  # ID 104, dist^2 = 1.0 (duplicate of 101)
        ],
        dtype=np.float32,
    )
    ids = np.array([101, 102, 103, 104], dtype=np.int64)
    query = np.array([1.0, 1.0], dtype=np.float32)

    # k = 2: nearest are tied (101, 103, 104) -> tie-broken by ID -> 101, 103
    res_k2 = exact_search(query, database, ids=ids, k=2)
    assert isinstance(res_k2, SearchResult)
    np.testing.assert_array_equal(res_k2.ids, np.array([101, 103], dtype=np.int64))
    np.testing.assert_allclose(res_k2.distances, np.array([1.0, 1.0], dtype=np.float32))

    # k = 4: all 4 returned in order
    res_k4 = exact_search(query, database, ids=ids, k=4)
    np.testing.assert_array_equal(res_k4.ids, np.array([101, 103, 104, 102], dtype=np.int64))
    np.testing.assert_allclose(
        res_k4.distances, np.array([1.0, 1.0, 1.0, 34.0], dtype=np.float32)
    )


def test_single_query_output_shapes():
    """2. Single-query (1D array) output and result shapes."""
    database = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    query = np.array([2.0, 3.0], dtype=np.float32)

    result = exact_search(query, database, k=2)
    assert isinstance(result, SearchResult)
    assert result.distances.shape == (2,)
    assert result.ids.shape == (2,)
    assert result.distances.dtype == np.float32
    assert result.ids.dtype == np.int64


def test_batch_query_output_shapes():
    """3. Batch-query (2D array) output and result shapes."""
    database = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    queries = np.array([[2.0, 3.0], [1.0, 2.0]], dtype=np.float32)

    result = exact_search(queries, database, k=2)
    assert isinstance(result, SearchResult)
    assert result.distances.shape == (2, 2)
    assert result.ids.shape == (2, 2)
    assert result.distances.dtype == np.float32
    assert result.ids.dtype == np.int64


def test_non_contiguous_and_custom_ids():
    """4. Correct IDs when database IDs are non-contiguous and unsorted."""
    database = np.array(
        [
            [10.0, 10.0],  # pos 0 -> ID 9999
            [1.0, 1.0],    # pos 1 -> ID 42 (nearest to [0, 0])
            [5.0, 5.0],    # pos 2 -> ID 500
        ],
        dtype=np.float32,
    )
    ids = np.array([9999, 42, 500], dtype=np.int64)
    query = np.array([0.0, 0.0], dtype=np.float32)

    result = exact_search(query, database, ids=ids, k=3)
    np.testing.assert_array_equal(result.ids, np.array([42, 500, 9999], dtype=np.int64))


def test_deterministic_equal_distance_tie_breaking():
    """5. Deterministic tie-breaking with equal distances by ascending ID."""
    # 4 symmetric points equidistant to [0, 0]
    database = np.array(
        [
            [1.0, 0.0],   # dist^2 = 1.0, ID 50
            [-1.0, 0.0],  # dist^2 = 1.0, ID 10
            [0.0, 1.0],   # dist^2 = 1.0, ID 30
            [0.0, -1.0],  # dist^2 = 1.0, ID 20
        ],
        dtype=np.float32,
    )
    ids = np.array([50, 10, 30, 20], dtype=np.int64)
    query = np.array([0.0, 0.0], dtype=np.float32)

    result = exact_search(query, database, ids=ids, k=4)
    # Strictly ascending by ID: 10, 20, 30, 50
    np.testing.assert_array_equal(result.ids, np.array([10, 20, 30, 50], dtype=np.int64))
    np.testing.assert_allclose(result.distances, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))


def test_k_equals_1():
    """6. Minimal selection with k=1."""
    database = np.array([[5.0, 5.0], [1.0, 1.0], [9.0, 9.0]], dtype=np.float32)
    query_1d = np.array([0.0, 0.0], dtype=np.float32)
    query_2d = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)

    res_1d = exact_search(query_1d, database, k=1)
    assert res_1d.ids.shape == (1,)
    assert res_1d.ids[0] == 1  # default id for [1.0, 1.0]

    res_2d = exact_search(query_2d, database, k=1)
    assert res_2d.ids.shape == (2, 1)
    assert res_2d.ids[0, 0] == 1  # closest to [0, 0] is pos 1
    assert res_2d.ids[1, 0] == 2  # closest to [10, 10] is pos 2


def test_k_equals_n():
    """7. Full set selection with k=N."""
    database = np.array([[3.0, 3.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    query = np.array([0.0, 0.0], dtype=np.float32)

    result = exact_search(query, database, k=3)
    assert result.ids.shape == (3,)
    np.testing.assert_array_equal(result.ids, np.array([1, 2, 0], dtype=np.int64))


def test_k_greater_than_n():
    """8. Oversized query k > N returns all N records without error or padding."""
    database = np.array([[3.0, 3.0], [1.0, 1.0]], dtype=np.float32)
    query_1d = np.array([0.0, 0.0], dtype=np.float32)
    query_2d = np.array([[0.0, 0.0]], dtype=np.float32)

    res_1d = exact_search(query_1d, database, k=10)
    assert res_1d.ids.shape == (2,)
    assert res_1d.distances.shape == (2,)
    np.testing.assert_array_equal(res_1d.ids, np.array([1, 0], dtype=np.int64))

    res_2d = exact_search(query_2d, database, k=10)
    assert res_2d.ids.shape == (1, 2)
    assert res_2d.distances.shape == (1, 2)
    np.testing.assert_array_equal(res_2d.ids, np.array([[1, 0]], dtype=np.int64))


def test_empty_database():
    """9. Empty database behavior (N = 0)."""
    empty_db = np.empty((0, 3), dtype=np.float32)
    query_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    query_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    # 1D query
    res_1d = exact_search(query_1d, empty_db, k=5)
    assert res_1d.distances.shape == (0,)
    assert res_1d.ids.shape == (0,)
    assert res_1d.distances.dtype == np.float32
    assert res_1d.ids.dtype == np.int64

    # 2D batch query (Q=2, N=0)
    res_2d = exact_search(query_2d, empty_db, k=5)
    assert res_2d.distances.shape == (2, 0)
    assert res_2d.ids.shape == (2, 0)
    assert res_2d.distances.dtype == np.float32
    assert res_2d.ids.dtype == np.int64


def test_empty_query_batch():
    """10. Empty query batch behavior (Q = 0)."""
    database = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    empty_queries = np.empty((0, 2), dtype=np.float32)

    result = exact_search(empty_queries, database, k=5)
    assert result.distances.shape == (0, 2)
    assert result.ids.shape == (0, 2)
    assert result.distances.dtype == np.float32
    assert result.ids.dtype == np.int64


def test_invalid_dimensions_and_dtypes():
    """11. Input validation for dimensions, dtypes, non-finites, and invalid k."""
    valid_db = np.array([[1.0, 2.0]], dtype=np.float32)
    valid_q = np.array([1.0, 2.0], dtype=np.float32)

    # Dimension mismatch: d_q=3 != d_db=2
    with pytest.raises(ValueError):
        exact_search(np.array([1.0, 2.0, 3.0], dtype=np.float32), valid_db)

    # Non-float32 query
    with pytest.raises(TypeError):
        exact_search(np.array([1.0, 2.0], dtype=np.float64), valid_db)

    # Non-float32 database
    with pytest.raises(TypeError):
        exact_search(valid_q, np.array([[1.0, 2.0]], dtype=np.float64))

    # Non-finite numbers
    with pytest.raises(ValueError):
        exact_search(np.array([np.nan, 2.0], dtype=np.float32), valid_db)

    # Invalid k <= 0
    with pytest.raises(ValueError, match=r"^k must be positive$"):
        exact_search(valid_q, valid_db, k=0)

    with pytest.raises(ValueError, match=r"^k must be positive$"):
        exact_search(valid_q, valid_db, k=-1)

    # Non-int64 IDs
    with pytest.raises(TypeError):
        exact_search(valid_q, valid_db, ids=np.array([1], dtype=np.int32))

    # ID length mismatch
    with pytest.raises(ValueError):
        exact_search(valid_q, valid_db, ids=np.array([1, 2], dtype=np.int64))

    # Duplicate IDs
    with pytest.raises(ValueError, match=r"[uU]nique"):
        exact_search(
            valid_q,
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            ids=np.array([10, 10], dtype=np.int64),
        )


def test_consistency_with_independent_composition():
    """12. Consistency between exact_search output and independently composed kernel + topk."""
    rng = np.random.default_rng(seed=123)
    database = rng.standard_normal((50, 8)).astype(np.float32)
    ids = np.arange(1000, 1050, dtype=np.int64)
    queries = rng.standard_normal((5, 8)).astype(np.float32)
    k = 7

    # Directly call exact_search
    exact_res = exact_search(queries, database, ids=ids, k=k)

    # Independently compose reference kernel + select_top_k
    dist_matrix = pairwise_squared_l2_ref(queries, database)
    manual_results = [select_top_k(dist_matrix[i], ids, k=k) for i in range(5)]
    expected_dists = np.stack([r.distances for r in manual_results], axis=0)
    expected_ids = np.stack([r.ids for r in manual_results], axis=0)

    np.testing.assert_array_equal(exact_res.ids, expected_ids)
    np.testing.assert_array_equal(exact_res.distances, expected_dists)


def test_ground_truth_consistency():
    """13. Ground truth consistency with exact search."""
    rng = np.random.default_rng(seed=456)
    database = rng.standard_normal((30, 4)).astype(np.float32)
    queries = rng.standard_normal((3, 4)).astype(np.float32)
    ids = np.arange(100, 130, dtype=np.int64)
    k = 5

    gt_res = generate_ground_truth(queries, database, ids=ids, k=k)
    exact_res = exact_search(queries, database, ids=ids, k=k)

    np.testing.assert_array_equal(gt_res.ids, exact_res.ids)
    np.testing.assert_array_equal(gt_res.distances, exact_res.distances)


def test_exact_flat_index_class():
    """14. Test ExactFlatIndex wrapper class properties and search method."""
    database = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    ids = np.array([10, 20, 30], dtype=np.int64)

    index = ExactFlatIndex(database=database, ids=ids)
    assert index.n == 3
    assert index.d == 2
    assert len(index) == 3
    np.testing.assert_array_equal(index.database, database)
    np.testing.assert_array_equal(index.ids, ids)

    # Search with 1D query
    q = np.array([2.0, 2.0], dtype=np.float32)
    res = index.search(q, k=2)
    assert isinstance(res, SearchResult)
    assert res.ids.shape == (2,)
    assert res.ids[0] == 10  # closest is [1, 2]

    # Search with 2D query
    res_batch = index.search(np.array([[2.0, 2.0]], dtype=np.float32), k=2)
    assert res_batch.ids.shape == (1, 2)
    assert res_batch.ids[0, 0] == 10

    # Negative int64 ID support
    neg_index = ExactFlatIndex(
        database=database,
        ids=np.array([-50, 100, -200], dtype=np.int64),
    )
    assert neg_index.ids[0] == -50


def test_exact_flat_index_validation():
    """Verify ExactFlatIndex constructor validation for dtypes, shapes, and unique IDs."""
    valid_db = np.array([[1.0, 2.0]], dtype=np.float32)

    # Non-float32 database
    with pytest.raises(TypeError):
        ExactFlatIndex(database=np.array([[1.0, 2.0]], dtype=np.float64))

    # 1D database
    with pytest.raises(ValueError):
        ExactFlatIndex(database=np.array([1.0, 2.0], dtype=np.float32))

    # Duplicate IDs
    with pytest.raises(ValueError, match=r"[uU]nique"):
        ExactFlatIndex(
            database=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            ids=np.array([42, 42], dtype=np.int64),
        )

    # ID length mismatch
    with pytest.raises(ValueError):
        ExactFlatIndex(
            database=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            ids=np.array([42], dtype=np.int64),
        )


def test_exact_flat_index_empty_database():
    """Verify ExactFlatIndex behavior on empty database."""
    empty_db = np.empty((0, 4), dtype=np.float32)
    idx = ExactFlatIndex(database=empty_db)
    assert idx.n == 0
    assert idx.d == 4
    assert len(idx) == 0

    res = idx.search(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), k=3)
    assert res.ids.shape == (0,)
    assert res.distances.shape == (0,)
