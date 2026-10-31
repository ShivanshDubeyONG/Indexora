"""Tests for Phase 2: Exact Flat Search and Ground Truth.

These tests verify `exact_search`, `ExactFlatIndex`, and `generate_ground_truth`
against the contracts in `plan/project srs.md`, `plan/tech design.md`, and
`docs/decisions.md`.
"""

import numpy as np
import pytest

from indexora import (
    ExactBlasIndex,
    ExactFlatIndex,
    SearchResult,
    exact_search,
    exact_search_blas,
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


def test_exact_search_chunk_size_invariance():
    """Verify exact_search produces bit-for-bit identical results across multiple chunk sizes."""
    rng = np.random.default_rng(707)
    N, d = 80, 8
    database = rng.standard_normal((N, d)).astype(np.float32)
    ids = rng.permutation(np.arange(1000, 1000 + N, dtype=np.int64))

    single_q = rng.standard_normal((d,)).astype(np.float32)
    batch_q = rng.standard_normal((5, d)).astype(np.float32)

    k = 10
    baseline_single = exact_search(single_q, database, ids=ids, k=k, max_db_chunk=10000)
    baseline_batch = exact_search(batch_q, database, ids=ids, k=k, max_db_chunk=10000)

    for chunk_size in [1, 3, 7, 19, 40, 80]:
        res_single = exact_search(single_q, database, ids=ids, k=k, max_db_chunk=chunk_size)
        res_batch = exact_search(batch_q, database, ids=ids, k=k, max_db_chunk=chunk_size)

        np.testing.assert_array_equal(res_single.ids, baseline_single.ids)
        np.testing.assert_array_equal(res_single.distances, baseline_single.distances)

        np.testing.assert_array_equal(res_batch.ids, baseline_batch.ids)
        np.testing.assert_array_equal(res_batch.distances, baseline_batch.distances)


def test_exact_search_incremental_equivalence_with_boundary_ties():
    """Verify incremental merging preserves exact deterministic tie-breaking across chunk boundaries."""
    # Place tied vectors across different blocks of size 2
    database = np.array(
        [
            [1.0, 1.0],  # dist^2 = 0.0, ID 500 (block 0)
            [2.0, 2.0],  # dist^2 = 2.0, ID 200 (block 0)
            [2.0, 2.0],  # dist^2 = 2.0, ID 100 (block 1) - tied with above, smaller ID
            [2.0, 2.0],  # dist^2 = 2.0, ID 300 (block 1) - tied with above
            [5.0, 5.0],  # dist^2 = 32.0, ID 50 (block 2)
        ],
        dtype=np.float32,
    )
    ids = np.array([500, 200, 100, 300, 50], dtype=np.int64)
    query = np.array([1.0, 1.0], dtype=np.float32)

    # With k=3: should get ID 500 (dist 0), then among the 3 tied at dist 2.0, IDs 100 and 200
    res_chunked = exact_search(query, database, ids=ids, k=3, max_db_chunk=2)
    res_full = exact_search(query, database, ids=ids, k=3, max_db_chunk=10)

    expected_ids = np.array([500, 100, 200], dtype=np.int64)
    expected_dists = np.array([0.0, 2.0, 2.0], dtype=np.float32)

    np.testing.assert_array_equal(res_chunked.ids, expected_ids)
    np.testing.assert_array_equal(res_chunked.distances, expected_dists)
    np.testing.assert_array_equal(res_chunked.ids, res_full.ids)
    np.testing.assert_array_equal(res_chunked.distances, res_full.distances)


def test_exact_search_max_db_chunk_validation():
    """Verify validation on invalid max_db_chunk parameters."""
    q = np.array([1.0, 2.0], dtype=np.float32)
    db = np.array([[1.0, 2.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="max_db_chunk must be a positive integer"):
        exact_search(q, db, max_db_chunk=0)

    with pytest.raises(ValueError, match="max_db_chunk must be a positive integer"):
        exact_search(q, db, max_db_chunk=-5)


def test_exact_search_incremental_edge_cases_comprehensive():
    """Verify incremental search on k > N, empty queries, non-sequential IDs, and chunk-spanning ties."""
    rng = np.random.default_rng(808)
    N, d = 25, 4
    database = rng.standard_normal((N, d)).astype(np.float32)
    ids = rng.permutation(np.arange(100, 100 + N, dtype=np.int64))

    # 1. k > N with small chunk size (max_db_chunk = 6)
    k_oversized = 50
    q_single = rng.standard_normal((d,)).astype(np.float32)
    res_k_over = exact_search(q_single, database, ids=ids, k=k_oversized, max_db_chunk=6)
    res_k_over_full = exact_search(q_single, database, ids=ids, k=k_oversized, max_db_chunk=1000)
    assert len(res_k_over.ids) == N
    np.testing.assert_array_equal(res_k_over.ids, res_k_over_full.ids)
    np.testing.assert_allclose(res_k_over.distances, res_k_over_full.distances)

    # 2. Empty queries with chunked database
    empty_q = np.empty((0, d), dtype=np.float32)
    res_empty_q = exact_search(empty_q, database, ids=ids, k=5, max_db_chunk=5)
    assert res_empty_q.ids.shape == (0, 5)
    assert res_empty_q.distances.shape == (0, 5)

    # 3. Batched queries with k > N and k < N
    q_batch = rng.standard_normal((4, d)).astype(np.float32)
    res_batch_chunk = exact_search(q_batch, database, ids=ids, k=10, max_db_chunk=5)
    res_batch_full = exact_search(q_batch, database, ids=ids, k=10, max_db_chunk=1000)
    np.testing.assert_array_equal(res_batch_chunk.ids, res_batch_full.ids)
    np.testing.assert_allclose(res_batch_chunk.distances, res_batch_full.distances)


# ============================================================================
# Phase 6.6: Fast BLAS-Accelerated Exact Search & Query Chunking Tests
# ============================================================================


def test_exact_search_blas_parity_with_reference():
    """Verify exact_search_blas returns identical neighbor IDs and distances as exact_search."""
    rng = np.random.default_rng(42)
    N, d, Q = 100, 16, 10
    database = rng.standard_normal((N, d)).astype(np.float32)
    queries = rng.standard_normal((Q, d)).astype(np.float32)
    ids = np.arange(1000, 1000 + N, dtype=np.int64)

    ref_res = exact_search(queries, database, ids=ids, k=10)
    blas_res = exact_search_blas(queries, database, ids=ids, k=10)

    np.testing.assert_array_equal(blas_res.ids, ref_res.ids)
    np.testing.assert_allclose(blas_res.distances, ref_res.distances, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(blas_res.mask, ref_res.mask)
    assert blas_res.distance_computations == Q * N


def test_exact_blas_index_interface_and_properties():
    """Verify ExactBlasIndex provides identical memory, search single, and search batch behavior."""
    rng = np.random.default_rng(101)
    N, d = 50, 8
    database = rng.standard_normal((N, d)).astype(np.float32)
    ids = np.arange(N, dtype=np.int64)

    idx_ref = ExactFlatIndex(database, ids=ids)
    idx_blas = ExactBlasIndex(database, ids=ids)

    assert idx_blas.index_memory_bytes == idx_ref.index_memory_bytes
    assert idx_blas.index_memory_mib == idx_ref.index_memory_mib
    assert idx_blas.bytes_per_vector == idx_ref.bytes_per_vector
    assert idx_blas.database.shape == (N, d)

    # Single search
    q_single = rng.standard_normal(d).astype(np.float32)
    res_ref_s = idx_ref.search(q_single, k=5)
    res_blas_s = idx_blas.search(q_single, k=5)
    np.testing.assert_array_equal(res_blas_s.ids, res_ref_s.ids)
    np.testing.assert_allclose(res_blas_s.distances, res_ref_s.distances, rtol=1e-4, atol=1e-4)

    # Batch search
    q_batch = rng.standard_normal((5, d)).astype(np.float32)
    res_ref_b = idx_ref.search(q_batch, k=5)
    res_blas_b = idx_blas.search(q_batch, k=5)
    np.testing.assert_array_equal(res_blas_b.ids, res_ref_b.ids)
    np.testing.assert_allclose(res_blas_b.distances, res_ref_b.distances, rtol=1e-4, atol=1e-4)


def test_exact_search_query_chunking_bounds():
    """Verify max_query_chunk parameter bounds query batching without altering results."""
    rng = np.random.default_rng(202)
    N, d, Q = 80, 8, 25
    database = rng.standard_normal((N, d)).astype(np.float32)
    queries = rng.standard_normal((Q, d)).astype(np.float32)

    # Reference exact search across different query chunk sizes
    res_q1 = exact_search(queries, database, k=5, max_query_chunk=1)
    res_q7 = exact_search(queries, database, k=5, max_query_chunk=7)
    res_full = exact_search(queries, database, k=5, max_query_chunk=1000)

    np.testing.assert_array_equal(res_q1.ids, res_full.ids)
    np.testing.assert_array_equal(res_q7.ids, res_full.ids)

    # BLAS exact search across different query chunk sizes
    res_blas_q2 = exact_search_blas(queries, database, k=5, max_query_chunk=2)
    res_blas_full = exact_search_blas(queries, database, k=5, max_query_chunk=1000)

    np.testing.assert_array_equal(res_blas_q2.ids, res_blas_full.ids)
    np.testing.assert_array_equal(res_blas_q2.ids, res_full.ids)

    # Validation errors on invalid max_query_chunk
    with pytest.raises(ValueError, match="max_query_chunk must be a positive integer"):
        exact_search(queries, database, max_query_chunk=0)
    with pytest.raises(ValueError, match="max_query_chunk must be a positive integer"):
        exact_search_blas(queries, database, max_query_chunk=-3)


def test_exact_search_blocked_database_chunking_equivalence():
    """Verify blocked GEMM across database chunks matches reference oracle bit-for-bit."""
    rng = np.random.default_rng(303)
    N, d, Q = 150, 16, 20
    database = rng.standard_normal((N, d)).astype(np.float32)
    queries = rng.standard_normal((Q, d)).astype(np.float32)
    ids = np.arange(100, 100 + N, dtype=np.int64)

    # Reference exact search
    ref_res = exact_search(queries, database, ids=ids, k=10)

    # Blocked BLAS search with multiple chunk sizes smaller than N
    for db_chunk in [17, 30, 50, 75]:
        blas_res = exact_search_blas(queries, database, ids=ids, k=10, max_db_chunk=db_chunk)
        np.testing.assert_array_equal(blas_res.ids, ref_res.ids)
        np.testing.assert_allclose(blas_res.distances, ref_res.distances, rtol=1e-4, atol=1e-4)


def test_exact_search_blocked_gemm_deterministic_tie_breaking():
    """Verify equidistant candidates in different database chunks break ties by ascending ID."""
    # 4 points with identical distances to [0, 0] distributed across chunks of size 2
    database = np.array(
        [
            [1.0, 0.0],  # ID 999, dist=1.0 (chunk 0)
            [0.0, 1.0],  # ID 500, dist=1.0 (chunk 0)
            [-1.0, 0.0], # ID 20,  dist=1.0 (chunk 1)
            [0.0, -1.0], # ID 100, dist=1.0 (chunk 1)
        ],
        dtype=np.float32,
    )
    ids = np.array([999, 500, 20, 100], dtype=np.int64)
    query = np.array([[0.0, 0.0]], dtype=np.float32)

    # With max_db_chunk=2, chunk 0 has [999, 500], chunk 1 has [20, 100]
    ref_k2 = exact_search(query, database, ids=ids, k=2)
    blas_k2 = exact_search_blas(query, database, ids=ids, k=2, max_db_chunk=2)

    # Smallest IDs among tied candidates: 20, 100
    np.testing.assert_array_equal(blas_k2.ids, ref_k2.ids)
    np.testing.assert_array_equal(blas_k2.ids, np.array([[20, 100]], dtype=np.int64))

    ref_k4 = exact_search(query, database, ids=ids, k=4)
    blas_k4 = exact_search_blas(query, database, ids=ids, k=4, max_db_chunk=2)
    np.testing.assert_array_equal(blas_k4.ids, ref_k4.ids)
    np.testing.assert_array_equal(blas_k4.ids, np.array([[20, 100, 500, 999]], dtype=np.int64))


def test_exact_blas_index_add_delete_norm_consistency():
    """Verify ExactBlasIndex maintains exact norm consistency and search equivalence after add/delete."""
    rng = np.random.default_rng(404)
    N, d = 40, 8
    X = rng.standard_normal((N, d)).astype(np.float32)
    ids = np.arange(N, dtype=np.int64)

    index = ExactBlasIndex(X, ids=ids)
    np.testing.assert_allclose(index._db_norms, np.sum(index._database ** 2, axis=1), rtol=1e-5)

    # 1. Add vectors
    new_X = rng.standard_normal((10, d)).astype(np.float32)
    new_ids = np.arange(100, 110, dtype=np.int64)
    index.add(new_X, ids=new_ids)
    assert len(index) == 50
    np.testing.assert_allclose(index._db_norms, np.sum(index._database ** 2, axis=1), rtol=1e-5)

    # 2. Delete vectors
    deleted = index.delete([100, 101, 102, 5, 10])
    assert deleted == 5
    assert len(index) == 45
    np.testing.assert_allclose(index._db_norms, np.sum(index._database ** 2, axis=1), rtol=1e-5)

    # 3. Search equivalence against a fresh index and reference flat index
    queries = rng.standard_normal((5, d)).astype(np.float32)
    fresh_blas = ExactBlasIndex(index.database, ids=index.ids)
    fresh_flat = ExactFlatIndex(index.database, ids=index.ids)

    res_mut = index.search(queries, k=5, max_db_chunk=15)
    res_fresh = fresh_blas.search(queries, k=5, max_db_chunk=15)
    res_flat = fresh_flat.search(queries, k=5)

    np.testing.assert_array_equal(res_mut.ids, res_fresh.ids)
    np.testing.assert_array_equal(res_mut.ids, res_flat.ids)
    np.testing.assert_allclose(res_mut.distances, res_fresh.distances, rtol=1e-5)


def test_exact_search_blocked_randomized_stress():
    """Randomized stress tests comparing blocked GEMM BLAS against exact reference oracle."""
    rng = np.random.default_rng(505)
    for trial in range(5):
        N = rng.integers(60, 150)
        d = rng.integers(8, 24)
        Q = rng.integers(5, 15)
        k = rng.integers(3, 12)
        chunk = rng.integers(11, 35)

        db = rng.standard_normal((N, d)).astype(np.float32)
        queries = rng.standard_normal((Q, d)).astype(np.float32)
        ids = rng.choice(10000, size=N, replace=False).astype(np.int64)

        ref_res = exact_search(queries, db, ids=ids, k=k)
        blas_res = exact_search_blas(queries, db, ids=ids, k=k, max_db_chunk=chunk)

        np.testing.assert_array_equal(
            blas_res.ids,
            ref_res.ids,
            err_msg=f"Trial {trial} failed ID agreement: N={N}, d={d}, Q={Q}, k={k}, chunk={chunk}",
        )
        np.testing.assert_allclose(
            blas_res.distances,
            ref_res.distances,
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Trial {trial} failed distance agreement",
        )
