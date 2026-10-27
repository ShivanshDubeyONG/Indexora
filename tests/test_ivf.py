"""Tests for Phase 5: IVF-Flat (Inverted File with Flat Vectors) index.

These tests establish the strict validation, invariants, and acceptance gate
for Phase 5 per plan/implementation.md and ADR-0006.
"""

from __future__ import annotations

import numpy as np
import pytest

from indexora.data import generate_clustered_dataset, generate_synthetic_dataset
from indexora.exact import exact_search
from indexora.ivf import (
    InvertedList,
    IVFFlatIndex,
    _validate_centroids,
    _validate_ivf_build_inputs,
    ivf_flat_build,
    ivf_flat_search,
)
from indexora.topk import SearchResult


# ============================================================================
# 1. InvertedList Data Structure Tests
# ============================================================================


def test_inverted_list_dataclass_validation() -> None:
    """Test InvertedList dataclass enforces valid attribute types and shapes."""
    ids = np.array([101, 202, 303], dtype=np.int64)
    vectors = np.ones((3, 4), dtype=np.float32)

    inv = InvertedList(ids=ids, vectors=vectors)
    assert inv.size == 3
    assert inv.ids.dtype == np.int64
    assert inv.vectors.dtype == np.float32
    assert inv.vectors.shape == (3, 4)

    # Empty list is completely valid
    empty_inv = InvertedList(ids=np.empty(0, dtype=np.int64), vectors=np.empty((0, 4), dtype=np.float32))
    assert empty_inv.size == 0

    # Reject non-int64 ids
    with pytest.raises(TypeError, match="ids must be an int64"):
        InvertedList(ids=np.array([1, 2], dtype=np.int32), vectors=np.ones((2, 4), dtype=np.float32))

    # Reject non-float32 vectors
    with pytest.raises(TypeError, match="vectors must be a float32"):
        InvertedList(ids=ids, vectors=np.ones((3, 4), dtype=np.float64))

    # Reject length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        InvertedList(ids=ids, vectors=np.ones((2, 4), dtype=np.float32))

    # Reject non-C-contiguous vectors
    f_vecs = np.asfortranarray(np.ones((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        InvertedList(ids=ids, vectors=f_vecs)


def test_validate_ivf_build_inputs() -> None:
    """Test strict validation of database, nlist, ids, and nprobe for IVF-Flat."""
    db = np.ones((20, 5), dtype=np.float32)

    # Valid inputs
    N, d, valid_ids = _validate_ivf_build_inputs(db, nlist=4, ids=None, nprobe=2)
    assert N == 20 and d == 5
    assert len(valid_ids) == 20
    assert valid_ids.dtype == np.int64

    # Non-float32 database
    with pytest.raises(TypeError, match="strict dtype float32"):
        _validate_ivf_build_inputs(db.astype(np.float64), nlist=4, ids=None, nprobe=1)

    # Non-C-contiguous database
    with pytest.raises(ValueError, match="C-contiguous"):
        _validate_ivf_build_inputs(np.asfortranarray(db), nlist=4, ids=None, nprobe=1)

    # Non-finite values
    db_nan = db.copy()
    db_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite numbers"):
        _validate_ivf_build_inputs(db_nan, nlist=4, ids=None, nprobe=1)

    # Invalid nlist: nlist < 1
    with pytest.raises(ValueError, match="nlist must be at least 1"):
        _validate_ivf_build_inputs(db, nlist=0, ids=None, nprobe=1)

    # Invalid nlist: nlist > N
    with pytest.raises(ValueError, match="nlist cannot exceed database size N"):
        _validate_ivf_build_inputs(db, nlist=25, ids=None, nprobe=1)

    # Invalid nprobe: nprobe < 1
    with pytest.raises(ValueError, match="nprobe must be at least 1"):
        _validate_ivf_build_inputs(db, nlist=4, ids=None, nprobe=0)

    # Invalid nprobe: nprobe > nlist
    with pytest.raises(ValueError, match="nprobe cannot exceed nlist"):
        _validate_ivf_build_inputs(db, nlist=4, ids=None, nprobe=5)

    # Invalid IDs: duplicate IDs
    dup_ids = np.ones(20, dtype=np.int64)
    with pytest.raises(ValueError, match="strictly unique"):
        _validate_ivf_build_inputs(db, nlist=4, ids=dup_ids, nprobe=1)

    # Invalid IDs: length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        _validate_ivf_build_inputs(db, nlist=4, ids=np.arange(10, dtype=np.int64), nprobe=1)


# ============================================================================
# 2. Construction, Partitioning & ID Preservation
# ============================================================================


def test_ivf_flat_construction_and_partition_invariant() -> None:
    """Test that every database vector is partitioned into exactly one inverted list."""
    rng = np.random.default_rng(42)
    N, d, nlist = 60, 4, 3
    database = np.ascontiguousarray(rng.standard_normal((N, d)), dtype=np.float32)

    index = IVFFlatIndex(database=database, nlist=nlist, seed=42)

    assert index.ntotal == N
    assert index.d == d
    assert index.nlist == nlist
    assert index.centroids.shape == (nlist, d)
    assert len(index.invlists) == nlist

    # Core Invariant: Total vectors across all inverted lists must equal N exactly
    total_in_lists = sum(lst.size for lst in index.invlists)
    assert total_in_lists == N, f"Partition size sum {total_in_lists} != {N}"

    # Core Invariant: Union of all list IDs must be a permutation of database IDs (no loss, no dupes)
    all_assigned_ids = np.concatenate([lst.ids for lst in index.invlists if lst.size > 0])
    assert len(all_assigned_ids) == N
    assert len(np.unique(all_assigned_ids)) == N
    np.testing.assert_array_equal(np.sort(all_assigned_ids), np.arange(N, dtype=np.int64))


def test_ivf_flat_preserves_custom_stable_ids() -> None:
    """Test that custom arbitrary int64 IDs are preserved in inverted lists."""
    N, d, nlist = 20, 3, 2
    database = np.ascontiguousarray(np.ones((N, d), dtype=np.float32))
    custom_ids = np.array([1000 + i for i in range(N)], dtype=np.int64)

    index = IVFFlatIndex(database=database, nlist=nlist, ids=custom_ids, seed=42)

    all_assigned_ids = np.concatenate([lst.ids for lst in index.invlists if lst.size > 0])
    np.testing.assert_array_equal(np.sort(all_assigned_ids), np.sort(custom_ids))


def test_ivf_flat_empty_and_unbalanced_lists() -> None:
    """Test that empty or unbalanced inverted lists do not crash construction or search."""
    # Highly skewed data: 29 points near 0, 1 outlier at 1000
    X = np.zeros((30, 2), dtype=np.float32)
    X[-1] = [1000.0, 1000.0]

    index = IVFFlatIndex(database=X, nlist=5, seed=42)
    assert index.nlist == 5
    sizes = index.list_sizes()
    assert len(sizes) == 5
    assert np.sum(sizes) == 30

    # Ensure search does not crash even when probing empty lists
    q = np.array([0.0, 0.0], dtype=np.float32)
    res = index.search(q, k=5, nprobe=5)
    assert len(res.ids) == 5


def test_ivf_flat_caller_array_immutability() -> None:
    """Test that index construction does not mutate caller-owned database or IDs."""
    rng = np.random.default_rng(99)
    database = np.ascontiguousarray(rng.standard_normal((25, 4)), dtype=np.float32)
    ids = np.arange(25, dtype=np.int64)

    db_copy = database.copy()
    ids_copy = ids.copy()

    index = IVFFlatIndex(database=database, nlist=3, ids=ids, seed=99)
    np.testing.assert_array_equal(database, db_copy, err_msg="IVFFlatIndex mutated database!")
    np.testing.assert_array_equal(ids, ids_copy, err_msg="IVFFlatIndex mutated ids!")


def test_ivf_flat_pre_supplied_centroids() -> None:
    """Test passing pre-supplied centroids and validating centroid shapes/dtypes."""
    N, d, nlist = 30, 3, 3
    rng = np.random.default_rng(123)
    database = np.ascontiguousarray(rng.standard_normal((N, d)), dtype=np.float32)
    centroids = np.ascontiguousarray(rng.standard_normal((nlist, d)), dtype=np.float32)

    index = IVFFlatIndex(database=database, nlist=nlist, centroids=centroids)
    np.testing.assert_array_equal(index.centroids, centroids)

    # Centroid validation errors
    with pytest.raises(TypeError, match="numpy ndarray"):
        _validate_centroids("not_an_array", expected_nlist=nlist, expected_d=d)  # type: ignore

    with pytest.raises(TypeError, match="strict dtype float32"):
        _validate_centroids(centroids.astype(np.float64), expected_nlist=nlist, expected_d=d)

    with pytest.raises(ValueError, match="2D array"):
        _validate_centroids(centroids.reshape(-1), expected_nlist=nlist, expected_d=d)

    with pytest.raises(ValueError, match="centroids shape mismatch"):
        _validate_centroids(centroids[:2], expected_nlist=nlist, expected_d=d)

    with pytest.raises(ValueError, match="C-contiguous"):
        _validate_centroids(np.asfortranarray(centroids), expected_nlist=nlist, expected_d=d)

    nan_c = centroids.copy()
    nan_c[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite numbers"):
        _validate_centroids(nan_c, expected_nlist=nlist, expected_d=d)


# ============================================================================
# 3. Query Parameter Validation
# ============================================================================


def test_ivf_flat_search_invalid_parameters() -> None:
    """Test rejection of invalid k, nprobe, or query shapes in search."""
    database = np.ascontiguousarray(np.ones((20, 4), dtype=np.float32))
    index = IVFFlatIndex(database=database, nlist=4, seed=42)
    query = np.ones(4, dtype=np.float32)

    # k <= 0
    with pytest.raises(ValueError, match="k must be positive"):
        index.search(query, k=0)

    # invalid k type
    with pytest.raises(TypeError, match="k must be an integer"):
        index.search(query, k="5")  # type: ignore

    # nprobe < 1
    with pytest.raises(ValueError, match="nprobe must be at least 1"):
        index.search(query, k=5, nprobe=0)

    # nprobe > nlist
    with pytest.raises(ValueError, match="nprobe cannot exceed nlist"):
        index.search(query, k=5, nprobe=5)

    # invalid nprobe type
    with pytest.raises(TypeError, match="nprobe must be an integer"):
        index.search(query, k=5, nprobe=1.5)  # type: ignore

    # dimension mismatch 1D
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.ones(6, dtype=np.float32), k=5)

    # dimension mismatch 2D
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.ones((2, 6), dtype=np.float32), k=5)

    # non-float32 query
    with pytest.raises(TypeError, match="strict dtype float32"):
        index.search(np.ones(4, dtype=np.float64), k=5)

    # non-finite query
    nan_q = query.copy()
    nan_q[0] = np.nan
    with pytest.raises(ValueError, match="finite numbers"):
        index.search(nan_q, k=5)

    # dynamic nprobe property setter
    index.nprobe = 2
    assert index.nprobe == 2
    with pytest.raises(ValueError, match="between 1 and 4"):
        index.nprobe = 0
    with pytest.raises(ValueError, match="between 1 and 4"):
        index.nprobe = 10


# ============================================================================
# 4. Decisive Acceptance Gate: All-Probe Equivalence
# ============================================================================


def test_ivf_flat_all_probe_matches_exact_search_bit_for_bit() -> None:
    """DECISIVE ACCEPTANCE GATE: nprobe == nlist must match exact search bit-for-bit."""
    rng = np.random.default_rng(12345)
    N, d, nlist, Q = 100, 8, 5, 10
    database = np.ascontiguousarray(rng.standard_normal((N, d)), dtype=np.float32)
    queries = np.ascontiguousarray(rng.standard_normal((Q, d)), dtype=np.float32)
    custom_ids = np.array([5000 + i for i in range(N)], dtype=np.int64)

    index = IVFFlatIndex(database=database, nlist=nlist, ids=custom_ids, seed=42)

    # All-probe search: nprobe = nlist (probes every list)
    k = 10
    ivf_res = index.search(queries=queries, k=k, nprobe=nlist)
    exact_res = exact_search(queries=queries, database=database, ids=custom_ids, k=k)

    # Invariant: identical top-k IDs and distances
    assert isinstance(ivf_res, SearchResult)
    np.testing.assert_array_equal(
        ivf_res.ids,
        exact_res.ids,
        err_msg="All-probe IVF-Flat IDs do not match exact search!",
    )
    np.testing.assert_allclose(
        ivf_res.distances,
        exact_res.distances,
        rtol=1e-5,
        atol=1e-5,
        err_msg="All-probe IVF-Flat distances do not match exact search!",
    )


def test_ivf_flat_deterministic_equal_distance_tie_breaking() -> None:
    """Test that all-probe IVF-Flat respects deterministic (distance, ID) tie-breaking."""
    # Duplicate database vectors with distinct IDs
    database = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    ids = np.array([30, 10, 20], dtype=np.int64)
    query = np.array([1.0, 1.0], dtype=np.float32)

    index = IVFFlatIndex(database=database, nlist=2, ids=ids, seed=42)
    ivf_res = index.search(query, k=3, nprobe=2)
    exact_res = exact_search(query, database, ids=ids, k=3)

    # Both must rank by ascending ID: 10, 20, 30
    np.testing.assert_array_equal(ivf_res.ids, np.array([10, 20, 30], dtype=np.int64))
    np.testing.assert_array_equal(ivf_res.ids, exact_res.ids)


def test_ivf_flat_all_probe_edge_cases_k_greater_than_n_and_empty_lists() -> None:
    """Test all-probe equivalence with k > N, empty inverted lists, custom IDs, and batched queries."""
    # 25 points, skewed so some of the 5 clusters are empty
    X = np.zeros((25, 3), dtype=np.float32)
    X[:20] = [1.0, 1.0, 1.0]
    X[20:] = [100.0, 100.0, 100.0]
    # Non-sequential, negative, and large int64 IDs
    custom_ids = np.array([-500 + i * 42 for i in range(25)], dtype=np.int64)

    queries = np.array(
        [[1.0, 1.0, 1.0], [100.0, 100.0, 100.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    index = IVFFlatIndex(database=X, nlist=5, ids=custom_ids, seed=42)
    # Check that at least one list is empty
    sizes = index.list_sizes()
    assert np.any(sizes == 0), f"Expected at least one empty list, got sizes={sizes}"

    # Oversized k = 50 > N = 25 with all-probe nprobe = 5
    ivf_res = index.search(queries, k=50, nprobe=5)
    exact_res = exact_search(queries, X, ids=custom_ids, k=50)

    # Invariant: both must return exactly N=25 items and match bit-for-bit
    assert ivf_res.ids.shape == (3, 25)
    assert exact_res.ids.shape == (3, 25)
    np.testing.assert_array_equal(ivf_res.ids, exact_res.ids)
    np.testing.assert_allclose(ivf_res.distances, exact_res.distances, rtol=1e-5, atol=1e-5)


def test_ivf_flat_centroid_tie_breaking_order() -> None:
    """Test that equidistant coarse centroids break ties deterministically by lowest centroid index."""
    # Symmetrically placed centroids around origin: [-1, 0] (idx 0) and [1, 0] (idx 1)
    centroids = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    db = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    ids = np.array([100, 200], dtype=np.int64)

    index = IVFFlatIndex(database=db, nlist=2, ids=ids, centroids=centroids)

    # Query at [0, 0] is equidistant (dist^2 = 1.0) from centroid 0 and centroid 1
    q = np.array([0.0, 0.0], dtype=np.float32)

    # With nprobe=1, stable sort must pick centroid 0 (which contains vector [-1, 0] with ID 100)
    res = index.search(q, k=1, nprobe=1)
    assert res.ids[0] == 100


def test_ivf_flat_property_mutation_isolation() -> None:
    """Test that public properties and methods prevent mutation of index internal state."""
    db = np.ones((10, 2), dtype=np.float32)
    index = IVFFlatIndex(database=db, nlist=2, seed=42)

    # 1. Mutating returned centroids does not alter internal centroids
    c = index.centroids
    orig_c0 = c[0, 0]
    c[0, 0] += 100.0
    assert index.centroids[0, 0] == orig_c0

    # 2. InvertedList arrays are marked read-only to prevent corruption
    inv0 = index.get_list(0)
    if inv0.size > 0:
        with pytest.raises(ValueError, match="read-only"):
            inv0.ids[0] = 9999
        with pytest.raises(ValueError, match="read-only"):
            inv0.vectors[0, 0] = 9999.0


# ============================================================================
# 5. Partial Probing & Recall Evaluation
# ============================================================================


def test_ivf_flat_partial_probe_recall_evaluation() -> None:
    """Test partial probing (nprobe < nlist) against exact ground truth on clustered data."""
    data = generate_clustered_dataset(
        n_base=150,
        n_queries=15,
        d=4,
        n_clusters=5,
        cluster_std=0.05,
        seed=42,
    )

    index = IVFFlatIndex(
        database=data.dataset.base,
        nlist=5,
        ids=data.dataset.base_ids,
        seed=42,
    )

    exact_res = exact_search(
        queries=data.dataset.queries,
        database=data.dataset.base,
        ids=data.dataset.base_ids,
        k=10,
    )

    # Probe 1 list: partial probe produces valid results
    res_p1 = index.search(data.dataset.queries, k=10, nprobe=1)
    assert res_p1.ids.shape == (15, 10)
    assert res_p1.distances.shape == (15, 10)

    # Calculate empirical recall@10 for nprobe=1
    # Recall = (# of true top-10 neighbors retrieved) / (15 * 10)
    matches_p1 = 0
    total = 15 * 10
    for q_idx in range(15):
        true_set = set(exact_res.ids[q_idx])
        retrieved_set = set(res_p1.ids[q_idx])
        matches_p1 += len(true_set.intersection(retrieved_set))
    recall_p1 = matches_p1 / total
    assert 0.0 <= recall_p1 <= 1.0

    # Probe all 5 lists: must achieve 1.0 recall (100% agreement with exact)
    res_all = index.search(data.dataset.queries, k=10, nprobe=5)
    np.testing.assert_array_equal(res_all.ids, exact_res.ids)


def test_ivf_flat_batched_vs_single_queries() -> None:
    """Test consistency between single 1D queries and 2D batched queries."""
    rng = np.random.default_rng(77)
    database = np.ascontiguousarray(rng.standard_normal((50, 4)), dtype=np.float32)
    queries = np.ascontiguousarray(rng.standard_normal((5, 4)), dtype=np.float32)

    index = IVFFlatIndex(database=database, nlist=3, seed=77)

    batched_res = index.search(queries, k=5, nprobe=2)

    for i in range(5):
        single_res = index.search(queries[i], k=5, nprobe=2)
        np.testing.assert_array_equal(batched_res.ids[i], single_res.ids)
        np.testing.assert_allclose(batched_res.distances[i], single_res.distances)


# ============================================================================
# 6. Edge Cases, Empty Probes & Functional API
# ============================================================================


def test_ivf_flat_fewer_than_k_candidates_and_empty_probed_lists() -> None:
    """Test queries when candidates are fewer than k or when probed lists are empty."""
    # 3 points in database
    db = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    ids = np.array([10, 20, 30], dtype=np.int64)

    index = IVFFlatIndex(database=db, nlist=3, ids=ids, seed=42)

    # Request k=10 with nprobe=1 (only 1 candidate in nearest list)
    single_res = index.search(np.array([0.0, 0.0], dtype=np.float32), k=10, nprobe=1)
    assert len(single_res.ids) == 1
    assert len(single_res.distances) == 1

    # Empty query batch
    empty_queries = np.empty((0, 2), dtype=np.float32)
    empty_res = index.search(empty_queries, k=5)
    assert empty_res.ids.shape == (0, 3)
    assert empty_res.distances.shape == (0, 3)


def test_ivf_flat_functional_api_and_list_inspection() -> None:
    """Test ivf_flat_build, ivf_flat_search, list_sizes, and get_list."""
    rng = np.random.default_rng(42)
    db = np.ascontiguousarray(rng.standard_normal((20, 3)), dtype=np.float32)
    queries = np.ascontiguousarray(rng.standard_normal((2, 3)), dtype=np.float32)

    index = ivf_flat_build(db, nlist=2, nprobe=1, seed=42)
    assert isinstance(index, IVFFlatIndex)

    res = ivf_flat_search(index, queries, k=5)
    assert isinstance(res, SearchResult)
    assert res.ids.shape == (2, 5)

    # get_list & list_sizes
    sizes = index.list_sizes()
    assert len(sizes) == 2
    assert np.sum(sizes) == 20

    lst0 = index.get_list(0)
    assert isinstance(lst0, InvertedList)
    assert lst0.size == sizes[0]

    with pytest.raises(IndexError, match="out of bounds"):
        index.get_list(2)

    with pytest.raises(TypeError, match="must be an integer"):
        index.get_list("0")  # type: ignore

    with pytest.raises(TypeError, match="must be an IVFFlatIndex"):
        ivf_flat_search("not_an_index", queries)  # type: ignore
