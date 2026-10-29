"""Tests for Phase 10: Hierarchical Navigable Small World (HNSW) vector search index."""

from __future__ import annotations

import numpy as np
import pytest

from indexora.eval import compute_recall_at_k
from indexora.exact import exact_search
from indexora.hnsw import HNSWIndex, hnsw_build, hnsw_search
from indexora.topk import SearchResult


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def small_dataset(rng):
    N, d = 200, 16
    db = rng.standard_normal((N, d), dtype=np.float32)
    db = np.ascontiguousarray(db)
    queries = rng.standard_normal((20, d), dtype=np.float32)
    queries = np.ascontiguousarray(queries)
    return db, queries


# -----------------------------------------------------------------------------
# Validation Tests
# -----------------------------------------------------------------------------

def test_hnsw_parameter_validation():
    db = np.zeros((10, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="Connectivity M must be an integer >= 2"):
        HNSWIndex(db, M=1)

    with pytest.raises(ValueError, match="ef_construction must be an integer >= 1"):
        HNSWIndex(db, ef_construction=0)

    with pytest.raises(ValueError, match="ef_search must be an integer >= 1"):
        HNSWIndex(db, ef_search=0)

    with pytest.raises(TypeError, match="database must be a numpy ndarray"):
        HNSWIndex([[1.0, 2.0]])  # type: ignore

    with pytest.raises(TypeError, match="strict dtype float32"):
        HNSWIndex(db.astype(np.float64))

    with pytest.raises(ValueError, match="database must be a 2D array"):
        HNSWIndex(np.zeros(10, dtype=np.float32))

    with pytest.raises(ValueError, match="database must contain only finite"):
        bad_db = db.copy()
        bad_db[0, 0] = np.nan
        HNSWIndex(bad_db)

    with pytest.raises(ValueError, match="ids must be strictly unique"):
        HNSWIndex(db, ids=np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64))


def test_hnsw_search_validation(small_dataset):
    db, queries = small_dataset
    idx = HNSWIndex(db, M=8, ef_construction=32)

    with pytest.raises(ValueError, match="k must be positive"):
        idx.search(queries, k=0)

    with pytest.raises(ValueError, match="ef_search must be at least 1"):
        idx.search(queries, k=5, ef_search=0)

    with pytest.raises(TypeError, match="queries must have strict dtype float32"):
        idx.search(queries.astype(np.float64), k=5)

    with pytest.raises(ValueError, match="Query feature dimension"):
        idx.search(np.zeros((5, db.shape[1] + 1), dtype=np.float32), k=5)

    with pytest.raises(ValueError, match="queries must contain only finite"):
        bad_q = queries.copy()
        bad_q[0, 0] = np.inf
        idx.search(bad_q, k=5)


# -----------------------------------------------------------------------------
# Graph Invariants & Structure
# -----------------------------------------------------------------------------

def test_hnsw_graph_invariants(small_dataset):
    db, _ = small_dataset
    N = db.shape[0]
    M = 8
    M0 = 2 * M
    idx = HNSWIndex(db, M=M, ef_construction=32, random_seed=42)

    assert idx.ntotal == N
    assert idx.d == db.shape[1]
    assert idx.M == M
    assert idx.M0 == M0
    assert idx.max_level >= 0
    assert idx.entry_point is not None
    assert 0 <= idx.entry_point < N

    # Verify entry point is present at max_level
    assert idx.node_level(idx.entry_point) == idx.max_level

    # Check invariants across all levels
    for lvl in range(idx.max_level + 1):
        m_limit = M0 if lvl == 0 else M
        for node in range(N):
            if idx.node_level(node) >= lvl:
                neighbors = idx.get_neighbors(node, lvl)
                # Degree bound invariant
                assert len(neighbors) <= m_limit, (
                    f"Node {node} at level {lvl} has {len(neighbors)} neighbors > {m_limit}"
                )
                # No self-loops
                assert node not in neighbors, f"Node {node} has self-loop at level {lvl}"
                # Valid neighbor node indices
                for nbr in neighbors:
                    assert 0 <= nbr < N, f"Neighbor {nbr} out of bounds"
                    assert idx.node_level(nbr) >= lvl, (
                        f"Neighbor {nbr} (level {idx.node_level(nbr)}) not in level {lvl}"
                    )


def test_hnsw_memory_accounting(small_dataset):
    db, _ = small_dataset
    N, d = db.shape
    idx = HNSWIndex(db, M=8, ef_construction=32)

    assert idx.vector_memory_bytes == N * d * 4
    assert idx.id_memory_bytes == N * 8
    assert idx.graph_edge_count > 0
    assert idx.graph_memory_bytes == idx.graph_edge_count * 8
    assert idx.index_memory_bytes == (
        idx.vector_memory_bytes + idx.id_memory_bytes + idx.graph_memory_bytes
    )
    assert idx.index_memory_mib == idx.index_memory_bytes / (1024.0 * 1024.0)
    assert idx.bytes_per_vector == idx.index_memory_bytes / N


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------

def test_hnsw_empty_database():
    empty_db = np.empty((0, 8), dtype=np.float32)
    idx = HNSWIndex(empty_db, M=4)

    assert idx.ntotal == 0
    assert idx.max_level == -1
    assert idx.entry_point is None
    assert idx.graph_edge_count == 0
    assert idx.bytes_per_vector == 0.0

    q_single = np.ones(8, dtype=np.float32)
    res_single = idx.search(q_single, k=5)
    assert isinstance(res_single, SearchResult)
    assert res_single.distances.shape == (0,)
    assert res_single.ids.shape == (0,)
    assert res_single.mask.shape == (0,)

    q_batch = np.ones((3, 8), dtype=np.float32)
    res_batch = idx.search(q_batch, k=5)
    assert res_batch.distances.shape == (3, 0)
    assert res_batch.ids.shape == (3, 0)
    assert res_batch.mask.shape == (3, 0)


def test_hnsw_single_vector():
    db = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    custom_id = np.array([999], dtype=np.int64)
    idx = HNSWIndex(db, ids=custom_id, M=4)

    assert idx.ntotal == 1
    assert idx.entry_point == 0
    assert idx.node_level(0) >= 0

    q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    res = idx.search(q, k=5)
    assert len(res.ids) == 1
    assert res.ids[0] == 999
    assert res.distances[0] == pytest.approx(0.0, abs=1e-6)
    assert res.mask[0] is True or res.mask[0] == 1


def test_hnsw_k_greater_than_database(small_dataset):
    db, queries = small_dataset
    N = db.shape[0]
    idx = HNSWIndex(db, M=8, ef_construction=32)

    # Search with k > N on approximate graph
    k = N + 50
    res = idx.search(queries[:2], k=k)
    assert res.ids.shape[0] == 2
    assert res.ids.shape[1] <= N
    assert res.ids.shape[1] >= N - 5  # Traverses candidate graph without error
    assert res.distances.shape == res.ids.shape
    assert np.all(res.mask)

    # Fully connected small database (N <= M0) returns exactly all N vectors
    db_small = db[:10]
    idx_small = HNSWIndex(db_small, M=8)
    res_small = idx_small.search(queries[:2], k=k)
    assert res_small.ids.shape == (2, 10)
    assert res_small.distances.shape == (2, 10)
    assert np.all(res_small.mask)


def test_hnsw_arbitrary_and_negative_ids():
    N, d = 10, 4
    db = np.arange(N * d, dtype=np.float32).reshape(N, d)
    # Include negative numbers and legitimate -1
    ids = np.array([-100, -50, -1, 0, 1, 2, 3, 4, 10, 100], dtype=np.int64)
    idx = HNSWIndex(db, ids=ids, M=4, ef_construction=16)

    # Exact match for the vector having ID -1
    q_minus_1 = db[2]
    res = idx.search(q_minus_1, k=1)
    assert res.ids[0] == -1
    assert res.distances[0] == pytest.approx(0.0, abs=1e-6)

    # Exact match for the vector having ID -100
    q_minus_100 = db[0]
    res_100 = idx.search(q_minus_100, k=1)
    assert res_100.ids[0] == -100


# -----------------------------------------------------------------------------
# Parity, Determinism & Accuracy
# -----------------------------------------------------------------------------

def test_hnsw_single_vs_batched_parity(small_dataset):
    db, queries = small_dataset
    idx = HNSWIndex(db, M=8, ef_construction=32, ef_search=32, random_seed=42)

    # Batched search
    res_batch = idx.search(queries, k=5, ef_search=32)

    # Single search per query
    for i in range(len(queries)):
        res_single = idx.search(queries[i], k=5, ef_search=32)
        np.testing.assert_array_equal(res_single.ids, res_batch.ids[i])
        np.testing.assert_allclose(res_single.distances, res_batch.distances[i], rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(res_single.mask, res_batch.mask[i])


def test_hnsw_duplicate_vectors_tie_breaking():
    # Construct identical duplicate vectors
    vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    db = np.repeat(vec[np.newaxis, :], 5, axis=0)
    ids = np.array([50, 10, 40, 20, 30], dtype=np.int64)

    idx = HNSWIndex(db, ids=ids, M=4, ef_construction=16, random_seed=42)
    res = idx.search(vec, k=5, ef_search=16)

    # Distances must all be 0.0
    np.testing.assert_allclose(res.distances, np.zeros(5, dtype=np.float32), atol=1e-6)
    # IDs should be deterministically ordered ascending by external ID: [10, 20, 30, 40, 50]
    np.testing.assert_array_equal(res.ids, np.array([10, 20, 30, 40, 50], dtype=np.int64))


def test_hnsw_approximate_recall(small_dataset):
    db, queries = small_dataset
    k = 10

    # Ground truth exact search
    gt = exact_search(queries, db, k=k)

    # Build HNSW with adequate parameters
    idx = HNSWIndex(db, M=16, ef_construction=64, random_seed=42)

    # Search with ef_search=32
    res = idx.search(queries, k=k, ef_search=32)
    recall = compute_recall_at_k(res.ids, gt.ids, k=k)

    # HNSW on 200 vectors should easily achieve high recall (> 95%)
    assert recall >= 0.95, f"Expected Recall@{k} >= 0.95, got {recall:.4f}"

    # Search with higher ef_search should maintain or improve recall
    res_high = idx.search(queries, k=k, ef_search=64)
    recall_high = compute_recall_at_k(res_high.ids, gt.ids, k=k)
    assert recall_high >= recall


def test_hnsw_functional_helpers(small_dataset):
    db, queries = small_dataset
    idx = hnsw_build(db, M=8, ef_construction=32, ef_search=16)
    assert isinstance(idx, HNSWIndex)

    res = hnsw_search(idx, queries[:3], k=5)
    assert isinstance(res, SearchResult)
    assert res.ids.shape == (3, 5)
