"""Comprehensive test suite for filtered search across all Indexora index implementations.

Verifies caller-supplied eligibility filters (`filter_ids`) with sequences, sets,
NumPy arrays, and callables, ensuring strict deterministic tie-breaking, proper
empty/partial result handling, and correct interaction with deleted vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from indexora.exact import ExactBlasIndex, ExactFlatIndex
from indexora.hnsw import HNSWIndex
from indexora.ivf import IVFFlatIndex
from indexora.ivfpq import IVFPQIndex

INDEX_CLASSES = [
    lambda db, ids: ExactFlatIndex(database=db, ids=ids),
    lambda db, ids: ExactBlasIndex(database=db, ids=ids),
    lambda db, ids: IVFFlatIndex(database=db, ids=ids, nlist=3, nprobe=3),
    lambda db, ids: IVFPQIndex(database=db, ids=ids, nlist=3, M=2, k_sub=4, nprobe=3),
    lambda db, ids: HNSWIndex(database=db, ids=ids, M=8, ef_construction=32, ef_search=32, random_seed=42),
]


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic dataset with distinct IDs including negative IDs."""
    rng = np.random.default_rng(12345)
    # 24 vectors of dim 8
    vectors = rng.standard_normal((24, 8)).astype(np.float32)
    ids = np.array(
        [-100, -50, -10, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        dtype=np.int64,
    )
    return vectors, ids


@pytest.mark.parametrize("index_builder", INDEX_CLASSES)
def test_filter_all_eligible_matches_unfiltered(dataset, index_builder):
    """When filter contains all index IDs, results must match unfiltered search."""
    vectors, ids = dataset
    idx = index_builder(vectors, ids)

    q = vectors[0]
    res_unfiltered = idx.search(q, k=5)
    res_filtered = idx.search(q, k=5, filter_ids=set(ids))

    np.testing.assert_array_equal(res_filtered.ids, res_unfiltered.ids)
    np.testing.assert_allclose(res_filtered.distances, res_unfiltered.distances, atol=1e-5)


@pytest.mark.parametrize("index_builder", INDEX_CLASSES)
def test_filter_none_eligible_returns_empty(dataset, index_builder):
    """When filter matches no IDs, search must return empty result with valid shapes."""
    vectors, ids = dataset
    idx = index_builder(vectors, ids)

    # Filter with IDs outside the index
    filter_set = {99999, 88888}

    # Single query
    res_single = idx.search(vectors[0], k=5, filter_ids=filter_set)
    assert len(res_single.ids) == 0
    assert len(res_single.distances) == 0
    assert len(res_single.mask) == 0

    # Batch queries
    res_batch = idx.search(vectors[:3], k=5, filter_ids=filter_set)
    assert res_batch.ids.shape == (3, 0)
    assert res_batch.distances.shape == (3, 0)
    assert res_batch.mask.shape == (3, 0)


@pytest.mark.parametrize("index_builder", INDEX_CLASSES)
def test_filter_fewer_than_k_eligible(dataset, index_builder):
    """When fewer than k vectors match filter, return only the matching count."""
    vectors, ids = dataset
    idx = index_builder(vectors, ids)

    # Only 2 vectors eligible
    allowed = {-100, 40}
    res = idx.search(vectors[0], k=5, filter_ids=allowed)

    assert len(res.ids) == 2
    assert set(res.ids.tolist()) == allowed
    assert np.all(res.mask)


@pytest.mark.parametrize("index_builder", INDEX_CLASSES)
def test_filter_excludes_true_nearest_neighbor(dataset, index_builder):
    """Filter should skip the geometric top-1 neighbor and return the best allowed vector."""
    vectors, ids = dataset
    idx = index_builder(vectors, ids)

    q = vectors[0]
    true_top1_id = ids[0]

    # Filter allows everything except the true nearest neighbor
    allowed = set(ids) - {true_top1_id}
    res = idx.search(q, k=3, filter_ids=allowed)

    assert true_top1_id not in res.ids
    assert len(res.ids) == 3
    for cand_id in res.ids:
        assert cand_id in allowed


@pytest.mark.parametrize("index_builder", INDEX_CLASSES)
@pytest.mark.parametrize(
    "filter_arg_fn",
    [
        lambda ids: list(ids[:4]),
        lambda ids: set(ids[:4]),
        lambda ids: np.array(ids[:4], dtype=np.int64),
        lambda ids: (lambda x: x in set(ids[:4])),
    ],
)
def test_filter_supported_container_types(dataset, index_builder, filter_arg_fn):
    """Test that list, set, numpy array, and callable filters yield identical results."""
    vectors, ids = dataset
    idx = index_builder(vectors, ids)

    q = vectors[0]
    filter_arg = filter_arg_fn(ids)
    res = idx.search(q, k=3, filter_ids=filter_arg)

    expected_allowed = set(ids[:4])
    assert len(res.ids) <= 3
    assert len(res.ids) > 0
    for cand_id in res.ids:
        assert cand_id in expected_allowed


@pytest.mark.parametrize("index_builder", INDEX_CLASSES)
def test_filter_with_deleted_vector(dataset, index_builder):
    """A deleted vector should NEVER be returned even if it matches filter_ids."""
    vectors, ids = dataset
    idx = index_builder(vectors, ids)

    target_id = ids[0]
    # Delete the target ID
    idx.delete([target_id])

    # Search with filter that explicitly includes target_id
    res = idx.search(vectors[0], k=3, filter_ids={target_id, ids[1], ids[2]})

    assert target_id not in res.ids
    # Should only return ids[1] and ids[2]
    assert set(res.ids.tolist()) == {ids[1], ids[2]}


def test_filter_tied_distance_deterministic_ordering():
    """Verify that filtering preserves deterministic (distance, ID) ordering under exact ties."""
    # 3 duplicate vectors at identical distance to origin
    db = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=np.float32)
    ids = np.array([50, -20, 10], dtype=np.int64)

    idx = ExactFlatIndex(database=db, ids=ids)
    q = np.array([0.0, 0.0], dtype=np.float32)

    # Filter to {50, -20}
    res = idx.search(q, k=2, filter_ids={50, -20})
    # Tie breaking by external ID: -20 < 50
    np.testing.assert_array_equal(res.ids, np.array([-20, 50], dtype=np.int64))
