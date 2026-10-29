"""Tests for Phase 11: Vector Index Mutations (Addition & Deletion)."""

from __future__ import annotations

import numpy as np
import pytest

from indexora.exact import ExactBlasIndex, ExactFlatIndex
from indexora.hnsw import HNSWIndex
from indexora.ivf import IVFFlatIndex
from indexora.ivfpq import IVFPQIndex


@pytest.fixture
def base_vectors():
    # 20 vectors in 8 dimensions
    rng = np.random.default_rng(42)
    return rng.standard_normal((20, 8), dtype=np.float32)


@pytest.fixture
def base_ids():
    # Includes negative IDs and legitimate -1
    return np.array(
        [-100, -50, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70],
        dtype=np.int64,
    )


INDEX_CLASSES = [
    lambda db, ids: ExactFlatIndex(database=db, ids=ids),
    lambda db, ids: ExactBlasIndex(database=db, ids=ids),
    lambda db, ids: IVFFlatIndex(database=db, ids=ids, nlist=4, seed=42),
    lambda db, ids: IVFPQIndex(database=db, ids=ids, nlist=4, M=2, k_sub=16, seed=42),
    lambda db, ids: HNSWIndex(database=db, ids=ids, M=4, ef_construction=16, random_seed=42),
]


@pytest.mark.parametrize("index_builder", INDEX_CLASSES)
def test_mutations_add_and_search(base_vectors, base_ids, index_builder):
    """Test incremental addition of vectors and verify they become searchable."""
    idx = index_builder(base_vectors, base_ids)
    initial_count = len(idx) if hasattr(idx, "__len__") else idx.ntotal

    # New vector designed to be an exact match for query
    new_vec = np.ones((1, 8), dtype=np.float32) * 42.0
    new_id = np.array([9999], dtype=np.int64)

    added_ids = idx.add(new_vec, ids=new_id)
    np.testing.assert_array_equal(added_ids, new_id)

    # Search for new vector
    res = idx.search(new_vec[0], k=1)
    assert len(res.ids) == 1
    assert res.ids[0] == 9999
    if not isinstance(idx, IVFPQIndex):
        assert res.distances[0] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("index_builder", INDEX_CLASSES)
def test_mutations_delete_removes_from_search(base_vectors, base_ids, index_builder):
    """Test deletion by ID and verify deleted vectors never appear in search results."""
    idx = index_builder(base_vectors, base_ids)

    # Search before deletion: target ID -1 is in the database
    target_vec = base_vectors[2]
    res_before = idx.search(target_vec, k=1)
    assert res_before.ids[0] == -1

    # Delete ID -1
    deleted_count = idx.delete([-1])
    assert deleted_count == 1

    # Search after deletion: ID -1 must NOT appear
    res_after = idx.search(target_vec, k=5)
    assert -1 not in res_after.ids


@pytest.mark.parametrize("index_builder", INDEX_CLASSES)
def test_mutations_delete_unknown_and_repeated(base_vectors, base_ids, index_builder):
    """Test deleting unknown IDs and repeated deletion behaves idempotently."""
    idx = index_builder(base_vectors, base_ids)

    # Unknown ID
    assert idx.delete([888888]) == 0

    # Delete valid ID
    assert idx.delete([70]) == 1

    # Repeated deletion of the same ID must return 0
    assert idx.delete([70]) == 0


@pytest.mark.parametrize("index_builder", INDEX_CLASSES)
def test_mutations_add_duplicate_id_raises(base_vectors, base_ids, index_builder):
    """Test that adding an ID already present in the index raises ValueError."""
    idx = index_builder(base_vectors, base_ids)
    vec = np.zeros((1, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="already exists"):
        idx.add(vec, ids=np.array([0], dtype=np.int64))


@pytest.mark.parametrize("index_builder", INDEX_CLASSES)
def test_mutations_add_after_delete_does_not_resurrect(base_vectors, base_ids, index_builder):
    """Test that deleting an ID and re-adding it with new coordinates does not resurrect old data."""
    idx = index_builder(base_vectors, base_ids)

    old_vec = base_vectors[0]
    target_id = base_ids[0]  # -100

    # Delete target ID
    assert idx.delete([target_id]) == 1

    # Re-add target ID with completely different vector coordinates
    new_vec = np.ones((1, 8), dtype=np.float32) * 1000.0
    idx.add(new_vec, ids=np.array([target_id], dtype=np.int64))

    # Search with old vector: target_id must not be the top-1 match
    res_old = idx.search(old_vec, k=1)
    assert res_old.ids[0] != target_id

    # Search with new vector: target_id must match exactly
    res_new = idx.search(new_vec[0], k=1)
    assert res_new.ids[0] == target_id
    if not isinstance(idx, IVFPQIndex):
        assert res_new.distances[0] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("index_builder", INDEX_CLASSES)
def test_mutations_delete_all_vectors_empty_transition(base_vectors, base_ids, index_builder):
    """Test that deleting all vectors transitions to an empty index cleanly."""
    idx = index_builder(base_vectors, base_ids)

    # Delete all IDs
    deleted = idx.delete(base_ids)
    assert deleted == len(base_ids)

    # Search empty index
    q = base_vectors[0]
    res_single = idx.search(q, k=5)
    assert len(res_single.ids) == 0

    res_batch = idx.search(base_vectors[:3], k=5)
    assert res_batch.ids.shape == (3, 0)

    # Add back to empty index
    v = np.zeros((2, 8), dtype=np.float32)
    new_ids = idx.add(v, ids=np.array([10, 20], dtype=np.int64))
    assert len(new_ids) == 2

    res_restored = idx.search(v[0], k=2)
    assert len(res_restored.ids) == 2
