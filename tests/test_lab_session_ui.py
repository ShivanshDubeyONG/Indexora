"""Unit tests for LabSession mutation state management, load atomicity, and UI correctness.

Verifies:
- Loading a saved index (via load or load_named) resets added_ids and deleted_ids to empty lists.
- Failed loads remain atomic and leave active index and mutation tracking untouched.
- Mutation tracking prevents duplicate IDs and correctly updates active counts.
- Unsuccessful deletions (e.g. non-existent IDs) return 0 and do not modify deleted_ids.
- Index-specific deletion mechanics (HNSW logical tombstones vs. Exact physical removal vs. IVF posting lists).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from indexora.exact import ExactBlasIndex, ExactFlatIndex
from indexora.hnsw import HNSWIndex
from indexora.ivf import IVFFlatIndex
from indexora.lab import LabSession, generate_lab_dataset


def test_lab_session_load_named_resets_mutation_history(tmp_path):
    """Loading a named index must reset added_ids and deleted_ids to empty lists."""
    ds = generate_lab_dataset(n_base=30, n_queries=2, d=4, n_clusters=2, seed=12)
    idx = ExactBlasIndex(database=ds.base, ids=ds.base_ids)
    session = LabSession(index=idx, index_name="ExactBlasIndex", d=ds.d, storage_root=tmp_path)

    # Add a vector and delete a vector
    new_v = np.ones(ds.d, dtype=np.float32)
    assigned_id = session.add_vector(new_v, ext_id=5001)
    del_id = int(ds.base_ids[0])
    session.delete_vector(del_id)

    assert session.added_ids == [assigned_id]
    assert session.deleted_ids == [del_id]

    # Save to storage
    session.save_named("test_idx_01")

    # Add more mutations to the in-memory session
    session.add_vector(new_v, ext_id=5002)
    del_id_2 = int(ds.base_ids[1])
    session.delete_vector(del_id_2)

    assert len(session.added_ids) == 2
    assert len(session.deleted_ids) == 2

    # Load the saved index
    restored = session.load_named("test_idx_01")
    assert restored is not None

    # Mutation history MUST be reset to empty lists
    assert session.added_ids == []
    assert session.deleted_ids == []


def test_lab_session_load_resets_mutation_history(tmp_path):
    """Loading via path must also reset added_ids and deleted_ids to empty lists."""
    ds = generate_lab_dataset(n_base=30, n_queries=2, d=4, n_clusters=2, seed=14)
    idx = IVFFlatIndex(database=ds.base, ids=ds.base_ids, nlist=3, nprobe=2)
    session = LabSession(index=idx, index_name="IVFFlatIndex", d=ds.d, storage_root=tmp_path)

    new_v = np.full(ds.d, 2.0, dtype=np.float32)
    session.add_vector(new_v, ext_id=7001)
    session.delete_vector(int(ds.base_ids[2]))

    assert session.added_ids == [7001]
    assert session.deleted_ids == [int(ds.base_ids[2])]

    target_dir = tmp_path / "custom_ivf_save"
    session.save(target_dir)

    # Mutate further
    session.add_vector(new_v, ext_id=7002)
    assert len(session.added_ids) == 2

    # Load from directory
    session.load(target_dir)
    assert session.added_ids == []
    assert session.deleted_ids == []


def test_lab_session_failed_load_atomic_preserves_state(tmp_path):
    """Failed load operations must not alter the active index or its mutation state."""
    ds = generate_lab_dataset(n_base=25, n_queries=2, d=4, n_clusters=2, seed=20)
    idx = ExactBlasIndex(database=ds.base, ids=ds.base_ids)
    session = LabSession(index=idx, index_name="ExactBlasIndex", d=ds.d, storage_root=tmp_path)

    session.add_vector(np.ones(ds.d, dtype=np.float32), ext_id=8001)
    session.delete_vector(int(ds.base_ids[0]))

    orig_added = list(session.added_ids)
    orig_deleted = list(session.deleted_ids)
    orig_index = session.active_index
    orig_count = session.active_count

    # 1. Attempt loading non-existent index
    with pytest.raises((FileNotFoundError, ValueError)):
        session.load_named("non_existent_index_123")

    assert session.active_index is orig_index
    assert session.added_ids == orig_added
    assert session.deleted_ids == orig_deleted
    assert session.active_count == orig_count

    # 2. Attempt loading corrupted index (checksum failure)
    valid_dir = tmp_path / "corrupted_idx"
    session.save(valid_dir)

    # Tamper with array file to cause ChecksumMismatchError
    arr_file = valid_dir / "database.npy"
    if arr_file.exists():
        with open(arr_file, "ab") as f:
            f.write(b"\x00\xff\xfe")

    with pytest.raises(Exception):  # ChecksumMismatchError
        session.load(valid_dir, verify_checksums=True)

    # Session remains untouched
    assert session.active_index is orig_index
    assert session.added_ids == orig_added
    assert session.deleted_ids == orig_deleted
    assert session.active_count == orig_count


def test_lab_session_mutations_no_duplicates_and_unsuccessful_deletions():
    """Mutations should prevent duplicate tracking and handle unsuccessful deletions cleanly."""
    ds = generate_lab_dataset(n_base=20, n_queries=2, d=4, n_clusters=2, seed=33)
    idx = ExactBlasIndex(database=ds.base, ids=ds.base_ids)
    session = LabSession(index=idx, index_name="ExactBlasIndex", d=ds.d)

    # Add vector
    new_v = np.zeros(ds.d, dtype=np.float32)
    assigned = session.add_vector(new_v, ext_id=9001)
    assert assigned == 9001
    assert session.added_ids == [9001]
    assert session.active_count == 21

    # Attempt deleting non-existent ID
    del_cnt = session.delete_vector(999999)
    assert del_cnt == 0
    assert 999999 not in session.deleted_ids
    assert session.deleted_ids == []
    assert session.active_count == 21

    # Delete the newly added vector
    del_cnt = session.delete_vector(9001)
    assert del_cnt == 1
    assert 9001 in session.deleted_ids
    assert 9001 not in session.added_ids
    assert session.active_count == 20

    # Deleting 9001 again: already deleted, returns 0
    del_cnt2 = session.delete_vector(9001)
    assert del_cnt2 == 0
    # deleted_ids should not duplicate 9001
    assert session.deleted_ids == [9001]
    assert session.active_count == 20


def test_index_specific_deletion_mechanics():
    """Verify distinct deletion mechanics across HNSW, Exact BLAS, and IVF."""
    ds = generate_lab_dataset(n_base=20, n_queries=2, d=4, n_clusters=2, seed=55)

    # 1. Exact BLAS: physical array slice removal
    exact_idx = ExactBlasIndex(database=ds.base, ids=ds.base_ids)
    init_exact_len = len(exact_idx.ids)
    del_target_exact = int(ds.base_ids[0])
    exact_del_cnt = exact_idx.delete([del_target_exact])
    assert exact_del_cnt == 1
    assert len(exact_idx.ids) == init_exact_len - 1
    assert del_target_exact not in exact_idx.ids

    # 2. HNSW: logical tombstones (_deleted_nodes set)
    hnsw_idx = HNSWIndex(database=ds.base, ids=ds.base_ids, M=4, ef_construction=16, ef_search=16)
    init_hnsw_active = hnsw_idx.n_active
    del_target_hnsw = int(ds.base_ids[1])
    internal_node_idx = hnsw_idx._id_to_node[del_target_hnsw]

    hnsw_del_cnt = hnsw_idx.delete([del_target_hnsw])
    assert hnsw_del_cnt == 1
    assert hnsw_idx.n_active == init_hnsw_active - 1
    # Check internal logical tombstone state
    assert internal_node_idx in hnsw_idx._deleted_nodes
    assert del_target_hnsw not in hnsw_idx._id_to_node
    # Node still physically exists in graph adjacency structure for routing
    assert internal_node_idx < len(hnsw_idx._graphs[0])
    assert len(hnsw_idx._graphs[0]) == len(ds.base)

    # 3. IVF-Flat: removal from inverted posting list
    ivf_idx = IVFFlatIndex(database=ds.base, ids=ds.base_ids, nlist=3, nprobe=3)
    init_ivf_active = ivf_idx.ntotal
    del_target_ivf = int(ds.base_ids[2])
    ivf_del_cnt = ivf_idx.delete([del_target_ivf])
    assert ivf_del_cnt == 1
    assert ivf_idx.ntotal == init_ivf_active - 1
    # ID not in any inverted list
    for invlist in ivf_idx.invlists:
        assert del_target_ivf not in invlist.ids
