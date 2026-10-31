"""Comprehensive test suite for index persistence across all Indexora index implementations.

Verifies atomic directory persistence, manifest schema versioning, SHA-256 checksum
integrity validation, zero-pickle NumPy array serialization, round-trip search fidelity,
and exact restoration of HNSW graphs and IVF-PQ codebooks without recomputation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from indexora.exact import ExactBlasIndex, ExactFlatIndex
from indexora.hnsw import HNSWIndex
from indexora.ivf import IVFFlatIndex
from indexora.ivfpq import IVFPQIndex
from indexora.persistence import load_index, save_index


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic vectors and IDs including negative and non-sequential IDs."""
    rng = np.random.default_rng(999)
    vectors = rng.standard_normal((32, 8)).astype(np.float32)
    ids = np.array(
        [-500, -100, -1, 0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70,
         80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230],
        dtype=np.int64,
    )
    return vectors, ids


@pytest.mark.parametrize(
    "index_builder,index_cls",
    [
        (lambda db, ids: ExactFlatIndex(database=db, ids=ids), ExactFlatIndex),
        (lambda db, ids: ExactBlasIndex(database=db, ids=ids), ExactBlasIndex),
        (lambda db, ids: IVFFlatIndex(database=db, ids=ids, nlist=4, nprobe=3), IVFFlatIndex),
        (lambda db, ids: IVFPQIndex(database=db, ids=ids, nlist=4, M=2, k_sub=4, nprobe=3), IVFPQIndex),
        (lambda db, ids: HNSWIndex(database=db, ids=ids, M=8, ef_construction=32, ef_search=32, random_seed=42), HNSWIndex),
    ],
)
def test_persistence_roundtrip_search_fidelity(dataset, tmp_path, index_builder, index_cls):
    """Saving and loading an index should preserve bit-identical search results."""
    vectors, ids = dataset
    idx = index_builder(vectors, ids)

    save_dir = tmp_path / f"index_{index_cls.__name__}"
    manifest = idx.save(save_dir)
    assert (save_dir / "manifest.json").exists()
    assert manifest["schema_version"] == "1.0"
    assert manifest["format"] == "indexora_index"
    assert manifest["index_type"] == index_cls.__name__

    # Load via classmethod
    loaded_idx = index_cls.load(save_dir)
    assert isinstance(loaded_idx, index_cls)
    assert len(loaded_idx) == len(idx)

    # Load via standalone function
    loaded_idx2 = load_index(save_dir)
    assert isinstance(loaded_idx2, index_cls)

    # Verify search fidelity on single and batch queries
    q_single = vectors[5]
    res_orig_single = idx.search(q_single, k=5)
    res_loaded_single = loaded_idx.search(q_single, k=5)

    np.testing.assert_array_equal(res_loaded_single.ids, res_orig_single.ids)
    np.testing.assert_allclose(res_loaded_single.distances, res_orig_single.distances, atol=1e-5)

    q_batch = vectors[:4]
    res_orig_batch = idx.search(q_batch, k=5)
    res_loaded_batch = loaded_idx.search(q_batch, k=5)

    np.testing.assert_array_equal(res_loaded_batch.ids, res_orig_batch.ids)
    np.testing.assert_allclose(res_loaded_batch.distances, res_orig_batch.distances, atol=1e-5)


def test_persistence_hnsw_graph_restoration(dataset, tmp_path):
    """Verify that HNSW graph topology (layers, enter node, connections) is restored exactly without rebuilding."""
    vectors, ids = dataset
    idx = HNSWIndex(database=vectors, ids=ids, M=8, ef_construction=32, ef_search=32, random_seed=42)

    save_dir = tmp_path / "hnsw_exact_graph"
    save_index(idx, save_dir)

    loaded_idx = HNSWIndex.load(save_dir)

    assert loaded_idx.entry_point == idx.entry_point
    assert loaded_idx.max_level == idx.max_level

    # Check that layer-by-layer graph matches original adjacency exactly
    for lvl in range(idx.max_level + 1):
        for node in range(idx.ntotal):
            assert loaded_idx.get_neighbors(node, lvl) == idx.get_neighbors(node, lvl)


def test_persistence_ivfpq_codebooks_restoration(dataset, tmp_path):
    """Verify that IVF-PQ coarse centroids, PQ codebooks, and posting lists are restored without retraining."""
    vectors, ids = dataset
    idx = IVFPQIndex(database=vectors, ids=ids, nlist=4, M=2, k_sub=4, nprobe=3)

    save_dir = tmp_path / "ivfpq_exact_codebooks"
    save_index(idx, save_dir)

    loaded_idx = IVFPQIndex.load(save_dir)

    assert loaded_idx.nlist == idx.nlist
    assert loaded_idx.M == idx.M
    assert loaded_idx.k_sub == idx.k_sub
    assert loaded_idx.use_residuals == idx.use_residuals

    # Coarse centroids match exactly
    np.testing.assert_array_equal(loaded_idx.coarse_centroids, idx.coarse_centroids)
    # PQ codebooks match exactly
    np.testing.assert_array_equal(loaded_idx.pq.codebooks, idx.pq.codebooks)

    # Posting lists match exactly
    for c in range(idx.nlist):
        orig_lst = idx.get_list(c)
        loaded_lst = loaded_idx.get_list(c)
        np.testing.assert_array_equal(loaded_lst.ids, orig_lst.ids)
        np.testing.assert_array_equal(loaded_lst.codes, orig_lst.codes)


def test_persistence_with_mutations(dataset, tmp_path):
    """Test that deleted and incrementally added vectors persist correctly."""
    vectors, ids = dataset
    idx = IVFFlatIndex(database=vectors, ids=ids, nlist=4, nprobe=3)

    # Delete 2 vectors
    del_ids = [ids[0], ids[1]]
    idx.delete(del_ids)

    # Add a new vector
    new_v = np.ones((1, 8), dtype=np.float32) * 5.0
    new_id = np.array([8888], dtype=np.int64)
    idx.add(new_v, ids=new_id)

    save_dir = tmp_path / "mutated_ivf"
    idx.save(save_dir)

    loaded_idx = IVFFlatIndex.load(save_dir)
    assert len(loaded_idx) == len(idx)

    # Deleted vectors should not be present
    res = loaded_idx.search(vectors[0], k=10)
    for d_id in del_ids:
        assert d_id not in res.ids

    # Added vector should be searchable
    res_new = loaded_idx.search(new_v[0], k=1)
    assert res_new.ids[0] == 8888


def test_persistence_hnsw_with_tombstones(dataset, tmp_path):
    """Test that tombstoned nodes and active node counts persist cleanly in HNSW."""
    vectors, ids = dataset
    idx = HNSWIndex(database=vectors, ids=ids, M=8, ef_construction=32, ef_search=32, random_seed=42)

    del_id = ids[3]
    idx.delete([del_id])

    save_dir = tmp_path / "hnsw_tombstones"
    idx.save(save_dir)

    loaded_idx = HNSWIndex.load(save_dir)
    assert len(loaded_idx) == len(idx)

    res = loaded_idx.search(vectors[3], k=5)
    assert del_id not in res.ids


def test_persistence_checksum_tampering_detection(dataset, tmp_path):
    """Modifying an array file must trigger checksum verification failure."""
    vectors, ids = dataset
    idx = ExactFlatIndex(database=vectors, ids=ids)

    save_dir = tmp_path / "tampered_index"
    idx.save(save_dir)

    # Tamper with database.npy
    data_file = save_dir / "database.npy"
    with open(data_file, "r+b") as f:
        f.seek(128)  # Seek into data body beyond header
        byte = f.read(1)
        f.seek(128)
        f.write(bytes([byte[0] ^ 0xFF]))

    with pytest.raises(ValueError, match="Integrity check failed"):
        load_index(save_dir, verify_checksums=True)

    # If verification disabled, load does not raise checksum error
    loaded_tampered = load_index(save_dir, verify_checksums=False)
    assert isinstance(loaded_tampered, ExactFlatIndex)


def test_persistence_unsupported_version(tmp_path):
    """Unsupported manifest format version raises ValueError."""
    save_dir = tmp_path / "bad_version_index"
    save_dir.mkdir()
    manifest_path = save_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"format": "indexora_index", "schema_version": "99.9", "index_type": "ExactFlatIndex"}))

    with pytest.raises(ValueError, match="Unsupported schema version"):
        load_index(save_dir)


def test_persistence_empty_index_roundtrip(tmp_path):
    """Empty indexes should save and load without error."""
    idx = ExactFlatIndex(database=np.empty((0, 8), dtype=np.float32))
    save_dir = tmp_path / "empty_exact_index"
    idx.save(save_dir)

    loaded_idx = ExactFlatIndex.load(save_dir)
    assert len(loaded_idx) == 0
    assert loaded_idx.ntotal == 0


def test_persistence_path_traversal_detection(tmp_path):
    """Manifest referencing filenames outside the directory raises ValueError."""
    save_dir = tmp_path / "traversal_index"
    save_dir.mkdir()
    manifest_data = {
        "format": "indexora_index",
        "schema_version": "1.0",
        "index_type": "ExactFlatIndex",
        "d": 8,
        "ntotal": 1,
        "arrays": {
            "../outside.npy": {"shape": [1, 8], "dtype": "float32", "sha256": "fake"}
        }
    }
    (save_dir / "manifest.json").write_text(json.dumps(manifest_data))

    with pytest.raises(ValueError, match="Path traversal detected"):
        load_index(save_dir)
