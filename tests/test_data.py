"""Tests for Phase 3: Data and Dataset Utilities.

These tests verify VectorDataset, ClusteredDataset, synthetic generators,
L2 normalization (including zero-vector policies), checksummed caching,
and compatibility with Indexora's exact-search engine.
"""

from pathlib import Path
import numpy as np
import pytest

from indexora import (
    ClusteredDataset,
    ExactFlatIndex,
    SearchResult,
    VectorDataset,
    exact_search,
    generate_clustered_dataset,
    generate_synthetic_dataset,
    load_dataset,
    normalize_l2,
    save_dataset,
)


def test_vector_dataset_construction_and_shapes():
    """Verify valid VectorDataset construction, properties, and default ID generation."""
    base = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    queries = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    train = np.array([[1.1, 2.1]], dtype=np.float32)

    ds = VectorDataset(base=base, queries=queries, train=train)

    assert ds.d == 2
    assert ds.n_base == 3
    assert ds.n_queries == 2
    assert ds.n_train == 1

    # Default IDs
    np.testing.assert_array_equal(ds.base_ids, np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(ds.query_ids, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(ds.train_ids, np.array([0], dtype=np.int64))


def test_vector_dataset_custom_stable_ids():
    """Verify custom, non-contiguous, unsorted IDs in VectorDataset."""
    base = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    queries = np.array([[2.0, 2.0]], dtype=np.float32)
    base_ids = np.array([9999, 42], dtype=np.int64)
    query_ids = np.array([101], dtype=np.int64)

    ds = VectorDataset(base=base, queries=queries, base_ids=base_ids, query_ids=query_ids)
    np.testing.assert_array_equal(ds.base_ids, np.array([9999, 42], dtype=np.int64))
    np.testing.assert_array_equal(ds.query_ids, np.array([101], dtype=np.int64))
    assert ds.train is None
    assert ds.train_ids is None
    assert ds.n_train == 0


def test_vector_dataset_validation_errors():
    """Verify strict validation for dtypes, shapes, finiteness, contiguity, and ID alignment."""
    valid_base = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    valid_q = np.array([[1.0, 2.0]], dtype=np.float32)

    # 1. Non-float32 base
    with pytest.raises(TypeError):
        VectorDataset(base=np.array([[1.0, 2.0]], dtype=np.float64), queries=valid_q)

    # 2. Non-float32 queries
    with pytest.raises(TypeError):
        VectorDataset(base=valid_base, queries=np.array([[1.0, 2.0]], dtype=np.float64))

    # 3. Dimension mismatch: base has d=2, queries has d=3
    with pytest.raises(ValueError, match=r"[dD]imension"):
        VectorDataset(base=valid_base, queries=np.array([[1.0, 2.0, 3.0]], dtype=np.float32))

    # 4. Dimension mismatch in train split
    with pytest.raises(ValueError, match=r"[dD]imension"):
        VectorDataset(
            base=valid_base,
            queries=valid_q,
            train=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        )

    # 5. Non-2D array (1D base)
    with pytest.raises(ValueError):
        VectorDataset(base=np.array([1.0, 2.0], dtype=np.float32), queries=valid_q)

    # 6. Non-finite values
    with pytest.raises(ValueError, match=r"finite"):
        VectorDataset(
            base=np.array([[np.nan, 2.0]], dtype=np.float32),
            queries=valid_q,
        )

    # 7. Non-C-contiguous array
    non_contig = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)[:, ::2]
    assert not non_contig.flags.c_contiguous
    with pytest.raises(ValueError, match=r"C-contiguous"):
        VectorDataset(base=non_contig, queries=np.array([[1.0, 3.0]], dtype=np.float32))

    # 8. Non-int64 IDs
    with pytest.raises(TypeError):
        VectorDataset(
            base=valid_base,
            queries=valid_q,
            base_ids=np.array([1, 2], dtype=np.int32),
        )

    # 9. ID length mismatch
    with pytest.raises(ValueError, match=r"[lL]ength"):
        VectorDataset(
            base=valid_base,
            queries=valid_q,
            base_ids=np.array([1], dtype=np.int64),
        )

    # 10. Duplicate IDs
    with pytest.raises(ValueError, match=r"[uU]nique"):
        VectorDataset(
            base=valid_base,
            queries=valid_q,
            base_ids=np.array([10, 10], dtype=np.int64),
        )

    # 11. train_ids provided without train vectors
    with pytest.raises(ValueError, match=r"train_ids"):
        VectorDataset(
            base=valid_base,
            queries=valid_q,
            train_ids=np.array([1], dtype=np.int64),
        )


def test_vector_dataset_empty_splits():
    """Verify dataset construction with zero-length splits."""
    empty_base = np.empty((0, 4), dtype=np.float32)
    empty_queries = np.empty((0, 4), dtype=np.float32)

    ds = VectorDataset(base=empty_base, queries=empty_queries)
    assert ds.d == 4
    assert ds.n_base == 0
    assert ds.n_queries == 0
    assert ds.base_ids.shape == (0,)
    assert ds.query_ids.shape == (0,)


def test_synthetic_dataset_generation():
    """Verify reproducible synthetic dataset generation."""
    ds1 = generate_synthetic_dataset(n_base=20, n_queries=5, d=8, n_train=10, seed=123)
    ds2 = generate_synthetic_dataset(n_base=20, n_queries=5, d=8, n_train=10, seed=123)

    assert ds1.n_base == 20
    assert ds1.n_queries == 5
    assert ds1.d == 8
    assert ds1.n_train == 10

    # Bit-for-bit reproducibility
    np.testing.assert_array_equal(ds1.base, ds2.base)
    np.testing.assert_array_equal(ds1.queries, ds2.queries)
    np.testing.assert_array_equal(ds1.train, ds2.train)

    # Different seed yields different data
    ds3 = generate_synthetic_dataset(n_base=20, n_queries=5, d=8, n_train=10, seed=999)
    assert not np.array_equal(ds1.base, ds3.base)

    # Validation on negative counts or non-positive dimension
    with pytest.raises(ValueError):
        generate_synthetic_dataset(n_base=-1, n_queries=5, d=8)

    with pytest.raises(ValueError):
        generate_synthetic_dataset(n_base=10, n_queries=5, d=0)


def test_clustered_dataset_generation():
    """Verify reproducible clustered dataset generation for k-means validation."""
    clustered = generate_clustered_dataset(
        n_base=60,
        n_queries=6,
        d=4,
        n_clusters=3,
        n_train=15,
        cluster_std=0.05,
        seed=42,
    )

    assert isinstance(clustered, ClusteredDataset)
    ds = clustered.dataset
    assert ds.n_base == 60
    assert ds.n_queries == 6
    assert ds.d == 4
    assert ds.n_train == 15

    # Check ground truth centroids and labels
    assert clustered.centroids.shape == (3, 4)
    assert clustered.base_labels.shape == (60,)
    assert set(np.unique(clustered.base_labels)).issubset({0, 1, 2})

    # Tuple unpacking support
    dataset_unpacked, labels_unpacked, centroids_unpacked = clustered
    assert dataset_unpacked is ds
    assert labels_unpacked is clustered.base_labels
    assert centroids_unpacked is clustered.centroids

    # Invalid parameter validation
    with pytest.raises(ValueError):
        generate_clustered_dataset(n_base=0, n_queries=1, d=2, n_clusters=2)

    with pytest.raises(ValueError):
        generate_clustered_dataset(n_base=10, n_queries=1, d=2, n_clusters=0)

    with pytest.raises(ValueError):
        generate_clustered_dataset(n_base=10, n_queries=1, d=2, n_clusters=2, cluster_std=-0.1)


def test_normalize_l2_unit_vectors():
    """Verify L2 unit normalization for 1D and 2D arrays."""
    # 1D vector
    v1 = np.array([3.0, 4.0], dtype=np.float32)
    norm1 = normalize_l2(v1)
    assert norm1.shape == (2,)
    assert norm1.dtype == np.float32
    np.testing.assert_allclose(norm1, np.array([0.6, 0.8], dtype=np.float32), rtol=1e-6)
    assert np.isclose(np.linalg.norm(norm1), 1.0, rtol=1e-6)

    # 2D batch
    v2 = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    norm2 = normalize_l2(v2)
    assert norm2.shape == (2, 2)
    assert norm2.dtype == np.float32
    assert norm2.flags.c_contiguous
    lengths = np.linalg.norm(norm2, axis=-1)
    np.testing.assert_allclose(lengths, np.array([1.0, 1.0], dtype=np.float32), rtol=1e-6)


def test_normalize_l2_zero_vector_policy():
    """Verify documented zero-vector policy ('zero' vs 'error')."""
    zero_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # 1. Default policy "zero": returns all zeros without NaN
    norm_zero = normalize_l2(zero_vec, zero_vector_policy="zero")
    assert norm_zero.shape == (3,)
    assert norm_zero.dtype == np.float32
    np.testing.assert_array_equal(norm_zero, zero_vec)
    assert np.all(np.isfinite(norm_zero))

    # 2. Batch with mixed zero and non-zero vectors
    batch_mixed = np.array([[0.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    norm_mixed = normalize_l2(batch_mixed, zero_vector_policy="zero")
    np.testing.assert_array_equal(norm_mixed[0], np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(norm_mixed[1], np.array([0.0, 1.0], dtype=np.float32))

    # 3. Policy "error": raises ValueError
    with pytest.raises(ValueError, match=r"[zZ]ero-norm"):
        normalize_l2(zero_vec, zero_vector_policy="error")

    with pytest.raises(ValueError, match=r"[zZ]ero-norm"):
        normalize_l2(batch_mixed, zero_vector_policy="error")

    # 4. Unknown policy
    with pytest.raises(ValueError, match=r"Unknown zero_vector_policy"):
        normalize_l2(zero_vec, zero_vector_policy="invalid_policy")

    # 5. Type and finiteness validation
    with pytest.raises(TypeError):
        normalize_l2(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ValueError):
        normalize_l2(np.array([np.nan, 2.0], dtype=np.float32))


def test_dataset_cache_round_trip(tmp_path: Path):
    """Verify save_dataset and load_dataset round-trip with SHA-256 integrity."""
    ds = generate_synthetic_dataset(n_base=25, n_queries=5, d=6, n_train=10, seed=777)
    cache_dir = tmp_path / "cached_dataset"

    manifest = save_dataset(ds, cache_dir)
    assert (cache_dir / "metadata.json").exists()
    assert (cache_dir / "base.npy").exists()
    assert manifest["dimension"] == 6
    assert len(manifest["checksums"]) == 6  # base, base_ids, queries, query_ids, train, train_ids

    # Load with checksum verification
    loaded_ds = load_dataset(cache_dir, verify_checksums=True)
    assert loaded_ds.d == ds.d
    assert loaded_ds.n_base == ds.n_base
    assert loaded_ds.n_queries == ds.n_queries
    assert loaded_ds.n_train == ds.n_train

    np.testing.assert_array_equal(loaded_ds.base, ds.base)
    np.testing.assert_array_equal(loaded_ds.base_ids, ds.base_ids)
    np.testing.assert_array_equal(loaded_ds.queries, ds.queries)
    np.testing.assert_array_equal(loaded_ds.query_ids, ds.query_ids)
    np.testing.assert_array_equal(loaded_ds.train, ds.train)
    np.testing.assert_array_equal(loaded_ds.train_ids, ds.train_ids)


def test_dataset_cache_corruption_detection(tmp_path: Path):
    """Verify that tampered or corrupted cache files are detected and rejected."""
    ds = generate_synthetic_dataset(n_base=10, n_queries=2, d=4, seed=42)
    cache_dir = tmp_path / "corrupt_test"
    save_dataset(ds, cache_dir)

    # Tamper with base.npy by appending a byte
    base_file = cache_dir / "base.npy"
    with open(base_file, "ab") as f:
        f.write(b"\x00")

    with pytest.raises(ValueError, match=r"[cC]hecksum mismatch.*corrupted"):
        load_dataset(cache_dir, verify_checksums=True)

    # Missing metadata.json
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_dataset(empty_dir)


def test_compatibility_with_exact_search():
    """Verify VectorDataset interoperability with exact_search and ExactFlatIndex."""
    ds = generate_synthetic_dataset(n_base=30, n_queries=4, d=5, seed=101)

    # 1. Run exact_search directly on dataset splits
    search_res = exact_search(
        queries=ds.queries,
        database=ds.base,
        ids=ds.base_ids,
        k=3,
    )
    assert isinstance(search_res, SearchResult)
    assert search_res.ids.shape == (4, 3)
    assert search_res.distances.shape == (4, 3)

    # 2. Run ExactFlatIndex on dataset splits
    index = ExactFlatIndex(database=ds.base, ids=ds.base_ids)
    index_res = index.search(ds.queries, k=3)

    np.testing.assert_array_equal(search_res.ids, index_res.ids)
    np.testing.assert_array_equal(search_res.distances, index_res.distances)
