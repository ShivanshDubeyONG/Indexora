"""Tests for Phase 7: Reference Adapters for FAISS and hnswlib.

These tests verify:
- Safe optional dependency handling (graceful error when dependencies are missing).
- FAISS exact flat search parity against Indexora's ground truth and exact search.
- FAISS IVF-Flat approximate search, all-probe parity, and parameter tuning.
- hnswlib adapter interface, parameter delegation, and ID translation.
- Memory property consistency and SearchResult packaging.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from indexora import (
    ExactFlatIndex,
    FaissFlatIndex,
    FaissIVFFlatIndex,
    HnswlibIndex,
    SearchResult,
    compute_recall_at_k,
    generate_clustered_dataset,
    generate_ground_truth,
    is_faiss_available,
    is_hnswlib_available,
)
import indexora.adapters as adapters_mod


# ============================================================================
# 1. Dependency Availability and Missing Dependency Handling
# ============================================================================


def test_dependency_availability_flags():
    """Verify boolean return values of availability functions."""
    assert isinstance(is_faiss_available(), bool)
    assert isinstance(is_hnswlib_available(), bool)


def test_missing_faiss_dependency_raises_clean_import_error(monkeypatch: pytest.MonkeyPatch):
    """Verify clean ImportError with install guidance when FAISS is unavailable."""
    monkeypatch.setattr(adapters_mod, "is_faiss_available", lambda: False)

    def mock_require_faiss():
        raise ImportError(
            "FAISS is not installed. Install it via 'pip install faiss-cpu' "
            "(or 'pip install indexora[adapters]')."
        )

    monkeypatch.setattr(adapters_mod, "_require_faiss", mock_require_faiss)

    dummy_db = np.ones((10, 4), dtype=np.float32)
    with pytest.raises(ImportError, match="pip install faiss-cpu"):
        FaissFlatIndex(dummy_db)

    with pytest.raises(ImportError, match="pip install faiss-cpu"):
        FaissIVFFlatIndex(dummy_db, nlist=2)


def test_missing_hnswlib_dependency_raises_clean_import_error(monkeypatch: pytest.MonkeyPatch):
    """Verify clean ImportError with install guidance when hnswlib is unavailable."""
    monkeypatch.setattr(adapters_mod, "is_hnswlib_available", lambda: False)

    dummy_db = np.ones((10, 4), dtype=np.float32)
    with pytest.raises(ImportError, match="pip install hnswlib"):
        HnswlibIndex(dummy_db)


# ============================================================================
# 2. FAISS Flat Exact Search Adapter Tests
# ============================================================================


@pytest.mark.skipif(not is_faiss_available(), reason="FAISS is not installed")
def test_faiss_flat_exact_search_parity_with_oracle():
    """Verify FaissFlatIndex matches ExactFlatIndex and ground truth."""
    rng = np.random.default_rng(42)
    N, d, Q = 80, 8, 10
    database = rng.standard_normal((N, d)).astype(np.float32)
    queries = rng.standard_normal((Q, d)).astype(np.float32)
    ids = np.arange(500, 500 + N, dtype=np.int64)

    exact_idx = ExactFlatIndex(database=database, ids=ids)
    faiss_idx = FaissFlatIndex(database=database, ids=ids)

    res_oracle = exact_idx.search(queries, k=10)
    res_faiss = faiss_idx.search(queries, k=10)

    assert isinstance(res_faiss, SearchResult)
    np.testing.assert_array_equal(res_faiss.ids, res_oracle.ids)
    np.testing.assert_allclose(res_faiss.distances, res_oracle.distances, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(res_faiss.mask, res_oracle.mask)
    assert res_faiss.distance_computations == Q * N

    # Perfect recall against exact ground truth
    recall = compute_recall_at_k(res_faiss, res_oracle, k=10)
    assert recall == 1.0


@pytest.mark.skipif(not is_faiss_available(), reason="FAISS is not installed")
def test_faiss_flat_custom_and_negative_ids():
    """Verify FaissFlatIndex properly maps non-sequential and negative int64 IDs."""
    database = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]], dtype=np.float32)
    ids = np.array([-1, 42, -999999], dtype=np.int64)
    queries = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)

    faiss_idx = FaissFlatIndex(database=database, ids=ids)
    res = faiss_idx.search(queries, k=2)

    assert res.ids[0, 0] == -1
    assert res.ids[1, 0] == -999999
    assert np.all(res.mask)
    assert res.distances[0, 0] == 0.0
    assert res.distances[1, 0] == 0.0


@pytest.mark.skipif(not is_faiss_available(), reason="FAISS is not installed")
def test_faiss_flat_single_query_1d():
    """Verify FaissFlatIndex works with 1D single query vectors."""
    db = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    q = np.array([1.0, 2.0], dtype=np.float32)

    idx = FaissFlatIndex(db)
    res = idx.search(q, k=1)

    assert res.ids.shape == (1,)
    assert res.distances.shape == (1,)
    assert res.ids[0] == 0
    assert res.distances[0] == 0.0


@pytest.mark.skipif(not is_faiss_available(), reason="FAISS is not installed")
def test_faiss_flat_validation_errors():
    """Verify input validation on FaissFlatIndex."""
    valid_db = np.ones((10, 4), dtype=np.float32)

    # Non-float32
    with pytest.raises(TypeError, match="float32"):
        FaissFlatIndex(valid_db.astype(np.float64))

    # Dimension mismatch
    with pytest.raises(ValueError, match="dimension"):
        idx = FaissFlatIndex(valid_db)
        idx.search(np.ones(5, dtype=np.float32))

    # Invalid k
    with pytest.raises(ValueError, match="k must be positive"):
        idx = FaissFlatIndex(valid_db)
        idx.search(np.ones(4, dtype=np.float32), k=0)


# ============================================================================
# 3. FAISS IVF-Flat Approximate Search Adapter Tests
# ============================================================================


@pytest.mark.skipif(not is_faiss_available(), reason="FAISS is not installed")
def test_faiss_ivf_flat_all_probe_exact_parity():
    """Verify FaissIVFFlatIndex with nprobe=nlist matches exact search."""
    data = generate_clustered_dataset(
        n_base=120,
        n_queries=10,
        d=4,
        n_clusters=4,
        cluster_std=0.05,
        seed=42,
    )

    faiss_ivf = FaissIVFFlatIndex(
        database=data.dataset.base,
        nlist=4,
        ids=data.dataset.base_ids,
        nprobe=4,
        seed=42,
    )

    gt = generate_ground_truth(
        queries=data.dataset.queries,
        database=data.dataset.base,
        ids=data.dataset.base_ids,
        k=5,
    )

    # All-probe search achieves 1.0 recall
    res_all = faiss_ivf.search(data.dataset.queries, k=5, nprobe=4)
    recall_all = compute_recall_at_k(res_all, gt, k=5)
    assert recall_all == 1.0

    # Partial-probe search executes and achieves valid recall
    res_p1 = faiss_ivf.search(data.dataset.queries, k=5, nprobe=1)
    recall_p1 = compute_recall_at_k(res_p1, gt, k=5)
    assert 0.0 <= recall_p1 <= 1.0


@pytest.mark.skipif(not is_faiss_available(), reason="FAISS is not installed")
def test_faiss_memory_properties():
    """Verify storage memory accounting properties on FAISS adapters."""
    N, d, nlist = 100, 16, 5
    db = np.ones((N, d), dtype=np.float32)

    flat_idx = FaissFlatIndex(db)
    expected_flat_bytes = N * (d * 4 + 8)
    assert flat_idx.index_memory_bytes == expected_flat_bytes
    assert flat_idx.index_memory_mib == expected_flat_bytes / (1024 * 1024)
    assert flat_idx.bytes_per_vector == (d * 4 + 8)

    ivf_idx = FaissIVFFlatIndex(db, nlist=nlist)
    expected_ivf_bytes = nlist * d * 4 + N * (d * 4 + 8)
    assert ivf_idx.index_memory_bytes == expected_ivf_bytes
    assert ivf_idx.index_memory_mib == expected_ivf_bytes / (1024 * 1024)
    assert ivf_idx.bytes_per_vector == expected_ivf_bytes / N


# ============================================================================
# 4. hnswlib Adapter Structure & ID Translation Tests
# ============================================================================


def test_hnswlib_adapter_lifecycle_and_id_mapping(monkeypatch: pytest.MonkeyPatch):
    """Verify HnswlibIndex maps 0..N-1 labels back to arbitrary int64 IDs."""
    N, d = 4, 2
    database = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    custom_ids = np.array([-1, 100, -999, 42], dtype=np.int64)

    # Mock hnswlib.Index
    mock_inner_index = MagicMock()
    # Mock knn_query returns internal labels [0, 2] and squared distances [0.1, 0.5]
    mock_inner_index.knn_query.return_value = (
        np.array([[0, 2]], dtype=np.uint64),
        np.array([[0.1, 0.5]], dtype=np.float32),
    )

    mock_hnswlib = MagicMock()
    mock_hnswlib.Index.return_value = mock_inner_index

    monkeypatch.setattr(adapters_mod, "_require_hnswlib", lambda: mock_hnswlib)

    idx = HnswlibIndex(
        database=database,
        ids=custom_ids,
        M=8,
        ef_construction=100,
        ef_search=20,
    )

    assert idx.M == 8
    assert idx.ef_construction == 100
    assert idx.ef_search == 20
    assert idx.index_memory_bytes == N * (d * 4 + 8 * 8)

    # Search with k=2
    res = idx.search(database[:1], k=2, ef_search=30)

    # Label 0 mapped to custom_id -1, label 2 mapped to custom_id -999
    assert res.ids[0, 0] == -1
    assert res.ids[0, 1] == -999
    assert res.distances[0, 0] == pytest.approx(0.1, abs=1e-5)
    assert res.distances[0, 1] == pytest.approx(0.5, abs=1e-5)
    assert np.all(res.mask)
    mock_inner_index.set_ef.assert_called_with(30)
