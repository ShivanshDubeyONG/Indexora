"""Tests for Inverted File with Product Quantization (IVF-PQ) in Indexora."""
import numpy as np
import pytest

from indexora.exact import exact_search
from indexora.ivfpq import IVFPQIndex, InvertedPQList, ivf_pq_build, ivf_pq_search
from indexora.pq import ProductQuantizer
from indexora.topk import SearchResult


def test_inverted_pq_list_validation() -> None:
    """Test validation of InvertedPQList container."""
    ids = np.array([10, 20, 30], dtype=np.int64)
    codes = np.ones((3, 4), dtype=np.uint8)

    lst = InvertedPQList(ids=ids, codes=codes)
    assert lst.size == 3

    # Reject non-int64 IDs
    with pytest.raises(TypeError, match="ids must have dtype int64"):
        InvertedPQList(ids=np.array([1, 2], dtype=np.int32), codes=np.ones((2, 4), dtype=np.uint8))

    # Reject non-integer codes
    with pytest.raises(TypeError, match="codes must have integer dtype"):
        InvertedPQList(ids=ids, codes=np.ones((3, 4), dtype=np.float32))

    # Reject length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        InvertedPQList(ids=ids, codes=np.ones((2, 4), dtype=np.uint8))


def test_ivf_pq_build_and_properties() -> None:
    """Test IVFPQIndex initialization, parameters, and list partition invariants."""
    rng = np.random.default_rng(42)
    N, d, nlist, M, k_sub = 100, 32, 4, 4, 16
    db = rng.normal(size=(N, d)).astype(np.float32)

    index = IVFPQIndex(database=db, nlist=nlist, M=M, k_sub=k_sub, seed=42)

    assert index.d == d
    assert index.ntotal == N
    assert index.nlist == nlist
    assert index.M == M
    assert index.d_sub == 8
    assert index.k_sub == k_sub
    assert index.nprobe == 1
    assert index.use_residuals is True
    assert index.code_dtype == np.uint8
    assert len(index.invlists) == nlist

    # Total vectors in lists must equal N
    total_in_lists = sum(lst.size for lst in index.invlists)
    assert total_in_lists == N

    # Union of all list IDs must equal 0..N-1
    all_ids = np.concatenate([lst.ids for lst in index.invlists if lst.size > 0])
    np.testing.assert_array_equal(np.sort(all_ids), np.arange(N, dtype=np.int64))


def test_ivf_pq_custom_and_negative_ids() -> None:
    """Test that custom arbitrary IDs including negative values are preserved."""
    rng = np.random.default_rng(42)
    N, d, nlist, M, k_sub = 80, 16, 4, 2, 8
    db = rng.normal(size=(N, d)).astype(np.float32)
    custom_ids = np.array([-50 + i * 2 for i in range(N)], dtype=np.int64)
    # Include explicit ID -1
    custom_ids[10] = -1

    index = IVFPQIndex(database=db, nlist=nlist, M=M, k_sub=k_sub, ids=custom_ids, seed=42)

    all_ids = np.concatenate([lst.ids for lst in index.invlists if lst.size > 0])
    np.testing.assert_array_equal(np.sort(all_ids), np.sort(custom_ids))

    # Query search returns custom IDs
    q = db[10]
    res = index.search(q, k=3, nprobe=nlist)
    assert isinstance(res, SearchResult)
    assert len(res.ids) == 3
    # All retrieved IDs must be members of custom_ids
    for retrieved_id in res.ids:
        assert retrieved_id in custom_ids


def test_ivf_pq_memory_and_compression_accounting() -> None:
    """Test precise memory accounting and compression ratio calculation."""
    rng = np.random.default_rng(42)
    N, d, nlist, M, k_sub = 500, 64, 5, 8, 32
    db = rng.normal(size=(N, d)).astype(np.float32)

    index = IVFPQIndex(database=db, nlist=nlist, M=M, k_sub=k_sub, seed=42)

    # Expected component memory:
    # Coarse centroids: nlist * d * 4 bytes
    expected_centroids_bytes = nlist * d * 4
    assert index.coarse_centroid_memory_bytes == expected_centroids_bytes

    # Codebooks: M * k_sub * (d // M) * 4 bytes
    expected_codebook_bytes = M * k_sub * (d // M) * 4
    assert index.codebook_memory_bytes == expected_codebook_bytes

    # Stored codes: N * M * 1 bytes (uint8)
    expected_codes_bytes = N * M * 1
    assert index.codes_memory_bytes == expected_codes_bytes

    # Stored IDs: N * 8 bytes (int64)
    expected_ids_bytes = N * 8
    assert index.ids_memory_bytes == expected_ids_bytes

    # Total logical index memory
    expected_total_bytes = (
        expected_centroids_bytes
        + expected_codebook_bytes
        + expected_codes_bytes
        + expected_ids_bytes
    )
    assert index.index_memory_bytes == expected_total_bytes
    assert pytest.approx(index.index_memory_mib, rel=1e-5) == expected_total_bytes / (1024 * 1024)
    assert pytest.approx(index.bytes_per_vector, rel=1e-5) == expected_total_bytes / N

    # Raw uncompressed: N * (d * 4 + 8) = 500 * (256 + 8) = 132,000 bytes
    expected_raw_bytes = N * (d * 4 + 8)
    assert index.raw_uncompressed_bytes == expected_raw_bytes

    # Compression ratio
    expected_ratio = expected_raw_bytes / expected_total_bytes
    assert pytest.approx(index.compression_ratio, rel=1e-5) == expected_ratio
    assert index.compression_ratio > 5.0  # Significant compression achieved


def test_ivf_pq_search_single_vs_batched() -> None:
    """Test that single query (d,) and batched (Q, d) return identical results."""
    rng = np.random.default_rng(42)
    N, d, nlist, M, k_sub, Q = 200, 32, 4, 4, 16, 5
    db = rng.normal(size=(N, d)).astype(np.float32)
    queries = rng.normal(size=(Q, d)).astype(np.float32)

    index = IVFPQIndex(database=db, nlist=nlist, M=M, k_sub=k_sub, seed=42)

    batched_res = index.search(queries, k=5, nprobe=2)

    for i in range(Q):
        single_res = index.search(queries[i], k=5, nprobe=2)
        np.testing.assert_array_equal(batched_res.ids[i], single_res.ids)
        np.testing.assert_allclose(batched_res.distances[i], single_res.distances, rtol=1e-5, atol=1e-5)


def test_ivf_pq_residuals_vs_non_residuals() -> None:
    """Test both residual and non-residual IVF-PQ modes."""
    rng = np.random.default_rng(42)
    N, d, nlist, M, k_sub, Q = 200, 16, 4, 2, 8, 5
    db = rng.normal(size=(N, d)).astype(np.float32)
    queries = rng.normal(size=(Q, d)).astype(np.float32)

    # Residual mode
    idx_res = IVFPQIndex(database=db, nlist=nlist, M=M, k_sub=k_sub, use_residuals=True, seed=42)
    res_res = idx_res.search(queries, k=5, nprobe=2)
    assert res_res.ids.shape == (Q, 5)

    # Non-residual mode
    idx_raw = IVFPQIndex(database=db, nlist=nlist, M=M, k_sub=k_sub, use_residuals=False, seed=42)
    res_raw = idx_raw.search(queries, k=5, nprobe=2)
    assert res_raw.ids.shape == (Q, 5)


def test_ivf_pq_empty_database_and_edge_cases() -> None:
    """Test edge cases: k > N_cand, empty probed lists, functional helpers."""
    # Database with 5 vectors
    db = np.ones((5, 8), dtype=np.float32)
    idx = ivf_pq_build(database=db, nlist=2, M=2, k_sub=4, nprobe=1, seed=42)

    # Search with k=10 > 5
    q = np.ones(8, dtype=np.float32)
    res = ivf_pq_search(idx, q, k=10)
    assert len(res.ids) <= 5

    # 0 queries
    q_empty = np.empty((0, 8), dtype=np.float32)
    res_empty = idx.search(q_empty, k=5)
    assert res_empty.ids.shape == (0, 5)


def test_ivf_pq_approximate_recall_evaluation() -> None:
    """Verify that IVF-PQ achieves reasonable approximate recall against exact search."""
    from indexora.data import generate_clustered_dataset
    from indexora.eval import compute_recall_at_k

    ds = generate_clustered_dataset(
        n_base=500, n_queries=20, d=32, n_clusters=10, seed=42
    ).dataset

    idx = IVFPQIndex(database=ds.base, nlist=10, M=4, k_sub=16, seed=42)
    exact_res = exact_search(ds.queries, ds.base, k=10)

    # Approximate search with nprobe=1
    res_p1 = idx.search(ds.queries, k=10, nprobe=1)
    # Approximate search with nprobe=10 (all coarse centroids probed)
    res_p10 = idx.search(ds.queries, k=10, nprobe=10)

    recall_p1 = compute_recall_at_k(res_p1, exact_res, k=10)
    recall_p10 = compute_recall_at_k(res_p10, exact_res, k=10)

    assert recall_p10 >= recall_p1
    assert recall_p10 > 0.35  # Meaningful approximate recall
