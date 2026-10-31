"""Unit tests for the Interactive Index Lab backend services (src/indexora/lab.py).

Verifies deterministic clustered dataset generation, 2D PCA projection, ground-truth
generation, memory measurement, benchmarking harness, and session mutation/persistence.
"""

from __future__ import annotations

import numpy as np
import pytest

from indexora.exact import ExactBlasIndex, ExactFlatIndex
from indexora.hnsw import HNSWIndex
from indexora.ivf import IVFFlatIndex
from indexora.ivfpq import IVFPQIndex
from indexora.lab import (
    IndexBenchmarkMetrics,
    IndexCache,
    LabDataset,
    LabSession,
    compute_exact_ground_truth,
    evaluate_index_search,
    generate_lab_dataset,
    measure_index_memory,
)


def test_generate_lab_dataset_deterministic():
    """Verify that generate_lab_dataset produces bit-for-bit identical results with the same seed."""
    ds1 = generate_lab_dataset(n_base=60, n_queries=5, d=8, n_clusters=3, seed=123)
    ds2 = generate_lab_dataset(n_base=60, n_queries=5, d=8, n_clusters=3, seed=123)

    assert ds1.N == 60
    assert ds1.Q == 5
    assert ds1.d == 8
    assert ds1.n_clusters == 3

    np.testing.assert_array_equal(ds1.base, ds2.base)
    np.testing.assert_array_equal(ds1.base_ids, ds2.base_ids)
    np.testing.assert_array_equal(ds1.queries, ds2.queries)
    np.testing.assert_array_equal(ds1.pca_mean, ds2.pca_mean)
    np.testing.assert_array_equal(ds1.pca_components, ds2.pca_components)


def test_generate_lab_dataset_validation():
    """Invalid arguments must raise explicit ValueErrors."""
    with pytest.raises(ValueError, match="n_base must be >= 1"):
        generate_lab_dataset(n_base=0)

    with pytest.raises(ValueError, match="n_queries must be >= 1"):
        generate_lab_dataset(n_queries=0)

    with pytest.raises(ValueError, match="Dimensionality d must be >= 2"):
        generate_lab_dataset(d=1)

    with pytest.raises(ValueError, match="cluster_std must be positive"):
        generate_lab_dataset(cluster_std=-0.5)


def test_lab_dataset_pca_projection():
    """Verify that PCA projection maps high-dimensional points to 2D deterministically."""
    ds = generate_lab_dataset(n_base=50, n_queries=4, d=6, n_clusters=3, seed=42)

    # Project 2D batch
    proj_batch = ds.project_2d(ds.base)
    assert proj_batch.shape == (50, 2)
    assert proj_batch.dtype == np.float32

    # Project 1D single vector
    proj_single = ds.project_2d(ds.queries[0])
    assert proj_single.shape == (2,)
    assert proj_single.dtype == np.float32


def test_lab_dataset_ground_truth():
    """Ground-truth generation must yield exact top-k matches with sorted distances."""
    ds = generate_lab_dataset(n_base=40, n_queries=4, d=4, n_clusters=2, seed=99)
    gt = ds.get_ground_truth(k=5)

    assert gt.ids.shape == (4, 5)
    assert gt.distances.shape == (4, 5)
    # Check distances are ascending
    for q_idx in range(4):
        dists = gt.distances[q_idx]
        assert np.all(np.diff(dists) >= -1e-6)


@pytest.mark.parametrize(
    "index_builder",
    [
        lambda ds: ExactFlatIndex(database=ds.base, ids=ds.base_ids),
        lambda ds: ExactBlasIndex(database=ds.base, ids=ds.base_ids),
        lambda ds: IVFFlatIndex(database=ds.base, ids=ds.base_ids, nlist=3, nprobe=3),
        lambda ds: IVFPQIndex(database=ds.base, ids=ds.base_ids, nlist=3, M=2, k_sub=4, nprobe=3),
        lambda ds: HNSWIndex(database=ds.base, ids=ds.base_ids, M=4, ef_construction=16, ef_search=16),
    ],
)
def test_measure_index_memory(index_builder):
    """Memory accounting should return positive KiB and meaningful description."""
    ds = generate_lab_dataset(n_base=30, n_queries=2, d=4, n_clusters=2, seed=1)
    idx = index_builder(ds)

    mem_kb, desc = measure_index_memory(idx)
    assert mem_kb > 0.0
    assert len(desc) > 0


def test_evaluate_index_search():
    """Benchmark evaluation produces complete metrics against exact ground truth."""
    ds = generate_lab_dataset(n_base=40, n_queries=4, d=4, n_clusters=2, seed=7)
    gt = ds.get_ground_truth(k=3)
    idx = ExactBlasIndex(database=ds.base, ids=ds.base_ids)

    metrics = evaluate_index_search(
        index=idx,
        index_name="Exact BLAS",
        queries=ds.queries,
        ground_truth=gt,
        k=3,
        build_time_ms=1.5,
        repetitions=2,
    )

    assert isinstance(metrics, IndexBenchmarkMetrics)
    assert metrics.index_name == "Exact BLAS"
    assert metrics.build_time_ms == 1.5
    assert metrics.recall_at_k == pytest.approx(1.0)
    assert metrics.median_latency_ms > 0.0
    assert metrics.p95_latency_ms >= metrics.median_latency_ms
    assert metrics.qps > 0.0
    assert metrics.distance_computations_per_query == pytest.approx(40.0)


def test_lab_session_mutation_and_persistence(tmp_path):
    """LabSession correctly tracks active count, executes mutations, and persists/restores index."""
    ds = generate_lab_dataset(n_base=30, n_queries=2, d=4, n_clusters=2, seed=10)
    idx = IVFFlatIndex(database=ds.base, ids=ds.base_ids, nlist=3, nprobe=3)
    session = LabSession(index=idx, index_name="IVFFlatIndex", d=ds.d)

    assert session.active_count == 30

    # Add a new vector with specific ID
    new_v = np.zeros(ds.d, dtype=np.float32)
    assigned_id = session.add_vector(new_v, ext_id=9999)
    assert assigned_id == 9999
    assert session.active_count == 31
    assert 9999 in session.added_ids

    # Search for new vector
    res = session.search(new_v, k=1)
    assert res.ids[0] == 9999

    # Delete the vector
    deleted_count = session.delete_vector(9999)
    assert deleted_count == 1
    assert session.active_count == 30
    assert 9999 in session.deleted_ids

    # Search again: 9999 must not appear
    res_after = session.search(new_v, k=1)
    assert res_after.ids[0] != 9999

    # Test persistence round-trip
    save_dir = tmp_path / "lab_test_index"
    manifest = session.save(save_dir)
    assert manifest["format"] == "indexora_index"

    # Reload into session
    restored = session.load(save_dir)
    assert isinstance(restored, IVFFlatIndex)
    assert session.active_count == 30


# -----------------------------------------------------------------------------
# Filtered Search Ground Truth Tests
# -----------------------------------------------------------------------------

def test_compute_exact_ground_truth_unfiltered():
    """Unfiltered exact ground truth returns the true nearest neighbors."""
    ds = generate_lab_dataset(n_base=50, n_queries=3, d=4, n_clusters=2, seed=42)
    q = ds.queries[0]
    gt = compute_exact_ground_truth(database=ds.base, ids=ds.base_ids, query=q, k=5)

    assert len(gt.ids) == 5
    assert len(gt.distances) == 5
    # Verify ascending sorted order
    assert np.all(np.diff(gt.distances) >= -1e-6)


def test_compute_exact_ground_truth_excluded_nearest_neighbor():
    """Excluding the true nearest neighbor forces ground truth to promote the next best vector."""
    ds = generate_lab_dataset(n_base=50, n_queries=3, d=4, n_clusters=2, seed=42)
    q = ds.queries[0]

    # Find unfiltered top-1 neighbor
    gt_unfiltered = compute_exact_ground_truth(database=ds.base, ids=ds.base_ids, query=q, k=2)
    true_top1_id = gt_unfiltered.ids[0]
    true_top2_id = gt_unfiltered.ids[1]

    # Compute filtered ground truth excluding top-1 ID
    gt_filtered = compute_exact_ground_truth(
        database=ds.base,
        ids=ds.base_ids,
        query=q,
        k=2,
        filter_ids=lambda eid: eid != true_top1_id,
    )

    # Excluded ID must not be returned; previous rank 2 becomes rank 1
    assert true_top1_id not in gt_filtered.ids
    assert gt_filtered.ids[0] == true_top2_id


def test_compute_exact_ground_truth_allowed_id_filter():
    """Filtered ground truth with allowed IDs restricts results exclusively to that set."""
    ds = generate_lab_dataset(n_base=50, n_queries=3, d=4, n_clusters=2, seed=42)
    q = ds.queries[0]

    allowed_ids = {int(ds.base_ids[5]), int(ds.base_ids[10]), int(ds.base_ids[15])}
    gt = compute_exact_ground_truth(database=ds.base, ids=ds.base_ids, query=q, k=5, filter_ids=allowed_ids)

    # Since only 3 are allowed, result contains exactly 3 candidates
    assert len(gt.ids) == 3
    for ret_id in gt.ids:
        assert ret_id in allowed_ids


def test_compute_exact_ground_truth_callable_filter():
    """Filtered ground truth with callable predicate accurately evaluates each external ID."""
    ds = generate_lab_dataset(n_base=50, n_queries=3, d=4, n_clusters=2, seed=42)
    q = ds.queries[0]

    # Allow only even IDs
    gt = compute_exact_ground_truth(
        database=ds.base,
        ids=ds.base_ids,
        query=q,
        k=5,
        filter_ids=lambda eid: eid % 2 == 0,
    )

    assert len(gt.ids) == 5
    for ret_id in gt.ids:
        assert ret_id % 2 == 0


def test_compute_exact_ground_truth_empty_eligible_set():
    """When filter excludes all vectors, ground truth returns an empty result with valid shapes."""
    ds = generate_lab_dataset(n_base=50, n_queries=3, d=4, n_clusters=2, seed=42)
    q = ds.queries[0]

    # Filter with IDs outside the dataset
    gt = compute_exact_ground_truth(
        database=ds.base,
        ids=ds.base_ids,
        query=q,
        k=5,
        filter_ids={999999, 888888},
    )

    assert len(gt.ids) == 0
    assert len(gt.distances) == 0
    assert len(gt.mask) == 0


# -----------------------------------------------------------------------------
# IndexCache Invalidation Tests
# -----------------------------------------------------------------------------

def test_index_cache_hit_and_miss():
    """IndexCache returns existing instance on hit and builds new instance on miss."""
    ds = generate_lab_dataset(n_base=30, n_queries=2, d=4, n_clusters=2, seed=42)
    cache = IndexCache()

    build_count = 0

    def factory():
        nonlocal build_count
        build_count += 1
        return ExactBlasIndex(database=ds.base, ids=ds.base_ids)

    # First call: cache miss, builds index
    idx1, t1, was_cached1 = cache.get_or_build("Exact BLAS", ds, {}, factory)
    assert build_count == 1
    assert was_cached1 is False

    # Second call with identical params: cache hit
    idx2, t2, was_cached2 = cache.get_or_build("Exact BLAS", ds, {}, factory)
    assert build_count == 1
    assert was_cached2 is True
    assert idx1 is idx2


def test_index_cache_invalidation_on_build_param_change():
    """Changing build-time parameters invalidates cache and triggers rebuild."""
    ds = generate_lab_dataset(n_base=30, n_queries=2, d=4, n_clusters=2, seed=42)
    cache = IndexCache()

    build_count = 0

    def factory():
        nonlocal build_count
        build_count += 1
        return IVFFlatIndex(database=ds.base, ids=ds.base_ids, nlist=3)

    # Build with nlist=3
    idx1, _, was_cached1 = cache.get_or_build("IVF-Flat", ds, {"nlist": 3}, factory)
    assert build_count == 1
    assert was_cached1 is False

    # Call with nlist=5 (build parameter changed): cache miss, rebuilds
    idx2, _, was_cached2 = cache.get_or_build("IVF-Flat", ds, {"nlist": 5}, factory)
    assert build_count == 2
    assert was_cached2 is False
    assert idx1 is not idx2


def test_index_cache_invalidation_on_dataset_change():
    """Regenerating or switching the dataset invalidates the cache."""
    ds1 = generate_lab_dataset(n_base=30, n_queries=2, d=4, n_clusters=2, seed=42)
    ds2 = generate_lab_dataset(n_base=40, n_queries=2, d=4, n_clusters=2, seed=43)
    cache = IndexCache()

    build_count = 0

    def factory():
        nonlocal build_count
        build_count += 1
        return ExactBlasIndex(database=ds1.base, ids=ds1.base_ids)

    idx1, _, was_cached1 = cache.get_or_build("Exact BLAS", ds1, {}, factory)
    assert build_count == 1
    assert was_cached1 is False

    # Switching to ds2: cache miss due to distinct dataset_id
    idx2, _, was_cached2 = cache.get_or_build("Exact BLAS", ds2, {}, factory)
    assert build_count == 2
    assert was_cached2 is False
    assert idx1 is not idx2


def test_index_cache_query_params_do_not_rebuild():
    """Query-time parameters (e.g. nprobe, ef_search) do not trigger index rebuilds."""
    ds = generate_lab_dataset(n_base=30, n_queries=2, d=4, n_clusters=2, seed=42)
    cache = IndexCache()

    build_count = 0

    def factory():
        nonlocal build_count
        build_count += 1
        return IVFFlatIndex(database=ds.base, ids=ds.base_ids, nlist=3)

    # Build params only contain nlist=3
    build_params = {"nlist": 3}
    idx, _, _ = cache.get_or_build("IVF-Flat", ds, build_params, factory)
    assert build_count == 1

    # Varying nprobe at query time uses the same cached index
    for nprobe in [1, 2, 3]:
        reused_idx, _, was_cached = cache.get_or_build("IVF-Flat", ds, build_params, factory)
        assert was_cached is True
        assert build_count == 1
        # Search executes with different nprobe without rebuilding
        res = reused_idx.search(ds.queries[0], k=2, nprobe=nprobe)
        assert len(res.ids) == 2
