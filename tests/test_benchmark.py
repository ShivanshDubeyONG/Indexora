"""Tests for Phase 6: Benchmark Harness v0.1 and Evaluation Metrics.

These tests verify:
- `compute_recall_at_k`: Accuracy, sentinel padding exclusion, custom IDs, ties, and edge cases.
- `count_distance_computations`: Distance computation accounting for exact and IVF indexes.
- `BenchmarkConfig` & `BenchmarkResult`: Configuration validation, metric tracking, and JSON/CSV/plot export.
- End-to-end benchmark sweeps and CLI `--smoke` execution.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from indexora import (
    BenchmarkConfig,
    BenchmarkResult,
    ExactBlasIndex,
    ExactFlatIndex,
    IVFFlatIndex,
    RunMetrics,
    SearchResult,
    compute_distance_aware_recall_at_k,
    compute_recall_at_k,
    count_distance_computations,
    exact_search,
    generate_clustered_dataset,
    generate_ground_truth,
    run_benchmark,
)
from indexora.benchmark import _resolve_latency_sample, get_process_peak_rss_bytes, main


# ============================================================================
# 1. Recall@k Evaluation & Sentinel Exclusion
# ============================================================================


def test_recall_at_k_perfect_match():
    """Verify Recall@k equals 1.0 when retrieved IDs match ground truth exactly."""
    gt = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
    ret = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)

    assert compute_recall_at_k(ret, gt, k=3) == 1.0
    assert compute_recall_at_k(ret, gt, k=2) == 1.0


def test_recall_at_k_partial_match():
    """Verify Recall@k correctly computes fractional recall."""
    gt = np.array([[1, 2, 3, 4]], dtype=np.int64)
    # Retrieved has 2 of the 4 ground truth items
    ret = np.array([[1, 99, 3, 88]], dtype=np.int64)

    assert compute_recall_at_k(ret, gt, k=4) == 0.5


def test_recall_at_k_sentinel_padding_exclusion():
    """Verify sentinel ID -1 is strictly excluded and does not count as a match or inflate recall."""
    # Query 0: true neighbors [101, 102, 103]
    # Query 1: true neighbors [201, 202, 203]
    gt = np.array([[101, 102, 103], [201, 202, 203]], dtype=np.int64)

    # Retrieved with sentinel -1 padding (e.g. from uneven IVF inverted lists)
    # Query 0 retrieved [101] + sentinels [-1, -1] -> recall should be 1/3
    # Query 1 retrieved [201, 202, 203] -> recall should be 3/3 = 1.0
    ret = np.array([[101, -1, -1], [201, 202, 203]], dtype=np.int64)

    # Average recall = (1/3 + 3/3) / 2 = 4/6 = 0.6666...
    recall = compute_recall_at_k(ret, gt, k=3, sentinel_id=-1)
    np.testing.assert_allclose(recall, 4.0 / 6.0)


def test_recall_at_k_single_query_1d():
    """Verify Recall@k on 1D single-query arrays."""
    gt_1d = np.array([50, 60, 70], dtype=np.int64)
    ret_1d = np.array([50, 99, 70], dtype=np.int64)

    assert compute_recall_at_k(ret_1d, gt_1d, k=3) == pytest.approx(2.0 / 3.0)


def test_recall_at_k_empty_and_zero_queries():
    """Verify Recall@k on empty queries and edge cases."""
    empty_gt = np.empty((0, 5), dtype=np.int64)
    empty_ret = np.empty((0, 5), dtype=np.int64)
    assert compute_recall_at_k(empty_ret, empty_gt, k=5) == 1.0


def test_recall_at_k_validation_errors():
    """Verify input validation for Recall@k."""
    valid = np.array([[1, 2]], dtype=np.int64)

    # Non-int64 dtypes
    with pytest.raises(TypeError, match="int64"):
        compute_recall_at_k(valid.astype(np.int32), valid)
    with pytest.raises(TypeError, match="int64"):
        compute_recall_at_k(valid, valid.astype(np.int32))

    # Dimension mismatch
    with pytest.raises(ValueError, match="Dimension mismatch"):
        compute_recall_at_k(valid[0], valid)

    # Query count mismatch
    with pytest.raises(ValueError, match="Query count mismatch"):
        compute_recall_at_k(valid, np.array([[1, 2], [3, 4]], dtype=np.int64))

    # Invalid k
    with pytest.raises(ValueError, match="k must be a positive integer"):
        compute_recall_at_k(valid, valid, k=0)


# ============================================================================
# 2. Distance Computation Accounting
# ============================================================================


def test_count_distance_computations_exact_flat():
    """Verify exact flat index distance computations equal database size N."""
    db = np.ones((50, 4), dtype=np.float32)
    q = np.ones((10, 4), dtype=np.float32)
    idx = ExactFlatIndex(database=db)

    dc = count_distance_computations(idx, q)
    assert dc == 50.0


def test_count_distance_computations_ivf_flat():
    """Verify IVF distance computations equal nlist coarse comparisons plus candidate counts."""
    # 3 clusters of sizes 5, 10, 15 (total 30 vectors)
    db = np.ones((30, 2), dtype=np.float32)
    # Pre-defined centroids
    c = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]], dtype=np.float32)
    # Assign first 5 to c0, next 10 to c1, last 15 to c2
    db[:5] = [0.0, 0.0]
    db[5:15] = [5.0, 5.0]
    db[15:30] = [10.0, 10.0]

    idx = IVFFlatIndex(database=db, nlist=3, centroids=c)

    # Query at [0.0, 0.0] with nprobe=1 probes list 0 (size 5)
    # DC = 3 (centroids) + 5 (candidates) = 8
    q1 = np.array([0.0, 0.0], dtype=np.float32)
    dc1 = count_distance_computations(idx, q1, nprobe=1)
    assert dc1 == 8.0

    # Query with nprobe=3 (all probe) probes all 3 lists (size 30)
    # DC = 3 (centroids) + 30 (candidates) = 33
    dc_all = count_distance_computations(idx, q1, nprobe=3)
    assert dc_all == 33.0


# ============================================================================
# 3. Benchmark Config & Result Structures
# ============================================================================


def test_benchmark_config_validation():
    """Verify BenchmarkConfig parameter validation."""
    with pytest.raises(ValueError, match="k must be positive"):
        BenchmarkConfig(k=0)

    with pytest.raises(ValueError, match="warmup_runs must be non-negative"):
        BenchmarkConfig(warmup_runs=-1)

    with pytest.raises(ValueError, match="repeat_runs must be at least 1"):
        BenchmarkConfig(repeat_runs=0)

    with pytest.raises(ValueError, match="nlist_values cannot be empty"):
        BenchmarkConfig(nlist_values=())

    with pytest.raises(ValueError, match="nprobe_values cannot be empty"):
        BenchmarkConfig(nprobe_values=())


# ============================================================================
# 4. End-to-End Benchmark Execution & Exporters
# ============================================================================


def test_run_benchmark_end_to_end(tmp_path: Path):
    """Verify run_benchmark produces valid metrics, recall=1.0 for exact and all-probe IVF, and exports files."""
    data = generate_clustered_dataset(
        n_base=60,
        n_queries=6,
        d=4,
        n_clusters=3,
        seed=42,
    )

    config = BenchmarkConfig(
        name="test_suite_run",
        k=5,
        warmup_runs=1,
        repeat_runs=2,
        seed=42,
        nlist_values=(3,),
        nprobe_values=(1, 3),
        smoke=True,
    )

    result = run_benchmark(dataset=data.dataset, config=config, verbose=False)

    assert isinstance(result, BenchmarkResult)
    assert len(result.runs) == 4  # exact_reference, exact_blas, ivf(p=1), ivf(p=3)

    # 1. Exact Reference and BLAS run verification
    exact_ref_run = result.runs[0]
    assert exact_ref_run.index_name == "exact_reference"
    assert exact_ref_run.recall_at_k == 1.0
    assert exact_ref_run.distance_aware_recall_at_k == 1.0
    assert exact_ref_run.distance_computations_per_query == 60.0
    assert exact_ref_run.p50_latency_ms > 0.0
    assert exact_ref_run.qps > 0.0

    exact_blas_run = result.runs[1]
    assert exact_blas_run.index_name == "exact_blas"
    assert exact_blas_run.recall_at_k == 1.0
    assert exact_blas_run.distance_aware_recall_at_k == 1.0
    assert exact_blas_run.distance_computations_per_query == 60.0
    assert exact_blas_run.p50_latency_ms > 0.0
    assert exact_blas_run.qps > 0.0

    # 2. IVF run verification
    ivf_p1 = result.runs[2]
    assert ivf_p1.index_name == "ivf_flat"
    assert ivf_p1.parameters["nprobe"] == 1
    assert 0.0 <= ivf_p1.recall_at_k <= 1.0
    assert ivf_p1.distance_computations_per_query < 60.0
    assert ivf_p1.kmeans_n_iter is not None

    ivf_p3 = result.runs[3]
    assert ivf_p3.index_name == "ivf_flat"
    assert ivf_p3.parameters["nprobe"] == 3
    # All-probe must achieve 1.0 recall
    assert ivf_p3.recall_at_k == 1.0
    assert ivf_p3.distance_aware_recall_at_k == 1.0
    assert ivf_p3.distance_computations_per_query == 63.0  # 3 centroids + 60 vectors

    # 3. Export to JSON
    json_path = tmp_path / "results.json"
    result.to_json(json_path)
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert "environment" in loaded
    assert "dataset_metadata" in loaded
    assert "runs" in loaded
    assert len(loaded["runs"]) == 4
    assert loaded["environment"]["numpy_version"] == np.__version__

    # 4. Export to CSV
    csv_path = tmp_path / "summary.csv"
    result.to_csv(csv_path)
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8")
    assert "index_name,nlist,nprobe" in content
    assert "exact_reference" in content
    assert "exact_blas" in content
    assert "ivf_flat" in content

    # 5. Plotting (if matplotlib installed)
    plots = result.plot(tmp_path)
    for p in plots:
        assert Path(p).exists()


# ============================================================================
# 5. CLI Smoke Execution
# ============================================================================


def test_benchmark_cli_smoke(tmp_path: Path):
    """Verify CLI --smoke flag runs end-to-end and outputs files."""
    out_dir = str(tmp_path / "cli_smoke")
    ret_code = main(["--smoke", "--output-dir", out_dir, "--seed", "42"])
    assert ret_code == 0

    assert (Path(out_dir) / "benchmark_results.json").exists()
    assert (Path(out_dir) / "benchmark_summary.csv").exists()


# ============================================================================
# 6. Advanced Edge Cases & Reproducibility
# ============================================================================


def test_recall_at_k_custom_and_negative_stable_ids():
    """Verify Recall@k correctly handles large and negative custom IDs."""
    # Custom IDs including negative (non-sentinel) and large values
    gt = np.array([[-999999, 0, 1000000000000]], dtype=np.int64)
    # Retrieved 2 of 3 valid IDs, plus sentinel -1 padding
    ret = np.array([[-999999, -1, 1000000000000]], dtype=np.int64)

    recall = compute_recall_at_k(ret, gt, k=3, sentinel_id=-1)
    np.testing.assert_allclose(recall, 2.0 / 3.0)


def test_recall_at_k_k_greater_than_gt_columns_raises():
    """Verify ValueError is raised when requested k exceeds ground-truth column count."""
    gt = np.array([[10, 20]], dtype=np.int64)
    ret = np.array([[10, 20]], dtype=np.int64)

    with pytest.raises(ValueError, match="Requested k=5 exceeds ground-truth column count"):
        compute_recall_at_k(ret, gt, k=5)


def test_recall_at_k_fewer_than_k_database_vectors():
    """Verify Recall@k behavior when database has fewer than k elements (N < k)."""
    # When N=2 and k=5, exact search returns shape (1, 2)
    gt = np.array([[100, 200]], dtype=np.int64)
    # IVF search might return padded shape (1, 5) with sentinels
    ret = np.array([[100, 200, -1, -1, -1]], dtype=np.int64)

    # Calling compute_recall_at_k with k=None (eff_k = gt.shape[1] = 2)
    recall = compute_recall_at_k(ret, gt, sentinel_id=-1)
    assert recall == 1.0


def test_benchmark_reproducibility():
    """Verify run_benchmark produces bit-identical recall and metrics given identical seeds."""
    data = generate_clustered_dataset(
        n_base=50,
        n_queries=5,
        d=4,
        n_clusters=3,
        seed=123,
    )

    config = BenchmarkConfig(
        name="reproducibility_test",
        k=3,
        warmup_runs=1,
        repeat_runs=2,
        seed=123,
        nlist_values=(3,),
        nprobe_values=(1, 2, 3),
        smoke=True,
    )

    res1 = run_benchmark(dataset=data.dataset, config=config, verbose=False)
    res2 = run_benchmark(dataset=data.dataset, config=config, verbose=False)

    assert len(res1.runs) == len(res2.runs)
    for r1, r2 in zip(res1.runs, res2.runs):
        assert r1.index_name == r2.index_name
        assert r1.parameters == r2.parameters
        assert r1.recall_at_k == r2.recall_at_k
        assert r1.distance_computations_per_query == r2.distance_computations_per_query


def test_benchmark_separation_of_build_and_query_timing():
    """Verify that build time and query latency are tracked as distinct, positive measurements."""
    data = generate_clustered_dataset(
        n_base=40,
        n_queries=4,
        d=4,
        n_clusters=2,
        seed=99,
    )

    config = BenchmarkConfig(
        name="timing_separation_test",
        k=2,
        warmup_runs=1,
        repeat_runs=3,
        seed=99,
        nlist_values=(2,),
        nprobe_values=(1,),
        smoke=True,
    )

    result = run_benchmark(dataset=data.dataset, config=config, verbose=False)
    for r in result.runs:
        # Build time is recorded
        assert r.build_time_seconds >= 0.0
        # Query latency is recorded with sample count samples
        assert len(r.query_latencies_ms) == r.per_query_sample_count
        assert r.per_query_sample_count == 4
        assert r.per_query_p50_ms > 0.0
        assert r.per_query_min_ms <= r.per_query_p50_ms <= r.per_query_max_ms
        assert r.batch_repeat_runs == 3
        assert r.batch_qps > 0.0
        assert r.index_memory_bytes > 0
        assert r.index_memory_mib > 0.0
        assert r.bytes_per_vector > 0.0


# ============================================================================
# 7. Phase 6.5: -1 ID Correctness vs. Padding Validity
# ============================================================================


def test_real_id_negative_one_exact_search_validity():
    """Verify that ID -1 is a legitimate entity ID in ExactFlatIndex and is marked valid."""
    db = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    ids = np.array([10, 20, -1], dtype=np.int64)
    idx = ExactFlatIndex(database=db, ids=ids)

    # Query closest to [-1.0, 0.0] (which has ID -1)
    q = np.array([-1.0, 0.0], dtype=np.float32)
    res = idx.search(q, k=1)

    assert res.ids[0] == -1
    assert res.mask[0] is True or res.mask[0] == 1
    assert res.distances[0] == pytest.approx(0.0)


def test_real_id_negative_one_ivf_recall_hit():
    """Verify that ID -1 in an IVF-Flat index is a valid entity and counts as a recall hit."""
    db = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32)
    ids = np.array([-1, 42], dtype=np.int64)
    idx = IVFFlatIndex(database=db, nlist=2, ids=ids, centroids=db)

    gt = generate_ground_truth(queries=db, database=db, ids=ids, k=1)
    assert gt.ids[0, 0] == -1
    assert gt.mask[0, 0] is True or gt.mask[0, 0] == 1

    res = idx.search(db, k=1, nprobe=2)
    assert res.ids[0, 0] == -1
    assert res.mask[0, 0] is True or res.mask[0, 0] == 1

    # compute_recall_at_k using SearchResult directly
    recall = compute_recall_at_k(res, gt, k=1)
    assert recall == 1.0


def test_padding_sentinels_distinguished_from_real_negative_one():
    """Verify that explicit SearchResult mask distinguishes real ID -1 from padded slots."""
    # Query 0 retrieved valid ID -1 and valid ID 10
    # Query 1 retrieved valid ID 20, followed by a padded slot (marked mask=False)
    # Even though both use ID value -1, slot 1 in query 1 is padding!
    ret_ids = np.array([[-1, 10], [20, -1]], dtype=np.int64)
    ret_dists = np.array([[0.1, 0.2], [0.3, np.inf]], dtype=np.float32)
    ret_mask = np.array([[True, True], [True, False]], dtype=bool)
    ret_res = SearchResult(distances=ret_dists, ids=ret_ids, mask=ret_mask)

    # Ground truth: query 0 has [-1, 10], query 1 has [20, 30]
    gt_ids = np.array([[-1, 10], [20, 30]], dtype=np.int64)
    gt_dists = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    gt_res = SearchResult(distances=gt_dists, ids=gt_ids)

    # Query 0 matched 2/2 (including real -1!) -> recall 1.0
    # Query 1 matched 1/2 ([20], padded -1 is excluded) -> recall 0.5
    # Overall recall = (1.0 + 0.5) / 2 = 0.75
    recall = compute_recall_at_k(ret_res, gt_res, k=2)
    assert recall == pytest.approx(0.75)


def test_mixed_batch_uneven_candidates_preserves_validity_mask():
    """Verify that batched IVF search sets mask=False only for padded missing candidate slots."""
    # 2 clusters: cluster 0 has 1 vector, cluster 1 has 3 vectors
    db = np.array([[0.0, 0.0], [10.0, 10.0], [10.1, 10.1], [10.2, 10.2]], dtype=np.float32)
    ids = np.array([100, 201, 202, 203], dtype=np.int64)
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    idx = IVFFlatIndex(database=db, nlist=2, ids=ids, centroids=centroids)

    # Query 0 close to cluster 0 (probes list 0 with nprobe=1 -> 1 candidate)
    # Query 1 close to cluster 1 (probes list 1 with nprobe=1 -> 3 candidates)
    queries = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    res = idx.search(queries, k=3, nprobe=1)

    assert res.ids.shape == (2, 3)
    assert res.mask.shape == (2, 3)

    # Query 0 retrieved 1 vector, padded 2 slots
    assert res.mask[0, 0] is True or res.mask[0, 0] == 1
    assert not res.mask[0, 1]
    assert not res.mask[0, 2]

    # Query 1 retrieved 3 vectors, all valid
    assert np.all(res.mask[1])


# ============================================================================
# 8. Phase 6.5: Index Storage Memory and Actual Distance Work
# ============================================================================


def test_index_storage_memory_accounting():
    """Verify exact formula for index storage bytes, MiB, and bytes per vector."""
    N = 100
    d = 16
    db = np.ones((N, d), dtype=np.float32)
    ids = np.arange(N, dtype=np.int64)

    # Exact index storage: N * (d * 4 + 8) = 100 * (64 + 8) = 7,200 bytes
    exact_idx = ExactFlatIndex(database=db, ids=ids)
    expected_exact_bytes = N * (d * 4 + 8)
    assert exact_idx.index_memory_bytes == expected_exact_bytes
    assert exact_idx.index_memory_mib == expected_exact_bytes / (1024 * 1024)
    assert exact_idx.bytes_per_vector == (d * 4 + 8)

    # IVF index storage: nlist * d * 4 + N * (d * 4 + 8) = 5 * 64 + 7,200 = 7,520 bytes
    nlist = 5
    ivf_idx = IVFFlatIndex(database=db, nlist=nlist, ids=ids, seed=42)
    expected_ivf_bytes = nlist * d * 4 + N * (d * 4 + 8)
    assert ivf_idx.index_memory_bytes == expected_ivf_bytes
    assert ivf_idx.index_memory_mib == expected_ivf_bytes / (1024 * 1024)
    assert ivf_idx.bytes_per_vector == float(expected_ivf_bytes / N)


def test_search_path_distance_computation_instrumentation():
    """Verify that reported distance computations reflect actual search path work."""
    N = 40
    d = 4
    db = np.ones((N, d), dtype=np.float32)
    q = np.ones((2, d), dtype=np.float32)
    exact_idx = ExactFlatIndex(database=db)

    # Exact search on 2 queries against 40 database vectors -> 80 total, 40 per query
    exact_res = exact_idx.search(q, k=5)
    assert exact_res.distance_computations == 80
    assert count_distance_computations(exact_idx, q) == 40.0

    # IVF search with nlist=4
    ivf_idx = IVFFlatIndex(database=db, nlist=4, seed=42)
    # All-probe (nprobe=4) -> each query computes 4 coarse + 40 candidates = 44
    # 2 queries -> total 88
    all_res = ivf_idx.search(q, k=5, nprobe=4)
    assert all_res.distance_computations == 88
    assert count_distance_computations(ivf_idx, q, nprobe=4) == 44.0


def test_benchmark_cli_sift_missing_args():
    """Verify CLI returns exit code 1 when --dataset sift_fvecs is used without paths."""
    ret = main(["--dataset", "sift_fvecs"])
    assert ret == 1


def test_ivf_all_probe_exact_equivalence_negative_one_and_ties():
    """Verify bit-for-bit equivalence between all-probe IVF and exact search with ID -1 and distance ties."""
    db = np.array(
        [
            [1.0, 1.0],   # ID 10
            [2.0, 2.0],   # ID -1 (tied distance with ID -50)
            [2.0, 2.0],   # ID -50 (tied distance with ID -1)
            [3.0, 3.0],   # ID 100
        ],
        dtype=np.float32,
    )
    ids = np.array([10, -1, -50, 100], dtype=np.int64)
    queries = np.array([[2.0, 2.0], [0.0, 0.0]], dtype=np.float32)

    ivf = IVFFlatIndex(database=db, nlist=3, ids=ids, seed=42)
    ivf_res = ivf.search(queries, k=4, nprobe=3)
    exact_res = exact_search(queries, db, ids=ids, k=4)

    np.testing.assert_array_equal(ivf_res.ids, exact_res.ids)
    np.testing.assert_allclose(ivf_res.distances, exact_res.distances)
    np.testing.assert_array_equal(ivf_res.mask, exact_res.mask)
    # Check that ID -1 is present and valid
    assert -1 in ivf_res.ids[0]
    assert np.all(ivf_res.mask)


def test_exact_search_empty_queries_and_db_distance_computations():
    """Verify exact_search records distance_computations=0 for empty queries or empty database."""
    db = np.ones((10, 4), dtype=np.float32)
    empty_q = np.empty((0, 4), dtype=np.float32)
    res_eq = exact_search(empty_q, db, k=5)
    assert res_eq.distance_computations == 0

    empty_db = np.empty((0, 4), dtype=np.float32)
    q = np.ones((3, 4), dtype=np.float32)
    res_edb = exact_search(q, empty_db, k=5)
    assert res_edb.distance_computations == 0


def test_benchmark_cli_no_plot_flag(tmp_path: Path):
    """Verify CLI --no-plot flag avoids plot generation and exits cleanly."""
    out_dir = str(tmp_path / "cli_no_plot")
    ret_code = main(["--smoke", "--no-plot", "--output-dir", out_dir])
    assert ret_code == 0
    assert (Path(out_dir) / "benchmark_results.json").exists()
    assert (Path(out_dir) / "benchmark_summary.csv").exists()
    assert not (Path(out_dir) / "recall_vs_latency.png").exists()


def test_benchmark_cli_plot_failure_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify CLI exits with code 1 when plot generation fails and --no-plot was not passed."""
    out_dir = str(tmp_path / "cli_plot_fail")

    def mock_plot(self, path):
        raise RuntimeError("Simulated missing matplotlib dependency")

    monkeypatch.setattr(BenchmarkResult, "plot", mock_plot)
    ret_code = main(["--smoke", "--output-dir", out_dir])
    assert ret_code == 1


def test_deterministic_distance_computation_counts_mathematical_oracle():
    """Verify mathematically exact distance computation breakdown with N=200, Q=20, nlist=5."""
    N = 200
    Q = 20
    d = 4
    nlist = 5
    centroids = np.zeros((nlist, d), dtype=np.float32)
    for c_idx in range(nlist):
        centroids[c_idx] = float(c_idx * 100)

    db_list = []
    ids_list = []
    for c_idx in range(nlist):
        cluster_vecs = np.full((40, d), float(c_idx * 100), dtype=np.float32)
        cluster_ids = np.arange(c_idx * 40, (c_idx + 1) * 40, dtype=np.int64)
        db_list.append(cluster_vecs)
        ids_list.append(cluster_ids)
    database = np.concatenate(db_list, axis=0)
    ids = np.concatenate(ids_list, axis=0)

    query_list = []
    for c_idx in range(nlist):
        query_list.append(np.full((4, d), float(c_idx * 100), dtype=np.float32))
    queries = np.concatenate(query_list, axis=0)

    # 1. Exact Flat verification
    exact_idx = ExactFlatIndex(database=database, ids=ids)
    exact_res = exact_idx.search(queries, k=5)
    assert exact_res.distance_computations == 4000
    assert exact_res.centroid_distance_computations == 0
    assert exact_res.candidate_distance_computations == 4000

    # 2. IVF-Flat with nprobe=1 (each query probes exactly 1 list of size 40)
    ivf_idx = IVFFlatIndex(database=database, nlist=nlist, ids=ids, centroids=centroids)
    ivf_p1_res = ivf_idx.search(queries, k=5, nprobe=1)
    assert ivf_p1_res.centroid_distance_computations == 100
    assert ivf_p1_res.candidate_distance_computations == 800
    assert ivf_p1_res.distance_computations == 900

    # 3. IVF-Flat with nprobe=5 (all-probe: probes all 5 lists of total size 200)
    ivf_all_res = ivf_idx.search(queries, k=5, nprobe=5)
    assert ivf_all_res.centroid_distance_computations == 100
    assert ivf_all_res.candidate_distance_computations == 4000
    assert ivf_all_res.distance_computations == 4100

    # Bit-for-bit all-probe equivalence to exact search
    np.testing.assert_array_equal(ivf_all_res.ids, exact_res.ids)
    np.testing.assert_allclose(ivf_all_res.distances, exact_res.distances)
    np.testing.assert_array_equal(ivf_all_res.mask, exact_res.mask)


# ============================================================================
# Phase 6.6: Distance-Aware Recall & Benchmark Enhancements
# ============================================================================


def test_distance_aware_recall_boundary_tie_resolution():
    """Verify distance-aware recall yields 1.0 when retrieved candidates match distance at rank k."""
    # Query has 2 candidates at exact distance 1.0, but with different IDs (e.g. 101 and 102)
    # Ground truth top-1 picked ID 101.
    # Retrieved top-1 picked ID 102.
    # Under pure ID recall@1: recall is 0.0.
    # Under distance-aware recall@1: retrieved item distance 1.0 <= GT max distance 1.0 -> recall is 1.0!
    gt_res = SearchResult(
        distances=np.array([[1.0]], dtype=np.float32),
        ids=np.array([[101]], dtype=np.int64),
    )
    ret_res = SearchResult(
        distances=np.array([[1.0]], dtype=np.float32),
        ids=np.array([[102]], dtype=np.int64),
    )

    standard_recall = compute_recall_at_k(ret_res, gt_res, k=1)
    dist_aware_recall = compute_distance_aware_recall_at_k(ret_res, gt_res, k=1)

    assert standard_recall == 0.0
    assert dist_aware_recall == 1.0


def test_distance_aware_recall_with_inferior_candidate():
    """Verify distance-aware recall is not inflated when retrieved distance exceeds GT max distance."""
    gt_res = SearchResult(
        distances=np.array([[1.0]], dtype=np.float32),
        ids=np.array([[101]], dtype=np.int64),
    )
    # Retrieved item has distance 5.0 (greater than gt threshold 1.0)
    ret_res = SearchResult(
        distances=np.array([[5.0]], dtype=np.float32),
        ids=np.array([[102]], dtype=np.int64),
    )

    dist_aware_recall = compute_distance_aware_recall_at_k(ret_res, gt_res, k=1)
    assert dist_aware_recall == 0.0


def test_count_distance_computations_exact_blas():
    """Verify count_distance_computations supports ExactBlasIndex."""
    db = np.ones((40, 8), dtype=np.float32)
    q = np.ones((3, 8), dtype=np.float32)
    idx = ExactBlasIndex(database=db)

    dc = count_distance_computations(idx, q)
    assert dc == 40.0


def test_resolve_latency_sample_deterministic_and_bounded():
    """Verify _resolve_latency_sample deterministically samples queries."""
    rng = np.random.default_rng(42)
    queries = rng.standard_normal((100, 4)).astype(np.float32)

    sample1 = _resolve_latency_sample(queries, sample_count=20, seed=42)
    sample2 = _resolve_latency_sample(queries, sample_count=20, seed=42)
    sample_diff_seed = _resolve_latency_sample(queries, sample_count=20, seed=99)

    assert sample1.shape == (20, 4)
    np.testing.assert_array_equal(sample1, sample2)
    assert not np.array_equal(sample1, sample_diff_seed)

    # When sample_count >= n_queries, returns original queries
    full_sample = _resolve_latency_sample(queries, sample_count=200, seed=42)
    np.testing.assert_array_equal(full_sample, queries)


def test_get_process_peak_rss_bytes():
    """Verify get_process_peak_rss_bytes returns a valid positive integer or None."""
    rss = get_process_peak_rss_bytes()
    if rss is not None:
        assert isinstance(rss, int)
        assert rss > 0
