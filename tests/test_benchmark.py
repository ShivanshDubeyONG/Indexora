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
    ExactFlatIndex,
    IVFFlatIndex,
    RunMetrics,
    compute_recall_at_k,
    count_distance_computations,
    generate_clustered_dataset,
    run_benchmark,
)
from indexora.benchmark import main


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
    assert len(result.runs) == 3  # exact, ivf(p=1), ivf(p=3)

    # 1. Exact run verification
    exact_run = result.runs[0]
    assert exact_run.index_name == "exact"
    assert exact_run.recall_at_k == 1.0
    assert exact_run.distance_computations_per_query == 60.0
    assert exact_run.p50_latency_ms > 0.0
    assert exact_run.qps > 0.0

    # 2. IVF run verification
    ivf_p1 = result.runs[1]
    assert ivf_p1.index_name == "ivf_flat"
    assert ivf_p1.parameters["nprobe"] == 1
    assert 0.0 <= ivf_p1.recall_at_k <= 1.0
    assert ivf_p1.distance_computations_per_query < 60.0

    ivf_p3 = result.runs[2]
    assert ivf_p3.index_name == "ivf_flat"
    assert ivf_p3.parameters["nprobe"] == 3
    # All-probe must achieve 1.0 recall
    assert ivf_p3.recall_at_k == 1.0
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
    assert len(loaded["runs"]) == 3
    assert loaded["environment"]["numpy_version"] == np.__version__

    # 4. Export to CSV
    csv_path = tmp_path / "summary.csv"
    result.to_csv(csv_path)
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8")
    assert "index_name,nlist,nprobe" in content
    assert "exact" in content
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
