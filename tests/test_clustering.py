"""Tests for Phase 4: K-means++ and Lloyd's clustering algorithm.

These tests establish the strict validation, invariants, and mathematical contracts
for Phase 4 per ADR-0005 and plan/implementation.md.

Algorithmic tests for `kmeans_plus_plus` and `kmeans` are marked with `xfail(raises=NotImplementedError)`
because Phase 4 is a human-core phase. As the owner implements the core loops, these
tests will transition from XFAIL to XPASS.
"""

from __future__ import annotations

import numpy as np
import pytest

from indexora.clustering import KMeansResult, _validate_kmeans_inputs, kmeans, kmeans_plus_plus
from indexora.data import generate_clustered_dataset
from indexora.kernels import pairwise_squared_l2_ref


# ============================================================================
# 1. Input Validation Tests (Passes immediately against scaffolding)
# ============================================================================


def test_kmeans_result_dataclass_validation() -> None:
    """Test KMeansResult dataclass enforces valid attribute types and shapes."""
    centroids = np.zeros((3, 4), dtype=np.float32)
    labels = np.zeros(10, dtype=np.int64)

    res = KMeansResult(
        centroids=centroids,
        labels=labels,
        inertia=12.5,
        n_iter=5,
        converged=True,
        inertia_history=(20.0, 15.0, 12.5),
    )
    assert res.centroids.shape == (3, 4)
    assert res.labels.shape == (10,)
    assert res.inertia == 12.5
    assert res.n_iter == 5
    assert res.converged is True
    assert len(res.inertia_history) == 3

    # Reject non-float32 centroids
    with pytest.raises(TypeError, match="centroids must be a float32"):
        KMeansResult(
            centroids=np.zeros((3, 4), dtype=np.float64),
            labels=labels,
            inertia=0.0,
            n_iter=1,
            converged=True,
        )

    # Reject non-int64 labels
    with pytest.raises(TypeError, match="labels must be an int64"):
        KMeansResult(
            centroids=centroids,
            labels=np.zeros(10, dtype=np.int32),
            inertia=0.0,
            n_iter=1,
            converged=True,
        )

    # Reject negative inertia
    with pytest.raises(ValueError, match="inertia cannot be negative"):
        KMeansResult(
            centroids=centroids,
            labels=labels,
            inertia=-1.0,
            n_iter=1,
            converged=True,
        )


def test_validate_kmeans_inputs_type_and_dtype() -> None:
    """Test strict rejection of non-array or non-float32 matrix inputs (no silent casting)."""
    # Non-array
    with pytest.raises(TypeError, match="X must be a numpy ndarray"):
        _validate_kmeans_inputs([[1.0, 2.0]], k=1)  # type: ignore[arg-type]

    # float64
    x_f64 = np.ones((5, 2), dtype=np.float64)
    with pytest.raises(TypeError, match="X must have strict dtype float32"):
        _validate_kmeans_inputs(x_f64, k=2)

    # int32
    x_int = np.ones((5, 2), dtype=np.int32)
    with pytest.raises(TypeError, match="X must have strict dtype float32"):
        _validate_kmeans_inputs(x_int, k=2)


def test_validate_kmeans_inputs_shapes_and_contiguity() -> None:
    """Test rejection of non-2D, non-contiguous, or non-finite inputs."""
    # 1D array
    with pytest.raises(ValueError, match="X must be a 2D array"):
        _validate_kmeans_inputs(np.ones(10, dtype=np.float32), k=2)

    # 3D array
    with pytest.raises(ValueError, match="X must be a 2D array"):
        _validate_kmeans_inputs(np.ones((2, 2, 2), dtype=np.float32), k=2)

    # Fortran-contiguous (non-C-contiguous)
    x_f = np.asfortranarray(np.ones((5, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="X must be C-contiguous"):
        _validate_kmeans_inputs(x_f, k=2)

    # NaN / Inf
    x_nan = np.ones((5, 3), dtype=np.float32)
    x_nan[1, 1] = np.nan
    with pytest.raises(ValueError, match="X must contain only finite numbers"):
        _validate_kmeans_inputs(x_nan, k=2)

    x_inf = np.ones((5, 3), dtype=np.float32)
    x_inf[2, 0] = np.inf
    with pytest.raises(ValueError, match="X must contain only finite numbers"):
        _validate_kmeans_inputs(x_inf, k=2)


def test_validate_kmeans_inputs_k_boundaries() -> None:
    """Test boundary conditions on k: k < 1, non-integer k, and k > N."""
    X = np.ones((10, 4), dtype=np.float32)

    # k < 1
    with pytest.raises(ValueError, match="k must be at least 1"):
        _validate_kmeans_inputs(X, k=0)

    # non-integer k
    with pytest.raises(TypeError, match="k must be an integer"):
        _validate_kmeans_inputs(X, k=2.5)  # type: ignore[arg-type]

    # k > N
    with pytest.raises(ValueError, match="k cannot exceed number of samples N"):
        _validate_kmeans_inputs(X, k=11)

    # Valid boundary: k == N is valid
    N, d = _validate_kmeans_inputs(X, k=10)
    assert N == 10 and d == 4


def test_validate_kmeans_inputs_hyperparameters() -> None:
    """Test validation of max_iter, tol, and empty_cluster_policy."""
    X = np.ones((10, 4), dtype=np.float32)

    # max_iter < 1
    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        _validate_kmeans_inputs(X, k=2, max_iter=0)

    # tol < 0
    with pytest.raises(ValueError, match="tol must be non-negative"):
        _validate_kmeans_inputs(X, k=2, tol=-1e-4)

    # invalid empty_cluster_policy
    with pytest.raises(ValueError, match="Unknown empty_cluster_policy"):
        _validate_kmeans_inputs(X, k=2, empty_cluster_policy="random_drop")


# ============================================================================
# ============================================================================
# 2. K-means++ Initialization Tests (Implemented)
# ============================================================================


def test_kmeans_plus_plus_shape_and_dtype() -> None:
    """Test that kmeans_plus_plus returns array of shape (k, d) with dtype float32."""
    rng = np.random.default_rng(42)
    X = np.ascontiguousarray(rng.standard_normal((50, 8)), dtype=np.float32)
    k = 5

    centroids = kmeans_plus_plus(X, k=k, seed=42)

    assert isinstance(centroids, np.ndarray)
    assert centroids.shape == (k, 8)
    assert centroids.dtype == np.float32
    assert centroids.flags.c_contiguous
    assert np.all(np.isfinite(centroids))


def test_kmeans_plus_plus_centroids_are_rows_of_X() -> None:
    """Test that all initialized centroids are exact rows from the input matrix X."""
    rng = np.random.default_rng(123)
    X = np.ascontiguousarray(rng.uniform(low=-10.0, high=10.0, size=(40, 4)), dtype=np.float32)
    k = 6

    centroids = kmeans_plus_plus(X, k=k, seed=123)

    # Every centroid row must exactly match at least one row in X
    for j in range(k):
        c = centroids[j]
        matches = np.isclose(X, c, atol=1e-6)
        row_matches = np.all(matches, axis=1)
        assert np.any(row_matches), f"Centroid {j} does not correspond to any input row in X"


def test_kmeans_plus_plus_seeded_reproducibility() -> None:
    """Test that fixed seeds produce bit-for-bit identical initializations."""
    rng = np.random.default_rng(999)
    X = np.ascontiguousarray(rng.standard_normal((30, 6)), dtype=np.float32)

    c1 = kmeans_plus_plus(X, k=4, seed=42)
    c2 = kmeans_plus_plus(X, k=4, seed=42)
    c3 = kmeans_plus_plus(X, k=4, seed=100)

    # Identical seed -> identical centroids
    np.testing.assert_array_equal(c1, c2)

    # Different seed -> different initialization
    assert not np.array_equal(c1, c3)


def test_kmeans_plus_plus_duplicate_vectors_degenerate_distribution() -> None:
    """Test that duplicate or identical points do not crash or produce NaNs when sum(D^2) == 0."""
    # 1. All 10 points identical: fallback must choose distinct unselected row indices
    X_identical = np.tile(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), (10, 1))

    centroids = kmeans_plus_plus(X_identical, k=3, seed=42)
    assert centroids.shape == (3, 3)
    assert np.all(np.isfinite(centroids))
    expected_identical = np.tile(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), (3, 1))
    np.testing.assert_allclose(centroids, expected_identical)

    # 2. Distinct points exhausted: N=5 with only 2 distinct values, but k=4
    X_few_distinct = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )
    c_few = kmeans_plus_plus(X_few_distinct, k=4, seed=42)
    assert c_few.shape == (4, 2)
    assert np.all(np.isfinite(c_few))
    for row in c_few:
        assert np.all(row == 1.0) or np.all(row == 2.0)


def test_kmeans_plus_plus_boundary_k1_and_kn() -> None:
    """Test boundary conditions k=1 and k=N for kmeans_plus_plus."""
    rng = np.random.default_rng(55)
    X = np.ascontiguousarray(rng.standard_normal((10, 3)), dtype=np.float32)

    # k = 1
    c1 = kmeans_plus_plus(X, k=1, seed=55)
    assert c1.shape == (1, 3)
    assert c1.dtype == np.float32
    assert np.any(np.all(np.isclose(X, c1[0]), axis=1))

    # k = N
    cN = kmeans_plus_plus(X, k=10, seed=55)
    assert cN.shape == (10, 3)
    assert cN.dtype == np.float32
    # When k=N, every row in X must be selected as a centroid
    sorted_X = np.sort(X, axis=0)
    sorted_cN = np.sort(cN, axis=0)
    np.testing.assert_allclose(sorted_cN, sorted_X, atol=1e-6)


def test_kmeans_plus_plus_generator_and_invalid_seed() -> None:
    """Test passing np.random.Generator as seed and rejection of invalid seed types."""
    X = np.ascontiguousarray(np.ones((6, 2), dtype=np.float32))

    # Valid Generator instance
    gen = np.random.default_rng(1234)
    c = kmeans_plus_plus(X, k=2, seed=gen)
    assert c.shape == (2, 2)

    # Invalid seed types
    with pytest.raises(TypeError, match="seed must be an integer, np.random.Generator, or None"):
        kmeans_plus_plus(X, k=2, seed="invalid_seed")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="seed must be an integer, np.random.Generator, or None"):
        kmeans_plus_plus(X, k=2, seed=12.34)  # type: ignore[arg-type]


def test_kmeans_plus_plus_caller_array_immutability() -> None:
    """Test that kmeans_plus_plus does not mutate caller-owned input arrays in place."""
    rng = np.random.default_rng(88)
    X = np.ascontiguousarray(rng.standard_normal((25, 4)), dtype=np.float32)
    X_copy = X.copy()

    kmeans_plus_plus(X, k=5, seed=88)
    np.testing.assert_array_equal(X, X_copy, err_msg="kmeans_plus_plus mutated X!")


def test_kmeans_plus_plus_extreme_finite_float32_values() -> None:
    """Test that extreme finite float32 values near max float32 do not overflow float64 calculations."""
    max_f32 = np.finfo(np.float32).max  # ~3.4028235e+38
    X_extreme = np.array(
        [
            [max_f32, -max_f32],
            [-max_f32, max_f32],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    centroids = kmeans_plus_plus(X_extreme, k=2, seed=42)
    assert centroids.shape == (2, 2)
    assert centroids.dtype == np.float32
    assert np.all(np.isfinite(centroids))


# ============================================================================
# 3. Lloyd's Algorithm Invariants and Correctness Tests (Implemented)
# ============================================================================


def test_kmeans_hand_calculated_2d_example() -> None:
    """Test Lloyd iteration on a hand-calculated 6-point, 2-cluster 2D dataset.

    Points:
    Cluster 0: [0, 0], [1, 0], [0, 1]
    Cluster 1: [10, 10], [11, 10], [10, 11]

    True means:
    Mean 0 = (1/3, 1/3)
    Mean 1 = (31/3, 31/3)
    """
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],
        ],
        dtype=np.float32,
    )

    res = kmeans(X, k=2, max_iter=20, tol=1e-6, seed=42)

    assert isinstance(res, KMeansResult)
    assert res.centroids.shape == (2, 2)
    assert res.labels.shape == (6,)
    assert res.converged is True

    # Points 0, 1, 2 must share one label, and 3, 4, 5 share the other
    assert res.labels[0] == res.labels[1] == res.labels[2]
    assert res.labels[3] == res.labels[4] == res.labels[5]
    assert res.labels[0] != res.labels[3]

    # Verify converged centroids match exact means (1/3, 1/3) and (31/3, 31/3)
    expected_means = np.sort(
        np.array([[1.0 / 3.0, 1.0 / 3.0], [31.0 / 3.0, 31.0 / 3.0]], dtype=np.float32),
        axis=0,
    )
    actual_means = np.sort(res.centroids, axis=0)
    np.testing.assert_allclose(actual_means, expected_means, atol=1e-5)


def test_kmeans_inertia_independent_calculation() -> None:
    """Test that result.inertia matches independent calculation from pairwise distances."""
    rng = np.random.default_rng(42)
    X = np.ascontiguousarray(rng.standard_normal((30, 4)), dtype=np.float32)

    res = kmeans(X, k=3, max_iter=50, seed=42)

    # Independent calculation using Phase 1 reference kernel
    dist_to_centroids = pairwise_squared_l2_ref(X, res.centroids)
    min_dists = np.min(dist_to_centroids, axis=1)
    expected_inertia = float(np.sum(min_dists.astype(np.float64)))

    assert np.isclose(res.inertia, expected_inertia, rtol=1e-4)


def test_kmeans_monotonic_inertia_decrease() -> None:
    """Test that inertia does not increase across recorded iterations in standard Lloyd steps."""
    rng = np.random.default_rng(77)
    X = np.ascontiguousarray(rng.standard_normal((60, 5)), dtype=np.float32)

    res = kmeans(X, k=4, max_iter=30, tol=1e-8, seed=77)

    assert len(res.inertia_history) >= 1
    # Check monotonic decrease with slight float tolerance
    for t in range(len(res.inertia_history) - 1):
        assert res.inertia_history[t + 1] <= res.inertia_history[t] + 1e-5, (
            f"Inertia increased from iteration {t} ({res.inertia_history[t]}) "
            f"to {t+1} ({res.inertia_history[t+1]})"
        )


def test_kmeans_deterministic_tie_breaking() -> None:
    """Test that vectors equidistant to multiple centroids assign to lowest centroid index."""
    # Point at (0, 0), two candidate centroids at (-2, 0) and (2, 0) equidistant (dist = 4.0)
    dists = pairwise_squared_l2_ref(
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.array([[-2.0, 0.0], [2.0, 0.0]], dtype=np.float32),
    )
    assert np.isclose(dists[0, 0], dists[0, 1])
    # Deterministic assignment: argmin chooses lowest column index 0
    assert np.argmin(dists, axis=1)[0] == 0

    # End-to-end check with kmeans
    X = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    res = kmeans(X, k=2, max_iter=10, seed=42)
    assert isinstance(res, KMeansResult)
    assert res.labels.shape == (3,)


def test_kmeans_empty_cluster_policy_reseed_furthest() -> None:
    """Test empty-cluster handling under 'reseed_furthest'."""
    # Pathological dataset: 4 points clustered tightly at (0, 0) and 1 outlier at (100, 100)
    # With k=4, empty clusters are provoked during iterations if centroids shift
    X = np.array(
        [
            [0.0, 0.0],
            [0.01, 0.0],
            [0.0, 0.01],
            [0.01, 0.01],
            [100.0, 100.0],
        ],
        dtype=np.float32,
    )
    res = kmeans(X, k=4, max_iter=20, empty_cluster_policy="reseed_furthest", seed=42)
    assert res.centroids.shape == (4, 2)
    assert len(np.unique(res.labels)) == 4
    assert np.all(np.isfinite(res.centroids))
    assert not np.isnan(res.inertia)


def test_kmeans_empty_cluster_policy_error() -> None:
    """Test that 'error' policy raises RuntimeError when an empty cluster occurs."""
    X = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="Empty cluster encountered"):
        kmeans(X, k=3, max_iter=10, empty_cluster_policy="error", seed=42)


def test_kmeans_empty_cluster_policy_keep() -> None:
    """Test that 'keep' policy retains previous centroid when a cluster is empty."""
    X = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    res_keep = kmeans(X, k=3, max_iter=10, empty_cluster_policy="keep", seed=42)
    assert isinstance(res_keep, KMeansResult)
    assert res_keep.centroids.shape == (3, 2)
    assert np.all(np.isfinite(res_keep.centroids))


def test_kmeans_iteration_limit_and_convergence_status() -> None:
    """Test that max_iter limit is respected and convergence flag is accurate."""
    rng = np.random.default_rng(101)
    X = np.ascontiguousarray(rng.standard_normal((40, 4)), dtype=np.float32)

    # max_iter = 1: must stop at 1 iteration
    res_1 = kmeans(X, k=3, max_iter=1, seed=101)
    assert res_1.n_iter == 1
    assert len(res_1.inertia_history) == 1

    # Large max_iter on easily separable data: must converge early
    data = generate_clustered_dataset(n_base=60, n_queries=0, d=3, n_clusters=2, cluster_std=0.01, seed=101)
    res_conv = kmeans(data.dataset.base, k=2, max_iter=100, tol=1e-4, seed=101)
    assert res_conv.converged is True
    assert res_conv.n_iter < 100


def test_kmeans_boundary_k1_and_kn() -> None:
    """Test Lloyd boundary cases k=1 and k=N."""
    rng = np.random.default_rng(202)
    X = np.ascontiguousarray(rng.standard_normal((15, 3)), dtype=np.float32)

    # k = 1: centroid is exact arithmetic mean of all points, all labels 0
    res_k1 = kmeans(X, k=1, max_iter=10)
    assert res_k1.centroids.shape == (1, 3)
    assert np.all(res_k1.labels == 0)
    expected_mean = np.mean(X.astype(np.float64), axis=0).astype(np.float32)
    np.testing.assert_allclose(res_k1.centroids[0], expected_mean, atol=1e-5)

    # k = N: each point is its own centroid, inertia is near zero
    res_kN = kmeans(X, k=15, max_iter=10)
    assert res_kN.centroids.shape == (15, 3)
    assert len(np.unique(res_kN.labels)) == 15
    assert np.isclose(res_kN.inertia, 0.0, atol=1e-4)


def test_kmeans_duplicate_and_identical_vectors() -> None:
    """Test Lloyd on datasets with identical or duplicate vectors."""
    # 8 identical points
    X_ident = np.tile(np.array([[2.5, -1.5]], dtype=np.float32), (8, 1))
    res_ident = kmeans(X_ident, k=2, max_iter=10)
    assert res_ident.centroids.shape == (2, 2)
    assert np.isclose(res_ident.inertia, 0.0, atol=1e-5)
    assert np.all(np.isfinite(res_ident.centroids))


def test_kmeans_final_consistency_labels_centroids_inertia() -> None:
    """Test that final labels and inertia strictly match independent calculation against final centroids."""
    rng = np.random.default_rng(303)
    X = np.ascontiguousarray(rng.standard_normal((35, 4)), dtype=np.float32)

    res = kmeans(X, k=3, max_iter=25, seed=303)

    # Independent assignment pass
    dists = pairwise_squared_l2_ref(X, res.centroids)
    expected_labels = np.argmin(dists, axis=1).astype(np.int64)
    expected_inertia = float(np.sum(np.min(dists, axis=1).astype(np.float64)))

    np.testing.assert_array_equal(res.labels, expected_labels)
    assert np.isclose(res.inertia, expected_inertia, rtol=1e-4)


def test_kmeans_caller_array_immutability() -> None:
    """Test that k-means does not mutate caller-owned input arrays in place."""
    rng = np.random.default_rng(88)
    X = np.ascontiguousarray(rng.standard_normal((25, 4)), dtype=np.float32)
    X_copy = X.copy()

    kmeans_plus_plus(X, k=3, seed=88)
    np.testing.assert_array_equal(X, X_copy, err_msg="kmeans_plus_plus mutated X!")

    kmeans(X, k=3, max_iter=10, seed=88)
    np.testing.assert_array_equal(X, X_copy, err_msg="kmeans mutated X!")


def test_kmeans_on_synthetic_clustered_dataset() -> None:
    """Test k-means recovery of well-separated synthetic clusters from generate_clustered_dataset."""
    # 120 points, 3 clusters well-separated (cluster_std=0.05)
    data = generate_clustered_dataset(
        n_base=120,
        n_queries=0,
        d=4,
        n_clusters=3,
        cluster_std=0.05,
        seed=42,
    )

    res = kmeans(data.dataset.base, k=3, max_iter=50, tol=1e-5, seed=42)

    assert res.converged is True
    assert res.centroids.shape == (3, 4)

    # For each cluster, check that all assigned points belong to the same true ground-truth cluster
    for cluster_id in range(3):
        mask = (res.labels == cluster_id)
        assert np.sum(mask) > 0, f"Cluster {cluster_id} is empty in well-separated data"
        true_labels = data.base_labels[mask]
        # All points in this discovered cluster should share the same ground-truth label
        assert len(np.unique(true_labels)) == 1, (
            f"Cluster {cluster_id} contains mixed true labels: {np.unique(true_labels)}"
        )


def test_kmeans_distance_method_blas_and_reference():
    """Verify distance_method parameter accepts 'blas' and 'reference' with equivalent behavior."""
    rng = np.random.default_rng(123)
    X = rng.standard_normal((60, 4)).astype(np.float32)

    res_blas = kmeans(X, k=3, max_iter=20, distance_method="blas", seed=42)
    res_ref = kmeans(X, k=3, max_iter=20, distance_method="reference", seed=42)

    assert res_blas.converged == res_ref.converged
    assert res_blas.n_iter == res_ref.n_iter
    np.testing.assert_allclose(res_blas.centroids, res_ref.centroids, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(res_blas.labels, res_ref.labels)

    # Invalid distance method raises ValueError
    with pytest.raises(ValueError, match="Unknown distance_method"):
        kmeans(X, k=3, distance_method="euclidean")
