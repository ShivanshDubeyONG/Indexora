"""Tests for Indexora distance kernels (Phase 1).

These tests verify the reference squared-L2 distance kernel (`pairwise_squared_l2_ref`)
against the contracts and test matrix specified in `docs/phase1_test_spec.md` and
`docs/decisions.md`.
"""

import numpy as np
import pytest

from indexora.kernels import pairwise_squared_l2_ref


def test_t1_1_identical_vectors():
    """T1.1: Distance self-identity. Distance between identical vectors must be 0.0 exactly."""
    q = np.array([1.5, -2.0, 3.0], dtype=np.float32)
    X = np.array([[1.5, -2.0, 3.0]], dtype=np.float32)

    # Single query (1D)
    dist_single = pairwise_squared_l2_ref(q, X)
    assert dist_single.shape == (1,)
    assert dist_single.dtype == np.float32
    assert dist_single[0] == 0.0

    # Batched query (2D)
    Q = q.reshape(1, 3)
    dist_batch = pairwise_squared_l2_ref(Q, X)
    assert dist_batch.shape == (1, 1)
    assert dist_batch.dtype == np.float32
    assert dist_batch[0, 0] == 0.0


def test_t1_2_hand_calculated_distances():
    """T1.2: Hand-calculated distances against 4 known 2D vectors from the learning guide."""
    X = np.array(
        [
            [1.0, 2.0],  # Point A
            [4.0, 6.0],  # Point B
            [2.0, 1.0],  # Point C
            [1.0, 2.0],  # Point D (duplicate of A)
        ],
        dtype=np.float32,
    )
    expected_values = [1.0, 34.0, 1.0, 1.0]

    # Batched query: shape (1, 2)
    Q = np.array([[1.0, 1.0]], dtype=np.float32)
    dist_batch = pairwise_squared_l2_ref(Q, X)
    assert dist_batch.shape == (1, 4)
    assert dist_batch.dtype == np.float32
    np.testing.assert_allclose(
        dist_batch,
        np.array([expected_values], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

    # Single query: shape (2,)
    q = np.array([1.0, 1.0], dtype=np.float32)
    dist_single = pairwise_squared_l2_ref(q, X)
    assert dist_single.shape == (4,)
    assert dist_single.dtype == np.float32
    np.testing.assert_allclose(
        dist_single,
        np.array(expected_values, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_t1_3_single_vs_batched_queries():
    """T1.3: Consistency and shape flexibility between 1D single query and 2D batched query."""
    X = np.array(
        [
            [0.5, -1.2, 3.3],
            [2.0, 0.0, -1.0],
            [-4.5, 2.1, 0.8],
        ],
        dtype=np.float32,
    )
    q = np.array([1.0, -1.0, 2.0], dtype=np.float32)
    Q = q.reshape(1, -1)

    dist_1d = pairwise_squared_l2_ref(q, X)
    dist_2d = pairwise_squared_l2_ref(Q, X)

    assert dist_1d.shape == (3,)
    assert dist_2d.shape == (1, 3)
    assert dist_1d.dtype == np.float32
    assert dist_2d.dtype == np.float32
    np.testing.assert_array_equal(dist_1d, dist_2d[0])


def test_multiple_queries_and_database_vectors():
    """Verify pairwise distance matrix with multiple queries (Q=2) and database vectors (N=3)."""
    Q = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [-3.0, -4.0],
        ],
        dtype=np.float32,
    )
    # Expected:
    # q0 -> x0: 0^2 + 0^2 = 0
    # q0 -> x1: 1^2 + 2^2 = 5
    # q0 -> x2: (-3)^2 + (-4)^2 = 25
    # q1 -> x0: 3^2 + 4^2 = 25
    # q1 -> x1: (3-1)^2 + (4-2)^2 = 4 + 4 = 8
    # q1 -> x2: (3 - (-3))^2 + (4 - (-4))^2 = 6^2 + 8^2 = 36 + 64 = 100
    expected = np.array(
        [
            [0.0, 5.0, 25.0],
            [25.0, 8.0, 100.0],
        ],
        dtype=np.float32,
    )

    actual = pairwise_squared_l2_ref(Q, X)
    assert actual.shape == (2, 3)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_t1_4_dimension_mismatch():
    """T1.4: Dimension mismatch between queries and database must raise ValueError."""
    q_4d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    Q_4d = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    X_3d = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    with pytest.raises(ValueError, match=r"[dD]imension|[sS]hape"):
        pairwise_squared_l2_ref(q_4d, X_3d)

    with pytest.raises(ValueError, match=r"[dD]imension|[sS]hape"):
        pairwise_squared_l2_ref(Q_4d, X_3d)


def test_t1_5_empty_inputs():
    """T1.5: Boundary robustness on empty query or database arrays."""
    X_empty = np.empty((0, 3), dtype=np.float32)
    X_non_empty = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    Q_empty = np.empty((0, 3), dtype=np.float32)
    Q_non_empty = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    q_single = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # 1. N = 0, Q >= 1 -> (Q, 0)
    out1 = pairwise_squared_l2_ref(Q_non_empty, X_empty)
    assert out1.shape == (1, 0)
    assert out1.dtype == np.float32

    # 2. Q = 0, N >= 1 -> (0, N)
    out2 = pairwise_squared_l2_ref(Q_empty, X_non_empty)
    assert out2.shape == (0, 2)
    assert out2.dtype == np.float32

    # 3. Single query (d,), N = 0 -> (0,)
    out3 = pairwise_squared_l2_ref(q_single, X_empty)
    assert out3.shape == (0,)
    assert out3.dtype == np.float32

    # 4. Q = 0, N = 0 -> (0, 0)
    out4 = pairwise_squared_l2_ref(Q_empty, X_empty)
    assert out4.shape == (0, 0)
    assert out4.dtype == np.float32


def test_t1_6_non_finite_values():
    """T1.6: Non-finite values (NaN, Inf, -Inf) must raise ValueError."""
    X_valid = np.array([[1.0, 2.0]], dtype=np.float32)
    q_valid = np.array([1.0, 2.0], dtype=np.float32)

    # Query with NaN
    q_nan = np.array([np.nan, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        pairwise_squared_l2_ref(q_nan, X_valid)

    # Query with Inf
    q_inf = np.array([np.inf, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        pairwise_squared_l2_ref(q_inf, X_valid)

    # Database with NaN
    X_nan = np.array([[1.0, np.nan]], dtype=np.float32)
    with pytest.raises(ValueError):
        pairwise_squared_l2_ref(q_valid, X_nan)

    # Database with -Inf
    X_inf = np.array([[1.0, -np.inf]], dtype=np.float32)
    with pytest.raises(ValueError):
        pairwise_squared_l2_ref(q_valid, X_inf)


def test_t1_7_strict_float32_input_validation():
    """T1.7: Strict float32 input enforcement. Non-float32 arrays must raise TypeError or ValueError."""
    X_f32 = np.array([[1.0, 2.0]], dtype=np.float32)
    q_f32 = np.array([1.0, 2.0], dtype=np.float32)

    # Query float64
    q_f64 = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises((TypeError, ValueError)):
        pairwise_squared_l2_ref(q_f64, X_f32)

    # Database float64
    X_f64 = np.array([[1.0, 2.0]], dtype=np.float64)
    with pytest.raises((TypeError, ValueError)):
        pairwise_squared_l2_ref(q_f32, X_f64)

    # Query int32
    q_int = np.array([1, 2], dtype=np.int32)
    with pytest.raises((TypeError, ValueError)):
        pairwise_squared_l2_ref(q_int, X_f32)

    # Database int64
    X_int = np.array([[1, 2]], dtype=np.int64)
    with pytest.raises((TypeError, ValueError)):
        pairwise_squared_l2_ref(q_f32, X_int)


def test_invalid_input_structures():
    """Verify rejection of malformed or non-array inputs (e.g. 3D arrays or scalars)."""
    X_valid = np.array([[1.0, 2.0]], dtype=np.float32)
    q_valid = np.array([1.0, 2.0], dtype=np.float32)

    # 3D database
    X_3d = np.ones((2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        pairwise_squared_l2_ref(q_valid, X_3d)

    # 1D database (database must be 2D of shape (N, d))
    X_1d = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        pairwise_squared_l2_ref(q_valid, X_1d)

    # 3D query
    Q_3d = np.ones((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        pairwise_squared_l2_ref(Q_3d, X_valid)


def test_negative_coordinates():
    """Verify distance computation with negative coordinate components."""
    q = np.array([-5.0, 10.0], dtype=np.float32)
    X = np.array([[5.0, -10.0]], dtype=np.float32)
    # (-5 - 5)^2 + (10 - (-10))^2 = (-10)^2 + 20^2 = 100 + 400 = 500
    dist = pairwise_squared_l2_ref(q, X)
    assert dist.shape == (1,)
    assert dist.dtype == np.float32
    assert dist[0] == 500.0


def test_single_vs_batched_random_vectors():
    """Verify bit-for-bit equivalence between 1D single query and 2D batch query across random vectors."""
    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((20, 16)).astype(np.float32)
    q = rng.standard_normal((16,)).astype(np.float32)

    dist_1d = pairwise_squared_l2_ref(q, X)
    dist_2d = pairwise_squared_l2_ref(q.reshape(1, 16), X)

    assert dist_1d.shape == (20,)
    assert dist_2d.shape == (1, 20)
    np.testing.assert_array_equal(dist_1d, dist_2d[0])


def test_chunked_kernel_parameter_validation():
    """Verify max_buffer_bytes validation on non-positive and non-integer inputs."""
    q = np.ones((4,), dtype=np.float32)
    X = np.ones((5, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="max_buffer_bytes must be positive"):
        pairwise_squared_l2_ref(q, X, max_buffer_bytes=0)

    with pytest.raises(ValueError, match="max_buffer_bytes must be positive"):
        pairwise_squared_l2_ref(q, X, max_buffer_bytes=-1024)

    with pytest.raises(TypeError, match="max_buffer_bytes must be an integer"):
        pairwise_squared_l2_ref(q, X, max_buffer_bytes=1024.5)  # type: ignore

    with pytest.raises(TypeError, match="max_buffer_bytes must be an integer"):
        pairwise_squared_l2_ref(q, X, max_buffer_bytes="1024")  # type: ignore


def test_chunked_kernel_equivalence_to_unchunked_oracle():
    """Verify bit-for-bit equivalence between chunked kernel and unchunked broadcast tensor oracle."""
    def unchunked_oracle(queries: np.ndarray, database: np.ndarray) -> np.ndarray:
        is_single = queries.ndim == 1
        q_2d = queries[np.newaxis, :] if is_single else queries
        diff = q_2d[:, np.newaxis, :].astype(np.float64) - database[np.newaxis, :, :].astype(np.float64)
        out = np.sum(diff * diff, axis=-1, dtype=np.float64).astype(np.float32)
        return out[0] if is_single else out

    rng = np.random.default_rng(101)

    # Test across multiple Q, N, d shapes and buffer thresholds
    test_cases = [
        (1, 15, 3),      # single query
        (5, 20, 8),      # small batch
        (13, 37, 7),     # prime dimensions
        (3, 100, 1),     # 1D feature
        (8, 25, 128),    # higher dimension
    ]

    # Deliberately restrictive buffer sizes to force diverse chunking paths
    buffer_sizes = [64, 256, 1024, 8192, 1024 * 1024]

    for Q, N, d in test_cases:
        queries = rng.standard_normal((Q, d)).astype(np.float32)
        database = rng.standard_normal((N, d)).astype(np.float32)
        oracle_batch = unchunked_oracle(queries, database)
        oracle_single = unchunked_oracle(queries[0], database)

        for buf in buffer_sizes:
            res_batch = pairwise_squared_l2_ref(queries, database, max_buffer_bytes=buf)
            res_single = pairwise_squared_l2_ref(queries[0], database, max_buffer_bytes=buf)

            np.testing.assert_array_equal(res_batch, oracle_batch)
            np.testing.assert_array_equal(res_single, oracle_single)


def test_chunked_kernel_boundary_cases():
    """Verify chunked kernel behavior on empty inputs, d=1, large d, and non-multiple chunks."""
    rng = np.random.default_rng(202)

    # Empty inputs with tight buffer
    empty_q = np.empty((0, 10), dtype=np.float32)
    db = rng.standard_normal((15, 10)).astype(np.float32)
    assert pairwise_squared_l2_ref(empty_q, db, max_buffer_bytes=128).shape == (0, 15)

    empty_db = np.empty((0, 10), dtype=np.float32)
    q = rng.standard_normal((8, 10)).astype(np.float32)
    assert pairwise_squared_l2_ref(q, empty_db, max_buffer_bytes=128).shape == (8, 0)
    assert pairwise_squared_l2_ref(q[0], empty_db, max_buffer_bytes=128).shape == (0,)

    # Large feature dimension (d = 256)
    q_large_d = rng.standard_normal((4, 256)).astype(np.float32)
    db_large_d = rng.standard_normal((10, 256)).astype(np.float32)
    res_large = pairwise_squared_l2_ref(q_large_d, db_large_d, max_buffer_bytes=2048)
    assert res_large.shape == (4, 10)
    assert res_large.dtype == np.float32
    # Verify non-negative squared distances
    assert np.all(res_large >= 0.0)


def test_chunked_kernel_memory_bound_structural():
    """Verify chunked distance kernel on a workload where unchunked (Q, N, d) float64 tensor exceeds 16MB."""
    rng = np.random.default_rng(303)
    Q, N, d = 50, 400, 128
    # Unchunked float64 (Q, N, d) tensor = 50 * 400 * 128 * 8 = 20,480,000 bytes (~20.5 MB)
    queries = rng.standard_normal((Q, d)).astype(np.float32)
    database = rng.standard_normal((N, d)).astype(np.float32)

    # Restrict buffer to 512 KiB (524,288 bytes)
    bounded_res = pairwise_squared_l2_ref(queries, database, max_buffer_bytes=512 * 1024)
    assert bounded_res.shape == (Q, N)
    assert bounded_res.dtype == np.float32

    # Check a deterministic slice against direct float64 scalar computation
    sample_diff = queries[0].astype(np.float64) - database[0].astype(np.float64)
    expected_dist_0_0 = np.float32(np.sum(sample_diff * sample_diff))
    np.testing.assert_allclose(bounded_res[0, 0], expected_dist_0_0)
