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
