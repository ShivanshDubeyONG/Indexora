"""Tests for Product Quantization (PQ) in Indexora."""
import numpy as np
import pytest

from indexora.kernels import pairwise_squared_l2_ref
from indexora.pq import ProductQuantizer, _resolve_code_dtype, _validate_pq_inputs


def test_pq_input_validation() -> None:
    """Test validation of d, M, and k_sub structural parameters."""
    # Valid
    d, M, k_sub = _validate_pq_inputs(128, 8, 256)
    assert d == 128 and M == 8 and k_sub == 256

    # d not divisible by M
    with pytest.raises(ValueError, match="must be divisible"):
        _validate_pq_inputs(128, 7, 256)

    # d <= 0
    with pytest.raises(ValueError, match="d must be a positive integer"):
        _validate_pq_inputs(0, 4, 16)

    # M <= 0
    with pytest.raises(ValueError, match="M must be a positive integer"):
        _validate_pq_inputs(16, 0, 16)

    # k_sub <= 0 or > 65536
    with pytest.raises(ValueError, match="k_sub must be a positive integer"):
        _validate_pq_inputs(16, 4, 0)
    with pytest.raises(ValueError, match="k_sub cannot exceed 65,536"):
        _validate_pq_inputs(16, 4, 70000)


def test_resolve_code_dtype() -> None:
    """Test resolution of minimal unsigned integer dtype based on k_sub."""
    assert _resolve_code_dtype(16) == np.dtype(np.uint8)
    assert _resolve_code_dtype(256) == np.dtype(np.uint8)
    assert _resolve_code_dtype(257) == np.dtype(np.uint16)
    assert _resolve_code_dtype(65536) == np.dtype(np.uint16)
    with pytest.raises(ValueError, match="exceeds supported maximum"):
        _resolve_code_dtype(65537)


def test_pq_initialization_and_pre_supplied_codebooks() -> None:
    """Test initialization with and without pre-supplied codebooks."""
    pq = ProductQuantizer(d=32, M=4, k_sub=16)
    assert pq.d == 32
    assert pq.M == 4
    assert pq.d_sub == 8
    assert pq.k_sub == 16
    assert not pq.is_trained
    assert pq.code_dtype == np.uint8
    assert pq.code_size_bytes == 4

    with pytest.raises(RuntimeError, match="has not been trained yet"):
        _ = pq.codebooks

    # Pre-supplied codebooks
    cbs = np.zeros((4, 16, 8), dtype=np.float32)
    pq_explicit = ProductQuantizer(d=32, M=4, k_sub=16, codebooks=cbs)
    assert pq_explicit.is_trained
    assert pq_explicit.codebooks.shape == (4, 16, 8)
    assert pq_explicit.memory_bytes == 4 * 16 * 8 * 4

    # Invalid codebooks shape
    with pytest.raises(ValueError, match="codebooks shape mismatch"):
        ProductQuantizer(d=32, M=4, k_sub=16, codebooks=np.zeros((4, 15, 8), dtype=np.float32))


def test_pq_training_and_encoding_shapes() -> None:
    """Test PQ training and encoding dimensions and dtypes."""
    rng = np.random.default_rng(42)
    N, d, M, k_sub = 300, 32, 4, 16
    X = rng.normal(size=(N, d)).astype(np.float32)

    pq = ProductQuantizer(d=d, M=M, k_sub=k_sub)
    pq.fit(X, max_iter=20, seed=42)

    assert pq.is_trained
    assert pq.codebooks.shape == (M, k_sub, d // M)

    codes = pq.encode(X)
    assert codes.shape == (N, M)
    assert codes.dtype == np.uint8
    assert np.all(codes >= 0) and np.all(codes < k_sub)

    # 1D single vector encoding
    single_code = pq.encode(X[0])
    assert single_code.shape == (M,)
    assert single_code.dtype == np.uint8
    np.testing.assert_array_equal(single_code, codes[0])


def test_pq_decode_and_reconstruction_error() -> None:
    """Test that decoding quantized codes reconstructs vectors with bounded error."""
    rng = np.random.default_rng(42)
    N, d, M, k_sub = 200, 16, 4, 8
    X = rng.normal(size=(N, d)).astype(np.float32)

    pq = ProductQuantizer(d=d, M=M, k_sub=k_sub)
    pq.fit(X, max_iter=25, seed=42)

    codes = pq.encode(X)
    recon = pq.decode(codes)

    assert recon.shape == (N, d)
    assert recon.dtype == np.float32

    # Reconstruction MSE
    mse = pq.reconstruction_error(X)
    manual_mse = float(np.mean((X - recon) ** 2))
    assert pytest.approx(mse, rel=1e-5) == manual_mse
    assert mse < 1.5  # Bounded error on standard normal data

    # 1D single code decoding
    single_recon = pq.decode(codes[0])
    assert single_recon.shape == (d,)
    np.testing.assert_array_equal(single_recon, recon[0])


def test_pq_adc_distance_table_mathematical_identity() -> None:
    """Verify that ADC lookup table distances equal exact distances to reconstructed vectors."""
    rng = np.random.default_rng(123)
    N, d, M, k_sub, Q = 50, 32, 4, 16, 10
    X = rng.normal(size=(N, d)).astype(np.float32)
    queries = rng.normal(size=(Q, d)).astype(np.float32)

    pq = ProductQuantizer(d=d, M=M, k_sub=k_sub)
    pq.fit(X, max_iter=20, seed=42)

    codes = pq.encode(X)
    recon = pq.decode(codes)

    # 1. Single query ADC
    for q in queries:
        lut = pq.compute_distance_table(q)
        assert lut.shape == (M, k_sub)

        adc_dists = pq.compute_distances_from_table(lut, codes)
        assert adc_dists.shape == (N,)
        assert adc_dists.dtype == np.float32

        # Exact distance to reconstructed vectors
        exact_recon_dists = pairwise_squared_l2_ref(q, recon)

        # ADC identity: distance from query to reconstructed vectors must match ADC table lookup
        np.testing.assert_allclose(
            adc_dists,
            exact_recon_dists,
            rtol=1e-4,
            atol=1e-4,
            err_msg="ADC distances do not match exact squared L2 distances to reconstructed vectors!",
        )

    # 2. Batched queries ADC
    batch_lut = pq.compute_distance_table(queries)
    assert batch_lut.shape == (Q, M, k_sub)

    batch_adc_dists = pq.compute_distances_from_table(batch_lut, codes)
    assert batch_adc_dists.shape == (Q, N)

    exact_batch_recon_dists = pairwise_squared_l2_ref(queries, recon)
    np.testing.assert_allclose(
        batch_adc_dists,
        exact_batch_recon_dists,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Batched ADC distances do not match exact squared L2 distances to reconstructed vectors!",
    )


def test_pq_edge_cases_empty_and_validation() -> None:
    """Test PQ edge cases: empty candidate lists, non-float32 vectors, out of bounds codes."""
    pq = ProductQuantizer(d=8, M=2, k_sub=4)
    cbs = np.zeros((2, 4, 4), dtype=np.float32)
    pq_init = ProductQuantizer(d=8, M=2, k_sub=4, codebooks=cbs)

    # Empty candidate codes
    empty_codes = np.empty((0, 2), dtype=np.uint8)
    lut = pq_init.compute_distance_table(np.ones(8, dtype=np.float32))
    dists = pq_init.compute_distances_from_table(lut, empty_codes)
    assert dists.shape == (0,)

    # Out of bounds code
    bad_codes = np.array([[5, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="outside valid centroid range"):
        pq_init.decode(bad_codes)
