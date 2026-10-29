"""Product Quantization (PQ) for vector compression and approximate search in Indexora.

This module provides the Phase 9 Product Quantization foundation:
- `ProductQuantizer`: Subspace decomposition, codebook training via Lloyd's k-means,
  vector encoding to compact uint8/uint16 codes, vector reconstruction, and
  Asymmetric Distance Computation (ADC) lookup table evaluation.
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np

from indexora.clustering import kmeans
from indexora.kernels import pairwise_squared_l2_blas, pairwise_squared_l2_ref


def _resolve_code_dtype(k_sub: int) -> np.dtype:
    """Determine the minimal unsigned integer dtype required to store subspace centroid indices."""
    if k_sub <= 256:
        return np.dtype(np.uint8)
    elif k_sub <= 65536:
        return np.dtype(np.uint16)
    else:
        raise ValueError(
            f"k_sub={k_sub} exceeds supported maximum of 65,536 centroids per subspace."
        )


def _validate_pq_inputs(
    d: int,
    M: int,
    k_sub: int,
) -> tuple[int, int, int]:
    """Validate Product Quantization structural hyperparameters."""
    if not isinstance(d, (int, np.integer)) or d <= 0:
        raise ValueError(f"Vector dimension d must be a positive integer, got {d}")
    if not isinstance(M, (int, np.integer)) or M <= 0:
        raise ValueError(f"Subspace count M must be a positive integer, got {M}")
    if d % M != 0:
        raise ValueError(
            f"Vector dimension d={d} must be divisible by subspace count M={M} (d % M == {d % M})."
        )
    if not isinstance(k_sub, (int, np.integer)) or k_sub <= 0:
        raise ValueError(f"Centroids per subspace k_sub must be a positive integer, got {k_sub}")
    if k_sub > 65536:
        raise ValueError(f"k_sub cannot exceed 65,536, got {k_sub}")

    d_sub = d // M
    return int(d), int(M), int(k_sub)


def _validate_vectors(X: np.ndarray, expected_d: int, name: str = "vectors") -> tuple[int, bool]:
    """Validate input vectors array for float32 dtype, shape, and contiguity."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(X).__name__}")
    if X.dtype != np.float32:
        raise TypeError(
            f"{name} must have strict dtype float32, got {X.dtype}. Silent casting is not permitted."
        )
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} must contain only finite numbers (no NaN or Inf).")

    is_1d = X.ndim == 1
    if is_1d:
        if X.shape[0] != expected_d:
            raise ValueError(
                f"{name} dimension mismatch: expected {expected_d}, got {X.shape[0]}"
            )
        if not X.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous in memory.")
        return 1, True

    if X.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D array, got ndim={X.ndim}")
    if X.shape[1] != expected_d:
        raise ValueError(
            f"{name} feature dimension mismatch: expected {expected_d}, got {X.shape[1]}"
        )
    if not X.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous in memory.")
    return X.shape[0], False


class ProductQuantizer:
    """Product Quantizer for dense floating-point vector compression.

    A D-dimensional vector space is decomposed into the Cartesian product of M
    orthogonal subspaces, each of dimension d_sub = D // M. Each subspace is
    quantized into k_sub centroids trained via k-means. Vectors are encoded as
    M-byte discrete tuples of centroid indices.

    Parameters
    ----------
    d : int
        Original vector feature dimensionality (must be divisible by M).
    M : int
        Number of orthogonal subspaces.
    k_sub : int, default=256
        Number of centroids per subspace (1 <= k_sub <= 65536).
    codebooks : np.ndarray | None, default=None
        Pre-trained subspace codebooks of shape `(M, k_sub, d_sub)` with `dtype=np.float32`.
        If None, codebooks must be trained via `.fit(X)`.
    """

    def __init__(
        self,
        d: int,
        M: int,
        k_sub: int = 256,
        codebooks: np.ndarray | None = None,
    ) -> None:
        self._d, self._M, self._k_sub = _validate_pq_inputs(d, M, k_sub)
        self._d_sub = self._d // self._M
        self._code_dtype = _resolve_code_dtype(self._k_sub)

        if codebooks is not None:
            if not isinstance(codebooks, np.ndarray):
                raise TypeError(f"codebooks must be a numpy ndarray, got {type(codebooks).__name__}")
            if codebooks.dtype != np.float32:
                raise TypeError(
                    f"codebooks must have strict dtype float32, got {codebooks.dtype}."
                )
            expected_shape = (self._M, self._k_sub, self._d_sub)
            if codebooks.shape != expected_shape:
                raise ValueError(
                    f"codebooks shape mismatch: expected {expected_shape}, got {codebooks.shape}"
                )
            if not codebooks.flags.c_contiguous:
                raise ValueError("codebooks must be C-contiguous in memory.")
            if not np.all(np.isfinite(codebooks)):
                raise ValueError("codebooks must contain only finite numbers.")
            self._codebooks = codebooks.copy()
            self._is_trained = True
        else:
            self._codebooks = np.empty((self._M, self._k_sub, self._d_sub), dtype=np.float32)
            self._is_trained = False

    @property
    def d(self) -> int:
        """Original vector dimensionality."""
        return self._d

    @property
    def M(self) -> int:
        """Number of equal-sized subspaces."""
        return self._M

    @property
    def d_sub(self) -> int:
        """Dimensionality of each subspace (`d // M`)."""
        return self._d_sub

    @property
    def k_sub(self) -> int:
        """Number of centroids per subspace."""
        return self._k_sub

    @property
    def is_trained(self) -> bool:
        """Whether codebooks have been trained or supplied."""
        return self._is_trained

    @property
    def code_dtype(self) -> np.dtype:
        """Minimal integer dtype for code indices (`uint8` or `uint16`)."""
        return self._code_dtype

    @property
    def code_size_bytes(self) -> int:
        """Bytes required to store one encoded vector (`M * code_dtype.itemsize`)."""
        return int(self._M * self._code_dtype.itemsize)

    @property
    def codebooks(self) -> np.ndarray:
        """Trained codebook tensor of shape `(M, k_sub, d_sub)`."""
        if not self._is_trained:
            raise RuntimeError("ProductQuantizer has not been trained yet. Call .fit() first.")
        return self._codebooks.copy()

    @property
    def memory_bytes(self) -> int:
        """Storage bytes of the persistent codebooks."""
        return int(self._codebooks.nbytes) if self._is_trained else 0

    def fit(
        self,
        X: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: int | None = 42,
    ) -> ProductQuantizer:
        """Train per-subspace codebooks using k-means clustering.

        Parameters
        ----------
        X : np.ndarray
            Training vectors of shape `(N, d)` with strict `dtype=np.float32`.
        max_iter : int, default=100
            Maximum k-means iterations per subspace.
        tol : float, default=1e-4
            Convergence tolerance for centroid updates.
        seed : int | None, default=42
            Random seed for k-means++ centroid initialization.

        Returns
        -------
        ProductQuantizer
            The trained quantizer instance.
        """
        N, is_1d = _validate_vectors(X, self._d, name="Training data X")
        if is_1d or N < self._k_sub:
            raise ValueError(
                f"Training data requires at least k_sub={self._k_sub} vectors, got N={N}"
            )

        rng = np.random.default_rng(seed)

        # Train k-means independently on each subspace
        for m in range(self._M):
            start_col = m * self._d_sub
            end_col = start_col + self._d_sub
            # Slice subspace block (guaranteed C-contiguous)
            X_sub = np.ascontiguousarray(X[:, start_col:end_col], dtype=np.float32)

            sub_seed = int(rng.integers(0, 2**31 - 1)) if seed is not None else None
            km_res = kmeans(
                X_sub,
                k=self._k_sub,
                max_iter=max_iter,
                tol=tol,
                seed=sub_seed,
                empty_cluster_policy="reseed_furthest",
                distance_method="blas",
            )
            self._codebooks[m] = km_res.centroids

        self._is_trained = True
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Quantize vectors into discrete subspace centroid indices.

        Parameters
        ----------
        X : np.ndarray
            Vectors of shape `(d,)` or `(N, d)` with strict `dtype=np.float32`.

        Returns
        -------
        np.ndarray
            Quantized codes of shape `(M,)` or `(N, M)` with dtype `code_dtype`.
        """
        if not self._is_trained:
            raise RuntimeError("ProductQuantizer is not trained. Call .fit() first.")

        N, is_1d = _validate_vectors(X, self._d, name="Vectors to encode")
        if is_1d:
            X_2d = X.reshape(1, self._d)
        else:
            X_2d = X

        if N == 0:
            empty_shape = (0, self._M) if not is_1d else (0,)
            return np.empty(empty_shape, dtype=self._code_dtype)

        codes = np.empty((N, self._M), dtype=self._code_dtype)

        for m in range(self._M):
            start_col = m * self._d_sub
            end_col = start_col + self._d_sub
            sub_vecs = np.ascontiguousarray(X_2d[:, start_col:end_col], dtype=np.float32)
            # Compute pairwise distances between subvectors and subspace centroids
            # Shape: (N, k_sub)
            sub_dists = pairwise_squared_l2_blas(sub_vecs, self._codebooks[m])
            # Assign nearest centroid index (ties broken deterministically by lowest index)
            codes[:, m] = np.argmin(sub_dists, axis=1).astype(self._code_dtype)

        if is_1d:
            return codes[0]
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct approximate vectors from discrete subspace centroid indices.

        Parameters
        ----------
        codes : np.ndarray
            Quantized codes of shape `(M,)` or `(N, M)` with unsigned integer dtype.

        Returns
        -------
        np.ndarray
            Reconstructed vectors of shape `(d,)` or `(N, d)` with `dtype=np.float32`.
        """
        if not self._is_trained:
            raise RuntimeError("ProductQuantizer is not trained. Call .fit() first.")

        if not isinstance(codes, np.ndarray):
            raise TypeError(f"codes must be a numpy ndarray, got {type(codes).__name__}")
        if not np.issubdtype(codes.dtype, np.integer):
            raise TypeError(f"codes must have integer dtype, got {codes.dtype}")

        is_1d = codes.ndim == 1
        if is_1d:
            if codes.shape[0] != self._M:
                raise ValueError(f"codes shape mismatch: expected ({self._M},), got {codes.shape}")
            codes_2d = codes.reshape(1, self._M)
        else:
            if codes.ndim != 2:
                raise ValueError(f"codes must be 1D or 2D array, got ndim={codes.ndim}")
            if codes.shape[1] != self._M:
                raise ValueError(
                    f"codes feature dimension mismatch: expected {self._M}, got {codes.shape[1]}"
                )
            codes_2d = codes

        N = codes_2d.shape[0]
        if N == 0:
            empty_shape = (0, self._d) if not is_1d else (0,)
            return np.empty(empty_shape, dtype=np.float32)

        # Validate code bounds
        if np.any(codes_2d < 0) or np.any(codes_2d >= self._k_sub):
            raise ValueError(
                f"codes contain values outside valid centroid range [0, {self._k_sub - 1}]"
            )

        recon = np.empty((N, self._d), dtype=np.float32)
        for m in range(self._M):
            start_col = m * self._d_sub
            end_col = start_col + self._d_sub
            sub_indices = codes_2d[:, m].astype(np.int64)
            recon[:, start_col:end_col] = self._codebooks[m, sub_indices]

        if is_1d:
            return recon[0]
        return recon

    def compute_distance_table(self, query: np.ndarray) -> np.ndarray:
        """Compute query-to-centroid Asymmetric Distance Computation (ADC) lookup table.

        Parameters
        ----------
        query : np.ndarray
            Query vector of shape `(d,)` or batched queries of shape `(Q, d)` with
            strict `dtype=np.float32`.

        Returns
        -------
        np.ndarray
            Squared-L2 distance lookup table:
            - Single query `(d,)`: shape `(M, k_sub)` float32.
            - Batched queries `(Q, d)`: shape `(Q, M, k_sub)` float32.
        """
        if not self._is_trained:
            raise RuntimeError("ProductQuantizer is not trained. Call .fit() first.")

        num_q, is_1d = _validate_vectors(query, self._d, name="Query")
        if is_1d:
            q_2d = query.reshape(1, self._d)
        else:
            q_2d = query

        if num_q == 0:
            empty_shape = (0, self._M, self._k_sub) if not is_1d else (self._M, self._k_sub)
            return np.empty(empty_shape, dtype=np.float32)

        # Compute distance tables for all subspaces
        # Shape: (Q, M, k_sub)
        lut = np.empty((num_q, self._M, self._k_sub), dtype=np.float32)
        for m in range(self._M):
            start_col = m * self._d_sub
            end_col = start_col + self._d_sub
            q_sub = np.ascontiguousarray(q_2d[:, start_col:end_col], dtype=np.float32)
            # Shape: (Q, k_sub)
            lut[:, m, :] = pairwise_squared_l2_blas(q_sub, self._codebooks[m])

        if is_1d:
            return lut[0]
        return lut

    def compute_distances_from_table(
        self,
        distance_table: np.ndarray,
        codes: np.ndarray,
    ) -> np.ndarray:
        """Evaluate approximate squared-L2 distances using precomputed ADC distance tables.

        Parameters
        ----------
        distance_table : np.ndarray
            Precomputed distance table from `.compute_distance_table(query)`:
            - Single query: shape `(M, k_sub)` float32.
            - Batched queries: shape `(Q, M, k_sub)` float32.
        codes : np.ndarray
            Candidate codes of shape `(N_cand, M)` with integer dtype.

        Returns
        -------
        np.ndarray
            Approximate squared-L2 distances with strict `dtype=np.float32`:
            - Single query table `(M, k_sub)`: shape `(N_cand,)`.
            - Batched query table `(Q, M, k_sub)`: shape `(Q, N_cand)`.
        """
        if not isinstance(distance_table, np.ndarray):
            raise TypeError(
                f"distance_table must be a numpy ndarray, got {type(distance_table).__name__}"
            )
        if not isinstance(codes, np.ndarray):
            raise TypeError(f"codes must be a numpy ndarray, got {type(codes).__name__}")
        if codes.ndim != 2 or codes.shape[1] != self._M:
            raise ValueError(f"codes must have shape (N_cand, {self._M}), got {codes.shape}")

        N_cand = codes.shape[0]
        if N_cand == 0:
            if distance_table.ndim == 2:
                return np.empty((0,), dtype=np.float32)
            elif distance_table.ndim == 3:
                return np.empty((distance_table.shape[0], 0), dtype=np.float32)
            else:
                raise ValueError(
                    f"distance_table must be 2D (M, k_sub) or 3D (Q, M, k_sub), got ndim={distance_table.ndim}"
                )

        m_indices = np.arange(self._M)
        int_codes = codes.astype(np.int64)

        if distance_table.ndim == 2:
            # Single query table: (M, k_sub)
            if distance_table.shape != (self._M, self._k_sub):
                raise ValueError(
                    f"distance_table shape mismatch: expected ({self._M}, {self._k_sub}), "
                    f"got {distance_table.shape}"
                )
            # Advanced indexing: distance_table[m_indices, codes] produces (N_cand, M)
            per_subspace_dists = distance_table[m_indices, int_codes]
            return np.sum(per_subspace_dists, axis=-1, dtype=np.float32)

        elif distance_table.ndim == 3:
            # Batched query table: (Q, M, k_sub)
            Q = distance_table.shape[0]
            if distance_table.shape[1:] != (self._M, self._k_sub):
                raise ValueError(
                    f"distance_table shape mismatch: expected ({Q}, {self._M}, {self._k_sub}), "
                    f"got {distance_table.shape}"
                )
            # Vectorized multi-query indexing
            # distance_table[:, m_indices, int_codes] -> shape (Q, N_cand, M)
            per_subspace_dists = distance_table[:, m_indices, int_codes]
            return np.sum(per_subspace_dists, axis=-1, dtype=np.float32)

        else:
            raise ValueError(
                f"distance_table must be 2D (M, k_sub) or 3D (Q, M, k_sub), got ndim={distance_table.ndim}"
            )

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Compute mean squared reconstruction error (MSE) across all coordinates.

        MSE = ||X - decode(encode(X))||_F^2 / (N * d)
        """
        N, is_1d = _validate_vectors(X, self._d, name="X")
        if N == 0:
            return 0.0
        X_2d = X.reshape(1, self._d) if is_1d else X
        recon = self.decode(self.encode(X_2d))
        diff = X_2d.astype(np.float64) - recon.astype(np.float64)
        return float(np.sum(diff * diff) / (N * self._d))
