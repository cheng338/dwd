"""Consistent finite-precision construction of a named RBF kernel.

Used only after an internally generated RBF self-kernel fails the existing
symmetry tolerance. This is not a repair of a caller-supplied kernel matrix.
"""

import numpy as np
from sklearn.utils.extmath import row_norms, safe_sparse_dot


def self_kernel_asymmetry(K):
    """Match the solver's symmetry threshold with bounded temporary storage."""
    scale = 0.0
    asymmetry = 0.0
    for start in range(0, len(K), 128):
        block = K[start:start + 128]
        if not np.isfinite(block).all():
            # Leave nonfinite-input rejection to the unchanged solver validator.
            return 0.0, np.inf
        scale = max(scale, float(np.max(np.abs(block))))
        asymmetry = max(asymmetry, float(np.max(np.abs(
            block - K[:, start:start + 128].T))))
    return asymmetry, 100 * np.finfo(float).eps * max(1.0, scale)


def norm_sum_rbf(X, Y, *, gamma=None, self_kernel=False):
    """Evaluate exp(-gamma * ||x-y||^2), summing the two norms first.

    X/Y are the estimator's validated float64 dense or CSR features. Adding
    the norms together before the dot-product term removes the demonstrated
    reversed-addition error. It does not eliminate all cancellation for nearly
    identical large vectors, or promise bitwise BLAS/batch invariance.

    One evaluated self-kernel triangle is retained and mirrored, using the
    known analytic RBF. Cross-kernels use the same distance formula. No caller
    matrix is averaged or modified, and sparse features are not densified.
    """
    if self_kernel and Y is not X:
        raise ValueError('self_kernel requires identical feature objects.')
    gamma = 1.0 / X.shape[1] if gamma is None else gamma
    x_norm = row_norms(X, squared=True)
    y_norm = x_norm if self_kernel else row_norms(Y, squared=True)
    K = safe_sparse_dot(X, Y.T, dense_output=True)
    K *= -2.0
    for start in range(0, len(x_norm), 128):
        stop = min(start + 128, len(x_norm))
        K[start:stop] += x_norm[start:stop, None] + y_norm[None, :]
    np.maximum(K, 0.0, out=K)
    K *= -gamma
    np.exp(K, out=K)
    if self_kernel:
        # Copy each upper pair once. Avoid an additional n-by-n temporary.
        for start in range(0, len(x_norm), 128):
            stop = min(start + 128, len(x_norm))
            block = K[start:stop, start:stop]
            lower = np.tril_indices(stop - start, -1)
            block[lower] = block.T[lower]
            K[stop:, start:stop] = K[start:stop, stop:].T
        np.fill_diagonal(K, 1.0)
    return K
