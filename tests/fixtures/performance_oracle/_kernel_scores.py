"""Accurate original-kernel score evaluation for cancellation-prone models.

This changes matrix-vector evaluation, never the kernel, coefficients or
intercept. Row-wise high/low products plus fsum use O(n_training) temporary
storage; no dense product matrix or extended-precision dtype is required.
"""
import numpy as np
from scipy.sparse import issparse

from ._compensated_residual import _gradual_underflow, _products, _fsum


def _inputs(K, alpha):
    raw_alpha = np.asarray(alpha)
    if (raw_alpha.ndim != 1 or not raw_alpha.size or np.iscomplexobj(raw_alpha)
            or raw_alpha.dtype.kind not in 'biuf'):
        raise ValueError('alpha must be a nonempty real numeric vector.')
    alpha = np.asarray(raw_alpha, dtype=float)
    if not np.isfinite(alpha).all():
        raise FloatingPointError('Nonfinite kernel coefficients.')
    sparse = issparse(K)
    if sparse:
        if K.format != 'csr':
            raise ValueError('Sparse query kernels must be CSR.')
        values = K.data
    else:
        K = np.asarray(K)
        values = K
    if (getattr(K, 'ndim', None) != 2 or K.shape[1] != len(alpha)
            or np.iscomplexobj(values) or values.dtype.kind not in 'biuf'):
        raise ValueError('K must be a real numeric query-by-training matrix matching alpha.')
    _gradual_underflow()
    return K, alpha, sparse


def _expanded_row(K, alpha, sparse, i):
    if sparse:
        start, stop = K.indptr[i:i+2]
        row = K.data[start:stop]
        coefficients = alpha[K.indices[start:stop]]
    else:
        row, coefficients = K[i], alpha
    high, low = _products(row, coefficients)
    return _fsum([*high, *low])


def compensated_kernel_matvec(K, alpha):
    """Return dense/CSR query-by-training K times one finite real vector.

    CSR duplicates are accumulated as their stored contributions, without a
    preliminary rounded sum_duplicates operation. Other sparse formats should
    be converted to CSR by their owner before calling this row-wise helper.
    Nonfinite input/products or an overflowing sum fail explicitly.
    """
    K, alpha, sparse = _inputs(K, alpha)
    result = np.empty(K.shape[0], dtype=float)
    for i in range(K.shape[0]):
        result[i] = _expanded_row(K, alpha, sparse, i)
    return result


def adaptive_kernel_matvec(K, alpha):
    """Use ordinary products only where a conservative row error estimate passes.

    The estimate uses gamma_n times an upper estimate of sum(abs(K_ij*alpha_j)),
    including roundoff in the positive bound calculation and gradual underflow.
    It assumes normal IEEE binary64 arithmetic and is numerical screening, not
    an interval certificate for arbitrary BLAS implementations. The per-row
    threshold is 5e-7*max(1, abs(ordinary_score)); uncertain rows are evaluated
    using expanded products. An overflowing bound therefore requests accurate
    evaluation rather than rejecting a potentially finite cancellation result.

    Absolute-value work is limited to at most 128 query rows at a time. Sparse
    duplicates count as separate terms in both the bound and expanded product.
    """
    K, alpha, sparse = _inputs(K, alpha)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        scores = np.asarray(K @ alpha, dtype=float).reshape(-1)
    magnitude = np.abs(alpha)
    eps = np.finfo(float).eps
    eta = float(np.nextafter(0., 1.))
    for start in range(0, K.shape[0], 128):
        stop = min(start+128, K.shape[0])
        if sparse:
            block = K[start:stop].astype(float, copy=True)
            if not np.isfinite(block.data).all():
                raise FloatingPointError('Nonfinite query kernel values.')
            block.data = np.abs(block.data)
            terms = np.diff(block.indptr).astype(float)
        else:
            block = np.asarray(K[start:stop], dtype=float)
            if not np.isfinite(block).all():
                raise FloatingPointError('Nonfinite query kernel values.')
            block = np.abs(block)
            terms = np.full(stop-start, len(alpha), dtype=float)
        with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
            absolute_sum = np.asarray(block @ magnitude).reshape(-1)
            # Enlarge n*eps, gamma_n, the positive sum and every bound operation.
            # The same gamma bound covers the ordinary and positive dot products.
            neps = np.nextafter(terms*eps, np.inf)
            denominator = np.nextafter(1.-neps, 0.)
            gamma = np.nextafter(neps/denominator, np.inf)
            underflow = np.nextafter((4.*terms+4.)*eta, np.inf)
            sum_denominator = np.nextafter(1.-gamma, 0.)
            upper_sum = np.nextafter(np.nextafter(absolute_sum+underflow, np.inf)/sum_denominator, np.inf)
            bound = np.nextafter(np.nextafter(gamma*upper_sum, np.inf)+underflow, np.inf)
            threshold = np.nextafter(5e-7*np.maximum(1., np.abs(scores[start:stop])), 0.)
        reliable = (np.isfinite(scores[start:stop]) & np.isfinite(bound)
                    & (denominator > 0) & (sum_denominator > 0)
                    & (absolute_sum >= 0) & (bound <= threshold))
        for local in np.flatnonzero(~reliable):
            index = start+int(local)
            scores[index] = _expanded_row(K, alpha, sparse, index)
    return scores
