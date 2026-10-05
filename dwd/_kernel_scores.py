"""Accurate original-kernel score evaluation for cancellation-prone models.

This changes matrix-vector evaluation, never the kernel, coefficients or
intercept. Bounded tiles of high/low products retain each row's fsum order.
Temporary storage remains O(n_training); no full extra kernel-sized product
matrix or extended-precision dtype is required.
"""
import numpy as np
from scipy.sparse import issparse

from ._compensated_residual import _gradual_underflow, _products, _fsum, _split_operand


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


def _expanded_row(K, alpha, sparse, i, *, _split_b=None):
    if sparse:
        start, stop = K.indptr[i:i+2]
        row = K.data[start:stop]
        coefficients = alpha[K.indices[start:stop]]
    else:
        row, coefficients = K[i], alpha
    if sparse or _split_b is None:
        high, low = _products(row, coefficients)
    else:
        high, low = _products(row, coefficients, _split_b=_split_b)
    return _fsum(high.tolist() + low.tolist())



# Bound the extra product-array work. The byte allowance conservatively accounts
# for elementwise temporaries; per-row Python summation storage remains O(n).
_SCORE_TILE_BYTES = 1024 * 1024
_SCORE_TILE_MAX_ROWS = 8


def _expanded_dense_rows(K, alpha, indices, split_alpha, result, *, batch_enabled=True):
    """Evaluate selected rows in order, with bounded temporary dense products.

    Elementwise products and each row's fsum term order are unchanged. If a
    tile fails, retry its rows in the original sequence: a later product error
    must not replace an earlier row's summation error. Allocation failure also
    disables batching for the remaining rows in this call.
    """
    # Wider floating dtypes can warn or fail when cast to binary64. Keep
    # their original row-by-row conversion and error ordering on all platforms.
    if K.dtype.kind == 'f' and K.dtype.itemsize > 8:
        batch_enabled = False
    capacity = min(_SCORE_TILE_MAX_ROWS,
                   max(1, _SCORE_TILE_BYTES // (128 * len(alpha))))
    position = 0
    while position < len(indices):
        stop = min(position + (capacity if batch_enabled else 1), len(indices))
        if stop - position < 2:
            index = int(indices[position])
            result[index] = _expanded_row(K, alpha, False, index, _split_b=split_alpha)
            position = stop
            continue
        tile = high = low = coefficients = None
        fallback = False
        try:
            selected = indices[position:stop]
            tile = np.asarray(K[selected], dtype=float)
            coefficients = np.broadcast_to(alpha, tile.shape)
            high, low = _products(tile, coefficients, _split_b=split_alpha)
        except (MemoryError, FloatingPointError):
            fallback = True
        if not fallback:
            try:
                for local, index in enumerate(selected):
                    result[int(index)] = _fsum(high[local].tolist() + low[local].tolist())
            except MemoryError:
                fallback = True
        if fallback:
            # Release large tile buffers before attempting the original path.
            # Leave the exception handler first, so its traceback releases
            # product temporaries as well. Numerical summation errors above
            # propagate directly in the original row order.
            # Retrying successful earlier rows is deterministic and does not
            # expose partial results outside this helper's owner.
            tile = high = low = coefficients = None
            batch_enabled = False
            for index in indices[position:stop]:
                index = int(index)
                result[index] = _expanded_row(K, alpha, False, index, _split_b=split_alpha)
        tile = high = low = coefficients = None
        position = stop
    return batch_enabled


def compensated_kernel_matvec(K, alpha):
    """Return dense/CSR query-by-training K times one finite real vector.

    CSR duplicates are accumulated as their stored contributions, without a
    preliminary rounded sum_duplicates operation. Other sparse formats should
    be converted to CSR by their owner before calling this row-wise helper.
    Nonfinite input/products or an overflowing sum fail explicitly.
    """
    K, alpha, sparse = _inputs(K, alpha)
    result = np.empty(K.shape[0], dtype=float)
    split_alpha = None
    if not sparse and K.shape[0] > 1:
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            split_alpha = _split_operand(alpha)
        _expanded_dense_rows(K, alpha, range(K.shape[0]), split_alpha, result)
        return result
    for i in range(K.shape[0]):
        if not sparse and split_alpha is None:
            # The same dense coefficients serve every row in this call.
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                split_alpha = _split_operand(alpha)
        result[i] = _expanded_row(K, alpha, sparse, i, _split_b=split_alpha)
    return result


def _dense_l1_upper(magnitude):
    """Upper bound the exact coefficient L1 sum under the existing model."""
    eps = np.finfo(float).eps
    with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
        eta = np.nextafter(0., 1.)
        terms = float(len(magnitude))
        neps = np.nextafter(terms*eps, np.inf)
        denominator = np.nextafter(1.-neps, 0.)
        gamma = np.nextafter(neps/denominator, np.inf)
        underflow = np.nextafter((4.*terms+4.)*eta, np.inf)
        sum_denominator = np.nextafter(1.-gamma, 0.)
        upper = np.nextafter(np.nextafter(np.sum(magnitude)+underflow, np.inf)/sum_denominator, np.inf)
    return upper if denominator > 0 and sum_denominator > 0 else np.inf


def _dense_first_tier(block, coefficient_upper, scores, gamma, underflow,
                      denominator, sum_denominator, threshold):
    """A deliberately looser bound than the existing positive-dot screen.

    S=sum(abs(K_ij*alpha_j)) <= max(abs(row))*upper_L1. Enlarge this
    for the positive BLAS product's possible upward error BEFORE applying
    the original screen's inflation. Thus cheap acceptance is a subset of
    the original screen's acceptance under its same IEEE/gamma assumptions.
    Max/min propagate NaNs and infinities, preserving finite-input validation.
    """
    low, high = np.min(block, axis=1), np.max(block, axis=1)
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise FloatingPointError('Nonfinite query kernel values.')
    maximum = np.maximum(-low, high)
    with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
        exact_sum_upper = np.nextafter(maximum*coefficient_upper, np.inf)
        positive_dot_upper = np.nextafter(np.nextafter(
            np.nextafter(1.+gamma, np.inf)*exact_sum_upper, np.inf)+underflow, np.inf)
        original_sum_upper = np.nextafter(np.nextafter(
            positive_dot_upper+underflow, np.inf)/sum_denominator, np.inf)
        bound = np.nextafter(np.nextafter(gamma*original_sum_upper, np.inf)+underflow, np.inf)
    return (np.isfinite(scores) & np.isfinite(bound) & (denominator > 0)
            & (sum_denominator > 0) & (bound <= threshold))


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
    coefficient_upper = None if sparse else _dense_l1_upper(magnitude)
    cheap_enabled = not sparse
    split_alpha = None
    batch_enabled = True
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
            terms = np.full(stop-start, len(alpha), dtype=float)
        with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
            # Enlarge n*eps, gamma_n, the positive sum and every bound operation.
            # The same gamma bound covers the ordinary and positive dot products.
            neps = np.nextafter(terms*eps, np.inf)
            denominator = np.nextafter(1.-neps, 0.)
            gamma = np.nextafter(neps/denominator, np.inf)
            underflow = np.nextafter((4.*terms+4.)*eta, np.inf)
            sum_denominator = np.nextafter(1.-gamma, 0.)
            threshold = np.nextafter(5e-7*np.maximum(1., np.abs(scores[start:stop])), 0.)
        if not sparse:
            if cheap_enabled:
                cheap = _dense_first_tier(block, coefficient_upper, scores[start:stop],
                                         gamma, underflow, denominator, sum_denominator, threshold)
                if np.all(cheap):
                    continue
                # Bound speculative work to one unsuccessful block per CALL.
                # Later blocks use exactly the original finite/fine screen.
                cheap_enabled = False
            elif not np.isfinite(block).all():
                raise FloatingPointError('Nonfinite query kernel values.')
            # Preserve the original full-block shape/layout for the fine
            # screen. Selecting only uncertain rows could alter BLAS rounding.
            block = np.abs(block)
        with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
            absolute_sum = np.asarray(block @ magnitude).reshape(-1)
            upper_sum = np.nextafter(np.nextafter(absolute_sum+underflow, np.inf)/sum_denominator, np.inf)
            bound = np.nextafter(np.nextafter(gamma*upper_sum, np.inf)+underflow, np.inf)
        reliable = (np.isfinite(scores[start:stop]) & np.isfinite(bound)
                    & (denominator > 0) & (sum_denominator > 0)
                    & (absolute_sum >= 0) & (bound <= threshold))
        uncertain = np.flatnonzero(~reliable)
        if not sparse and len(uncertain) > 1:
            if split_alpha is None:
                with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                    split_alpha = _split_operand(alpha)
            batch_enabled = _expanded_dense_rows(
                K, alpha, start + uncertain, split_alpha, scores,
                batch_enabled=batch_enabled)
            continue
        for local in uncertain:
            index = start+int(local)
            if not sparse and split_alpha is None:
                # Prepare only when an accurate dense row is first needed.
                with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                    split_alpha = _split_operand(alpha)
            scores[index] = _expanded_row(K, alpha, sparse, index, _split_b=split_alpha)
    return scores
