"""Joint affine scores for represented binary64 K, alpha and b.

No kernel construction, fitted coefficient, intercept or tie rule is changed.
Adaptive ordinary results pass an affine accuracy AND sign screen. Other rows
sum expanded products and the intercept together, with an exact rational slow
path for exceptional range or unresolved subnormal cancellation. This is not a
universal correctly-rounded ordinary path or an arbitrary-BLAS certificate.
"""
from fractions import Fraction
import math

import numpy as np
from scipy.sparse import issparse

from dwd._compensated_residual import _products, _fsum, _split_operand
from dwd._kernel_scores import _inputs, _dense_l1_upper


_EPS = float(np.finfo(float).eps)
_ETA = float(np.finfo(float).smallest_subnormal)

_TILE_BYTES = 1024 * 1024
_TILE_MAX_ROWS = 8


def _intercept(value):
    raw = np.asarray(value)
    if raw.ndim != 0 or np.iscomplexobj(raw) or raw.dtype.kind not in 'biuf':
        raise ValueError('b must be a finite real scalar.')
    result = float(raw)
    if not math.isfinite(result):
        raise FloatingPointError('Nonfinite affine intercept.')
    return result


def _row_data(K, alpha, sparse, index):
    if sparse:
        start, stop = K.indptr[index:index + 2]
        return K.data[start:stop], alpha[K.indices[start:stop]]
    return K[index], alpha


def _exact_row(K, alpha, b, sparse, index):
    """Round the exact affine sum of represented binary64 input values once.

    CSR duplicates remain distinct products. No preliminary rounded duplicate
    summation, coefficient modification, snapping or sign-based output occurs.
    A nonzero exact sum below half the minimum subnormal can round to zero.
    """
    row, coefficients = _row_data(K, alpha, sparse, index)
    row = np.asarray(row, dtype=float)
    if not np.isfinite(row).all():
        raise FloatingPointError('Nonfinite query kernel values.')
    total = Fraction.from_float(b)
    for value, coefficient in zip(row, coefficients):
        if value != 0.0 and coefficient != 0.0:
            total += Fraction.from_float(float(value)) * Fraction.from_float(float(coefficient))
    try:
        result = float(total)
    except OverflowError as error:
        raise FloatingPointError('Affine score is outside finite binary64 range.') from error
    if not math.isfinite(result):
        raise FloatingPointError('Nonfinite exact affine score.')
    return result


def _finish_row(high, low, K, alpha, b, sparse, index):
    """Sum products and b jointly; recover only justified exceptional cases."""
    try:
        score = _fsum(high.tolist() + low.tolist() + [b])
    except FloatingPointError:
        # fsum can overflow internally before finite terms cancel. Its inputs
        # are finite product terms here, so re-evaluate the original affine sum.
        return _exact_row(K, alpha, b, sparse, index), True
    # _products loses at most 2 eta per product under its documented model.
    # Match the residual helper's enlarged final-sum allowance, including
    # the scalar intercept and outward rounding of this bound itself.
    allowance = math.nextafter(4.0 * _EPS * abs(score) + (2.0 * len(high) + 6.0) * _ETA,
                               math.inf)
    if not math.isfinite(allowance) or abs(score) <= allowance:
        return _exact_row(K, alpha, b, sparse, index), True
    return score, False


def _expanded_row(K, alpha, b, sparse, index, split_alpha=None):
    row, coefficients = _row_data(K, alpha, sparse, index)
    try:
        if sparse or split_alpha is None:
            high, low = _products(row, coefficients)
        else:
            high, low = _products(row, coefficients, _split_b=split_alpha)
    except FloatingPointError:
        # This also handles overflowing individual products that cancel in
        # the final affine sum. Nonfinite original inputs still fail in _exact_row.
        return _exact_row(K, alpha, b, sparse, index), True
    return _finish_row(high, low, K, alpha, b, sparse, index)


def _expanded_dense_rows(K, alpha, b, indices, split_alpha, output, exact_used,
                         batch_enabled=True):
    """Bound product scratch and preserve row-order fallback behavior."""
    if K.dtype.kind == 'f' and K.dtype.itemsize > 8:
        batch_enabled = False
    capacity = min(_TILE_MAX_ROWS, max(1, _TILE_BYTES // (128 * len(alpha))))
    position = 0
    while position < len(indices):
        stop = min(position + (capacity if batch_enabled else 1), len(indices))
        if stop - position < 2:
            index = int(indices[position])
            output[index], exact_used[index] = _expanded_row(
                K, alpha, b, False, index, split_alpha)
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
                    index = int(index)
                    output[index], exact_used[index] = _finish_row(
                        high[local], low[local], K, alpha, b, False, index)
            except MemoryError:
                fallback = True
        if fallback:
            tile = high = low = coefficients = None
            batch_enabled = False
            for index in indices[position:stop]:
                index = int(index)
                output[index], exact_used[index] = _expanded_row(
                    K, alpha, b, False, index, split_alpha)
        tile = high = low = coefficients = None
        position = stop
    return batch_enabled


def _dot_bound(absolute_sum, gamma, underflow, sum_denominator):
    with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
        upper = np.nextafter(np.nextafter(absolute_sum + underflow, np.inf) /
                             sum_denominator, np.inf)
        return np.nextafter(np.nextafter(gamma * upper, np.inf) + underflow, np.inf)


def _affine_accept(dot_scores, scores, b, dot_bound, denominator, sum_denominator):
    """Bound dot error plus final addition; accept only a separated sign."""
    eps, eta = _EPS, _ETA
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        # For fl(d+b), eps*(abs(d)+abs(b)) + eta exceeds the ordinary
        # round-to-nearest addition allowance. Inflate every positive step.
        addition_scale = np.nextafter(np.abs(dot_scores) + abs(b), np.inf)
        addition_bound = np.nextafter(np.nextafter(eps * addition_scale, np.inf) + eta,
                                      np.inf)
        bound = np.nextafter(dot_bound + addition_bound, np.inf)
        threshold = np.nextafter(5e-7 * np.maximum(1.0, np.abs(scores)), 0.0)
    return (np.isfinite(scores) & np.isfinite(bound) & (denominator > 0)
            & (sum_denominator > 0) & (bound <= threshold) & (np.abs(scores) > bound))


def _cheap_accept(block, coefficient_upper, dot_scores, scores, b, gamma,
                  underflow, denominator, sum_denominator):
    low, high = np.min(block, axis=1), np.max(block, axis=1)
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise FloatingPointError('Nonfinite query kernel values.')
    maximum = np.maximum(-low, high)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        exact_sum_upper = np.nextafter(maximum * coefficient_upper, np.inf)
        positive_dot_upper = np.nextafter(np.nextafter(
            np.nextafter(1.0 + gamma, np.inf) * exact_sum_upper, np.inf) + underflow,
            np.inf)
    bound = _dot_bound(positive_dot_upper, gamma, underflow, sum_denominator)
    return _affine_accept(dot_scores, scores, b, bound, denominator, sum_denominator)


def affine_scores(K, alpha, b, *, policy='adaptive', return_info=False):
    """Evaluate query-by-training K @ alpha + b without a new prediction tie rule.

    ``policy='adaptive'`` retains an ordinary result only when the documented
    IEEE/gamma dot bound and outward addition bound pass both the affine error
    tolerance and a strict sign separation check. ``'compensated'`` evaluates
    every row jointly. Exact-rational recovery is limited to exceptional range
    or unresolved refined sign cases; it can be expensive and has no time cap.
    Ordinary accepted scores are not promised correctly rounded.

    Like the existing helpers, inputs are interpreted through their binary64
    working representation. Dense and CSR (including duplicate stored entries)
    are supported. There is no extra full-sized product matrix. Native, Python
    and rational temporary storage remain proportional to a row, with bounded
    dense tiles; memory exhaustion propagates after row-wise retry where possible.

    Returns a float64 vector, or (vector, diagnostic counts) with return_info.
    The caller should retain the existing ``scores > 0`` classification rule.
    """
    if policy not in ('adaptive', 'compensated'):
        raise ValueError("policy must be 'adaptive' or 'compensated'.")
    b = _intercept(b)
    K, alpha, sparse = _inputs(K, alpha)
    output = np.empty(K.shape[0], dtype=float)
    refined = np.zeros(K.shape[0], dtype=bool)
    exact_used = np.zeros(K.shape[0], dtype=bool)
    split_alpha = None
    batch_enabled = True
    cheap_rows = fine_rows = 0
    if policy == 'compensated':
        refined[:] = True
        if not sparse and K.shape[0] > 1:
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                split_alpha = _split_operand(alpha)
            _expanded_dense_rows(K, alpha, b, np.arange(K.shape[0]), split_alpha,
                                 output, exact_used)
        else:
            for index in range(K.shape[0]):
                if not sparse and split_alpha is None:
                    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                        split_alpha = _split_operand(alpha)
                output[index], exact_used[index] = _expanded_row(
                    K, alpha, b, sparse, index, split_alpha)
    else:
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            dots = np.asarray(K @ alpha, dtype=float).reshape(-1)
            output[:] = dots + b
        magnitude = np.abs(alpha)
        coefficient_upper = None if sparse else _dense_l1_upper(magnitude)
        cheap_enabled = not sparse
        eps, eta = _EPS, _ETA
        for start in range(0, K.shape[0], 128):
            stop = min(start + 128, K.shape[0])
            if sparse:
                block = K[start:stop].astype(float, copy=True)
                if not np.isfinite(block.data).all():
                    raise FloatingPointError('Nonfinite query kernel values.')
                block.data = np.abs(block.data)
                terms = np.diff(block.indptr).astype(float)
            else:
                block = np.asarray(K[start:stop], dtype=float)
                terms = np.full(stop - start, len(alpha), dtype=float)
            with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
                neps = np.nextafter(terms * eps, np.inf)
                denominator = np.nextafter(1.0 - neps, 0.0)
                gamma = np.nextafter(neps / denominator, np.inf)
                underflow = np.nextafter((4.0 * terms + 4.0) * eta, np.inf)
                sum_denominator = np.nextafter(1.0 - gamma, 0.0)
            if not sparse:
                if cheap_enabled:
                    cheap = _cheap_accept(block, coefficient_upper, dots[start:stop],
                                          output[start:stop], b, gamma, underflow,
                                          denominator, sum_denominator)
                    if np.all(cheap):
                        cheap_rows += stop - start
                        continue
                    cheap_enabled = False
                elif not np.isfinite(block).all():
                    raise FloatingPointError('Nonfinite query kernel values.')
                block = np.abs(block)
            with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
                absolute_sum = np.asarray(block @ magnitude).reshape(-1)
                bound = _dot_bound(absolute_sum, gamma, underflow, sum_denominator)
            accepted = ((absolute_sum >= 0) & _affine_accept(
                dots[start:stop], output[start:stop], b, bound, denominator, sum_denominator))
            fine_rows += int(np.count_nonzero(accepted))
            uncertain = np.flatnonzero(~accepted)
            indices = start + uncertain
            refined[indices] = True
            if not sparse and len(uncertain) > 1:
                if split_alpha is None:
                    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                        split_alpha = _split_operand(alpha)
                batch_enabled = _expanded_dense_rows(
                    K, alpha, b, indices, split_alpha, output, exact_used, batch_enabled)
                continue
            for index in indices:
                index = int(index)
                if not sparse and split_alpha is None:
                    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                        split_alpha = _split_operand(alpha)
                output[index], exact_used[index] = _expanded_row(
                    K, alpha, b, sparse, index, split_alpha)
    if not np.isfinite(output).all():
        raise FloatingPointError('Nonfinite affine kernel scores.')
    if return_info:
        joint_rows = int(np.count_nonzero(refined))
        return output, {'policy': policy, 'rows': int(K.shape[0]),
                        'ordinary_rows': int(K.shape[0]) - joint_rows,
                        'joint_rows': joint_rows,
                        'exact_rows': int(np.count_nonzero(exact_used)),
                        'cheap_accepted_rows': int(cheap_rows),
                        'fine_accepted_rows': int(fine_rows)}
    return output


def adaptive_kernel_affine(K, alpha, b):
    return affine_scores(K, alpha, b, policy='adaptive')


def compensated_kernel_affine(K, alpha, b):
    return affine_scores(K, alpha, b, policy='compensated')
