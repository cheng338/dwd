"""Shared compensated float64 dot/quadratic bounds; no solver dependency."""
from math import fsum
import numpy as np


def _upward_nonnegative(value):
    """Round a finite nonnegative bound outward, rejecting bound overflow."""
    if not np.isfinite(value) or value < 0:
        raise FloatingPointError('Initial-state error bound is not finite.')
    with np.errstate(over='ignore', under='ignore'):
        result = float(np.nextafter(float(value), np.inf))
    if not np.isfinite(result):
        raise FloatingPointError('Initial-state error bound exceeds finite arithmetic.')
    return result


def _compensated_dot(a, b):
    """Return a float64 dot and an absolute forward-error bound.

    Products are rounded float64 values; math.fsum accumulates those values.
    Use eps (twice unit roundoff) conservatively: product error is at most
    eps*|exact product| + eta, and fsum's possible last-bit/double-rounding
    error is covered by 2*eps*sum(abs(products)) + eta. Here eta is the
    smallest subnormal, so gradual-underflow errors are included explicitly.
    Inflating the measured positive sum gives S_upper >= sum(abs(a*b));
    4*eps*S_upper + 2*(n+1)*eta covers both stages. Scalar bound operations
    are rounded outward. No wider platform-dependent dtype is assumed.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError('Compensated dot inputs must be matching vectors.')
    n, eps = len(a), np.finfo(float).eps
    # The bound includes gradual underflow, not flush-to-zero/denormals-are-zero.
    # Check both output and input subnormals without changing floating-point mode.
    half_normal = np.array([0x0008000000000000], dtype=np.uint64).view(np.float64)[0]
    with np.errstate(under='ignore'):
        halved = np.multiply(np.float64(np.finfo(float).tiny), np.float64(.5))
        doubled = np.multiply(half_normal, np.float64(2.))
    if (np.asarray(halved).view(np.uint64).item() != 0x0008000000000000
            or np.asarray(doubled).view(np.uint64).item() != 0x0010000000000000):
        raise FloatingPointError('Compensated initial-state checking requires gradual float64 underflow.')
    with np.errstate(under='ignore'):
        eta = float(np.nextafter(0., 1.))
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        products = a * b
    if not np.isfinite(products).all():
        raise FloatingPointError('Nonfinite products in compensated initial-state check.')
    try:
        value = fsum(products.tolist())
        measured_absolute = fsum(np.abs(products).tolist())
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Compensated initial-state accumulation overflowed.') from exc
    if not np.isfinite(value) or not np.isfinite(measured_absolute):
        raise FloatingPointError('Nonfinite compensated initial-state accumulation.')
    # Local error policy: underflow is included in eta's allowance, while each
    # outward-bound operation explicitly rejects nonfinite/overflow results.
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        numerator = _upward_nonnegative(
            measured_absolute + _upward_nonnegative((n + 1) * eta))
        denominator = float(np.nextafter((1. - eps) * (1. - 2 * eps), 0.))
        absolute_upper = _upward_nonnegative(numerator / denominator)
        error = _upward_nonnegative(
            _upward_nonnegative((4 * eps) * absolute_upper)
            + _upward_nonnegative((2 * n + 2) * eta))
    return float(value), error


def _compensated_quadratic(K, alpha):
    """Check the represented K quadratic with only O(n) extra storage."""
    scores, errors = np.empty(len(alpha)), np.empty(len(alpha))
    for i, row in enumerate(K):
        scores[i], errors[i] = _compensated_dot(row, alpha)
    quadratic, dot_error = _compensated_dot(alpha, scores)
    propagation, propagation_error = _compensated_dot(np.abs(alpha), errors)
    bound = _upward_nonnegative(
        dot_error + _upward_nonnegative(propagation + propagation_error))
    return scores, quadratic, bound


