"""Inexpensive forward-error screen for ordinary kernel-system arithmetic.

Assumes IEEE binary64, round-to-nearest, gradual underflow, a faithfully rounded
math.fsum and a BLAS reduction with at most 2*n rounding stages per term.
Uncertain/unsupported screens use the existing compensated measurement. These
are conditional error estimates, not certificates for arbitrary BLAS or K.
"""
import math

import numpy as np

from ._compensated_residual import _gradual_underflow
from ._native_residual import _round_to_nearest
from ._quadratic_bounds import _compensated_dot

_EPS = np.finfo(float).eps


def _up(value):
    value = float(value)
    if not math.isfinite(value) or value < 0.:
        raise FloatingPointError('Unrepresentable ordinary residual allowance.')
    result = math.nextafter(value, math.inf)
    if not math.isfinite(result):
        raise FloatingPointError('Overflowing ordinary residual allowance.')
    return result


def _gamma(count):
    # eps is twice unit roundoff. Scalar bounds round outwards.
    neps = _up(count * _EPS)
    if neps >= .01:
        raise FloatingPointError('Ordinary reduction error bound is too large.')
    return _up(neps / math.nextafter(1. - neps, 0.))


def _add(a, b):
    return _up(a + b)


def _mul(a, b):
    return _up(a * b)


def _array_up(value):
    result = np.nextafter(value, np.inf)
    if not np.isfinite(result).all() or np.any(result < 0.):
        raise FloatingPointError('Unrepresentable ordinary vector allowance.')
    return result


def _norm_upper(values):
    """Scaled positive reduction avoids overflow/underflow in norm squares."""
    values = np.abs(np.asarray(values, dtype=float))
    scale = float(np.max(values))
    if scale == 0.:
        return 0.
    if not math.isfinite(scale):
        raise FloatingPointError('Nonfinite ordinary residual norm.')
    scaled = values / scale
    squared = float(np.dot(scaled, scaled))
    squared_upper = _mul(squared, _add(1., _gamma(4 * len(values) + 8)))
    return _mul(scale, _up(math.sqrt(squared_upper)))


class OrdinaryResidualBounds:
    """O(n) cached state, with an optional bounded-memory BLAS recheck."""

    def __init__(self, K, mean):
        if not _round_to_nearest():
            raise FloatingPointError('Ordinary residual bounds require round-to-nearest.')
        n = len(K)
        self.n = n
        self.gamma = _gamma(2 * n + 8)
        self.row_max = np.empty(n)
        self.row_sum = np.empty(n)
        self.minimum = math.inf
        eta = _gradual_underflow()
        denominator = math.nextafter(1. - self.gamma, 0.)
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            for start in range(0, n, 128):
                block = np.abs(K[start:start + 128])
                self.row_max[start:start + len(block)] = np.max(block, axis=1)
                absolute = np.sum(block, axis=1)
                inflated = _array_up(absolute + _mul(2 * n + 2, eta))
                self.row_sum[start:start + len(block)] = _array_up(inflated / denominator)
                self.minimum = min(self.minimum, float(np.min(
                    np.where(block == 0., np.inf, block))))
        self.maximum = float(np.max(self.row_max))
        if not math.isfinite(self.maximum):
            raise FloatingPointError('Nonfinite ordinary kernel bound.')
        # Bound both reductions in H*mean and the two mean rounding errors.
        # The exact mean refers to stored K, not a symmetrized substitute.
        mean_error = _add(_mul(self.gamma, self.maximum), _mul(2 * n + 8, eta))
        average = float(np.mean(mean))
        average_error = _add(mean_error, _add(
            _mul(self.gamma, float(np.max(np.abs(mean)))), _mul(2 * n + 8, eta)))
        centered = mean - average
        centered_error = _add(_add(mean_error, average_error), _add(
            _mul(_gamma(2), max(float(np.max(np.abs(mean))), abs(average))), eta))
        self.centered_mean_norm = _add(
            _norm_upper(centered), _mul(_up(math.sqrt(n)), centered_error))
        upper_mean = math.nextafter(average + average_error, math.inf)
        if not math.isfinite(upper_mean) or upper_mean < 0.:
            raise FloatingPointError('Invalid PSD kernel mean bound.')
        self.mean_root = _up(math.sqrt(upper_mean))

    def measure(self, rhs, target_sum, x, shift, s, product, residual, constraint, summed,
                score_bounds=None):
        """Return (acceptable, maximum, RKHS bound), or decline by exception."""
        if not _round_to_nearest():
            raise FloatingPointError('Ordinary residual bounds require round-to-nearest.')
        eta = _gradual_underflow()
        absolute_x = np.abs(x)
        largest_x = float(np.max(absolute_x))
        nonzero_x = absolute_x[absolute_x != 0.]
        if len(nonzero_x):
            smallest_x = float(np.min(nonzero_x))
            # Product underflow/DAZ cases use expanded portable measurement.
            # Subsequent subtraction underflow is covered by the eta terms.
            for first in (self.minimum, shift):
                if math.isfinite(first) and first != 0. and (
                        math.frexp(abs(first))[1] + math.frexp(smallest_x)[1] < -1020):
                    raise FloatingPointError('Ordinary product may underflow.')
            if smallest_x < np.finfo(float).tiny or self.minimum < np.finfo(float).tiny:
                raise FloatingPointError('Ordinary operand is subnormal.')
        absolute_sum = math.fsum(absolute_x.tolist())
        absolute_error = _add(_mul(2 * _EPS, absolute_sum), eta)
        absolute_upper = _add(absolute_sum, absolute_error)
        absolute_lower = max(0., math.nextafter(absolute_sum - absolute_error, -math.inf))
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            if score_bounds is None:
                sums = np.minimum(_array_up(self.row_max * absolute_upper),
                                  _array_up(self.row_sum * largest_x))
                score_bounds = _array_up(_array_up(self.gamma * sums) + _mul(4 * self.n + 8, eta))
            # fl(rhs - fl(K*x) - fl(shift*x) - s): bound all operations,
            # including shift multiplication, using a four-stage allowance.
            shifted_upper = _array_up(abs(shift) * absolute_x)
            magnitude = _array_up(np.abs(rhs) + np.abs(product))
            magnitude = _array_up(magnitude + shifted_upper)
            magnitude = _array_up(magnitude + abs(s))
            bounds = _array_up(score_bounds + _array_up(_gamma(8) * magnitude))
            bounds = _array_up(bounds + _mul(16, eta))
        # fsum's possible last-bit error and target subtraction are separate.
        sum_error = _add(_mul(2 * _EPS, abs(summed)), eta)
        constraint_bound = _add(sum_error, _add(
            _mul(_gamma(2), _add(abs(target_sum), abs(summed))), _mul(2, eta)))
        maximum = float(np.max(_array_up(np.abs(residual) + bounds)))
        constraint_upper = _add(abs(constraint), constraint_bound)
        constraint_scale = max(1., absolute_lower, abs(target_sum))
        constraint_tolerance = math.nextafter(64 * _EPS * constraint_scale, 0.)
        equation_tolerance = math.nextafter(1e-10 * max(1., float(np.max(np.abs(rhs)))), 0.)

        # ||H||_2=1: use unprojected residual plus its uncertainty. This avoids
        # an unbounded rounded projection authorizing fast acceptance. Tighter
        # compensated checks remain available when this estimate is uncertain.
        numerator = _add(_add(_norm_upper(residual), _norm_upper(bounds)),
                         _mul(constraint_upper, self.centered_mean_norm))
        denominator = math.nextafter(2 * math.sqrt(shift), 0.)
        if denominator <= 0. or not math.isfinite(denominator):
            raise FloatingPointError('Unrepresentable ordinary RKHS denominator.')
        feature_error = _add(_up(numerator / denominator),
                             _mul(constraint_upper, self.mean_root))

        quadratic, dot_error = _compensated_dot(x, product)
        propagation, propagation_error = _compensated_dot(absolute_x, score_bounds)
        quadratic_error = _add(dot_error, _add(propagation, propagation_error))
        quadratic_lower = math.nextafter(quadratic - quadratic_error, -math.inf)
        scale = max(1., math.nextafter(math.sqrt(max(0., quadratic_lower)), 0.))
        feature_tolerance = math.nextafter(5e-7 * scale, 0.)
        return (maximum <= equation_tolerance and constraint_upper <= constraint_tolerance
                and feature_error <= feature_tolerance), maximum, feature_error

    def blocked_product(self, K, x):
        """Recompute K*x in short dots, with a bound independent of BLAS order.

        At most 128 partial vectors are stored. Each dot has at most block
        terms; their errors sum to gamma_(2*block+8)*sum(abs(K_ij*x_j)).
        The final partial-vector reduction has its own independent allowance.
        Called only after measure has checked the operand/runtime conditions.
        No matrix entry, coefficient or objective is changed.
        """
        eta = _gradual_underflow()
        if not _round_to_nearest():
            raise FloatingPointError('Blocked residual bounds require round-to-nearest.')
        block = max(32, (self.n + 127)//128)
        count = (self.n + block - 1)//block
        parts = np.empty((self.n, count))
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            for i, start in enumerate(range(0, self.n, block)):
                parts[:, i] = K[:, start:start + block]@x[start:start + block]
            scores = np.sum(parts, axis=1)
            partial_absolute = np.sum(np.abs(parts), axis=1)
            gamma_sum = _gamma(2 * count + 8)
            partial_upper = _array_up(_array_up(partial_absolute + _mul(2 * count + 8, eta))
                                      / math.nextafter(1. - gamma_sum, 0.))
            absolute = np.abs(x)
            total = math.fsum(absolute.tolist())
            total_upper = _add(total, _add(_mul(2 * _EPS, total), eta))
            magnitude = np.minimum(_array_up(self.row_max * total_upper),
                                   _array_up(self.row_sum * float(np.max(absolute))))
            errors = _array_up(_array_up(_gamma(2 * block + 8) * magnitude)
                                + _array_up(gamma_sum * partial_upper))
            errors = _array_up(errors + _mul(8 * self.n + 8 * count + 16, eta))
        if not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite blocked kernel product.')
        return scores, errors
