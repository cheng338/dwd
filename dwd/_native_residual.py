"""Guarded native compensated dot products for original-equation residuals.

CPython 3.12's ``math.sumprod`` uses the triple-length dot-product algorithm
from Ogita, Rump and Oishi (2005), Algorithm 5.10. Proposition 5.11 supplies
its forward error estimate. This module is an optional numerical screen:
unsupported arithmetic, uncertain bounds, or exceptional states return None
for the caller to evaluate with the portable compensated residual routine.

References:
https://github.com/python/cpython/blob/3.12/Modules/mathmodule.c
https://doi.org/10.1137/030601818
"""
import math
import sys
import types

import numpy as np

from ._compensated_residual import _fsum, _gradual_underflow
from ._compiled_residual import compiled_values


_MIN_OPERAND = 2.**-200
_MAX_OPERAND = 2.**200
_MAX_TERMS = 2**20
_EPS = np.finfo(float).eps


def _native_supported():
    """Enable only the audited implementation and IEEE binary64 arithmetic."""
    return (sys.implementation.name == 'cpython'
            and sys.version_info[:2] == (3, 12)
            and isinstance(getattr(math, 'sumprod', None), types.BuiltinFunctionType)
            and sys.float_info.radix == 2 and sys.float_info.mant_dig == 53
            and sys.float_info.max_exp == 1024)


def _round_to_nearest():
    # Use runtime operations, not expressions folded into constants. The last
    # test distinguishes round-to-nearest from rounding toward zero.
    one, half_ulp = float(1), float(2.**-53)
    return (one + half_ulp == one and -one - half_ulp == -one
            and one + (half_ulp + half_ulp + half_ulp)
            == one + (half_ulp + half_ulp + half_ulp + half_ulp))


def _up(value):
    return math.nextafter(float(value), math.inf)


def _add(a, b):
    return _up(a + b)


def _mul(a, b):
    return _up(a * b)


def _safe_operand(values):
    absolute = np.abs(values)
    # NaNs and infinities fail these comparisons. Zero is safe, including -0.
    return bool(np.all((absolute == 0.)
                       | ((absolute >= _MIN_OPERAND) & (absolute <= _MAX_OPERAND))))


def _bound_constants(terms, eta):
    """Outward constants for E <= (a*abs(computed) + b)/(1-a).

    Proposition 5.11 has gamma_(4*m-2), a=u+2*gamma**2, and
    b=gamma**3*sum(abs(products))+5*m*eta. Use eps=2*u conservatively.
    Replacing abs(exact) with abs(computed)+E gives the displayed bound.
    """
    neps = _mul(float(4 * terms - 2), _EPS)
    denominator = math.nextafter(1. - neps, 0.)
    gamma = _up(neps / denominator)
    square = _mul(gamma, gamma)
    a = _add(_EPS, _mul(2., square))
    return (a, _mul(square, gamma), _mul(float(5 * terms), eta),
            math.nextafter(1. - a, 0.))


def _error_bound(computed, absolute_sum_upper, constants):
    a, gamma_cube, underflow, denominator = constants
    numerator = _add(_add(_mul(a, abs(computed)),
                          _mul(gamma_cube, absolute_sum_upper)), underflow)
    return _up(numerator / denominator)


def native_compensated_residual(K, shift, rhs, x, s, target_sum):
    """Return residual, constraint, scores, and their three error allowances.

    The six outputs are ``(residual, constraint, scores, residual_bounds,
    constraint_bound, score_bounds)``. Return None when a precondition or the
    score precision screen fails; no solver state or tolerance is changed.

    Scores and residuals use separate native dot products. The residual includes
    the original equation's rhs, free intercept and shifted coefficient as
    additional products; it never subtracts an already-rounded kernel score.

    All nonzero operands must have magnitude between 2**-200 and 2**200 and
    there are at most 2**20 terms. These deliberately conservative limits keep
    the native product splitting and accumulation away from overflow and
    underflow, including the alternate Dekker path on builds without reliable
    fma. At most O(n) additional storage is used, with no new dense matrix.
    """
    if not _native_supported() or not _round_to_nearest():
        return None
    try:
        eta = _gradual_underflow()
        # Do not silently allocate a second dense float64 Gram matrix.
        if not isinstance(K, np.ndarray) or K.dtype != np.dtype(float) or K.ndim != 2:
            return None
        x, rhs = np.asarray(x, dtype=float), np.asarray(rhs, dtype=float)
        if x.ndim != 1 or rhs.shape != x.shape:
            return None
        n = len(x)
        if n == 0 or n + 3 > _MAX_TERMS or K.shape != (n, n):
            return None
        shift, s, target_sum = float(shift), float(s), float(target_sum)
        if not (_safe_operand(x) and _safe_operand(rhs)
                and _safe_operand(np.array([shift, s, target_sum]))):
            return None

        positive_x = x.tolist()
        negative_x = (-x).tolist()
        augmented_x = negative_x + [1., 1., 0.]
        # The faithful fsum contract already used by the portable routine gives
        # an upper sum after two outward representable steps.
        absolute_x = _up(_up(_fsum(np.abs(x).tolist())))
        score_constants = _bound_constants(n, eta)
        residual_constants = _bound_constants(n + 3, eta)
        residual, scores = np.empty(n), np.empty(n)
        residual_bounds, score_bounds = np.empty(n), np.empty(n)
        compiled = compiled_values(K, x, rhs, shift, s)
        for i, row in enumerate(K):
            if compiled is None:
                if not _safe_operand(row):
                    return None
                row_terms = row.tolist()
                score = math.sumprod(row_terms, positive_x)
                row_maximum = float(np.max(np.abs(row)))
            else:
                score = float(compiled[0][i])
                row_maximum = float(compiled[2][i])
            score_sum_upper = _mul(row_maximum, absolute_x)
            score_error = _error_bound(score, score_sum_upper, score_constants)
            # Native scores may feed later solver arithmetic. Require at least
            # the portable routine's enlarged faithful-sum precision budget.
            faithful_budget = _up(4 * _EPS * abs(score) + (2 * n + 6) * eta)
            if not math.isfinite(score) or not math.isfinite(score_error) or score_error > faithful_budget:
                return None
            if compiled is None:
                row_terms.extend([float(rhs[i]), -s, -shift])
                augmented_x[-1] = float(x[i])
                value = math.sumprod(row_terms, augmented_x)
            else:
                value = float(compiled[1][i])
            total_upper = _add(_add(_add(score_sum_upper, abs(float(rhs[i]))), abs(s)),
                               _mul(abs(shift), abs(float(x[i]))))
            allowance = _error_bound(value, total_upper, residual_constants)
            if not math.isfinite(value) or not math.isfinite(allowance):
                return None
            scores[i], score_bounds[i] = score, score_error
            residual[i], residual_bounds[i] = value, allowance

        constraint = _fsum([target_sum] + negative_x)
        constraint_bound = _up(4 * _EPS * abs(constraint) + 2 * eta)
        if not math.isfinite(constraint_bound):
            return None
        return residual, constraint, scores, residual_bounds, constraint_bound, score_bounds
    except (FloatingPointError, ValueError, OverflowError, TypeError):
        return None
