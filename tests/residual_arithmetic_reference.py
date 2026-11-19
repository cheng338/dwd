"""Independent exact equations and two explicit binary64 product models.

The models preserve the accumulation order of the reviewed core. They are not
claims about which model any particular CPython build uses. Fraction provides
the independent mathematical oracle, including on builds with math.sumprod.
"""
from fractions import Fraction
import math

import numpy as np


def exact(value):
    return Fraction.from_float(float(value))


def exact_sumprod(left, right):
    return sum((exact(a) * exact(b) for a, b in zip(left, right, strict=True)), Fraction())


def _two_sum(a, b):
    high = a + b
    z = high - a
    return high, (a - (high - z)) + (b - z)


def _split(value):
    temporary = value * 134217729.0
    high = temporary - (temporary - value)
    return high, value - high


def _dekker_product(a, b):
    ah, al = _split(a)
    bh, bl = _split(b)
    p = ah * bh
    q = ah * bl + al * bh
    high = p + q
    low = p - high + q + al * bl
    return high, low


def _fma_product(a, b):
    # In the guarded normal-product domain this is exactly (a*b, fma(a,b,-a*b)).
    # No hardware FMA or Python-version-specific math.fma is required.
    product = exact(a) * exact(b)
    high = a * b  # Preserve the IEEE-754 sign of a zero product.
    return high, float(product - exact(high))


def _sumprod(left, right, product):
    high = low = tiny = 0.0
    for a, b in zip(left, right, strict=True):
        ph, pl = product(float(a), float(b))
        sh, sl = _two_sum(high, ph)
        r1h, r1l = _two_sum(low, pl)
        r2h, r2l = _two_sum(r1h, sl)
        high, low, tiny = sh, r2h, tiny + r1l + r2l
    last_high, last_low = _two_sum(low, high)
    return tiny + last_low + last_high


def dekker_sumprod(left, right):
    return _sumprod(left, right, _dekker_product)


def fma_sumprod(left, right):
    return _sumprod(left, right, _fma_product)


def reference_rows(sumprod, K, x, rhs, shift, intercept):
    positive = [float(value) for value in x]
    negative = [-value for value in positive]
    scores, residuals = [], []
    for i, row in enumerate(K):
        row = [float(value) for value in row]
        scores.append(sumprod(row, positive))
        residuals.append(sumprod(
            row + [float(rhs[i]), -float(intercept), -float(shift)],
            negative + [1.0, 1.0, positive[i]]))
    return np.asarray(scores, dtype=np.float64), np.asarray(residuals, dtype=np.float64)


def _theoretical_error(computed, left, right):
    """Exact-rational conservative Prop. 5.11 expression, without dwd helpers."""
    terms = [exact(a) * exact(b) for a, b in zip(left, right, strict=True)]
    m = len(terms)
    eps, eta = Fraction(1, 2**52), Fraction(1, 2**1074)
    neps = (4 * m - 2) * eps
    gamma = neps / (1 - neps)
    a = eps + 2 * gamma**2
    assert 0 <= a < 1
    return (a * abs(exact(computed)) + gamma**3 * sum(map(abs, terms), Fraction())
            + 5 * m * eta) / (1 - a)


def assert_compiled_values_bounded(K, x, rhs, shift, intercept, scores, residuals):
    """Check original equations against exact theory, not another float result."""
    positive = [float(value) for value in x]
    negative = [-value for value in positive]
    for i, row in enumerate(K):
        row = [float(value) for value in row]
        augmented_row = row + [float(rhs[i]), -float(intercept), -float(shift)]
        augmented_x = negative + [1.0, 1.0, positive[i]]
        assert abs(exact(scores[i]) - exact_sumprod(row, positive)) <= _theoretical_error(scores[i], row, positive)
        assert abs(exact(residuals[i]) - exact_sumprod(augmented_row, augmented_x)) <= _theoretical_error(residuals[i], augmented_row, augmented_x)


def assert_public_result_bounded(K, shift, rhs, x, intercept, target_sum, result):
    """Check the six-output bounded residual contract and its faithful gate."""
    assert isinstance(result, tuple) and len(result) == 6
    residuals, constraint, scores, residual_bounds, constraint_bound, score_bounds = result
    n = len(x)
    for array in (residuals, scores, residual_bounds, score_bounds):
        assert isinstance(array, np.ndarray) and array.dtype == np.dtype('float64')
        assert array.shape == (n,) and np.isfinite(array).all()
    assert isinstance(constraint, float) and isinstance(constraint_bound, float)
    assert math.isfinite(constraint) and math.isfinite(constraint_bound)
    assert (residual_bounds >= 0).all() and (score_bounds >= 0).all() and constraint_bound >= 0
    eta = float.fromhex('0x0.0000000000001p-1022')
    eps = 2.0**-52
    for i, row in enumerate(K):
        score = exact_sumprod(row, x)
        residual = exact(rhs[i]) - exact(intercept) - exact(shift) * exact(x[i]) - score
        assert abs(exact(scores[i]) - score) <= exact(score_bounds[i])
        assert abs(exact(residuals[i]) - residual) <= exact(residual_bounds[i])
        faithful = math.nextafter(4 * eps * abs(float(scores[i])) + (2 * n + 6) * eta, math.inf)
        assert score_bounds[i] <= faithful
    exact_constraint = exact(target_sum) - sum(map(exact, x), Fraction())
    assert abs(exact(constraint) - exact_constraint) <= exact(constraint_bound)
