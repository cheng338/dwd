"""Feasible lower dual bounds for free-intercept kernel DWD diagnostics.

For a symmetric positive-semidefinite represented kernel, the primal objective
and reported gap remain numerical estimates. Only the dual
endpoint is rounded outward, conditional on ordinary IEEE binary64 with gradual
underflow and correctly rounded Decimal ln/exp/sqrt. No iterate is modified.
"""
from decimal import Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
from math import fsum
import numpy as np
from ._compensated_residual import _gradual_underflow
from ._native_residual import _round_to_nearest
from ._quadratic_bounds import _compensated_dot

_DENOMINATOR = 1 << 52
# A diagnostic-work budget, never a rank or positive-eigenvalue cutoff.
_EXACT_ENTRY_BUDGET = 4096


class _AsymmetricKernel(ValueError):
    """The nonzero dual formula does not certify this literal score map."""


def _fraction(value):
    return Fraction.from_float(float(value))


def _float_lower(value):
    result = float(value)
    if _fraction(result) > value:
        result = float(np.nextafter(result, -np.inf))
    return result


def feasible_dual_slopes(raw, y):
    """Project, quantize down to 2**-52, and exactly balance class masses.

    Reducing excess class mass preserves [0,1]. Feasibility is an exact property
    of represented dyadic numbers, not a tolerance on a floating-point sum.
    """
    raw, y = np.asarray(raw, float), np.asarray(y, float)
    if (raw.ndim != 1 or y.shape != raw.shape or not np.isfinite(raw).all()
            or not np.isin(y, [-1., 1.]).all()):
        raise ValueError('Dual slopes and signed labels must be finite matching vectors.')
    lo, hi = -1., 1.
    for _ in range(64):
        mid = (lo + hi) * .5
        projected = np.clip(raw - mid * y, 0., 1.)
        if fsum((y * projected).tolist()) > 0:
            lo = mid
        else:
            hi = mid
    projected = np.clip(raw - ((lo + hi) * .5) * y, 0., 1.)
    units = [int(value * _DENOMINATOR) for value in projected]
    balance = sum(unit if label > 0 else -unit for unit, label in zip(units, y))
    excess_label, remaining = (1. if balance > 0 else -1.), abs(balance)
    for index, label in enumerate(y):
        if not remaining:
            break
        if label == excess_label:
            reduction = min(units[index], remaining)
            units[index] -= reduction
            remaining -= reduction
    if remaining:
        raise ArithmeticError('Unable to balance finite dual slopes.')
    return np.array([unit / _DENOMINATOR for unit in units], dtype=float)


def _conjugate_mean_lower(rho, q):
    q_exact = _fraction(q)
    exponent_exact = q_exact / (q_exact + 1)
    with localcontext() as context:
        context.prec, context.rounding = 50, ROUND_CEILING
        exponent = Decimal(exponent_exact.numerator) / Decimal(exponent_exact.denominator)
    cache = {0.: Decimal(0), 1.: Decimal(1)}
    with localcontext() as context:
        context.prec, context.rounding = 50, ROUND_FLOOR
        total = Decimal(0)
        for value in rho:
            value = float(value)
            if value not in cache:
                represented = Decimal.from_float(value)
                # Decimal transcendental functions round to nearest even when
                # context.rounding differs; move their endpoint outwards.
                if q == 1.:
                    lower = represented.sqrt().next_minus()
                else:
                    lower = (exponent * represented.ln().next_minus()).exp().next_minus()
                cache[value] = lower
            total += cache[value]
        return Fraction(total / Decimal(len(rho)))


def _quadratic_upper(K, signed, target):
    """Upper represented quadratic, with bounded temporary storage.

    A constant kernel has an exact zero quadratic for an exactly balanced
    signed vector. Otherwise bound BLAS products and a compensated outer dot.
    Small uncertain cases can escalate to exact arithmetic within a fixed work
    budget. Larger uncertain cases retain a conservative, possibly weak bound.
    """
    _gradual_underflow()
    n, constant = len(signed), float(K[0, 0])
    is_constant = True
    for start in range(0, n, 128):
        block = K[start:start + 128]
        if not np.array_equal(block, K[:, start:start + 128].T):
            raise _AsymmetricKernel('Stored kernel is not exactly symmetric.')
        if is_constant and not np.all(block == constant):
            is_constant = False
    if is_constant:
        if constant < 0:
            raise ValueError('A negative constant kernel is not positive semidefinite.')
        total = sum((_fraction(value) for value in signed), Fraction(0))
        return _fraction(constant) * total * total, 'exact_constant'
    if not _round_to_nearest():
        raise FloatingPointError('Dual quadratic bounds require round-to-nearest.')
    with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
        product = K @ signed
        eta, eps = np.nextafter(0., 1.), np.finfo(float).eps
        neps = np.nextafter(n * eps, np.inf)
        denominator = np.nextafter(1. - neps, 0.)
        gamma = np.nextafter(neps / denominator, np.inf)
        sum_denominator = np.nextafter(1. - gamma, 0.)
        underflow = np.nextafter((4. * n + 4.) * eta, np.inf)
        errors = np.empty(n)
        magnitude = np.abs(signed)
        for start in range(0, n, 128):
            absolute = np.abs(K[start:start + 128]) @ magnitude
            absolute_upper = np.nextafter(
                np.nextafter(absolute + underflow, np.inf) / sum_denominator, np.inf)
            errors[start:start + 128] = np.nextafter(
                np.nextafter(gamma * absolute_upper, np.inf) + underflow, np.inf)
    if (denominator <= 0 or sum_denominator <= 0 or not np.isfinite(product).all()
            or not np.isfinite(errors).all()):
        raise FloatingPointError('Unable to bound dual quadratic arithmetic.')
    value, outer_error = _compensated_dot(signed, product)
    propagation, propagation_error = _compensated_dot(magnitude, errors)
    allowance = _fraction(outer_error) + _fraction(propagation) + _fraction(propagation_error)
    if allowance > target and K.size <= _EXACT_ENTRY_BUDGET:
        exact = [_fraction(v) for v in signed]
        result = sum((left * _fraction(entry) * right
                      for left, row in zip(exact, K) for entry, right in zip(row, exact)), Fraction(0))
        return result, 'exact_small_escalation'
    return _fraction(value) + allowance, 'bounded_binary64'


def kernel_dual_lower_bound(K, y, raw_slopes, lambd, q):
    """Return a feasible dual lower bound, never a convergence declaration.

    Zero slopes give a valid bound of zero. Select that explicitly if it is
    stronger than the candidate bound or nonzero arithmetic cannot be bounded.
    A certified negative quadratic is rejected, never clipped or hidden.
    """
    rho = feasible_dual_slopes(raw_slopes, y)
    penalty, n = _fraction(lambd), len(y)
    method, fallback = None, None
    try:
        quadratic_upper, method = _quadratic_upper(
            K, y * rho, _fraction(1e-10) * (4 * penalty * n * n))
        if quadratic_upper < 0:
            raise ValueError('A certified negative dual quadratic is incompatible with a PSD kernel.')
        candidate = _conjugate_mean_lower(rho, q) - quadratic_upper / (4 * penalty * n * n)
        if candidate < 0:
            dual, fallback = 0., 'zero_slopes_give_a_stronger_certified_bound'
        else:
            dual = _float_lower(candidate)
    except _AsymmetricKernel:
        # The current fit policy accepts tiny asymmetry. Do not strengthen that
        # policy or claim the symmetric-kernel dual for an asymmetric score map.
        dual, fallback = 0., 'asymmetric_kernel_no_nonzero_dual_certificate'
    except (OverflowError, FloatingPointError):
        dual, fallback = 0., 'nonzero_dual_bound_unavailable'
    return dual, {
        'feasibility': 'exact_dyadic_class_mass_balance',
        'slope_quantum': 2. ** -52,
        'dual_quadratic_method': method,
        'fallback_reason': fallback,
        'dual_objective_is_lower_bound': True,
        'kernel_assumption': ('nonnegative represented RKHS penalty'
                              if fallback == 'asymmetric_kernel_no_nonzero_dual_certificate'
                              else 'symmetric positive-semidefinite represented kernel'),
        'primal_objective_and_gap_are_estimates': True,
        'arithmetic_assumptions': 'IEEE binary64 with gradual underflow; faithfully rounded math.fsum; correctly rounded Decimal ln/exp/sqrt',
    }
