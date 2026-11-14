"""Shared compensated float64 dot/quadratic bounds; no solver dependency."""
from math import fsum
import math
import sys
import types
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


# Own the scalar binding token beside its original definition, before callers
# can replace that module attribute. This is a finite compatibility gate, not
# attestation of arbitrary mutations of NumPy or the floating environment.
_ORIGINAL_UPWARD = _upward_nonnegative
_ORIGINAL_UPWARD_CODE = _upward_nonnegative.__code__
_ORIGINAL_FSUM = fsum
_ORIGINAL_NEXTAFTER = math.nextafter
_ORIGINAL_UFUNC_TYPE = np.ufunc
_BOUND_UFUNCS = tuple((attribute, name, getattr(np, attribute, None))
                      for attribute, name in (
                          ('add', 'add'), ('divide', 'divide'),
                          ('multiply', 'multiply'), ('absolute', 'absolute'),
                          ('abs', 'absolute'),
                          ('nextafter', 'nextafter'), ('isfinite', 'isfinite'),
                          ('less', 'less')))
_ORIGINAL_ERRSTATE = np.errstate
_ERRSTATE_METHODS = tuple((name, getattr(np.errstate, name, None),
                          getattr(getattr(np.errstate, name, None), '__code__', None))
                         for name in ('__init__', '__enter__', '__exit__'))


def _builtin_math(function, name):
    return (type(function) is types.BuiltinFunctionType
            and function.__module__ == 'math' and function.__name__ == name)


def _standard_feature_bound_bindings():
    """Decline vector bounds for ordinary pre/post-import replacement hooks."""
    if (_upward_nonnegative is not _ORIGINAL_UPWARD
            or _upward_nonnegative.__code__ is not _ORIGINAL_UPWARD_CODE
            or _upward_nonnegative.__defaults__ is not None
            or _upward_nonnegative.__kwdefaults__ is not None
            or fsum is not _ORIGINAL_FSUM or not _builtin_math(fsum, 'fsum')
            or math.nextafter is not _ORIGINAL_NEXTAFTER
            or not _builtin_math(math.nextafter, 'nextafter')):
        return False
    for attribute, name, original in _BOUND_UFUNCS:
        function = getattr(np, attribute, None)
        if (function is not original or type(function) is not _ORIGINAL_UFUNC_TYPE
                or function.__name__ != name):
            return False
    cls = np.errstate
    if cls is not _ORIGINAL_ERRSTATE or type(cls) is not type or cls.__name__ != 'errstate':
        return False
    initializer = cls.__dict__.get('__init__')
    owner_name = getattr(initializer, '__globals__', {}).get('__name__')
    if owner_name not in ('numpy._core._ufunc_config', 'numpy.core._ufunc_config'):
        return False
    owner = sys.modules.get(owner_name)
    if getattr(owner, 'errstate', None) is not cls:
        return False
    for name, original, code in _ERRSTATE_METHODS:
        method = cls.__dict__.get(name)
        if (method is not original or getattr(method, '__code__', None) is not code
                or getattr(method, '__globals__', None) is not initializer.__globals__
                or hasattr(method, '__wrapped__') or code is None
                or code.co_qualname != 'errstate.'+name
                or code.co_filename != initializer.__code__.co_filename):
            return False
    return True


def _upward_nonnegative_array(values):
    """Vector sibling of the scalar bound, preserving its two refusals."""
    if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 1:
        raise ValueError('Vector bound inputs must be binary64 vectors.')
    if not np.isfinite(values).all() or np.less(values, 0.).any():
        raise FloatingPointError('Initial-state error bound is not finite.')
    with np.errstate(over='ignore', under='ignore'):
        result = np.nextafter(values, np.inf)
    if not np.isfinite(result).all():
        raise FloatingPointError('Initial-state error bound exceeds finite arithmetic.')
    return result


def _dual_coordinate_bounds(values, measured_absolute, constants):
    """Four outward stages and the distinct final coordinate nextafter.

    The caller supplies checked finite fsum results. Every arithmetic operation
    is explicit: no reassociation, fused operation or combined error factor.
    Final coordinate infinity is deliberately returned; the unchanged endpoint
    handles it after all coordinates, unlike an overflowing outward stage.
    """
    if (type(values) is not np.ndarray or type(measured_absolute) is not np.ndarray
            or values.dtype != np.float64 or measured_absolute.dtype != np.float64
            or values.ndim != 1 or values.shape != measured_absolute.shape):
        raise ValueError('Coordinate bound inputs must be matching binary64 vectors.')
    eps, allowance, denominator, tail = constants
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        numerator = _upward_nonnegative_array(np.add(measured_absolute, allowance))
        absolute_upper = _upward_nonnegative_array(np.divide(numerator, denominator))
        inner_error = _upward_nonnegative_array(np.multiply(4*eps, absolute_upper))
        error = _upward_nonnegative_array(np.add(inner_error, tail))
        upper = np.nextafter(np.add(np.absolute(values), error), np.inf)
    return error, upper


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


