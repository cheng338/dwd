"""Portable float64 residuals with compensated products and accumulation.

Used only after the ordinary residual check fails. No extended-precision dtype,
BLAS provider change or dense extra matrix is needed. Dekker splitting operates
on frexp mantissas, so the splitter cannot overflow. Scaled subnormal terms are
covered by an absolute underflow allowance; nonfinite products fail closed.
"""
import math
import numpy as np


def _gradual_underflow():
    with np.errstate(under='ignore'):
        tiny = np.finfo(float).tiny
        half = np.float64(tiny) * .5
        eta = np.nextafter(0., 1.)
        valid = half != 0. and half * 2. == tiny and eta > 0.
    if not valid:
        raise FloatingPointError('Compensated residuals require gradual float64 underflow.')
    return float(eta)


def _products(a, b):
    """Return high/low product terms; each exact product differs by at most 2 eta.

    Before scaling, normal mantissa products have an error-free Dekker split.
    Scaling each of the two terms can round only on underflow (at most half a
    minimum subnormal each); use a larger 2 eta allowance per product.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError('Invalid compensated product inputs.')
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        am, ae = np.frexp(a)
        bm, be = np.frexp(b)
        splitter = 134217729.  # 2**27 + 1, for binary64 significands.
        ca, cb = splitter * am, splitter * bm
        ah, bh = ca - (ca - am), cb - (cb - bm)
        al, bl = am - ah, bm - bh
        high = am * bm
        low = ((ah * bh - high) + ah * bl + al * bh) + al * bl
        exponent = ae + be
        high, low = np.ldexp(high, exponent), np.ldexp(low, exponent)
    if not np.isfinite(high).all() or not np.isfinite(low).all():
        raise FloatingPointError('Nonfinite or overflowing compensated products.')
    return high, low


def _fsum(terms):
    try:
        value = math.fsum(terms)
    except (ValueError, OverflowError) as exc:
        raise FloatingPointError('Nonfinite or overflowing compensated residual sum.') from exc
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite compensated residual sum.')
    return value


def compensated_residual(K, shift, rhs, x, s, target_sum):
    """Return residual, constraint, scores, and outward roundoff allowances.

    Residuals sum the original equation's expanded terms directly, rather than
    subtracting an already-rounded score. The allowances assume ordinary IEEE
    binary64 round-to-nearest and correctly/faithfully rounded math.fsum; they
    cover product underflow and a conservatively enlarged final-sum ulp. They
    are not a certificate for the input Gram matrix or later solver arithmetic.
    Additional resident storage is O(n).
    """
    eta = _gradual_underflow()
    n = len(x)
    shifted_hi, shifted_lo = _products(np.full(n, shift), x)
    residual, scores = np.empty(n), np.empty(n)
    for i, row in enumerate(K):
        hi, lo = _products(row, x)
        scores[i] = _fsum([*hi, *lo])
        residual[i] = _fsum([float(rhs[i]), -float(s), -float(shifted_hi[i]),
                             -float(shifted_lo[i]), *(-hi), *(-lo)])
    constraint = _fsum([float(target_sum), *(-x)])
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        # One extra product is shift*x; allow another ulp for outward rounding.
        bounds = np.nextafter(4 * np.finfo(float).eps * np.abs(residual)
                             + (2 * n + 6) * eta, np.inf)
        constraint_bound = np.nextafter(4 * np.finfo(float).eps * abs(constraint) + 2 * eta, np.inf)
    if not np.isfinite(bounds).all() or not np.isfinite(constraint_bound):
        raise FloatingPointError('Unrepresentable compensated residual error allowance.')
    return residual, constraint, scores, bounds, float(constraint_bound)
