"""Independent F11 regression oracle and fixed finite input fixtures.

Gaussian rational elimination and manual binary64 midpoint cells are independent
of the package LDL/rounding implementation. No runtime module is imported.
"""
from fractions import Fraction
import hashlib
import struct

import numpy as np
from numpy.testing import assert_allclose


def solve_exact(matrix, rhs):
    if not matrix or len(rhs) != len(matrix) or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("The exact system must be nonempty, square, and match its RHS.")
    rows = [list(row)+[value] for row, value in zip(matrix, rhs)]
    for j in range(len(rows)):
        pivot = next(i for i in range(j, len(rows)) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scale = rows[j][j]
        rows[j] = [value/scale for value in rows[j]]
        for i in range(len(rows)):
            if i != j and rows[i][j]:
                scale = rows[i][j]
                rows[i] = [a-scale*b for a, b in zip(rows[i], rows[j])]
    return [row[-1] for row in rows]


_SIGN = 1 << 63
_MAX = 0x7FEFFFFFFFFFFFFF


def word(x):
    return struct.unpack('>Q', struct.pack('>d', float(x)))[0]


def bits(x):
    return format(word(x), '016x')


def dyadic(magnitude_word):
    """Independent positive finite binary64 raw-word decoder."""
    exponent, fraction = magnitude_word >> 52, magnitude_word & ((1 << 52)-1)
    assert 0 <= exponent < 2047
    mantissa = fraction if exponent == 0 else (1 << 52)+fraction
    power = -1074 if exponent == 0 else exponent-1023-52
    return Fraction(mantissa << power, 1) if power >= 0 else Fraction(mantissa, 1 << -power)


def raw_fraction(value):
    raw = word(value)
    magnitude = dyadic(raw & (_SIGN-1))
    return -magnitude if raw & _SIGN else magnitude


class ExactAction:
    """Raw-word target adapter to the frozen independent Gaussian solver.

    The direct next-point RHS is assembled independently of production LDL.
    No float-to-Fraction arithmetic/conversion defines the subnormal target.
    """
    def __init__(self, X, shift, h, old):
        self.input_X = tuple(tuple(float(v) for v in row) for row in X)
        self.input_h = tuple(float(v) for v in h)
        self.input_old = tuple(float(v) for v in old)
        assert len(X) > 0 and len(X[0]) > 0
        self.p = len(X[0])
        assert all(len(row) == self.p for row in X)
        assert len(h) == len(X) and len(old) == self.p+1
        self.shift = float(shift)
        s = raw_fraction(shift)
        assert s > 0
        self.A = [[Fraction(1)]+[raw_fraction(v) for v in row] for row in X]
        theta = list(map(raw_fraction, old))
        target = [sum((a*b for a, b in zip(row, theta)), Fraction())-raw_fraction(hi)
                  for row, hi in zip(self.A, h)]
        P = [[sum((row[j]*row[k] for row in self.A), Fraction())+(s if j == k and j else 0)
              for k in range(self.p+1)] for j in range(self.p+1)]
        rhs = [sum((row[j]*t for row, t in zip(self.A, target)), Fraction())
               for j in range(self.p+1)]
        self.reference_next = solve_exact(P, rhs)

    def assess(self, theta):
        assert len(theta) == self.p+1
        errors = [raw_fraction(a)-b for a, b in zip(theta, self.reference_next)]
        scores = [sum((a*b for a, b in zip(row, errors)), Fraction()) for row in self.A]
        return dict(coefficient_error_squared=sum((e*e for e in errors[1:]), Fraction()),
                    score_error_squared=sum((e*e for e in scores), Fraction()),
                    intercept_error_abs=abs(errors[0]))


def assert_rounding_cell(exact, returned):
    """No float(Fraction), production rounding verifier or nextafter used."""
    exact = Fraction(exact)
    raw = word(returned)
    magnitude = raw & (_SIGN-1)
    assert magnitude <= _MAX
    if exact == 0:
        assert raw == 0  # Explicit contract for exact rational zero.
        return
    assert bool(raw & _SIGN) == (exact < 0)
    exact = abs(exact)
    if magnitude == 0:
        assert exact <= Fraction(1, 1 << 1075)
        return
    value = dyadic(magnitude)
    lower = (dyadic(magnitude-1)+value)/2
    upper = (value+dyadic(magnitude+1))/2 if magnitude < _MAX else Fraction((1 << 1024)-(1 << 970))
    even = magnitude % 2 == 0
    assert exact > lower or (even and exact == lower)
    assert exact < upper or (even and exact == upper)


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()



def assert_receipt(action, theta, receipt, *, require_immutable=True):
    assert type(theta) is np.ndarray and theta.dtype == np.dtype('float64')
    assert theta.shape == (len(action.input_old),)
    if require_immutable:
        assert not theta.flags.writeable
    assert receipt['schema'] == 'f11-exact-coordinate-next-v1'
    assert receipt['status'] == 'exact_coordinates_rounded'
    assert receipt['target'] == 'original_feature_fixed_h'
    assert receipt['input_sha256'] == digest(action.input_X)
    assert receipt['shift_bits'] == bits(action.shift)
    assert receipt['h_sha256'] == digest(action.input_h)
    assert receipt['beta_sha256'] == digest(action.input_old[1:])
    assert receipt['offset_bits'] == bits(action.input_old[0])
    assert receipt['theta_bits'] == [bits(v) for v in theta]
    assert receipt['theta_sha256'] == digest(theta)
    assert receipt['coordinate_rounding_verified'] is True
    assert receipt['rounding'] == 'binary64_round_to_nearest_ties_to_even'
    assert [Fraction(v) for v in receipt['exact_theta']] == action.reference_next
    for target, actual in zip(action.reference_next, theta):
        assert_rounding_cell(target, actual)
    assessed = action.assess(theta)
    for key, oracle_key in [('beta_error_squared', 'coefficient_error_squared'),
                            ('score_error_squared', 'score_error_squared'),
                            ('intercept_error_abs', 'intercept_error_abs')]:
        assert Fraction(receipt['reference_errors'][key]) == assessed[oracle_key]
    assert receipt['factorizations_completed'] == 1
    return assessed




def case(name):
    i = np.arange(16)
    walsh = np.column_stack([1.0 - 2.0 * ((i >> bit) & 1) for bit in (0, 1, 2)])
    y = np.where(walsh[:, 0] > 0, 1.0, -1.0)
    if name == "ordinary":
        return walsh + [0.25, -0.5, 0.125], y, 0.05
    if name == "original_far_outlier":
        X = np.random.RandomState(31).normal(size=(10, 3))
        X[4] *= 2e8
        return X, 2.0 * np.tile([0, 1], 5) - 1.0, 0.05
    if name == "dyadic_near_collinear":
        X = np.column_stack((walsh[:, 0] + 0.25,
                             walsh[:, 0] + 2.0**-30 * walsh[:, 1], walsh[:, 2]))
        return X, y, 2.0**-40
    if name == "wide_small_positive":
        X = np.random.RandomState(110221).randint(-4, 5, size=(8, 12)).astype(float) / 4
        return X, np.resize(np.array([-1.0, 1.0]), 8), 2.0**-60
    if name == "constant_positive":
        X = np.column_stack((walsh[:, :2], np.full(16, 2.0**20)))
        # An unbalanced target also exercises the unpenalized intercept.
        return X, np.r_[np.ones(10), -np.ones(6)], 0.05
    if name == "zero_duplicate":
        return np.column_stack((walsh, walsh[:, 0])), y, 0.0
    raise AssertionError(name)


def branch_case(branch):
    y = np.array([0, 1, 0, 1])
    if branch == "pi":
        X = np.ones((4, 1))
        lambd = 2.0**-54
        shift = 2.0**-53
        root_half = np.sqrt(0.5)
        U = np.array([[root_half, root_half], [-root_half, root_half]])
        D = np.array([-shift, 8.0])
    else:
        # An approximate eigensystem of this represented constant-column Gram:
        # the intercept component of its normalized null vector rounds to1.
        X = np.full((4, 1), 2.0**30)
        lambd, shift = 0.5, 1.0
        U = np.array([[1.0, 2.0**-30], [-2.0**-30, 1.0]])
        D = np.array([0.0, 2.0**62])
    assert np.isfinite(U).all() and np.isfinite(D).all()
    assert_allclose(U.T @ U, np.eye(2), rtol=0, atol=4*np.finfo(float).eps)
    return X, y, lambd, shift, U, D


def ordinary():
    i = np.arange(8)
    X = np.column_stack((1.0-2.0*(i & 1), 1.0-2.0*((i >> 1) & 1)))
    return X + [0.25, -0.125], (i & 1)
