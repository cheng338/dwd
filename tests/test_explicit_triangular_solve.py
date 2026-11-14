"""Triangular-solve regressions on independent represented equations.

These focused cases preserve the validated solver and fault contracts without
campaign environment switches. Private SPD fixtures are not claimed to be
reachable GenDWD designs, convergence proofs, or performance measurements.
"""
from fractions import Fraction

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import dwd._explicit_factor as factor
from test_explicit_linear_system import _REFUSAL

EPS = np.finfo(np.float64).eps


@pytest.fixture(autouse=True)
def numpy_policy():
    with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
        yield


def exact_residual(P, rhs, solution):
    """Independent 128*d*eps allowance, not the production 64*d*eps target."""
    values = [Fraction(float(v)) for v in solution]
    ratios = []
    for row, target in zip(P, rhs):
        target = Fraction(float(target))
        products = [Fraction(float(a))*b for a, b in zip(row, values) if a != 0]
        residual = abs(sum(products, Fraction())-target)
        denominator = sum(map(abs, products), Fraction())+abs(target)
        assert denominator or not residual
        ratios.append(float(residual/denominator) if denominator else 0.)
    assert max(ratios, default=0.) <= 128*len(P)*EPS
    return ratios


def system(dimension):
    # Strictly SPD, exactly represented tridiagonal matrix.  Fraction constructs
    # the RHS independently; cancellation spans the former 128-row boundary.
    P = np.diag(np.resize(np.array([2., 4., 8., 16.]), dimension))
    rows = np.arange(dimension-1)
    P[rows, rows+1] = P[rows+1, rows] = .125
    x = np.resize(np.array([1., -.5, .25, -2.]), dimension)
    rhs = np.array([float(sum((Fraction(float(a))*Fraction(float(b))
                              for a, b in zip(row, x) if a), Fraction())) for row in P])
    return P, rhs, x


def replace_action(monkeypatch, instance, transform):
    """Inject at the second actual triangular primitive after construction."""
    calls = []
    original = instance._trsv

    def triangular(*args, **kwargs):
        value = original(*args, **kwargs)
        if kwargs["trans"] == 1:
            value = np.asarray(value).copy()
            calls.append(value.copy())
            return transform(value, len(calls))
        return value

    monkeypatch.setattr(instance, "_trsv", triangular)
    return calls


@pytest.mark.parametrize('dimension', (4, 127, 128, 129, 785))
@pytest.mark.parametrize('zero', (False, True))
def test_original_equations_across_residual_block_boundary(dimension, zero):
    P, rhs, expected = system(dimension)
    if zero:
        rhs.fill(0.)
        expected.fill(0.)
    saved_P, saved_rhs = P.copy(), rhs.copy()
    solver = factor.ExplicitFactor(P)
    answer = solver.solve(rhs)
    exact_residual(saved_P, saved_rhs, answer)
    assert_allclose(answer, expected, rtol=512*EPS, atol=512*EPS)
    assert_array_equal(P, saved_P)
    assert_array_equal(rhs, saved_rhs)


@pytest.mark.parametrize('row', (0, 127, 128, 784))
def test_corrupt_action_at_each_boundary_is_not_hidden(monkeypatch, row):
    P, rhs, _ = system(785)
    solver = factor.ExplicitFactor(P)
    def corrupt(value, call):
        value[row] += 1.
        return value
    calls = replace_action(monkeypatch, solver, corrupt)
    with pytest.raises(_REFUSAL):
        solver.solve(rhs)
    assert len(calls) == 1


@pytest.mark.parametrize('fault', ('negative', 'zero', 'nan', 'inf', 'scaled', 'mixed', 'one_coordinate'))
def test_first_triangular_action_corruption_is_rejected(monkeypatch, fault):
    P, rhs, _ = system(5)
    solver = factor.ExplicitFactor(P)
    original = solver._trsv
    stages = []
    def changed(*args, **kwargs):
        value = np.asarray(original(*args, **kwargs)).copy()
        stages.append(kwargs['trans'])
        if kwargs['trans'] == 0:
            if fault == 'negative':
                value *= -1
            elif fault == 'zero':
                value[:] = 0.
            elif fault == 'nan':
                value[0] = np.nan
            elif fault == 'inf':
                value[0] = np.inf
            elif fault == 'scaled':
                value *= 1.5
            elif fault == 'mixed':
                value += .25*np.roll(value, 1)
            else:
                value[0] += 1.
        return value
    monkeypatch.setattr(solver, '_trsv', changed)
    with pytest.raises(_REFUSAL):
        solver.solve(rhs)
    assert stages == [0, 1]


@pytest.mark.parametrize('layout', ('ordinary', 'readonly', 'strided'))
def test_triangular_orientation_and_private_scratch_preserve_factor_and_rhs(monkeypatch, layout):
    P, rhs, _ = system(5)
    if layout == 'readonly':
        rhs.flags.writeable = False
    elif layout == 'strided':
        owner = np.zeros(2*len(rhs))
        owner[::2] = rhs
        rhs = owner[::2]
    saved_P, saved_rhs = P.copy(), rhs.copy()
    rhs_storage = (rhs.flags.writeable, rhs.strides)
    solver = factor.ExplicitFactor(P)
    assert solver.factor[0].flags.f_contiguous
    factor_before = solver.factor[0].tobytes(order='A')
    original = solver._trsv
    stages = []
    def observed(matrix, scratch, **kwargs):
        assert matrix is solver.factor[0]
        assert not np.shares_memory(scratch, rhs)
        assert kwargs['lower'] is True and kwargs['diag'] == 0 and kwargs['overwrite_x'] is True
        stages.append(kwargs['trans'])
        return original(matrix, scratch, **kwargs)
    monkeypatch.setattr(solver, '_trsv', observed)
    for _ in range(2):
        exact_residual(P, rhs, solver.solve(rhs))
    assert stages == [0, 1, 0, 1]
    assert solver.factor[0].tobytes(order='A') == factor_before
    assert_array_equal(P, saved_P)
    assert_array_equal(rhs, saved_rhs)
    assert (rhs.flags.writeable, rhs.strides) == rhs_storage
