"""Explicit-factor fault, equation, hook and input-preservation regressions.

Independent Fraction utilities come from test_explicit_linear_system. Faults
intercept the triangular kernel selected by the actual factor implementation;
no package hooks or arithmetic policies escape each monkeypatch context.
"""
import numpy as np
from numpy.testing import assert_allclose
import pytest
import math

import dwd._explicit_factor as factor
import dwd.gen_dwd as gdwd
from test_explicit_linear_system import (
    _EPS, _REFUSAL, _assert_descent, _assert_equation, _assert_unchanged,
    _equation_exact, _f, _ordinary, _representation, _snapshot,
    _solve, _solve_fraction, _structured,
)


@pytest.fixture(autouse=True)
def _local_numpy_policy():
    with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
        yield


def _wrap_triangular_kernel(monkeypatch, observe):
    """Wrap the selected BLAS primitive, including calls made by public fits."""
    original_selection = factor.get_blas_funcs

    def selected(*args, **kwargs):
        assert args[0] == ("trsv",)
        original, = original_selection(*args, **kwargs)

        def triangular(*action_args, **action_kwargs):
            return observe(original, *action_args, **action_kwargs)

        return (triangular,)

    monkeypatch.setattr(factor, "get_blas_funcs", selected)


@pytest.mark.parametrize("fault", ["negative", "zero", "nan", "inf", "mixed_coordinates", "scaled"])
@pytest.mark.parametrize("kind", ["dense", "csr"])
@pytest.mark.parametrize("mode", ["schur"])
def test_corrupt_factor_solution_refuses_before_objective_update(monkeypatch, fault, kind, mode):
    dense, y = _ordinary()
    X = _representation(dense, kind)
    before = _snapshot(X)
    original_objective = gdwd.dwd_obj
    solved_actions, objective_calls = [], []

    def corrupt(original_solve, *args, **kwargs):
        good = original_solve(*args, **kwargs)
        if kwargs["trans"] == 0:
            return good
        bad = np.asarray(good).copy()
        solved_actions.append(bad.copy())
        assert np.any(bad != 0), "Fault fixture must exercise a nonzero RHS"
        if fault == "negative":
            bad = -bad
        elif fault == "zero":
            bad.fill(0.0)
        elif fault == "nan":
            bad[0] = np.nan
        elif fault == "inf":
            bad[0] = np.inf
        elif fault == "mixed_coordinates":
            bad += 0.25 * np.roll(bad, 1)
        else:
            bad *= 1.5
        return bad

    def objective(*args, **kwargs):
        objective_calls.append(True)
        return original_objective(*args, **kwargs)

    # Corrupt the second triangular action, before the original-system guard.
    _wrap_triangular_kernel(monkeypatch, corrupt)
    monkeypatch.setattr(gdwd, "dwd_obj", objective)
    with pytest.raises(_REFUSAL):
        _solve(X, y, 0.1, mode=mode, steps=3)
    assert len(solved_actions) == 1
    assert len(objective_calls) == 1  # Only the initial objective: solve guard caught it.
    _assert_unchanged(X, before)


@pytest.mark.parametrize("kind", ["dense", "csr"])
@pytest.mark.parametrize("mode", ["schur"])
@pytest.mark.parametrize("stationary", [False, True])
def test_fixed3_reuses_one_factor_and_checks_each_original_rhs(monkeypatch, kind, mode, stationary):
    dense, y = _ordinary()
    if stationary:
        # The fourth Walsh column is orthogonal to the intercept and all three
        # feature columns, giving exactly zero gradient at zero initialization.
        y = 1.0 - 2.0 * ((np.arange(len(dense)) >> 3) & 1)
    X = _representation(dense, kind)
    before = _snapshot(X)
    original_factor = factor.cho_factor
    original_action = factor.ExplicitFactor.solve
    factorizations, triangular_rhs, actions = [], [], []

    def factor_spy(*args, **kwargs):
        factorizations.append(np.asarray(args[0]).copy())
        return original_factor(*args, **kwargs)

    def triangular_spy(original_solve, *args, **kwargs):
        triangular_rhs.append((kwargs["trans"], np.asarray(args[1]).copy()))
        return original_solve(*args, **kwargs)

    def action_spy(self, rhs):
        original_P = np.asarray(self.P).copy()
        original_rhs = np.asarray(rhs).copy()
        solution = original_action(self, rhs)
        # Validate the actual original stored matrix, independently of its
        # equilibrated factor, using exact binary64 inputs to this equation.
        P_f = [[_f(value) for value in row] for row in original_P]
        rhs_f = [_f(value) for value in original_rhs]
        _assert_equation(P_f, rhs_f, solution)
        actions.append((original_P, original_rhs, np.asarray(solution).copy()))
        return solution

    def no_inverse(*args, **kwargs):
        raise AssertionError("The ordinary factor path must not construct an inverse")

    monkeypatch.setattr(factor, "cho_factor", factor_spy)
    _wrap_triangular_kernel(monkeypatch, triangular_spy)
    monkeypatch.setattr(factor.ExplicitFactor, "solve", action_spy)
    monkeypatch.setattr(np.linalg, "inv", no_inverse)
    result = _solve(X, y, 0.1, mode=mode, steps=3)
    assert len(factorizations) == 1
    assert len(actions) == 3
    assert [stage for stage, _ in triangular_rhs] == [0, 1] * 3
    assert len(result[2]) == 4

    theta = np.zeros(dense.shape[1] + 1)
    for stored_P, rhs, solved in actions:
        P_f, step_rhs = _equation_exact(dense, y, theta[1:], theta[0], 0.1)
        delta = (len(dense) / 4.0) * solved  # Unchanged q=1 MM scaling.
        expected = _solve_fraction(P_f, step_rhs)
        _assert_equation(P_f, step_rhs, delta)
        assert_allclose(delta, expected, rtol=512 * _EPS, atol=512 * _EPS)
        gradient = np.array([float(value) * 4.0 / len(dense) for value in step_rhs])
        assert_allclose(rhs, gradient, rtol=512 * _EPS, atol=512 * _EPS)
        theta -= delta
    assert_allclose(np.r_[result[1], result[0]], theta, rtol=512 * _EPS, atol=512 * _EPS)
    _assert_descent(dense, y, result, 0.1)
    _assert_unchanged(X, before)


@pytest.mark.parametrize("structure", ["duplicate", "constant", "onehot"])
@pytest.mark.parametrize("kind", ["dense", "csr"])
@pytest.mark.parametrize("mode", ["schur"])
def test_natural_small_ridge_factor_outcomes_use_independent_equations(structure, kind, mode, record_property):
    dense, y = _structured(structure)
    X = _representation(dense, kind)
    before = _snapshot(X)
    penalty = 1e-12
    # All six dense and CSR cases must solve the original equation and descend
    # on the explicit Schur path, even with this small ridge penalty.
    one = _solve(X, y, penalty, mode=mode, steps=1)
    record_property("factor_one_step", "accepted")
    assert len(one[2]) == 2
    P, rhs = _equation_exact(dense, y, np.zeros(dense.shape[1]), 0.0, penalty)
    _assert_equation(P, rhs, -np.r_[one[1], one[0]])
    _assert_descent(dense, y, one, penalty)
    result = _solve(X, y, penalty, mode=mode, steps=8)
    record_property("factor_eight_step", "accepted")
    assert len(result[2]) == 9
    _assert_descent(dense, y, result, penalty)
    _assert_unchanged(X, before)


@pytest.mark.parametrize("fault", ["unchanged", "zero_first", "double_first"])
def test_private_factor_extreme_scales_cannot_hide_original_equation_error(monkeypatch, fault, record_property):
    """Private-helper robustness: not asserted to be a reachable free-intercept P.

    True x=(2^-700, 2^500). Scaling every component by max(abs(x)) loses
    the first component and its normalized RHS, potentially hiding 0/0.
    The exact original equation remains decisive even if a practical helper
    legitimately refuses such an extreme scaling regime.
    """
    P = np.diag([math.ldexp(1.0, 1000), math.ldexp(1.0, -1000)])
    rhs = np.array([math.ldexp(1.0, 300), math.ldexp(1.0, -500)])
    calls = []

    def changed(original_solve, *args, **kwargs):
        solved = original_solve(*args, **kwargs)
        if kwargs["trans"] == 0:
            return solved
        calls.append(np.asarray(solved).copy())
        solved = np.asarray(solved).copy()
        if fault == "zero_first":
            solved[0] = 0.0
        elif fault == "double_first":
            solved[0] *= 2.0
        return solved

    _wrap_triangular_kernel(monkeypatch, changed)
    try:
        solution = factor.ExplicitFactor(P).solve(rhs)
    except _REFUSAL as exc:
        record_property("private_extreme_action", type(exc).__name__ + ": " + str(exc))
    else:
        record_property("private_extreme_action", "accepted")
        _assert_equation([[_f(value) for value in row] for row in P],
                         [_f(value) for value in rhs], solution)
    assert len(calls) == 1


@pytest.mark.parametrize("first_multiplier", [1.0, 9.0 / 8.0, 0.0])
def test_private_factor_nonzero_subnormal_denominator_cannot_hide_quantization(monkeypatch, first_multiplier, record_property):
    """Private-helper robustness, not an observed reachable GenDWD fit system.

    True x=(2^-573, 2^500); normalized first x and RHS are 2^-1073.
    Multiplying the true first component by 9/8 leaves its normalized binary64
    value at the same subnormal after rounding. Thus a nonzero denominator can
    still hide a 12.5% original-equation error; the exact input equation is the
    oracle, and explicit refusal is allowed for this unresolved scaling regime.
    """
    P = np.diag([math.ldexp(1.0, 1000), math.ldexp(1.0, -1000)])
    rhs = np.array([math.ldexp(1.0, 427), math.ldexp(1.0, -500)])
    calls = []

    def changed(original_solve, *args, **kwargs):
        solved = np.asarray(original_solve(*args, **kwargs)).copy()
        if kwargs["trans"] == 0:
            return solved
        calls.append(solved.copy())
        solved[0] *= first_multiplier
        return solved

    _wrap_triangular_kernel(monkeypatch, changed)
    try:
        solution = factor.ExplicitFactor(P).solve(rhs)
    except _REFUSAL as exc:
        record_property("private_subnormal_action", type(exc).__name__ + ": " + str(exc))
    else:
        record_property("private_subnormal_action", "accepted")
        _assert_equation([[_f(value) for value in row] for row in P],
                         [_f(value) for value in rhs], solution)
    assert len(calls) == 1


def test_private_factor_first_rhs_division_cannot_erase_error_before_rescaling(monkeypatch, record_property):
    """Private SPD action fixture; no claim of reachable GenDWD fit inputs.

    The first RHS division underflows, but a subsequent division by the tiny
    solution scale would magnify it. The final denominator is normal, so a
    final-denominator-only check misses this original componentwise error.
    """
    huge, coupling, small = (math.ldexp(1.0, power) for power in (1000, 400, -500))
    P = np.array([[huge, coupling, coupling], [coupling, 1.0, 0.0], [coupling, 0.0, 1.0]])
    rhs = np.array([math.ldexp(1.0, -100), small, -small])
    bad_solution = np.array([0.0, small, -small])
    P_f = [[_f(value) for value in row] for row in P]
    rhs_f = [_f(value) for value in rhs]

    def componentwise_ratios(solution):
        values = [_f(value) for value in solution]
        for row, target in zip(P_f, rhs_f):
            products = [a * b for a, b in zip(row, values)]
            residual = abs(sum(products, _f(0)) - target)
            denominator = sum(map(abs, products), _f(0)) + abs(target)
            yield residual / denominator if denominator else residual

    assert next(componentwise_ratios(bad_solution)) == _f(1) / 3
    # The supplied solution is in equilibrated coordinates here; since its
    # first component is zero and the other diagonal scales are one, the
    # desired wrong original solution has exactly the same stored vector.
    calls = []
    def changed(original_solve, *args, **kwargs):
        if kwargs["trans"] == 0:
            return original_solve(*args, **kwargs)
        calls.append(True)
        return bad_solution.copy()

    _wrap_triangular_kernel(monkeypatch, changed)
    try:
        solution = factor.ExplicitFactor(P).solve(rhs)
    except _REFUSAL as exc:
        record_property("private_staged_rhs_action", type(exc).__name__ + ": " + str(exc))
    else:
        record_property("private_staged_rhs_action", "accepted")
        assert all(float(ratio) <= 128 * len(P) * _EPS
                   for ratio in componentwise_ratios(solution))
    assert len(calls) == 1
