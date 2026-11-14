"""Focused F-6 contract checks; independent equations, not a proposed remedy.

Run normally under pytest after selecting one isolated source tree externally.
The historical wide cases need not ascend on every BLAS implementation: a
structurally singular unregularized explicit inverse must be refused. A tiny
positive ridge may be lost in forming the represented Gram matrix; those cases
may refuse or return a numerically valid descent step. Exact diagonal, scaled
SPD controls must remain usable. No pseudoinverse, ridge, or solver policy is
introduced by these tests. Existing sparse-score/cosine regressions stay separate.
"""
from fractions import Fraction
import math

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError

from dwd.gen_dwd import GenDWD, GenDWDCV, solve_gen_dwd


_REFUSAL = (ValueError, np.linalg.LinAlgError, FloatingPointError)
_EPS = np.finfo(np.float64).eps
_WIDE = [(12, 20, seed) for seed in (20003, 12050, 12051, 12052)] + [
    (20, 30, 20003), (25, 40, 7), (12, 50, 3)]


@pytest.fixture(autouse=True)
def _local_numpy_policy():
    # The separate F-4 nextafter-underflow issue is outside this finding.
    with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
        yield


def _f(value):
    return Fraction(float(value))


def _labels(n):
    return np.resize(np.array([-1.0, 1.0]), n)


def _representation(X, kind):
    return X.copy() if kind == "dense" else sparse.csr_matrix(X)


def _snapshot(X):
    if sparse.issparse(X):
        return tuple((a.copy(), a.flags.writeable) for a in
                     (X.data, X.indices, X.indptr))
    return ((X.copy(), X.flags.writeable),)


def _assert_unchanged(X, before):
    arrays = (X.data, X.indices, X.indptr) if sparse.issparse(X) else (X,)
    for actual, (saved, writable) in zip(arrays, before):
        assert_array_equal(actual, saved)
        assert actual.flags.writeable == writable


def _orthogonal(scale=1.0):
    # Four independent Walsh columns including the intercept, all dyadic.
    i = np.arange(16)
    X = np.column_stack([1.0 - 2.0 * ((i >> bit) & 1) for bit in (0, 1, 2)])
    return X * scale, np.where(X[:, 0] > 0, 1.0, -1.0)


def _ordinary():
    X, y = _orthogonal()
    return X + np.array([0.25, -0.5, 0.125]), y


def _structured(kind):
    X, y = _orthogonal()
    if kind == "duplicate":
        return np.column_stack((X, X[:, 0])), y
    if kind == "constant":
        return np.column_stack((X, np.ones(len(X)))), y
    # All one-hot columns sum to the intercept: tall but augmented-rank deficient.
    return np.eye(4)[np.arange(16) % 4], y


def _q_parts(margin, q):
    q = Fraction(q)
    cutoff = q / (q + 1)
    if margin <= cutoff:
        return 1 - margin, Fraction(-1)
    return (cutoff / margin) ** int(q) / (q + 1), -(cutoff / margin) ** int(q + 1)


def _objective_exact(X, y, beta, offset, penalty, q=1):
    rows = [[_f(v) for v in row] for row in np.asarray(X)]
    beta_f, b = list(map(_f, beta)), _f(offset)
    losses = [_q_parts(_f(yi) * (b + sum((a * c for a, c in zip(row, beta_f)),
                                        Fraction())), q)[0]
              for row, yi in zip(rows, y)]
    return sum(losses, Fraction()) / len(rows) + _f(penalty) * sum(
        (v * v for v in beta_f), Fraction())


def _equation_exact(X, y, beta, offset, penalty, q=1):
    """P and rhs of P delta = (n/M) gradient, in exact input arithmetic."""
    n, p = X.shape
    A = [[Fraction(1)] + list(map(_f, row)) for row in X]
    theta = [_f(offset)] + list(map(_f, beta))
    M = Fraction(q) + 2 + Fraction(1, q)
    shift = 2 * n * _f(penalty) / M
    P = [[sum((row[j] * row[k] for row in A), Fraction())
          + (shift if j == k and j else 0) for k in range(p + 1)]
         for j in range(p + 1)]
    z = [_f(yi) * _q_parts(_f(yi) * sum((a * b for a, b in zip(row, theta)),
                                      Fraction()), q)[1] / n
         for row, yi in zip(A, y)]
    gradient = [sum((row[j] * zi for row, zi in zip(A, z)), Fraction())
                + (2 * _f(penalty) * theta[j] if j else 0)
                for j in range(p + 1)]
    return P, [Fraction(n, 1) * value / M for value in gradient]


def _solve_fraction(P, rhs):
    # Tiny (4-by-4) independent Gaussian elimination; never a package inverse.
    rows = [list(row) + [value] for row, value in zip(P, rhs)]
    n = len(rows)
    for j in range(n):
        pivot = next(i for i in range(j, n) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scale = rows[j][j]
        rows[j] = [value / scale for value in rows[j]]
        for i in range(n):
            if i != j:
                scale = rows[i][j]
                rows[i] = [a - scale * b for a, b in zip(rows[i], rows[j])]
    return np.array([float(row[-1]) for row in rows])


def _assert_equation(P, rhs, delta):
    # A backward-error check scaled by actual equation terms, not a condition
    # number cutoff or a changed solver acceptance tolerance.
    delta_f = list(map(_f, delta))
    residual = [sum((a * b for a, b in zip(row, delta_f)), Fraction()) - target
                for row, target in zip(P, rhs)]
    scale = max((sum((abs(a * b) for a, b in zip(row, delta_f)), Fraction())
                 + abs(target) for row, target in zip(P, rhs)), default=Fraction())
    maximum = max(map(abs, residual), default=Fraction())
    if not scale:
        assert not maximum
    else:
        assert float(maximum / scale) <= 128 * len(P) * _EPS


def _assert_descent(X, y, result, penalty, beta0=None, offset0=0.0, q=1):
    beta, offset, history, _ = result
    beta0 = np.zeros(X.shape[1]) if beta0 is None else beta0
    history = np.asarray(history)
    assert np.isfinite(beta).all() and np.isfinite(offset) and np.isfinite(history).all()
    budget = 128 * len(X) * _EPS * max(1.0, float(np.max(np.abs(history))))
    assert np.all(np.diff(history) <= budget)
    first = float(_objective_exact(X, y, beta0, offset0, penalty, q))
    last = float(_objective_exact(X, y, beta, offset, penalty, q))
    assert last <= first + 128 * len(X) * _EPS * max(1.0, abs(first), abs(last))
    # Compare reported final objective with the independent exact-margin value.
    assert abs(history[-1] - last) <= 256 * len(X) * _EPS * max(1.0, abs(last))


def _solve(X, y, penalty, *, implicit=False, mode="schur", steps=8,
           beta0=None, offset0=0.0, q=1, stopping="fixed"):
    return solve_gen_dwd(X, y, penalty, q=q, implicit_P=implicit,
                         initialization="zero", stopping=stopping,
                         obj_tol=0.0, max_iter=steps, beta_init=beta0,
                         offset_init=offset0)


@pytest.mark.parametrize("shape", _WIDE)
@pytest.mark.parametrize("kind", ["dense", "csr"])
@pytest.mark.parametrize("mode", ["schur"])
def test_wide_unregularized_explicit_system_refuses(shape, kind, mode):
    n, p, seed = shape
    X = _representation(np.random.RandomState(seed).normal(size=(n, p)), kind)
    before = _snapshot(X)
    with pytest.raises(ValueError, match="zero-shift augmented design"):
        _solve(X, _labels(n), 0.0, mode=mode)
    _assert_unchanged(X, before)


@pytest.mark.parametrize("seed", [20003, 12050, 12051, 12052])
@pytest.mark.parametrize("kind", ["dense", "csr"])
def test_original_implicit_refusal_contract(seed, kind):
    X = _representation(np.random.RandomState(seed).normal(size=(12, 20)), kind)
    with pytest.raises(_REFUSAL):
        _solve(X, _labels(12), 0.0, implicit=True)


@pytest.mark.parametrize("shape", _WIDE)
@pytest.mark.parametrize("kind", ["dense", "csr"])
def test_tiny_ridge_wide_system_refuses_or_validates_accepted_step(shape, kind, record_property):
    n, p, seed = shape
    dense = np.random.RandomState(seed).normal(size=(n, p))
    X, y = _representation(dense, kind), _labels(n)
    before = _snapshot(X)
    try:
        one = _solve(X, y, 1e-18, steps=1)
    except _REFUSAL as exc:
        record_property("ill_conditioned_one_step", type(exc).__name__ + ": " + str(exc))
    else:
        record_property("ill_conditioned_one_step", "accepted")
        P, rhs = _equation_exact(dense, y, np.zeros(p), 0.0, 1e-18)
        _assert_equation(P, rhs, -np.r_[one[1], one[0]])
        _assert_descent(dense, y, one, 1e-18)
    try:
        result = _solve(X, y, 1e-18)
    except _REFUSAL as exc:
        record_property("ill_conditioned_eight_step", type(exc).__name__ + ": " + str(exc))
    else:
        record_property("ill_conditioned_eight_step", "accepted")
        _assert_descent(dense, y, result, 1e-18)
    _assert_unchanged(X, before)


@pytest.mark.parametrize("structure", ["duplicate", "constant", "onehot"])
@pytest.mark.parametrize("kind", ["dense", "csr"])
@pytest.mark.parametrize("mode", ["schur"])
def test_tall_exactly_rank_deficient_explicit_system_refuses(structure, kind, mode):
    dense, y = _structured(structure)
    X = _representation(dense, kind)
    before = _snapshot(X)
    with pytest.raises(ValueError, match="zero-shift augmented design"):
        _solve(X, y, 0.0, mode=mode)
    _assert_unchanged(X, before)


@pytest.mark.parametrize("structure", ["duplicate", "constant", "onehot"])
@pytest.mark.parametrize("kind", ["dense", "csr"])
@pytest.mark.parametrize("penalty", [1e-12, 0.1])
def test_ridge_resolves_structural_deficiency_or_refuses_unreliable_inverse(structure, kind, penalty, record_property):
    dense, y = _structured(structure)
    X = _representation(dense, kind)
    try:
        one = _solve(X, y, penalty, steps=1)
    except _REFUSAL as exc:
        if penalty == 0.1:
            raise
        record_property("small_ridge_one_step", type(exc).__name__ + ": " + str(exc))
    else:
        record_property("small_ridge_one_step", "accepted")
        P, rhs = _equation_exact(dense, y, np.zeros(dense.shape[1]), 0.0, penalty)
        _assert_equation(P, rhs, -np.r_[one[1], one[0]])
        _assert_descent(dense, y, one, penalty)
    try:
        result = _solve(X, y, penalty)
    except _REFUSAL as exc:
        if penalty == 0.1:
            raise
        record_property("small_ridge_eight_step", type(exc).__name__ + ": " + str(exc))
    else:
        record_property("small_ridge_eight_step", "accepted")
        _assert_descent(dense, y, result, penalty)


@pytest.mark.parametrize("kind", ["dense", "csr"])
@pytest.mark.parametrize("penalty", [0.0, 1e-12, 0.1])
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("q", [1, 2])
def test_full_rank_step_matches_fraction_system(kind, penalty, implicit, q):
    dense, y = _ordinary()
    X = _representation(dense, kind)
    beta0, offset0 = np.array([0.125, -0.25, 0.0625]), -0.125
    before = _snapshot(X)
    P, rhs = _equation_exact(dense, y, beta0, offset0, penalty, q)
    expected = _solve_fraction(P, rhs)
    result = _solve(X, y, penalty, implicit=implicit, steps=1,
                    beta0=beta0, offset0=offset0, q=q)
    delta = np.r_[offset0 - result[1], beta0 - result[0]]
    _assert_equation(P, rhs, delta)
    # These predeclared tiny full-rank fixtures have modest conditioning.
    absolute = 128 * len(P) * _EPS * max(1.0, float(np.max(np.abs(expected))))
    assert_allclose(delta, expected, rtol=128 * len(P) * _EPS, atol=absolute)
    _assert_descent(dense, y, result, penalty, beta0, offset0, q)
    _assert_unchanged(X, before)
    assert_array_equal(beta0, [0.125, -0.25, 0.0625])


@pytest.mark.parametrize("power", [-20, 0, 20])
@pytest.mark.parametrize("kind", ["dense", "csr"])
@pytest.mark.parametrize("implicit", [False, True])
def test_scaled_diagonal_spd_is_not_singular_merely_for_small_entries(power, kind, implicit):
    scale = math.ldexp(1.0, power)
    dense, y = _orthogonal(scale)
    result = _solve(_representation(dense, kind), y, 0.0, implicit=implicit, steps=1)
    # P is exactly diagonal; at zero, the exact update is beta[0]=1/(4*scale).
    assert_allclose(result[0] * scale, [0.25, 0.0, 0.0], rtol=512 * _EPS, atol=512 * _EPS)
    assert abs(result[1]) <= 512 * _EPS
    _assert_descent(dense, y, result, 0.0)


@pytest.mark.parametrize("power", [-12, -26])
@pytest.mark.parametrize("penalty", [0.0, 1e-12, 0.1])
@pytest.mark.parametrize("kind", ["dense", "csr"])
def test_nearly_dependent_columns_do_not_return_unchecked_steps(power, penalty, kind, record_property):
    base, y = _orthogonal()
    dense = np.column_stack((base[:, 0], base[:, 0] + math.ldexp(1.0, power) * base[:, 1]))
    try:
        one = _solve(_representation(dense, kind), y, penalty, steps=1)
    except _REFUSAL as exc:
        # At 2^-26 the squared distinguishing term can disappear from X.T X.
        # Even the 2^-12 case can have an unreliable explicitly formed inverse;
        # mathematical SPD alone does not guarantee inverse admissibility.
        # The ordinary-ridge case is well conditioned and must remain usable.
        if penalty == 0.1:
            raise
        record_property("near_singular_one_step", type(exc).__name__ + ": " + str(exc))
    else:
        record_property("near_singular_one_step", "accepted")
        P, rhs = _equation_exact(dense, y, np.zeros(2), 0.0, penalty)
        _assert_equation(P, rhs, -np.r_[one[1], one[0]])
        _assert_descent(dense, y, one, penalty)
    try:
        result = _solve(_representation(dense, kind), y, penalty)
    except _REFUSAL as exc:
        if penalty == 0.1:
            raise
        record_property("near_singular_eight_step", type(exc).__name__ + ": " + str(exc))
    else:
        record_property("near_singular_eight_step", "accepted")
        _assert_descent(dense, y, result, penalty)


@pytest.mark.parametrize("kind", ["dense", "csr"])
def test_positive_lambda_with_zero_represented_shift_does_not_admit_wide_inverse(kind):
    penalty, q = float.fromhex("0x0.0000000000001p-1022"), 128
    dense = np.random.RandomState(20003).normal(size=(12, 20))
    M = float(q) + 2.0 + 1.0 / q
    assert penalty > 0.0 and (2.0 * len(dense) * penalty) / M == 0.0
    with pytest.raises(_REFUSAL):
        _solve(_representation(dense, kind), _labels(len(dense)), penalty, q=q)


@pytest.mark.parametrize("kind", ["dense", "csr"])
@pytest.mark.parametrize("power", [-20, 0, 20])
def test_zero_represented_shift_keeps_scaled_spd_control_valid(kind, power):
    penalty, q = float.fromhex("0x0.0000000000001p-1022"), 128
    scale = math.ldexp(1.0, power)
    dense, y = _orthogonal(scale)
    M = float(q) + 2.0 + 1.0 / q
    assert penalty > 0.0 and (2.0 * len(dense) * penalty) / M == 0.0
    result = _solve(_representation(dense, kind), y, penalty, q=q, steps=1)
    assert_allclose(result[0] * scale, [1.0 / M, 0.0, 0.0],
                    rtol=512 * _EPS, atol=512 * _EPS)
    assert abs(result[1]) <= 512 * _EPS
    P, rhs = _equation_exact(dense, y, np.zeros(dense.shape[1]), 0.0, penalty, q)
    _assert_equation(P, rhs, -np.r_[result[1], result[0]])
    _assert_descent(dense, y, result, penalty, q=q)


@pytest.mark.parametrize("kind", ["dense", "csr"])
@pytest.mark.parametrize("refit", [False, True])
def test_public_singular_fit_clears_learned_state_without_mutating_inputs(kind, refit):
    model = GenDWD(lambd=0.1, implicit_P=False, max_iter=2, stopping="fixed")
    if refit:
        good, good_y = _ordinary()
        model.fit(_representation(good, kind), good_y)
        assert hasattr(model, "coef_")
    dense, y = _structured("constant")
    X = _representation(dense, kind)
    if sparse.issparse(X):
        for array in (X.data, X.indices, X.indptr):
            array.flags.writeable = False
    else:
        X.flags.writeable = False
    before = _snapshot(X)
    model.set_params(lambd=0.0)
    with pytest.raises(_REFUSAL):
        model.fit(X, y)
    assert not [name for name in vars(model) if name.endswith("_") and not name.startswith("_")]
    with pytest.raises(NotFittedError):
        model.decision_function(X)
    _assert_unchanged(X, before)


@pytest.mark.parametrize("stopping", ["fixed", "objective", "optimality"])
@pytest.mark.parametrize("implicit", [False, True])
def test_existing_stopping_and_initialization_contracts(stopping, implicit):
    X, y = _ordinary()
    model = GenDWD(lambd=0.1, implicit_P=implicit, max_iter=3,
                   stopping=stopping, initialization="zero", obj_tol=0.0, tol=0.0).fit(X, y)
    assert model.n_iter_ <= 3
    assert model.initialization_ == "zero"
    assert model.stationarity_checked_
    assert np.isfinite(model.stationarity_residual_)
    _assert_descent(X, y, (model.coef_.ravel(), model.intercept_[0],
                          model.objective_history_, model.C_), 0.1)


@pytest.mark.parametrize("implicit", [False, True])
def test_valid_cv_cache_preserves_ridge_path(implicit):
    X, y = _ordinary()
    model = GenDWD(lambd=0.1, implicit_P=implicit, max_iter=2, stopping="fixed")
    model.cv_init(X)
    assert model._cv_cache_matches(X)
    cache = tuple(array.copy() for array in model._P0_eig)
    model.fit(X, y)
    plain = GenDWD(lambd=0.1, implicit_P=implicit, max_iter=2, stopping="fixed").fit(X, y)
    assert_allclose(model.coef_, plain.coef_, rtol=2e-12, atol=2e-13)
    for saved, actual in zip(cache, model._P0_eig):
        assert_array_equal(saved, actual)


@pytest.mark.parametrize("implicit", [False, True])
def test_tiny_public_cv_preserves_selected_solver_contract(implicit):
    X, y = _ordinary()
    model = GenDWDCV(lambd_vals=[0.05, 0.1], q_vals=[1], implicit_P=implicit,
                     max_iter=2, stopping="fixed", initialization="zero", cv=2).fit(X, y)
    assert len(model.agg_cv_results_["params"]) == 2
    assert model.best_estimator_.implicit_P is implicit
    assert model.predict(X).shape == y.shape


@pytest.mark.parametrize("kind", ["dense", "csr"])
def test_singular_cv_candidate_refuses_and_clears_previous_fit(kind):
    X, y = _ordinary()
    model = GenDWDCV(lambd_vals=[0.1], q_vals=[1], implicit_P=False,
                     max_iter=2, stopping="fixed", cv=2).fit(_representation(X, kind), y)
    assert hasattr(model, "best_estimator_")
    singular, labels = _structured("constant")
    model.set_params(lambd_vals=[0.0])
    with pytest.raises(_REFUSAL):
        model.fit(_representation(singular, kind), labels)
    assert not [name for name in vars(model) if name.endswith("_") and not name.startswith("_")]


@pytest.mark.parametrize("fault", ["negative", "zero", "nan", "inf", "offdiagonal", "scaled"])
def test_corrupt_inverse_cannot_be_accepted_as_an_unchecked_update(monkeypatch, fault, record_property):
    X, y = _ordinary()
    P, rhs = _equation_exact(X, y, np.zeros(3), 0.0, 0.1)
    original = np.linalg.inv
    calls = []

    def corrupt(matrix):
        inverse = original(matrix)
        calls.append(matrix.shape)
        if fault == "negative":
            return -inverse
        if fault == "zero":
            return np.zeros_like(inverse)
        if fault == "nan":
            inverse[0, 0] = np.nan
        elif fault == "inf":
            inverse[0, 0] = np.inf
        elif fault == "offdiagonal":
            inverse[0, 1] += 1.0
        else:
            inverse *= 1.5
        return inverse

    monkeypatch.setattr(np.linalg, "inv", corrupt)
    try:
        result = _solve(X, y, 0.1, steps=1)
    except _REFUSAL:
        record_property("injected_inverse_outcome", "refused")
    else:
        # A future valid implementation need not call np.linalg.inv. The
        # required behavior is correct accepted equations, not a helper name.
        record_property("injected_inverse_outcome", "accepted_checked_step")
        _assert_equation(P, rhs, -np.r_[result[1], result[0]])
        _assert_descent(X, y, result, 0.1)
    record_property("injected_inverse_calls", len(calls))
