"""Shared zero-shift augmented-design rank regression tests.

The pinned pre-change module in tests/fixtures supplies behavioral comparisons.
Exact Fraction and analytic controls provide independent mathematical checks.
"""
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV

import dwd.gen_dwd as linear
import dwd._explicit_factor as factor
from dwd._eigen import validated_eigh


EPS = float(np.finfo(float).eps)
BASELINE_SHA256 = 'e32956c68e0f259fef3d610e2b778529b42e0e324e2c732f3870dd39bcf608ba'


@pytest.fixture(scope='session')
def baseline():
    path = Path(__file__).resolve().parent / 'fixtures' / 'ir21_baseline_gen_dwd.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BASELINE_SHA256
    spec = importlib.util.spec_from_file_location('ir21_baseline_gen_dwd', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Deliberately not registered in sys.modules.
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BASELINE_SHA256
    return module


def walsh():
    rows = np.arange(8)
    X = np.column_stack([1. - 2. * ((rows >> bit) & 1) for bit in range(3)])
    return X, X[:, 0].copy()


def ordinary():
    X, y = walsh()
    return X + np.array([.25, -.5, .125]), y


def structured(name):
    X, y = walsh()
    if name == 'duplicate':
        return np.column_stack((X[:, 0], X[:, 0], X[:, 1])), y
    if name == 'constant':
        return np.column_stack((X[:, 0], np.ones(8))), y
    if name == 'onehot':
        return np.eye(4)[np.arange(8) % 4], y
    if name == 'zero_column':
        return np.column_stack((X[:, 0], np.zeros(8))), y
    if name == 'p_equals_n':
        return np.eye(4), np.array([-1., 1., -1., 1.])
    if name == 'p_exceeds_n':
        return np.column_stack((np.eye(4), np.array([1., 2., 3., 4.]))), np.array([-1., 1., -1., 1.])
    raise AssertionError(name)


def representation(X, kind):
    return X.copy() if kind == 'dense' else sparse.csr_matrix(X)


def snapshot(X):
    arrays = (X.data, X.indices, X.indptr) if sparse.issparse(X) else (X,)
    return [(a.copy(), a.flags.writeable) for a in arrays]


def unchanged(X, saved):
    arrays = (X.data, X.indices, X.indptr) if sparse.issparse(X) else (X,)
    for a, (b, writable) in zip(arrays, saved, strict=True):
        assert_array_equal(a, b)
        assert a.flags.writeable == writable


def fraction_rank(X):
    """Small exact elimination on represented [1,X], independent of LAPACK."""
    rows = [[Fraction(1)] + [Fraction(float(x)) for x in row] for row in X]
    rank = 0
    for column in range(len(rows[0])):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        d = rows[rank][column]
        rows[rank] = [x / d for x in rows[rank]]
        for i in range(rank + 1, len(rows)):
            d = rows[i][column]
            rows[i] = [x - d * y for x, y in zip(rows[i], rows[rank])]
        rank += 1
        if rank == len(rows):
            break
    return rank


def exact_step(X, y, beta, offset, q):
    """Exact zero-penalty augmented normal equations for the selected margins."""
    A = [[Fraction(1)] + [Fraction(float(x)) for x in row] for row in X]
    theta = [Fraction(float(offset))] + [Fraction(float(x)) for x in beta]
    n, d = len(A), len(A[0])
    cutoff = Fraction(q, q + 1)
    margins = [Fraction(float(yi)) * sum((a * b for a, b in zip(row, theta)), Fraction())
               for row, yi in zip(A, y)]
    assert max(margins) <= cutoff  # Independent exact slope is -1 here.
    P = [[sum((row[j] * row[k] for row in A), Fraction()) for k in range(d)] for j in range(d)]
    M = Fraction(q) + 2 + Fraction(1, q)
    rhs = [-sum((row[j] * Fraction(float(yi)) for row, yi in zip(A, y)), Fraction()) / M
           for j in range(d)]
    rows = [list(row) + [value] for row, value in zip(P, rhs)]
    for j in range(d):
        pivot = next(i for i in range(j, d) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scale = rows[j][j]
        rows[j] = [v / scale for v in rows[j]]
        for i in range(d):
            if i != j:
                scale = rows[i][j]
                rows[i] = [a - scale * b for a, b in zip(rows[i], rows[j])]
    return P, rhs, np.array([float(row[-1]) for row in rows])


def assert_equation(P, rhs, delta):
    exact_delta = [Fraction(float(x)) for x in delta]
    residual = [sum((a * b for a, b in zip(row, exact_delta)), Fraction()) - target
                for row, target in zip(P, rhs)]
    scale = max(sum((abs(a * b) for a, b in zip(row, exact_delta)), Fraction()) + abs(target)
                for row, target in zip(P, rhs))
    assert float(max(map(abs, residual)) / scale) <= 256 * len(P) * EPS


def solve(module, X, y, penalty=0., *, implicit=True, mode='schur', steps=1, **kwargs):
    return module.solve_gen_dwd(X, y, penalty, implicit_P=implicit,
                               initialization='zero', stopping='fixed', max_iter=steps,
                               obj_tol=0., **kwargs)


def assert_results_equal(a, b, X):
    for aa, bb in zip(a, b, strict=True):
        assert_allclose(aa, bb, rtol=2e-12, atol=2e-13)
    assert_allclose(X @ a[0] + a[1], X @ b[0] + b[1], rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize('name', ['duplicate', 'constant', 'onehot', 'zero_column', 'p_equals_n', 'p_exceeds_n'])
@pytest.mark.parametrize('kind', ['dense', 'csr'])
@pytest.mark.parametrize('mode', ['schur'])
def test_structural_rank_loss_refuses_implicit_before_eigen_preparation(name, kind, mode, monkeypatch):
    dense, y = structured(name)
    assert fraction_rank(dense) < dense.shape[1] + 1
    X = representation(dense, kind); before = snapshot(X)
    monkeypatch.setattr(linear, 'get_P0_eig', lambda *a, **kw: pytest.fail('Rank loss reached eigen preparation'))
    with pytest.raises(ValueError, match='zero-shift augmented design'):
        solve(linear, X, y, mode=mode, steps=0)
    unchanged(X, before)


def test_fixed_24_seed_duplicate_reproduction_records_original_sign_dependence(baseline, monkeypatch, record_property):
    observations = []
    for seed in range(213000, 213024):
        Z = np.random.default_rng(seed).normal(size=(12, 2))
        X = np.column_stack((Z, Z[:, 0])); y = np.tile([-1., 1.], 6)
        assert_array_equal(X[:, 0], X[:, 2])  # Exact augmented null vector (0,1,0,-1).
        before = X.tobytes(); captured = []
        old_eigen = baseline.get_P0_eig
        def observe(*args, **kwargs):
            U, D = old_eigen(*args, **kwargs)
            captured.append(float(np.min(D)))
            return U, D
        with monkeypatch.context() as patch:
            patch.setattr(baseline, 'get_P0_eig', observe)
            try:
                result = solve(baseline, X, y, steps=0)
            except ValueError as exc:
                assert 'shifted augmented Gram matrix must be positive definite' in str(exc)
                accepted = False
            else:
                assert np.isfinite(result[0]).all() and np.isfinite(result[1])
                accepted = True
        assert len(captured) == 1 and accepted is (captured[0] > 0.)
        with pytest.raises(ValueError, match='zero-shift augmented design'):
            solve(linear, X, y, steps=0)
        assert X.tobytes() == before
        observations.append({'seed': seed, 'baseline_minimum_eigenvalue': captured[0],
                             'baseline_accepted': accepted, 'candidate_rejected': True})
    record_property('seed_reproduction', json.dumps(observations, allow_nan=False))
    assert len(observations) == 24  # No required host-specific accepted/refused split.


@pytest.mark.parametrize('sign', [-1, 0, 1])
@pytest.mark.parametrize('source', ['computed', 'supplied', 'cached'])
def test_null_mode_roundoff_sign_never_changes_design_refusal(sign, source, baseline, monkeypatch):
    X = np.ones((8, 1)); y = np.tile([-1., 1.], 4)
    U = np.array([[1., 1.], [1., -1.]]) / math.sqrt(2.)
    D = np.array([16., sign * EPS])
    P0 = np.full((2, 2), 8.)
    # These small sign perturbations are valid numerical representations;
    # augmented-design rank, not their sign, must settle zero-shift admission.
    validated_eigh(P0, supplied=(U, D))
    before = X.tobytes(), U.tobytes(), D.tobytes()
    def call(module):
        if source == 'computed':
            with monkeypatch.context() as patch:
                patch.setattr(module, 'get_P0_eig', lambda *a, **kw: (U, D))
                return solve(module, X, y, steps=0)
        if source == 'supplied':
            return solve(module, X, y, steps=0, P0_eig=(U, D))
        model = module.GenDWD(lambd=0., max_iter=0, stopping='fixed', initialization='zero')
        model.cv_init(X)
        model._P0_eig = (U, D)
        assert model._cv_cache_matches(X)
        return model.fit(X, y)
    if sign > 0:
        call(baseline)
    else:
        with pytest.raises(ValueError, match='shifted augmented Gram matrix must be positive definite'):
            call(baseline)
    with pytest.raises(ValueError, match='zero-shift augmented design'):
        call(linear)
    assert before == (X.tobytes(), U.tobytes(), D.tobytes())


@pytest.mark.parametrize('implicit', [False, True])
@pytest.mark.parametrize('kind', ['dense', 'csr'])
@pytest.mark.parametrize('mode', ['schur'])
def test_positive_shift_skips_rank_guard_and_preserves_original_results(implicit, kind, mode, baseline, monkeypatch):
    dense, y = structured('duplicate'); X = representation(dense, kind)
    before = snapshot(X)
    monkeypatch.setattr(factor, 'validate_unregularized_design', lambda *a: pytest.fail('Positive shift invoked rank guard'))
    monkeypatch.setattr(linear, 'validate_unregularized_design', lambda *a: pytest.fail('Positive shift invoked rank guard'))
    a = solve(baseline, X, y, .2, implicit=implicit, mode=mode, steps=4)
    b = solve(linear, X, y, .2, implicit=implicit, mode=mode, steps=4)
    assert_results_equal(a, b, dense)
    unchanged(X, before)


@pytest.mark.parametrize('implicit', [False, True])
@pytest.mark.parametrize('kind', ['dense', 'csr'])
@pytest.mark.parametrize('q', [1, 2])
def test_zero_penalty_full_rank_step_matches_exact_augmented_equations(implicit, kind, q):
    dense, y = ordinary(); X = representation(dense, kind)
    assert fraction_rank(dense) == 4
    beta = np.array([.125, -.25, .0625]); offset = -.125
    P, rhs, expected = exact_step(dense, y, beta, offset, q)
    before = snapshot(X)
    result = solve(linear, X, y, implicit=implicit, q=q, beta_init=beta, offset_init=offset)
    actual = np.r_[offset - result[1], beta - result[0]]
    assert_equation(P, rhs, actual)
    assert_allclose(actual, expected, rtol=512 * EPS, atol=512 * EPS)
    assert result[2][-1] <= result[2][0] + 512 * EPS
    unchanged(X, before)


@pytest.mark.parametrize('power', [-40, 0, 40])
@pytest.mark.parametrize('implicit', [False, True])
@pytest.mark.parametrize('kind', ['dense', 'csr'])
def test_scaled_full_rank_diagonal_control_disproves_raw_gram_relative_rank_cutoff(power, implicit, kind):
    dense, y = walsh(); scale = math.ldexp(1., power); dense *= scale
    X = representation(dense, kind)
    result = solve(linear, X, y, implicit=implicit)
    assert_allclose(result[0] * scale, [.25, 0., 0.], rtol=512 * EPS, atol=512 * EPS)
    assert abs(result[1]) <= 512 * EPS
    assert_allclose(dense @ result[0] + result[1], .25 * y, rtol=512 * EPS, atol=512 * EPS)
    if power == -40:
        assert math.ldexp(1., 2 * power) < EPS * 8


@pytest.mark.parametrize('power,accepted', [(-8, True), (-50, False)])
@pytest.mark.parametrize('kind', ['dense', 'csr'])
def test_numerically_resolved_and_unresolved_full_rank_designs(power, accepted, kind):
    W, _ = walsh(); X = np.column_stack((W[:, 0], W[:, 0] + math.ldexp(1., power) * W[:, 1]))
    assert fraction_rank(X) == 3  # Both are exactly full rank; policy is numerical.
    if accepted:
        factor.validate_unregularized_design(representation(X, kind))
    else:
        with pytest.raises(ValueError, match='rank deficient or numerically unresolved'):
            factor.validate_unregularized_design(representation(X, kind))


@pytest.mark.parametrize('powers', [(-400, -200, 200), (-500, 0, 500), (400, 400, 400)])
@pytest.mark.parametrize('kind', ['dense', 'csr'])
def test_guard_only_extreme_column_units_preserve_resolved_rank(powers, kind):
    W, _ = walsh()
    X = np.ldexp(W, np.array(powers))
    before = X.tobytes()
    factor.validate_unregularized_design(representation(X, kind))
    assert X.tobytes() == before


@pytest.mark.parametrize('deficient', [False, True])
@pytest.mark.parametrize('implicit', [False, True])
@pytest.mark.parametrize('kind', ['dense', 'csr'])
def test_positive_penalty_with_zero_represented_shift_uses_same_rank_contract(deficient, implicit, kind):
    dense, y = structured('duplicate') if deficient else walsh()
    penalty, q = math.ldexp(1., -1074), 128
    M = float(q) + 2. + 1. / q
    assert penalty > 0 and 2 * len(dense) * penalty / M == 0
    X = representation(dense, kind); before = snapshot(X)
    if deficient:
        with pytest.raises(ValueError, match='zero-shift augmented design'):
            solve(linear, X, y, penalty, implicit=implicit, q=q)
    else:
        result = solve(linear, X, y, penalty, implicit=implicit, q=q)
        assert_allclose(result[0], [1. / M, 0., 0.], rtol=1024 * EPS, atol=1024 * EPS)
        assert abs(result[1]) <= 1024 * EPS
    unchanged(X, before)


@pytest.mark.parametrize('refit', [False, True])
@pytest.mark.parametrize('kind', ['dense', 'csr'])
def test_public_implicit_refusal_cleans_fitted_state_and_preserves_readonly_input(refit, kind):
    model = linear.GenDWD(lambd=.2, max_iter=2, stopping='fixed')
    if refit:
        good, labels = ordinary(); model.fit(representation(good, kind), labels)
        assert hasattr(model, 'coef_')
    dense, y = structured('onehot'); X = representation(dense, kind)
    for a in ((X.data, X.indices, X.indptr) if sparse.issparse(X) else (X,)):
        a.flags.writeable = False
    before = snapshot(X)
    model.set_params(lambd=0.)
    with pytest.raises(ValueError, match='zero-shift augmented design'):
        model.fit(X, y)
    assert not [name for name in vars(model) if name.endswith('_') and not name.startswith('_')]
    with pytest.raises(NotFittedError):
        model.decision_function(X)
    unchanged(X, before)


def assert_models_equal(a, b, X):
    for field in ('coef_', 'intercept_', 'objective_history_', 'C_', 'stationarity_residual_'):
        assert_allclose(getattr(a, field), getattr(b, field), rtol=2e-12, atol=2e-13)
    for field in ('n_iter_', 'returned_iteration_', 'termination_reason_', 'converged_'):
        assert getattr(a, field) == getattr(b, field)
    assert_array_equal(a.predict(X), b.predict(X))
    assert_allclose(a.decision_function(X), b.decision_function(X), rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize('kind', ['dense', 'csr'])
def test_resolved_zero_penalty_cache_matches_fresh_and_baseline(kind, baseline):
    dense, y = ordinary(); X = representation(dense, kind)
    options = dict(lambd=0., implicit_P=True, max_iter=3, stopping='fixed', initialization='zero')
    model = linear.GenDWD(**options).cv_init(X)
    cache = model._P0_eig
    saved = tuple(a.copy() for a in cache)
    model.fit(X, y)
    assert model._P0_eig is cache
    for a, b in zip(cache, saved):
        assert_array_equal(a, b)
    assert_models_equal(model, linear.GenDWD(**options).fit(X, y), X)
    assert_models_equal(model, baseline.GenDWD(**options).fit(X, y), X)


def cv_data():
    X, y = ordinary()
    return np.vstack((X, X)), np.tile(y, 2), [(np.arange(8), np.arange(8, 16)), (np.arange(8, 16), np.arange(8))]


@pytest.mark.parametrize('framework', ['package', 'sklearn'])
@pytest.mark.parametrize('implicit', [False, True])
def test_zero_and_positive_penalty_public_cv_matches_original(framework, implicit, baseline):
    X, y, folds = cv_data()
    def fit(module):
        if framework == 'package':
            return module.GenDWDCV(lambd_vals=[0., .2], q_vals=[1.], cv=folds,
                                  implicit_P=implicit, initialization='zero', stopping='fixed', max_iter=2).fit(X, y)
        return GridSearchCV(module.GenDWD(implicit_P=implicit, initialization='zero', stopping='fixed', max_iter=2),
                            {'lambd': [0., .2]}, cv=folds, error_score='raise', n_jobs=1).fit(X, y)
    a, b = fit(baseline), fit(linear)
    assert a.best_params_ == b.best_params_ and a.best_score_ == b.best_score_
    if framework == 'package':
        assert a.agg_cv_results_.keys() == b.agg_cv_results_.keys()
        for key in a.agg_cv_results_:
            if key not in ('mean_runtime', 'std_runtime'):
                assert a.agg_cv_results_[key] == b.agg_cv_results_[key]
    else:
        for key in ('params', 'mean_test_score', 'std_test_score', 'rank_test_score',
                    'split0_test_score', 'split1_test_score'):
            assert_array_equal(a.cv_results_[key], b.cv_results_[key])
    assert_models_equal(a.best_estimator_, b.best_estimator_, X)


@pytest.mark.parametrize('framework', ['package', 'sklearn'])
@pytest.mark.parametrize('kind', ['dense', 'csr'])
def test_rank_deficient_zero_penalty_public_cv_fails_explicitly(framework, kind):
    dense, y = structured('constant')
    X = representation(np.vstack((dense, dense)), kind); y = np.tile(y, 2)
    folds = [(np.arange(8), np.arange(8, 16)), (np.arange(8, 16), np.arange(8))]
    if framework == 'package':
        model = linear.GenDWDCV(lambd_vals=[0., .2], cv=folds, max_iter=2, stopping='fixed')
    else:
        model = GridSearchCV(linear.GenDWD(max_iter=2, stopping='fixed'), {'lambd': [0., .2]},
                             cv=folds, error_score='raise', n_jobs=1)
    with pytest.raises(ValueError, match='zero-shift augmented design'):
        model.fit(X, y)
    assert not hasattr(model, 'best_estimator_')
