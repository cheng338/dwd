"""Automatic bounded recovery through the real GenDWD public entrypoints."""
import hashlib
import importlib
import importlib.util
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal
import pytest
from scipy import sparse

from dwd._exact_linear_mm import ExactCoordinateMM
from dwd import _implicit_recovery


candidate = importlib.import_module('dwd.gen_dwd')
_spec = importlib.util.spec_from_file_location(
    '_f11_exact_test_oracle', Path(__file__).with_name('fixtures') / 'f11_exact_oracle.py')
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


@pytest.fixture(scope='session')
def baseline():
    path = Path(__file__).with_name('fixtures') / 'f11_baseline_gen_dwd.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'cdf90ba00808c1dc81fa36f42a64c13245a33d9581ea6d79b968650045edde1e'
    spec = importlib.util.spec_from_file_location('dwd._f11_pre_recovery_baseline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observe_actions(monkeypatch):
    """Test-only observer: the identical readonly result reaches the caller."""
    original = ExactCoordinateMM.next_iterate
    records = []
    def observed(system, h, beta, offset):
        old = np.r_[float(offset), beta]
        saved_h = h.copy()
        theta, receipt = original(system, h, beta, offset)
        assert not theta.flags.writeable
        records.append(dict(X=system.X.copy(), shift=system.s, old=old, h=saved_h,
                            proposed=theta.copy(), receipt=receipt))
        return theta, receipt
    monkeypatch.setattr(ExactCoordinateMM, 'next_iterate', observed)
    return records


def model(module, **kwargs):
    return module.GenDWD(initialization='zero', stopping='fixed', max_iter=2,
                         obj_tol=0.0, tol=1e-12, **kwargs)


def assert_same(a, b, X):
    for name in ('coef_', 'intercept_', 'objective_history_', 'classes_'):
        assert_array_equal(getattr(a, name), getattr(b, name))
    for name in ('n_iter_', 'C_', 'final_objective_', 'gradient_inf_norm_',
                 'stationarity_residual_', 'converged_', 'termination_reason_'):
        assert getattr(a, name) == getattr(b, name)
    assert_array_equal(a.decision_function(X), b.decision_function(X))
    assert_array_equal(a.predict(X), b.predict(X))


def assert_actual_actions(fitted, X, records):
    if not records:
        return  # Native preparation may succeed on another backend.
    assert len(records) == fitted.n_iter_ == fitted.returned_iteration_
    previous = None
    for record in records:
        assert_array_equal(record['X'], X)
        if previous is not None:
            assert [oracle.bits(v) for v in record['old']] == [oracle.bits(v) for v in previous]
        action = oracle.ExactAction(X, float(record['shift']), record['h'], record['old'])
        oracle.assert_receipt(action, record['proposed'], record['receipt'], require_immutable=False)
        previous = record['proposed']
    actual = np.r_[fitted.intercept_[0], fitted.coef_.ravel()]
    assert [oracle.bits(v) for v in actual] == [oracle.bits(v) for v in previous]


@pytest.mark.parametrize('entry', ['computed', 'cached'])
def test_ordinary_success_is_identical(baseline, entry, monkeypatch):
    X, y = oracle.ordinary()
    records = observe_actions(monkeypatch)
    for implicit in (False, True):
        a, b = [model(module, lambd=0.125, implicit_P=implicit) for module in (baseline, candidate)]
        if entry == 'cached':
            a.cv_init(X)
            b.cv_init(X)
        a.fit(X, y)
        b.fit(X, y)
        assert not records
        assert_same(a, b, X)


@pytest.mark.parametrize('case', ['original', 'structural'])
def test_actual_public_defaults(case, monkeypatch):
    if case == 'original':
        X, signed, penalty = oracle.case('original_far_outlier')
        y = (signed+1)/2
    else:
        X = np.random.RandomState(110230).normal(size=(128, 8))
        X[:, 0] = 2.0**30
        y, penalty = np.tile([0, 1], 64), 0.05
    saved = X.copy()
    records = observe_actions(monkeypatch)
    fitted = candidate.GenDWD(lambd=penalty)
    params = fitted.get_params()
    assert {key: params[key] for key in ('initialization', 'stopping', 'max_iter', 'obj_tol', 'tol')} == {
        'initialization': 'zero', 'stopping': 'objective', 'max_iter': 100, 'obj_tol': 1e-5, 'tol': 1e-6}
    fitted.fit(X, y)
    assert 0 <= fitted.n_iter_ <= 100
    assert np.isfinite(fitted.decision_function(X)).all()
    assert_actual_actions(fitted, X, records)
    assert_array_equal(X, saved)
    # Convergence and objective stopping are reported by the unchanged solver;
    # coordinate correctness does not imply either stopping condition is met.
    assert fitted.converged_ == (np.isfinite(fitted.stationarity_residual_)
                                and fitted.stationarity_residual_ <= fitted.tol)


def test_malformed_pairs_do_not_recover(monkeypatch):
    X, y = oracle.ordinary()
    U, D = candidate.get_P0_eig(X)
    records = observe_actions(monkeypatch)
    with pytest.raises(ValueError):
        model(candidate, lambd=0.125).fit(X, y, P0_eig=(U, D+1))
    assert not records


def test_automatic_branch_scope_and_late_hooks(monkeypatch):
    for branch in ('pi', 'denominator'):
        with monkeypatch.context() as local:
            X, y, penalty, _, U, D = oracle.branch_case(branch)
            local.setattr(candidate, 'validated_eigh', lambda matrix, supplied=None: (D.copy(), U.copy()))
            unobserved = model(candidate, lambd=penalty).fit(X, y)
            records = observe_actions(local)
            observed = model(candidate, lambd=penalty).fit(X, y)
            assert len(records) == 2
            assert_actual_actions(observed, X, records)
            assert_same(unobserved, observed, X)
    X, y, penalty, _, U, D = oracle.branch_case('pi')
    for kind in ('sparse', 'step', 'loss', 'gradient', 'error_callback', 'late_gradient'):
        with monkeypatch.context() as local:
            records = observe_actions(local)
            local.setattr(candidate, 'validated_eigh', lambda matrix, supplied=None: (D.copy(), U.copy()))
            if kind in ('step', 'loss', 'gradient'):
                local.setattr(candidate, {'step':'get_step_implicit', 'loss':'V_grad',
                                         'gradient':'_linear_gradient'}[kind], lambda *a, **k: None)
            if kind == 'late_gradient':
                original = candidate.dwd_obj
                def mutate(*args, **kwargs):
                    value = original(*args, **kwargs)
                    local.setattr(candidate, '_linear_gradient', lambda *a, **k: (0.0, np.zeros(1), 0.0))
                    return value
                local.setattr(candidate, 'dwd_obj', mutate)
            fitted = candidate.GenDWD(lambd=penalty, initialization='zero', max_iter=2,
                stopping='optimality')
            with (np.errstate(over='call') if kind == 'error_callback' else np.errstate()):
                with pytest.raises(ValueError, match='changed numerical hooks|numerical preparation failure'):
                    fitted.fit(sparse.csr_matrix(X) if kind == 'sparse' else X, y)
            assert not records and not hasattr(fitted, 'coef_')
    records = observe_actions(monkeypatch)
    with pytest.raises(ValueError, match=r'^The zero-shift augmented design is rank deficient or numerically unresolved at float64 precision\.$'):
        model(candidate, lambd=0.0).fit(np.ones((8, 2)), np.tile([0, 1], 4))
    assert not records


def test_feature_limit_refuses_without_fitted_state(monkeypatch):
    X, y = np.ones((4, 17)), np.array([0, 1, 0, 1])
    records = observe_actions(monkeypatch)
    monkeypatch.setattr(candidate, 'validated_eigh', lambda matrix, supplied=None:
                        (np.r_[-1.0, np.ones(17)], np.eye(18)))
    fitted = model(candidate, lambd=0.05)
    with pytest.raises(ValueError, match='bounded resource policy'):
        fitted.fit(X, y)
    assert not records and not hasattr(fitted, 'coef_')


def test_cached_cv_ordinary_smoke(monkeypatch):
    X, y = oracle.ordinary()
    records = observe_actions(monkeypatch)
    original = candidate.get_P0_eig
    calls = []
    def counted(features):
        calls.append(features.shape)
        return original(features)
    monkeypatch.setattr(candidate, 'get_P0_eig', counted)
    fitted = candidate.GenDWDCV(lambd_vals=[0.125, 0.25], q_vals=[1], cv=2,
                                initialization='zero', stopping='fixed', max_iter=2).fit(X, y)
    assert not records and len(calls) == 3
    assert fitted.best_estimator_.n_iter_ == 2 and np.isfinite(fitted.best_score_)


def test_forged_authority_and_changed_feature_target_are_rejected(monkeypatch):
    X, y, penalty, _, U, D = oracle.branch_case('pi')
    with monkeypatch.context() as local:
        local.setattr(candidate, 'validated_eigh', lambda matrix, supplied=None: (D.copy(), U.copy()))
        original = ExactCoordinateMM.next_iterate
        def forged(system, *args, **kwargs):
            theta, receipt = original(system, *args, **kwargs)
            receipt['input_sha256'] = '0'*64
            return theta, receipt
        local.setattr(ExactCoordinateMM, 'next_iterate', forged)
        fitted = model(candidate, lambd=penalty)
        with pytest.raises(ValueError, match='unbound authority'):
            fitted.fit(X, y)
        assert not hasattr(fitted, 'coef_')
    X = np.array([[-1.0], [1.0]])
    system = _implicit_recovery.prepare(vars(candidate), X, 1.0)
    X[0, 0] = -2.0
    with pytest.raises(ValueError, match='feature target changed'):
        _implicit_recovery.update(vars(candidate), system, X, 1.0,
                                  np.zeros(2), np.zeros(1), 0.0)
