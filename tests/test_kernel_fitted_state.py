"""Generic sklearn fitted detection must agree with usable model lifecycle."""
import copy
import pickle

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from dwd.gen_dwd import GenDWD
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV
from dwd.kern_md import KernMD

X = np.array([[-2., .1], [-1., -.2], [-.8, .3], [-.4, -.1],
              [.5, .2], [.9, -.3], [1.3, .1], [2., -.2]])
Y = np.repeat([-1, 1], 4)


def make_case(kind, backend='auto'):
    opts = dict(max_iter=3, stopping='fixed', backend=backend, random_state=42)
    if kind == 'precomputed':
        return KernGDWD(kernel=kind, **opts), X @ X.T + np.eye(len(X)), Y
    if kind == 'rank_deficient':
        return KernGDWD(kernel='linear', **opts), np.zeros_like(X), Y
    if kind == 'sparse':
        return KernGDWD(kernel='rbf', **opts), csr_matrix(X), Y
    return KernGDWD(kernel=kind, **opts), X.copy(), Y


def assert_unfitted(model, query):
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    with pytest.raises(NotFittedError):
        model.predict(query)


@pytest.mark.parametrize('backend', ['auto', 'cholesky', 'spectral', 'lbfgs'])
@pytest.mark.parametrize('kind', ['linear', 'rbf', 'precomputed', 'sparse', 'rank_deficient'])
def test_prepare_fit_prepare_refit(kind, backend, monkeypatch):
    model, data, labels = make_case(kind, backend)
    assert_unfitted(model, data)
    params = model.get_params(deep=False).copy()
    assert model.cv_init(data) is model
    cache = model._cv_K
    assert_unfitted(model, data)
    assert model.get_params(deep=False) == params
    model.fit(data, labels)
    check_is_fitted(model)
    scores = model.decision_function(data).copy()
    assert np.isfinite(scores).all()
    assert model._cv_K is cache
    assert_unfitted(clone(model), data)
    model.cv_init(data)
    assert_unfitted(model, data)
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise AssertionError('prepared cache rebuilt')
        patch.setattr(model, '_compute_training_kernel', fail)
        model.fit(data, labels)
    check_is_fitted(model)
    np.testing.assert_allclose(model.decision_function(data), scores, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('prior', ['fresh', 'prepared', 'fitted'])
@pytest.mark.parametrize('failure', ['invalid_fit', 'invalid_prepare', 'prepare_interrupt', 'late_result'])
def test_failure_cleanup(prior, failure, monkeypatch):
    model, data, labels = make_case('rbf')
    if prior == 'prepared':
        model.cv_init(data)
    elif prior == 'fitted':
        model.fit(data, labels)
    if failure == 'invalid_fit':
        with pytest.raises(ValueError):
            model.fit(data, np.ones(len(labels)))
    elif failure == 'invalid_prepare':
        with pytest.raises(ValueError):
            model.cv_init(np.full(data.shape, np.nan))
    elif failure == 'prepare_interrupt':
        error = KeyboardInterrupt('controlled preparation interruption')
        def fail(*args, **kwargs):
            raise error
        monkeypatch.setattr(model, '_compute_training_kernel', fail)
        with pytest.raises(KeyboardInterrupt) as caught:
            model.cv_init(data)
        assert caught.value is error
    else:
        original = model._set_fit_result
        error = RuntimeError('controlled error after coefficient storage')
        def fail(result):
            original(result)
            raise error
        monkeypatch.setattr(model, '_set_fit_result', fail)
        with pytest.raises(RuntimeError) as caught:
            model.fit(data, labels)
        assert caught.value is error
    assert_unfitted(model, data)


@pytest.mark.parametrize('state', ['fresh', 'prepared', 'fitted', 'failed'])
@pytest.mark.parametrize('transport', ['pickle', 'deepcopy'])
def test_roundtrip(state, transport):
    model, data, labels = make_case('rbf')
    if state == 'prepared':
        model.cv_init(data)
    elif state in ('fitted', 'failed'):
        model.fit(data, labels)
        if state == 'failed':
            with pytest.raises(ValueError):
                model.fit(data, labels[:-1])
    restored = pickle.loads(pickle.dumps(model)) if transport == 'pickle' else copy.deepcopy(model)
    assert restored.get_params() == model.get_params()
    if state == 'fitted':
        check_is_fitted(restored)
        np.testing.assert_array_equal(restored.decision_function(data), model.decision_function(data))
    else:
        assert_unfitted(restored, data)


def test_explicit_attribute_queries_preserve_sklearn_semantics():
    model = KernGDWD().cv_init(X)
    check_is_fitted(model, 'kernel_')  # Explicit query, not generic predictor readiness.
    with pytest.raises(NotFittedError):
        check_is_fitted(model, ['kernel_', 'dual_coef_'])
    check_is_fitted(model, ['kernel_', 'dual_coef_'], all_or_any=any)
    assert_unfitted(model, X)


@pytest.mark.parametrize('attribute', ['dual_coef_', 'intercept_', 'classes_', '_Xfit'])
def test_missing_prediction_state_refuses(attribute):
    model = KernGDWD(max_iter=0).fit(X, Y)
    delattr(model, attribute)
    assert_unfitted(model, X)


def test_nonconverged_model_is_still_fitted():
    model = KernGDWD(max_iter=0).fit(X, Y)
    assert not model.converged_
    check_is_fitted(model)
    assert np.isfinite(model.decision_function(X)).all()


def test_pipeline_observes_preparation_and_invalidation():
    scaler = StandardScaler().fit(X)
    model = KernGDWD(max_iter=2, stopping='fixed')
    pipeline = Pipeline([('scale', scaler), ('dwd', model)])
    model.cv_init(scaler.transform(X))
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline)
    pipeline.fit(X, Y)
    check_is_fitted(pipeline)
    assert len(pipeline.predict(X)) == len(Y)
    model.cv_init(scaler.transform(X))
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline)


def test_boundary_estimators():
    linear = GenDWD(max_iter=2).cv_init(X)
    assert_unfitted(linear, X)
    linear.fit(X, Y)
    check_is_fitted(linear)
    for model in [KernMD(), KernGDWDCV(lambd_vals=[1.], q_vals=[1.], cv=2, max_iter=2)]:
        assert_unfitted(model, X)
        model.fit(X, Y)
        check_is_fitted(model)
        assert len(model.predict(X)) == len(Y)


def test_subclass_inherits_contract():
    class Derived(KernGDWD):
        pass
    model = Derived(max_iter=0).cv_init(X)
    assert_unfitted(model, X)
    model.fit(X, Y)
    check_is_fitted(model)


def test_hook_is_not_added_to_numerical_or_prediction_paths(monkeypatch):
    def fail(*args):
        raise AssertionError('generic hook entered hot path')
    monkeypatch.setattr(KernGDWD, '__sklearn_is_fitted__', fail)
    model = KernGDWD(max_iter=2, stopping='fixed').cv_init(X).fit(X, Y)
    assert np.isfinite(model.decision_function(X)).all()
    assert len(model.predict(X)) == len(Y)

