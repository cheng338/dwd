"""Unsupported fit weights reject cleanly without rebinding genuine solver options."""
import importlib
import inspect

import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics.pairwise import rbf_kernel
from sklearn import config_context
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted, has_fit_parameter

from dwd.gen_dwd import GenDWD, GenDWDCV, get_P0_eig
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV, get_K_eig
from dwd.kernel_utils import KernelClfMixin


SOLVERS = ('linear', 'kernel', 'socp', 'svm')
MODULES = {'linear': ('dwd.gen_dwd', 'solve_gen_dwd'),
           'kernel': ('dwd.gen_kern_dwd', 'solve_gen_kern_dwd'),
           'socp': ('dwd.socp_dwd', 'solve_dwd_socp'),
           'svm': ('dwd.svm', 'solve_svm')}


@pytest.fixture
def data():
    X = np.random.default_rng(46034).normal(scale=.35, size=(12, 4))
    y = np.tile(np.array([-1, 1]), 6)
    X[:, 0] += .7 * y
    K = rbf_kernel(X, gamma=.25) + .1 * np.eye(len(y))
    return X, y, K


def _load(name):
    if name in ('socp', 'svm'):
        cp = pytest.importorskip('cvxpy', reason='Optional conic dependency is unavailable.')
        if 'CLARABEL' not in cp.installed_solvers():
            pytest.skip('CLARABEL is unavailable; conic cases are not validated.')
    module_name, function_name = MODULES[name]
    module = importlib.import_module(module_name)
    return module, getattr(module, function_name)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _arguments(name, data):
    X, y, K = data
    if name == 'linear':
        return [X, y, .3, 1], dict(max_iter=3, stopping='fixed', random_state=23)
    if name == 'kernel':
        return [K, y, .3, 1], dict(max_iter=3, obj_tol=0., random_state=23)
    return [X, y, .3], dict(solver_kws={'solver': 'CLARABEL'})


def _numeric_result(name, result, X, K=None):
    coefficient = np.asarray(result[0], dtype=float).reshape(-1)
    intercept = float(np.asarray(result[1]))
    scores = np.asarray((K if name == 'kernel' else X) @ coefficient + intercept)
    assert coefficient.ndim == 1 and np.isfinite(coefficient).all()
    assert np.isfinite(intercept) and np.isfinite(scores).all()
    payload = dict(coefficient=coefficient, intercept=intercept, scores=scores,
                   signed_predictions=np.where(scores >= 0., 1, -1))
    if name in ('linear', 'kernel'):
        history = np.asarray(result[2], dtype=float)
        conversion = float(result[3])
        assert history.ndim == 1 and len(history) >= 1 and np.isfinite(history).all()
        assert np.isfinite(conversion) and conversion >= 0
        payload.update(history=history, conversion=conversion)
    else:
        problem = result[-1]
        assert problem.status in ('optimal', 'optimal_inaccurate')
        assert np.isfinite(problem.value)
        payload.update(status=problem.status, objective=float(problem.value))
        if name == 'socp':
            assert np.isfinite(result[2]).all() and np.isfinite(result[3]).all()
            payload.update(slack=result[2], distance=result[3])
    return _jsonable(payload)


@pytest.mark.parametrize('name', SOLVERS)
def test_signature_preserves_real_options_and_defaults(name):
    _, solver = _load(name)
    parameters = inspect.signature(solver).parameters
    genuine = {
        'linear': [('X', inspect.Parameter.empty), ('y', inspect.Parameter.empty),
                   ('lambd', inspect.Parameter.empty), ('q', 1), ('beta_init', None),
                   ('offset_init', None), ('implicit_P', True), ('obj_tol', 1e-5),
                   ('max_iter', 100), ('P0_eig', None), ('random_state', None),
                   ('initialization', 'zero'), ('stopping', 'objective'), ('tol', 1e-6)],
        'kernel': [('K', inspect.Parameter.empty), ('y', inspect.Parameter.empty),
                   ('lambd', inspect.Parameter.empty), ('q', 1), ('alpha_init', None),
                   ('offset_init', None), ('obj_tol', 1e-5), ('max_iter', 100),
                   ('K_eig', None), ('random_state', None), ('acceleration', 'auto'),
                   ('linear_action', 'factor')],
        'socp': [('X', inspect.Parameter.empty), ('y', inspect.Parameter.empty),
                 ('C', 1.0), ('solver_kws', None)],
        'svm': [('X', inspect.Parameter.empty), ('y', inspect.Parameter.empty),
                ('C', inspect.Parameter.empty), ('solver_kws', None)],
    }[name]
    assert [(key, value.default) for key, value in parameters.items()
            if key != 'sample_weight'] == genuine
    assert 'sample_weight' not in parameters
    assert all(p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters.values())
    boundary = {'linear': 4, 'kernel': 6, 'socp': 3, 'svm': 3}[name]
    assert all(p.kind == p.POSITIONAL_OR_KEYWORD for p in list(parameters.values())[:boundary])
    assert all(p.kind == p.KEYWORD_ONLY for p in list(parameters.values())[boundary:])


class _ValidationReached(Exception):
    pass


def _validation_sentinel(monkeypatch, module):
    calls = []
    def stop(*args, **kwargs):
        calls.append(True)
        raise _ValidationReached('The mathematical input validator was reached.')
    monkeypatch.setattr(module, 'check_real_X_y', stop)
    return calls


def _weight(kind, n):
    return None if kind == 'none' else [1.] * n if kind == 'list' else np.linspace(.5, 1.5, n)


@pytest.mark.parametrize('name', SOLVERS)
@pytest.mark.parametrize('kind', ('none', 'list', 'array'))
def test_keyword_weight_never_becomes_a_solver_option(name, kind, data, monkeypatch):
    module, solver = _load(name)
    calls = _validation_sentinel(monkeypatch, module)
    args, kwargs = _arguments(name, data)
    kwargs['sample_weight'] = _weight(kind, len(data[1]))
    with pytest.raises(TypeError, match='sample_weight'):
        solver(*args, **kwargs)
    assert calls == []


@pytest.mark.parametrize('name', SOLVERS)
@pytest.mark.parametrize('kind', ('none', 'list', 'array'))
def test_old_positional_weight_slot_never_rebinds(name, kind, data, monkeypatch):
    module, solver = _load(name)
    calls = _validation_sentinel(monkeypatch, module)
    args, kwargs = _arguments(name, data)
    if name == 'kernel':
        args += [None, None]  # Genuine alpha/offset positional inputs.
    args.append(_weight(kind, len(data[1])))
    with pytest.raises(TypeError, match='positional'):
        solver(*args, **kwargs)
    assert calls == []


@pytest.mark.parametrize('storage', ('dense', 'csr'))
@pytest.mark.parametrize('mode', ('implicit', 'explicit', 'initialized', 'supplied_eigen'))
def test_linear_keyword_options_preserve_results(storage, mode, data):
    module, solver = _load('linear')
    X, y, _ = data
    features = X.copy() if storage == 'dense' else sparse.csr_matrix(X)
    before = features.copy()
    options = dict(max_iter=3, stopping='fixed', random_state=23, obj_tol=0.)
    if mode == 'explicit':
        options['implicit_P'] = False
    if mode == 'initialized':
        options.update(beta_init=np.array([.02, -.03, .04, .01]), offset_init=.07,
                       initialization='random')
    if mode == 'supplied_eigen':
        options['P0_eig'] = module.get_P0_eig(features)
    result = solver(features, y, .3, 1, **options)
    np.testing.assert_array_equal(features.toarray() if sparse.issparse(features) else features,
                                  before.toarray() if sparse.issparse(before) else before)
    _numeric_result('linear', result, X)


@pytest.mark.parametrize('mode', ('auto', 'initialized', 'supplied_eigen', 'inverse'))
def test_kernel_keyword_options_preserve_results(mode, data):
    module, solver = _load('kernel')
    X, y, K = data
    before = K.copy()
    options = dict(max_iter=3, obj_tol=0., random_state=23)
    if mode == 'initialized':
        options.update(alpha_init=np.linspace(-.02, .02, len(y)), offset_init=.07,
                       acceleration=None)
    if mode == 'supplied_eigen':
        options['K_eig'] = module.get_K_eig(K)
    if mode == 'inverse':
        options.update(linear_action='inverse', acceleration=None)
    result = solver(K, y, .3, 1, **options)
    np.testing.assert_array_equal(K, before)
    _numeric_result('kernel', result, X, K)


@pytest.mark.parametrize('name', ('socp', 'svm'))
@pytest.mark.parametrize('storage', ('dense', 'csr'))
def test_conic_keyword_options_preserve_results(name, storage, data):
    _, solver = _load(name)
    X, y, _ = data
    features = X.copy() if storage == 'dense' else sparse.csr_matrix(X)
    before = features.copy()
    options = {'solver': 'CLARABEL'}
    result = solver(features, y, .3, solver_kws=options)
    assert options == {'solver': 'CLARABEL'}
    np.testing.assert_array_equal(features.toarray() if sparse.issparse(features) else features,
                                  before.toarray() if sparse.issparse(before) else before)
    _numeric_result(name, result, X)


def test_linear_full_old_null_weight_positional_call_requires_keyword_migration(data):
    _, solver = _load('linear')
    X, y, _ = data
    beta = np.array([.02, -.03, .04, .01])
    old_args = [X, y, .3, 1, None, beta, .07, False, 0., 3, None, 23]
    with pytest.raises(TypeError, match='positional'):
        solver(*old_args, stopping='fixed')
    actual = solver(X, y, .3, 1, beta_init=beta, offset_init=.07, implicit_P=False,
                    obj_tol=0., max_iter=3, P0_eig=None, random_state=23, stopping='fixed')
    _numeric_result('linear', actual, X)


@pytest.mark.parametrize('name', ('socp', 'svm'))
def test_conic_old_null_weight_and_positional_options_require_keyword_migration(name, data):
    _, solver = _load(name)
    X, y, _ = data
    options = {'solver': 'CLARABEL'}
    with pytest.raises(TypeError, match='positional'):
        solver(X, y, .3, None, options)
    actual = solver(X, y, .3, solver_kws=options)
    _numeric_result(name, actual, X)


def test_kernel_genuine_six_positional_slots_still_work(data):
    _, solver = _load('kernel')
    X, y, K = data
    alpha = np.linspace(-.02, .02, len(y))
    options = dict(max_iter=3, obj_tol=0., acceleration=None, random_state=23)
    positional = solver(K, y, .3, 1, alpha, .07, **options)
    keyword = solver(K, y, .3, q=1, alpha_init=alpha, offset_init=.07, **options)
    payload = _numeric_result('kernel', positional, X, K)
    assert payload == _numeric_result('kernel', keyword, X, K)


@pytest.mark.parametrize('name, option', [('linear', 'beta_init'), ('linear', 'offset_init'),
                                          ('kernel', 'alpha_init'), ('kernel', 'offset_init')])
def test_invalid_real_initialization_still_fails_specifically(name, option, data):
    _, solver = _load(name)
    args, options = _arguments(name, data)
    options[option] = np.nan if option == 'offset_init' else np.array([np.nan])
    with pytest.raises(ValueError, match=option):
        solver(*args, **options)


@pytest.mark.parametrize('name', ('socp', 'svm'))
def test_real_tuning_helper_forwards_solver_options_by_keyword(name, data, monkeypatch):
    _, solver = _load(name)
    pytest.importorskip('matplotlib', reason='Optional plotting dependency is unavailable.')
    helper = importlib.import_module('dwd.sim_fun')
    # Exercise the real caller without creating a figure or displaying a graph.
    for operation in ('plot', 'xscale', 'legend', 'xlabel', 'ylim'):
        monkeypatch.setattr(helper.plt, operation, lambda *args, **kwargs: None)
    X, y, _ = data
    options = {'solver': 'CLARABEL'}
    train, test = helper.tuning_curve(X, y, X, y, [.3], solver, solver_kws=options)
    assert train.shape == test.shape == (1,)
    assert np.isfinite(train).all() and np.isfinite(test).all()
    assert np.all((0 <= train) & (train <= 1))
    np.testing.assert_array_equal(train, test)
    assert options == {'solver': 'CLARABEL'}


@pytest.mark.parametrize('name', ('linear', 'linear_cv', 'socp', 'svm'))
def test_real_estimator_callers_preserve_unweighted_fit(name, data):
    module, _ = _load('linear' if name == 'linear_cv' else name)
    if name == 'linear':
        model = module.GenDWD(lambd=.3, q=1, max_iter=3, stopping='fixed', random_state=23)
    elif name == 'linear_cv':
        model = module.GenDWDCV(lambd_vals=(.3, .6), q_vals=(1,), cv=2,
                               max_iter=3, stopping='fixed', random_state=23)
    else:
        estimator = module.DWD if name == 'socp' else module.SVM
        model = estimator(C=.3, solver_kws={'solver': 'CLARABEL'})
    X, y, _ = data
    before = X.copy()
    assert model.fit(X, y) is model
    coefficient = np.asarray(model.coef_)
    intercept = np.asarray(model.intercept_)
    scores = np.asarray(model.decision_function(X))
    predictions = model.predict(X)
    assert np.isfinite(coefficient).all() and np.isfinite(intercept).all()
    assert scores.shape == y.shape and np.isfinite(scores).all()
    assert set(np.unique(predictions)) <= set(y)
    np.testing.assert_array_equal(X, before)


FIT_CLASSES = ('GenDWD', 'GenDWDCV', 'KernGDWD', 'KernGDWDCV', 'DWD', 'SVM')


def _estimator(name):
    if name in ('DWD', 'SVM'):
        module, _ = _load('socp' if name == 'DWD' else 'svm')
        return getattr(module, name)(C=.3, solver_kws={'solver': 'CLARABEL'})
    classes = {'GenDWD': GenDWD, 'GenDWDCV': GenDWDCV,
               'KernGDWD': KernGDWD, 'KernGDWDCV': KernGDWDCV}
    options = dict(max_iter=3, stopping='fixed', random_state=23)
    if name.startswith('Kern'):
        options.update(kernel='rbf', backend='cholesky')
        options['kernel_kws_vals' if name.endswith('CV') else 'kernel_kws'] = (
            ({'gamma': .25},) if name.endswith('CV') else {'gamma': .25})
    if name.endswith('CV'):
        options.update(cv=2, lambd_vals=(.3,), q_vals=(1.,))
    else:
        options.update(lambd=.3, q=1.)
    return classes[name](**options)


def _assert_unfitted(model, X):
    assert not [key for key in vars(model) if key.endswith('_') and not key.startswith('_')]
    assert not hasattr(model, '_Xfit')
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    for method in (model.predict, model.decision_function):
        with pytest.raises(NotFittedError):
            method(X)


@pytest.mark.parametrize('name', FIT_CLASSES)
def test_estimator_fit_signature_and_metadata_do_not_advertise_weights(name):
    model = _estimator(name)
    before = model.get_params(deep=False)
    for actual in (model, clone(model)):
        assert 'sample_weight' not in inspect.signature(actual.fit).parameters
        assert not has_fit_parameter(actual, 'sample_weight')
        assert inspect.signature(type(actual).fit) == inspect.signature(type(actual).fit.__wrapped__)
        assert 'sample_weight' not in actual.get_metadata_routing().fit.requests
        assert 'sample_weight' in inspect.signature(actual.score).parameters
        with config_context(enable_metadata_routing=True):
            actual.set_score_request(sample_weight=True)
            assert actual.get_metadata_routing().consumes('score', ['sample_weight'])
            assert not actual.get_metadata_routing().consumes('fit', ['sample_weight'])
    assert all(model.get_params(deep=False)[key] is value for key, value in before.items())


def test_abstract_kernel_fit_does_not_advertise_plural_weights():
    parameters = inspect.signature(KernelClfMixin.fit).parameters
    assert 'sample_weights' not in parameters and 'sample_weight' not in parameters


@pytest.mark.parametrize('name', FIT_CLASSES)
@pytest.mark.parametrize('storage', ('dense', 'csr'))
def test_unweighted_fit_keeps_weighted_scoring(name, storage, data):
    X, y, _ = data
    features = sparse.csr_matrix(X) if storage == 'csr' else X
    model = _estimator(name)
    before = model.get_params(deep=False)
    assert model.fit(features, y) is model
    weights = np.arange(1., len(y) + 1.)
    score_y = y.copy()
    score_y[:3] *= -1
    predictions = model.predict(features)
    assert model.score(features, score_y, sample_weight=weights) == accuracy_score(
        score_y, predictions, sample_weight=weights)
    assert model.score(features, score_y, sample_weight=None) == accuracy_score(score_y, predictions)
    assert all(model.get_params(deep=False)[key] is value for key, value in before.items())


@pytest.mark.parametrize('name', FIT_CLASSES)
@pytest.mark.parametrize('call_kind', ('weight', 'none_keyword', 'none_positional', 'unexpected_keyword'))
def test_rejected_fit_arguments_clear_fitted_state_and_allow_recovery(name, call_kind, data):
    X, y, _ = data
    model = _estimator(name).fit(X, y)
    expected = model.decision_function(X).copy()
    before = model.get_params(deep=False)
    sentinel = object()
    model._user_private_sentinel = sentinel
    if call_kind == 'weight':
        call = lambda: model.fit(X, y, sample_weight=np.ones(len(y)))
    elif call_kind == 'none_keyword':
        call = lambda: model.fit(X, y, sample_weight=None)
    elif call_kind == 'none_positional':
        call = lambda: model.fit(X, y, None)
    else:
        call = lambda: model.fit(X, y, unsupported_option=True)
    with pytest.raises(TypeError):
        call()
    _assert_unfitted(model, X)
    assert model._user_private_sentinel is sentinel
    assert all(model.get_params(deep=False)[key] is value for key, value in before.items())
    assert model.fit(X, y) is model
    np.testing.assert_allclose(model.decision_function(X), expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('name', ('GenDWD', 'KernGDWD'))
def test_real_keyword_only_fit_inputs_remain_supported(name, data):
    X, y, _ = data
    model = _estimator(name)
    if name == 'GenDWD':
        options = {'P0_eig': get_P0_eig(X), 'beta_init': np.zeros(X.shape[1]), 'offset_init': 0.}
    else:
        K = rbf_kernel(X, gamma=.25)
        options = {'K': K, 'K_eig': get_K_eig(K), 'alpha_init': np.zeros(len(y)),
                   'offset_init': 0., 'validation_data': None}
    for key in options:
        assert inspect.signature(model.fit).parameters[key].kind is inspect.Parameter.KEYWORD_ONLY
    assert model.fit(X, y, **options) is model
    assert np.isfinite(model.decision_function(X)).all()


@pytest.mark.parametrize('name', ('GenDWD', 'KernGDWD'))
def test_weight_refusal_preserves_private_cv_cache_ownership(name, data):
    X, y, _ = data
    model = _estimator(name)
    model.cv_init(X)
    model.fit(X, y)
    cache = {key: value for key, value in vars(model).items()
             if key.startswith('_cv_') or key in ('_P0_eig', '_K_eig')}
    assert cache
    with pytest.raises(TypeError, match='sample_weight'):
        model.fit(X, y, sample_weight=np.ones(len(y)))
    _assert_unfitted(model, X)
    assert all(getattr(model, key) is value for key, value in cache.items())
    assert model.fit(X, y) is model
    np.testing.assert_allclose(model.decision_function(X),
                               _estimator(name).fit(X, y).decision_function(X),
                               rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('bootstrap', (False, True))
def test_bagging_uses_selected_rows_without_synthetic_fit_weights(bootstrap):
    X = np.random.default_rng(46).normal(size=(24, 3))
    y = np.tile(np.array([0, 1]), 12)
    X[:, 0] += .6 * (2 * y - 1)
    model = _estimator('GenDWD')
    bag = BaggingClassifier(estimator=model, n_estimators=3, n_jobs=1,
                            max_samples=.75, bootstrap=bootstrap, random_state=23)
    with config_context(enable_metadata_routing=False):
        bag.fit(X, y)
    assert len(bag.estimators_) == 3
    for fitted, rows, features in zip(bag.estimators_, bag.estimators_samples_,
                                      bag.estimators_features_):
        assert len(rows) == 18
        assert np.unique(y[rows]).size == 2
        if not bootstrap:
            assert np.unique(rows).size == len(rows)
        manual = clone(fitted).fit(X[np.ix_(rows, features)], y[rows])
        query = X[:, features]
        np.testing.assert_allclose(fitted.decision_function(query),
                                   manual.decision_function(query), rtol=1e-11, atol=1e-12)
        np.testing.assert_array_equal(fitted.predict(query), manual.predict(query))
