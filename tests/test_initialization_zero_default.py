"""Public zero-default migration and retained initialization contracts.

These tests use the real estimators. The test-only private reference scaffold
keeps its deliberate random start in its own numerical-comparison module.
"""
import inspect
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from _search_resources import single_worker_search_parent

from dwd.cv import run_cv
from dwd.gen_dwd import GenDWD, GenDWDCV, solve_gen_dwd
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV


X = np.array([[-1., 0.], [-.5, .5], [0., -.5], [.5, .25],
              [1., .5], [.25, 1.], [-.25, -.75], [.75, -.25]])
Y = np.array([-3, 7, -3, 7, -3, 7, -3, 7])
PUBLIC = (GenDWD, GenDWDCV, KernGDWD, KernGDWDCV)
REMOVED = "initialization.*auto.*removed.*zero"


def _model(family, **kwargs):
    options = dict(lambd=.1, max_iter=0, stopping='fixed')
    options.update(kwargs)
    if family == 'linear':
        return GenDWD(**options)
    return KernGDWD(kernel='rbf', kernel_kws={'gamma': .25}, **options)


def _dimension(family):
    return X.shape[1] if family == 'linear' else len(Y)


def _coefficients(model, family):
    return model.coef_[0] if family == 'linear' else model.dual_coef_[0]


def _supplied(family, value):
    return {('beta_init' if family == 'linear' else 'alpha_init'): value,
            'offset_init': .125}


def _assert_rng_equal(actual, expected):
    assert actual[0] == expected[0]
    assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def _cv_model(cls, **kwargs):
    options = dict(lambd_vals=[.1], q_vals=[1.], cv=2, max_iter=0,
                   stopping='fixed', random_state=43)
    options.update(kwargs)
    if cls is KernGDWDCV:
        options.update(kernel='rbf', kernel_kws_vals=[{'gamma': .25}])
    return cls(**options)


@pytest.mark.parametrize('entry', (*PUBLIC, solve_gen_dwd))
def test_public_default_spelling_is_zero_and_other_defaults_are_unchanged(entry):
    signature = inspect.signature(entry)
    assert signature.parameters['initialization'].default == 'zero'
    assert signature.parameters['stopping'].default == 'objective'
    assert signature.parameters['max_iter'].default == 100
    assert signature.parameters['obj_tol'].default == 1e-5
    assert signature.parameters['tol'].default == 1e-6


@pytest.mark.parametrize('cls', PUBLIC)
def test_clone_and_set_params_keep_supported_initialization_choices(cls):
    model = cls()
    assert model.get_params()['initialization'] == 'zero'
    assert clone(model).get_params()['initialization'] == 'zero'
    assert model.set_params(initialization='random') is model
    assert clone(model).get_params()['initialization'] == 'random'
    model.set_params(initialization='zero')
    assert clone(model).get_params()['initialization'] == 'zero'


@pytest.mark.parametrize('cls', PUBLIC)
@pytest.mark.parametrize('assignment', ['constructor', 'set_params'])
def test_removed_auto_is_rejected_at_fit_including_cloned_requests(cls, assignment):
    options = {'initialization': 'auto'} if assignment == 'constructor' else {}
    model = (_cv_model(cls, **options) if cls in (GenDWDCV, KernGDWDCV)
             else _model('linear' if cls is GenDWD else 'kernel', **options))
    if assignment == 'set_params':
        model.set_params(initialization='auto')
    model = clone(model)
    with pytest.raises(ValueError, match=REMOVED):
        model.fit(X, Y)
    assert not hasattr(model, 'coef_')
    assert not hasattr(model, 'dual_coef_')
    assert not hasattr(model, 'best_estimator_')


@pytest.mark.parametrize('family', ['linear', 'kernel'])
def test_explicit_initial_values_do_not_bypass_removed_auto(family):
    initial = np.arange(_dimension(family), dtype=float) / 32.
    saved = initial.copy()
    with pytest.raises(ValueError, match=REMOVED):
        _model(family, initialization='auto').fit(X, Y, **_supplied(family, initial))
    assert_array_equal(initial, saved)


@pytest.mark.parametrize('supplied', [False, True])
def test_public_linear_function_also_rejects_removed_auto(supplied):
    options = _supplied('linear', np.array([.125, -.25])) if supplied else {}
    with pytest.raises(ValueError, match=REMOVED):
        solve_gen_dwd(X, Y, lambd=.1, max_iter=0,
                      initialization='auto', **options)


@pytest.mark.parametrize('family', ['linear', 'kernel'])
@pytest.mark.parametrize('route', ['run_cv', 'sklearn', 'pipeline'])
def test_removed_auto_grid_value_is_not_accepted_or_reinterpreted(family, route):
    model = _model(family)
    with pytest.raises(ValueError, match=REMOVED):
        if route == 'run_cv':
            run_cv(model, X, Y, {'initialization': ['auto']}, cv=2)
        elif route == 'sklearn':
            GridSearchCV(model, {'initialization': ['auto']}, cv=2, n_jobs=1,
                         error_score='raise').fit(X, Y)
        else:
            GridSearchCV(Pipeline([('model', model)]),
                         {'model__initialization': ['auto']}, cv=2, n_jobs=1,
                         error_score='raise').fit(X, Y)


@pytest.mark.parametrize('family', ['linear', 'kernel'])
@pytest.mark.parametrize('q', [1., 2.3])
def test_omitted_initialization_matches_explicit_zero_after_one_update(family, q):
    default = _model(family, q=q, max_iter=1, random_state=71).fit(X, Y)
    explicit = _model(family, q=q, max_iter=1, random_state=71,
                      initialization='zero').fit(X, Y)
    for a, b in ((_coefficients(default, family), _coefficients(explicit, family)),
                 (default.intercept_, explicit.intercept_),
                 (default.objective_history_, explicit.objective_history_)):
        assert a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()
    assert default.n_iter_ == explicit.n_iter_ == 1
    assert_array_equal(default.decision_function(X), explicit.decision_function(X))


def test_public_linear_function_default_matches_explicit_zero():
    default = solve_gen_dwd(X, Y, lambd=.1, max_iter=1, stopping='fixed')
    explicit = solve_gen_dwd(X, Y, lambd=.1, max_iter=1, stopping='fixed',
                             initialization='zero')
    for actual, expected in zip(default, explicit):
        assert_array_equal(actual, expected)


@pytest.mark.parametrize('family', ['linear', 'kernel'])
@pytest.mark.parametrize('rng_kind', ['global', 'object'])
def test_default_zero_does_not_consume_global_or_supplied_rng(family, rng_kind):
    saved_global = np.random.get_state()
    try:
        np.random.seed(71)
        rng = np.random.RandomState(19)
        before_global, before_rng = np.random.get_state(), rng.get_state()
        model = _model(family, random_state=None if rng_kind == 'global' else rng).fit(X, Y)
        assert_array_equal(_coefficients(model, family), np.zeros(_dimension(family)))
        assert model.intercept_[0] == 0.
        _assert_rng_equal(np.random.get_state(), before_global)
        _assert_rng_equal(rng.get_state(), before_rng)
    finally:
        np.random.set_state(saved_global)


@pytest.mark.parametrize('family', ['linear', 'kernel'])
@pytest.mark.parametrize('rng_kind', ['global', 'object', 'integer'])
def test_random_start_keeps_exact_draw_and_rng_advancement_contract(family, rng_kind):
    saved_global = np.random.get_state()
    try:
        seed = 23
        np.random.seed(seed)
        before_global = np.random.get_state()
        expected_rng = np.random.RandomState(seed)
        expected = expected_rng.normal(size=_dimension(family))
        if family == 'kernel':
            expected /= np.linalg.norm(expected)
        rng = np.random.RandomState(seed)
        argument = {'global': None, 'object': rng, 'integer': seed}[rng_kind]
        model = _model(family, initialization='random', random_state=argument).fit(X, Y)
        assert_array_equal(_coefficients(model, family), expected)
        assert model.intercept_[0] == 0.
        if rng_kind == 'global':
            _assert_rng_equal(np.random.get_state(), expected_rng.get_state())
        else:
            _assert_rng_equal(np.random.get_state(), before_global)
            if rng_kind == 'object':
                _assert_rng_equal(rng.get_state(), expected_rng.get_state())
            else:
                repeated = _model(family, initialization='random', random_state=seed).fit(X, Y)
                assert_array_equal(_coefficients(repeated, family), expected)
    finally:
        np.random.set_state(saved_global)


@pytest.mark.parametrize('family', ['linear', 'kernel'])
def test_offset_only_does_not_suppress_requested_random_coefficients(family):
    rng = np.random.RandomState(23)
    expected_rng = np.random.RandomState(23)
    expected = expected_rng.normal(size=_dimension(family))
    if family == 'kernel':
        expected /= np.linalg.norm(expected)
    model = _model(family, initialization='random', random_state=rng).fit(
        X, Y, offset_init=.125)
    assert_array_equal(_coefficients(model, family), expected)
    assert model.intercept_[0] == .125
    _assert_rng_equal(rng.get_state(), expected_rng.get_state())


@pytest.mark.parametrize('family', ['linear', 'kernel'])
@pytest.mark.parametrize('initialization', ['zero', 'random'])
@pytest.mark.parametrize('rng_kind', ['global', 'object'])
def test_supplied_state_has_precedence_without_mutation_or_rng_use(family, initialization, rng_kind):
    saved_global = np.random.get_state()
    try:
        np.random.seed(71)
        rng = np.random.RandomState(19)
        before_global, before_rng = np.random.get_state(), rng.get_state()
        initial = (np.arange(_dimension(family), dtype=float) - 2.) / 32.
        before = initial.copy()
        model = _model(family, initialization=initialization,
                       random_state=None if rng_kind == 'global' else rng).fit(
                           X, Y, **_supplied(family, initial))
        assert_array_equal(_coefficients(model, family), before)
        assert model.intercept_[0] == .125
        assert_array_equal(initial, before)
        _assert_rng_equal(np.random.get_state(), before_global)
        _assert_rng_equal(rng.get_state(), before_rng)
    finally:
        np.random.set_state(saved_global)


@pytest.mark.parametrize('family', ['linear', 'kernel'])
def test_solver_updates_do_not_mutate_the_actual_supplied_coefficient_array(family):
    initial = (np.arange(_dimension(family), dtype=float) - 2.) / 32.
    before = initial.copy()
    model = _model(family, initialization='random', max_iter=1).fit(
        X, Y, **_supplied(family, initial))
    assert model.n_iter_ == 1
    assert_array_equal(initial, before)


@pytest.mark.parametrize('cls,base', [(GenDWDCV, GenDWD), (KernGDWDCV, KernGDWD)])
@pytest.mark.parametrize('requested', [None, 'zero', 'random'])
def test_cv_forwards_the_initialization_to_every_fold_and_final_refit(cls, base, requested):
    calls = []
    original = base.fit

    def record(model, *args, **kwargs):
        calls.append((model.initialization, model.random_state, len(args[1])))
        return original(model, *args, **kwargs)

    options = {} if requested is None else {'initialization': requested}
    model = _cv_model(cls, **options)
    with patch.object(base, 'fit', new=record):
        model.fit(X, Y)
    expected = 'zero' if requested is None else requested
    assert calls == [(expected, 43, 4), (expected, 43, 4), (expected, 43, 8)]
    assert type(model.best_estimator_) is base
    assert model.best_estimator_.initialization == expected
    assert model.best_estimator_.random_state == 43
    assert_array_equal(model.predict(X), model.best_estimator_.predict(X))


@pytest.mark.parametrize('family', ['linear', 'kernel'])
@single_worker_search_parent()
def test_valid_grid_keeps_zero_and_random_as_distinct_requests(family):
    search = GridSearchCV(_model(family, random_state=29),
                          {'initialization': ['zero', 'random']},
                          cv=2, n_jobs=1, error_score='raise').fit(X, Y)
    assert [entry['initialization'] for entry in search.cv_results_['params']] == ['zero', 'random']
    assert search.best_estimator_.initialization == search.best_params_['initialization']
    assert type(search.best_estimator_) is (GenDWD if family == 'linear' else KernGDWD)
