"""Public joint-default migration without changing private or fitted policies."""
import inspect
import pickle
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from dwd import _kernel_solver as solver
from dwd.cv import run_cv
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV, solve_gen_kern_dwd
from dwd.kern_md import KernMD


X = np.array([[-1., 0.], [-.5, .5], [0., -.5], [.5, .25],
              [1., .5], [.25, 1.], [-.25, -.75], [.75, -.25]])
Y = np.array([-3, 7, -3, 7, -3, 7, -3, 7])
QUERY = np.array([[.125, -.25], [.75, .375], [-.5, 0.]])
VALIDATION_Y = np.array([-3, 7, -3])
REMOVED = 'affine_computation.*auto.*removed.*joint'


def _model(**kwargs):
    options = dict(kernel='rbf', kernel_kws={'gamma': .25}, lambd=.1,
                   max_iter=2, stopping='fixed', random_state=31)
    options.update(kwargs)
    return KernGDWD(**options)


def _cv_model(**kwargs):
    options = dict(kernel='rbf', kernel_kws_vals=[{'gamma': .25}],
                   lambd_vals=[.1], q_vals=[1.], cv=2,
                   max_iter=1, stopping='fixed', random_state=31)
    options.update(kwargs)
    return KernGDWDCV(**options)


def _assert_same_fit(left, right):
    for name in ('dual_coef_', 'intercept_', 'objective_history_'):
        a, b = getattr(left, name), getattr(right, name)
        assert a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()
    for name in ('n_iter_', 'returned_iteration_', 'termination_reason_',
                 'backend_', 'prediction_precision_'):
        assert getattr(left, name) == getattr(right, name)
    assert left.affine_computation_ == right.affine_computation_ == 'joint'
    assert_array_equal(left.decision_function(QUERY), right.decision_function(QUERY))
    assert_array_equal(left.predict(QUERY), right.predict(QUERY))


@pytest.mark.parametrize('cls', [KernGDWD, KernGDWDCV])
def test_public_joint_default_and_other_control_defaults_are_explicit(cls):
    assert inspect.signature(cls).parameters['affine_computation'].default == 'joint'
    model = cls()
    assert model.get_params()['affine_computation'] == 'joint'
    assert clone(model).affine_computation == 'joint'
    assert model.backend == 'auto'
    assert model.acceleration == 'auto'
    assert model.residual_check_order == 'auto'
    assert model.initialization == 'zero'
    assert model.stopping == 'objective'
    assert model.max_iter == 100
    assert model.tol == 1e-6
    assert model.obj_tol == 1e-5
    assert model.set_params(affine_computation='standard') is model
    assert clone(model).affine_computation == 'standard'
    model.set_params(affine_computation='joint')
    assert clone(model).affine_computation == 'joint'


def test_private_low_level_and_mean_difference_defaults_are_not_promoted():
    assert inspect.signature(solver.solve_kernel).parameters['affine_computation'].default == 'standard'
    assert 'affine_computation' not in inspect.signature(solve_gen_kern_dwd).parameters
    assert 'affine_computation' not in KernMD().get_params()
    assert not hasattr(KernMD(), 'affine_computation_')
    # Missing requested fields use current defaults; fitted fallbacks are separate.
    for cls in (KernGDWD, KernGDWDCV):
        assert cls.affine_computation == 'joint'
        model = cls()
        del model.__dict__['affine_computation']
        assert clone(model).affine_computation == 'joint'


@pytest.mark.parametrize('cls', [KernGDWD, KernGDWDCV])
@pytest.mark.parametrize('assignment', ['constructor', 'set_params'])
def test_removed_auto_rejects_at_fit_including_cloned_requests(cls, assignment):
    options = {'affine_computation': 'auto'} if assignment == 'constructor' else {}
    model = _model(**options) if cls is KernGDWD else _cv_model(**options)
    if assignment == 'set_params':
        model.set_params(affine_computation='auto')
    model = clone(model)
    with patch.object(KernGDWD, '_compute_training_kernel',
                      side_effect=AssertionError('removed policy reached preparation')) as prepare:
        with pytest.raises(ValueError, match=REMOVED):
            model.fit(X, Y)
    prepare.assert_not_called()
    assert not hasattr(model, 'affine_computation_')
    assert not hasattr(model, 'dual_coef_')
    assert not hasattr(model, 'best_estimator_')


def test_removed_auto_rejects_cv_init_before_preparation():
    model = _model(affine_computation='auto')
    with patch.object(KernGDWD, '_compute_training_kernel',
                      side_effect=AssertionError('removed policy reached preparation')) as prepare:
        with pytest.raises(ValueError, match=REMOVED):
            model.cv_init(X)
    prepare.assert_not_called()
    assert not hasattr(model, '_cv_K')


def test_supplied_kernel_and_initial_state_do_not_bypass_removed_auto():
    alpha = (np.arange(len(Y), dtype=float) - 3.) / 32.
    saved = alpha.copy()
    with pytest.raises(ValueError, match=REMOVED):
        _model(affine_computation='auto').fit(
            X, Y, K=np.eye(len(Y)), alpha_init=alpha, offset_init=.125)
    assert_array_equal(alpha, saved)


@pytest.mark.parametrize('route', ['run_cv', 'sklearn', 'pipeline'])
def test_removed_auto_in_direct_and_nested_grids_is_rejected(route):
    model = _model()
    with pytest.raises(ValueError, match=REMOVED):
        if route == 'run_cv':
            run_cv(model, X, Y, {'affine_computation': ['auto']}, cv=2)
        elif route == 'sklearn':
            GridSearchCV(model, {'affine_computation': ['auto']},
                         cv=2, n_jobs=1, error_score='raise').fit(X, Y)
        else:
            GridSearchCV(Pipeline([('model', model)]),
                         {'model__affine_computation': ['auto']},
                         cv=2, n_jobs=1, error_score='raise').fit(X, Y)


@pytest.mark.parametrize('backend', ['auto', 'cholesky', 'spectral', 'lbfgs'])
@pytest.mark.parametrize('q', [1., 2.3])
def test_default_matches_explicit_joint_state_and_query_across_backends(backend, q):
    default = _model(backend=backend, q=q).fit(X, Y)
    explicit = _model(backend=backend, q=q, affine_computation='joint').fit(X, Y)
    _assert_same_fit(default, explicit)


@pytest.mark.parametrize('backend', ['auto', 'cholesky', 'spectral', 'lbfgs'])
def test_default_matches_explicit_joint_validation_and_selected_state(backend):
    options = dict(backend=backend, stopping='validation', max_iter=3, patience=2)
    default = _model(**options).fit(X, Y, validation_data=(QUERY, VALIDATION_Y))
    explicit = _model(affine_computation='joint', **options).fit(
        X, Y, validation_data=(QUERY, VALIDATION_Y))
    assert default.validation_history_
    assert default.validation_history_ == explicit.validation_history_
    _assert_same_fit(default, explicit)


def test_default_cv_matches_explicit_joint_folds_and_final_refit():
    default = _cv_model().fit(X, Y)
    explicit = _cv_model(affine_computation='joint').fit(X, Y)
    assert default.affine_computation == explicit.affine_computation == 'joint'
    assert default.affine_computation_ == explicit.affine_computation_ == 'joint'
    assert default.best_params_ == explicit.best_params_
    assert_array_equal(default.agg_cv_results_['mean_test_score'],
                       explicit.agg_cv_results_['mean_test_score'])
    _assert_same_fit(default.best_estimator_, explicit.best_estimator_)
    assert_array_equal(default.decision_function(QUERY), explicit.decision_function(QUERY))


@pytest.mark.parametrize('backend', ['auto', 'spectral'])
def test_default_joint_cache_reuses_equivalent_request_but_not_standard(backend):
    model = _model(backend=backend).cv_init(X)
    stored = model._cv_K
    assert model._cv_affine_computation == 'joint'
    model.set_params(affine_computation='joint')
    assert model._cv_cache_matches(X)
    with patch.object(model, '_compute_training_kernel',
                      side_effect=AssertionError('equivalent joint cache must be reused')):
        model.fit(X, Y)
    assert model._cv_K is stored
    assert model.affine_computation_ == 'joint'
    model.set_params(affine_computation='standard')
    assert not model._cv_cache_matches(X)
    model.cv_init(X).fit(X, Y)
    assert model._cv_affine_computation == model.affine_computation_ == 'standard'


@pytest.mark.parametrize('backend', ['auto', 'spectral'])
def test_synthetic_stale_auto_cache_is_rebuilt_not_relabelled(backend):
    model = _model(backend=backend).cv_init(X)
    # Synthetic old metadata only; this is not a claimed historical artifact.
    model._cv_affine_computation = 'auto'
    model._cv_K = np.full((len(Y), len(Y)), np.nan)
    model._K_eig = (np.full((len(Y), len(Y)), np.nan), np.full(len(Y), np.nan))
    assert not model._cv_cache_matches(X)
    construct = model._compute_training_kernel
    with patch.object(model, '_compute_training_kernel', wraps=construct) as prepare, \
            patch.object(solver, 'solve_kernel', wraps=solver.solve_kernel) as solve:
        model.fit(X, Y)
    prepare.assert_called_once()
    assert solve.call_args.kwargs['K_eig'] is None
    assert solve.call_args.kwargs['affine_computation'] == 'joint'
    assert np.isfinite(model.objective_history_).all()
    assert np.isfinite(model.decision_function(QUERY)).all()
    assert model._cv_affine_computation == 'auto'  # rejected evidence was not rewritten
    assert not model._cv_cache_matches(X)


@pytest.mark.parametrize('cls', [KernGDWD, KernGDWDCV])
def test_removed_request_does_not_reinterpret_fitted_policy_but_refit_fails_cleanly(cls):
    model = _model().fit(X, Y) if cls is KernGDWD else _cv_model().fit(X, Y)
    scores, labels = model.decision_function(QUERY), model.predict(QUERY)
    model.set_params(affine_computation='auto')
    restored = pickle.loads(pickle.dumps(model))
    for fitted in (model, restored):
        assert fitted.affine_computation_ == 'joint'
        assert_array_equal(fitted.decision_function(QUERY), scores)
        assert_array_equal(fitted.predict(QUERY), labels)
    with pytest.raises(ValueError, match=REMOVED):
        restored.fit(X, Y)
    assert not hasattr(restored, 'affine_computation_')
    assert not hasattr(restored, 'dual_coef_')
    assert not hasattr(restored, 'best_estimator_')
    with pytest.raises(NotFittedError):
        restored.predict(QUERY)


def test_removed_cv_init_request_clears_existing_predictor():
    model = _model().fit(X, Y).set_params(affine_computation='auto')
    with pytest.raises(ValueError, match=REMOVED):
        model.cv_init(X)
    assert not hasattr(model, 'affine_computation_')
    with pytest.raises(NotFittedError):
        model.predict(QUERY)
