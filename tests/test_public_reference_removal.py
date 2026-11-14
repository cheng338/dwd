"""Actual public API after retiring the implementation selector.

These tests use real production estimators, never the private-reference proxy.
Synthetic stale attributes/cache metadata model saved-request policy; they are
not claimed to be historical model artifacts or exact historical reproduction.
"""
import inspect
import pickle
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from dwd import _kernel_solver as solver
from dwd import gen_kern_dwd as kernel
from dwd.cv import run_cv
from dwd.gen_dwd import GenDWDCV
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV


X = np.array([[-1., 0.], [-.5, .5], [0., -.5], [.5, .25],
              [1., .5], [.25, 1.], [-.25, -.75], [.75, -.25]])
Y = np.array([-3, 7, -3, 7, -3, 7, -3, 7])
OPTIONS = dict(kernel='rbf', kernel_kws={'gamma': .25}, lambd=.1,
               max_iter=3, stopping='fixed', random_state=37)


@pytest.mark.parametrize('cls', [KernGDWD, KernGDWDCV])
def test_real_signatures_params_and_clone_have_no_retired_controls(cls):
    model = cls()
    for removed in ('implementation', 'solver_mode', 'implicit_P'):
        assert removed not in inspect.signature(cls).parameters
        assert removed not in model.get_params()
        assert removed not in clone(model).get_params()
    assert model.initialization == 'zero'
    assert model.backend == 'auto'
    assert model.acceleration == 'auto'
    assert model.stopping == 'objective'
    assert model.max_iter == 100
    assert model.obj_tol == 1e-5


@pytest.mark.parametrize('cls', [KernGDWD, KernGDWDCV])
@pytest.mark.parametrize('name,value', [
    ('implementation', 'reference'), ('implementation', 'optimized'),
    ('implementation', None), ('solver_mode', 'legacy'), ('solver_mode', 'schur'),
    ('implicit_P', True), ('implicit_P', False)])
def test_removed_constructor_and_set_params_requests_are_rejected(cls, name, value):
    with pytest.raises(TypeError, match=name):
        cls(**{name: value})
    model = cls()
    before = model.get_params()
    with pytest.raises(ValueError, match=name):
        model.set_params(**{name: value})
    assert model.get_params().keys() == before.keys()
    assert not hasattr(model, name)


@pytest.mark.parametrize('route', ['run_cv', 'sklearn', 'pipeline'])
@pytest.mark.parametrize('name', ['implementation', 'solver_mode', 'implicit_P'])
def test_removed_grid_keys_fail_before_any_kernel_preparation(route, name):
    model = KernGDWD(**OPTIONS)
    value = {'implementation': 'reference', 'solver_mode': 'legacy', 'implicit_P': True}[name]
    with patch.object(KernGDWD, '_compute_training_kernel',
                      side_effect=AssertionError('removed key reached preparation')) as prepare:
        with pytest.raises(ValueError, match=name):
            if route == 'run_cv':
                run_cv(model, X, Y, {name: [value]}, cv=2)
            elif route == 'sklearn':
                GridSearchCV(model, {name: [value]}, cv=2, n_jobs=1,
                             error_score='raise').fit(X, Y)
            else:
                GridSearchCV(Pipeline([('model', model)]),
                             {'model__' + name: [value]}, cv=2, n_jobs=1,
                             error_score='raise').fit(X, Y)
    prepare.assert_not_called()


def test_synthetic_saved_reference_attribute_cannot_select_future_public_fit():
    model = KernGDWD(**OPTIONS)
    model.implementation = 'reference'
    restored = pickle.loads(pickle.dumps(model))
    assert restored.implementation == 'reference'  # stored evidence is not erased
    assert 'implementation' not in restored.get_params()
    copied = clone(restored)
    assert type(copied) is KernGDWD
    assert not hasattr(copied, 'implementation')
    original = solver.solve_kernel
    with patch.object(solver, 'solve_kernel', wraps=original) as solve:
        restored.fit(X, Y)
    assert all(call.kwargs['implementation'] == 'optimized'
               for call in solve.call_args_list)
    assert_array_equal(solve.call_args_list[0].kwargs['alpha_init'], np.zeros(len(Y)))
    assert restored.diagnostics_['implementation'] == 'optimized'
    assert np.isfinite(restored.decision_function(X)).all()


@pytest.mark.parametrize('backend', ['auto', 'spectral'])
@pytest.mark.parametrize('old_provenance', ['reference', None])
def test_old_or_missing_reference_cache_provenance_rebuilds_before_solve(backend, old_provenance):
    model = KernGDWD(**OPTIONS, backend=backend).cv_init(X)
    if old_provenance is None:
        del model._cv_implementation
    else:
        model._cv_implementation = old_provenance
    model.implementation = 'reference'
    # Poison only the rejected old cache. Reusing either value would fail.
    model._cv_K = np.full((len(Y), len(Y)), np.nan)
    model._K_eig = (np.full((len(Y), len(Y)), np.nan), np.full(len(Y), np.nan))
    assert not model._cv_cache_matches(X)
    construct = model._compute_training_kernel
    with patch.object(model, '_compute_training_kernel', wraps=construct) as prepare, \
            patch.object(solver, 'solve_kernel', wraps=solver.solve_kernel) as solve:
        model.fit(X, Y)
    prepare.assert_called_once()
    assert solve.call_args.kwargs['K_eig'] is None
    assert solve.call_args.kwargs['implementation'] == 'optimized'
    assert np.isfinite(model.objective_history_).all()


def test_valid_optimized_eigen_cache_survives_irrelevant_stale_attribute():
    model = KernGDWD(**OPTIONS, backend='spectral').cv_init(X)
    eigen = model._K_eig
    assert eigen is not None
    assert model._cv_implementation == 'optimized'
    model.implementation = 'reference'
    assert model._cv_cache_matches(X)
    with patch.object(model, '_compute_training_kernel',
                      side_effect=AssertionError('valid optimized cache was lost')), \
            patch.object(solver, 'solve_kernel', wraps=solver.solve_kernel) as solve:
        model.fit(X, Y)
    assert solve.call_args.kwargs['K_eig'] is eigen
    assert solve.call_args.kwargs['implementation'] == 'optimized'


def test_public_supplied_eigenpairs_and_private_reference_keep_original_equations():
    K = np.eye(8) + np.ones((8, 8)) / 8.
    values, vectors = np.linalg.eigh(K)
    alpha = (np.arange(8) - 3.) / 32.
    offset = .125
    observed = []
    model = KernGDWD(kernel='precomputed', lambd=.1, backend='spectral',
                    acceleration=None, residual_check_order='refinement_first',
                    stopping='fixed', max_iter=3,
                    callback=lambda state: observed.append(state)).fit(
                        K, Y, K_eig=(vectors, values),
                        alpha_init=alpha, offset_init=offset)
    reference = solver.solve_kernel(
        K, np.where(Y == 7, 1., -1.), .1, implementation='reference',
        K_eig=(vectors, values), alpha_init=alpha, offset_init=offset,
        acceleration=None, residual_check_order='refinement_first',
        stopping='fixed', max_iter=3)
    assert [state['iteration'] for state in observed] == [0, 1, 2, 3]
    assert_array_equal(observed[0]['alpha'], alpha)
    assert observed[0]['offset'] == offset
    assert model.diagnostics_['implementation'] == 'optimized'
    assert reference['diagnostics']['implementation'] == 'reference'
    assert_allclose(model.dual_coef_[0], reference['alpha'], rtol=2e-9, atol=2e-10)
    assert_allclose(model.intercept_[0], reference['offset'], rtol=2e-9, atol=2e-10)
    assert_allclose(model.objective_history_, reference['objective_history'], rtol=2e-9, atol=2e-10)


def test_removed_ignored_eigensystem_keyword_rejects_before_eigen_work():
    with patch.object(kernel, 'validated_eigh',
                      side_effect=AssertionError('removed keyword reached LAPACK')) as eigen:
        with pytest.raises(TypeError, match='reference'):
            kernel.get_K_eig(np.eye(3), reference=True)
    eigen.assert_not_called()


@pytest.mark.parametrize('cls', [GenDWDCV, KernGDWDCV])
@pytest.mark.parametrize('name', ['solver_mode', 'not_a_parameter', 'unknown__value'])
def test_cv_invalid_root_rejects_without_rendering_or_changing_default_grids(cls, name):
    model = cls()
    before = model.get_params(deep=False)
    grid_values = {key: np.asarray(before[key]).copy() for key in ('lambd_vals', 'q_vals')}
    # Default grids must not be rendered merely to reject
    # a parameter name. Their public types, identities and values are retained.
    assert all(isinstance(before[key], tuple) for key in grid_values)
    with patch.object(cls, '__repr__', side_effect=AssertionError('invalid key rendered estimator')):
        with pytest.raises(ValueError, match=name):
            model.set_params(**{name: 'legacy'})
    after = model.get_params(deep=False)
    assert after.keys() == before.keys()
    assert all(after[key] is value for key, value in before.items())
    for key, value in grid_values.items():
        assert_array_equal(after[key], value)
    assert not hasattr(model, name)


@pytest.mark.parametrize('cls', [GenDWDCV, KernGDWDCV])
def test_cv_mixed_valid_and_invalid_roots_reject_before_any_assignment(cls):
    model = cls()
    before = model.get_params(deep=False)
    marker = model.observed_state_ = object()
    # Deliberately place a valid mutation first. The public-name preflight
    # rejects the whole request before sklearn could apply a partial update.
    with pytest.raises(ValueError, match='unknown'):
        model.set_params(max_iter=17, unknown='invalid')
    after = model.get_params(deep=False)
    assert after.keys() == before.keys()
    assert all(after[key] is value for key, value in before.items())
    assert model.observed_state_ is marker
    assert not hasattr(model, 'unknown')


@pytest.mark.parametrize('cls', [GenDWDCV, KernGDWDCV])
def test_cv_empty_and_supported_updates_preserve_sklearn_parameter_contract(cls):
    model = cls()
    assert model.set_params() is model
    lambd = np.array([.25, .5])
    q = np.array([1., 2.3])
    assert model.set_params(lambd_vals=lambd, q_vals=q, max_iter=7) is model
    assert model.lambd_vals is lambd
    assert model.q_vals is q
    assert model.max_iter == 7
    copied = clone(model)
    assert type(copied) is cls
    assert_array_equal(copied.lambd_vals, lambd)
    assert_array_equal(copied.q_vals, q)
    assert copied.max_iter == 7


class _NestedControl(BaseEstimator):
    def __init__(self, value=1):
        self.value = value


class _KernelCVWithNestedControl(KernGDWDCV):
    def __init__(self, nested=None):
        super().__init__()
        self.nested = nested


class _LinearCVWithNestedControl(GenDWDCV):
    def __init__(self, nested=None):
        super().__init__()
        self.nested = nested


@pytest.mark.parametrize('cls', [_LinearCVWithNestedControl, _KernelCVWithNestedControl])
def test_cv_subclass_nested_replacement_still_delegates_to_sklearn(cls):
    original = _NestedControl(1)
    replacement = _NestedControl(2)
    model = cls(nested=original)
    # Validate declared subclass roots, leaving nested-tail resolution to
    # sklearn. Its deferred nested updates target the replacement even when
    # the nested key precedes the replacement in the request.
    assert model.set_params(nested__value=3, nested=replacement) is model
    assert model.nested is replacement
    assert replacement.value == 3
    assert original.value == 1
    with pytest.raises(ValueError, match='missing'):
        model.set_params(nested__missing=9)
    assert replacement.value == 3
    assert not hasattr(replacement, 'missing')


def test_linear_cv_implicit_projection_remains_a_supported_parameter():
    model = GenDWDCV()
    assert model.implicit_P is True
    assert model.set_params(implicit_P=False) is model
    assert model.get_params()['implicit_P'] is False
    assert clone(model).implicit_P is False
