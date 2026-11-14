"""Missing-request policy regression checks; no historical refitting.

Fixed scope: 95 pytest cases, at most 108 solver-consuming KernGDWD.fit calls
plus 12 KernGDWDCV wrapper calls and two rejected pre-solver fit attempts.
All new fits use max_iter <= 3 on eight-row manufactured data, except the one
16-row historical-input refit pair (also max_iter=3). Historical pickle bytes
are trusted repository fixtures, verified before deserialization. Their three
selected prediction states are read only; removing fields from copies is
explicitly a synthetic projection, not a claim that the captured object lacked
those fields. The one historical-input refit pair uses the CURRENT solver.

Tests support source checkouts and installed packages with a separate test bundle.
Trusted historical fixtures are resolved relative to this test module.
Independent new fits use the predeclared ordinary comparison bound
rtol=2e-11, atol=2e-12; native eigendecompositions need not be bitwise identical.
Same saved-model state/readout checks and discrete policy/label checks remain
exact. These reporting comparisons do not alter solver acceptance criteria.
"""
import copy
import hashlib
import inspect
import io
import itertools
import json
import pickle
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.metrics.pairwise import rbf_kernel

from dwd import _kernel_solver as solver
from dwd.gen_dwd import GenDWD, GenDWDCV
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV, solve_gen_kern_dwd
from dwd.kern_md import KernMD
from dwd.kernel_utils import KernelScaler


CURRENT = {'affine_computation': 'joint', 'acceleration': 'auto',
           'residual_check_order': 'auto'}
EXPLICIT_OLD = {'affine_computation': 'standard', 'acceleration': None,
                'residual_check_order': 'refinement_first'}
FIELDS = tuple(CURRENT)
SUBSETS = tuple(subset for size in (1, 2, 3)
                for subset in itertools.combinations(FIELDS, size))
X = np.array([[-1., 0.], [-.5, .5], [0., -.5], [.5, .25],
              [1., .5], [.25, 1.], [-.25, -.75], [.75, -.25]])
Y = np.array([-3, 7, -3, 7, -3, 7, -3, 7])
Q = np.array([[.125, -.25], [.75, .375], [-.5, 0.]])
VY = np.array([-3, 7, -3])
FIXTURE_SHA = '904f7f707acfcbb865d5f3a54e191c025d3936198015bbc544442c311318581f'
MANIFEST_SHA = 'd274703a2d5661deff21025408c5b214376fe48d3444dcd8e32b6f9a4a0f410c'
RTOL, ATOL = 2e-11, 2e-12


def rbf_callable(first, second):
    return rbf_kernel(first, second, gamma=.25)


def _geometry(layout):
    if layout == 'precomputed':
        return rbf_kernel(X, X, gamma=.25), rbf_kernel(Q, X, gamma=.25), {'kernel': 'precomputed', 'kernel_kws': None}
    if layout == 'callable':
        return X.copy(), Q.copy(), {'kernel': rbf_callable, 'kernel_kws': None}
    options = {'kernel': 'rbf', 'kernel_kws': {'gamma': .25}}
    if layout == 'csr':
        return csr_matrix(X), csr_matrix(Q), options
    assert layout == 'dense'
    return X.copy(), Q.copy(), options


def _model(**kwargs):
    options = dict(kernel='rbf', kernel_kws={'gamma': .25}, lambd=.1, q=1.,
                   max_iter=3, stopping='fixed', initialization='zero', random_state=29)
    options.update(kwargs)
    return KernGDWD(**options)


def _cv_model(**kwargs):
    options = dict(kernel='rbf', kernel_kws_vals=[{'gamma': .25}],
                   lambd_vals=[.1], q_vals=[1.], cv=2, max_iter=2,
                   stopping='fixed', initialization='zero', random_state=29)
    options.update(kwargs)
    return KernGDWDCV(**options)


def _remove(model, fields=FIELDS, *, nested=False):
    for field in fields:
        model.__dict__.pop(field, None)
    if nested and hasattr(model, 'best_estimator_'):
        _remove(model.best_estimator_, fields)
    return model


def _roundtrip(model, transport):
    if transport == 'clone':
        return clone(model)
    if transport == 'pickle':
        return pickle.loads(pickle.dumps(model, protocol=4))
    assert transport == 'joblib'
    stream = io.BytesIO()
    joblib.dump(model, stream, compress=0, protocol=4)
    stream.seek(0)
    return joblib.load(stream)


def _equal_values(left, right):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert left.dtype == right.dtype
        assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal_values(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal_values(a, b)
    else:
        assert left == right


def _assert_same_fit(left, right, query):
    for field in ('dual_coef_', 'intercept_', 'objective_history_'):
        first, second = getattr(left, field), getattr(right, field)
        assert first.dtype == second.dtype
        assert_allclose(first, second, rtol=RTOL, atol=ATOL, err_msg=field)
    for field in ('n_iter_', 'returned_iteration_', 'termination_reason_',
                  'backend_', 'prediction_precision_', 'affine_computation_',
                  'requested_acceleration_', 'acceleration_',
                  'requested_residual_check_order_', 'residual_check_order_',
                  'converged_', 'stationarity_checked_'):
        _equal_values(getattr(left, field), getattr(right, field))
    assert_allclose(left.decision_function(query), right.decision_function(query),
                    rtol=RTOL, atol=ATOL)
    assert_array_equal(left.predict(query), right.predict(query))


def _assert_same_validation(first, second):
    assert len(first) == len(second)
    for left, right in zip(first, second):
        assert left.keys() == right.keys()
        for name in left:
            if isinstance(left[name], (float, np.floating)):
                assert_allclose(left[name], right[name], rtol=RTOL, atol=ATOL,
                                err_msg='validation ' + name)
            else:
                _equal_values(left[name], right[name])


def _assert_current_requests(model):
    params = model.get_params(deep=False)
    for field, expected in CURRENT.items():
        assert params[field] == expected


def _assert_resolved_current(model, *, supplied_eigen=False):
    _assert_current_requests(model)
    assert model.affine_computation_ == 'joint'
    assert model.requested_acceleration_ == 'auto'
    assert model.acceleration_ == (None if model.backend_ == 'lbfgs' else 'restart')
    assert model.requested_residual_check_order_ == 'auto'
    expected = ('adaptive' if model.backend_ == 'cholesky' and not supplied_eigen
                else 'refinement_first')
    assert model.residual_check_order_ == expected


def _fit_pair(layout='dense', *, q=1., backend='auto', missing=FIELDS,
              supplied_eigen=False):
    train, query, geometry = _geometry(layout)
    options = dict(geometry, q=q, backend=backend)
    absent = _remove(_model(**options, **EXPLICIT_OLD), missing)
    requested = dict(EXPLICIT_OLD)
    requested.update({field: CURRENT[field] for field in missing})
    fresh = _model(**options, **requested)
    fit_options = {}
    if supplied_eigen:
        matrix = rbf_kernel(X, X, gamma=.25)
        values, vectors = np.linalg.eigh(matrix)
        fit_options = {'K': matrix, 'K_eig': (vectors, values)}
    absent.fit(train, Y, **fit_options)
    fresh.fit(train, Y, **fit_options)
    _assert_same_fit(absent, fresh, query)
    if tuple(missing) == FIELDS:
        _assert_resolved_current(absent, supplied_eigen=supplied_eigen)
    return absent, fresh


def _assert_missing_fitted_fallback(model, query):
    """Synthetic old readout projection: preserve standard/ordinary fallback."""
    absent, explicit = copy.deepcopy(model), copy.deepcopy(model)
    _remove(absent, nested=True)
    states = [absent]
    if hasattr(absent, 'best_estimator_'):
        states.append(absent.best_estimator_)
    for state in states:
        state.__dict__.pop('affine_computation_', None)
        state.__dict__.pop('prediction_precision_', None)
        assert state.get_params()['affine_computation'] == 'joint'
    states = [explicit]
    if hasattr(explicit, 'best_estimator_'):
        states.append(explicit.best_estimator_)
    for state in states:
        state.affine_computation_ = 'standard'
        state.prediction_precision_ = 'ordinary'
    with patch('dwd._affine_scores.affine_scores',
               side_effect=AssertionError('missing fitted policy was reinterpreted as joint')):
        assert_array_equal(absent.decision_function(query), explicit.decision_function(query))
        assert_array_equal(absent.predict(query), explicit.predict(query))
    for field in ('dual_coef_', 'intercept_', 'classes_'):
        assert_array_equal(getattr(absent, field), getattr(model, field))


@pytest.mark.parametrize('cls', [KernGDWD, KernGDWDCV])
@pytest.mark.parametrize('missing', SUBSETS, ids=lambda value: '+'.join(value))
@pytest.mark.parametrize('transport', ['clone', 'pickle', 'joblib'])
def test_missing_requests_use_current_defaults_and_explicit_fields_survive(cls, missing, transport):
    model = cls(**EXPLICIT_OLD)
    _remove(model, missing)
    expected = dict(EXPLICIT_OLD)
    expected.update({field: CURRENT[field] for field in missing})
    for field, value in expected.items():
        assert getattr(model, field) == value
        assert model.get_params()[field] == value
    restored = _roundtrip(model, transport)
    for field, value in expected.items():
        assert restored.get_params()[field] == value
    # Reading/cloning/serializing never fills the original instance's gaps.
    assert all(field not in vars(model) for field in missing)
    assert not hasattr(restored, 'dual_coef_')


def test_other_classes_and_private_function_defaults_are_outside_this_migration():
    for cls in (GenDWD, GenDWDCV, KernMD, KernelScaler):
        assert not set(FIELDS).intersection(inspect.signature(cls).parameters)
    for cls in (KernGDWD, KernGDWDCV):
        signature = inspect.signature(cls)
        for field, current in CURRENT.items():
            assert getattr(cls, field) == signature.parameters[field].default == current
        for field in ('rbf_computation', 'exact_recovery', 'linear_action'):
            assert getattr(cls, field) == signature.parameters[field].default
    private = inspect.signature(solver.solve_kernel).parameters
    assert private['affine_computation'].default == 'standard'
    assert private['residual_check_order'].default == 'refinement_first'
    assert 'affine_computation' not in inspect.signature(solve_gen_kern_dwd).parameters


@pytest.mark.parametrize('layout', ['dense', 'csr', 'precomputed', 'callable'])
@pytest.mark.parametrize('q', [1., 2.3])
def test_all_missing_requests_refit_like_current_defaults_across_geometry(layout, q):
    _fit_pair(layout, q=q)


@pytest.mark.parametrize('backend', ['cholesky', 'spectral', 'lbfgs'])
@pytest.mark.parametrize('q', [1., 2.3])
def test_all_missing_requests_refit_like_current_defaults_across_backends(backend, q):
    _fit_pair(q=q, backend=backend)


@pytest.mark.parametrize('field', FIELDS)
def test_one_missing_request_does_not_replace_other_explicit_policies(field):
    absent, fresh = _fit_pair(missing=(field,))
    for retained in set(FIELDS) - {field}:
        assert absent.get_params()[retained] == fresh.get_params()[retained] == EXPLICIT_OLD[retained]


@pytest.mark.parametrize('backend', ['auto', 'cholesky'])
@pytest.mark.parametrize('q', [1., 2.3])
def test_missing_auto_policies_respect_supplied_eigenbasis_dispatch(backend, q):
    absent, _ = _fit_pair(q=q, backend=backend, supplied_eigen=True)
    assert absent.backend_ == ('spectral' if backend == 'auto' else 'cholesky')
    assert absent.residual_check_order_ == 'refinement_first'


@pytest.mark.parametrize('backend', ['auto', 'spectral'])
@pytest.mark.parametrize('q', [1., 2.3])
def test_missing_cv_requests_control_folds_and_current_final_refit(backend, q):
    absent = _remove(_cv_model(backend=backend, q_vals=[q], **EXPLICIT_OLD))
    fresh = _cv_model(backend=backend, q_vals=[q], **CURRENT)
    absent.fit(X, Y)
    fresh.fit(X, Y)
    _assert_current_requests(absent)
    _assert_resolved_current(absent.best_estimator_, supplied_eigen=(backend == 'spectral'))
    _assert_same_fit(absent.best_estimator_, fresh.best_estimator_, Q)
    assert absent.best_params_ == fresh.best_params_
    for field in ('mean_test_score', 'mean_train_score'):
        assert_array_equal(absent.agg_cv_results_[field], fresh.agg_cv_results_[field])
    for first, second in zip(absent.all_cv_results_, fresh.all_cv_results_):
        for field in ('params', 'test_score', 'train_score'):
            _equal_values(first[field], second[field])
    assert_allclose(absent.decision_function(Q), fresh.decision_function(Q),
                    rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('cls', [KernGDWD, KernGDWDCV])
@pytest.mark.parametrize('policy', ['standard', 'joint'])
@pytest.mark.parametrize('transport', ['pickle', 'joblib'])
def test_missing_requests_do_not_reinterpret_fitted_prediction_or_diagnostics(cls, policy, transport):
    options = dict(affine_computation=policy, acceleration=None,
                   residual_check_order='refinement_first')
    model = _model(**options) if cls is KernGDWD else _cv_model(**options)
    model.fit(X, Y)
    scores, labels = model.decision_function(Q), model.predict(Q)
    fitted = getattr(model, 'best_estimator_', model)
    learned = {name: copy.deepcopy(getattr(fitted, name)) for name in
               ('dual_coef_', 'intercept_', 'objective_history_', 'diagnostics_',
                'affine_computation_', 'requested_acceleration_', 'acceleration_',
                'requested_residual_check_order_', 'residual_check_order_')}
    _remove(model, nested=True)
    restored = _roundtrip(model, transport)
    for candidate in (model, restored):
        _assert_current_requests(candidate)
        assert candidate.affine_computation_ == policy
        assert_array_equal(candidate.decision_function(Q), scores)
        assert_array_equal(candidate.predict(Q), labels)
        returned = getattr(candidate, 'best_estimator_', candidate)
        for field, expected in learned.items():
            _equal_values(getattr(returned, field), expected)
    if policy == 'standard':
        _assert_missing_fitted_fallback(restored, Q)


@pytest.mark.parametrize('backend', ['auto', 'spectral'])
@pytest.mark.parametrize('old_marker', ['standard', 'auto', None])
def test_missing_current_affine_request_rebuilds_old_or_unattributed_cache(backend, old_marker):
    absent = _model(backend=backend, **EXPLICIT_OLD).cv_init(X)
    if old_marker is None:
        del absent._cv_affine_computation
    else:
        absent._cv_affine_computation = old_marker
    _remove(absent)
    absent._cv_K = np.full((len(Y), len(Y)), np.nan)
    absent._K_eig = (np.full((len(Y), len(Y)), np.nan), np.full(len(Y), np.nan))
    assert not absent._cv_cache_matches(X)
    construct = absent._compute_training_kernel
    with patch.object(absent, '_compute_training_kernel', wraps=construct) as prepare, \
            patch.object(solver, 'solve_kernel', wraps=solver.solve_kernel) as solve:
        absent.fit(X, Y)
    prepare.assert_called_once()
    assert solve.call_args.kwargs['K_eig'] is None
    assert solve.call_args.kwargs['affine_computation'] == 'joint'
    fresh = _model(backend=backend, **CURRENT).fit(X, Y)
    _assert_same_fit(absent, fresh, Q)
    assert getattr(absent, '_cv_affine_computation', None) == old_marker
    assert not absent._cv_cache_matches(X)


@pytest.mark.parametrize('backend', ['auto', 'spectral'])
def test_missing_solver_requests_can_reuse_policy_compatible_preparation(backend):
    absent = _model(backend=backend, affine_computation='joint',
                    acceleration=None, residual_check_order='refinement_first').cv_init(X)
    _remove(absent, ('acceleration', 'residual_check_order'))
    cache = absent._cv_K
    assert absent._cv_cache_matches(X)
    with patch.object(absent, '_compute_training_kernel',
                      side_effect=AssertionError('compatible cache must be reused')):
        absent.fit(X, Y)
    assert absent._cv_K is cache
    fresh = _model(backend=backend, **CURRENT).cv_init(X).fit(X, Y)
    _assert_same_fit(absent, fresh, Q)
    _assert_resolved_current(absent, supplied_eigen=(backend == 'spectral'))


@pytest.mark.parametrize('backend', ['auto', 'spectral'])
@pytest.mark.parametrize('q', [1., 2.3])
def test_new_validation_fit_uses_current_policy_without_reinterpreting_old_results(backend, q):
    options = dict(backend=backend, q=q, stopping='validation', patience=2)
    absent = _remove(_model(**options, **EXPLICIT_OLD))
    fresh = _model(**options, **CURRENT)
    absent.fit(X, Y, validation_data=(Q, VY))
    fresh.fit(X, Y, validation_data=(Q, VY))
    assert absent.validation_history_
    _assert_same_validation(absent.validation_history_, fresh.validation_history_)
    _assert_same_fit(absent, fresh, Q)


def test_optional_cvxpy_classes_are_outside_missing_kernel_policy_migration():
    pytest.importorskip('cvxpy', exc_type=ImportError)
    from dwd.socp_dwd import DWD
    from dwd.svm import SVM
    for cls in (DWD, SVM):
        assert not set(FIELDS).intersection(inspect.signature(cls).parameters)


def _historical(name):
    directory = Path(__file__).resolve().parent / 'fixtures'
    archive, receipt = directory / 'legacy_solver_retirement.npz', directory / 'legacy_solver_retirement.json'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == FIXTURE_SHA
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == MANIFEST_SHA
    manifest = json.loads(receipt.read_text(encoding='utf-8'))
    assert manifest['fixture_sha256'] == FIXTURE_SHA
    assert manifest['cases'][name]['fitted_solver_mode'] == 'legacy'
    with np.load(archive, allow_pickle=False) as arrays:
        raw = arrays[name + '__pickle'].tobytes()
        assert hashlib.sha256(raw).hexdigest() == manifest['cases'][name]['pickle_sha256']
        saved = {field: arrays[name + '__' + field].copy()
                 for field in ('X', 'y', 'query', 'scores', 'labels')}
    saved['model'] = pickle.loads(raw)
    assert saved['model'].solver_mode == 'legacy'
    return saved


@pytest.mark.parametrize('name', ['kernel_rbf', 'kernel_precomputed', 'kernel_cv'])
def test_genuine_historical_state_readout_survives_missing_request_projection(name):
    saved = _historical(name)
    model = saved['model']
    query = saved['query']
    scores, labels = model.decision_function(query), model.predict(query)
    arrays = {field: getattr(model, field).copy()
              for field in ('dual_coef_', 'intercept_', 'classes_')}
    original_requests = {field: getattr(model, field) for field in FIELDS}
    unmodified = _roundtrip(model, 'pickle')
    for field, value in original_requests.items():
        assert getattr(unmodified, field) == value
    assert_array_equal(unmodified.decision_function(query), scores)
    _remove(model, nested=True)
    for restored in (model, _roundtrip(model, 'pickle'), _roundtrip(model, 'joblib')):
        _assert_current_requests(restored)
        assert_array_equal(restored.decision_function(query), scores)
        assert_array_equal(restored.predict(query), labels)
        assert restored.solver_mode == 'legacy'  # historical attribution is retained
        for field, values in arrays.items():
            assert_array_equal(getattr(restored, field), values)
    _assert_missing_fitted_fallback(model, query)
    # This test promises current readout invariance, not cross-library bitwise
    # reproduction of the archived score array or a new historical fit.


def test_current_refit_of_genuine_historical_input_uses_current_missing_defaults():
    saved = _historical('kernel_rbf')
    historical = _remove(saved['model'])
    assert not historical._cv_cache_matches(saved['X'])
    parameters = historical.get_params(deep=False)
    _assert_current_requests(historical)
    assert parameters['max_iter'] == 3
    assert parameters['initialization'] == 'zero'
    current = KernGDWD(**parameters)
    historical.fit(saved['X'], saved['y'])
    current.fit(saved['X'], saved['y'])
    _assert_same_fit(historical, current, saved['query'])
    _assert_resolved_current(historical)
    assert historical.diagnostics_['implementation'] == 'optimized'


@pytest.mark.parametrize('explicit', [
    {'affine_computation': 'auto'}, {'initialization': 'auto'}])
def test_explicit_removed_values_are_not_silently_migrated(explicit):
    model = _remove(_model(), ('acceleration', 'residual_check_order'))
    model.set_params(**explicit)
    restored = _roundtrip(model, 'pickle')
    for name, value in explicit.items():
        assert restored.get_params()[name] == value
    with patch.object(solver, 'solve_kernel',
                      side_effect=AssertionError('invalid explicit policy reached the solver')) as solve:
        with pytest.raises(ValueError, match='auto.*removed'):
            restored.fit(X, Y)
    solve.assert_not_called()
