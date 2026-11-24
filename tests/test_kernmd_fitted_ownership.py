"""KernMD owns feature data and snapshots the kernel specification at fit.

Exact checks compare one fitted predictor before/after unrelated caller edits.
Independent kernel/layout/refit computations use stated floating-point bounds.
Callable objects themselves retain their identity; their external state is not
promised to be frozen. Precomputed fits retain only a shape-stable shared view.
"""
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from _search_resources import single_worker_search_parent

from dwd.kern_md import KernMD


TRAIN = np.array([
    [-3., 0., 1.], [-2., 1., 0.], [-1., -1., 2.], [-2., 0., -1.],
    [1., 2., 0.], [2., 1., -1.], [3., -1., 1.], [1., 0., 2.],
])
LABELS = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
QUERY = np.array([[-2., .5, 1.], [2., 1., -.5], [0., 0., 0.],
                  [.5, -1., 2.], [3., 2., -1.]])


def weighted_kernel(train, query, *, settings):
    """A real matrix-level kernel with nested mutable keyword values."""
    return (train * settings['weights'][0]) @ query.T + settings['offset'][0]


def _snapshot(array):
    return (array.shape, array.strides, array.dtype.str,
            array.flags.writeable, array.flags.c_contiguous,
            array.flags.f_contiguous, array.__array_interface__['data'][0],
            array.tobytes())


def _unfitted(model):
    learned = [name for name in vars(model)
               if (name.endswith('_') and not name.startswith('_')) or name == '_Xfit']
    assert learned == []
    with pytest.raises(NotFittedError):
        model.decision_function(QUERY)


def _make_training(layout):
    if layout == 'C64':
        owner = TRAIN.copy()
        return owner, owner
    if layout == 'F64':
        owner = np.array(TRAIN, order='F')
        return owner, owner
    if layout == 'strided64':
        owner = np.zeros((len(TRAIN), 6))
        owner[:, ::2] = TRAIN
        return owner[:, ::2], owner
    if layout == 'readonly64':
        owner = TRAIN.copy()
        view = owner.view()
        view.flags.writeable = False
        return view, owner
    dtype = {'C32': np.float32, 'integer': np.int64}[layout]
    owner = TRAIN.astype(dtype)
    return owner, owner


@pytest.mark.parametrize('layout', ['C64', 'F64', 'strided64', 'readonly64', 'C32', 'integer'])
@pytest.mark.parametrize('kernel', ['linear', 'rbf'])
def test_feature_storage_is_owned_without_changing_input_or_dtype(layout, kernel):
    train, owner = _make_training(layout)
    shared_alias = train.view()
    before = _snapshot(train)
    kws = {'gamma': .25} if kernel == 'rbf' else None
    model = KernMD(kernel=kernel, kernel_kws=kws).fit(train, LABELS)
    assert _snapshot(train) == before
    assert model._Xfit is not train
    assert not np.shares_memory(model._Xfit, owner)
    assert not np.shares_memory(model._Xfit, shared_alias)
    assert model._Xfit.dtype == train.dtype
    if layout in ('C64', 'C32', 'integer', 'readonly64'):
        assert model._Xfit.flags.c_contiguous
    if layout == 'F64':
        assert model._Xfit.flags.f_contiguous
    assert_array_equal(model._Xfit, train)
    query = QUERY.astype(train.dtype) if layout == 'C32' else QUERY.copy()
    query_before = _snapshot(query)
    scores = model.decision_function(query)
    labels = model.predict(query)
    alpha, intercept = model.dual_coef_.copy(), model.intercept_.copy()
    owner[...] = 17
    assert np.any(shared_alias != TRAIN)
    assert_array_equal(model._Xfit, TRAIN.astype(train.dtype))
    assert_array_equal(model.decision_function(query), scores)
    assert_array_equal(model.predict(query), labels)
    assert_array_equal(model.dual_coef_, alpha)
    assert_array_equal(model.intercept_, intercept)
    assert _snapshot(query) == query_before


@pytest.mark.parametrize('kernel', ['linear', 'rbf'])
def test_caller_shape_changes_cannot_change_training_row_or_feature_counts(kernel):
    train = TRAIN.copy()
    model = KernMD(kernel=kernel).fit(train, LABELS)
    before = model.decision_function(QUERY)
    train.shape = (train.size, 1)
    assert model._Xfit.shape == TRAIN.shape
    assert model.n_features_in_ == TRAIN.shape[1]
    assert_array_equal(model.decision_function(QUERY), before)


def test_named_kernel_settings_are_frozen_until_explicit_refit():
    kws = {'gamma': .25}
    model = KernMD(kernel='rbf', kernel_kws=kws).fit(TRAIN, LABELS)
    before = model.decision_function(QUERY)
    kws['gamma'] = 4.
    model.set_params(kernel='linear', kernel_kws=None)
    assert model.get_params()['kernel'] == 'linear'
    assert model.get_params()['kernel_kws'] is None
    assert model.kernel_ == 'rbf'
    assert model.kernel_kws_ == {'gamma': .25}
    assert_array_equal(model.decision_function(QUERY), before)
    clone_model = clone(model)
    _unfitted(clone_model)
    model.fit(TRAIN, LABELS)
    reference = clone_model.fit(TRAIN, LABELS)
    assert model.kernel_ == 'linear'
    assert model.kernel_kws_ is None
    assert_allclose(model.decision_function(QUERY), reference.decision_function(QUERY),
                    rtol=1e-13, atol=1e-13)
    assert np.max(np.abs(model.decision_function(QUERY) - before)) > .1


def test_nested_callable_keywords_are_deep_copied_but_callable_is_identical():
    settings = {'weights': [np.array([.5, 1., 2.])], 'offset': [.25]}
    kws = {'settings': settings}
    model = KernMD(kernel=weighted_kernel, kernel_kws=kws).fit(TRAIN, LABELS)
    before = model.decision_function(QUERY)
    assert model.kernel_ is weighted_kernel
    assert model.kernel_kws_ is not kws
    assert model.kernel_kws_['settings'] is not settings
    assert not np.shares_memory(model.kernel_kws_['settings']['weights'][0],
                                settings['weights'][0])
    settings['weights'][0][:] = 99
    settings['offset'][0] = -9
    settings['weights'].append(np.array([3., 4., 5.]))
    model.set_params(kernel='precomputed', kernel_kws={'unused': True})
    assert_array_equal(model.decision_function(QUERY), before)
    assert_array_equal(model.kernel_kws_['settings']['weights'][0], [.5, 1., 2.])
    assert model.kernel_kws_['settings']['offset'] == [.25]


def test_callable_self_kernel_receives_owned_identical_operands_and_snapshot_kwargs():
    train = TRAIN.copy()
    calls = []
    settings = {'weights': [np.array([.5, 1., 2.])], 'offset': [.25]}

    def callback(left, right, *, settings):
        calls.append((left, right, settings))
        return weighted_kernel(left, right, settings=settings)

    model = KernMD(kernel=callback, kernel_kws={'settings': settings}).fit(train, LABELS)
    assert len(calls) == 1
    left, right, received = calls[0]
    assert left is right is model._Xfit
    assert left is not train and not np.shares_memory(left, train)
    assert received is model.kernel_kws_['settings']
    assert received is not settings
    model.decision_function(QUERY)
    assert calls[1][0] is model._Xfit
    assert_array_equal(calls[1][1], QUERY)
    assert model.kernel_ is callback


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('readonly', [False, True])
def test_precomputed_view_shares_values_but_owns_shape_metadata(dtype, readonly):
    owner = rbf_kernel(TRAIN, gamma=.25).astype(dtype)
    gram = owner.view()
    gram.flags.writeable = not readonly
    before = _snapshot(gram)
    query = rbf_kernel(QUERY, TRAIN, gamma=.25).astype(dtype)
    model = KernMD(kernel='precomputed').fit(gram, LABELS)
    assert _snapshot(gram) == before
    assert model._Xfit is not gram
    assert np.shares_memory(model._Xfit, owner)
    assert model._Xfit.dtype == dtype
    scores = model.decision_function(query)
    expected = query @ model.dual_coef_[0] + model.intercept_[0]
    assert_allclose(scores, expected, rtol=1e-13, atol=1e-13)
    owner[:] = -999
    gram.shape = (gram.size, 1)
    assert model._Xfit.shape == (len(TRAIN), len(TRAIN))
    assert_array_equal(model.decision_function(query), scores)
    model.set_params(kernel='linear', kernel_kws={'unexpected': True})
    assert_array_equal(model.decision_function(query), scores)
    with pytest.raises(ValueError, match='one column per training observation'):
        model.decision_function(query[:, :-1])


@pytest.mark.parametrize('first,second', [('rbf', 'linear'), ('linear', 'precomputed'),
                                         ('precomputed', 'rbf')])
def test_refit_replaces_basis_and_specification_with_new_data(first, second):
    def inputs(features, kernel):
        return (rbf_kernel(features, gamma=.25) if kernel == 'precomputed'
                else features.copy())

    model = KernMD(kernel=first, kernel_kws=None).fit(inputs(TRAIN, first), LABELS)
    old_basis = model._Xfit
    changed = TRAIN[:6].copy()
    changed[:, 0] *= .5
    labels = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    kws = {'gamma': .25} if second == 'rbf' else None
    model.set_params(kernel=second, kernel_kws=kws)
    supplied = inputs(changed, second)
    model.fit(supplied, labels)
    reference = clone(model).fit(supplied, labels)
    query = (rbf_kernel(QUERY, changed, gamma=.25) if second == 'precomputed'
             else QUERY)
    assert model._Xfit is not old_basis
    assert model.kernel_ == second
    assert model._Xfit.shape[0] == 6
    assert_array_equal(model.classes_, ['a', 'b'])
    assert_allclose(model.decision_function(query), reference.decision_function(query),
                    rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('initially_fitted', [False, True])
def test_deepcopy_failure_preserves_exception_and_clears_all_fitted_state(initially_fitted):
    failure = RuntimeError('deliberate keyword deepcopy refusal')

    class RefuseCopy:
        def __deepcopy__(self, memo):
            raise failure

    model = KernMD()
    if initially_fitted:
        model.fit(TRAIN, LABELS)
    kwargs = {'settings': RefuseCopy()}
    model.set_params(kernel=weighted_kernel, kernel_kws=kwargs)
    before = _snapshot(TRAIN)
    with pytest.raises(RuntimeError) as caught:
        model.fit(TRAIN, LABELS)
    assert caught.value is failure
    assert model.kernel_kws is kwargs
    assert _snapshot(TRAIN) == before
    _unfitted(model)


@pytest.mark.parametrize('initially_fitted', [False, True])
@pytest.mark.parametrize('kind', ['error', 'interrupt'])
def test_kernel_callback_failure_clears_new_snapshots_and_preserves_exception(initially_fitted, kind):
    failure = (RuntimeError('deliberate callback failure') if kind == 'error'
               else KeyboardInterrupt('deliberate callback interrupt'))
    seen = []

    def callback(left, right):
        seen.append((left, right))
        raise failure

    model = KernMD()
    if initially_fitted:
        model.fit(TRAIN, LABELS)
    model.set_params(kernel=callback)
    before = _snapshot(TRAIN)
    with pytest.raises(type(failure)) as caught:
        model.fit(TRAIN, LABELS)
    assert caught.value is failure
    assert len(seen) == 1 and seen[0][0] is seen[0][1]
    assert not np.shares_memory(seen[0][0], TRAIN)
    assert model.kernel is callback
    assert _snapshot(TRAIN) == before
    _unfitted(model)
    model.set_params(kernel='linear').fit(TRAIN, LABELS)
    assert np.isfinite(model.decision_function(QUERY)).all()


@pytest.mark.parametrize('failure_kind', ['single_class', 'sparse_fit'])
def test_invalid_refit_removes_owned_basis_and_fitted_configuration(failure_kind):
    model = KernMD(kernel='rbf', kernel_kws={'gamma': .25}).fit(TRAIN, LABELS)
    if failure_kind == 'single_class':
        with pytest.raises(ValueError, match='exactly two classes'):
            model.fit(TRAIN, np.ones(len(TRAIN)))
    else:
        with pytest.raises(TypeError, match='[Ss]parse|dense'):
            model.fit(csr_matrix(TRAIN), LABELS)
    _unfitted(model)
    assert model.kernel == 'rbf' and model.kernel_kws == {'gamma': .25}


@pytest.mark.parametrize('kind', ['named', 'callable'])
def test_pickle_preserves_owned_basis_and_captured_settings(kind):
    train = TRAIN.copy()
    if kind == 'named':
        model = KernMD(kernel='rbf', kernel_kws={'gamma': .25})
    else:
        model = KernMD(kernel=weighted_kernel, kernel_kws={
            'settings': {'weights': [np.array([.5, 1., 2.])], 'offset': [.25]}})
    model.fit(train, LABELS)
    expected = model.decision_function(QUERY)
    model.set_params(kernel='precomputed', kernel_kws=None)
    restored = pickle.loads(pickle.dumps(model))
    train[:] = 17
    assert_array_equal(restored._Xfit, TRAIN)
    assert not np.shares_memory(restored._Xfit, train)
    assert_allclose(restored.decision_function(QUERY), expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('kind', ['named', 'callable', 'precomputed'])
def test_historical_models_without_snapshot_fields_keep_the_existing_fallback(kind):
    if kind == 'named':
        model = KernMD(kernel='rbf', kernel_kws={'gamma': .25}).fit(TRAIN, LABELS)
        query = QUERY
    elif kind == 'callable':
        model = KernMD(kernel=weighted_kernel, kernel_kws={
            'settings': {'weights': [np.array([.5, 1., 2.])], 'offset': [.25]}}).fit(TRAIN, LABELS)
        query = QUERY
    else:
        model = KernMD(kernel='precomputed').fit(rbf_kernel(TRAIN), LABELS)
        query = rbf_kernel(QUERY, TRAIN)
    expected = model.decision_function(query)
    del model.kernel_
    del model.kernel_kws_
    restored = pickle.loads(pickle.dumps(model))
    assert not hasattr(restored, 'kernel_') and not hasattr(restored, 'kernel_kws_')
    assert_allclose(restored.decision_function(query), expected, rtol=1e-13, atol=1e-13)
    # This preserves old-object compatibility, not retroactive snapshot ownership.


@pytest.mark.parametrize('batch', [None, 2, 10])
def test_fitted_laplacian_retains_sparse_query_ownership_and_batch_contract(batch):
    train = TRAIN.copy()
    model = KernMD(kernel='laplacian', kernel_kws={'gamma': .25}).fit(train, LABELS)
    expected = model.decision_function(QUERY)
    query = csr_matrix(QUERY)
    for row in range(len(QUERY)):
        start, stop = query.indptr[row:row + 2]
        query.indices[start:stop] = query.indices[start:stop][::-1]
        query.data[start:stop] = query.data[start:stop][::-1]
    query.has_sorted_indices = False
    query.has_canonical_format = False
    for array in (query.data, query.indices, query.indptr):
        array.flags.writeable = False
    before = [_snapshot(a) for a in (query.data, query.indices, query.indptr)]
    train[:] = 17
    model.set_params(kernel='linear', kernel_kws=None)
    actual = model.decision_function(query, prediction_batch_size=batch)
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert [_snapshot(a) for a in (query.data, query.indices, query.indptr)] == before
    assert not query.has_sorted_indices and not query.has_canonical_format


@pytest.mark.parametrize('batch', [None, 2])
def test_frozen_polynomial_keeps_nonfinite_prediction_refusal_and_recovery(batch):
    model = KernMD(kernel='poly', kernel_kws={'degree': 4}).fit(TRAIN, LABELS)
    expected = model.decision_function(QUERY)
    model.set_params(kernel='linear', kernel_kws=None)
    extreme = QUERY * 1e100
    before = _snapshot(extreme)
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(FloatingPointError, match='Nonfinite'):
            model.decision_function(extreme, prediction_batch_size=batch)
    assert _snapshot(extreme) == before
    assert_array_equal(model.decision_function(QUERY), expected)


@pytest.mark.parametrize('kind', ['features', 'precomputed'])
@single_worker_search_parent()
def test_sklearn_grid_search_clone_and_refit_keep_owned_final_predictor(kind):
    # KernMD has no package cv_init method; use its supported sklearn estimator API.
    if kind == 'features':
        train = TRAIN.copy()
        estimator = KernMD(kernel='rbf')
        grid = {'kernel_kws': [{'gamma': .25}, {'gamma': .5}]}
        query = QUERY
    else:
        train = rbf_kernel(TRAIN, gamma=.25)
        estimator = KernMD(kernel='precomputed')
        grid = {'kernel_kws': [None]}
        query = rbf_kernel(QUERY, TRAIN, gamma=.25)
    search = GridSearchCV(estimator, grid, cv=StratifiedKFold(n_splits=2),
                          n_jobs=1, error_score='raise', refit=True).fit(train, LABELS)
    model = search.best_estimator_
    assert model.kernel_ == estimator.kernel
    expected = model.decision_function(query)
    assert np.isfinite(search.cv_results_['mean_test_score']).all()
    _unfitted(clone(model))
    train[:] = 17
    train.shape = (train.size, 1)
    assert model._Xfit.shape[0] == len(TRAIN)
    assert_array_equal(search.decision_function(query), expected)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_linear_mean_difference_matches_independent_feature_space_formula(dtype):
    train = TRAIN.astype(dtype)
    query = QUERY.astype(dtype)
    model = KernMD(kernel='linear').fit(train, LABELS)
    positive = TRAIN[LABELS == 1].mean(axis=0)
    negative = TRAIN[LABELS == -1].mean(axis=0)
    expected = QUERY @ (positive - negative) - .5 * (positive @ positive - negative @ negative)
    # All input and class-mean values here are dyadic, so this is an exact oracle.
    assert_array_equal(model.decision_function(query), expected)
    assert_array_equal(model.predict(query), np.where(expected > 0, 1, -1))
