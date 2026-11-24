"""Public shared-fold search parity, ownership, lifecycle and fallback contracts.

Full decision tables are compared to stock sklearn; elapsed columns are not
equality assertions. Delegation spies check original objects and options reach
stock fit, without claiming unsupported inputs are accepted by sklearn. The
private token test checks the finite immutable-storage invariant directly.
"""

import copy
import pickle
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import dia_array
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from dwd.kern_md import KernMD
from dwd import model_selection as selection
from dwd.model_selection import fit_shared_fold_search


pytestmark = pytest.mark.usefixtures("numerical_search_parent")


def _data():
    labels = np.where(np.arange(24) % 2, 1, -1)
    features = ((np.arange(24 * 4).reshape(24, 4) * 7) % 29 - 14) / 8.
    features[:, 0] += labels * 2.
    folds = []
    for fold in range(3):
        test = np.array([i for i in range(len(labels)) if i % 3 == fold], dtype=np.intp)
        train = np.array([i for i in range(len(labels)) if i % 3 != fold], dtype=np.intp)
        folds.append((train, test))
    return features, labels, folds


def _search(kernel='rbf', scoring=None, *, candidates=2, folds=None, refit=True):
    if kernel == 'linear':
        grid = {'kernel_kws': [None, {}]}
    else:
        gammas = [.125, .5] if candidates == 2 else [.03125, .0625, .125, .25, .5, 1., 2., 4.]
        grid = {'kernel_kws': [{'gamma': gamma} for gamma in gammas]}
    return GridSearchCV(KernMD(kernel=kernel), grid, cv=folds, scoring=scoring,
                        n_jobs=1, refit=refit, error_score='raise',
                        return_train_score=True)


def _table_equal(actual, expected):
    assert set(actual.cv_results_) == set(expected.cv_results_)
    for name, expected_values in expected.cv_results_.items():
        if name.endswith('_time'):
            # Elapsed observations naturally differ; do not erase score fields.
            continue
        actual_values = actual.cv_results_[name]
        if name == 'params':
            assert actual_values == expected_values
        elif np.ma.isMaskedArray(expected_values):
            assert_array_equal(np.ma.getmaskarray(actual_values),
                               np.ma.getmaskarray(expected_values))
            assert_array_equal(np.ma.asarray(actual_values).compressed(),
                               np.ma.asarray(expected_values).compressed())
        else:
            assert_array_equal(actual_values, expected_values)
    assert actual.best_index_ == expected.best_index_
    assert actual.best_params_ == expected.best_params_
    assert actual.best_score_ == expected.best_score_
    assert actual.n_splits_ == expected.n_splits_ == 3
    assert type(actual.estimator) is KernMD
    assert hasattr(actual, 'best_estimator_') == hasattr(expected, 'best_estimator_')
    if hasattr(expected, 'best_estimator_'):
        assert type(actual.best_estimator_) is KernMD
        assert_array_equal(actual.best_estimator_.classes_, expected.best_estimator_.classes_)
        assert_array_equal(actual.best_estimator_.dual_coef_, expected.best_estimator_.dual_coef_)
        assert_allclose(actual.best_estimator_.intercept_, expected.best_estimator_.intercept_,
                        rtol=1e-13, atol=1e-13)


def _predictions_equal(actual, expected, query):
    assert_allclose(actual.decision_function(query), expected.decision_function(query),
                    rtol=1e-13, atol=1e-13)
    assert_array_equal(actual.predict(query), expected.predict(query))


def _snapshot(array):
    return (array.shape, array.strides, array.dtype.str, array.flags.writeable,
            array.tobytes())


def _is_fitted(estimator):
    try:
        check_is_fitted(estimator)
    except NotFittedError:
        return False
    return True


def _capture_private_stores(monkeypatch):
    stores = []
    original = selection._FoldDataset

    class ObservedStore(original):
        def __init__(self, X):
            super().__init__(X)
            stores.append(self)

    monkeypatch.setattr(selection, '_FoldDataset', ObservedStore)
    return stores


def _without_private_references(search, monkeypatch):
    assert not any(isinstance(value, (selection._FoldDataset, selection._FoldToken,
                                     selection._SharedFoldKernMD))
                   for value in vars(search).values())
    if hasattr(search, 'best_estimator_'):
        assert type(search.best_estimator_) is KernMD
    # A public search, including a failed one, must not require private helper
    # classes when deserialized. This also checks nested scorer references.
    payload = pickle.dumps(search)
    with monkeypatch.context() as context:
        context.setitem(sys.modules, selection.__name__, None)
        return pickle.loads(payload)


@pytest.mark.parametrize('kernel,scoring,candidates', [
    ('linear', None, 2), ('linear', 'accuracy', 2),
    ('rbf', None, 2), ('rbf', 'accuracy', 2), ('rbf', 'accuracy', 8),
])
def test_supported_search_matches_complete_stock_score_table(kernel, scoring, candidates):
    X, y, folds = _data()
    query = X[::3].copy() + .125
    X_before, y_before = _snapshot(X), _snapshot(y)
    folds_before = [(_snapshot(train), _snapshot(test)) for train, test in folds]
    shared = _search(kernel, scoring, candidates=candidates, folds=folds)
    original_estimator = shared.estimator
    stock = _search(kernel, scoring, candidates=candidates, folds=folds).fit(X, y)
    result = fit_shared_fold_search(shared, X, y)
    assert result is shared
    assert shared.estimator is original_estimator
    if scoring is None:
        assert shared.scorer_._estimator is original_estimator
    assert shared.shared_fold_info['route'] == 'shared'
    assert shared.shared_fold_info['completed'] is True
    assert shared.shared_fold_info['borrowed_fits'] == candidates * len(folds)
    assert shared.shared_fold_info['cache_hits'] > 0
    _table_equal(shared, stock)
    _predictions_equal(shared, stock, query)
    assert not np.shares_memory(shared.best_estimator_._Xfit, X)
    assert _snapshot(X) == X_before and _snapshot(y) == y_before
    assert [(_snapshot(train), _snapshot(test)) for train, test in folds] == folds_before


@pytest.mark.parametrize('layout', ['C', 'F', 'strided', 'readonly-alias'])
def test_shared_search_final_model_survives_caller_value_and_shape_edits(layout):
    values, y, folds = _data()
    if layout == 'strided':
        owner = np.zeros((len(values), 2 * values.shape[1]))
        owner[:, ::2] = values
        X = owner[:, ::2]
    elif layout == 'F':
        owner = np.array(values, order='F')
        X = owner
    else:
        owner = values.copy()
        X = owner.view() if layout == 'readonly-alias' else owner
        if layout == 'readonly-alias':
            X.flags.writeable = False
    before = _snapshot(X)
    query = values[::3].copy() + .125
    stock = _search(folds=folds).fit(values, y)
    shared = _search(folds=folds)
    fit_shared_fold_search(shared, X, y)
    assert shared.shared_fold_info['route'] == 'shared'
    assert _snapshot(X) == before
    _table_equal(shared, stock)
    _predictions_equal(shared, stock, query)
    expected = shared.decision_function(query).copy()
    assert not np.shares_memory(shared.best_estimator_._Xfit, owner)
    owner[...] = 17
    # Shape metadata of the caller's supplied view must also be independent.
    X.shape = (X.shape[0], 1, X.shape[1])
    assert shared.best_estimator_._Xfit.shape == values.shape
    assert_array_equal(shared.best_estimator_._Xfit, values)
    assert_array_equal(shared.decision_function(query), expected)


def test_shared_search_pickle_and_explicit_model_refit_remain_public_and_owned(monkeypatch):
    X, y, folds = _data()
    stores = _capture_private_stores(monkeypatch)
    shared = _search(folds=folds)
    fit_shared_fold_search(shared, X, y)
    assert len(stores) == 1 and stores[0]._tokens == {}
    # Published counters describe active-search storage, before release. The
    # test deliberately retains the store, so it makes no garbage-collection
    # or process-memory claim from this observation.
    assert shared.shared_fold_info['cached_subsets'] == 6
    assert shared.shared_fold_info['cached_bytes'] > 0
    query = X[::3].copy() + .125
    expected = shared.decision_function(query).copy()
    restored = _without_private_references(shared, monkeypatch)
    assert type(restored) is GridSearchCV
    assert type(restored.estimator) is KernMD
    assert type(restored.best_estimator_) is KernMD
    assert restored.shared_fold_info == shared.shared_fold_info
    _table_equal(restored, shared)
    X[:] = -19
    assert_array_equal(restored.decision_function(query), expected)
    replacement, labels, _ = _data()
    replacement[:, 1] *= .5
    model = restored.best_estimator_
    model.fit(replacement, labels)
    stock = KernMD(**model.get_params(deep=False)).fit(replacement, labels)
    assert not np.shares_memory(model._Xfit, replacement)
    _predictions_equal(model, stock, query)
    frozen = model.decision_function(query).copy()
    replacement[:] = 31
    assert_array_equal(model.decision_function(query), frozen)


def _callable_kernel(left, right, *, config):
    return (left * config['weights'][0]) @ right.T + config['offset'][0]


def _custom_accuracy(model, X, y):
    return float(np.count_nonzero(model.predict(X) == y) / len(y))


@pytest.mark.parametrize('kind', ['callable', 'precomputed', 'custom-scorer'])
def test_unsupported_search_uses_stock_and_preserves_complete_results(kind):
    X, y, folds = _data()
    query = X[::3].copy() + .125
    if kind == 'callable':
        estimator = KernMD(kernel=_callable_kernel, kernel_kws={
            'config': {'weights': [np.array([.5, 1., 2., .25])], 'offset': [.25]}})
        grid, scoring = {'naive_bayes': [False]}, None
    elif kind == 'precomputed':
        query = query @ X.T
        X = X @ X.T
        estimator, grid, scoring = KernMD(kernel='precomputed'), {'kernel_kws': [None]}, None
    else:
        estimator = KernMD(kernel='rbf')
        grid = {'kernel_kws': [{'gamma': .125}, {'gamma': .5}]}
        scoring = _custom_accuracy
    before = _snapshot(X)
    shared = GridSearchCV(estimator, grid, cv=folds, scoring=scoring, n_jobs=1,
                          refit=True, error_score='raise', return_train_score=True)
    stock = copy.deepcopy(shared).fit(X, y)
    original_estimator = shared.estimator
    assert fit_shared_fold_search(shared, X, y) is shared
    assert shared.estimator is original_estimator
    assert shared.shared_fold_info['route'] == 'stock'
    assert shared.shared_fold_info['reason']
    _table_equal(shared, stock)
    _predictions_equal(shared, stock, query)
    assert _snapshot(X) == before
    if kind != 'precomputed':
        assert not np.shares_memory(shared.best_estimator_._Xfit, X)


def test_svc_fallback_with_int32_dia_array_matches_dense_search():
    # Older SciPy widened the indices during DIA-to-CSR conversion on Linux.
    data = np.tile(np.arange(1, 5, dtype=np.int16), (3, 1))
    offsets = np.array([0, -1, 2], dtype=np.int32)
    X = dia_array((data, offsets), shape=(4, 4))
    y = np.array([0, 1, 0, 1])
    assert X.offsets.dtype == np.int32
    search = GridSearchCV(SVC(), {'C': [1.]}, cv=2, n_jobs=1, error_score='raise')
    dense = X.toarray()
    expected = copy.deepcopy(search).fit(dense, y)
    actual = fit_shared_fold_search(search, X, y)
    assert actual.shared_fold_info['route'] == 'stock'
    for split in ('split0_test_score', 'split1_test_score'):
        assert_array_equal(actual.cv_results_[split], expected.cv_results_[split])
    assert_array_equal(actual.predict(X), expected.predict(dense))
    assert_allclose(actual.decision_function(X), expected.decision_function(dense),
                    rtol=1e-10, atol=1e-10)


class _ArrayProvider:
    def __init__(self, values):
        self.values = values
        self.conversions = 0

    def __array__(self, dtype=None, copy=None):
        self.conversions += 1
        return np.asarray(self.values, dtype=dtype)


class _OnePass:
    def __init__(self, values):
        self.values = values
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self.values)


@pytest.mark.parametrize('kind', [
    'parallel', 'callable-refit', 'float32', 'numeric-object', 'array-provider',
    'fold-iterator', 'grid-iterator', 'fit-metadata', 'callback',
])
def test_unsupported_configuration_delegates_original_objects_without_preconsumption(
        kind, monkeypatch):
    X, y, folds = _data()
    search = _search(folds=folds)
    fit_params = {}
    observed = None
    if kind == 'parallel':
        search.n_jobs = 2
    elif kind == 'callable-refit':
        search.refit = lambda results: 0
    elif kind == 'float32':
        X = X.astype(np.float32)
    elif kind == 'numeric-object':
        X = X.astype(object)
    elif kind == 'array-provider':
        X = observed = _ArrayProvider(X)
    elif kind == 'fold-iterator':
        search.cv = observed = _OnePass(folds)
    elif kind == 'grid-iterator':
        search.param_grid = observed = _OnePass([search.param_grid])
    elif kind == 'fit-metadata':
        fit_params = {'sample_weight': np.ones(len(y))}
    else:
        search._skl_callbacks = [object()]
    original_estimator = search.estimator
    original_cv, original_grid = search.cv, search.param_grid
    original_jobs, original_refit = search.n_jobs, search.refit
    calls = []

    def observe_stock_fit(self, data, targets, **params):
        calls.append((self, data, targets, params))
        return self

    monkeypatch.setattr(GridSearchCV, 'fit', observe_stock_fit)
    assert fit_shared_fold_search(search, X, y, **fit_params) is search
    assert len(calls) == 1
    actual_search, actual_X, actual_y, params = calls[0]
    assert actual_search is search and actual_X is X and actual_y is y
    assert params.keys() == fit_params.keys()
    assert all(params[key] is fit_params[key] for key in fit_params)
    assert search.estimator is original_estimator
    assert search.cv is original_cv and search.param_grid is original_grid
    assert search.n_jobs == original_jobs and search.refit is original_refit
    assert search.shared_fold_info['route'] == 'stock'
    assert search.shared_fold_info['reason']
    assert search.shared_fold_info['completed'] is True
    if isinstance(observed, _ArrayProvider):
        assert observed.conversions == 0
    elif isinstance(observed, _OnePass):
        assert observed.iterations == 0


def test_private_tokens_are_immutable_shape_stable_and_registered():
    X, _, _ = _data()
    rows = np.array([4, 0, 4, 2], dtype=np.intp)
    original_rows = rows.copy()
    expected = X[rows].copy()
    store = selection._FoldDataset(X)
    token = store[(rows, Ellipsis)]
    assert store[(original_rows.copy(), Ellipsis)] is token
    view = np.asarray(token)
    assert_array_equal(view, expected)
    assert not np.shares_memory(view, X)
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view.flags.writeable = True
    borrowed = token._borrow()
    assert not borrowed.flags.writeable
    with pytest.raises(ValueError):
        borrowed.flags.writeable = True
    view.shape = (view.size,)
    X[:] = 17
    rows[:] = 1
    assert np.asarray(token).shape == expected.shape
    assert_array_equal(np.asarray(token), expected)
    assert_array_equal(token._borrow(), expected)
    forged = selection._FoldToken(store, token._key, token._frozen)
    with pytest.raises(ValueError, match='Unregistered'):
        forged._borrow()


@pytest.mark.parametrize('scoring', [None, 'accuracy'])
def test_refit_false_matches_stock_without_creating_a_final_estimator(scoring, monkeypatch):
    X, y, folds = _data()
    shared = _search(scoring=scoring, folds=folds, refit=False)
    original = shared.estimator
    stock = _search(scoring=scoring, folds=folds, refit=False).fit(X, y)
    assert fit_shared_fold_search(shared, X, y) is shared
    assert shared.estimator is original
    assert shared.shared_fold_info['route'] == 'shared'
    assert shared.shared_fold_info['completed'] is True
    assert shared.shared_fold_info['borrowed_fits'] == 6
    assert not hasattr(shared, 'best_estimator_')
    if scoring is None:
        assert shared.scorer_._estimator is original
    _table_equal(shared, stock)
    restored = _without_private_references(shared, monkeypatch)
    _table_equal(restored, stock)


def test_early_fit_failure_matches_stock_fitted_state_and_restores_constructor(monkeypatch):
    X, y, _ = _data()
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == -1)
    # The full input is valid binary data, but every supplied training fold is
    # invalid. This fails naturally inside the estimator rather than in a spy.
    folds = [(positive.copy(), negative.copy()) for _ in range(3)]
    stock = _search(folds=folds)
    shared = _search(folds=folds)
    original = shared.estimator
    with pytest.raises(ValueError) as stock_failure:
        stock.fit(X, y)
    with pytest.raises(ValueError) as shared_failure:
        fit_shared_fold_search(shared, X, y)
    assert str(shared_failure.value) == str(stock_failure.value)
    assert shared.estimator is original
    assert not hasattr(shared, 'scorer_')
    assert _is_fitted(shared) == _is_fitted(stock)
    assert shared.shared_fold_info['route'] == 'shared'
    assert shared.shared_fold_info['completed'] is False
    assert 'shared_fold_info_' not in vars(shared)
    _without_private_references(shared, monkeypatch)


@pytest.mark.parametrize('initially_fitted', [False, True])
@pytest.mark.parametrize('failure_kind', ['error', 'interrupt'])
def test_final_refit_failure_preserves_exception_and_public_search_state(
        initially_fitted, failure_kind, monkeypatch):
    X, y, folds = _data()
    stores = _capture_private_stores(monkeypatch)
    shared = _search(folds=folds)
    if initially_fitted:
        fit_shared_fold_search(shared, X, y)
    original = shared.estimator
    had_scorer = hasattr(shared, 'scorer_')
    prior_scorer = getattr(shared, 'scorer_', None)
    before = _snapshot(X)
    failure = (RuntimeError('deliberate final refit failure') if failure_kind == 'error'
               else KeyboardInterrupt('deliberate final refit interruption'))
    original_fit = KernMD.fit
    final_refit_calls = []

    def fail_only_final_refit(self, features, targets, **kwargs):
        if len(features) == len(X):
            final_refit_calls.append(self)
            raise failure
        return original_fit(self, features, targets, **kwargs)

    monkeypatch.setattr(KernMD, 'fit', fail_only_final_refit)
    with pytest.raises(type(failure)) as caught:
        fit_shared_fold_search(shared, X, y)
    assert caught.value is failure
    assert len(final_refit_calls) == 1
    assert shared.estimator is original
    assert hasattr(shared, 'scorer_') == had_scorer
    if had_scorer:
        assert shared.scorer_ is prior_scorer
    assert shared.shared_fold_info['route'] == 'shared'
    assert shared.shared_fold_info['completed'] is False
    assert _snapshot(X) == before
    assert len(stores) == 1 + int(initially_fitted)
    assert all(store._tokens == {} for store in stores)
    # sklearn may already have assigned learned search attributes. The helper
    # promises public estimator types, not transactional rollback of sklearn.
    restored = _without_private_references(shared, monkeypatch)
    assert type(restored.estimator) is KernMD
    if hasattr(restored, 'best_estimator_'):
        assert type(restored.best_estimator_) is KernMD


@pytest.mark.parametrize('second_route', ['shared', 'stock'])
def test_repeated_calls_refresh_data_and_match_the_selected_stock_search(second_route):
    X, y, folds = _data()
    original_values = X.copy()
    shared = _search(folds=folds)
    original_estimator = shared.estimator
    fit_shared_fold_search(shared, X, y)
    first_model = shared.best_estimator_
    first_info = copy.deepcopy(shared.shared_fold_info)
    query = X[::3].copy() + .125
    first_scores = first_model.decision_function(query).copy()
    X[:, 0] *= -1
    X[:, 1] += .25
    if second_route == 'stock':
        shared.scoring = _custom_accuracy
    stock = _search(scoring=shared.scoring, folds=folds).fit(X, y)
    assert fit_shared_fold_search(shared, X, y) is shared
    assert shared.estimator is original_estimator
    assert shared.shared_fold_info['route'] == second_route
    assert shared.shared_fold_info['completed'] is True
    _table_equal(shared, stock)
    _predictions_equal(shared, stock, query)
    assert shared.best_estimator_ is not first_model
    assert_array_equal(first_model._Xfit, original_values)
    assert_array_equal(first_model.decision_function(query), first_scores)
    assert_array_equal(shared.best_estimator_._Xfit, X)
    assert not np.shares_memory(shared.best_estimator_._Xfit, X)
    if second_route == 'shared':
        for key in ('borrowed_fits', 'cached_subsets', 'cache_hits', 'root_bytes', 'cached_bytes'):
            assert shared.shared_fold_info[key] == first_info[key]


def test_reused_search_switching_refit_off_preserves_stock_attribute_semantics():
    X, y, folds = _data()
    shared = _search(folds=folds)
    stock = _search(folds=folds)
    fit_shared_fold_search(shared, X, y)
    stock.fit(X, y)
    shared_previous, stock_previous = shared.best_estimator_, stock.best_estimator_
    shared.refit = stock.refit = False
    X[:, 0] *= -1
    stock.fit(X, y)
    fit_shared_fold_search(shared, X, y)
    _table_equal(shared, stock)
    assert shared.shared_fold_info['completed'] is True
    assert shared.shared_fold_info['borrowed_fits'] == 6
    if hasattr(stock, 'best_estimator_'):
        # Existing sklearn releases retain the old model when refit=False.
        # Follow actual stock semantics; do not invent a fresh-state guarantee.
        assert stock.best_estimator_ is stock_previous
        assert shared.best_estimator_ is shared_previous
