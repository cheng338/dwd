"""E-specific copy of the frozen shared-fold follow-up semantic cases.

Normal installed modules only. No candidate-specific imports or timing claims.
The existing shared-fold/ownership modules retain the broad layout and fallback
matrix; these cases add ordered duplicate folds, independent linear oracles,
mid-search failure cleanup and the byte-gather boundaries under investigation.
"""
import copy
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.model_selection import GridSearchCV, ParameterGrid

from dwd import model_selection as selection
from dwd.kern_md import KernMD


pytestmark = pytest.mark.usefixtures("numerical_search_parent")


def data():
    X = ((np.arange(16 * 3).reshape(16, 3) * 5) % 19 - 9) / 8.
    y = np.where(np.arange(16) % 2, 7, -3)
    X[:, 0] += np.where(y == 7, 1., -1.)
    first = (np.array([0, 1, 2, 3, 0, 1, 2, 3]),
             np.array([4, 5, 4, 5]))
    folds = [first, (np.array([11, 10, 9, 8, 11, 10, 9, 8]),
                     np.array([7, 6, 7, 6])),
             tuple(v.copy() for v in first),
             (np.array([0, 1, 0, 1, 2, 3, 2, 3]), first[1].copy())]
    return X, y, folds


def snapshot(a):
    return (a.shape, a.strides, a.dtype.str, a.flags.writeable,
            a.__array_interface__['data'][0], a.tobytes())


def exact_linear_scores(X, y, query):
    # The feature-space mean-difference formula is independent of KernMD's
    # Gram-matrix computation and of the shared store/gather implementation.
    classes = sorted(set(y.tolist()))
    means = []
    for label in classes:
        rows = X[y == label]
        means.append([sum((Fraction(float(v)) for v in rows[:, j]), Fraction())
                      / len(rows) for j in range(X.shape[1])])
    negative, positive = means
    normal = [p - n for p, n in zip(positive, negative)]
    offset = -sum((p*p - n*n for p, n in zip(positive, negative)), Fraction()) / 2
    scores = [sum((Fraction(float(v))*w for v, w in zip(row, normal)), offset)
              for row in query]
    return np.array([float(v) for v in scores]), np.array([
        classes[int(v > 0)] for v in scores])


def make_search(folds, candidates=2, scoring=None):
    return GridSearchCV(KernMD(kernel='linear'),
                        {'kernel_kws': [None, {}] * (candidates // 2)},
                        cv=folds, n_jobs=1, scoring=scoring, refit=True,
                        error_score='raise', return_train_score=True)


@pytest.mark.parametrize('candidates', [2, 8])
@pytest.mark.parametrize('scoring', [None, 'accuracy'])
def test_ordered_duplicate_repeated_folds_match_independent_linear_oracle(candidates, scoring):
    X, y, folds = data()
    before = (snapshot(X), snapshot(y), [(snapshot(a), snapshot(b)) for a, b in folds])
    searches = [make_search(folds, candidates, scoring) for _ in range(2)]
    searches[0].fit(X, y)
    selection.fit_shared_fold_search(searches[1], X, y)
    expected_splits = {'train': [], 'test': []}
    for train, test in folds:
        for side, rows in [('train', train), ('test', test)]:
            _, labels = exact_linear_scores(X[train], y[train], X[rows])
            expected_splits[side].append(float(np.mean(labels == y[rows])))
    for search in searches:
        assert search.cv_results_['params'] == list(ParameterGrid(search.param_grid))
        for side, scores in expected_splits.items():
            for index, score in enumerate(scores):
                assert_array_equal(search.cv_results_[f'split{index}_{side}_score'],
                                   np.full(candidates, score))
            assert_array_equal(search.cv_results_[f'mean_{side}_score'],
                               np.full(candidates, np.mean(scores)))
            assert_array_equal(search.cv_results_[f'std_{side}_score'],
                               np.full(candidates, np.std(scores)))
        # Duplicate parameter candidates deliberately tie: sklearn selects the
        # first in declared order, not a value-deduplicated or reordered grid.
        assert_array_equal(search.cv_results_['rank_test_score'], np.ones(candidates))
        assert search.best_index_ == 0 and search.best_params_ == {'kernel_kws': None}
        expected, labels = exact_linear_scores(X, y, X[::-1])
        assert_allclose(search.decision_function(X[::-1]), expected, rtol=1e-13, atol=1e-13)
        assert_array_equal(search.predict(X[::-1]), labels)
        assert type(search.best_estimator_) is KernMD
        assert search.best_estimator_._Xfit.flags.writeable
        assert not np.shares_memory(search.best_estimator_._Xfit, X)
        assert_array_equal(search.best_estimator_._Xfit, X)
        assert_array_equal(search.best_estimator_.classes_, [-3, 7])
    info = searches[1].shared_fold_info
    assert info['route'] == 'shared' and info['completed'] is True
    assert info['borrowed_fits'] == candidates * len(folds)
    assert info['cached_subsets'] == 5  # order changes the training cache key
    assert info['cache_hits'] > 0
    assert (snapshot(X), snapshot(y), [(snapshot(a), snapshot(b)) for a, b in folds]) == before


@pytest.mark.parametrize('route', ['stock', 'shared'])
def test_exact_zero_and_representable_near_zero_decisions_keep_label_rule(route):
    X = np.array([[-1., 0.], [1., 0.]] * 8)
    y = np.array([-3, 7] * 8)
    folds = [(np.arange(8), np.arange(8, 16)), (np.arange(8, 16), np.arange(8))]
    query = np.array([[-2.**-48, 0.], [0., 0.], [2.**-48, 0.]])
    search = make_search(folds)
    search.fit(X, y) if route == 'stock' else selection.fit_shared_fold_search(search, X, y)
    expected, labels = exact_linear_scores(X, y, query)
    assert_array_equal(search.decision_function(query), expected)
    assert_array_equal(search.predict(query), labels)
    assert_array_equal(labels, [-3, -3, 7])


@pytest.mark.parametrize('layout', ['C', 'F', 'strided', 'readonly'])
@pytest.mark.parametrize('rows', [
    [0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 1, 3, 1],
    [0, -1, -5, 2], [],
])
def test_private_gather_is_exact_ordered_numpy_subset(layout, rows):
    values = np.arange(20, dtype=np.float64).reshape(5, 4)
    if layout == 'F':
        X = np.array(values, order='F')
    elif layout == 'strided':
        owner = np.zeros((5, 8)); owner[:, ::2] = values; X = owner[:, ::2]
    else:
        X = values.copy()
    if layout == 'readonly':
        X.flags.writeable = False
    indices = np.array(rows, dtype=np.intp)
    before = snapshot(X)
    store = selection._FoldDataset(X)
    token = store[(indices, Ellipsis)]
    expected = X[indices, :]
    assert_array_equal(np.asarray(token), expected)
    assert np.asarray(token).dtype == X.dtype
    assert np.asarray(token).shape == expected.shape
    assert np.asarray(token).tobytes() == expected.tobytes(order='C')
    assert not np.asarray(token).flags.writeable
    assert store[(indices.copy(), Ellipsis)] is token
    assert snapshot(X) == before


@pytest.mark.parametrize('rows', [[5], [-6], [0, 5], [-1, -6]])
def test_private_gather_keeps_invalid_index_refusal(rows):
    X = np.arange(20, dtype=np.float64).reshape(5, 4)
    indices = np.array(rows, dtype=np.intp)
    with pytest.raises(IndexError):
        X[indices, :]
    store = selection._FoldDataset(X)
    with pytest.raises(IndexError):
        store[(indices, Ellipsis)]
    assert store._tokens == {}


def test_private_cache_key_preserves_order_duplicates_and_integer_width():
    X = np.arange(20, dtype=np.float64).reshape(5, 4)
    store = selection._FoldDataset(X)
    first = store[(np.array([0, 2, 0], dtype=np.int8), Ellipsis)]
    assert store[(np.array([0, 2, 0], dtype=np.uint64), Ellipsis)] is first
    other = store[(np.array([0, 0, 2], dtype=np.int64), Ellipsis)]
    assert other is not first
    assert_array_equal(np.asarray(first), X[[0, 2, 0]])
    assert_array_equal(np.asarray(other), X[[0, 0, 2]])


@pytest.mark.parametrize('failure_type', [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize('prior_fit', [False, True])
def test_midfold_failure_releases_registry_restores_scorer_and_allows_reuse(
        monkeypatch, failure_type, prior_fit):
    X, y, folds = data()
    search = make_search(folds)
    if prior_fit:
        selection.fit_shared_fold_search(search, X, y)
    original_estimator = search.estimator
    previous_scorer = vars(search).get('scorer_')
    previous_model = vars(search).get('best_estimator_')
    previous_scores = None if previous_model is None else previous_model.decision_function(X).copy()
    stores = []
    original_store = selection._FoldDataset
    class ObservedStore(original_store):
        def __init__(self, values):
            super().__init__(values); stores.append(self)
    original_solver = selection.kern_md
    calls = []
    failure = failure_type('finite midfold interruption')
    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise failure
        return original_solver(*args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(selection, '_FoldDataset', ObservedStore)
        m.setattr(selection, 'kern_md', interrupted)
        with pytest.raises(failure_type) as caught:
            selection.fit_shared_fold_search(search, X, y)
    assert caught.value is failure and len(calls) == 3
    assert len(stores) == 1 and stores[0]._tokens == {}
    assert search.estimator is original_estimator
    assert vars(search).get('scorer_') is previous_scorer
    assert ('scorer_' in vars(search)) == prior_fit
    assert search.shared_fold_info['completed'] is False
    if previous_model is not None:
        assert search.best_estimator_ is previous_model
        assert_array_equal(previous_model.decision_function(X), previous_scores)
    # Real successful reuse after the failure; no fake fit/scoring return.
    selection.fit_shared_fold_search(search, X, y)
    stock = make_search(folds).fit(X, y)
    assert search.shared_fold_info['completed'] is True
    assert search.estimator is original_estimator
    assert search.scorer_._estimator is original_estimator
    for key in stock.cv_results_:
        if not key.endswith('_time'):
            assert_array_equal(search.cv_results_[key], stock.cv_results_[key])
    assert_array_equal(search.predict(X), stock.predict(X))


def test_best_kernel_keywords_remain_snapshot_after_grid_and_constructor_edits():
    X, y, folds = data()
    settings = [{'gamma': .125}, {'gamma': .5}]
    search = GridSearchCV(KernMD(kernel='rbf'), {'kernel_kws': settings},
                          cv=folds, n_jobs=1, error_score='raise')
    selection.fit_shared_fold_search(search, X, y)
    model = search.best_estimator_
    expected = model.decision_function(X).copy()
    frozen_kws = copy.deepcopy(model.kernel_kws_)
    for settings_entry in settings:
        settings_entry['gamma'] = 4.
    search.estimator.set_params(kernel='linear', kernel_kws=None)
    model.set_params(kernel='linear', kernel_kws=None)
    assert model.kernel_ == 'rbf' and model.kernel_kws_ == frozen_kws
    assert_array_equal(model.decision_function(X), expected)


@pytest.mark.parametrize('shape,rows', [((0, 4), []), ((5, 0), []),
                                      ((5, 0), [0, 1, 0, -1]), ((0, 0), [])])
def test_private_zero_size_gather_preserves_shape(shape, rows):
    X = np.empty(shape, dtype=np.float64)
    indices = np.asarray(rows, dtype=np.intp)
    store = selection._FoldDataset(X)
    actual = np.asarray(store[(indices, Ellipsis)])
    expected = X[indices, :]
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    assert not actual.flags.writeable


def test_private_unsigned_key_keeps_existing_intp_normalization():
    X = np.arange(20, dtype=np.float64).reshape(5, 4)
    unsigned = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)
    # This private entry has always normalized keys to intp. Public eligibility
    # rejects this input; the test does not promote wrapped indices publicly.
    normalized = unsigned.astype(np.intp, copy=False)
    store = selection._FoldDataset(X)
    token = store[(unsigned, Ellipsis)]
    assert_array_equal(np.asarray(token), X[normalized, :])
    assert store[(normalized, Ellipsis)] is token


@pytest.mark.parametrize('rows', [[0, 1, 2, 3], [3, 1, 1, 0]])
def test_e_gather_uses_single_span_or_original_numpy_fallback(monkeypatch, rows):
    X = np.arange(32, dtype=np.float64).reshape(8, 4)
    store = selection._FoldDataset(X)
    original = selection._FrozenMatrix.view
    root_reads = []
    def observed(matrix):
        if matrix is store._root:
            root_reads.append(1)
        return original(matrix)
    monkeypatch.setattr(selection._FrozenMatrix, 'view', observed)
    token = store[(np.array(rows, dtype=np.intp), Ellipsis)]
    assert_array_equal(np.asarray(token), X[rows])
    # E shares only the consecutive nonnegative selector. The reordered
    # selector must retain the original NumPy gather, including one root view.
    # All other cases are text-identical to the frozen 43-case general module.
    expected = 0 if rows == [0, 1, 2, 3] else 1
    assert len(root_reads) == expected
