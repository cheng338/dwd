"""Public CV controls for pandas train/test selectors with mapped storage."""
import os

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from dwd.model_selection import fit_shared_fold_search


pd = pytest.importorskip('pandas')
ROWS = np.arange(8, dtype=np.int64)


def _signature(train, test):
    return int(''.join(str(int(i) + 1) for i in (*train, *test)))


class _OrderedRows(BaseEstimator):
    def __init__(self, parent, process):
        self.parent = parent
        self.process = process

    def fit(self, X, y):
        assert (os.getpid() != self.parent) == self.process
        np.testing.assert_array_equal(X[:, 0], y)
        self.rows_ = tuple(y)
        return self

    def score(self, X, y):
        np.testing.assert_array_equal(X[:, 0], y)
        return _signature(self.rows_, y)


def _results(folds, process, scorer_factory=None):
    search = GridSearchCV(_OrderedRows(os.getpid(), process), {}, cv=folds,
                          n_jobs=2 if process else 1, refit=False, error_score='raise')
    if scorer_factory is not None:
        search.scoring = scorer_factory(search)
    with parallel_config(backend='loky', inner_max_num_threads=1):
        if process:
            fit_shared_fold_search(search, ROWS[:, None], ROWS)
        else:
            search.fit(ROWS[:, None], ROWS)
    assert search.cv is folds
    return [search.cv_results_[f'split{i}_test_score'][0] for i in range(2)]


@pytest.mark.parametrize('kind', ['Index', 'Series'])
@pytest.mark.parametrize('case', ['reversed-readonly', 'ordinary-bool', 'private-bool'])
def test_ordered_train_and_test_rows(tmp_path, kind, case):
    path = tmp_path / 'selectors.npy'
    if case == 'reversed-readonly':
        np.save(path, ROWS.reshape(2, 4))
        mapping = np.load(path, mmap_mode='r')
        values = [row[::-1] for row in mapping]
        selected = [[3, 2, 1, 0], [7, 6, 5, 4]]
    else:
        mask = np.array([1, 0, 0, 1, 0, 1, 1, 0], dtype=bool)
        selected = [[0, 3, 5, 6], [1, 2, 4, 7]]
        values = np.stack([mask, ~mask])
        if case == 'private-bool':
            np.save(path, ~values)
            mapping = np.load(path, mmap_mode='c')
            mapping[:] = values
            values = mapping

    first, second = [getattr(pd, kind)(row, name='selector', copy=False) for row in values]
    if case != 'ordinary-bool':
        for selector in (first, second):
            assert np.shares_memory(selector.to_numpy(copy=False), mapping)
    folds = [(first, second), (second, first)]
    expected = [_signature(*selected), _signature(*reversed(selected))]
    assert _results(folds, process=False) == expected
    assert _results(folds, process=True) == expected
    if case == 'private-bool':
        np.testing.assert_array_equal(np.load(path), ~values)


class _SeriesMetadata:
    def __init__(self, search):
        self.search = search

    def __call__(self, estimator, X, y):
        for selector, labels in zip(self.search.cv[0], [(19, 17, 13, 11), (37, 31, 29, 23)]):
            assert type(selector) is pd.Series
            assert selector.name == 'selector' and selector.index.name == 'assay'
            assert tuple(selector.index) == labels
            assert selector.attrs == {'batch': 'morning'}
            assert not selector.flags.allows_duplicate_labels
        return estimator.score(X, y)


def test_worker_public_series_keeps_mapped_index_values_and_metadata(tmp_path):
    path = tmp_path / 'series_labels.npy'
    np.save(path, ROWS.reshape(2, 4))
    labels = np.load(path, mmap_mode='c')
    labels[:] = [[19, 17, 13, 11], [37, 31, 29, 23]]
    selectors = []
    for data, axis in zip(ROWS.reshape(2, 4), labels):
        index = pd.Index(axis, name='assay', copy=False)
        selector = pd.Series(data, index=index, name='selector', copy=False)
        selector.attrs['batch'] = 'morning'
        selector.flags.allows_duplicate_labels = False
        assert np.shares_memory(selector.index.to_numpy(copy=False), labels)
        selectors.append(selector)
    first, second = selectors
    folds = [(first, second), (second, first)]
    reference = _results(folds, process=False, scorer_factory=_SeriesMetadata)
    assert _results(folds, process=True, scorer_factory=_SeriesMetadata) == reference
    np.testing.assert_array_equal(np.load(path), ROWS.reshape(2, 4))


class _CapturedView:
    def __init__(self, view):
        self.view = view

    def __call__(self, estimator, X, y):
        assert tuple(self.view) == (0, 3)
        return estimator.score(X, y)


@pytest.mark.parametrize('kind', ['Index', 'Series'])
def test_independent_overlap_with_selector_storage(tmp_path, kind):
    path = tmp_path / 'private_rows.npy'
    np.save(path, ROWS.reshape(2, 4))
    mapping = np.load(path, mmap_mode='c')
    mapping[:] = [[2, 0, 3, 1], [6, 4, 7, 5]]
    first, second = [getattr(pd, kind)(row, copy=False) for row in mapping]
    view = mapping[0, 1:3]
    assert np.shares_memory(first.to_numpy(copy=False), view)
    folds = [(first, second), (second, first)]
    def scorer_factory(search):
        return _CapturedView(view)

    reference = _results(folds, process=False, scorer_factory=scorer_factory)
    assert _results(folds, process=True, scorer_factory=scorer_factory) == reference
    np.testing.assert_array_equal(np.load(path), ROWS.reshape(2, 4))
