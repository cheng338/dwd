"""Public GridSearchCV witness for privately edited pandas train/test selectors."""

import os

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from dwd.model_selection import fit_shared_fold_search


pytestmark = pytest.mark.usefixtures("search_parent_native_thread")


EDITED = ((2, 0, 3, 1), (6, 4, 7, 5))


class RowModel(BaseEstimator):
    def __init__(self, residue=0):
        self.residue = residue

    def fit(self, X, y):
        np.testing.assert_array_equal(X[:, 0], y)
        self.train_rows_ = tuple(X[:, 0])
        self.fit_pid_ = os.getpid()
        return self

    def predict(self, X):
        return np.full(len(X), self.residue)


def fingerprint(rows):
    # Each row ID is in [0, 7]. Adding one preserves leading zeroes.
    return int(''.join(str(int(row) + 1) for row in rows))


class AuditScorer:
    def __init__(self, search, first, second, process):
        self.search = search
        self.first = first
        self.second = second
        self.selector_type = type(first)
        self.parent = os.getpid()
        self.process = process

    def __call__(self, estimator, X, y):
        np.testing.assert_array_equal(X[:, 0], y)
        scored = tuple(X[:, 0])
        train = estimator.train_rows_
        folds = self.search.cv
        aliases = (self.first is folds[0][0] is folds[1][1]
                   and self.second is folds[0][1] is folds[1][0])
        metadata = all(type(s) is self.selector_type and s.name == n
                       and s.dtype == np.dtype('int64')
                       for s, n in ((self.first, 'a'), (self.second, 'b')))
        return {
            'winner': float(estimator.residue == train[0] % 4 == scored[0] % 4),
            'fit_rows': fingerprint(train),
            'scored_rows': fingerprint(scored),
            'aliases': float(aliases),
            'metadata': float(metadata),
            'selector_values': float((tuple(self.first), tuple(self.second)) == EDITED),
            'placement': float(estimator.fit_pid_ == os.getpid()
                               and (os.getpid() != self.parent) == self.process),
        }


@pytest.mark.parametrize('kind', ['Index', 'Series'])
@pytest.mark.parametrize('mode', ['stock-serial-private', 'stock-process-owning', 'dwd-process-private'])
def test_pandas_row_selectors_preserve_winner_and_refit(tmp_path, kind, mode):
    pd = pytest.importorskip('pandas')
    disk = np.arange(8, dtype=np.int64).reshape(2, 4)
    path = tmp_path / 'selectors.npy'
    np.save(path, disk)
    mapped = np.load(path, mmap_mode='c')
    mapped[:] = EDITED
    owning = mode == 'stock-process-owning'
    first, second = (getattr(pd, kind)(row.copy() if owning else row, name=name, copy=False)
                     for row, name in zip(mapped, ('a', 'b')))
    for selector in (first, second):
        assert bool(np.shares_memory(selector.to_numpy(copy=False), mapped)) != owning

    folds = [(first, second), (second, first)]
    process = mode != 'stock-serial-private'
    search = GridSearchCV(RowModel(), {'residue': [0, 2]}, cv=folds, n_jobs=2 if process else 1,
                          return_train_score=True, refit='winner', error_score='raise')
    search.scoring = AuditScorer(search, first, second, process)
    X = np.arange(8)[:, None]
    y = X[:, 0].copy()
    with parallel_config(backend='loky', inner_max_num_threads=1):
        if mode == 'dwd-process-private':
            fit_shared_fold_search(search, X, y)
        else:
            search.fit(X, y)

    # Stale disk rows award both folds to residue 0. The private edits award
    # both folds to residue 2. Exact row signatures distinguish every position.
    results = search.cv_results_
    for i in range(2):
        for side, scored_rows in [('train', EDITED[i]), ('test', EDITED[1 - i])]:
            prefix = f'split{i}_{side}_'
            np.testing.assert_array_equal(results[prefix + 'winner'], [0., 1.])
            for metric, value in [('fit_rows', fingerprint(EDITED[i])),
                                  ('scored_rows', fingerprint(scored_rows)),
                                  ('aliases', 1.), ('metadata', 1.),
                                  ('selector_values', 1.), ('placement', 1.)]:
                np.testing.assert_array_equal(results[prefix + metric], [value, value])
    np.testing.assert_array_equal(results['mean_test_winner'], [0., 1.])
    np.testing.assert_array_equal(results['rank_test_winner'], [2, 1])
    assert search.best_index_ == 1 and search.best_params_ == {'residue': 2}
    assert search.best_estimator_.fit_pid_ == os.getpid()
    assert search.best_estimator_.train_rows_ == tuple(y)
    np.testing.assert_array_equal(search.predict(X), np.full(8, 2))
    assert search.cv is folds and search.scoring.first is first and search.scoring.second is second
    np.testing.assert_array_equal(mapped, EDITED)
    np.testing.assert_array_equal(np.load(path), disk)
