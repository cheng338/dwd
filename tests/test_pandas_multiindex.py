"""Preserve the MultiIndex codes pandas uses for groups with missing keys."""
import os

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV

from dwd.model_selection import fit_shared_fold_search


pytestmark = pytest.mark.usefixtures("search_parent_native_thread")


pd = pytest.importorskip("pandas")
_ROWS = np.arange(8)
_LABELS = np.array([11, 23, 37, 41])


def _frame(labels):
    records = pd.DataFrame({"site": [1, np.nan, 2, np.nan, 2, 1, np.nan, 2],
                            "visit": [1, 3, 2, 2, 3, 3, 4, 4]})
    grouped = records.groupby(["site", "visit"], dropna=False).size().index
    index = grouped.set_levels(pd.Index(labels, copy=False), level="visit")
    return pd.DataFrame({"value": 3 * _ROWS}, index=index)


def _loc(index, key):
    try:
        return index.get_loc(key)
    except KeyError:
        return None


class _CheckGroupIndex(RegressorMixin, BaseEstimator):
    def __init__(self, parent):
        self.parent = parent

    def _check(self, X, rows):
        assert (os.getpid() == self.parent) == (len(rows) == len(_ROWS))
        expected = _frame(_LABELS).iloc[rows]
        for actual, want in zip(X.index.codes, expected.index.codes):
            np.testing.assert_array_equal(actual, want)
        for visit in _LABELS:
            key = (np.nan, visit)
            assert _loc(X.index, key) == _loc(expected.index, key)
        pd.testing.assert_frame_equal(X, expected, check_exact=True)

    def fit(self, X, y):
        self._check(X, y)
        if len(y) == len(_ROWS):
            self.refit_frame_ = X
        return self

    def score(self, X, y):
        self._check(X, y)
        return float(y[0])


def test_mapped_level_preserves_missing_group_codes_in_another_level(tmp_path):
    path = tmp_path / "labels.npy"
    np.save(path, _LABELS - 10)
    original = path.read_bytes()
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = _LABELS
    frame = _frame(np.asarray(mapped))
    assert np.shares_memory(frame.index.levels[1].to_numpy(), mapped)

    left = np.array([3, 0, 5, 6])
    right = np.array([1, 7, 2, 4])
    search = GridSearchCV(_CheckGroupIndex(os.getpid()), {}, cv=[(left, right), (right, left)],
                          n_jobs=2, error_score="raise")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        fit_shared_fold_search(search, frame, _ROWS)
    np.testing.assert_array_equal(search.cv_results_["split0_test_score"], [right[0]])
    np.testing.assert_array_equal(search.cv_results_["split1_test_score"], [left[0]])
    assert search.best_estimator_.refit_frame_ is frame
    assert path.read_bytes() == original
