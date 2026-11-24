"""Process CV must give a custom splitter and final refit the parent's DataFrame."""
import os

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV

from dwd.model_selection import fit_shared_fold_search


pytestmark = pytest.mark.usefixtures("search_parent_native_thread")


pd = pytest.importorskip("pandas")


def _signal(rows):
    return 11 * rows + 7


class _EditPrivateFeatures:
    def __init__(self, frame, mapped):
        self.parent_frame_id = id(frame)
        self.mapped = mapped
        self.split_calls = 0

    def get_n_splits(self, X=None, y=None, groups=None):
        return 2

    def split(self, X, y, groups=None):
        assert id(X) == self.parent_frame_id, "split must receive the caller's DataFrame"
        self.split_calls += 1
        self.mapped[:] = _signal(y)
        np.testing.assert_array_equal(X.iloc[:, 0].to_numpy(), _signal(y))
        left = np.array([4, 1, 6, 3])
        right = np.array([7, 0, 5, 2])
        yield left, right
        yield right, left


class _CheckParentFrame(RegressorMixin, BaseEstimator):
    def __init__(self, parent_pid, offset=0):
        self.parent_pid = parent_pid
        self.offset = offset

    def fit(self, X, y):
        np.testing.assert_array_equal(X.iloc[:, 0].to_numpy(), _signal(y))
        if len(y) == 8:
            assert os.getpid() == self.parent_pid
            self.refit_frame_ = X
        else:
            assert os.getpid() != self.parent_pid
        return self

    def score(self, X, y):
        assert os.getpid() != self.parent_pid
        np.testing.assert_array_equal(X.iloc[:, 0].to_numpy(), _signal(y))
        return -float(np.mean(np.abs(self.predict(X) - _signal(y))))

    def predict(self, X):
        return X.iloc[:, 0].to_numpy() + self.offset


def test_split_and_final_refit_use_parent_frame_without_callbacks(tmp_path):
    rows = np.arange(8)
    path = tmp_path / "private.npy"
    np.save(path, -100 - rows)
    original_bytes = path.read_bytes()
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = rows + 2
    frame = pd.DataFrame({"value": np.asarray(mapped)}, copy=False)
    assert np.shares_memory(frame.iloc[:, 0].to_numpy(), mapped)

    cv = _EditPrivateFeatures(frame, mapped)
    search = GridSearchCV(_CheckParentFrame(os.getpid()), {"offset": [0, 4]},
                          cv=cv, n_jobs=2, refit=True, error_score="raise")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        assert fit_shared_fold_search(search, frame, rows) is search

    assert search.cv is cv and cv.split_calls == 1
    assert search.best_estimator_.refit_frame_ is frame
    assert search.best_params_ == {"offset": 0}
    for split in range(2):
        np.testing.assert_array_equal(search.cv_results_[f"split{split}_test_score"], [0, -4])
    np.testing.assert_array_equal(search.predict(frame), _signal(rows))
    assert np.shares_memory(frame.iloc[:, 0].to_numpy(), mapped)
    assert path.read_bytes() == original_bytes
