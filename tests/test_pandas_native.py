"""Public checks for pandas temporal storage and parent-side object rejection."""
import os

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV, _search

from dwd.model_selection import fit_shared_fold_search


pytestmark = pytest.mark.usefixtures("search_parent_native_thread")


pd = pytest.importorskip("pandas")
ROWS = np.arange(8)
FOLDS = [(np.array([4, 0, 6, 2]), np.array([7, 1, 5, 3])),
         (np.array([7, 1, 5, 3]), np.array([4, 0, 6, 2]))]


def _ordinals():
    data = np.array([37 + 3 * ROWS, 239 + 19 * ((3 * ROWS) % 8)], dtype="int64")
    data[0, 4] = data[1, 7] = np.iinfo(np.int64).min  # pandas NaT
    return data


def _temporal_array(kind, ordinals):
    if kind == "period":
        return pd.arrays.PeriodArray(ordinals, dtype=pd.PeriodDtype("D"), copy=False)
    dtype = "datetime64[ns]" if kind == "datetime" else "timedelta64[ns]"
    return ordinals.view(dtype)


def _frame(kind, ordinals):
    index = pd.Index(_temporal_array(kind, ordinals[1]), name="stamp", copy=False)
    return pd.DataFrame({"time": _temporal_array(kind, ordinals[0])}, index=index, copy=False)


class _ObserveTime(RegressorMixin, BaseEstimator):
    def __init__(self, kind, process, parent_pid):
        self.kind = kind
        self.process = process
        self.parent_pid = parent_pid

    def _check(self, X, y):
        expected = _frame(self.kind, _ordinals()).iloc[y]
        pd.testing.assert_frame_equal(X, expected, check_exact=True)
        assert (os.getpid() != self.parent_pid) == (self.process and len(y) < len(ROWS))

    def fit(self, X, y):
        self._check(X, y)
        if len(y) == len(ROWS):
            self.refit_frame_ = X
        return self

    def score(self, X, y):
        self._check(X, y)
        return float(y[0])


@pytest.mark.parametrize("kind,storage,jobs", [
    ("datetime", "private", 2), ("timedelta", "private", 2), ("period", "private", 2),
    ("period", "private", 1), ("datetime", "owning", 2),
])
def test_native_time_column_and_index(tmp_path, kind, storage, jobs):
    if storage == "private":
        path = tmp_path / "ordinals.npy"
        np.save(path, np.arange(16, dtype="int64").reshape(2, 8))
        original_bytes = path.read_bytes()
        mapped = np.load(path, mmap_mode="c")
        mapped[:] = _ordinals()
        frame = _frame(kind, np.asarray(mapped))
        assert np.shares_memory(frame.iloc[:, 0].array.asi8, mapped[0])
        assert np.shares_memory(frame.index.asi8, mapped[1])
    else:
        frame = _frame(kind, _ordinals())

    search = GridSearchCV(_ObserveTime(kind, jobs > 1, os.getpid()), {}, cv=FOLDS,
                          n_jobs=jobs, error_score="raise")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        assert fit_shared_fold_search(search, frame, ROWS) is search

    for split, (_, test) in enumerate(FOLDS):
        np.testing.assert_array_equal(search.cv_results_[f"split{split}_test_score"], [test[0]])
    if jobs == 1 or storage == "owning":
        assert search.best_estimator_.refit_frame_ is frame
    pd.testing.assert_frame_equal(frame, _frame(kind, _ordinals()), check_exact=True)
    if storage == "private":
        assert path.read_bytes() == original_bytes


def test_raw_mapped_object_block_is_rejected_before_custom_splitter(tmp_path, monkeypatch):
    # A zero-length object-dtype view has the real mmap base and no object
    # slots. No Python pointer is created, persisted, or sent to a worker.
    path = tmp_path / "empty-block.bin"
    path.write_bytes(bytes(8))
    mapped = np.memmap(path, mode="r", dtype=object, shape=(0, 1))
    frame = pd.DataFrame(np.asarray(mapped), columns=["empty"], dtype=object, copy=False)
    view = frame.iloc[:, 0].to_numpy(copy=False)
    assert view.dtype == np.dtype(object) and view.size == 0
    # shares_memory is always false for zero elements. Check the retained
    # root by identity, independently of the helper's own map detection.
    while isinstance(view, np.ndarray) and view is not mapped:
        view = view.base
    assert view is mapped

    class NeverSplit:
        def get_n_splits(self, X=None, y=None, groups=None):
            pytest.fail("split counting ran before mapped-object rejection")

        def split(self, X, y=None, groups=None):
            pytest.fail("split ran before mapped-object rejection")

    def prevent_executor(*args, **kwargs):
        pytest.fail("executor construction preceded mapped-object rejection")

    monkeypatch.setattr(_search, "Parallel", prevent_executor)
    search = GridSearchCV(DummyRegressor(), {}, cv=NeverSplit(), n_jobs=2)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with pytest.raises(ValueError, match="file-backed object arrays"):
            fit_shared_fold_search(search, frame, np.empty(0))
