"""Public process-CV regression for pandas feature tables with mapped blocks."""
import os

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV

from dwd.model_selection import fit_shared_fold_search


pytestmark = pytest.mark.usefixtures("search_parent_native_thread")


pd = pytest.importorskip("pandas")

_ROWS = np.arange(12)
_A = (6, 2, 11, 1, 9, 4)
_B = (5, 10, 0, 7, 3, 8)
_FOLDS = ((_A, _B), (_B, _A))


def _numbers(phase):
    return np.column_stack((
        10. * _ROWS - 3 * phase,
        _ROWS ** 2 - 2.25 + 5 * phase,
        30.5 - 1.5 * _ROWS - 7 * phase,
    ))


def _frame(numbers, mixed=False):
    index = pd.Index([f"donor-{31 - 2 * row}" for row in _ROWS], name="donor")
    frame = pd.DataFrame(numbers, copy=False, index=index, columns=["value", "value", "rate"])
    if mixed:
        counts = pd.array([None if i % 3 == 0 else i for i in _ROWS], dtype="Int16")
        frame.insert(1, "count", counts)
        groups = pd.Categorical(_ROWS % 2, categories=[1, 0], ordered=True)
        frame.insert(3, "group", groups)
    frame.columns = frame.columns.rename("measurement")
    return frame


class _ObserveFrame(RegressorMixin, BaseEstimator):
    """Check the exact raw rows where sklearn hands them to fit and score."""

    def __init__(self, parent_pid, process, mixed, bias=0):
        self.parent_pid = parent_pid
        self.process = process
        self.mixed = mixed
        self.bias = bias

    def _check_rows(self, X, y):
        assert type(X) is pd.DataFrame
        rows, phases = np.divmod(y, 10)
        phase = phases[0]
        assert phase in (0, 1) and np.all(phases == phase)
        # This independent oracle is built from the data definition, with no
        # reference to the input mapping, worker frame, or a serialized view.
        expected = _frame(_numbers(phase), self.mixed).iloc[rows]
        pd.testing.assert_frame_equal(X, expected, check_exact=True)
        assert (os.getpid() != self.parent_pid) == (self.process and len(rows) < len(_ROWS))
        return tuple(rows)

    def fit(self, X, y):
        self.training_rows_ = self._check_rows(X, y)
        return self

    def predict(self, X):
        return X.iloc[:, 0].to_numpy() + self.bias

    def score(self, X, y):
        test_rows = self._check_rows(X, y)
        assert (self.training_rows_, test_rows) in _FOLDS
        return -float(np.mean(np.abs(self.predict(X) - y)))


def _search(backend, n_jobs, mixed):
    estimator = _ObserveFrame(os.getpid(), backend == "loky" and n_jobs > 1, mixed)
    cv = [(np.array(train), np.array(test)) for train, test in _FOLDS]
    return GridSearchCV(estimator, {"bias": [0, 4]}, cv=cv, n_jobs=None,
                        pre_dispatch=2, error_score="raise")


def _fit(search, frame, phase, backend="loky", n_jobs=2, stock=False):
    cv = search.cv
    y = 10 * _ROWS + phase
    config = {"inner_max_num_threads": 1} if backend == "loky" else {}
    with parallel_config(backend=backend, n_jobs=n_jobs, **config):
        if stock:
            assert search.fit(frame, y) is search
        else:
            assert fit_shared_fold_search(search, frame, y) is search
    assert search.cv is cv
    scores = [0., -4.] if phase == 0 else [-4., 0.]
    for field in ("split0_test_score", "split1_test_score", "mean_test_score"):
        np.testing.assert_array_equal(search.cv_results_[field], scores)
    assert search.best_params_ == {"bias": 4 * phase}
    assert search.best_estimator_.training_rows_ == tuple(_ROWS)
    np.testing.assert_array_equal(search.predict(frame), y)


def _check_mapped_columns(frame, mapped, mixed=False):
    for column in ((0, 2, 4) if mixed else (0, 1, 2)):
        assert np.shares_memory(frame.iloc[:, column].to_numpy(copy=False), mapped)


@pytest.mark.parametrize("mixed", [False, True], ids=["numeric", "mixed"])
def test_private_frame_edits_reach_workers_on_repeated_fit(tmp_path, mixed):
    path = tmp_path / "private.npy"
    np.save(path, -1000. - _numbers(0))
    original_bytes = path.read_bytes()
    mapped = np.load(path, mmap_mode="c")
    frame = _frame(np.asarray(mapped), mixed)
    search = _search("loky", 2, mixed)
    for phase in (0, 1):
        mapped[:] = _numbers(phase)
        _check_mapped_columns(frame, mapped, mixed)
        _fit(search, frame, phase)
        pd.testing.assert_frame_equal(frame, _frame(_numbers(phase), mixed), check_exact=True)
        _check_mapped_columns(frame, mapped, mixed)
        assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("layout", ["c-map", "reversed"])
def test_read_only_frame_layout_reaches_workers(tmp_path, monkeypatch, layout):
    path = tmp_path / "readonly.npy"
    numbers = _numbers(0)
    np.save(path, numbers if layout == "c-map" else numbers[::-1])
    original_bytes = path.read_bytes()
    mapped = np.load(path, mmap_mode="r")
    values = np.asarray(mapped)
    if layout == "reversed":
        values = values[::-1]
        # Some joblib versions reconstruct negative strides from the low
        # address. Reject that unsafe reducer call before a worker reads it.
        from joblib import _memmapping_reducer

        reduce_mapped = _memmapping_reducer._reduce_memmap_backed

        def checked_reduce(array, backing):
            if np.shares_memory(array, mapped):
                assert all(s >= 0 or n < 2 for s, n in zip(array.strides, array.shape)), (
                    "reversed feature block reached joblib's mapped reducer")
            return reduce_mapped(array, backing)

        monkeypatch.setattr(_memmapping_reducer, "_reduce_memmap_backed", checked_reduce)
    frame = _frame(values)
    assert mapped.flags.c_contiguous and not mapped.flags.writeable
    _check_mapped_columns(frame, mapped)
    if layout == "c-map":
        assert frame.to_numpy(copy=False).T.flags.f_contiguous
        assert not frame.to_numpy(copy=False).T.flags.c_contiguous
    else:
        assert frame.to_numpy(copy=False).strides[0] < 0
    _fit(_search("loky", 2, False), frame, 0)
    pd.testing.assert_frame_equal(frame, _frame(numbers), check_exact=True)
    _check_mapped_columns(frame, mapped)
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "kind,backend,n_jobs",
    [("owning", "loky", 2), ("matching-map", "loky", 2),
     ("private", "loky", 1), ("private", "threading", 2)],
    ids=["owning-process", "matching-map-process", "private-serial", "private-threading"],
)
def test_stock_compatible_feature_frame(tmp_path, kind, backend, n_jobs):
    numbers = _numbers(1)
    if kind == "owning":
        assert numbers.flags.owndata
        frame = _frame(numbers)
    else:
        path = tmp_path / "control.npy"
        # A C map of features by rows gives pandas a matching C-order block.
        disk = numbers.T.copy() if kind == "matching-map" else -numbers
        np.save(path, disk)
        original_bytes = path.read_bytes()
        mapped = np.load(path, mmap_mode="r" if kind == "matching-map" else "c")
        if kind == "matching-map":
            frame = _frame(np.asarray(mapped).T)
            assert frame.to_numpy(copy=False).T.flags.c_contiguous
        else:
            mapped[:] = numbers
            frame = _frame(np.asarray(mapped))
        _check_mapped_columns(frame, mapped)
    for stock in (True, False):
        _fit(_search(backend, n_jobs, False), frame, 1, backend, n_jobs, stock=stock)
        pd.testing.assert_frame_equal(frame, _frame(numbers), check_exact=True)
        if kind != "owning":
            _check_mapped_columns(frame, mapped)
            assert path.read_bytes() == original_bytes
