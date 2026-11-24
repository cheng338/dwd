"""Process searches on interval endpoints and compressed sparse storage."""
import os

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV

from dwd.model_selection import fit_shared_fold_search


pd = pytest.importorskip("pandas")
# This constructor accepts compressed block starts and lengths on both pandas
# 2.2 and 3.1. The latter's from_indices constructor recomputes the blocks.
pytestmark = [
    pytest.mark.filterwarnings("ignore:The 'sparse_index' parameter of SparseArray"),
    pytest.mark.usefixtures("search_parent_native_thread"),
]
_ROWS = np.arange(8, dtype=np.int64)
_FOLDS = [(np.array([3, 0, 5, 6]), np.array([1, 7, 2, 4])),
          (np.array([1, 7, 2, 4]), np.array([3, 0, 5, 6]))]


class _CompareFrame(RegressorMixin, BaseEstimator):
    def __init__(self, reference, parent):
        self.reference = reference
        self.parent = parent

    def _check(self, X, rows):
        in_parent = os.getpid() == self.parent
        assert in_parent == (len(rows) == len(_ROWS))
        if in_parent:
            np.testing.assert_array_equal(rows, _ROWS)
            expected = self.reference
        else:
            expected = self.reference.iloc[rows]
        pd.testing.assert_frame_equal(X, expected, check_exact=True)

    def fit(self, X, y):
        self._check(X, y)
        if len(y) == len(_ROWS):
            self.refit_frame_ = X
        return self

    def score(self, X, y):
        self._check(X, y)
        return float(y[0])


def _assert_search(frame, reference):
    search = GridSearchCV(_CompareFrame(reference, os.getpid()), {}, cv=_FOLDS,
                          n_jobs=2, error_score="raise")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        fit_shared_fold_search(search, frame, _ROWS.copy())
    for split, (_, test) in enumerate(_FOLDS):
        np.testing.assert_array_equal(search.cv_results_[f"split{split}_test_score"], [test[0]])
    assert search.best_estimator_.refit_frame_ is frame
    pd.testing.assert_frame_equal(frame, reference, check_exact=True)


def _interval_frame(place, mapped):
    if place == "index":
        left = (100 + _ROWS).view("datetime64[ns]")
        right = mapped.view("datetime64[ns]")
    else:
        left, right = mapped, 200 + _ROWS
    intervals = pd.IntervalIndex.from_arrays(left, right, closed="both", name="band", copy=False)
    if place == "column":
        frame = pd.DataFrame({"value": intervals.array}, copy=False)
        bounds = frame.iloc[:, 0].array.left
    elif place == "index":
        frame = pd.DataFrame({"value": _ROWS}, index=intervals)
        bounds = frame.index.right
    elif place == "categories":
        column = pd.Categorical.from_codes(_ROWS[::-1], categories=intervals, ordered=True)
        frame = pd.DataFrame({"value": column}, copy=False)
        bounds = frame.iloc[:, 0].array.categories.left
    else:
        index = pd.MultiIndex(levels=[intervals, pd.RangeIndex(2)],
                              codes=[(_ROWS + 1) % 8, _ROWS % 2], names=["band", "visit"])
        frame = pd.DataFrame({"value": _ROWS}, index=index)
        bounds = frame.index.levels[0].left
    return frame, bounds.to_numpy(copy=False)


@pytest.mark.parametrize("place", ["column", "index", "categories", "level"])
def test_private_interval_endpoints(tmp_path, place):
    values = 150 + _ROWS
    path = tmp_path / "endpoints.npy"
    np.save(path, values - 5)
    original = path.read_bytes()
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = values
    frame, storage = _interval_frame(place, np.asarray(mapped))
    reference, _ = _interval_frame(place, values)
    assert np.shares_memory(storage, mapped)
    _assert_search(frame, reference)
    assert path.read_bytes() == original


def _sparse_frame(field, data):
    integer = field not in ("block-starts", "block-lengths")
    kind = "integer" if integer else "block"
    # Derive the concrete sparse index class from pandas' public constructor.
    index_type = type(pd.arrays.SparseArray([0, 1], fill_value=0, kind=kind).sp_index)
    if integer:
        indices = data if field == "positions" else np.array([0, 2, 5, 7], dtype=np.int32)
        index = index_type(len(_ROWS), indices)
    else:
        starts = data if field == "block-starts" else np.array([0, 4], dtype=np.int32)
        lengths = data if field == "block-lengths" else np.array([2, 2], dtype=np.int32)
        index = index_type(len(_ROWS), starts, lengths)
    values = data if field in ("column", "index") else np.array([19, 27, 33, 45])
    array = pd.arrays.SparseArray(values, sparse_index=index, fill_value=-1, copy=False)
    if field == "index":
        frame = pd.DataFrame({"value": _ROWS}, index=pd.Index(array, name="sample", copy=False))
        stored = frame.index.array
    else:
        frame = pd.DataFrame({"value": array}, copy=False)
        stored = frame.iloc[:, 0].array
    if field == "positions":
        backing = stored.sp_index.indices
    elif field == "block-starts":
        backing = stored.sp_index.blocs
    elif field == "block-lengths":
        backing = stored.sp_index.blengths
    else:
        backing = stored.sp_values
    return frame, backing


@pytest.mark.parametrize("field,disk,edited", [
    ("column", [1, 2, 3, 4], [19, 27, 33, 45]),
    ("index", [1, 2, 3, 4], [19, 27, 33, 45]),
    ("positions", [0, 1, 3, 6], [0, 2, 5, 7]),
    ("block-starts", [1, 5], [0, 4]),
    ("block-lengths", [1, 3], [2, 2]),
])
def test_private_sparse_storage(tmp_path, field, disk, edited):
    # Both sets of compressed coordinates are valid. Reopening the original
    # file yields different rows without relying on out-of-bounds indexing.
    values = np.array(edited, dtype=np.int32)
    path = tmp_path / "compressed.npy"
    np.save(path, np.array(disk, dtype=np.int32))
    original = path.read_bytes()
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = values
    frame, storage = _sparse_frame(field, np.asarray(mapped))
    reference, _ = _sparse_frame(field, values)
    assert np.shares_memory(storage, mapped)
    _assert_search(frame, reference)
    assert path.read_bytes() == original


def test_empty_mapped_sparse_object_values_rejected_before_split(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(bytes(8))
    mapped = np.memmap(path, mode="r", dtype=object, shape=(0,))
    positions = pd.arrays.SparseArray(np.zeros(len(_ROWS)), fill_value=0).sp_index
    values = pd.arrays.SparseArray(np.asarray(mapped), sparse_index=positions,
                                  fill_value=0, dtype=object, copy=False)
    frame = pd.DataFrame({"value": values}, copy=False)
    stored = frame.iloc[:, 0].array.sp_values
    assert stored.size == 0 and stored.dtype == object
    while isinstance(stored, np.ndarray) and stored is not mapped:
        stored = stored.base
    assert stored is mapped

    class NeverSplit:
        def get_n_splits(self, X=None, y=None, groups=None):
            pytest.fail("attempted split counting on mapped object storage")

        def split(self, X, y=None, groups=None):
            pytest.fail("attempted splitting on mapped object storage")

    search = GridSearchCV(DummyRegressor(), {}, cv=NeverSplit(), n_jobs=2)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with pytest.raises(ValueError, match="file-backed object arrays"):
            fit_shared_fold_search(search, frame, _ROWS)
