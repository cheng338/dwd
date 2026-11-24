"""Public CV tests for pandas-native storage and the executor selected at fit."""
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
_LEFT = (6, 2, 11, 1, 9, 4)
_RIGHT = (5, 10, 0, 7, 3, 8)
_FOLDS = ((_LEFT, _RIGHT), (_RIGHT, _LEFT))


def _values(kind):
    if kind in ("int-mask", "boolean-data"):
        return _ROWS % 2 == 1
    if kind == "float-data":
        return _ROWS + .25
    if kind in ("category-codes", "multi-codes"):
        return (_ROWS % 3).astype(np.int8)
    if kind in ("category-labels", "multi-level", "column-index"):
        return np.array([20, 31, 45])
    return 19 + 3 * _ROWS


def _frame(kind, values):
    if kind == "int-data":
        column = pd.arrays.IntegerArray(values, np.zeros(len(_ROWS), bool), copy=False)
    elif kind == "int-mask":
        column = pd.arrays.IntegerArray(20 + _ROWS, values, copy=False)
    elif kind == "float-data":
        column = pd.arrays.FloatingArray(values, np.zeros(len(_ROWS), bool), copy=False)
    elif kind == "boolean-data":
        column = pd.arrays.BooleanArray(values, np.zeros(len(_ROWS), bool), copy=False)
    elif kind in ("category-codes", "category-labels"):
        codes = values if kind == "category-codes" else _values("category-codes")
        labels = values if kind == "category-labels" else _values("category-labels")
        column = pd.Categorical.from_codes(codes, pd.Index(labels, copy=False), ordered=True)
    elif kind in ("multi-level", "multi-codes"):
        codes = values if kind == "multi-codes" else _values("multi-codes")
        labels = values if kind == "multi-level" else _values("multi-level")
        index = pd.MultiIndex(levels=[pd.Index(labels, copy=False), pd.RangeIndex(4)],
                              codes=[codes, _ROWS // 3], names=["site", "visit"], copy=False)
        return pd.DataFrame({"response": 8 + _ROWS}, index=index)
    elif kind == "row-index":
        return pd.DataFrame({"response": 8 + _ROWS}, index=pd.Index(values, name="sample", copy=False))
    elif kind == "column-index":
        columns = pd.Index(values, name="assay", copy=False)
        return pd.DataFrame(np.column_stack((_ROWS + 1, _ROWS - 4, _ROWS % 3)), columns=columns)
    elif kind == "plain":
        column = values
    else:
        raise AssertionError(kind)
    return pd.DataFrame({"value": column}, copy=False)


def _backing(frame, kind):
    if kind == "row-index":
        return frame.index.to_numpy(copy=False)
    if kind == "column-index":
        return frame.columns.to_numpy(copy=False)
    if kind == "multi-level":
        return frame.index.levels[0].to_numpy(copy=False)
    if kind == "multi-codes":
        return frame.index.codes[0]
    column = frame.iloc[:, 0]
    if kind == "plain":
        return column.to_numpy(copy=False)
    array = column.array
    if kind == "category-codes":
        return array.codes
    if kind == "category-labels":
        return array.categories.to_numpy(copy=False)
    return array._mask if kind == "int-mask" else array._data


def _signal(frame, kind):
    if kind == "row-index":
        return frame.index.to_numpy(dtype=float)
    if kind == "column-index":
        return frame.to_numpy() @ frame.columns.to_numpy(dtype=float)
    if kind in ("multi-level", "multi-codes"):
        return frame.index.get_level_values("site").to_numpy(dtype=float)
    return frame.iloc[:, 0].to_numpy(dtype=float, na_value=-11)


class _StorageRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, kind, parent_pid, sign=1):
        self.kind = kind
        self.parent_pid = parent_pid
        self.sign = sign

    def _check(self, X, rows):
        expected = _frame(self.kind, _values(self.kind)).iloc[rows]
        assert type(X) is pd.DataFrame
        pd.testing.assert_frame_equal(X, expected, check_exact=True)
        assert (os.getpid() != self.parent_pid) == (len(rows) < len(_ROWS))
        return expected

    def fit(self, X, y):
        self._check(X, y)
        self.train_ = tuple(y)
        return self

    def predict(self, X):
        return self.sign * _signal(X, self.kind)

    def score(self, X, y):
        expected = self._check(X, y)
        assert (self.train_, tuple(y)) in _FOLDS
        return -float(np.mean(np.abs(self.predict(X) - _signal(expected, self.kind))))


class _SwitchToProcesses:
    def setup(self, estimator, context):
        estimator.n_jobs = 2

    def teardown(self, estimator, context):
        pass

    def on_fit_task_begin(self, estimator, context):
        pass

    def on_fit_task_end(self, estimator, context):
        return False


def _check_search(tmp_path, kind, switch=False):
    values = _values(kind)
    path = tmp_path / "backing.npy"
    # Only primitive numeric/bool data are ever written to a file mapping.
    np.save(path, values[::-1])
    original_bytes = path.read_bytes()
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = values
    frame = _frame(kind, np.asarray(mapped))
    assert np.shares_memory(_backing(frame, kind), mapped)
    oracle = _frame(kind, _values(kind))
    folds = [(np.array(a), np.array(b)) for a, b in _FOLDS]
    search = GridSearchCV(_StorageRegressor(kind, os.getpid()), {"sign": [1, -1]},
                          cv=folds, n_jobs=1 if switch else 2, error_score="raise")
    if switch:
        search.set_callbacks(_SwitchToProcesses())
    with parallel_config(backend="loky", inner_max_num_threads=1):
        assert fit_shared_fold_search(search, frame, _ROWS) is search
    expected = _signal(oracle, kind)
    for i, (_, rows) in enumerate(folds):
        np.testing.assert_array_equal(search.cv_results_[f"split{i}_test_score"],
                                      [0., -2. * np.mean(np.abs(expected[rows]))])
    assert search.best_params_ == {"sign": 1}
    np.testing.assert_array_equal(search.predict(frame), expected)
    pd.testing.assert_frame_equal(frame, oracle, check_exact=True)
    assert np.shares_memory(_backing(frame, kind), mapped)
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("kind", [
    "int-data", "int-mask", "float-data", "boolean-data", "category-codes",
    "category-labels", "row-index", "column-index", "multi-level", "multi-codes",
])
def test_pandas_private_feature_storage_reaches_workers(tmp_path, kind):
    _check_search(tmp_path, kind)


@pytest.mark.skipif(not hasattr(GridSearchCV, "set_callbacks"), reason="requires sklearn fit callbacks")
def test_pandas_features_follow_executor_chosen_during_setup(tmp_path):
    _check_search(tmp_path, "plain", switch=True)
