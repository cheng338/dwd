"""Active pandas reducers can reopen a file before pickle reaches a fold buffer."""
import pickle

import numpy as np
import pytest
from joblib import parallel_config
from joblib.externals.loky.backend.reduction import get_loky_pickler
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV

from dwd.model_selection import fit_shared_fold_search


pd = pytest.importorskip("pandas")


def _reload_carrier(path, kind):
    data = np.load(path)
    if kind == "nullable-index":
        return pd.arrays.IntegerArray(data, np.zeros(data.size, dtype=bool))
    return pd.Index(data)


@pytest.mark.parametrize(("kind", "veto"), [
    pytest.param("nullable-index", False, id="nullable-index"),
    pytest.param("series-labels", False, id="series-labels"),
    pytest.param("interval-left", False, id="interval-left"),
    pytest.param("interval-left", True, id="interval-left-veto"),
])
def test_nested_reducer_cannot_reopen_private_fold_storage(tmp_path, monkeypatch, kind, veto):
    path = tmp_path / "rows.npy"
    np.save(path, np.arange(3, dtype=np.int64))
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = [3, 4, 5]
    if kind == "nullable-index":
        values = pd.arrays.IntegerArray(mapped, np.zeros(3, dtype=bool), copy=False)
        selector = pd.Index(values, copy=False)
        carrier = selector._data
        assert type(carrier) is pd.arrays.IntegerArray
        assert np.shares_memory(carrier._data, mapped)
    else:
        if kind == "interval-left":
            labels = pd.IntervalIndex.from_arrays(mapped, np.arange(10, 13), copy=False)
        else:
            labels = pd.Index(mapped, copy=False)
        selector = pd.Series(np.arange(3), index=labels, copy=False)
        # IntervalIndex.__reduce__ saves the public endpoint Index objects.
        carrier = selector.index.left if kind == "interval-left" else selector.index
        assert type(carrier) is pd.Index
        assert np.shares_memory(carrier.to_numpy(copy=False), mapped)

    pickler = get_loky_pickler()
    dump = pickler.dump
    slot = pickle.Pickler.dispatch_table
    seen = []
    reject = False
    marker = "backend vetoed the selected interval endpoint"

    def install(self, batch):
        table = slot.__get__(self)
        fallback = table.get(type(carrier))

        def reduce_carrier(value):
            if value is carrier:
                seen.append(True)
                if reject:
                    raise ValueError(marker)
                return _reload_carrier, (str(path), kind)
            return value.__reduce__() if fallback is None else fallback(value)

        slot.__set__(self, {**table, type(carrier): reduce_carrier})
        return dump(self, batch)

    def score(model, X, y):
        if kind == "nullable-index":
            return float(model.constant_[0, 0])  # Mean of the actual training targets.
        if kind == "interval-left":
            return float(np.dot(selector.index.left.to_numpy(), [100, 10, 1]))
        return float(np.mean(selector.index.to_numpy()))

    rows = np.arange(8)
    folds = [(selector, np.array([6, 7]))]
    search = GridSearchCV(DummyRegressor(), {}, cv=folds, scoring=score, n_jobs=2,
                          pre_dispatch=1, refit=False, error_score="raise")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        if kind == "interval-left" and not veto:
            stock = GridSearchCV(**search.get_params(deep=False))
            fit_shared_fold_search(stock, rows[:, None], rows)
            np.testing.assert_array_equal(stock.cv_results_["split0_test_score"], [345.0])

        # Prove the installed recipe actually substitutes disk rows or disk labels.
        monkeypatch.setattr(pickler, "dump", install)
        reference = GridSearchCV(**search.get_params(deep=False))
        reference.fit(rows[:, None], rows)
        expected = 12.0 if kind == "interval-left" else 1.0
        np.testing.assert_array_equal(reference.cv_results_["split0_test_score"], [expected])
        assert seen == [True]
        seen.clear()

        reject = veto
        with pytest.raises(pickle.PicklingError) as error:
            fit_shared_fold_search(search, rows[:, None], rows)

    message = f"{error.value}\n{error.value.__cause__}"
    if veto:
        assert marker in message
    else:
        assert "owning" in message and "fold" in message
    assert seen == [True] and search.cv is folds
    assert not search.shared_fold_info["completed"]
    np.testing.assert_array_equal(mapped, [3, 4, 5])
    np.testing.assert_array_equal(np.load(path), [0, 1, 2])
