"""Exercise custom reducers on the actual objects selected for worker fits."""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from joblib import parallel_config
from joblib.externals.loky.backend.reduction import get_loky_pickler
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV

from dwd.model_selection import fit_shared_fold_search


pd = pytest.importorskip("pandas")
DISK = ((0, 1, 2, 3), (4, 5, 6, 7))
EDITED = ((2, 0, 3, 1), (6, 4, 7, 5))
ROWS = np.arange(8)


def _folds(path, kind):
    np.save(path, np.asarray(DISK, dtype=np.int64))
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = EDITED
    first, second = (np.asarray(row) if kind == "ndarray" else getattr(pd, kind)(row, copy=False)
                     for row in mapped)
    assert all(np.shares_memory(np.asarray(row), mapped) for row in (first, second))
    return [(first, second)]


def _reload_selector(path, kind, i):
    return getattr(pd, kind)(np.load(path)[i], copy=False)


@pytest.mark.parametrize("kind", ["Index", "Series"])
@pytest.mark.parametrize("veto", [False, True], ids=["reopen-disk", "veto"])
def test_live_wrapper_reducer_handles_original_once(tmp_path, monkeypatch, kind, veto):
    path = tmp_path / "indices.npy"
    folds = _folds(path, kind)
    selectors = folds[0]
    wrapper = getattr(pd, kind)
    assert all(type(selector) is wrapper for selector in selectors)
    search = GridSearchCV(DummyRegressor(), {}, cv=folds, n_jobs=2,
                          pre_dispatch=1, refit=False, error_score="raise")
    seen = []
    marker = "backend vetoed original pandas wrapper"
    pickler = get_loky_pickler()
    dump = pickler.dump
    slot = pickle.Pickler.dispatch_table

    def install(self, batch):
        table = slot.__get__(self)
        fallback = table.get(wrapper)

        def reduce_wrapper(value):
            for i, original in enumerate(selectors):
                if value is original:
                    seen.append(i)
                    if veto:
                        raise ValueError(marker)
                    # This recipe never exposes the selector's edited array.
                    return _reload_selector, (str(path), kind, i)
            return value.__reduce__() if fallback is None else fallback(value)

        slot.__set__(self, {**table, wrapper: reduce_wrapper})
        return dump(self, batch)

    monkeypatch.setattr(pickler, "dump", install)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with pytest.raises(pickle.PicklingError) as error:
            fit_shared_fold_search(search, ROWS[:, None], ROWS)
    message = f"{error.value}\n{error.value.__cause__}"
    if veto:
        assert marker in message
    else:
        assert "owning" in message and "fold" in message
    assert len(seen) == 1  # One batch fails at the first selected wrapper.
    assert search.cv is folds and not search.shared_fold_info["completed"]
    np.testing.assert_array_equal(selectors, EDITED)
    np.testing.assert_array_equal(np.load(path), DISK)


class _OpaqueCaptureSearch(GridSearchCV):
    def __getstate__(self):
        state = super().__getstate__().copy()
        state["opaque"] = SimpleNamespace(raw=np.asarray(self.cv[0][0]).view())
        return state

    def read_capture(self, model, X, y):
        np.testing.assert_array_equal(self.cv[0][0], EDITED[0])
        np.testing.assert_array_equal(y, EDITED[1])
        # This distinct view belongs to the custom hook's opaque record.
        np.testing.assert_array_equal(self.opaque.raw, DISK[0])
        return 1.0


@pytest.mark.parametrize("kind", ["Index", "Series", "ndarray"])
def test_custom_search_state_keeps_independent_capture(tmp_path, kind):
    path = tmp_path / "indices.npy"
    folds = _folds(path, kind)
    search = _OpaqueCaptureSearch(DummyRegressor(), {}, cv=folds, n_jobs=2,
                                  pre_dispatch=1, refit=False, error_score="raise")
    search.scoring = search.read_capture
    with parallel_config(backend="loky", inner_max_num_threads=1):
        try:
            fit_shared_fold_search(search, ROWS[:, None], ROWS)
        except pickle.PicklingError as error:
            if kind == "ndarray":
                raise
            message = f"{error}\n{error.__cause__}"
            assert "owning" in message and "fold" in message
            assert not search.shared_fold_info["completed"]
        else:
            np.testing.assert_array_equal(search.cv_results_["split0_test_score"], [1.0])
            assert search.shared_fold_info["completed"]
    assert search.cv is folds and not hasattr(search, "opaque")
    np.testing.assert_array_equal(folds[0], EDITED)
    np.testing.assert_array_equal(np.load(path), DISK)
