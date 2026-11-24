"""Public row selection when pandas folds use different process serializers."""
import os
import pickle

import numpy as np
import pytest
from joblib import parallel_config
from joblib._multiprocessing_helpers import mp
from joblib.externals.loky.backend.reduction import get_loky_pickler
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, check_cv

from dwd.model_selection import fit_shared_fold_search


pd = pytest.importorskip("pandas")
ROWS = np.arange(8, dtype=np.int64)
EDITED = ((2, 0, 3, 1), (6, 4, 7, 5))
DISK = ((0, 1, 2, 3), (4, 5, 6, 7))


def _fingerprint(rows):
    return int("".join(str(int(row) + 1) for row in rows))


class _RowModel(BaseEstimator):
    def fit(self, X, y):
        np.testing.assert_array_equal(X[:, 0], y)
        self.train_ = _fingerprint(y)
        self.pid_ = os.getpid()
        return self


class _ReadRows:
    def __init__(self, raw):
        self.raw = raw
        self.parent = os.getpid()

    def __call__(self, model, X, y):
        assert model.pid_ == os.getpid() != self.parent
        np.testing.assert_array_equal(X[:, 0], y)
        cv = self.search.cv
        assert cv[0][0] is cv[1][1] and cv[0][1] is cv[1][0]
        out = {"train": model.train_, "test": _fingerprint(y),
               "public_first": _fingerprint(cv[0][0]), "public_second": _fingerprint(cv[0][1])}
        if self.raw is not None:
            out["captured"] = _fingerprint(self.raw)
        return out


class _PassthroughSearch(GridSearchCV):
    def __getstate__(self):
        return super().__getstate__()


class _OverrideSearch(GridSearchCV):
    def _run_search(self, evaluate_candidates):
        # Only the selected folds travel to the tasks. search.cv stays public.
        selected = vars(self).pop("selected_cv")
        evaluate_candidates([{}], cv=selected)


@pytest.fixture(params=["Index", "Series"])
def folds(tmp_path, request):
    path = tmp_path / "selectors.npy"
    np.save(path, np.asarray(DISK, dtype=np.int64))
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = EDITED
    first, second = [getattr(pd, request.param)(row, copy=False) for row in mapped]
    for selector in (first, second):
        assert np.shares_memory(selector.to_numpy(copy=False), mapped)
    yield [(first, second), (second, first)]
    np.testing.assert_array_equal(mapped, EDITED)
    np.testing.assert_array_equal(np.load(path), DISK)


def _search(cls, cv, raw=None):
    scorer = _ReadRows(raw)
    search = cls(_RowModel(), {}, cv=cv, scoring=scorer, refit=False, n_jobs=2,
                 pre_dispatch=1, error_score="raise")
    scorer.search = search
    return search


def _check_result(search, public, captured=False):
    for i in (0, 1):
        expected = {"train": EDITED[i], "test": EDITED[1 - i],
                    "public_first": public[0], "public_second": public[1]}
        if captured:
            expected["captured"] = EDITED[0]
        for name, rows in expected.items():
            np.testing.assert_array_equal(search.cv_results_[f"split{i}_test_{name}"], [_fingerprint(rows)])


@pytest.mark.parametrize("case", ["passthrough-hook", "checked-override", "multiprocessing"])
def test_worker_rows_with_selected_serializer(folds, case):
    public = folds
    expected_public = EDITED
    backend = "loky"
    options = {"inner_max_num_threads": 1}
    if case == "checked-override":
        first, second = [np.array(rows) for rows in DISK]
        public = [(first, second), (second, first)]
        search = _search(_OverrideSearch, public)
        search.selected_cv = check_cv(folds)
        expected_public = DISK
    elif case == "passthrough-hook":
        search = _search(_PassthroughSearch, public, raw=folds[0][0].to_numpy(copy=False))
    else:
        if mp is None:
            pytest.skip("joblib multiprocessing is unavailable")
        backend = "multiprocessing"
        options = {}
        search = _search(GridSearchCV, public)

    with parallel_config(backend=backend, **options):
        assert fit_shared_fold_search(search, ROWS[:, None], ROWS) is search
    _check_result(search, expected_public, captured=case == "passthrough-hook")
    assert search.cv is public and search.shared_fold_info["completed"]


@pytest.mark.parametrize("policy", ["delegate", "veto"])
def test_changed_live_array_reducer(folds, monkeypatch, policy):
    # Get the actual array in the selector's pickle state. to_numpy(copy=False)
    # can return a new read-only view. The separate scorer capture must also
    # pass through the effective reducer when the reducer inspects that object.
    first = folds[0][0]
    raw = first._data if type(first) is pd.Index else first._values
    assert type(raw) in (np.ndarray, np.memmap)
    search = _search(GridSearchCV, folds, raw)
    pickler = get_loky_pickler()
    dump = pickler.dump
    slot = pickle.Pickler.dispatch_table
    installed, seen = [], []
    marker = "policy rejected original pandas selector data"

    def change_active(self, batch):
        table = slot.__get__(self)
        kind = type(raw)
        if kind in table:
            native = table[kind]

            def inspect_array(value):
                if value is raw:
                    seen.append(True)
                    if policy == "veto":
                        raise ValueError(marker)
                return native(value)

            slot.__set__(self, {**table, kind: inspect_array})
            installed.append(True)
        return dump(self, batch)

    monkeypatch.setattr(pickler, "dump", change_active)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        try:
            fit_shared_fold_search(search, ROWS[:, None], ROWS)
        except pickle.PicklingError as error:
            # joblib may put the originating exception in its remote traceback.
            messages = f"{error}\n{error.__cause__}"
            if policy == "veto" and seen:
                assert marker in messages
            else:
                assert "owning" in messages and "fold" in messages
                assert "serializ" in messages or "verify" in messages
            assert not search.shared_fold_info["completed"]
        else:
            assert policy == "delegate", "the search silently bypassed the installed veto"
            _check_result(search, EDITED, captured=True)
            assert seen and search.shared_fold_info["completed"]
    assert installed and search.cv is folds and search.scoring.raw is raw
