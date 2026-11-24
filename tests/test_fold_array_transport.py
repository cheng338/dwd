"""Public process-search behavior for detached and native numeric fold arrays."""
import copyreg
import gc
import json
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from inspect import isdatadescriptor
from io import BytesIO
from pathlib import Path
from pickle import PicklingError
from tempfile import TemporaryDirectory
from threading import Barrier
from types import SimpleNamespace
from weakref import ref

import numpy as np
import pytest
from joblib import parallel_config
from joblib._parallel_backends import LokyBackend, MultiprocessingBackend
from joblib.externals.loky import get_reusable_executor
from joblib.externals.loky.backend.reduction import get_loky_pickler, get_loky_pickler_name, set_loky_pickler
from joblib.parallel import BatchedCalls
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV

from dwd import model_selection as dwd_search
from dwd.model_selection import fit_shared_fold_search

pytestmark = pytest.mark.usefixtures("search_parent_native_thread")


try:
    from joblib.externals.cloudpickle import CloudPickler  # bundled in joblib 1.5
except ModuleNotFoundError as error:
    if error.name != "joblib.externals.cloudpickle":
        raise
    from cloudpickle import CloudPickler  # a joblib dependency from 1.6


_FIRST = (9, 1, 15, 5, 13, 7, 11, 3)
_SECOND = (2, 12, 4, 14, 0, 10, 6, 8)


def _targets(rows):
    return ((rows < 10) & (rows % 2 == 0)).astype(int)


class _RememberRows(DummyRegressor):
    def fit(self, X, y):
        self.train_ids_ = tuple(X[:, 0])
        np.testing.assert_array_equal(y, _targets(X[:, 0]))
        return super().fit(X, y)


@pytest.mark.parametrize("threshold", [None, 8])
@pytest.mark.parametrize("form", ["tuples", "pairs", "stack", "memmap_rows"])
def test_worker_preserves_detached_fold_array(tmp_path, threshold, form):
    expected = np.array([[_FIRST, _SECOND], [_SECOND, _FIRST]])
    path = tmp_path / "folds.npy"
    np.save(path, 15 - expected)
    mapped = np.load(path, mmap_mode="c")
    try:
        mapped[:] = expected
        stack = mapped if form == "memmap_rows" else np.asarray(mapped)
        if form == "stack":
            folds = captured = stack
        elif form == "pairs":
            folds = list(stack)
            captured = folds[0]
        else:
            a, b = stack[0]
            folds = [(a, b), (b, a)]
            captured = a
        parent = os.getpid()

        def score(model, X, y):
            assert os.getpid() != parent
            cv = search.cv
            if form == "stack":
                assert cv is captured
            elif form == "pairs":
                assert cv[0] is captured
            else:
                assert cv[0][0] is cv[1][1] is captured
            assert type(captured) is np.ndarray
            np.testing.assert_array_equal(cv, expected)
            np.testing.assert_array_equal(y, _targets(X[:, 0]))
            actual = (model.train_ids_, tuple(X[:, 0]))
            assert actual in ((_FIRST, _SECOND), (_SECOND, _FIRST))
            plain = type(cv[0][0]) is np.ndarray
            return {"selection": float(model.constant == int(plain)),
                    "reverse": float(actual == (_SECOND, _FIRST)),
                    "value": -float(np.mean(np.abs(model.predict(X) - y)))}

        X = np.arange(16)[:, None]
        search = GridSearchCV(_RememberRows(strategy="constant"), {"constant": [0, 1]},
                              cv=folds, scoring=score, refit="selection", n_jobs=2,
                              pre_dispatch=1, error_score="raise")
        with parallel_config(backend="loky", inner_max_num_threads=1, max_nbytes=threshold):
            fit_shared_fold_search(search, X, _targets(X[:, 0]))
        results = search.cv_results_
        for fold, values in enumerate(([-.625, -.375], [0., -1.])):
            np.testing.assert_array_equal(results[f"split{fold}_test_selection"], [0., 1.])
            np.testing.assert_array_equal(results[f"split{fold}_test_reverse"], [fold, fold])
            np.testing.assert_array_equal(results[f"split{fold}_test_value"], values)
        assert search.best_params_ == {"constant": 1} and search.cv is folds
        assert search.best_estimator_.train_ids_ == tuple(range(16))
        np.testing.assert_array_equal(search.predict(X), np.ones(len(X)))
        np.testing.assert_array_equal(mapped, expected)
        np.testing.assert_array_equal(np.load(path), 15 - expected)
    finally:
        mapped._mmap.close()


class _PriorityWeightedError:
    def __init__(self, priority_rows):
        self.priority_rows = priority_rows

    def __call__(self, model, X, y):
        weights = np.where(np.isin(X[:, 0], self.priority_rows), 3., 1.)
        return -np.mean(weights * np.abs(y - model.predict(X)))


@pytest.mark.parametrize("threshold", [None, 8])
@pytest.mark.parametrize("recognized_loader", [True, False], ids=["native_loader", "unknown_loader"])
def test_priority_weights_use_private_fold_values(tmp_path, threshold, recognized_loader, monkeypatch):
    if not recognized_loader:
        monkeypatch.setattr(dwd_search, "_JOBLIB_TEMP_LOADER", None)
    rows = np.arange(16)
    path = tmp_path / "priority.npy"
    np.save(path, rows[::-1])
    mapped = np.load(path, mmap_mode="c")
    try:
        mapped[:] = rows
        priority, other = mapped[:8], mapped[8:]
        folds = [(priority, other), (other, priority)]
        search = GridSearchCV(DummyRegressor(strategy="constant"), {"constant": [0, 1]},
                              scoring=_PriorityWeightedError(priority), cv=folds, n_jobs=2,
                              pre_dispatch=1, error_score="raise")
        with parallel_config(backend="loky", inner_max_num_threads=1, max_nbytes=threshold):
            fit_shared_fold_search(search, rows[:, None], rows >= 8)
        results = search.cv_results_
        np.testing.assert_array_equal(results["split0_test_score"], [-1., 0.])
        np.testing.assert_array_equal(results["split1_test_score"], [0., -3.])
        np.testing.assert_array_equal(results["mean_test_score"], [-.5, -1.5])
        assert search.best_estimator_.constant == 0 and search.cv is folds
        np.testing.assert_array_equal(search.predict(rows[:, None]), np.zeros(len(rows)))
        np.testing.assert_array_equal(mapped, rows)
        np.testing.assert_array_equal(np.load(path), rows[::-1])
    finally:
        mapped._mmap.close()


@pytest.mark.parametrize("threshold", [None, 8])
@pytest.mark.parametrize("owning", [False, True], ids=["read_only_mapping", "owning"])
def test_safe_indices_follow_native_joblib_transport(tmp_path, threshold, owning):
    rows = np.arange(16)
    np.save(tmp_path / "indices.npy", rows)
    mapped = np.load(tmp_path / "indices.npy", mmap_mode="r")
    try:
        a, b = np.asarray(mapped[:8]), np.asarray(mapped[8:])
        if owning:
            a, b = a.copy(), b.copy()
        folds = [(a, b), (b, a)]
        parent = os.getpid()

        def score(model, X, y):
            assert os.getpid() != parent
            row = search.cv[0][0]
            assert row is search.cv[1][1] is a
            np.testing.assert_array_equal(row, np.arange(8))
            return {"plain": float(type(row) is np.ndarray),
                    "writeable": float(row.flags.writeable)}

        with parallel_config(backend="loky", inner_max_num_threads=1, max_nbytes=threshold):
            diagnostics = []
            for native in (True, False):
                search = GridSearchCV(DummyRegressor(), {}, cv=folds, scoring=score, refit=False,
                                      n_jobs=2, pre_dispatch=1, error_score="raise")
                if native:
                    search.fit(rows[:, None], rows)
                else:
                    fit_shared_fold_search(search, rows[:, None], rows)
                diagnostics.append([search.cv_results_[f"split{i}_test_{metric}"][0]
                                    for i in (0, 1) for metric in ("plain", "writeable")])
        assert diagnostics[0] == diagnostics[1]
    finally:
        mapped._mmap.close()


class _RegisteredSearch(GridSearchCV):
    pass


def _reduce_registered_search(search):
    state = dict(search.__getstate__())
    state["independent"] = SimpleNamespace(raw=search.cv[0][0])
    return copyreg.__newobj__, (type(search),), state


@pytest.mark.parametrize("table_mode", ["copyreg", "c-slot", "late-family"])
def test_registered_search_reducer_preserves_independent_raw_state(tmp_path, monkeypatch, table_mode):
    rows = np.arange(16)
    path = tmp_path / "opaque.npy"
    np.save(path, rows[::-1])
    mapped = np.load(path, mmap_mode="c")
    missing = object()
    previous = copyreg.dispatch_table.get(_RegisteredSearch, missing)
    try:
        if table_mode != "copyreg":
            pickler = get_loky_pickler()
            initialize, dump = pickler.__init__, pickler.dump
            slot = pickle.Pickler.dispatch_table

            class FamilyTable(dict):
                def __missing__(self, kind):
                    if issubclass(kind, _RegisteredSearch):
                        return _reduce_registered_search
                    raise KeyError(kind)

            def configure(self, *args, **kwargs):
                initialize(self, *args, **kwargs)
                if table_mode == "c-slot":
                    # Only the Python attribute changes. The C table still
                    # has both the array and the search policies.
                    self.dispatch_table = {}
                else:
                    slot.__get__(self).pop(_RegisteredSearch, None)

            def dump_batch(self, batch):
                if table_mode == "late-family":
                    slot.__set__(self, FamilyTable(slot.__get__(self)))
                return dump(self, batch)

            monkeypatch.setattr(pickler, "__init__", configure)
            monkeypatch.setattr(pickler, "dump", dump_batch)
        mapped[:] = rows
        a, b = np.asarray(mapped[:8]), np.asarray(mapped[8:])
        folds = [(a, b), (b, a)]
        parent = os.getpid()

        def score(model, X, y):
            assert os.getpid() != parent
            public = search.cv[0][0]
            raw = search.independent.raw
            assert raw is not public
            np.testing.assert_array_equal(public, rows[:8])
            np.testing.assert_array_equal(raw, rows[::-1][:8])
            return float(np.mean(X[:, 0]))

        copyreg.pickle(_RegisteredSearch, _reduce_registered_search)
        search = _RegisteredSearch(DummyRegressor(), {}, cv=folds, scoring=score, refit=False,
                                   n_jobs=2, pre_dispatch=1, error_score="raise")
        with parallel_config(backend="loky", inner_max_num_threads=1, max_nbytes=8):
            fit_shared_fold_search(search, rows[:, None], rows)
        assert copyreg.dispatch_table[_RegisteredSearch] is _reduce_registered_search
        for i, expected in enumerate((11.5, 3.5)):
            np.testing.assert_array_equal(search.cv_results_[f"split{i}_test_score"], [expected])
        assert search.cv is folds and not hasattr(search, "independent")
        np.testing.assert_array_equal(mapped, rows)
        np.testing.assert_array_equal(np.load(path), rows[::-1])
    finally:
        if previous is missing:
            copyreg.dispatch_table.pop(_RegisteredSearch, None)
        else:
            copyreg.pickle(_RegisteredSearch, previous)
    assert copyreg.dispatch_table.get(_RegisteredSearch, missing) is previous


_VETO_MARKER = "dwd-fold-test-custom-pickler-veto"
_TOKEN_MARKER = "dwd-fold-test-unresolved-persistent-id"
_SWITCH_MARKER = "dwd-fold-test-switch-persistent-id"


def _positive_token(value):
    if type(value) is str and value == _TOKEN_MARKER:
        return ("decoded-token",)
    return None


class Pickler(CloudPickler):
    # loky looks up this exact class name in a configured pickler module.
    def persistent_id(self, value):
        if type(value) is str and value == _VETO_MARKER:
            raise ValueError("reject fold scorer marker")
        return _positive_token(value)


class _InstanceTokenPickler(CloudPickler):
    def __init__(self, file, protocol=None):
        super().__init__(file, protocol=protocol)
        self.persistent_id = _positive_token


class _DumpPersistentPickler(CloudPickler):
    def dump(self, obj):
        self.persistent_id = self.begin_stream
        return super().dump(obj)

    def begin_stream(self, value):
        if type(value) is str and value == _VETO_MARKER:
            raise ValueError("late persistent ID rejected scorer")
        return _positive_token(value)


class _SwitchPersistentPickler(_DumpPersistentPickler):
    def begin_stream(self, value):
        if type(value) is str and value == _SWITCH_MARKER:
            self.persistent_id = super().begin_stream
        return None


def _uses_selected_policy(pickler):
    # CPython's writable slot reads the current callback for each object.
    # Its method-based pickler keeps the callback selected at dump entry.
    return pickler is not _SwitchPersistentPickler or isdatadescriptor(pickle.Pickler.persistent_id)


@pytest.fixture
def worker_pickler(monkeypatch):
    previous = get_loky_pickler_name()

    def select(cls):
        monkeypatch.setitem(globals(), "Pickler", cls)
        set_loky_pickler(__name__)
        return get_loky_pickler()

    try:
        yield select
    finally:
        set_loky_pickler(previous)


@pytest.fixture(params=[False, True], ids=["data", "search-state"])
def worker_round_trip(request):
    def round_trip(batch, search, external=None):
        worker = dwd_search._WorkerSearchBatch(batch, LokyBackend(), search,
                                               dict if request.param else None, data=external)
        loader, args = worker.encode(pickle.HIGHEST_PROTOCOL)
        return loader(*args)

    return round_trip


@pytest.mark.skipif(isdatadescriptor(pickle.Pickler.persistent_id), reason="requires method lookup at dump entry")
def test_worker_persistent_id_missing_fallback_runs_once(worker_pickler, worker_round_trip):
    lookups = []

    class MissingOnce(CloudPickler):
        def __getattribute__(self, name):
            if name == "persistent_id":
                raise AttributeError(name)
            return super().__getattribute__(name)

        def __getattr__(self, name):
            if name != "persistent_id":
                raise AttributeError(name)
            lookups.append(name)
            if len(lookups) == 1:
                raise AttributeError("first persistent_id fallback failed")
            return lambda value: None

    native = worker_pickler(MissingOnce)
    search = _MarkedScore()
    with pytest.raises(AttributeError, match="first persistent_id fallback failed"):
        native(BytesIO()).dump((search,))
    assert lookups == ["persistent_id"]
    lookups.clear()
    with pytest.raises(AttributeError, match="first persistent_id fallback failed"):
        worker_round_trip((search,), search)
    assert lookups == ["persistent_id"]


@pytest.mark.skipif(isdatadescriptor(pickle.Pickler.persistent_id), reason="requires method lookup at dump entry")
@pytest.mark.parametrize("fallback", [False, True], ids=["property", "fallback"])
def test_worker_persistent_id_can_inspect_native_reducer(worker_pickler, fallback):
    observed = []

    class ReadingProperty(CloudPickler):
        def select_hook(self):
            observed.append(self.reducer_override == super().reducer_override)
            return lambda value: None

        @property
        def persistent_id(self):
            return self.select_hook()

    class ReadingFallback(ReadingProperty):
        @property
        def persistent_id(self):
            raise AttributeError("persistent_id")

        def __getattr__(self, name):
            if name == "persistent_id":
                return self.select_hook()
            raise AttributeError(name)

    native = worker_pickler(ReadingFallback if fallback else ReadingProperty)
    original = [([0], [2])]
    prepared = [([2], [0])]
    search = _SavedFieldsSearch(DummyRegressor(), {}, cv=original)
    native(BytesIO()).dump((search,))
    assert observed == [True]
    observed.clear()
    batch = dwd_search._WorkerSearchBatch((search,), LokyBackend(), search,
                                          lambda: {id(original): lambda: prepared})
    loader, args = batch.encode(pickle.HIGHEST_PROTOCOL)
    restored, = loader(*args)
    assert (observed, restored.cv) == ([True], prepared)


def test_class_persistent_id_releases_worker_search(worker_pickler, worker_round_trip):
    instances = []

    class Observed(CloudPickler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(ref(self))

        def persistent_id(self, value):
            return None

    native = worker_pickler(Observed)
    enabled = gc.isenabled()
    gc.disable()
    try:
        for encode in (lambda batch, search: native(BytesIO()).dump(batch), worker_round_trip):
            search = _MarkedScore()
            saved = ref(search)
            encode((search,), search)
            del search
            assert len(instances) == 1
            assert instances.pop()() is None
            assert saved() is None
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("fit", [GridSearchCV.fit, fit_shared_fold_search], ids=["stock", "helper"])
def test_completed_search_releases_dynamic_pickler_and_input(tmp_path, monkeypatch, fit):
    def run_search():
        X = np.arange(8)[:, None]
        visits = []

        class CountInputs(CloudPickler):
            def persistent_id(self, value):
                if value is X:
                    visits.append(True)
                return None

        path = tmp_path / "indices.npy"
        np.save(path, X[::-1, 0])
        mapped = np.load(path, mmap_mode="c")
        previous = get_loky_pickler_name()
        try:
            mapped[:] = X[:, 0]
            a, b = np.asarray(mapped).reshape(2, 4)
            if fit is GridSearchCV.fit:
                a, b = a.copy(), b.copy()
            folds = [(a, b), (b, a)]
            search = _SavedFieldsSearch(DummyRegressor(strategy="constant"), {"constant": [0, 1]},
                                        cv=folds, n_jobs=2, pre_dispatch=1, error_score="raise")
            search.parent_pid = os.getpid()
            search.scoring = search.score_first_fold
            with monkeypatch.context() as patch:
                patch.setitem(globals(), "Pickler", CountInputs)
                set_loky_pickler(__name__)
                with parallel_config(backend="loky", inner_max_num_threads=1):
                    fit(search, X, X[:, 0])
            assert visits and search.best_params_ == {"constant": 1}
        finally:
            get_reusable_executor().shutdown(wait=True)
            set_loky_pickler(previous)
            mapped._mmap.close()
        return ref(X), ref(CountInputs)

    data, pickler_class = run_search()
    gc.collect()
    assert (data() is None, pickler_class() is None) == (True, True)


def _assert_token_collision_rejected(native, batch, search, external, worker_round_trip):
    stream = BytesIO()
    native(stream).dump(batch)
    with pytest.raises(pickle.UnpicklingError, match="persistent"):
        pickle.loads(stream.getvalue())
    # The helper may reject a custom callback while saving or an unknown
    # persistent ID while loading. Both leave the private binding untouched.
    with pytest.raises((PicklingError, pickle.UnpicklingError), match="(?i)persistent|owning|binding"):
        worker_round_trip(batch, search, {id(external): external})


def test_retained_backend_pickler_releases_finished_search(worker_pickler, worker_round_trip):
    retained = []

    class Observer(CloudPickler):
        def dump(self, value):
            result = super().dump(value)
            self.clear_memo()
            retained.append(self)
            return result

    native = worker_pickler(Observer)
    for encode in (lambda batch, search: native(BytesIO()).dump(batch), worker_round_trip):
        search = _MarkedScore()
        saved = ref(search)
        encode((search,), search)
        del search
        gc.collect()
        assert len(retained) == 1
        assert saved() is None
        retained.clear()


def test_worker_stream_preserves_backend_attributes(worker_pickler, worker_round_trip):
    calls = []

    class Configured(CloudPickler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.search = 'observe-fit-inputs'

        def persistent_id(self, value):
            return self.external_id(value)

        def external_id(self, value):
            if type(value) is str and value == _TOKEN_MARKER:
                calls.append(self.search)
            return None

    native = worker_pickler(Configured)
    search = _MarkedScore()
    batch = (search, _TOKEN_MARKER)
    for encode in (lambda batch, search: native(BytesIO()).dump(batch), worker_round_trip):
        encode(batch, search)
        assert calls == ['observe-fit-inputs']
        calls.clear()


@pytest.mark.skipif(not isdatadescriptor(pickle.Pickler.persistent_id), reason="requires the writable C callback slot")
@pytest.mark.parametrize("binding", ["descriptor", "direct-c"])
def test_rebound_native_token_cannot_claim_worker_data(worker_pickler, worker_round_trip, binding):
    def token(value):
        if type(value) is str and value == _TOKEN_MARKER:
            return 0
        return None

    class Descriptor(CloudPickler):
        @property
        def persistent_id(self):
            try:
                return pickle.Pickler.persistent_id.__get__(self, type(self))
            except AttributeError:
                return self.begin

        @persistent_id.setter
        def persistent_id(self, callback):
            pickle.Pickler.persistent_id.__set__(self, callback)

        def begin(self, value):
            if type(value) is str and value == _SWITCH_MARKER:
                self.persistent_id = token
            return None

    class DirectC(CloudPickler):
        def persistent_id(self, value):
            if type(value) is str and value == _SWITCH_MARKER:
                pickle.Pickler.persistent_id.__set__(self, token)
            return None

    native = worker_pickler(Descriptor if binding == "descriptor" else DirectC)
    external = bytearray(b"data")
    search = _MarkedScore()
    batch = (external, search, _SWITCH_MARKER, _TOKEN_MARKER)
    # The first item creates a legitimate private binding at position zero.
    # The final marker requests the same integer through the backend's C slot.
    _assert_token_collision_rejected(native, batch, search, external, worker_round_trip)


@pytest.mark.skipif(not isdatadescriptor(pickle.Pickler.persistent_id), reason="requires the writable C callback slot")
def test_native_token_rejected_after_callback_restoration(worker_pickler, worker_round_trip):
    active = None
    previous = []
    changes = []
    slot = pickle.Pickler.persistent_id

    class Observed(CloudPickler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            nonlocal active
            active = ref(self)

        def persistent_id(self, value):
            return None

    def token(value):
        if type(value) is str and value == _TOKEN_MARKER:
            return 0
        return None

    class ChangeCallback:
        def __init__(self, install):
            self.install = install

        def __reduce__(self):
            pickler = active()
            if self.install:
                previous.append(slot.__get__(pickler, type(pickler)))
                slot.__set__(pickler, token)
            else:
                slot.__set__(pickler, previous.pop())
            changes.append(self.install)
            return int, (0,)

    native = worker_pickler(Observed)
    external = bytearray(b"data")
    search = _MarkedScore()
    # Both ordinary reducers finish before dump returns. The C callback is
    # identical at entry and exit, although the intervening token is foreign.
    batch = (external, search, ChangeCallback(True), _TOKEN_MARKER, ChangeCallback(False))
    _assert_token_collision_rejected(native, batch, search, external, worker_round_trip)
    assert changes[:2] == [True, False]


class _MarkedScore:
    def __init__(self):
        self.marker = _VETO_MARKER

    def __call__(self, model, X, y):
        return 1.


class _SavedFieldsSearch(GridSearchCV):
    def __getstate__(self):
        fields = vars(self).copy()
        return tuple(fields), fields

    def __setstate__(self, state):
        names, fields = state
        assert names == tuple(fields)
        vars(self).update(fields)

    def score_token(self, model, X, y):
        return float(model.constant == int(self.marker == "decoded-token"))

    def score_first_fold(self, model, X, y):
        assert os.getpid() != self.parent_pid
        return float(model.constant == int(self.cv[0][0][0] == 0))


@pytest.mark.parametrize("fit", [GridSearchCV.fit, fit_shared_fold_search], ids=["stock", "helper"])
def test_worker_dispatch_table_family_veto(tmp_path, worker_pickler, fit):
    message = "family reducer rejected the search"

    def reject(search):
        raise ValueError(message)

    class FamilyTable(dict):
        def __missing__(self, kind):
            if issubclass(kind, GridSearchCV):
                return reject
            raise KeyError(kind)

    class FamilyPickler(CloudPickler):
        def dump(self, obj):
            self._set_dispatch_table(FamilyTable(self.dispatch_table))
            return super().dump(obj)

    worker_pickler(FamilyPickler)
    rows = np.arange(8)
    path = tmp_path / "indices.npy"
    np.save(path, rows[::-1])
    mapped = np.load(path, mmap_mode="c")
    try:
        mapped[:] = rows
        first, second = np.asarray(mapped).reshape(2, 4)
        search = _SavedFieldsSearch(DummyRegressor(), {}, cv=[(first, second), (second, first)],
                                    n_jobs=2, pre_dispatch=1, error_score="raise")
        search.scoring = search.score_first_fold
        search.parent_pid = os.getpid()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            with pytest.raises((ValueError, PicklingError)) as error:
                fit(search, rows[:, None], rows)
        assert str(error.value) == message or f"ValueError: {message}" in str(error.value.__cause__).splitlines()
        np.testing.assert_array_equal(mapped, rows)
        np.testing.assert_array_equal(np.load(path), rows[::-1])
    finally:
        get_reusable_executor().shutdown(wait=True)


class _DispatchScore:
    def __call__(self, model, X, y):
        public = self.search.cv[0][0]
        raw = getattr(self.search, "opaque", None)
        if raw is not None:
            assert raw.row is self.captured
        return {"winner": float(model.constant == int(self.captured[0] == 0)),
                "public": float(public[0]), "captured": float(self.captured[0]),
                "same": float(public is self.captured)}


class _RestoreArrayTemplate:
    def __init__(self, template, native):
        self.template, self.native = template, native

    def __reduce__(self):
        self.template[np.ndarray] = self.native
        return int, (0,)


class _TemplateScore:
    def __getstate__(self):
        fields = {"captured": self.captured, "search": self.search}
        if self.remove_during_save:
            fields["restore"] = _RestoreArrayTemplate(self.template, self.template.pop(np.ndarray))
        return fields

    def __call__(self, model, X, y):
        assert self.captured is self.search.cv[0][0]
        return float(self.captured[0])


@pytest.mark.parametrize("change", ["template-in-save", "template-in-init", "active-table"])
@pytest.mark.parametrize("fit", [GridSearchCV.fit, fit_shared_fold_search], ids=["owning", "mapped"])
def test_worker_dispatch_table_snapshots_array_template(tmp_path, monkeypatch, change, fit):
    rows = np.arange(8)
    path = tmp_path / "indices.npy"
    np.save(path, rows[::-1])
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = rows
    first, second = np.asarray(mapped).reshape(2, 4)
    if fit is GridSearchCV.fit:
        first, second = first.copy(), second.copy()
    folds = [(first, second), (second, first)]
    template = {}
    score = _TemplateScore()
    search = GridSearchCV(DummyRegressor(), {}, scoring=score, cv=folds, refit=False,
                          n_jobs=2, pre_dispatch=1, error_score="raise")
    score.captured, score.search, score.template = first, search, template
    score.remove_during_save = change != "template-in-init"
    backend = LokyBackend(forward_reducers=template)
    pickler = get_loky_pickler()
    initialize, dump = pickler.__init__, pickler.dump

    if change == "template-in-init":
        def initialize_batch(self, writer, reducers=None, **kwargs):
            initialize(self, writer, reducers=reducers, **kwargs)
            if reducers is template:
                self.saved_array_reducer = template.pop(np.ndarray)

        def dump_batch(self, batch):
            if "saved_array_reducer" in vars(self):
                template[np.ndarray] = vars(self).pop("saved_array_reducer")
            return dump(self, batch)

        monkeypatch.setattr(pickler, "__init__", initialize_batch)
        monkeypatch.setattr(pickler, "dump", dump_batch)
    elif change == "active-table":
        def change_active(self, batch):
            slot = pickle.Pickler.dispatch_table
            table = slot.__get__(self)
            if np.ndarray in table:
                native = table[np.ndarray]

                def veto(value):
                    if value is first:
                        raise ValueError("active table rejected the captured row")
                    return native(value)

                slot.__set__(self, {**table, np.ndarray: veto})
            return dump(self, batch)

        monkeypatch.setattr(pickler, "dump", change_active)
    try:
        with parallel_config(backend=backend, inner_max_num_threads=1):
            if change == "active-table":
                with pytest.raises(PicklingError) as error:
                    fit(search, rows[:, None], rows)
                assert "ValueError: active table rejected the captured row" in str(error.value.__cause__)
            else:
                fit(search, rows[:, None], rows)
                np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [0.])
                assert np.ndarray in template
        assert search.cv is folds and score.captured is first
        np.testing.assert_array_equal(mapped, rows)
        np.testing.assert_array_equal(np.load(path), rows[::-1])
    finally:
        get_reusable_executor().shutdown(wait=True)


def _restore_dispatch_fields(search, state):
    names, fields, raw = state
    for name in names:
        setattr(search, name, fields[name])
    search.opaque = raw


class _ReshapeDuringPickle:
    def __init__(self, row):
        self.row = row

    def __reduce__(self):
        self.row.resize(2, 2)
        return int, (0,)


class _CapturedShapeScore:
    def __getstate__(self):
        self.row.resize(4)
        return dict(row=self.row, update=_ReshapeDuringPickle(self.row),
                    search=self.search, parent=os.getpid())

    def __call__(self, model, X, y):
        assert os.getpid() != self.parent
        assert self.row is self.search.opaque.row
        assert self.row.flat[0] == 7
        return float(model.constant == int(self.row.ndim == 1))


def _reduce_search_with_raw_row(search):
    state = (tuple(vars(search)), search.__getstate__(), SimpleNamespace(row=search.scoring.row))
    return copyreg.__newobj__, (type(search),), state, None, None, _restore_dispatch_fields


class _GuardedOffsetMap(np.memmap):
    def __getattribute__(self, name):
        if name == "offset" and "offset_reads" in vars(self):
            self.offset_reads += 1
            raise ValueError("access to the raw backing offset was rejected")
        return super().__getattribute__(name)


@pytest.mark.parametrize("map_class", [np.memmap, _GuardedOffsetMap])
def test_worker_dispatch_table_captures_shape_at_first_visit(tmp_path, map_class):
    rows = np.arange(8, dtype=np.intp)
    path = tmp_path / "capture.bin"
    rows[::-1].tofile(path)
    mapped = map_class(path, mode="c", dtype=rows.dtype, shape=8)
    mapped[:] = rows
    first, second = np.asarray(mapped).reshape(2, 4)
    folds = [(first, second), (second, first)]
    scorer = _CapturedShapeScore()
    search = GridSearchCV(DummyRegressor(strategy="constant"), {"constant": [0, 1]},
                          cv=folds, scoring=scorer, n_jobs=2, pre_dispatch=1, error_score="raise")
    scorer.row, scorer.search = np.asarray(mapped)[:4], search
    backend = LokyBackend(forward_reducers={GridSearchCV: _reduce_search_with_raw_row})
    try:
        with parallel_config(backend=backend, inner_max_num_threads=1, max_nbytes=8):
            fit_shared_fold_search(search, rows[:, None], rows)
        for split in (0, 1):
            np.testing.assert_array_equal(search.cv_results_[f"split{split}_test_score"], [0., 1.])
        assert search.best_params_ == {"constant": 1}
        np.testing.assert_array_equal(search.predict(rows[:, None]), np.ones(8))
        assert search.cv is folds and first.shape == (4,) and not hasattr(search, "opaque")
        np.testing.assert_array_equal(mapped, rows)
        np.testing.assert_array_equal(np.fromfile(path, dtype=rows.dtype), rows[::-1])
    finally:
        get_reusable_executor().shutdown(wait=True)


@pytest.mark.parametrize("custom_map", [False, True])
@pytest.mark.parametrize("raw", [False, True])
def test_worker_dispatch_table_uses_only_selected_raw_offset(tmp_path, custom_map, raw):
    rows = np.arange(8, dtype=np.intp)
    path = tmp_path / "indices.bin"
    rows[::-1].tofile(path)
    cls = _GuardedOffsetMap if custom_map else np.memmap
    mapped = cls(path, mode="c", dtype=rows.dtype, shape=8)
    mapped[:] = rows
    if custom_map:
        mapped.offset_reads = 0
    else:
        mapped.offset = None
    first, second = np.asarray(mapped).reshape(2, 4)
    folds = [(first, second), (second, first)]
    score = _DispatchScore()
    search = GridSearchCV(DummyRegressor(strategy="constant"), {"constant": [0, 1]},
                          cv=folds, scoring=score, refit="winner", n_jobs=2, pre_dispatch=1,
                          error_score="raise")
    score.captured = score.row = first
    score.search = search
    backend = LokyBackend(forward_reducers={GridSearchCV: _reduce_search_with_raw_row} if raw else {})
    try:
        with parallel_config(backend=backend, inner_max_num_threads=1):
            if raw:
                with pytest.raises(PicklingError) as error:
                    fit_shared_fold_search(search, rows[:, None], rows)
                cause = str(error.value.__cause__)
                assert ("access to the raw backing offset" if custom_map else "TypeError") in cause
            else:
                fit_shared_fold_search(search, rows[:, None], rows)
                np.testing.assert_array_equal(search.cv_results_["mean_test_winner"], [0., 1.])
                np.testing.assert_array_equal(search.predict(rows[:, None]), np.ones(8))
                assert search.best_params_ == {"constant": 1}
        if custom_map:
            assert (mapped.offset_reads > 0) == raw
        assert search.cv is folds
        np.testing.assert_array_equal(mapped, rows)
        np.testing.assert_array_equal(np.fromfile(path, dtype=rows.dtype), rows[::-1])
    finally:
        get_reusable_executor().shutdown(wait=True)


@pytest.mark.parametrize("install", ["init", "dump", "inherited-dump"])
@pytest.mark.parametrize("captured_first", [True, False], ids=["capture-first", "search-first"])
@pytest.mark.parametrize("opaque", [False, True], ids=["miss", "registered"])
def test_worker_dispatch_table_preserves_fold_capture(tmp_path, monkeypatch, install, captured_first, opaque):
    rows = np.arange(8)
    path = tmp_path / "indices.npy"
    np.save(path, rows[::-1])
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = rows
    first, second = np.asarray(mapped).reshape(2, 4)
    folds = [(first, second), (second, first)]
    score = _DispatchScore()
    search = GridSearchCV(DummyRegressor(strategy="constant"), {"constant": [0, 1]}, cv=folds,
                          scoring=score, refit="winner", n_jobs=2, pre_dispatch=1, error_score="raise")
    if captured_first:
        score.captured, score.search = first, search
    else:
        score.search, score.captured = search, first
    visits = []

    def reduce_search(value):
        visits.append("reduce")
        state = tuple(vars(value)), value.__getstate__(), SimpleNamespace(row=value.cv[0][0])
        return copyreg.__newobj__, (type(value),), state, None, None, _restore_dispatch_fields

    class Table(dict):
        def __missing__(self, kind):
            if kind is GridSearchCV:
                visits.append("lookup")
                if opaque:
                    return reduce_search
            raise KeyError(kind)

    pickler = get_loky_pickler()
    initialize, dump = pickler.__init__, pickler.dump

    def configure(self):
        self._set_dispatch_table(Table(pickle.Pickler.dispatch_table.__get__(self)))

    if install == "init":
        def initialize_table(self, *args, **kwargs):
            initialize(self, *args, **kwargs)
            configure(self)
        monkeypatch.setattr(pickler, "__init__", initialize_table)
    else:
        def dump_batch(self, batch):
            if isinstance(self, pickler):
                configure(self)
            return dump(self, batch)
        monkeypatch.setattr(CloudPickler if install == "inherited-dump" else pickler, "dump", dump_batch)

    with parallel_config(backend="loky", inner_max_num_threads=1, max_nbytes=8):
        if opaque and captured_first:
            with pytest.raises(PicklingError) as error:
                fit_shared_fold_search(search, rows[:, None], rows)
            assert "Build the folds from owning indices" in str(error.value.__cause__)
        else:
            fit_shared_fold_search(search, rows[:, None], rows)
    if opaque and captured_first:
        assert visits == ["lookup", "reduce"]
        assert not hasattr(search, "best_params_")
    else:
        assert visits == (["lookup", "reduce"] if opaque else ["lookup"]) * 4
        assert search.best_params_ == {"constant": int(not opaque)}
        for metric, expected in (("public", 0), ("captured", 7 if opaque else 0), ("same", int(not opaque))):
            np.testing.assert_array_equal(search.cv_results_["mean_test_" + metric], [expected, expected])
    assert search.cv is folds and score.captured is first and not hasattr(search, "opaque")
    np.testing.assert_array_equal(mapped, rows)
    np.testing.assert_array_equal(np.load(path), rows[::-1])


def _reopen_zero_row(*args):
    from joblib._memmapping_reducer import _strided_from_memmap

    array = np.array(_strided_from_memmap(*args), copy=True)
    array.flat[0] = 0
    return array


def _zero_row_opener():
    return _reopen_zero_row


@pytest.mark.parametrize("policy,capture_first,action,expected", [
    ("declared", True, "transform-before-search", 0),
    ("declared", False, "transform-before-search", 7),
    ("declared", False, "transform", 0),
    ("declared", True, "veto", "reopen veto"),
    ("declared", False, "veto", "reopen veto"),
    ("declared", True, "persistent", "reopen veto"),
    ("empty", True, "veto", 0),
    ("add", True, "pass", "owning indices"),
    ("remove", True, "pass", "owning indices"),
    ("add", False, "transform", 0),
    ("remove", False, "veto", 0),
    ("add", True, "reject", "search reducer refused"),
    ("implicit-owning", True, "veto", 0),
])
def test_worker_dispatch_table_raw_recipe_uses_primary_pickler(tmp_path, monkeypatch, policy, capture_first,
                                                              action, expected):
    from joblib._memmapping_reducer import _strided_from_memmap

    rows = np.arange(8)
    path = tmp_path / "indices.npy"
    np.save(path, rows[::-1])
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = rows
    first, second = np.asarray(mapped).reshape(2, 4)
    if policy == "implicit-owning":
        first, second = first.copy(), second.copy()
    folds = [(first, second), (second, first)]
    score = _DispatchScore()
    search = GridSearchCV(DummyRegressor(strategy="constant"), {"constant": [0, 1]}, cv=folds,
                          scoring=score, refit="winner", n_jobs=2, pre_dispatch=1, error_score="raise")
    if capture_first:
        score.captured, score.search = first, search
    else:
        score.search, score.captured = search, first
    score.row = first
    pickler = get_loky_pickler()
    native, dump = pickler.reducer_override, pickler.dump
    events = []

    def reduce_search(value):
        events.append("registered")
        if action == "reject":
            raise ValueError("search reducer refused")
        return _reduce_search_with_raw_row(value)

    class Table(dict):
        def __getitem__(self, kind):
            if kind is GridSearchCV:
                events.append("lookup")
            return super().__getitem__(kind)

        def __missing__(self, kind):
            if kind is GridSearchCV and policy == "implicit-owning":
                return reduce_search
            raise KeyError(kind)

    def select_table(obj, batch):
        obj._set_dispatch_table(Table(pickle.Pickler.dispatch_table.__get__(obj)))
        return dump(obj, batch)

    def override(obj, value):
        if type(value) is BatchedCalls:
            obj.saving_batch = True
            events.append("batch")
        elif value is search:
            obj.saw_search = True
            table = pickle.Pickler.dispatch_table.__get__(obj)
            if policy == "add":
                table[GridSearchCV] = reduce_search
            elif policy == "remove":
                table.pop(GridSearchCV, None)
        elif value is _strided_from_memmap:
            in_batch = getattr(obj, "saving_batch", False)
            before = not getattr(obj, "saw_search", False)
            events.append(("reopen", in_batch, before))
            if in_batch and action == "veto":
                raise ValueError("reopen veto")
            if in_batch and (action == "transform" or (action == "transform-before-search" and before)):
                return _zero_row_opener, ()
        return native(obj, value)

    def check_id(obj, value):
        if action == "persistent" and value is _strided_from_memmap and getattr(obj, "saving_batch", False):
            raise ValueError("reopen veto")
        return None

    reducers = {GridSearchCV: reduce_search} if policy in ("declared", "remove") else {}
    backend = LokyBackend(forward_reducers=reducers)
    monkeypatch.setattr(pickler, "dump", select_table)
    monkeypatch.setattr(pickler, "reducer_override", override)
    monkeypatch.setattr(pickler, "persistent_id", check_id)
    with parallel_config(backend=backend, inner_max_num_threads=1, max_nbytes=8):
        if isinstance(expected, str):
            with pytest.raises(PicklingError) as error:
                fit_shared_fold_search(search, rows[:, None], rows)
            assert expected in str(error.value.__cause__)
            assert not hasattr(search, "best_params_")
        else:
            fit_shared_fold_search(search, rows[:, None], rows)
            assert search.best_params_ == {"constant": int(expected == 0)}
            np.testing.assert_array_equal(search.predict(rows[:, None]), np.full(8, int(expected == 0)))
            for metric, result in (("captured", expected), ("public", 0),
                                   ("same", int(policy in ("empty", "remove", "implicit-owning")))):
                np.testing.assert_array_equal(search.cv_results_["mean_test_" + metric], [result, result])
            assert events.count("lookup") == 4
    reopen = [event for event in events if type(event) is tuple]
    if policy == "declared" and action != "persistent":
        assert reopen == [("reopen", True, capture_first)] * (1 if isinstance(expected, str) else 4)
    elif policy == "empty" or policy == "implicit-owning" or (policy == "remove" and not capture_first):
        assert not reopen
    assert search.cv is folds and score.captured is first and not hasattr(search, "opaque")
    np.testing.assert_array_equal(mapped, rows)
    np.testing.assert_array_equal(np.load(path), rows[::-1])


class _ChangingOverlapScore:
    def __getstate__(self):
        batch = len(self.sent) + 1
        if self.change == "shape":
            self.row.resize((8,) if batch == 1 else (2, 4))
        elif self.change == "value":
            self.row.flat[-1] = 10 * batch
        elif self.change == "zero":
            self.row.flat[-1] = -0. if batch > 1 else 0.
        elif self.change == "nan":
            self.row.view(np.uint64).flat[-1] = 0x7ff8000000000000 + batch
        self.sent.append(self.observe())
        if self.capture_first:
            return dict(change=self.change, row=self.row, again=self.row, search=self.search)
        return dict(change=self.change, search=self.search, row=self.row, again=self.row)

    def observe(self):
        if self.change == "shape":
            return self.row.ndim
        if self.change == "zero":
            return int(np.signbit(self.row.flat[-1]))
        if self.change == "nan":
            return int(self.row.view(np.uint64).flat[-1]) & 255
        return int(self.row.flat[-1])

    def __call__(self, estimator, X, y):
        assert self.again is self.row
        if hasattr(self.search, "opaque"):
            assert self.row is self.search.opaque.row
        first, second = self.search.cv[0]
        np.testing.assert_array_equal(first, np.arange(4))
        np.testing.assert_array_equal(second, np.arange(4, 8))
        observed = self.observe()
        threshold = {"shape": 1, "value": 15, "zero": 0, "nan": 1, "stable": 0}[self.change]
        return dict(observed=float(observed), winner=float(estimator.constant == int(observed > threshold)))


@pytest.mark.parametrize("policy,change,capture_first,values,copies", [
    ("declared", "stable", True, [0, 0, 0, 0], 0),
    ("declared", "stable", False, [0, 0, 0, 0], 0),
    ("miss", "stable", True, [11, 11, 11, 11], 1),
    ("miss", "shape", True, [1, 2, 2, 2], 2),
    ("raw-first", "shape", True, [1, 2, 2, 2], 1),
    ("miss", "value", True, [10, 20, 30, 40], 4),
    ("raw-first", "value", True, [0, 20, 30, 40], 3),
    ("miss", "zero", True, [0, 1, 1, 1], 2),
    ("miss", "nan", True, [1, 2, 3, 4], 4),
])
def test_worker_dispatch_table_refreshes_separate_view(tmp_path, monkeypatch, policy, change,
                                                      capture_first, values, copies):
    from joblib import _memmapping_reducer

    rows = np.arange(12, dtype=np.int64)
    path = tmp_path / "indices.npy"
    np.save(path, rows[::-1])
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = rows
    first, second = np.asarray(mapped)[:8].reshape(2, 4)
    folds = [(first, second), (second, first)]
    row = np.asarray(mapped)[4:12]
    if change in ("zero", "nan"):
        row = row.view(np.float64)
    scorer = _ChangingOverlapScore()
    scorer.row, scorer.change, scorer.sent, scorer.capture_first = row, change, [], capture_first
    search = GridSearchCV(DummyRegressor(strategy="constant"), {"constant": [0, 1]}, cv=folds, scoring=scorer,
                          refit="winner", n_jobs=2, pre_dispatch=1, error_score="raise")
    scorer.search = search
    template = {}
    if policy != "miss":
        def registered(value):
            if policy == "raw-first":
                del template[GridSearchCV]
            return _reduce_search_with_raw_row(value)
        template[GridSearchCV] = registered

    snapshots, files = [], set()
    copy, dump = dwd_search._snapshot_search_input, _memmapping_reducer.dump

    def copy_row(value):
        detached = copy(value)
        if value is row:
            snapshots.append(detached)
        return detached

    def write_array(value, filename, *args, **kwargs):
        if any(value.base is snapshot for snapshot in snapshots):
            files.add(filename)
        return dump(value, filename, *args, **kwargs)

    monkeypatch.setattr(dwd_search, "_snapshot_search_input", copy_row)
    monkeypatch.setattr(_memmapping_reducer, "dump", write_array)
    with parallel_config(backend=LokyBackend(forward_reducers=template), inner_max_num_threads=1, max_nbytes=8):
        fit_shared_fold_search(search, rows[:, None], rows)
    for split in (0, 1):
        np.testing.assert_array_equal(search.cv_results_[f"split{split}_test_observed"], values[split::2])
    expected = int(change != "stable" or policy == "miss")
    assert search.best_params_ == {"constant": expected}
    np.testing.assert_array_equal(search.predict(rows[:, None]), np.full(12, expected))
    assert len(snapshots) == len(files) == copies
    assert scorer.row is row and scorer.search is search and search.cv is folds
    np.testing.assert_array_equal(mapped[:8], rows[:8])
    np.testing.assert_array_equal(np.load(path), rows[::-1])


class _IndependentFoldScore:
    def __call__(self, estimator, X, y):
        assert self.captured is self.again
        return float(estimator.constant == int(self.captured[0] == 0))


class _OnceModeMap(np.memmap):
    def __getattribute__(self, name):
        if name in ("mode", "offset") and vars(self).get("checking", False):
            if name == "mode":
                self.mode_reads += 1
                if self.mode_reads > 1:
                    raise ValueError("read the captured map's mode twice")
                if self.live_mode is not None:
                    return self.live_mode
            if name == "offset" and not self.allow_raw:
                raise ValueError("opened an unselected raw map")
        return super().__getattribute__(name)


class _GetterModeMap(_OnceModeMap):
    @property
    def mode(self):
        return self._stored_mode

    @mode.setter
    def mode(self, value):
        self._stored_mode = value


class _NormalizedModeMap(_OnceModeMap):
    def __setattr__(self, name, value):
        if name == "mode":
            value = {"c": "copyonwrite", "r+": "readwrite"}.get(value, value)
        super().__setattr__(name, value)

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if name == "mode":
            return {"copyonwrite": "c", "readwrite": "r+"}.get(value, value)
        return value


class _TextMode(str):
    pass


class _NumpyTextMode(np.str_):
    pass


class _NumpyModeWithTextComparisons(np.str_):
    __eq__ = str.__eq__
    __ne__ = str.__ne__
    __hash__ = str.__hash__


@pytest.mark.parametrize("equal", [str.__eq__, np.str_.__eq__])
@pytest.mark.parametrize("unequal", [str.__ne__, np.str_.__ne__])
def test_worker_dispatch_table_builtin_mode_comparisons(equal, unequal):
    from dwd.model_selection import _map_mode_text

    class Mode(np.str_):
        __eq__ = equal
        __ne__ = unequal

    assert type(_map_mode_text(Mode("r+"))) is str
    assert _map_mode_text(Mode("r+")) == "r+"

    def custom_comparison(self, other):
        raise AssertionError("Mode classification invoked a caller's comparison")

    for method in ("__eq__", "__ne__"):
        overridden = type("CustomMode", (Mode,), {method: custom_comparison})
        assert _map_mode_text(overridden("r+")) is None


@pytest.mark.parametrize("policy,capture_first", [
    ("declared", True), ("declared", False), ("miss", True), ("miss", False), ("implicit", True),
    ("disjoint", True), ("safe", True), ("getter-safe", True), ("getter-private", True),
    ("getter-now-private", True), ("getter-now-safe", True), ("getter-stack", True), ("getter-pair", True),
    ("normalized-private", True), ("normalized-safe", True),
    ("scalar-private", True), ("scalar-safe", True), ("inherited-safe", True),
    ("inherited-scalar-safe", True), ("inherited-scalar-private", True),
    ("scalar-python-comparisons", True),
])
def test_worker_dispatch_table_reads_capture_mode_once(tmp_path, monkeypatch, policy, capture_first):
    from joblib._memmapping_reducer import _strided_from_memmap

    nested = policy in ("getter-stack", "getter-pair")
    rows = np.arange(16 if nested else 8)
    path = tmp_path / "rows.bin"
    storage = np.arange(12) if policy == "disjoint" else rows
    safe = nested or policy in ("safe", "getter-safe", "getter-now-private", "normalized-safe", "scalar-safe",
                                "inherited-safe", "inherited-scalar-safe", "scalar-python-comparisons")
    mode = "r+" if safe else "c"
    if policy.startswith("scalar"):
        mode = np.str_(mode)
    if policy == "inherited-safe":
        mode = _TextMode(mode)
    if policy.startswith("inherited-scalar"):
        mode = _NumpyTextMode(mode)
    if policy == "scalar-python-comparisons":
        mode = _NumpyModeWithTextComparisons(mode)
    disk = storage if mode == "r+" else storage[::-1]
    disk.tofile(path)
    map_type = _GetterModeMap if policy.startswith("getter") else _OnceModeMap
    if policy.startswith("normalized"):
        map_type = _NormalizedModeMap
    mapped = map_type(path, mode=mode, dtype=rows.dtype, shape=storage.size)
    mapped[:] = storage
    first, second = np.asarray(mapped)[:8].reshape(2, 4)
    captured = np.asarray(mapped)[8:] if policy == "disjoint" else first.view()
    if mode == "r+":
        first, second = first[::-1], second[::-1]
    folds = [(first, second), (second, first)]
    if nested:
        stack = np.asarray(mapped).reshape(2, 2, 4)
        folds = stack[::-1] if policy == "getter-stack" else [pair[::-1] for pair in stack]
    public_start = int(folds[0][0][0])
    score = _DispatchScore()
    search = GridSearchCV(DummyRegressor(strategy="constant"), {"constant": [0, 1]}, cv=folds,
                          scoring=score, refit="winner", n_jobs=2, pre_dispatch=1, error_score="raise")
    if capture_first:
        score.captured, score.search = captured, search
    else:
        score.search, score.captured = search, captured
    score.row = captured
    pickler = get_loky_pickler()
    native, dump = pickler.reducer_override, pickler.dump
    observed = []

    class Table(dict):
        def __missing__(self, kind):
            if policy == "implicit" and kind is GridSearchCV:
                return _reduce_search_with_raw_row
            raise KeyError(kind)

    def configure(obj, value):
        obj._set_dispatch_table(Table(pickle.Pickler.dispatch_table.__get__(obj)))
        return dump(obj, value)

    def begin_check(obj, value):
        if value is captured:
            mapped.allow_raw = safe or policy in ("declared", "disjoint")
            mapped.live_mode = {"getter-now-private": "c", "getter-now-safe": "r+"}.get(policy)
            mapped.checking, mapped.mode_reads = True, 0
        return native(obj, value)

    def check_reconstruction(obj, value):
        if vars(mapped).get("checking"):
            observed.append((mapped.mode_reads, value is _strided_from_memmap))
            mapped.checking = False
        return None

    monkeypatch.setattr(pickler, "dump", configure)
    monkeypatch.setattr(pickler, "reducer_override", begin_check)
    monkeypatch.setattr(pickler, "persistent_id", check_reconstruction)
    reducers = {GridSearchCV: _reduce_search_with_raw_row} if policy == "declared" else {}
    with parallel_config(backend=LokyBackend(forward_reducers=reducers), inner_max_num_threads=1):
        if policy in ("implicit", "getter-now-private", "getter-now-safe"):
            with pytest.raises(PicklingError) as error:
                fit_shared_fold_search(search, rows[:, None], rows)
            expected = "owning indices" if policy == "implicit" else "native layout of an overlapping mapped view"
            assert expected in str(error.value.__cause__)
            assert not hasattr(search, "best_params_")
        else:
            fit_shared_fold_search(search, rows[:, None], rows)
            observed_value = {"declared": 7, "disjoint": 3}.get(policy, 0)
            winner = int(observed_value == 0)
            assert search.best_params_ == {"constant": winner}
            np.testing.assert_array_equal(search.predict(rows[:, None]), np.full(len(rows), winner))
            for metric, expected in (("captured", observed_value), ("public", public_start), ("same", 0)):
                np.testing.assert_array_equal(search.cv_results_["mean_test_" + metric], [expected, expected])
    if policy in ("getter-now-private", "getter-now-safe"):
        assert mapped.mode_reads == 1
        mapped.checking = False
    else:
        assert observed == [(1, mapped.allow_raw)] * (1 if policy == "implicit" else 4)
    assert score.captured is captured and search.cv is folds
    np.testing.assert_array_equal(mapped, storage)
    np.testing.assert_array_equal(np.fromfile(path, dtype=rows.dtype), disk)


@pytest.mark.parametrize("declared", [False, True])
def test_worker_dispatch_table_absent_search_keeps_detached_capture(tmp_path, declared):
    rows = np.arange(8)
    path = tmp_path / "indices.npy"
    np.save(path, rows[::-1])
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = rows
    first, second = np.asarray(mapped).reshape(2, 4)
    folds = [(first, second), (second, first)]
    score = _IndependentFoldScore()
    score.captured = score.again = first
    search = GridSearchCV(DummyRegressor(strategy="constant"), {"constant": [0, 1]}, cv=folds,
                          scoring=score, n_jobs=2, pre_dispatch=1, error_score="raise")
    visits = []

    def independent_jobs(batch):
        factory, (items, *args) = BatchedCalls.__reduce__(batch)
        jobs = []
        for fn, positional, original in items:
            keywords = original.copy()
            # Recent sklearn includes an optional callback caller in each job.
            keywords.pop("caller", None)
            jobs.append((fn, positional, keywords))
        return factory, (jobs, *args)

    def reduce_search(value):
        visits.append(value)
        return value.__reduce_ex__(pickle.HIGHEST_PROTOCOL)

    reducers = {BatchedCalls: independent_jobs}
    if declared:
        reducers[GridSearchCV] = reduce_search
    with parallel_config(backend=LokyBackend(forward_reducers=reducers), inner_max_num_threads=1):
        if declared:
            with pytest.raises(PicklingError) as error:
                fit_shared_fold_search(search, rows[:, None], rows)
            assert "owning indices" in str(error.value.__cause__)
            assert not hasattr(search, "best_params_")
        else:
            fit_shared_fold_search(search, rows[:, None], rows)
            np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [0., 1.])
            assert search.best_params_ == {"constant": 1}
    assert not visits and search.cv is folds and score.captured is first
    np.testing.assert_array_equal(mapped, rows)
    np.testing.assert_array_equal(np.load(path), rows[::-1])


class _MarkerSearch(GridSearchCV):
    def __reduce_ex__(self, protocol):
        return int, (1,)


class _MarkerScore:
    def __call__(self, estimator, X, y):
        assert os.getpid() != self.parent and self.markers == (2, 1, 2)
        return float(estimator.constant)


@pytest.mark.parametrize("fit,owning", [(GridSearchCV.fit, True), (fit_shared_fold_search, True),
                                        (fit_shared_fold_search, False)])
def test_worker_dispatch_table_revisits_search_without_fold_capture(tmp_path, monkeypatch, fit, owning):
    rows = np.arange(8)
    path = tmp_path / "indices.npy"
    np.save(path, rows[::-1])
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = rows
    first, second = np.asarray(mapped).reshape(2, 4)
    if owning:
        first, second = first.copy(), second.copy()
    folds = [(first, second), (second, first)]
    score = _MarkerScore()
    search = _MarkerSearch(DummyRegressor(strategy="constant"), {"constant": [0, 1]}, cv=folds,
                           scoring=score, n_jobs=2, pre_dispatch=1, error_score="raise")
    score.markers, score.parent = (search, search, search), os.getpid()
    pickler = get_loky_pickler()
    native = pickler.reducer_override
    visits = []

    def explicit(value):
        return int, (2,)

    def select(self, value):
        if type(value) is BatchedCalls:
            self.revisits = 0
            factory, (items, *args) = BatchedCalls.__reduce__(value)
            jobs = []
            for fn, positional, original in items:
                keywords = original.copy()
                keywords.pop("caller", None)
                jobs.append((fn, positional, keywords))
            return factory, (jobs, *args)
        if value is search:
            self.revisits += 1
            visits.append(self.revisits)
            table = pickle.Pickler.dispatch_table.__get__(self)
            if self.revisits == 2:
                del table[_MarkerSearch]
            elif self.revisits == 3:
                table[_MarkerSearch] = explicit
                self.fast = False
        return native(self, value)

    def begin_markers(self, value):
        if value is score.markers:
            self.fast = True
        return None

    monkeypatch.setattr(pickler, "reducer_override", select)
    monkeypatch.setattr(pickler, "persistent_id", begin_markers)
    backend = LokyBackend(forward_reducers={_MarkerSearch: explicit})
    with parallel_config(backend=backend, inner_max_num_threads=1, max_nbytes=None):
        fit(search, rows[:, None], rows)
    assert visits == [1, 2, 3] * 4
    np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [0., 1.])
    assert search.best_params_ == {"constant": 1}
    np.testing.assert_array_equal(search.predict(rows[:, None]), np.ones(8))
    assert search.cv is folds and all(marker is search for marker in score.markers)
    np.testing.assert_array_equal(mapped, rows)
    np.testing.assert_array_equal(np.load(path), rows[::-1])


@pytest.mark.parametrize("mode", ["absent-detached", "absent-declared", "stable", "changed-twice", "last-veto",
                                  "array-free", "unrelated", "safe-overlap", "applied-raw", "related-raw",
                                  "applied-detached", "settled", "early-unrelated", "late-after-unrelated"])
def test_worker_dispatch_table_checks_final_batch_policy(tmp_path, monkeypatch, mode):
    from joblib import Parallel

    class Search:
        def __reduce__(self):
            return int, (1,)

    def explicit(value):
        return int, (2,)

    rows = np.arange(4)
    path = tmp_path / "indices.npy"
    np.save(path, rows[::-1])
    mapped = np.load(path, mmap_mode="c")
    mapped[:] = rows
    row = np.asarray(mapped)
    search = Search()
    batch = (row,) if mode.startswith("absent") else (row, search, search, search)
    if mode == "array-free":
        batch = search, search, search
    elif mode == "unrelated":
        batch = search, np.array([11]), search, search
    elif mode == "safe-overlap":
        forward = np.asarray(np.load(path, mmap_mode="r"))
        reversed_row = forward[::-1]
        batch = search, forward, search, search
    elif mode == "applied-raw":
        batch = search, row, search, search
    elif mode == "related-raw":
        batch = search, row.view(), search, search
    elif mode == "applied-detached":
        batch = search, search, row, search
    elif mode == "settled":
        batch = search, search, search, row
    elif mode in ("early-unrelated", "late-after-unrelated"):
        batch = np.arange(3)[::2], search, search, search
    backend = LokyBackend(forward_reducers={} if mode in ("absent-detached", "late-after-unrelated")
                          else {Search: explicit})
    pickler = get_loky_pickler()
    dump, native = pickler.dump, pickler.reducer_override
    visits = []

    def dump_batch(self, value):
        if value is batch:
            self.fast = True  # Let the native pickler actually revisit the same search.
        return dump(self, value)

    def select(self, value):
        if value is search:
            visits.append(len(visits) + 1)
            if mode not in ("absent-detached", "absent-declared", "stable"):
                table = pickle.Pickler.dispatch_table.__get__(self)
                if mode == "late-after-unrelated" and len(visits) == 1:
                    table[Search] = explicit
                if len(visits) == 2:
                    del table[Search]
                elif len(visits) == 3:
                    if mode == "last-veto":
                        raise ValueError("last search rejected")
                    table[Search] = explicit
        return native(self, value)

    with Parallel(n_jobs=2, backend=backend) as parallel:
        monkeypatch.setattr(pickler, "dump", dump_batch)
        monkeypatch.setattr(pickler, "reducer_override", select)
        worker = dwd_search._WorkerSearchBatch(batch, parallel._backend, search, dict)
        worker.bind_originals = True
        worker.fold_arrays = dwd_search._FoldArrays()
        worker.fold_arrays.bind(row, row.copy())
        if mode == "safe-overlap":
            worker.fold_arrays.bind(reversed_row, reversed_row.copy())
        if mode in ("absent-declared", "changed-twice", "applied-raw", "related-raw", "applied-detached"):
            with pytest.raises(PicklingError, match="owning indices"):
                worker.encode(pickle.HIGHEST_PROTOCOL)
        elif mode == "last-veto":
            with pytest.raises(ValueError, match="last search rejected"):
                worker.encode(pickle.HIGHEST_PROTOCOL)
        else:
            loader, args = worker.encode(pickle.HIGHEST_PROTOCOL)
            restored = loader(*args)
            if mode == "array-free":
                assert restored == (2, 1, 2)
            elif mode in ("unrelated", "safe-overlap"):
                assert (restored[0], *restored[2:]) == (2, 1, 2)
                np.testing.assert_array_equal(restored[1], [11] if mode == "unrelated" else rows[::-1])
            elif mode == "settled":
                assert restored[:3] == (2, 1, 2)
                np.testing.assert_array_equal(restored[-1], rows[::-1])
            elif mode in ("early-unrelated", "late-after-unrelated"):
                np.testing.assert_array_equal(restored[0], [0, 2])
                assert restored[1:] == (2, 1, 2)
            else:
                np.testing.assert_array_equal(restored[0], rows if mode == "absent-detached" else rows[::-1])
                assert restored[1:] == (() if mode == "absent-detached" else (2, 2, 2))
    assert visits == ([] if mode.startswith("absent") else [1, 2, 3])
    np.testing.assert_array_equal(mapped, rows)
    np.testing.assert_array_equal(np.load(path), rows[::-1])


@pytest.mark.parametrize("policy", ["unset", "empty", "c-slot", "shadow", "lookup-key", "lookup-error",
                                    "reducer-key", "none", "hash-key-unset", "hash-key-explicit"])
def test_worker_dispatch_table_matches_native_fallback(worker_pickler, monkeypatch, policy):
    visits = []

    class Meta(type):
        def __hash__(cls):
            if policy.startswith("hash-key"):
                visits.append("hash")
                raise KeyError("hash failed")
            return type.__hash__(cls)

    class Probe(metaclass=Meta):
        def __reduce__(self):
            visits.append("instance")
            return str, ("instance",)

    def registered(value):
        visits.append("registered")
        if policy == "reducer-key":
            raise KeyError("reducer rejected the search")
        return str, ("registered",)

    class Table(dict):
        def __missing__(self, kind):
            if kind is Probe:
                visits.append("lookup")
                if policy == "lookup-error":
                    raise RuntimeError("table lookup failed")
            raise KeyError(kind)

    class Configured(CloudPickler):
        def dump(self, obj):
            slot = pickle.Pickler.dispatch_table
            if policy in ("unset", "hash-key-unset"):
                slot.__delete__(self)
            elif policy in ("empty", "hash-key-explicit"):
                slot.__set__(self, {})
            elif policy in ("lookup-key", "lookup-error"):
                slot.__set__(self, Table())
            elif policy == "none":
                slot.__set__(self, None)
            elif policy == "shadow":
                slot.__set__(self, {})
                self.dispatch_table = {Probe: registered}
            else:
                slot.__set__(self, {Probe: registered})
                self.dispatch_table = {}
            return super().dump(obj)

    base = worker_pickler(Configured)
    if not policy.startswith("hash-key"):
        monkeypatch.setitem(copyreg.dispatch_table, Probe, registered)
    search = Probe()
    batch = [search]
    results = []
    for adapted in (False, True):
        visits.clear()
        try:
            if adapted:
                encoder = dwd_search._WorkerSearchBatch(batch, LokyBackend(), search, dict)
                restore, args = encoder.encode(pickle.HIGHEST_PROTOCOL)
                value = restore(*args)
            else:
                stream = BytesIO()
                base(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(batch)
                value = pickle.loads(stream.getvalue())
        except (KeyError, RuntimeError, TypeError) as error:
            value = type(error), error.args
        results.append((value, tuple(visits)))
    assert results[0] == results[1]
    if policy == "hash-key-unset":
        assert results[0][0] == (KeyError, ("hash failed",))
    elif policy == "hash-key-explicit":
        assert results[0][0] == ["instance"]


_REDUCER_VISITS = []


def _copy_batch_items(batch):
    _REDUCER_VISITS.append("batch")
    factory, args = BatchedCalls.__reduce__(batch)
    return factory, (list(args[0]), *args[1:])


def _reduce_visible_job(native, value):
    if type(value) is BatchedCalls:
        return _copy_batch_items(value)
    if type(value) is _SavedFieldsSearch:
        _REDUCER_VISITS.append("search")
    return native(value)


class _DelegatingBatchPickler(CloudPickler):
    def reducer_override(self, value):
        return _reduce_visible_job(super().reducer_override, value)


class _AssignedBatchPickler(CloudPickler):
    def __init__(self, file, protocol=None):
        super().__init__(file, protocol=protocol)
        self.reducer_override = partial(_reduce_visible_job, super().reducer_override)


class _DecoratedBatchPickler(CloudPickler):
    def __init__(self, file, protocol=None):
        super().__init__(file, protocol=protocol)
        self.reducer_override = partial(_reduce_visible_job, self.reducer_override)


class _ReducerProperty(CloudPickler):
    @property
    def reducer_override(self):
        if "_configured_reducer" in vars(self):
            return self._configured_reducer
        return super().reducer_override


class _ReadOnlyBatchPickler(_ReducerProperty):
    def __init__(self, file, protocol=None):
        super().__init__(file, protocol=protocol)
        self._configured_reducer = partial(_reduce_visible_job, self.reducer_override)


class _WritableBatchPickler(_ReducerProperty):
    def __init__(self, file, protocol=None):
        super().__init__(file, protocol=protocol)
        self.reducer_override = partial(_reduce_visible_job, self.reducer_override)

    @_ReducerProperty.reducer_override.setter
    def reducer_override(self, hook):
        assert "_configured_reducer" not in vars(self), "reducer was configured twice"
        self._configured_reducer = hook


class _SetReducer:
    def __set__(self, pickler, hook):
        assert callable(hook) and "reducer_override" not in vars(pickler)
        vars(pickler)["reducer_override"] = hook


class _SetOnlyBatchPickler(_AssignedBatchPickler):
    reducer_override = _SetReducer()


class _DumpAssignedBatchPickler(CloudPickler):
    def dump(self, obj):
        self.reducer_override = partial(_reduce_visible_job, super().reducer_override)
        return super().dump(obj)


class _DumpDecoratedBatchPickler(CloudPickler):
    def dump(self, obj):
        self.reducer_override = partial(_reduce_visible_job, self.reducer_override)
        return super().dump(obj)


def _reject_late_search(native, value):
    if type(value) is _SavedFieldsSearch:
        raise ValueError("late-native-search-veto")
    return native(value)


class _DumpPropertyVetoPickler(_ReducerProperty):
    def dump(self, obj):
        self._configured_reducer = partial(_reject_late_search, super().reducer_override)
        return super().dump(obj)


class _ConditionalDumpVetoPickler(CloudPickler):
    def dump(self, obj):
        if self.reducer_override == super().reducer_override:
            self.reducer_override = partial(_reject_late_search, super().reducer_override)
        return super().dump(obj)


class _DirectCDumpPickler(CloudPickler):
    def dump(self, obj):
        return pickle.Pickler.dump(self, obj)


class _DirectBatchWithTrailerPickler(CloudPickler):
    def dump(self, obj):
        if type(obj) is not BatchedCalls:
            return super().dump(obj)
        pickle.Pickler.dump(self, obj)
        return super().dump(None)


class _CooperativeBatchWithTrailerPickler(CloudPickler):
    def dump(self, obj):
        result = super().dump(obj)
        if type(obj) is BatchedCalls:
            super().dump(None)
        return result


class _RetryBatch(Exception):
    pass


class _RetryBatchWithDirectDumpPickler(CloudPickler):
    def __init__(self, file, protocol=None):
        super().__init__(file, protocol=protocol)
        self.stream = file
        self.first_attempt = False

    def reducer_override(self, value):
        if self.first_attempt and type(value) is _SavedFieldsSearch:
            raise _RetryBatch
        return super().reducer_override(value)

    def dump(self, obj):
        if type(obj) is not BatchedCalls:
            return super().dump(obj)
        start = self.stream.tell()
        self.first_attempt = True
        try:
            super().dump(obj)
        except _RetryBatch:
            self.stream.seek(start)
            self.stream.truncate()
            self.clear_memo()
        else:
            raise AssertionError("expected the native search reducer to request a retry")
        finally:
            self.first_attempt = False
        return pickle.Pickler.dump(self, obj)


class _ObserveSubmit:
    def submit(self, batch, callback=None):
        self.submitted += 1
        assert type(batch) is BatchedCalls and type(batch.items) is list
        for function, args, kwargs in batch.items:
            assert function.__name__ == "_fit_and_score" and args
            assert kwargs["scorer"] is self.expected_scorer
            assert kwargs["parameters"]["constant"] in (0, 1)
        return super().submit(batch, callback=callback)


class _ObservedLoky(_ObserveSubmit, LokyBackend):
    pass


class _ObservedPool(_ObserveSubmit, MultiprocessingBackend):
    pass


def test_forwarding_adapter_restores_overlapping_dispatches_and_errors():
    original = SimpleNamespace(submit=lambda fn, *args, **kwargs: fn)
    backend = LokyBackend()
    backend._workers = original
    parallel = SimpleNamespace(_aborting=False)
    barrier = Barrier(2, timeout=10)

    def run(marker):
        batch = object()
        worker = dwd_search._WorkerSearchBatch(batch, backend, None, None)

        def dispatch(job):
            assert job is batch
            barrier.wait()
            assert backend._workers.submit(marker, func="task argument") is marker
            assert original.submit(job) is job
            assert backend._workers.submit(fn=job) is worker
            barrier.wait()
            return marker

        return dwd_search._dispatch_native_batch(parallel, dispatch, worker)

    markers = [object(), object()]
    with ThreadPoolExecutor(2) as tasks:
        assert list(tasks.map(run, markers)) == markers
    assert backend._workers is original

    def fail(job):
        raise ValueError("submission refused")

    worker = dwd_search._WorkerSearchBatch(object(), backend, None, None)
    with pytest.raises(ValueError, match="submission refused"):
        dwd_search._dispatch_native_batch(parallel, fail, worker)
    assert backend._workers is original
    with pytest.raises(RuntimeError, match="did not forward"):
        dwd_search._dispatch_native_batch(parallel, lambda job: None, worker)
    assert backend._workers is original
    parallel._aborting = True
    assert dwd_search._dispatch_native_batch(parallel, lambda job: None, worker) is None
    assert backend._workers is original


@pytest.mark.parametrize("pickler", [
    pytest.param(_DelegatingBatchPickler, id="loky"),
    pytest.param(_AssignedBatchPickler, id="loky-assigned"),
    pytest.param(_DecoratedBatchPickler, id="loky-decorated"),
    pytest.param(_ReadOnlyBatchPickler, id="loky-read-only"),
    pytest.param(_WritableBatchPickler, id="loky-writable"),
    pytest.param(_SetOnlyBatchPickler, id="loky-set-only"),
    pytest.param(_DumpAssignedBatchPickler, id="loky-dump-assigned"),
    pytest.param(_DumpDecoratedBatchPickler, id="loky-dump-decorated"),
    pytest.param(None, id="multiprocessing"),
])
def test_custom_batch_reducer_preserves_visible_jobs_and_worker_folds(tmp_path, monkeypatch, pickler):
    X = np.arange(8)[:, None]
    path = tmp_path / "indices.npy"
    np.save(path, X[::-1, 0])
    mapped = np.load(path, mmap_mode="c")
    previous = get_loky_pickler_name()
    visits = []
    monkeypatch.setitem(globals(), "_REDUCER_VISITS", visits)
    try:
        mapped[:] = X[:, 0]
        if pickler is not None:
            monkeypatch.setitem(globals(), "Pickler", pickler)
            set_loky_pickler(__name__)
        for fit in (GridSearchCV.fit, fit_shared_fold_search):
            visits.clear()
            rows = np.asarray(mapped).reshape(2, 4)
            first, second = rows.copy() if fit is GridSearchCV.fit else rows
            cv = [(first, second), (second, first)]
            search = _SavedFieldsSearch(DummyRegressor(strategy="constant"), {"constant": [0, 1]},
                                        cv=cv, n_jobs=2, pre_dispatch=1, error_score="raise")
            search.parent_pid = os.getpid()
            search.scoring = search.score_first_fold
            if pickler is not None:
                backend = _ObservedLoky()
                options = {"inner_max_num_threads": 1}
            else:
                backend = _ObservedPool(forward_reducers={BatchedCalls: _copy_batch_items})
                options = {}
            backend.expected_scorer = search.scoring
            backend.submitted = 0
            with parallel_config(backend=backend, **options):
                fit(search, X, X[:, 0])
            assert visits.count("batch") == backend.submitted > 0
            if pickler is not None:
                assert visits.count("search") == backend.submitted
            assert search.best_params_ == {"constant": 1}
            np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [0., 1.])
            np.testing.assert_array_equal(search.predict(X), np.ones(len(X)))
            assert search.cv is cv
        np.testing.assert_array_equal(mapped, X[:, 0])
        np.testing.assert_array_equal(np.load(path), X[::-1, 0])
    finally:
        if pickler is not None:
            get_reusable_executor().shutdown(wait=True)
            set_loky_pickler(previous)
        mapped._mmap.close()


@pytest.mark.parametrize("search_class, pickler, fit, message", [
    pytest.param(GridSearchCV, Pickler, fit_shared_fold_search,
                 "reject fold scorer marker", id="GridSearchCV"),
    pytest.param(_SavedFieldsSearch, Pickler, fit_shared_fold_search,
                 "reject fold scorer marker", id="_SavedFieldsSearch"),
    pytest.param(_SavedFieldsSearch, _DumpPropertyVetoPickler, GridSearchCV.fit,
                 "late-native-search-veto", id="late-stock"),
    pytest.param(_SavedFieldsSearch, _DumpPropertyVetoPickler, fit_shared_fold_search,
                 "late-native-search-veto", id="late-helper"),
    pytest.param(_SavedFieldsSearch, _ConditionalDumpVetoPickler, GridSearchCV.fit,
                 "late-native-search-veto", id="conditional-stock"),
    pytest.param(_SavedFieldsSearch, _ConditionalDumpVetoPickler, fit_shared_fold_search,
                 "late-native-search-veto", id="conditional-helper"),
    pytest.param(_SavedFieldsSearch, _DumpPersistentPickler, GridSearchCV.fit,
                 "late persistent ID rejected scorer", id="dump-persistent-stock"),
    pytest.param(_SavedFieldsSearch, _DumpPersistentPickler, fit_shared_fold_search,
                 "late persistent ID rejected scorer", id="dump-persistent-helper"),
    pytest.param(_SavedFieldsSearch, _SwitchPersistentPickler, GridSearchCV.fit,
                 "late persistent ID rejected scorer", id="switch-persistent-stock"),
    pytest.param(_SavedFieldsSearch, _SwitchPersistentPickler, fit_shared_fold_search,
                 "late persistent ID rejected scorer", id="switch-persistent-helper"),
])
def test_custom_pickler_can_reject_scorer_with_mapped_folds(tmp_path, monkeypatch, search_class, pickler, fit, message):
    rows = np.arange(16)
    np.save(tmp_path / "indices.npy", rows[::-1])
    mapped = np.load(tmp_path / "indices.npy", mmap_mode="c")
    previous = get_loky_pickler_name()
    try:
        mapped[:] = rows
        a, b = np.asarray(mapped[:8]), np.asarray(mapped[8:])
        if fit is GridSearchCV.fit:
            a, b = a.copy(), b.copy()
        folds = [(a, b), (b, a)]
        search = search_class(DummyRegressor(), {}, cv=folds, scoring=_MarkedScore(),
                              n_jobs=2, pre_dispatch=1, error_score="raise")
        if search_class is _SavedFieldsSearch:
            search.scoring.search = search
        if pickler is _SwitchPersistentPickler:
            search.scoring.marker = (_SWITCH_MARKER, _VETO_MARKER)
        monkeypatch.setitem(globals(), "Pickler", pickler)
        set_loky_pickler(__name__)
        with parallel_config(backend="loky", inner_max_num_threads=1, max_nbytes=8):
            if _uses_selected_policy(pickler):
                with pytest.raises((ValueError, PicklingError)) as error:
                    fit(search, rows[:, None], rows)
                assert str(error.value) == message or f"ValueError: {message}" in str(error.value.__cause__).splitlines()
            else:
                fit(search, rows[:, None], rows)
                np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [1.])
        assert search.cv is folds and "__getstate__" not in vars(search)
    finally:
        set_loky_pickler(previous)
        mapped._mmap.close()


@pytest.mark.parametrize("fit", [GridSearchCV.fit, fit_shared_fold_search], ids=["stock", "helper"])
def test_builtin_loky_dump_preserves_search_veto(tmp_path, monkeypatch, fit):
    X = np.arange(8)[:, None]
    path = tmp_path / "indices.npy"
    np.save(path, X[::-1, 0])
    mapped = np.load(path, mmap_mode="c")
    previous = get_loky_pickler_name()
    try:
        mapped[:] = X[:, 0]
        a, b = np.asarray(mapped).reshape(2, 4)
        if fit is GridSearchCV.fit:
            a, b = a.copy(), b.copy()
        folds = [(a, b), (b, a)]
        search = _SavedFieldsSearch(DummyRegressor(strategy="constant"), {"constant": [0, 1]},
                                    cv=folds, n_jobs=2, pre_dispatch=1, error_score="raise")
        search.parent_pid = os.getpid()
        search.scoring = search.score_first_fold
        set_loky_pickler("cloudpickle")
        pickler = get_loky_pickler()
        original_dump = pickler.dump
        rejected = []
        message = "built-in loky pickler rejected the worker search"

        def observed_dump(self, obj):
            native = self.reducer_override

            def reject(value):
                if value is search:
                    rejected.append(True)
                    raise ValueError(message)
                return native(value)

            self.reducer_override = reject
            try:
                return original_dump(self, obj)
            finally:
                del self.reducer_override

        monkeypatch.setattr(pickler, "dump", observed_dump)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            with pytest.raises((ValueError, PicklingError)) as error:
                fit(search, X, X[:, 0])
        assert rejected
        if isinstance(error.value, ValueError):
            assert str(error.value) == message
        else:
            assert f"ValueError: {message}" in str(error.value.__cause__).splitlines()
        assert search.cv is folds and "__getstate__" not in vars(search)
        np.testing.assert_array_equal(mapped, X[:, 0])
        np.testing.assert_array_equal(np.load(path), X[::-1, 0])
    finally:
        get_reusable_executor().shutdown(wait=True)
        set_loky_pickler(previous)
        mapped._mmap.close()


@pytest.mark.parametrize("pickler, accepts_mapped", [
    pytest.param(_DirectCDumpPickler, False, id="direct"),
    pytest.param(_DirectBatchWithTrailerPickler, False, id="direct-trailer"),
    pytest.param(_RetryBatchWithDirectDumpPickler, False, id="direct-retry"),
    pytest.param(_CooperativeBatchWithTrailerPickler, True, id="cooperative-trailer"),
])
@pytest.mark.parametrize("fit, owning", [
    pytest.param(GridSearchCV.fit, True, id="stock-owning"),
    pytest.param(fit_shared_fold_search, False, id="helper-mapped"),
    pytest.param(fit_shared_fold_search, True, id="helper-owning"),
])
def test_direct_c_dump_requires_owning_indices(tmp_path, monkeypatch, fit, owning, pickler, accepts_mapped):
    X = np.arange(8)[:, None]
    path = tmp_path / "indices.npy"
    np.save(path, X[::-1, 0])
    mapped = np.load(path, mmap_mode="c")
    previous = get_loky_pickler_name()
    try:
        mapped[:] = X[:, 0]
        a, b = np.asarray(mapped).reshape(2, 4)
        if owning:
            a, b = a.copy(), b.copy()
        folds = [(a, b), (b, a)]
        search = _SavedFieldsSearch(DummyRegressor(strategy="constant"), {"constant": [0, 1]},
                                    cv=folds, n_jobs=2, pre_dispatch=1, error_score="raise")
        search.parent_pid = os.getpid()
        search.scoring = search.score_first_fold
        monkeypatch.setitem(globals(), "Pickler", pickler)
        set_loky_pickler(__name__)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            if owning or accepts_mapped:
                fit(search, X, X[:, 0])
                assert search.best_params_ == {"constant": 1}
                np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [0., 1.])
                np.testing.assert_array_equal(search.predict(X), np.ones(len(X)))
            else:
                with pytest.raises((PicklingError, RuntimeError, TypeError, ValueError)) as error:
                    fit(search, X, X[:, 0])
                assert re.search(r"owning.*indices", f"{error.value}\n{error.value.__cause__}", re.IGNORECASE)
        assert search.cv is folds and "__getstate__" not in vars(search)
        assert folds[0][0] is a and folds[0][1] is b and folds[1][0] is b and folds[1][1] is a
        np.testing.assert_array_equal(mapped, X[:, 0])
        np.testing.assert_array_equal(np.load(path), X[::-1, 0])
    finally:
        get_reusable_executor().shutdown(wait=True)
        set_loky_pickler(previous)
        mapped._mmap.close()


@pytest.mark.parametrize("pickler", [Pickler, _InstanceTokenPickler, _DumpPersistentPickler, _SwitchPersistentPickler],
                         ids=["method", "instance", "dump-persistent", "switch-persistent"])
@pytest.mark.parametrize("fit", [GridSearchCV.fit, fit_shared_fold_search], ids=["stock", "helper"])
def test_unresolved_custom_pickler_token_rejects_search(tmp_path, monkeypatch, pickler, fit):
    X = np.arange(8)[:, None]
    path = tmp_path / "indices.npy"
    np.save(path, X[::-1, 0])
    mapped = np.load(path, mmap_mode="c")
    previous = get_loky_pickler_name()
    try:
        mapped[:] = X[:, 0]
        first, second = np.asarray(mapped).reshape(2, 4)
        if fit is GridSearchCV.fit:
            first, second = first.copy(), second.copy()
        cv = [(first, second), (second, first)]
        search = _SavedFieldsSearch(DummyRegressor(strategy="constant"), {"constant": [0, 1]},
                                    cv=cv, n_jobs=2, pre_dispatch=1, error_score="raise")
        search.scoring = search.score_token
        if pickler is _SwitchPersistentPickler:
            search.switch_marker = _SWITCH_MARKER
        search.marker = _TOKEN_MARKER
        # The private unpickler also has to resolve its own binding to X.
        search.dataset = X
        monkeypatch.setitem(globals(), "Pickler", pickler)
        set_loky_pickler(__name__)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            if _uses_selected_policy(pickler):
                with pytest.raises((BrokenProcessPool, PicklingError)) as error:
                    fit(search, X, X[:, 0])
                assert "persistent" in f"{error.value} {error.value.__cause__}".lower()
            else:
                fit(search, X, X[:, 0])
                assert search.best_params_ == {"constant": 0}
                np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [1., 0.])
        assert search.cv is cv and search.marker == _TOKEN_MARKER
    finally:
        set_loky_pickler(previous)
        mapped._mmap.close()


@pytest.mark.parametrize("policy", ["veto", "token"])
@pytest.mark.parametrize("fit, search_class", [
    pytest.param(GridSearchCV.fit, GridSearchCV, id="stock"),
    pytest.param(fit_shared_fold_search, GridSearchCV, id="helper-default"),
    pytest.param(fit_shared_fold_search, _SavedFieldsSearch, id="helper-subclass"),
])
def test_builtin_loky_dump_preserves_persistent_id(tmp_path, monkeypatch, policy, fit, search_class):
    X = np.arange(8)[:, None]
    path = tmp_path / "indices.npy"
    np.save(path, X[::-1, 0])
    mapped = np.load(path, mmap_mode="c")
    previous = get_loky_pickler_name()
    try:
        mapped[:] = X[:, 0]
        a, b = np.asarray(mapped).reshape(2, 4)
        if fit is GridSearchCV.fit:
            a, b = a.copy(), b.copy()
        folds = [(a, b), (b, a)]
        search = search_class(DummyRegressor(), {}, cv=folds, n_jobs=2, pre_dispatch=1, error_score="raise")
        search.scoring = _MarkedScore()
        search.scoring.search = search
        set_loky_pickler("cloudpickle")
        pickler = get_loky_pickler()
        original_dump = pickler.dump
        visits = []
        worker_state = []
        message = "built-in loky persistent ID rejected the scorer"

        def observed_dump(self, obj):
            state = vars(self).get("_dwd_state")
            if type(obj) is BatchedCalls and state is not None and state.search is search:
                worker_state.append(state.replacements is not None)

            def persistent_id(value):
                if value is search.scoring:
                    visits.append(True)
                    if policy == "veto":
                        raise ValueError(message)
                    return ("scorer-token",)
                return None

            self.persistent_id = persistent_id
            return original_dump(self, obj)

        monkeypatch.setattr(pickler, "dump", observed_dump)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            if policy == "veto":
                with pytest.raises((ValueError, PicklingError)) as error:
                    fit(search, X, X[:, 0])
                assert str(error.value) == message or f"ValueError: {message}" in str(error.value.__cause__).splitlines()
            else:
                with pytest.raises((BrokenProcessPool, PicklingError)) as error:
                    fit(search, X, X[:, 0])
                assert "persistent" in f"{error.value} {error.value.__cause__}".lower()
        assert visits
        # The native queue sends the stock jobs directly. DWD's ordinary search
        # uses the fold-array envelope and the subclass also adapts search state.
        assert set(worker_state) == (set() if fit is GridSearchCV.fit else {search_class is _SavedFieldsSearch})
        assert search.cv is folds and "__getstate__" not in vars(search)
        np.testing.assert_array_equal(mapped, X[:, 0])
        np.testing.assert_array_equal(np.load(path), X[::-1, 0])
    finally:
        get_reusable_executor().shutdown(wait=True)
        set_loky_pickler(previous)
        mapped._mmap.close()


@pytest.mark.parametrize("form", ["stack", "pairs"])
@pytest.mark.parametrize("step", [1, 2], ids=["whole_row", "every_other_index"])
@pytest.mark.parametrize("threshold", [None, 8])
def test_priority_weights_use_separately_sliced_fold(tmp_path, form, step, threshold):
    rows = np.arange(16)
    data = np.array([[rows[:8], rows[8:]], [rows[8:], rows[:8]]])
    path = tmp_path / "priority.npy"
    np.save(path, 15 - data)
    owner = np.load(path, mmap_mode="c")
    try:
        owner[:] = data
        stack = np.asarray(owner)
        folds = stack if form == "stack" else list(stack)
        priority = folds[0][0][::step]
        assert priority is not folds[0][0] and np.shares_memory(priority, folds[0][0])
        expected = [-.5, -1.5 if step == 1 else -1.]
        for jobs in (1, 2):
            search = GridSearchCV(DummyRegressor(strategy="constant"), {"constant": [0, 1]},
                                  cv=folds, scoring=_PriorityWeightedError(priority), n_jobs=jobs,
                                  pre_dispatch=1, error_score="raise")
            with parallel_config(backend="loky", inner_max_num_threads=1, max_nbytes=threshold):
                if jobs == 1:
                    search.fit(rows[:, None], rows >= 8)
                else:
                    fit_shared_fold_search(search, rows[:, None], rows >= 8)
            np.testing.assert_array_equal(search.cv_results_["mean_test_score"], expected)
            assert search.cv is folds and search.best_estimator_.constant == 0
            np.testing.assert_array_equal(search.predict(rows[:, None]), np.zeros(len(rows)))
        np.testing.assert_array_equal(owner, data)
        np.testing.assert_array_equal(np.load(path), 15 - data)
    finally:
        owner._mmap.close()


class _IndependentIndexScore:
    def __init__(self, priority):
        self.priority = priority

    def __call__(self, model, X, y):
        return {"row_sum": float(np.sum(self.priority)), "mapped": float(isinstance(self.priority, np.memmap))}


@pytest.mark.parametrize("relation", ["disjoint", "interleaved_disjoint", "independent_mapping"])
def test_independent_capture_keeps_native_transport(tmp_path, relation):
    rows = np.arange(16)
    data = np.array([[rows[:8], rows[8:]], [rows[8:], rows[:8]]])
    # Adjacent interleaved columns share a byte range but no elements.
    disk = np.stack((15 - data, 15 - data), axis=-1)
    path = tmp_path / "indices.npy"
    np.save(path, disk)
    owner = np.load(path, mmap_mode="c")
    second = np.load(path, mmap_mode="c")
    try:
        changed = np.stack((data, data), axis=-1)
        owner[:] = changed
        second[:] = changed + 16
        if relation == "interleaved_disjoint":
            folds = np.asarray(owner)[..., 0]
            priority = np.asarray(owner)[0, 0, :, 1]
        elif relation == "disjoint":
            folds = np.asarray(owner).reshape(4, 2, 8)[:2]
            priority = np.asarray(owner).reshape(4, 2, 8)[2, 0]
        else:
            folds = np.asarray(owner)[..., 0]
            priority = np.asarray(second)[0, 0, :, 0]
        assert not np.shares_memory(priority, folds)
        scores = []
        for native in (True, False):
            # Direct joblib transport controls the scorer's unrelated mapping.
            cv = np.array(folds, copy=True) if native else folds
            search = GridSearchCV(DummyRegressor(), {}, scoring=_IndependentIndexScore(priority),
                                  cv=cv, n_jobs=2, refit=False, pre_dispatch=1, error_score="raise")
            with parallel_config(backend="loky", inner_max_num_threads=1, max_nbytes=8):
                if native:
                    search.fit(rows[:, None], rows >= 8)
                else:
                    fit_shared_fold_search(search, rows[:, None], rows >= 8)
            scores.append([(search.cv_results_[f"split{i}_test_row_sum"][0],
                            search.cv_results_[f"split{i}_test_mapped"][0]) for i in (0, 1)])
        assert scores[0] == scores[1]
        np.testing.assert_array_equal(owner, changed)
        np.testing.assert_array_equal(second, changed + 16)
        np.testing.assert_array_equal(np.load(path), disk)
    finally:
        second._mmap.close()
        owner._mmap.close()


class _ScorerWithIndex:
    def __init__(self, index):
        self.index = index

    def __call__(self, estimator, X, y):
        return float(len(self.index))


@pytest.mark.parametrize("hook", ["submit_observer", "array_reducer"])
@pytest.mark.parametrize("search_class", [GridSearchCV, _SavedFieldsSearch])
def test_multiprocessing_preserves_caller_transport(hook, tmp_path, monkeypatch, search_class):
    rows = np.arange(16)
    path = tmp_path / "caller.npy"
    np.save(path, rows[::-1])
    raw = np.load(path, mmap_mode="c")
    try:
        raw[:] = rows
        a, b = np.asarray(raw[:8]), np.asarray(raw[8:])
        folds = [(a, b), (b, a)]
        reducers = {}
        backend = MultiprocessingBackend(forward_reducers=reducers)
        batches = []
        if hook == "submit_observer":
            submit = backend.submit

            def observe(batch, callback=None):
                assert type(batch) is BatchedCalls
                for function, args, kwargs in batch.items:
                    assert function.__name__ == "_fit_and_score" and args
                    assert kwargs["scorer"] is search.scoring
                    if "caller" in kwargs:
                        assert kwargs["caller"] is search
                batches.append(len(batch.items))
                return submit(batch, callback=callback)

            monkeypatch.setattr(backend, "submit", observe)
        else:
            def configure(search, evaluate):
                native = reducers[np.ndarray]

                def reject_index(value):
                    if value is a:
                        raise ValueError("caller array reducer rejected the captured index")
                    return native(value)

                reducers[np.ndarray] = reject_index
                return evaluate([{}])

            monkeypatch.setattr(GridSearchCV, "_run_search", configure)
        search = search_class(DummyRegressor(), {}, n_jobs=2, pre_dispatch=1,
                              cv=folds, scoring=_ScorerWithIndex(a), error_score="raise")
        if search_class is _SavedFieldsSearch:
            search.scoring.search = search
        with parallel_config(backend=backend, max_nbytes=8):
            if hook == "array_reducer":
                with pytest.raises(ValueError, match="caller array reducer rejected"):
                    fit_shared_fold_search(search, rows[:, None], rows)
            else:
                fit_shared_fold_search(search, rows[:, None], rows)
                assert batches and sum(batches) == 2
                np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [8.])
        assert search.cv is folds
        np.testing.assert_array_equal(raw, rows)
        np.testing.assert_array_equal(np.load(path), rows[::-1])
    finally:
        raw._mmap.close()


def _backing_map(value):
    while value is not None and not isinstance(value, np.memmap):
        value = getattr(value, "base", None)
    return value


class _TemporaryFoldScore:
    def __init__(self, row, control, observations, writable, row_modes, control_mode):
        self.row = self.alias = row
        self.control = control
        self.observations = observations
        self.writable = writable
        self.row_modes = row_modes
        self.control_mode = control_mode
        self.parent = os.getpid()

    def __call__(self, model, X, y):
        assert os.getpid() != self.parent
        assert type(self.row) is np.ndarray and self.alias is self.row
        assert self.row.flags.writeable == self.writable and self.row[0] == 0
        row_map = _backing_map(self.row)
        control_map = _backing_map(self.control)
        assert getattr(row_map, "mode", None) in self.row_modes
        assert getattr(control_map, "mode", None) == self.control_mode
        if self.writable:
            self.row[0] = 27
            assert self.alias[0] == 27
        else:
            with pytest.raises(ValueError):
                self.row[0] = 27
        files = [str(m.filename) for m in (row_map, control_map) if m is not None]
        assert all(Path(name).is_file() for name in files)
        path = self.observations / f"{model.constant}-{X[0, 0]}.json"
        path.write_text(json.dumps(files), encoding="utf8")
        return 1.


@pytest.mark.parametrize("backend", ["loky", "multiprocessing"])
@pytest.mark.parametrize("mode,threshold,writable,recognized_loader", [
    ("r", 8, True, True), ("r+", 8, True, True), ("w+", 8, True, True), ("c", 8, True, True),
    ("r", 8, False, True), ("r+", 8, False, True), ("r", None, True, True),
    ("r", 8, True, False), ("r+", 8, True, False),
])
def test_private_fold_mapping_and_cleanup(tmp_path, backend, mode, threshold, writable,
                                         recognized_loader, monkeypatch):
    if not recognized_loader:
        monkeypatch.setattr(dwd_search, "_JOBLIB_TEMP_LOADER", None)
    rows = np.arange(64)
    path = tmp_path / "source.npy"
    np.save(path, rows[::-1])
    mapped = np.load(path, mmap_mode="c")
    try:
        mapped[:] = rows
        a, b = np.asarray(mapped[:32]), np.asarray(mapped[32:])
        a.flags.writeable = writable
        folds = [(a, b), (b, a)]
        control = np.arange(41)
        observations = tmp_path / "observed"
        observations.mkdir()
        native_mode = None if threshold is None else {"w+": "r+"}.get(mode, mode)
        # Both an owning copy and a private mapping isolate writes. Joblib
        # can change its reconstruction protocol independently of DWD.
        row_modes = (None, "c") if writable else (None, native_mode)
        scorer = _TemporaryFoldScore(a, control, observations, writable, row_modes, native_mode)
        search = GridSearchCV(DummyRegressor(strategy="constant"), {"constant": [0, 1]},
                              cv=folds, scoring=scorer, error_score="raise", n_jobs=2, pre_dispatch=1)
        options = {"inner_max_num_threads": 1} if backend == "loky" else {}
        # Joblib appends generated names. Keep its folder independent of the checkout path.
        with TemporaryDirectory() as jobtemp:
            with parallel_config(backend=backend, temp_folder=jobtemp, max_nbytes=threshold,
                                 mmap_mode=mode, **options):
                fit_shared_fold_search(search, rows[:, None], rows)
            assert search.cv is folds and np.all(search.cv_results_["mean_test_score"] == 1.)
            reports = list(observations.glob("*.json"))
            assert len(reports) == 4
            generated = {Path(name) for report in reports for name in json.loads(report.read_text(encoding="utf8"))}
            assert bool(generated) == (threshold is not None)
            assert all(name != path and not name.exists() for name in generated)
        np.testing.assert_array_equal(mapped, rows)
        np.testing.assert_array_equal(np.load(path), rows[::-1])
        np.testing.assert_array_equal(control, np.arange(41))
    finally:
        mapped._mmap.close()




class _ReplacePublicCv(GridSearchCV):
    def _run_search(self, evaluate_candidates):
        # sklearn has already checked the original folds for this evaluation.
        self.cv = 2
        evaluate_candidates([{"constant": value} for value in self.param_grid["constant"]])


@pytest.mark.parametrize("backend", ["loky", "multiprocessing"])
def test_captured_checked_indices_survive_public_cv_replacement(tmp_path, backend):
    rows = np.arange(16)
    path = tmp_path / "checked.npy"
    np.save(path, rows[::-1])
    mapped = np.load(path, mmap_mode="c")
    try:
        mapped[:] = rows
        first, second = np.asarray(mapped[:8]), np.asarray(mapped[8:])
        search = _ReplacePublicCv(DummyRegressor(strategy="constant"), {"constant": [0, 1]},
                                  scoring=_PriorityWeightedError(first), cv=[(first, second), (second, first)],
                                  n_jobs=2, pre_dispatch=1, error_score="raise")
        options = {"inner_max_num_threads": 1} if backend == "loky" else {}
        with parallel_config(backend=backend, max_nbytes=8, **options):
            fit_shared_fold_search(search, rows[:, None], rows >= 8)
        np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [-.5, -1.5])
        assert search.best_estimator_.constant == 0 and search.cv == 2
        np.testing.assert_array_equal(mapped, rows)
        np.testing.assert_array_equal(np.load(path), rows[::-1])
    finally:
        mapped._mmap.close()


class _LargePriorityWeightedError(_PriorityWeightedError):
    def __init__(self, priority_rows):
        super().__init__(priority_rows)
        self.alias = priority_rows
        self.dtype_string = priority_rows.dtype.str
        self.writeable = priority_rows.flags.writeable
        self.parent = os.getpid()

    def __call__(self, model, X, y):
        priority = self.priority_rows
        assert os.getpid() != self.parent
        assert priority is self.alias and type(priority) is np.ndarray
        assert _backing_map(priority) is None
        assert priority.dtype.str == self.dtype_string and priority.flags.writeable == self.writeable
        np.testing.assert_array_equal(priority[:8], np.arange(8))
        assert np.all(priority[8:] == -1)
        if self.writeable:
            priority[-1] = -3
            assert self.alias[-1] == -3
            priority[-1] = -1
        else:
            with pytest.raises(ValueError):
                priority[-1] = -3
        return super().__call__(model, X, y)


@pytest.mark.parametrize("backend", ["loky", "multiprocessing"])
@pytest.mark.parametrize("writeable", [False, True], ids=["read_only", "writable"])
def test_large_overlapping_fold_capture_with_disabled_mapping(tmp_path, backend, writeable):
    rows = np.arange(16)
    X, y = rows[:, None], rows >= 8
    dtype = np.dtype(np.int32).newbyteorder("S")
    disk = np.full(520, -2, dtype=dtype)
    disk[:8], disk[-8:] = rows[8:], rows[:8]
    expected = np.full_like(disk, -1)
    expected[:8], expected[-8:] = rows[:8], rows[8:]
    path = tmp_path / "priority.npy"
    np.save(path, disk)
    mapped = np.load(path, mmap_mode="c")
    try:
        mapped[:] = expected
        mapped.flags.writeable = writeable
        first, second = np.asarray(mapped[:8]), np.asarray(mapped[-8:])
        priority = np.asarray(mapped[:512])
        folds = [(first, second), (second, first)]
        threshold = 512
        assert np.shares_memory(first, priority)
        assert max(X.nbytes, y.nbytes, first.nbytes, second.nbytes) < threshold < priority.nbytes
        search = GridSearchCV(DummyRegressor(strategy="constant"), {"constant": [0, 1]},
                              cv=folds, scoring=_LargePriorityWeightedError(priority), n_jobs=2,
                              pre_dispatch=1, error_score="raise")
        options = {"inner_max_num_threads": 1} if backend == "loky" else {}
        with parallel_config(backend=backend, mmap_mode=None, max_nbytes=threshold, **options):
            fit_shared_fold_search(search, X, y)
        results = search.cv_results_
        np.testing.assert_array_equal(results["split0_test_score"], [-1., 0.])
        np.testing.assert_array_equal(results["split1_test_score"], [0., -3.])
        np.testing.assert_array_equal(results["mean_test_score"], [-.5, -1.5])
        assert search.best_estimator_.constant == 0 and search.cv is folds
        np.testing.assert_array_equal(search.predict(X), np.zeros(len(rows)))
        np.testing.assert_array_equal(mapped, expected)
        np.testing.assert_array_equal(np.load(path), disk)
    finally:
        mapped._mmap.close()
