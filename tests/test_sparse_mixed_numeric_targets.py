"""Public label semantics for pandas extension arrays and their wrappers."""

from contextlib import nullcontext
from datetime import datetime, timedelta, tzinfo
from sys import version_info

import numpy as np
import pytest
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.multiclass import check_classification_targets

from dwd._validation import exact_sparse_uint64_target
from dwd.gen_kern_dwd import KernGDWD
from dwd.kern_md import KernMD

pd = pytest.importorskip("pandas")
BASE = 2**53


@pytest.mark.parametrize("base", [pd.Series, pd.Index])
def test_optional_vector_interface_defers_to_exact_storage_and_available_struct(base):
    calls = []
    enabled = False
    represented = np.array([10, 20] * 2)

    class Labels(base):
        @property
        def __array_struct__(self):
            calls.append("struct")
            if enabled:
                return represented.__array_struct__
            raise AttributeError("no struct")

        @property
        def __array_interface__(self):
            calls.append("interface")
            raise AttributeError("no interface")

    stored = pd.arrays.SparseArray(np.array([0, BASE + 1] * 2, np.int64), fill_value=0.)
    y = Labels(stored) if base is pd.Series else Labels._simple_new(stored)
    X = np.array([[-2.], [2.], [-1.], [1.]])
    for enabled in (False, True):
        calls.clear()
        np.asarray(y)
        native_calls = calls.copy()
        calls.clear()
        expected = [10, 20] if enabled else [0, BASE + 1]
        assert list(map(int, KernMD().fit(X, y).classes_)) == expected
        assert calls == native_calls


def test_unavailable_array_method_uses_the_selected_sequence_once():
    calls = []
    rows = np.array([0, BASE + 1] * 2, dtype=np.uint64)

    class Series(pd.Series):
        @property
        def __array__(self):
            calls.append("array")
            raise AttributeError("unavailable")

        def __iter__(self):
            calls.append("iteration")
            return iter(rows)

    y = Series(np.zeros(4))
    calls.clear()
    np.testing.assert_array_equal(np.asarray(y), rows)
    native_calls = calls.copy()
    calls.clear()
    np.testing.assert_array_equal(exact_sparse_uint64_target(y), rows)
    assert calls == native_calls


@pytest.mark.parametrize("columns", [1, 2])
def test_declined_series_array_preserves_mixed_rows_and_validates_shape(columns):
    reads = []
    scalars = [0., np.uint64(BASE + 1)] * 2

    class Series(pd.Series):
        @property
        def __array__(self):
            raise AttributeError("use sequence")

        def __iter__(self):
            reads.append("iteration")
            return iter([[v] * columns for v in scalars])

    y = Series(np.zeros(4))
    X = np.array([[-2.], [2.], [-1.], [1.]])
    if columns == 1:
        with pytest.warns(DataConversionWarning):
            model = KernMD().fit(X, y)
        assert list(map(int, model.predict(X))) == [0, BASE + 1] * 2
    else:
        with pytest.raises(ValueError, match="1d array"):
            KernMD().fit(X, y)
    assert reads == ["iteration"]


def test_partially_consumed_extension_iterator_propagates_its_own_type_error():
    marker = TypeError("invalid third label")
    calls = []

    class Sparse(pd.arrays.SparseArray):
        @property
        def __array__(self):
            raise AttributeError("use sequence")

        def __iter__(self):
            calls.append("iteration")
            yield 0.
            yield np.uint64(BASE)
            raise marker

    y = Sparse(np.array([0, BASE, BASE + 1], dtype=np.uint64), fill_value=0.)
    with pytest.raises(TypeError) as caught:
        KernMD().fit(np.eye(3), y)
    assert caught.value is marker
    assert calls == ["iteration"]


@pytest.mark.skipif(version_info < (3, 12), reason="Python-level buffer support requires 3.12")
def test_declined_series_buffer_falls_back_once_to_exact_sparse_labels():
    calls = []

    class Series(pd.Series):
        def __buffer__(self, flags):
            calls.append(flags)
            raise MemoryError("contiguous buffer unavailable")

    y = Series(pd.arrays.SparseArray(np.array([0, BASE + 1] * 2, dtype=np.int64), fill_value=0.))
    np.asarray(y)
    requested = calls.copy()
    calls.clear()
    assert list(map(int, exact_sparse_uint64_target(y))) == [0, BASE + 1] * 2
    assert len(calls) == 1 and calls == requested


@pytest.mark.parametrize(("fill", "hook"), [(np.int64(0), "decline"), (0., "decline"),
                                          (0., "active"), (0., "raise"), (0., "manager")])
def test_optional_frame_protocol_keeps_selected_manager_and_dense_target(fill, hook):
    calls = []
    marker = RuntimeError("selected sparse conversion")
    rounded = np.array([0., float(BASE)] * 2)

    class Sparse(pd.arrays.SparseArray):
        @property
        def __array_interface__(self):
            calls.append("inner")
            if hook == "active":
                return rounded.__array_interface__
            if hook == "raise":
                raise marker
            raise AttributeError("unavailable")

    class Frame(pd.DataFrame):
        @property
        def __array_interface__(self):
            raise AttributeError("unavailable")

        @property
        def _mgr(self):
            calls.append("manager")
            return self.__dict__["_mgr"]

        @_mgr.setter
        def _mgr(self, value):
            self.__dict__["_mgr"] = value

    def target():
        y = Frame({"y": Sparse(np.array([0, BASE + 1] * 2, np.int64), fill_value=fill)})
        if hook == "manager":
            def as_array(*args, **kwargs):
                calls.append("as_array")
                return np.array([7, 13] * 2)[:, None]
            y._mgr.as_array = as_array
        calls.clear()
        return y

    X = np.array([[-2.], [2.], [-1.], [1.]])
    try:
        visible = np.asarray(target()).ravel()
    except RuntimeError as error:
        assert error is marker
        native_reads = calls.copy()
        with pytest.raises(RuntimeError) as caught:
            KernMD().fit(X, target())
        assert caught.value is marker
        assert calls == native_reads
        return
    native_reads = calls.copy()
    with pytest.warns(DataConversionWarning):
        model = KernMD().fit(X, target())
    declared = hook == "active" and "inner" in native_reads or "as_array" in native_reads
    expected = list(map(int, visible)) if declared else [0, BASE + 1] * 2
    assert list(map(int, model.predict(X))) == expected
    assert calls == native_reads


def test_frame_manager_descriptor_selects_labels_without_an_outer_array_override():
    first = pd.DataFrame({"y": pd.arrays.SparseArray([0, 13] * 2, fill_value=0.)})
    second = pd.DataFrame({"y": pd.arrays.SparseArray([7, BASE + 1] * 2, fill_value=0.)})
    reads = []

    class SwitchingFrame(pd.DataFrame):
        @property
        def _mgr(self):
            reads.append("manager")
            return first._mgr if len(reads) <= 2 else second._mgr

        @_mgr.setter
        def _mgr(self, manager):
            self.__dict__["_mgr"] = manager

    def target():
        frame = SwitchingFrame(first)
        reads.clear()
        return frame

    public = np.asarray(target()).ravel()
    native_reads = reads.copy()
    expected = [0, 13] if public[0] == 0 else [7, BASE + 1]
    with pytest.warns(DataConversionWarning):
        model = KernMD().fit(np.array([[-2.], [2.], [-1.], [1.]]), target())
    assert list(map(int, model.classes_)) == expected
    assert reads == native_reads


@pytest.mark.parametrize("kind", ["sparse", "nullable", "dense"])
def test_plain_frame_calls_its_manager_conversion_only_when_selected(kind):
    native = np.array([0, BASE + 1] * 2, dtype=np.int64)
    if kind == "sparse":
        column = pd.arrays.SparseArray(native, fill_value=0.)
    elif kind == "nullable":
        column = pd.array(native, dtype="Int64")
    else:
        column = native
    frame = pd.DataFrame({"y": column})
    calls = []
    error = None

    def convert(*args, **kwargs):
        calls.append("manager")
        if error is not None:
            raise error
        return np.array([17, 43] * 2)[:, None]

    frame._mgr.as_array = convert
    public = np.asarray(frame).ravel()
    native_calls = calls.copy()
    calls.clear()
    X = np.array([[-2.], [2.], [-1.], [1.]])
    with pytest.warns(DataConversionWarning):
        model = KernMD().fit(X, frame)
    expected = public if native_calls else native
    assert list(map(int, model.predict(X))) == list(map(int, expected))
    assert calls == native_calls
    error = RuntimeError("selected manager failed")
    calls.clear()
    if native_calls:
        with pytest.raises(RuntimeError) as caught:
            model.score(X, frame)
        assert caught.value is error
    else:
        assert model.score(X, frame) == 1.
    assert calls == native_calls


def test_sparse_category_observes_instance_selection_and_propagates_its_exception():
    levels = pd.arrays.SparseArray(np.array([0, BASE + 3], dtype=np.int64), fill_value=0.)
    y = pd.Categorical.from_codes([0, 1] * 2, categories=pd.Index(levels))
    calls = []
    error = None

    def take(indices, fill_value=None):
        calls.append("take")
        if error is not None:
            raise error
        return np.array([0, BASE + 8], dtype=np.int64)[indices]

    y.categories._values._take_with_fill = take
    assert list(map(int, exact_sparse_uint64_target(y))) == [0, BASE + 8] * 2
    assert calls == ["take"]
    calls.clear()
    error = RuntimeError("selected level take")
    with pytest.raises(RuntimeError) as caught:
        exact_sparse_uint64_target(y)
    assert caught.value is error and calls == ["take"]


@pytest.mark.parametrize("hook", ["to_numpy", "astype", "absent_interface", "instance_array"])
def test_nested_sparse_levels_only_honor_hooks_used_during_take_and_projection(hook):
    events = []

    class Levels(pd.arrays.SparseArray):
        pass

    values = Levels(np.array([0, BASE + 1], dtype=np.int64), fill_value=0.)
    labels = pd.Categorical.from_codes([0, 1] * 2, categories=pd.Index(values))

    def unused(*args, **kwargs):
        events.append("unused")
        raise AssertionError("this categorical does not invoke that hook on its levels")

    if hook in ("to_numpy", "astype"):
        setattr(Levels, hook, unused)
    elif hook == "absent_interface":
        def unavailable(self):
            events.append("interface")
            raise AttributeError("no array interface")
        Levels.__array_interface__ = property(unavailable)
    if hook == "instance_array":
        labels.categories._values.__array__ = unused

    class Series(pd.Series):
        @property
        def _values(self):
            return labels

    y = Series(np.zeros(4))
    events.clear()
    assert list(map(int, np.asarray(y))) == [0, BASE] * 2
    native_events = events.copy()
    events.clear()
    assert list(map(int, exact_sparse_uint64_target(y))) == [0, BASE + 1] * 2
    assert events == native_events


@pytest.mark.parametrize("carrier", ["index", "series"])
@pytest.mark.parametrize("storage", ["native", "sparse", "nullable", "projected"])
def test_vector_selected_storage_descriptor_preserves_first_observations_and_errors(carrier, storage):
    ids = np.array([0, BASE + 1] * 2, dtype=np.uint64)
    later = np.array([0, BASE + 4] * 2, dtype=np.uint64)
    calls = []
    failure = None

    def selected(original):
        calls.append(1)
        if failure is not None:
            raise failure
        return original if len(calls) == 1 else later

    class Index(pd.Index):
        @property
        def _data(self):
            return selected(self.original)

        @_data.setter
        def _data(self, value):
            self.original = value

    class Series(pd.Series):
        @property
        def _values(self):
            return selected(super()._values)

    class Projected(pd.arrays.IntegerArray):
        def __array__(self, dtype=None, copy=None):
            return np.asarray(ids, dtype=dtype)

    if storage == "sparse":
        first = pd.arrays.SparseArray(ids.astype(np.int64), fill_value=0.)
    elif storage == "nullable":
        first = pd.array(ids, dtype="UInt64")
    elif storage == "projected":
        first = Projected(np.zeros(4, dtype=np.int64), np.zeros(4, dtype=bool))
    else:
        first = ids
    y = Index._simple_new(first) if carrier == "index" else Series(first, copy=False)
    X = np.array([[-2.], [2.], [-1.], [1.]])
    calls.clear()
    model = KernMD().fit(X, y)
    assert list(map(int, model.predict(X))) == ids.tolist()
    assert calls == [1]
    calls.clear()
    assert model.score(X, y, sample_weight=[1, 3, 5, 7]) == 1.
    assert calls == [1]
    calls.clear()
    monitor = KernGDWD(initialization="zero", stopping="validation", max_iter=2)
    monitor.fit(X, ids, validation_data=(X, y))
    assert [row["score"] for row in monitor.validation_history_] == [1., 1.]
    assert calls == [1]
    calls.clear()
    failure = RuntimeError("selected index storage")
    with pytest.raises(RuntimeError) as caught:
        model.score(X, y)
    assert caught.value is failure and calls == [1]


def test_specialized_index_array_conversion_bypasses_an_inactive_storage_descriptor():
    rows = np.array([0, BASE + 1] * 2, dtype=np.uint64)
    calls = []

    class Labels(pd.MultiIndex):
        @property
        def _data(self):
            raise AssertionError("unused storage descriptor")

        @property
        def values(self):
            calls.append("values")
            return rows

    y = Labels.from_tuples([(0, i) for i in range(4)])
    calls.clear()
    assert list(map(int, exact_sparse_uint64_target(y))) == rows.tolist()
    assert calls == ["values"]


def test_declared_index_array_protocol_precedes_its_storage_descriptor():
    ids = np.array([0, BASE + 1] * 2, dtype=np.uint64)
    calls = []

    class Labels(pd.Index):
        @property
        def _data(self):
            raise AssertionError("unselected index storage")

        @_data.setter
        def _data(self, value):
            pass

        def __array__(self, dtype=None, copy=None):
            calls.append("array")
            return np.asarray(ids, dtype=dtype)

    y = Labels._simple_new(np.zeros(4, dtype=np.int64))
    X = np.array([[-2.], [2.], [-1.], [1.]])
    assert list(map(int, KernMD().fit(X, y).predict(X))) == ids.tolist()
    assert calls == ["array"]


def test_series_selected_extension_retains_pandas_post_conversion_dtype_reads():
    ids = np.array([0, BASE + 1] * 2, dtype=np.uint64)
    calls = []
    marker = None

    class Extension(pd.arrays.IntegerArray):
        @property
        def dtype(self):
            calls.append("dtype")
            if marker is not None:
                raise marker
            return super().dtype

        def __array__(self, dtype=None, copy=None):
            calls.append("array")
            return np.asarray(ids, dtype=dtype)

    values = Extension(np.zeros(4, dtype=np.int64), np.zeros(4, dtype=bool))

    class Series(pd.Series):
        @property
        def _values(self):
            calls.append("values")
            return values

    class Explicit(Series):
        def __array__(self, dtype=None, copy=None):
            calls.append("outer")
            return np.asarray(ids, dtype=dtype)

    for cls in (Series, Explicit):
        marker = None
        y = cls(np.zeros(4, dtype=np.int64))
        calls.clear()
        np.testing.assert_array_equal(np.asarray(y), ids)
        selected = calls.copy()
        if cls is Explicit:
            assert selected == ["outer"]
        calls.clear()
        np.testing.assert_array_equal(exact_sparse_uint64_target(y), ids)
        assert calls == selected
        if "dtype" in selected:
            calls.clear()
            marker = RuntimeError("pandas selected the extension dtype")
            with pytest.raises(RuntimeError) as caught:
                exact_sparse_uint64_target(y)
            assert caught.value is marker and calls == selected


@pytest.mark.parametrize("subclass", [False, True])
def test_selected_sparse_catalogue_keeps_exact_levels_after_equivalent_fill_change(subclass):
    class Sparse(pd.arrays.SparseArray):
        pass

    cls = Sparse if subclass else pd.arrays.SparseArray
    levels = cls(np.array([0, BASE, BASE + 1], dtype=np.int64), fill_value=np.int64(0))
    labels = pd.Categorical.from_codes([0, 1, 2] * 2, categories=pd.Index(levels))
    levels.fill_value = 0.

    class Series(pd.Series):
        @property
        def _values(self):
            return labels

    assert list(map(int, exact_sparse_uint64_target(Series(np.zeros(6))))) == [0, BASE, BASE + 1] * 2


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
def test_selected_sparse_categorical_datetime_and_timedelta_retain_the_public_dtype(dtype):
    ticks = np.array([1, 2], dtype=dtype)
    levels = pd.arrays.SparseArray(ticks, fill_value=ticks[0])
    labels = pd.Categorical.from_codes([0, 1] * 2, categories=pd.Index(levels))

    class Series(pd.Series):
        @property
        def _values(self):
            return labels

    y = Series(np.zeros(4))
    public = np.asarray(y)
    selected = exact_sparse_uint64_target(y)
    assert selected.dtype == public.dtype
    np.testing.assert_array_equal(selected, public)


@pytest.mark.parametrize("carrier", ["lazy_index", "categorical_levels", "range_index", "series_manager"])
def test_inherited_pandas_vector_conversion_retains_its_indirectly_selected_storage(carrier):
    calls = []
    rows = np.array([0, BASE + 1] * 2, dtype=np.uint64)
    changed = np.array([0, BASE + 4] * 2, dtype=np.uint64)

    def selected(first, second):
        calls.append(1)
        return first if len(calls) == 1 else second

    if carrier == "range_index":
        class Labels(pd.RangeIndex):
            @property
            def _data(self):
                return selected(rows, changed)

        y = Labels(4)
    elif carrier == "series_manager":
        other = pd.Series(changed)._mgr

        class Labels(pd.Series):
            @property
            def _mgr(self):
                return selected(self.original, other)

            @_mgr.setter
            def _mgr(self, value):
                object.__setattr__(self, "original", value)

        y = Labels(rows)
    else:
        first = rows if carrier == "lazy_index" else rows[:2]
        second = changed if carrier == "lazy_index" else changed[:2]

        class Labels(pd.Index):
            def __getattr__(self, name):
                if name == "_data":
                    return selected(first, second)
                raise AttributeError(name)

        labels = Labels._simple_new(first)
        del labels._data
        y = labels if carrier == "lazy_index" else pd.Categorical.from_codes([0, 1] * 2, categories=labels)
    calls.clear()
    public = np.asarray(y)
    selected_reads = len(calls)
    assert selected_reads
    calls.clear()
    X = np.array([[-2.], [2.], [-1.], [1.]])
    model = KernMD().fit(X, public)
    assert model.score(X, y, sample_weight=[1, 3, 5, 7]) == 1.
    assert len(calls) == selected_reads


@pytest.mark.parametrize("source", ["categories", "level_index", "sparse_fill"])
def test_selected_pandas_storage_is_read_once_per_public_conversion(source):
    first = pd.Index([0, BASE + 1], dtype="uint64")
    later = pd.Index([0, BASE + 4], dtype="uint64")
    reads = []

    class Category(pd.Categorical):
        @property
        def categories(self):
            reads.append("categories")
            return first if len(reads) == 1 else later

    class Levels(pd.Index):
        @property
        def _values(self):
            reads.append("levels")
            return first._values if len(reads) == 1 else later._values

    class Sparse(pd.arrays.SparseArray):
        @property
        def fill_value(self):
            reads.append("fill")
            return np.uint64(BASE + (1 if len(reads) == 1 else 4))

    if source == "categories":
        values = Category.from_codes([0, 1] * 2, categories=first)
    elif source == "level_index":
        levels = Levels._simple_new(first._values)
        values = pd.Categorical.from_codes([0, 1] * 2, categories=levels)
    else:
        values = Sparse(np.array([0, 1] * 2, dtype=np.uint64), fill_value=np.uint64(0))
    y = pd.Series(values, copy=False).to_frame()
    reads.clear()
    projected = np.asarray(y).ravel()
    expected_reads = reads.copy()
    X = np.array([[-2.], [2.], [-1.], [1.]])
    trusted = KernMD().fit(X, projected)
    for operation in (lambda v: list(map(int, KernMD().fit(X, v).classes_)),
                      lambda v: trusted.score(X, v, sample_weight=[1, 3, 5, 7]),
                      lambda v: [row["score"] for row in KernGDWD(initialization="zero",
                          stopping="validation", max_iter=2).fit(X, projected, validation_data=(X, v)).validation_history_]):
        reads.clear()
        assert operation(y) == operation(projected)
        assert reads == expected_reads


def test_sparse_metadata_and_categorical_scalar_access_are_not_used_for_recovery():
    class Sparse(pd.arrays.SparseArray):
        @property
        def npoints(self):
            raise AssertionError("unused metadata")

        def __getitem__(self, key):
            if isinstance(key, (int, np.integer)):
                raise AssertionError("unused scalar selector")
            return super().__getitem__(key)

    values = Sparse(np.array([0, BASE + 1], dtype=np.int64), fill_value=0.)
    category = pd.Categorical.from_codes([0, 1] * 2, categories=pd.Index(values))
    assert list(map(int, exact_sparse_uint64_target(values))) == [0, BASE + 1]
    assert list(map(int, exact_sparse_uint64_target(category))) == [0, BASE + 1] * 2


def test_sparse_category_honors_the_fill_returned_during_level_selection():
    calls = []
    armed = False

    class Levels(pd.arrays.SparseArray):
        @property
        def fill_value(self):
            if not armed:
                return super().fill_value
            calls.append(1)
            return np.uint64(BASE + (1 if len(calls) == 1 else 4))

    levels = Levels(np.array([0, 1, 2], dtype=np.uint64), fill_value=np.uint64(0))
    y = pd.Categorical.from_codes([1, 0] * 2, categories=pd.Index(levels))
    armed = True
    projected = np.asarray(y)
    expected_reads = len(calls)
    calls.clear()
    np.testing.assert_array_equal(exact_sparse_uint64_target(y), projected)
    assert len(calls) == expected_reads


def test_indistinguishable_dynamic_sparse_category_levels_require_exact_or_projected_ids():
    class Levels(pd.arrays.SparseArray):
        def __getattribute__(self, name):
            return super().__getattribute__(name)

    levels = Levels(np.array([0, BASE + 1], dtype=np.int64), fill_value=0.)
    y = pd.Categorical.from_codes([0, 1] * 2, categories=pd.Index(levels))
    projected = np.asarray(y)
    assert list(map(int, projected)) == [0, BASE] * 2
    with pytest.raises(ValueError, match="changes a stored class ID"):
        exact_sparse_uint64_target(y)
    assert list(map(int, exact_sparse_uint64_target(projected))) == [0, BASE] * 2


@pytest.mark.parametrize("field", ["dtype", "ndim", "sp_index"])
@pytest.mark.parametrize("primed", [False, True])
def test_lossy_sparse_frame_propagates_the_last_selected_accessor_error(field, primed):
    calls = []
    fail_at = None
    marker = RuntimeError("selected sparse accessor")

    class Sparse(pd.arrays.SparseArray):
        def __getattribute__(self, name):
            if name == field:
                calls.append(name)
                if len(calls) == fail_at:
                    raise marker
            return super().__getattribute__(name)

    def frame():
        values = Sparse(np.array([1, 0, BASE + 1] * 2, dtype=np.uint64), fill_value=np.uint64(0))
        values.fill_value = float(BASE)
        y = pd.DataFrame({"y": values})
        if primed:
            np.asarray(y)
        calls.clear()
        return y

    np.asarray(frame())
    if not calls:
        pytest.skip("this manager does not read that accessor")
    count = len(calls)
    y = frame()
    fail_at = count
    with pytest.raises(RuntimeError) as caught:
        KernMD().fit(np.eye(6), y)
    assert caught.value is marker and len(calls) == fail_at


@pytest.mark.parametrize("stored,fill", [(2, np.int64(0)), (BASE + 1, 0.)])
def test_sparse_subclass_extra_observation_preserves_the_logical_fill_and_exact_points(stored, fill):
    class ExtraFill(pd.arrays.SparseArray):
        def __len__(self):
            return super().__len__() + 1

    y = ExtraFill(np.array([0, stored] * 2, dtype=np.int64), fill_value=fill)
    assert np.asarray(y).shape == (5,)
    X = np.array([[-2.], [2.], [-1.], [1.], [-3.]])
    model = KernMD().fit(X, y)
    assert list(map(int, model.predict(X))) == [0, stored, 0, stored, 0]
    assert model.score(X, y, sample_weight=[1, 2, 3, 4, 5]) == 1.


@pytest.mark.parametrize("wide", [False, True])
def test_repeated_sparse_frame_conversions_use_its_live_pandas_cache(wide):
    def frame():
        calls = []

        class Sparse(pd.arrays.SparseArray):
            def __getattribute__(self, name):
                if name in ("sp_index", "dtype"):
                    calls.append(name)
                return super().__getattribute__(name)

        stored = BASE + 1 if wide else 2
        values = Sparse(np.array([0, stored] * 2, dtype=np.int64), fill_value=0. if wide else np.int64(0))
        y = pd.DataFrame({"y": values})
        calls.clear()
        return y, calls

    native, expected = frame()
    y, actual = frame()
    X = np.array([[-2.], [2.], [-1.], [1.]])
    np.asarray(native)
    with pytest.warns(DataConversionWarning):
        model = KernMD().fit(X, y)
    assert actual == expected
    for _ in range(2):
        expected.clear()
        actual.clear()
        np.asarray(native)
        assert model.score(X, y) == 1.
        assert actual == expected


def test_active_sparse_fill_can_introduce_loss_over_homogeneous_raw_storage():
    class Sparse(pd.arrays.SparseArray):
        @property
        def fill_value(self):
            return float(BASE)

    values = Sparse(np.array([1, 0, BASE + 1] * 2, dtype=np.uint64), fill_value=np.uint64(0))
    y = pd.DataFrame({"y": values})
    assert list(map(int, np.asarray(y).ravel())) == [1, BASE, BASE] * 2
    assert list(map(int, exact_sparse_uint64_target(y).ravel())) == [1, BASE, BASE + 1] * 2


@pytest.mark.parametrize("subtype,fill", [(np.int8, 256), (np.int8, -129), (np.uint8, 257),
                                           (np.int16, 0.5), (np.int8, 2)])
def test_legacy_sparse_frame_preserves_fills_outside_its_storage_subtype(subtype, fill):
    try:
        pd.get_option("mode.data_manager")
    except pd.errors.OptionError:
        pytest.skip("ArrayManager is unavailable")

    class Passive(pd.arrays.SparseArray):
        def __getattribute__(self, name):
            return super().__getattribute__(name)

    with pd.option_context("mode.data_manager", "array"):
        for cls in (pd.arrays.SparseArray, Passive):
            values = cls(np.array([0, 1] * 2, dtype=subtype), fill_value=subtype(0))
            values.fill_value = fill
            y = pd.DataFrame({"y": values})
            np.testing.assert_array_equal(exact_sparse_uint64_target(y).ravel(), [fill, 1] * 2)


def sparse_labels(ids, storage):
    if storage == "int_points":
        return pd.arrays.SparseArray(np.asarray(ids, dtype=np.int64), fill_value=0.)
    # Compress against a sentinel first. A float comparison against BASE + 1
    # during construction could discard a stored BASE before an estimator sees it.
    raw = np.array([1. if value == BASE + 1 else float(value) for value in ids])
    array = pd.arrays.SparseArray(raw, fill_value=1.)
    array.fill_value = np.int64(BASE + 1)
    return array


@pytest.mark.parametrize("storage", ["int_points", "int_fill"])
@pytest.mark.parametrize("carrier", ["array", "series", "index", "frame"])
def test_binary_sparse_fit_and_score_use_source_ids(storage, carrier):
    def target(ids):
        array = sparse_labels(ids, storage)
        assert [int(array[i]) for i in range(len(ids))] == ids
        if carrier == "array":
            return array
        if carrier == "index":
            return pd.Index(array)
        series = pd.Series(array)
        return series.to_frame() if carrier == "frame" else series

    X = np.tile([[-1.], [1.]], (4, 1))
    ids = [0, BASE + 1] * 4
    y = target(ids)
    if carrier == "frame":
        with pytest.warns(DataConversionWarning):
            model = KernMD().fit(X, y)
    else:
        model = KernMD().fit(X, y)
    assert list(map(int, model.classes_)) == [0, BASE + 1]
    assert list(map(int, model.predict(X))) == ids
    assert model.score(X, y) == 1.
    truth = target([0, BASE, BASE + 1])
    assert model.score(X[[0, 1, 1]], truth, sample_weight=[1, 4, 3]) == .5
    with pytest.raises(ValueError, match="exactly two classes"):
        KernMD().fit(X[:3], truth)


def test_binary_scores_bare_sparse_rows_without_the_training_fill():
    ids = np.array([0, -(BASE + 1)] * 2, dtype=np.int64)
    y = pd.arrays.SparseArray(ids, fill_value=0.)
    X = np.array([[-1.], [1.]] * 2)
    model = KernMD().fit(X, y)
    assert model.predict(X).tolist() == ids.tolist()
    assert y.sp_index.ngaps == 2 and model.score(X, y) == 1.

    rows = [1, 3, 1, 3]
    held = y.take(rows)
    assert held.sp_index.ngaps == 0
    for weights, expected in ((None, .5), ([1, 2, 4, 8], 10 / 15)):
        assert model.score(X, ids[rows], sample_weight=weights) == expected
        assert model.score(X, held, sample_weight=weights) == expected


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
def test_binary_scores_bare_sparse_temporal_labels(dtype):
    ids = np.array([0, 1] * 2, dtype=dtype)
    y = pd.arrays.SparseArray(ids, fill_value=ids[0])
    X = np.array([[-1.], [1.]] * 2)
    model = KernMD().fit(X, y)
    assert model.score(X, y) == 1.
    held = y.take([1, 3, 1, 3])
    assert model.score(X, held, sample_weight=[1, 2, 4, 8]) == 10 / 15


@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
@pytest.mark.parametrize("carrier", ["plain", "inherited", "explicit"])
def test_binary_recovers_only_the_selected_sparse_temporal_fill(scalar, carrier):
    native = np.array([0, 1] * 2, dtype=f"{scalar.__name__}[ns]")
    X = np.array([[-1.], [1.]] * 2)

    class Inherited(pd.arrays.SparseArray):
        pass

    class Explicit(Inherited):
        def __array__(self, dtype=None, copy=None):
            return native.astype(dtype or native.dtype, copy=False)

    cls = {"plain": pd.arrays.SparseArray, "inherited": Inherited, "explicit": Explicit}[carrier]
    y = cls(native, fill_value=native[0])
    try:
        y.fill_value = scalar(1, "ps")
    except ValueError:
        pytest.skip("pandas rejects the finer fill at construction")
    model = KernMD().fit(X, native)
    explicit = carrier == "explicit"
    assert model.score(X, y) == (1. if explicit else .5)
    assert model.score(X, y, sample_weight=[1, 2, 4, 8]) == (1. if explicit else 10 / 15)
    fitted = KernMD().fit(X, y)
    assert fitted.classes_.dtype == np.dtype(f"{scalar.__name__}[{'ns' if explicit else 'ps'}]")
    assert fitted.classes_.view("i8").tolist() == ([0, 1] if explicit else [1, 1000])
    assert fitted.score(X, y) == 1.


@pytest.mark.parametrize("fill", [np.datetime64(2**55, "s"), np.datetime64(-(2**55), "s"), "1970-01-01", 0])
def test_sparse_temporal_labels_refuse_unrepresentable_or_nontemporal_occupied_fill(fill):
    native = np.array([0, 1] * 2, dtype="datetime64[ns]")
    y = pd.arrays.SparseArray(native, fill_value=native[0])
    try:
        y.fill_value = fill
    except ValueError:
        pytest.skip("pandas rejects this fill at construction")
    X = np.array([[-1.], [1.]] * 2)
    model = KernMD().fit(X, native)
    if isinstance(fill, np.datetime64):
        assert model.score(X, y) == .5
        assert model.score(X, y, sample_weight=[1, 2, 4, 8]) == 10 / 15
    else:
        with pytest.raises(ValueError, match="incompatible occupied fill"):
            model.score(X, y)
    with pytest.raises(ValueError, match="exact shared unit|incompatible occupied fill"):
        KernMD().fit(X, y)
    assert model.score(X, y.take([1] * 4)) == .5


def test_mixed_unit_temporal_validation_cannot_wrap_onto_a_training_class():
    X = np.array([[-1.], [1.]] * 2)
    native = np.array([0, 1] * 2, dtype="datetime64[ns]")
    y = pd.arrays.SparseArray(native, fill_value=native[0])
    y.fill_value = np.datetime64(1, "ps")
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    # In int64 picoseconds (2**61 + 1) ns wraps to 1000 ps.
    held = np.array([1, 2**61 + 1] * 2, dtype=native.dtype)
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, y, validation_data=(X, held))
    model.fit(X, y, validation_data=(X[1::2], native[1::2]))
    assert [row["score"] for row in model.validation_history_] == [1., 1.]
    scored = KernMD().fit(X, y)
    for carrier in (list(held), pd.Series(held), pd.Index(held)):
        assert scored.score(X, carrier) == 0.
    for vector in (pd.Series(y), pd.Index(y)):
        assert scored.score(X, vector) == 1.

    mixed = [np.datetime64(1, "ps"), np.datetime64(2**61 + 1, "ns")] * 2
    assert scored.score(X, mixed) == .5
    with pytest.raises(ValueError, match="exact shared unit|Validation labels must belong"):
        model.fit(X, y, validation_data=(X, mixed))

    reads = []
    class Iterated(pd.arrays.SparseArray):
        @property
        def __array__(self):
            raise AttributeError("use the selected iterator")

        def __iter__(self):
            reads.append(True)
            return iter(mixed)

    assert scored.score(X, Iterated(native, fill_value=native[0])) == .5
    assert reads == [True]


def test_sparse_calendar_datetime_accepts_an_exact_day_representation():
    native = np.array([0, 1] * 2, dtype="datetime64[2M]")
    try:
        y = pd.arrays.SparseArray(native, fill_value=native[0])
        y.fill_value = np.datetime64(1, "W")
    except (ValueError, TypeError):
        pytest.skip("pandas rejects this sparse calendar dtype or fill")
    X = np.array([[-1.], [1.]] * 2)
    model = KernMD().fit(X, y)
    np.testing.assert_array_equal(model.classes_.astype("datetime64[D]"),
                                  np.array(["1970-01-08", "1970-03-01"], dtype="datetime64[D]"))
    assert model.score(X, y) == 1.


@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
@pytest.mark.parametrize(("unit", "stride", "seconds"), [("ns", 3, 12_000_000_000),
                                                          ("ns", 11, -44_000_000_000),
                                                          ("as", 3, 12)])
def test_mixed_temporal_values_fit_score_and_monitor_on_an_exact_stepped_unit(scalar, unit, stride, seconds):
    width = 10**9 if unit == "ns" else 10**18
    ticks = [seconds * width // stride, 1] * 2
    dtype = np.dtype(f"{scalar.__name__}[{stride}{unit}]")
    native = np.array(ticks, dtype=dtype)
    held = [scalar(seconds, "s"), scalar(stride, unit)] * 2
    X = np.array([[-1.], [1.]] * 2)
    model = KernMD().fit(X, held)
    assert model.classes_.dtype == dtype
    assert model.classes_.view("i8").tolist() == sorted(set(ticks))
    assert model.score(X, native) == model.score(X, held) == 1.
    assert model.score(X, held[::-1]) == 0.
    monitored = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    monitored.fit(X, native, validation_data=(X, held))
    assert [row["score"] for row in monitored.validation_history_] == [1., 1.]


@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
def test_mixed_temporal_validation_uses_the_fitted_grid_even_when_inference_needs_a_divisor(scalar):
    first, second = 1_000_000_007, 1_000_000_009
    product = first * second
    held = [scalar(product, "s"), scalar(product, "ns")] * 2
    native = np.array([second * 10**9, second] * 2, dtype=f"{scalar.__name__}[{first}ns]")
    X = np.array([[-1.], [1.]] * 2)
    model = KernMD().fit(X, native)
    assert model.score(X, held) == 1.
    assert model.score(X, [held[0]] * 4, sample_weight=[1, 2, 4, 8]) == 5 / 15
    for missing in (scalar("NaT"), scalar("NaT", "M")):
        assert model.score(X, [missing, *held[1:]]) == .75
    monitored = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    monitored.fit(X, native, validation_data=(X, held))
    assert [row["score"] for row in monitored.validation_history_] == [1., 1.]
    unknown = [held[0], scalar(product + 1, "ns")] * 2
    with pytest.raises(ValueError, match="Validation labels must belong"):
        monitored.fit(X, native, validation_data=(X, unknown))


@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
def test_bare_sparse_temporal_labels_can_require_a_stepped_unit(scalar):
    native = np.array([0, 3] * 2, dtype=f"{scalar.__name__}[ns]")
    y = pd.arrays.SparseArray(native, fill_value=native[0])
    try:
        y.fill_value = scalar(12_000_000_000, "s")
    except ValueError:
        pytest.skip("pandas rejects this occupied fill at construction")
    X = np.array([[-1.], [1.]] * 2)
    model = KernMD().fit(X, y)
    assert model.classes_.dtype == np.dtype(f"{scalar.__name__}[3ns]")
    assert model.classes_.view("i8").tolist() == [1, 4 * 10**18]
    assert model.score(X, y) == 1.
    assert model.score(X[::-1], y) == 0.
    monitored = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    monitored.fit(X, y, validation_data=(X, y))
    assert [row["score"] for row in monitored.validation_history_] == [1., 1.]


@pytest.mark.parametrize(("scalar", "other_unit"), [(np.datetime64, "generic"), (np.timedelta64, "M")])
def test_missing_only_temporal_truth_has_zero_accuracy_in_either_unit_order(scalar, other_unit):
    X = np.array([[-1.], [1.]] * 2)
    native = np.array([0, 1] * 2, dtype=f"{scalar.__name__}[ns]")
    model = KernMD().fit(X, native)
    fixed, other = scalar("NaT", "ns"), scalar("NaT", other_unit)
    assert model.score(X, np.full(4, fixed)) == 0.
    for held in ([fixed, other] * 2, [other, fixed] * 2):
        assert model.score(X, held) == 0.
        assert model.score(X, held, sample_weight=[1, 2, 4, 8]) == 0.
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            model.score(X, held, sample_weight=[1])
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            model.score(X, held[:2])
        monitored = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
        with pytest.raises(ValueError, match="Validation labels must belong"):
            monitored.fit(X, native, validation_data=(X, held))
    gaps = pd.arrays.SparseArray(np.full(4, fixed), fill_value=fixed)
    points = pd.arrays.SparseArray(native, fill_value=fixed)
    assert gaps.sp_index.ngaps == 4 and model.score(X, gaps) == 0.
    assert points.sp_index.ngaps == 0 and model.score(X, points) == 1.


@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("carrier", ["bare", "series", "index", "frame"])
def test_sparse_temporal_carriers_preserve_observed_units_and_missing_rows(scalar, categorical, carrier):
    ns = np.dtype(f"{scalar.__name__}[ns]")
    ps = np.dtype(f"{scalar.__name__}[ps]")
    X = np.array([[-1.], [1.]] * 2)
    native = np.array([0, 1] * 2, dtype=ns)

    def target(codes):
        # The unused third level cannot be converted to picoseconds.
        levels = np.array([0, 1, 2**61 + 1], dtype=ns)
        rows = levels if categorical else np.array([levels[c] if c >= 0 else scalar("NaT") for c in codes], dtype=ns)
        sparse = pd.arrays.SparseArray(rows, fill_value=levels[0])
        try:
            sparse.fill_value = scalar(1, "ps")
        except ValueError:
            pytest.skip("pandas rejects the finer fill at construction")
        y = pd.Categorical.from_codes(codes, categories=pd.Index(sparse)) if categorical else sparse
        if carrier == "index":
            return pd.Index(y)
        if carrier == "series":
            return pd.Series(y)
        return pd.Series(y).to_frame() if carrier == "frame" else y

    model = KernMD().fit(X, native)
    y = target([0, 1] * 2)
    assert model.score(X, y) == .5
    assert model.score(X, y, sample_weight=[1, 2, 4, 8]) == 10 / 15
    with pytest.warns(DataConversionWarning) if carrier == "frame" else nullcontext():
        fitted = KernMD().fit(X, y)
    assert fitted.classes_.dtype == ps and fitted.classes_.view("i8").tolist() == [1, 1000]
    assert fitted.score(X, target([1] * 4)) == .5
    assert fitted.score(X, target([-1, 1, 0, 1])) == .75
    monitored = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    with pytest.raises(ValueError, match="Validation labels must belong"):
        monitored.fit(X, native, validation_data=(X, y))
    if carrier == "frame":
        with pytest.raises(ValueError):
            fitted.score(X, pd.concat([y, y], axis=1))


@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("subclass", ["levels", "category"])
@pytest.mark.parametrize("frame", [False, True])
def test_inherited_temporal_category_uses_only_the_selected_levels(scalar, sign, subclass, frame):
    class DisplayMeta(type(pd.arrays.SparseArray)):
        pass

    class Levels(pd.arrays.SparseArray, metaclass=DisplayMeta):
        display_name: str = "dates"
        _display_options = {"precision": "native"}

        def label_name(self):
            return self.display_name

        @property
        def display_unit(self):
            raise AssertionError("Unrelated property was consulted for temporal labels")

    class Category(pd.Categorical):
        pass

    # January 1 and 2 of 2020, or the corresponding negative offsets.
    day = 86_400_000_000_000
    values = np.array([0, sign * day * 18_262, sign * day * 18_263], dtype=f"{scalar.__name__}[ns]")
    array_type = Levels if subclass == "levels" else pd.arrays.SparseArray
    cat_type = Category if subclass == "category" else pd.Categorical
    storage = array_type(values, fill_value=values[0])
    try:
        storage.fill_value = scalar(1, "ps")
    except ValueError:
        if scalar is not np.timedelta64:
            raise
        # pandas 3 rejects picosecond duration fills for nanosecond points.
        storage.fill_value = np.timedelta64(1, "ms")
    categories = pd.Index(storage)

    def target(codes):
        y = cat_type.from_codes(codes, categories=categories)
        return pd.DataFrame({"y": y}) if frame else y

    X = np.array([[-1.], [1.]] * 2)
    native = values[[1, 2] * 2]
    with pytest.warns(DataConversionWarning) if frame else nullcontext():
        model = KernMD().fit(X, target([1, 2] * 2))
    np.testing.assert_array_equal(model.predict(X), native)
    assert model.classes_.dtype == native.dtype
    assert model.score(X, target([1, 2, 1, 1]), sample_weight=[1, 2, 4, 8]) == 7 / 15
    # The same fill actually participates when code zero is used.
    held = target([0, 2, 1, 2])
    assert model.score(X, held, sample_weight=[1, 2, 4, 8]) == 14 / 15
    monitored = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    monitored.fit(X, native, validation_data=(X, target([1, 2] * 2)))
    assert [row["score"] for row in monitored.validation_history_] == [1., 1.]
    with pytest.raises(ValueError, match="Validation labels must belong"):
        monitored.fit(X, native, validation_data=(X, held))


def test_temporal_category_keeps_custom_row_constructor_overflow():
    marker = OverflowError("Overflow when converting between datetime64 units")
    armed = False
    calls = []

    class Constructor(type(pd.arrays.SparseArray)):
        def __call__(cls, *args, **kwargs):
            if armed:
                calls.append(True)
                raise marker
            return super().__call__(*args, **kwargs)

    class Levels(pd.arrays.SparseArray, metaclass=Constructor):
        pass

    values = np.array(["1970-01-01", "2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    sparse = Levels(values, fill_value=values[0])
    sparse.fill_value = np.datetime64(1, "ps")
    y = pd.Categorical.from_codes([1, 2] * 2, categories=pd.Index(sparse))
    armed = True
    with pytest.raises(OverflowError) as raised:
        KernMD().fit(np.array([[-1.], [1.]] * 2), y)
    assert raised.value is marker and calls == [True]


@pytest.mark.parametrize("hook", ["kind", "_take_with_fill", "__instancecheck__"])
def test_temporal_category_runs_selected_sparse_take_and_type_hooks(hook):
    marker = RuntimeError(hook)
    armed = False
    calls = []

    def observe(value, *args, **kwargs):
        if armed:
            calls.append(hook)
            raise marker
        return inherited(value, *args, **kwargs)

    base = pd.arrays.SparseArray
    if hook == "__instancecheck__":
        inherited = type(base).__instancecheck__
        meta = type("Meta", (type(base),), {hook: observe})
        levels_type = meta("Levels", (base,), {})
    else:
        inherited = base.kind.fget if hook == "kind" else base._take_with_fill
        levels_type = type("Levels", (base,), {hook: property(observe) if hook == "kind" else observe})

    values = np.array([0, 2], dtype="datetime64[ns]")
    levels = levels_type(values, fill_value=values[0])
    y = pd.Categorical.from_codes([0, 1] * 2, categories=pd.Index(levels))
    armed = True
    with pytest.raises(RuntimeError) as public:
        np.asarray(y)
    count = len(calls)
    calls.clear()
    with pytest.raises(RuntimeError) as actual:
        KernMD().fit(np.array([[-1.], [1.]] * 2), y)
    assert actual.value is public.value is marker and len(calls) == count


@pytest.mark.parametrize("hook", ["shape", "_readonly"])
@pytest.mark.parametrize("case", ["occupied fine fill", "unused fine fill", "ordinary gaps", "ordinary points"])
@pytest.mark.parametrize("frame", [False, True])
def test_temporal_category_observes_only_selected_returned_row_getters(hook, case, frame):
    calls = []
    armed = False
    marker = OverflowError("Overflow when converting between datetime64 units")
    base = pd.arrays.SparseArray

    def read(self):
        if armed:
            calls.append(hook)
            raise marker
        return base.shape.fget(self) if hook == "shape" else False

    levels_type = type("Levels", (base,), {hook: property(read)})
    if case == "unused fine fill":
        epoch = 1_577_836_800_000_000_000
        ticks = [0, epoch + 1, epoch + 2]
    else:
        ticks = [0, 2, 5]
    values = np.array(ticks, dtype="datetime64[ns]")
    levels = levels_type(values, fill_value=values[0])
    if "fine" in case:
        levels.fill_value = np.datetime64(1, "ps")
    codes = ([0, 1] if case in ("occupied fine fill", "ordinary gaps") else [1, 2]) * 2
    y = pd.Categorical.from_codes(codes, categories=pd.Index(levels))
    if frame:
        y = pd.DataFrame({"y": y}, copy=False)
    exact = np.array([1, 2000] * 2, dtype="datetime64[ps]") if case == "occupied fine fill" else values[codes]
    X = np.array([[-1.], [1.]] * 2)
    model = KernMD().fit(X, exact)
    monitor = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    armed = True
    try:
        np.asarray(y)
    except OverflowError as error:
        # Native construction can overflow before returning any sparse rows.
        assert error is marker or (case == "unused fine fill" and not calls)
    expected_calls = calls.copy()
    actions = (lambda: KernMD().fit(X, y).predict(X),
               lambda: model.score(X, y, sample_weight=[1, 2, 4, 8]),
               lambda: [row["score"] for row in monitor.fit(X, exact, validation_data=(X, y)).validation_history_])
    for action, expected in zip(actions, (exact, 1., [1., 1.])):
        calls.clear()
        if expected_calls:
            with pytest.raises(OverflowError) as raised:
                action()
            assert raised.value is marker
        else:
            np.testing.assert_array_equal(action(), expected)
        assert calls == expected_calls


def test_sparse_temporal_category_with_missing_code_and_unused_nat_fill():
    class Levels(pd.arrays.SparseArray):
        pass

    values = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    levels = Levels(values, fill_value=pd.NaT)
    assert not levels.sp_index.ngaps
    y = pd.Categorical.from_codes([0, 1, -1, 1], categories=pd.Index(levels))
    X = np.array([[-1.], [1.]] * 2)
    assert KernMD().fit(X, values[[0, 1] * 2]).score(X, y) == .75


@pytest.mark.parametrize("kind", ["datetime", "timedelta"])
@pytest.mark.parametrize("family", ["pandas", "python"])
@pytest.mark.parametrize("far", [False, True])
@pytest.mark.parametrize("frame", [False, True])
def test_sparse_temporal_category_with_boxed_fill_and_nanosecond_points(kind, family, far, frame):
    class Levels(pd.arrays.SparseArray):
        pass

    if kind == "datetime":
        year = 2300 if far else 2000
        seconds = int(np.datetime64(f"{year}-01-01", "s").view("i8"))
        fill = pd.Timestamp(f"{year}-01-01") if family == "pandas" else datetime(year, 1, 1)
    else:
        seconds = 12_000_000_000 if far else 10
        fill = pd.Timedelta(np.timedelta64(seconds, "s")) if family == "pandas" else timedelta(seconds=seconds)
    ns = np.array([0, 1_577_836_800_000_000_001, 1_577_836_800_000_000_002], dtype=f"{kind}64[ns]")
    levels = Levels(ns, fill_value=ns[0])
    try:
        levels.fill_value = fill
        catalogue = pd.Index(levels)
        unused = pd.Categorical.from_codes([1, 2] * 2, categories=catalogue)
    except (OverflowError, ValueError):
        if not far:
            raise
        pytest.skip("pandas rejects this out-of-range fill during construction")
    occupied = pd.Categorical.from_codes([0, 2] * 2, categories=catalogue)
    X = np.array([[-1.], [1.]] * 2)
    # Both selected counts fit in 2 ns, even when the fill overflows native ns.
    reference = np.array([seconds * 500_000_000, int(ns[2].view("i8")) // 2] * 2, dtype=f"{kind}64[2ns]")
    for y, exact in ((unused, ns[[1, 2] * 2]), (occupied, reference)):
        if frame:
            y = pd.DataFrame({"y": y}, copy=False)
        model = KernMD().fit(X, y)
        assert model.score(X, exact, sample_weight=[1, 2, 4, 8]) == 1.
        assert KernMD().fit(X, exact).score(X, y, sample_weight=[1, 2, 4, 8]) == 1.
        monitor = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
        history = monitor.fit(X, exact, validation_data=(X, y)).validation_history_
        assert [row["score"] for row in history] == [1., 1.]


@pytest.mark.parametrize("custom", ["scalar", "timezone"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("subclass", [False, True])
def test_sparse_temporal_category_does_not_repeat_custom_fill_callbacks(custom, fail, subclass):
    calls = []
    armed = False
    marker = OverflowError("Overflow when converting between datetime64 units")

    def record():
        if armed:
            calls.append(True)
            if fail:
                raise marker

    class Day(datetime):
        __hash__ = datetime.__hash__

        def __ne__(self, other):
            record()
            return super().__ne__(other)

        def __eq__(self, other):
            record()
            return super().__eq__(other)

    class Zone(tzinfo):
        def utcoffset(self, dt):
            record()
            return timedelta(0)

        def dst(self, dt):
            return timedelta(0)

    class Levels(pd.arrays.SparseArray):
        pass

    fill = Day(1970, 1, 1) if custom == "scalar" else datetime(1970, 1, 1, tzinfo=Zone())
    ns = np.array([0, 2, 5], dtype="datetime64[ns]")
    levels = (Levels if subclass else pd.arrays.SparseArray)(ns, fill_value=ns[0])
    try:
        levels.fill_value = fill
        y = pd.Categorical.from_codes([1, 2] * 2, categories=pd.Index(levels))
    except (ValueError, TypeError):
        pytest.skip("pandas rejects this custom fill during construction")
    armed = True
    try:
        public = np.asarray(y)
    except OverflowError as exc:
        assert exc is marker
        public = None
    expected_calls = calls.copy()
    calls.clear()
    X = np.array([[-1.], [1.]] * 2)
    if public is None:
        with pytest.raises(OverflowError) as actual:
            KernMD().fit(X, y)
        assert actual.value is marker
    else:
        np.testing.assert_array_equal(KernMD().fit(X, y).predict(X), public)
    assert calls == expected_calls


def test_temporal_frame_uses_a_declared_projection_and_rejects_an_ambiguous_dynamic_one():
    ids = np.array([0, 1] * 2, dtype="datetime64[ns]")
    sparse = pd.arrays.SparseArray(ids, fill_value=ids[0])
    sparse.fill_value = np.datetime64(1, "ps")
    X = np.array([[-1.], [1.]] * 2)
    model = KernMD().fit(X, ids)
    reads = []

    class Explicit(pd.DataFrame):
        def __array__(self, dtype=None, copy=None):
            reads.append(True)
            return ids[:, None].astype(dtype or ids.dtype, copy=False)

    class Dynamic(pd.DataFrame):
        def __getattribute__(self, name):
            return super().__getattribute__(name)

    assert model.score(X, Explicit({"y": sparse})) == 1. and reads == [True]
    with pytest.raises(ValueError, match="changes a stored class ID"):
        model.score(X, Dynamic({"y": sparse}))


@pytest.mark.parametrize("effect", ["error", "change"])
def test_temporal_category_runs_the_selected_sparse_row_constructor_and_conversion(effect):
    armed = False
    calls = []
    marker = RuntimeError("active temporal row shape")

    class Levels(pd.arrays.SparseArray):
        @property
        def shape(self):
            if armed and effect == "error":
                calls.append("shape")
                raise marker
            return super().shape

        def __setattr__(self, name, value):
            if armed and effect == "change" and name == "_sparse_values":
                calls.append("setter")
                value = value + np.timedelta64(5, "ns")
            super().__setattr__(name, value)

    levels = np.array([0, 2], dtype="datetime64[ns]")
    storage = Levels(levels, fill_value=levels[0])
    y = pd.Categorical.from_codes([0, 1] * 2, categories=pd.Index(storage))
    X = np.array([[-1.], [1.]] * 2)
    armed = True
    if effect == "error":
        with pytest.raises(RuntimeError) as public:
            np.asarray(y)
        count = len(calls)
        calls.clear()
        with pytest.raises(RuntimeError) as actual:
            KernMD().fit(X, y)
        assert actual.value is public.value is marker and len(calls) == count
    else:
        represented = np.asarray(y)
        count = len(calls)
        calls.clear()
        model = KernMD().fit(X, y)
        np.testing.assert_array_equal(model.predict(X), represented)
        assert count == len(calls)


@pytest.mark.parametrize("storage", ["int_points", "int_fill"])
def test_validation_membership_uses_the_occupied_sparse_ids(storage):
    X = np.tile([[-1.], [1.]], (4, 1))
    train = np.array([0, BASE] * 4, dtype=np.int64)
    held = pd.Series(sparse_labels([0, BASE + 1], storage))
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, train, validation_data=(X[:2], held))
    train[1::2] += 1
    model.fit(X, train, validation_data=(X[:2], held))
    assert [row["score"] for row in model.validation_history_] == [1., 1.]


@pytest.mark.parametrize("sparse", [
    lambda: pd.arrays.SparseArray(np.array([0, 3], dtype="int64"), fill_value=0.),
    lambda: pd.arrays.SparseArray(np.array([0., 3.25]), fill_value=0),
])
def test_ordinary_mixed_sparse_target_keeps_native_projection(sparse):
    y = sparse()
    actual = np.asarray(exact_sparse_uint64_target(y))
    expected = np.asarray(y)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_unused_fill_same_dtype_and_native_array_keep_fast_paths():
    native = np.array([BASE, BASE + 1], dtype=np.int64)
    assert exact_sparse_uint64_target(native) is native
    for y in (pd.Series(pd.arrays.SparseArray(native, fill_value=0.)),
              pd.Series(pd.arrays.SparseArray(np.append(native, 0), fill_value=np.int64(0)))):
        assert exact_sparse_uint64_target(y) is y
        assert [int(i) for i in np.asarray(y)] == [int(y.iloc[i]) for i in range(len(y))]


@pytest.mark.parametrize(("dtype", "fill", "ids"), [
    (np.int64, 0, [-(2**63), 0]),
    (np.int64, BASE, [BASE, BASE + 1]),
    (np.int64, None, [BASE, BASE + 1]),
    (np.uint64, 0, [0, 2**64 - 1]),
    (np.uint64, 2**64 - 2, [2**64 - 2, 2**64 - 1]),
])
def test_native_sparse_frame_preserves_binary_classes_and_weighted_score(dtype, fill, ids):
    def frame(codes):
        values = np.asarray(ids, dtype=dtype)[codes]
        if fill is None:
            return pd.DataFrame({"y": values}).astype("Sparse[int64]")
        return pd.DataFrame({"y": pd.arrays.SparseArray(values, fill_value=dtype(fill))})

    X = np.array([[-2.], [2.], [-1.], [1.]])
    y = frame([0, 1, 0, 1])
    normalized = exact_sparse_uint64_target(y)
    assert normalized.shape == (4, 1) and normalized.dtype == dtype
    if fill is None:
        assert np.shares_memory(normalized, y["y"].array.sp_values)
    with pytest.warns(DataConversionWarning):
        model = KernMD().fit(X, y)
    assert list(map(int, model.classes_)) == ids
    assert list(map(int, model.predict(X))) == ids * 2
    assert model.score(X, frame([0, 0, 0, 1]), sample_weight=[1, 2, 4, 8]) == 13 / 15


def test_occupied_nan_in_native_sparse_frame_reaches_label_validation():
    y = pd.DataFrame({"y": pd.arrays.SparseArray([0., np.nan, 0., 1.], fill_value=np.float64(0))})
    with pytest.warns(DataConversionWarning), pytest.raises(ValueError, match="NaN"):
        KernMD().fit(np.eye(4), y)


def test_sparse_frame_keeps_a_wider_float_fill_distinct_from_its_points():
    point = 2**24
    array = pd.arrays.SparseArray(np.array([0, point] * 2, dtype=np.float32), fill_value=np.float32(0))
    array.fill_value = float(point + 1)
    y = pd.DataFrame({"y": array})
    X = np.array([[-2.], [2.], [-1.], [1.]])
    with pytest.warns(DataConversionWarning):
        model = KernMD().fit(X, y)
    assert list(model.classes_) == [point, point + 1]
    assert list(model.predict(X)) == [point + 1, point] * 2


def test_numeric_sparse_frame_considers_text_fill_only_at_occupied_gaps():
    def frame(values):
        array = pd.arrays.SparseArray(np.array(values, dtype=np.int64), fill_value=np.int64(0))
        try:
            array.fill_value = str(BASE)
        except ValueError:
            pytest.skip("pandas requires the fill to match the numeric subtype")
        return pd.DataFrame({"y": array})

    X = np.array([[-2.], [2.], [-1.], [1.]])
    with pytest.raises((TypeError, ValueError)):
        KernMD().fit(X, frame([0, BASE, BASE + 1, 0]))
    with pytest.warns(DataConversionWarning):
        model = KernMD().fit(X, frame([BASE, BASE + 1] * 2))
    assert list(map(int, model.predict(X))) == [BASE, BASE + 1] * 2


def test_all_gap_sparse_float_with_integer_fill_recovers_id():
    y = sparse_labels([BASE + 1] * 3, "int_fill")
    assert not y.npoints
    assert list(map(int, exact_sparse_uint64_target(y))) == [BASE + 1] * 3


@pytest.mark.parametrize("high", [2**53 + 1, -(2**53 + 1), 2**63 + 1])
def test_sparse_bool_points_keep_large_integer_fill(high):
    fill = np.uint64(high) if high > 2**63-1 else np.int64(high)
    try:
        y = pd.Series(pd.arrays.SparseArray([False, True]*2, dtype=bool, fill_value=False))
        y.array.fill_value = fill
    except ValueError as error:
        pytest.skip(f"pandas rejects this sparse fill at construction: {error}")
    ids = [int(y.iloc[i]) for i in range(4)]
    assert ids == [high, 1] * 2
    X = np.array([[-2.], [2.], [-1.], [1.]])
    model = KernMD().fit(X, y)
    assert list(map(int, model.predict(X))) == ids
    assert model.score(X, y) == 1.


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("categorical", [False, True])
def test_sparse_custom_fill_preserves_its_stored_integer(dtype, categorical):
    class Redirect(int):
        def __int__(self):
            return 10

    y = pd.arrays.SparseArray([0, 20] if categorical else [0, 20]*2,
                              dtype=dtype, fill_value=0)
    y.fill_value = Redirect(11)
    if categorical:
        y = pd.Categorical.from_codes([0, 1]*2, categories=pd.Index(y))
    target = pd.Series(y)
    X = np.array([[-2.], [2.], [-1.], [1.]])
    model = KernMD().fit(X, target)
    assert [int.__int__(v) if type(v) is Redirect else int(v)
            for v in model.predict(X)] == [11, 20]*2
    assert model.score(X, target) == 1.


def test_inherited_sparse_storage_and_explicit_array_representation():
    class Inherited(pd.arrays.SparseArray):
        pass

    class Projected(Inherited):
        def __array__(self, dtype=None, copy=None):
            self.reads += 1
            return np.asarray([10, 20]*2, dtype=dtype)

    X = np.array([[-2.], [2.], [-1.], [1.]])
    original = Inherited(np.array([0, BASE+1]*2, dtype=np.int64), fill_value=0.)
    model = KernMD().fit(X, original)
    assert list(map(int, model.predict(X))) == [0, BASE+1]*2
    target = Projected(np.array([1, 2]*2), fill_value=0.)
    target.reads = 0
    assert KernMD().fit(X, target).classes_.tolist() == [10, 20]
    assert target.reads == 1


@pytest.mark.parametrize("outer", [True, False])
@pytest.mark.parametrize("name", ["__array__", "__array_interface__", "__array_struct__"])
def test_explicit_sparse_projection_takes_precedence_and_propagates_errors(name, outer):
    class ProjectionError(Exception):
        pass

    data = np.array([10, 20]*2)
    calls = []
    def read(instance):
        calls.append(name)
        if len(calls) > 1:
            raise ProjectionError("projection failed")
        return getattr(data, name)

    base = pd.Series if outer else pd.arrays.SparseArray
    projected = type("Projected", (base,), {name: property(read)})
    raw = np.array([0, BASE+1]*2, dtype=np.int64)
    storage = pd.arrays.SparseArray(raw, fill_value=0.)
    target = projected(storage) if outer else projected(raw, fill_value=0.)
    X = np.array([[-2.], [2.], [-1.], [1.]])
    assert KernMD().fit(X, target).classes_.tolist() == [10, 20]
    assert calls == [name]
    with pytest.raises(ProjectionError, match="projection failed"):
        KernMD().fit(X, target)


def test_dynamic_sparse_representation_reads_available_interface_once():
    rows = np.array([10, 20]*2)
    calls = []
    class Dynamic(pd.arrays.SparseArray):
        def __getattr__(self, name):
            if name == "__array_interface__":
                calls.append(name)
                return rows.__array_interface__
            raise AttributeError(name)

    raw = np.array([0, BASE+1]*2, dtype=np.int64)
    target = Dynamic(raw, fill_value=0.)
    fitted = KernMD().fit(np.eye(4), target)
    assert calls == ["__array_interface__"]
    assert fitted.classes_.tolist() == [10, 20]


@pytest.mark.parametrize("delegate", [False, True])
def test_dynamic_sparse_frame_conversion_and_exact_stored_labels(delegate):
    stored_id = BASE + 1
    frame_id = 2**63 + 1
    reads = []

    class Sparse(pd.arrays.SparseArray):
        def __getattribute__(self, name):
            if name == "to_numpy" and delegate:
                reads.append(name)

                def to_numpy(dtype=None, copy=False, **kwargs):
                    return np.array([0, frame_id] * 2, dtype=np.uint64).astype(dtype or np.uint64)

                return to_numpy
            return super().__getattribute__(name)

    y = Sparse(np.array([0, stored_id] * 2, dtype=np.int64), fill_value=0.)
    X = np.array([[-2.], [2.], [-1.], [1.]])
    for target in (y, pd.Series(y), pd.Index(y)):
        reads.clear()
        assert list(map(int, KernMD().fit(X, target).classes_)) == [0, stored_id]
        assert not reads
    frame = pd.Series(y).to_frame()
    reads.clear()
    with pytest.warns(DataConversionWarning):
        model = KernMD().fit(X, frame)
    assert list(map(int, model.classes_)) == [0, frame_id if delegate else stored_id]
    assert len(reads) == int(delegate)


def test_categorical_projects_the_extension_returned_by_take():
    class Selected(pd.arrays.IntegerArray):
        def __array__(self, dtype=None, copy=None):
            # Missing codes are passed to take with pandas' usual fill.
            return np.array([30 if pd.isna(v) else int(v) for v in self], dtype=dtype)

    class Levels(pd.arrays.IntegerArray):
        def take(self, indices, *, allow_fill=False, fill_value=None):
            selected = super().take(indices, allow_fill=allow_fill, fill_value=fill_value)
            return Selected(selected.to_numpy(dtype=np.int64, na_value=0), selected.isna())

    levels = Levels(np.array([10, 20]), np.zeros(2, dtype=bool))
    y = pd.Categorical.from_codes([0, 1, -1, 0], categories=pd.Index(levels))
    assert np.asarray(y).tolist() == [10, 20, 30, 10]
    with pytest.raises(ValueError, match="exactly two classes"):
        KernMD().fit(np.eye(4), y)


@pytest.mark.parametrize("dtype", ["int64", "Int64", object])
def test_categorical_reads_levels_without_invoking_index_conversion(dtype):
    levels = pd.Index([0, BASE, BASE + 1], dtype=dtype)

    def unused(*args, **kwargs):
        raise AssertionError("categorical projection does not use Index.to_numpy")

    levels.to_numpy = unused
    y = pd.Categorical.from_codes([0, 1, 2, 0], categories=levels)
    assert list(map(int, np.asarray(y))) == [0, BASE, BASE + 1, 0]
    with pytest.raises(ValueError, match="exactly two classes"):
        KernMD().fit(np.eye(4), y)


def test_pandas_storage_recovery_uses_the_accessors_selected_by_the_carrier():
    class Series(pd.Series):
        @property
        def array(self):
            raise AssertionError("Series does not select its array accessor")

    class Frame(pd.DataFrame):
        @property
        def iloc(self):
            raise AssertionError("DataFrame does not select its iloc accessor")

    class Category(pd.Categorical):
        @property
        def codes(self):
            raise AssertionError("Categorical does not select its public codes accessor")

    values = pd.array([0, BASE, BASE + 1, 0], dtype="UInt64")
    class ProjectedFrame(Frame):
        @property
        def _values(self):
            return np.asarray(values)[:, None]

    projected = ProjectedFrame({"y": pd.array([0, BASE, BASE, 0], dtype="UInt64")})
    for y in (Series(values), Frame({"y": values}), Category(values), projected):
        assert list(map(int, np.asarray(y).ravel())) == [0, BASE, BASE + 1, 0]
        with pytest.warns(DataConversionWarning) if isinstance(y, Frame) else nullcontext():
            with pytest.raises(ValueError, match="exactly two classes"):
                KernMD().fit(np.eye(4), y)


@pytest.mark.parametrize("accessor", ["_values", "values"])
def test_dynamic_frame_uses_pandas_selected_values_once(accessor):
    reads = []
    failure = None
    labels = np.array([0, BASE, BASE + 1] * 2, dtype=np.uint64)
    stored = np.array([0, BASE, BASE] * 2, dtype=np.int64)

    class Frame(pd.DataFrame):
        def __getattribute__(self, key):
            if key == accessor:
                reads.append(key)
                if len(reads) > 1:
                    raise AssertionError("frame projection repeated")
                if failure is not None:
                    raise failure
                return labels[:, None]
            return super().__getattribute__(key)

    y = Frame({"y": pd.Categorical(stored)})
    reads.clear()
    visible = list(map(int, np.asarray(y).ravel()))
    selected = reads.copy()
    X = np.array([[-2.], [2.], [2.]] * 2)
    weights = [1, 2, 4] * 2
    fitted = KernMD().fit(X, stored)
    matches = [a == b for a, b in zip(map(int, fitted.predict(X)), visible)]
    reads.clear()
    assert fitted.score(X, y, sample_weight=weights) == np.average(matches, weights=weights)
    assert reads == selected
    if selected:
        reads.clear()
        with pytest.warns(DataConversionWarning), pytest.raises(ValueError, match="exactly two classes"):
            KernMD().fit(X, y)
        assert reads == selected
        reads.clear()
        failure = RuntimeError("frame values unavailable")
        with pytest.raises(RuntimeError) as caught:
            fitted.score(X, y)
        assert caught.value is failure and reads == selected


@pytest.mark.parametrize("category", [False, True])
@pytest.mark.parametrize("inherited", [False, True])
def test_ambiguous_dynamic_sparse_frame_requires_a_label_choice(category, inherited):
    class Passive(pd.DataFrame):
        def __getattribute__(self, key):
            return super().__getattribute__(key)

    class Fallback(pd.DataFrame):
        def __getattr__(self, key):
            return super().__getattr__(key)

    class Distinct(Passive):
        def __getattribute__(self, key):
            if key == "_values":
                return np.array([0, BASE + 3] * 2, dtype=np.uint64)[:, None]
            return super().__getattribute__(key)

    class Sparse(pd.arrays.SparseArray):
        pass

    class Category(pd.Categorical):
        pass

    ids = [0, BASE + 1] * 2
    sparse_cls = Sparse if inherited else pd.arrays.SparseArray
    categorical_cls = Category if inherited else pd.Categorical
    if category:
        levels = pd.Index(sparse_cls(np.array(ids[:2], dtype=np.int64), fill_value=0.))
        storage = categorical_cls.from_codes([0, 1] * 2, categories=levels)
    else:
        storage = sparse_cls(np.array(ids, dtype=np.int64), fill_value=0.)
    y = Passive({"y": storage})
    projected = np.asarray(y)
    assert list(map(int, projected[:, 0])) == [0, BASE] * 2
    X = np.array([[-2.], [2.], [-1.], [1.]])
    with pytest.raises(ValueError, match="changes a stored class ID"):
        KernMD().fit(X, y)
    with pytest.warns(DataConversionWarning):
        assert list(map(int, KernMD().fit(X, projected).classes_)) == [0, BASE]
    assert list(map(int, KernMD().fit(X, [storage[i] for i in range(4)]).classes_)) == ids[:2]
    with pytest.warns(DataConversionWarning):
        assert list(map(int, KernMD().fit(X, Fallback({"y": storage})).classes_)) == ids[:2]
    with pytest.warns(DataConversionWarning):
        assert list(map(int, KernMD().fit(X, Distinct({"y": storage})).classes_)) == [0, BASE + 3]


@pytest.mark.parametrize("high", [2**31, -(2**31) - 1, BASE, -(2**63), 2**63, 2**63 + 2048])
def test_lossless_sparse_frame_floats_remain_discrete_ids(high):
    class Passive(pd.DataFrame):
        def __getattribute__(self, key):
            return super().__getattribute__(key)

    X = np.array([[-2.], [2.], [-1.], [1.]])
    dtype = np.uint64 if high >= 2**63 else np.int64
    for frame in (pd.DataFrame, Passive):
        def target(ids):
            return frame({"y": pd.arrays.SparseArray(np.array(ids, dtype=dtype), fill_value=0.)})

        ids = [0, high] * 2
        y = target(ids)
        assert list(map(int, np.asarray(y).ravel())) == ids
        check_classification_targets(exact_sparse_uint64_target(y))
        with pytest.warns(DataConversionWarning):
            model = KernMD().fit(X, y)
        assert list(map(int, model.predict(X))) == ids
        assert model.score(X, target([0, high, high, 0]), sample_weight=[1, 3, 5, 7]) == .25


@pytest.mark.parametrize("dtype", [np.float32, np.longdouble,
                                  pytest.param(np.dtype(np.float64).newbyteorder("S"), id="swapped-float64")])
def test_wide_explicit_float_projection_only_normalizes_exact_integers(dtype):
    reads = []
    high = 2**63

    class Projected(pd.Series):
        def __array__(self, dtype=None, copy=None):
            reads.append(1)
            return values

    y = Projected(np.zeros(4))
    scalar = np.dtype(dtype).type
    next_id = max(high + 1, int(np.nextafter(scalar(high), scalar(np.inf))))
    for ids in ([high] * 4, [0, high] * 2, [0, 1, high, 0], [high, next_id] * 2):
        values = np.asarray(ids, dtype=np.uint64).astype(dtype)
        reads.clear()
        labels = exact_sparse_uint64_target(y)
        assert reads == [1] and list(map(int, labels)) == ids
        check_classification_targets(labels)
        if len(set(ids)) == 2:
            model = KernMD().fit(np.eye(4), y)
            assert list(map(int, model.classes_)) == sorted(set(ids))
            assert model.score(np.eye(4), values) == 1.
            if not np.dtype(dtype).isnative or np.finfo(dtype).nmant > np.finfo(np.float64).nmant:
                model.fit(np.eye(4), values)
                assert model.score(np.eye(4), values) == model.score(np.eye(4), y) == 1.
        else:
            with pytest.raises(ValueError, match="exactly two classes"):
                KernMD().fit(np.eye(4), y)
    for other in (0.5, np.nan, np.inf, -np.inf):
        values = np.array([high, other] * 2, dtype=dtype)
        reads.clear()
        labels = exact_sparse_uint64_target(y)
        assert reads == [1] and labels.dtype == values.dtype
        np.testing.assert_array_equal(labels, values)
        for target in (y, values):
            reads.clear()
            if np.isfinite(other):
                classes = KernMD().fit(np.eye(4), target).classes_
                assert classes[0] == other and int(classes[1]) == high
            else:
                with pytest.raises(ValueError, match="NaN|infinity"):
                    KernMD().fit(np.eye(4), target)
            assert reads == ([1] if target is y else [])


@pytest.mark.parametrize("base, field", [(pd.Series, "_values"), (pd.Index, "_data")])
def test_outer_vector_reads_its_selected_backing_values_once(base, field):
    reads = []
    data = np.array([0, BASE + 1] * 2, dtype=np.uint64)

    class Labels(base):
        def __getattribute__(self, key):
            if key == field:
                reads.append(key)
                if len(reads) > 1:
                    raise AssertionError("vector projection repeated")
                return data
            return super().__getattribute__(key)

    original = np.array([10, 20] * 2)
    y = Labels(original) if base is pd.Series else Labels._simple_new(original)
    X = np.array([[-2.], [2.], [-1.], [1.]])
    reads.clear()
    model = KernMD().fit(X, y)
    assert list(map(int, model.predict(X))) == list(map(int, data)) and reads == [field]


@pytest.mark.parametrize("storage", [
    pd.period_range("2020-01", periods=4, freq="M").array,
    pd.date_range("2020-01-01", periods=4).array,
    pd.timedelta_range("1D", periods=4).array,
])
@pytest.mark.parametrize("carrier", ["series", "index", "frame", "categorical"])
def test_temporal_extension_uses_the_public_carriers_conversion_once(storage, carrier):
    reads = []

    class NumericLabels(type(storage)):
        def __getattribute__(self, name):
            if name in ("__array__", "to_numpy"):
                def project(dtype=None, *args, **kwargs):
                    reads.append(name)
                    high = BASE + 1 if name == "__array__" else BASE
                    return np.asarray([0, high] * 2, dtype=dtype or np.uint64).reshape(self.shape)
                return project
            return super().__getattribute__(name)

    values = NumericLabels._simple_new(storage._ndarray, dtype=storage.dtype)
    if carrier == "categorical":
        y = pd.Categorical.from_codes([0, 1, 2, 3], categories=pd.Index(values))
    else:
        y = pd.Index(values) if carrier == "index" else pd.Series(values)
    if carrier == "frame":
        y = y.to_frame()
    reads.clear()
    expected = list(map(int, np.asarray(y).ravel()))
    methods = reads.copy()
    assert len(methods) == 1
    X = np.array([[-2.], [2.], [-1.], [1.]])
    reads.clear()
    with pytest.warns(DataConversionWarning) if carrier == "frame" else nullcontext():
        model = KernMD().fit(X, y)
    assert list(map(int, model.predict(X))) == expected
    assert reads == methods
    reads.clear()
    assert model.score(X, y) == 1.
    assert reads == methods


def test_legacy_array_manager_casts_before_selecting_frame_labels():
    try:
        pd.get_option("mode.data_manager")
    except pd.errors.OptionError:
        pytest.skip("pandas does not provide ArrayManager")

    class Dynamic(pd.arrays.IntegerArray):
        def __getattribute__(self, name):
            if name == "__array__":
                raise AssertionError("the frame does not use the inner __array__")
            return super().__getattribute__(name)

    class Cast(pd.arrays.IntegerArray):
        def astype(self, dtype, copy=True):
            return np.array([0, BASE, BASE + 1, 0], dtype=dtype)

    class Levels(pd.arrays.IntegerArray):
        def astype(self, dtype, copy=True):
            return np.array([0, BASE + 1], dtype=dtype)

    class IteratedLevels(pd.arrays.IntegerArray):
        astype = pd.api.extensions.ExtensionArray.astype

        @property
        def __array__(self):
            if armed:
                raise AttributeError("use sequence")
            return super().__array__

        def __iter__(self):
            return (0. if int(v) == 0 else v for v in super().__iter__())

    with pd.option_context("mode.data_manager", "array"):
        raw = np.array([0, BASE, BASE, 0])
        mask = np.zeros(len(raw), dtype=bool)
        frame = pd.DataFrame({"y": Dynamic(raw, mask)})
        with pytest.warns(DataConversionWarning):
            model = KernMD().fit(np.eye(4), frame)
        assert list(map(int, model.classes_)) == [0, BASE]
        frame = pd.DataFrame({"y": Cast(raw, mask)})
        projected = np.asarray(frame)
        assert projected.dtype == object and list(projected[:, 0]) == [0, BASE, BASE + 1, 0]
        labels = np.asarray(exact_sparse_uint64_target(frame))
        assert labels.dtype.kind in "iu"
        np.testing.assert_array_equal(labels, projected)
        with pytest.warns(DataConversionWarning), pytest.raises(ValueError, match="exactly two classes"):
            KernMD().fit(np.eye(4), frame)
        levels = Levels(np.array([0, BASE]), np.zeros(2, dtype=bool))
        category = pd.Categorical.from_codes([0, 1] * 2, categories=pd.Index(levels))
        frame = pd.DataFrame({"y": category})
        with pytest.warns(DataConversionWarning):
            model = KernMD().fit(np.eye(4), frame)
        assert list(map(int, model.classes_)) == [0, BASE + 1]
        # The cast iterates the catalogue before pandas applies the codes,
        # even when the catalogue and the observation vector have equal sizes.
        armed = False
        levels = IteratedLevels(np.array([0, BASE, BASE + 1], dtype=np.uint64), np.zeros(3, dtype=bool))
        labels = pd.Categorical.from_codes([1, 2, 0], categories=pd.Index(levels))
        frame = pd.DataFrame({"y": labels})
        armed = True
        actual = exact_sparse_uint64_target(frame)
        assert list(map(int, np.asarray(actual).ravel())) == [BASE, BASE + 1, 0]


@pytest.mark.parametrize("point", [1., np.inf, -np.inf])
def test_wide_sparse_fill_preserves_finite_target_validation(point):
    y = pd.arrays.SparseArray([0., point]*2, fill_value=0.)
    y.fill_value = 2**64
    target = pd.Series(y)
    X = np.array([[-2.], [2.], [-1.], [1.]])
    if np.isfinite(point):
        model = KernMD().fit(X, target)
        assert list(map(int, model.predict(X))) == [2**64, 1]*2
    else:
        with pytest.raises(ValueError, match="NaN or infinity"):
            KernMD().fit(X, target)


def extension_target(cls, stored, carrier):
    values = np.array(stored, dtype=np.int64)
    if issubclass(cls, pd.Categorical):
        array = cls(values)
    else:
        array = cls(values, np.zeros(len(values), dtype=bool))
    if carrier == "series":
        return pd.Series(array)
    if carrier == "index":
        return pd.Index(array)
    if carrier == "frame":
        return pd.DataFrame({"y": array})
    return array


def declared_extension(base, carrier, stored, represented):
    reads = []

    class Declared(base):
        def __array__(self, dtype=None, copy=None):
            reads.append(None)
            if isinstance(represented, Exception):
                raise represented
            result = np.asarray(represented, dtype=np.uint64)
            return result if dtype is None else result.astype(dtype, copy=False)

    target = extension_target(Declared, stored, carrier)
    reads.clear()
    return target, reads


@pytest.mark.parametrize("base", [pd.Categorical, pd.arrays.IntegerArray])
@pytest.mark.parametrize("carrier", ["direct", "series", "index", "frame"])
def test_declared_extension_controls_binary_fit_and_class_count(base, carrier):
    # A frame can choose the integer array's to_numpy storage even when the
    # direct array exposes different labels. Measure the supplied carrier.
    for third_label in (BASE + 1, BASE):
        stored = [0, BASE, BASE] * 2
        declared = [0, BASE + 1, third_label] * 2
        target, reads = declared_extension(base, carrier, stored, declared)
        visible = list(map(int, np.asarray(target).ravel()))
        expected_reads = reads.copy()
        reads.clear()
        if carrier != "frame" or base is pd.Categorical:
            assert visible == declared
        classes = sorted(set(visible))
        X = np.array([[-3.], [1.], [2.]] * 2)
        with pytest.warns(DataConversionWarning) if carrier == "frame" else nullcontext():
            if len(classes) == 3:
                with pytest.raises(ValueError, match="exactly two classes"):
                    KernMD().fit(X, target)
            else:
                model = KernMD().fit(X, target)
                assert list(map(int, model.classes_)) == classes
                assert list(map(int, model.predict(X))) == visible
        assert reads == expected_reads


@pytest.mark.parametrize("base", [pd.Categorical, pd.arrays.IntegerArray])
@pytest.mark.parametrize("carrier", ["direct", "series", "index", "frame"])
def test_declared_extension_controls_score_and_monitoring(base, carrier):
    X = np.array([[-2.], [2.], [-1.], [1.]])
    train = np.array([0, BASE] * 2, dtype=np.uint64)
    model = KernMD().fit(X, train)
    predicted = list(map(int, model.predict(X)))
    weights = [1, 2, 3, 4]
    for last_label in (0, BASE + 1):
        target, reads = declared_extension(base, carrier, train, [0, BASE, 0, last_label])
        visible = list(map(int, np.asarray(target).ravel()))
        expected_reads = reads.copy()
        correct = [a == b for a, b in zip(predicted, visible)]
        reads.clear()
        assert model.score(X, target) == sum(correct) / len(correct)
        assert reads == expected_reads
        reads.clear()
        assert model.score(X, target, sample_weight=weights) == sum(
            w for w, match in zip(weights, correct) if match) / sum(weights)
        assert reads == expected_reads
        reads.clear()
        monitor = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
        with pytest.warns(DataConversionWarning) if carrier == "frame" else nullcontext():
            if BASE + 1 in visible:
                with pytest.raises(ValueError, match="Validation labels must belong"):
                    monitor.fit(X, train, validation_data=(X, target))
            else:
                monitor.fit(X, train, validation_data=(X, target))
                scores = [row["score"] for row in monitor.validation_history_]
                assert scores == [sum(correct) / len(correct)] * 2
        assert reads == expected_reads


@pytest.mark.parametrize("base", [pd.Categorical, pd.arrays.IntegerArray])
@pytest.mark.parametrize("carrier", ["direct", "series"])
def test_declared_extension_preserves_projection_error(base, carrier):
    error = RuntimeError("declared extension projection failed")
    X = np.array([[-2.], [2.], [-1.], [1.]])
    labels = np.array([0, BASE] * 2, dtype=np.uint64)
    target, reads = declared_extension(base, carrier, labels, error)
    model = KernMD().fit(X, labels)
    monitor = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    for run in (lambda: KernMD().fit(X, target), lambda: model.score(X, target),
                lambda: monitor.fit(X, labels, validation_data=(X, target))):
        reads.clear()
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is error
        assert len(reads) == 1


@pytest.mark.parametrize("base", [pd.Categorical, pd.arrays.IntegerArray])
@pytest.mark.parametrize("inherited", [False, True])
def test_ordinary_extension_preserves_adjacent_ids(base, inherited):
    class Inherited(base):
        pass

    ids = [BASE, BASE + 1] * 2
    y = extension_target(Inherited if inherited else base, ids, "series")
    X = np.array([[-2.], [2.], [-1.], [1.]])
    model = KernMD().fit(X, y)
    assert list(map(int, model.classes_)) == ids[:2]
    assert list(map(int, model.predict(X))) == ids
    assert model.score(X, y) == 1.


@pytest.mark.parametrize("base", [pd.arrays.SparseArray, pd.arrays.IntegerArray])
def test_categorical_projects_selected_extension_rows_in_observation_order(base):
    reads = []

    class ProjectedLevels(base):
        def __array__(self, dtype=None, copy=None):
            reads.append(len(self))
            # The projection operates on the selected observations. Applying
            # it to the three levels and then indexing gives different labels.
            values = [BASE + int(value) + 2 * (i % 2) for i, value in enumerate(self)]
            return np.array(values, dtype=np.uint64 if dtype is None else dtype)

    stored = np.array([10, 11, 30], dtype=np.int64)
    if base is pd.arrays.SparseArray:
        levels = ProjectedLevels(stored, fill_value=0)
    else:
        levels = ProjectedLevels(stored, np.zeros(len(stored), dtype=bool))
    target = pd.Categorical.from_codes([1, 0, 1, 0], categories=pd.Index(levels))
    reads.clear()
    visible = list(map(int, np.asarray(target)))
    assert visible == [BASE + 11, BASE + 12] * 2
    assert reads == [4]
    reads.clear()
    X = np.array([[-2.], [2.], [-1.], [1.]])
    model = KernMD().fit(X, target)
    assert list(map(int, model.predict(X))) == visible
    assert reads == [4]
    reads.clear()
    assert model.score(X, target) == 1.
    assert reads == [4]


@pytest.mark.parametrize("carrier", ["direct", "series", "frame"])
def test_nullable_to_numpy_projection_retains_unsigned_ids(carrier):
    ids = [2**63, 2**63 + 1] * 2

    class UnsignedLabels(pd.arrays.IntegerArray):
        def to_numpy(self, dtype=None, copy=False, **options):
            values = np.array(ids, dtype=np.uint64)
            return values if dtype is None else values.astype(dtype, copy=False)

    target = extension_target(UnsignedLabels, [0, 1] * 2, carrier)
    assert list(map(int, np.asarray(target).ravel())) == ids
    X = np.array([[-2.], [2.], [-1.], [1.]])
    with pytest.warns(DataConversionWarning) if carrier == "frame" else nullcontext():
        model = KernMD().fit(X, target)
    assert list(map(int, model.classes_)) == ids[:2]
    assert list(map(int, model.predict(X))) == ids
    assert model.score(X, target) == 1.
