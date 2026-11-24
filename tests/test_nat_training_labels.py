"""Missing temporal rows cannot define a training class."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from dwd.gen_kern_dwd import KernGDWD
from dwd.kern_md import KernMD, kern_md
from dwd.utils import pm1


X = np.array([[-2.], [2.], [-1.], [1.]])


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
@pytest.mark.parametrize("model_type", [KernMD, KernGDWD])
def test_missing_training_label_invalidates_a_previous_fit(dtype, model_type):
    y = np.array([10, 20, 10, 20], dtype=dtype)
    model = (KernGDWD(initialization="zero", max_iter=2)
             if model_type is KernGDWD else KernMD())
    model.fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)
    y[::2] = "NaT"
    with pytest.raises(ValueError, match="NaT"):
        model.fit(X, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@pytest.mark.parametrize("dtype", ["datetime64[ms]", "timedelta64[ms]"])
@pytest.mark.parametrize("storage", ["gap", "point", "all_missing", "category", "sparse_category"])
def test_missing_observed_pandas_temporal_rows(dtype, storage):
    pd = pytest.importorskip("pandas")
    labels = np.array([10, 20, 10, 20], dtype=dtype)
    missing = labels.copy()
    missing[::2] = "NaT"
    if storage in ("category", "sparse_category"):
        levels = labels[:2]
        if storage == "sparse_category":
            levels = pd.Index(pd.arrays.SparseArray(levels, fill_value=labels[0]))
        y = pd.Categorical.from_codes([-1, 1, -1, 1], categories=levels)
    elif storage == "all_missing":
        y = pd.arrays.SparseArray(np.full_like(labels, "NaT"), fill_value=pd.NaT)
    else:
        y = pd.arrays.SparseArray(missing, fill_value=pd.NaT if storage == "gap" else labels[1])
    with pytest.raises(ValueError, match="NaT"):
        KernMD().fit(X, y)
    with pytest.raises(ValueError, match="NaT"):
        kern_md(X @ X.T, y)
    with pytest.raises(ValueError, match="NaT"):
        pm1(y)


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
def test_unused_nat_fill_and_held_out_nat(dtype):
    pd = pytest.importorskip("pandas")
    y = np.array([10, 20, 10, 20], dtype=dtype)
    sparse = pd.arrays.SparseArray(y, fill_value=pd.NaT)
    assert sparse.sp_index.ngaps == 0
    model = KernMD().fit(X, sparse)
    np.testing.assert_array_equal(model.predict(X), y)
    truth = y.copy()
    truth[0] = "NaT"
    assert model.score(X, truth, sample_weight=[2, 1, 1, 1]) == 3 / 5
    monitored = KernGDWD(initialization="zero", stopping="validation", max_iter=2)
    with pytest.raises(ValueError, match="Validation labels must belong"):
        monitored.fit(X, sparse, validation_data=(X, truth))


def test_fit_uses_the_selected_temporal_projection_once():
    pd = pytest.importorskip("pandas")
    calls = []
    valid = np.array([10, 20, 10, 20], dtype="datetime64[ns]")
    missing = valid.copy()
    missing[::2] = "NaT"

    class Projected(pd.Series):
        def __array__(self, dtype=None, copy=None):
            calls.append(None)
            if len(calls) > 1:
                raise AssertionError("Repeated target conversion")
            return np.asarray(selected, dtype=dtype)

    selected = valid
    model = KernMD().fit(X, Projected(missing))
    np.testing.assert_array_equal(model.classes_, valid[:2])
    assert len(calls) == 1
    calls.clear()
    selected = missing
    with pytest.raises(ValueError, match="NaT"):
        model.fit(X, Projected(valid))
    assert len(calls) == 1


@pytest.mark.parametrize("box", [False, True])
def test_direct_helpers_reject_boxed_nat(box):
    pd = pytest.importorskip("pandas")
    y = np.empty(4, dtype=object)
    missing = pd.NaT
    if box:
        missing = np.empty((), dtype=object)
        missing[()] = np.array("NaT", dtype="datetime64[ns]")
    y[:] = [pd.Timestamp("2000-01-01"), missing] * 2
    with pytest.raises(ValueError, match="NaT"):
        kern_md(X @ X.T, y)
    with pytest.raises(ValueError, match="NaT"):
        pm1(y)


@pytest.mark.parametrize("box", ["passive", "forwarded", "masked", "strided", "masked_finite"])
def test_direct_helpers_reject_a_single_missing_temporal_box(box):
    class Passive(np.ndarray):
        pass

    class Forwarded(np.ndarray):
        def __eq__(self, other):
            return super().__eq__(other)

    def wrap(value, missing=False):
        if box == "passive":
            return np.array(value).view(Passive)
        if box == "forwarded":
            return np.array(value).view(Forwarded)
        if box == "strided":
            return np.array([value, value])[::-2]
        return np.ma.array(value, mask=missing and box == "masked_finite")

    labels = np.array([10, 20, 10, 20], dtype="timedelta64[ns]")
    y = np.empty(4, dtype=object)
    for i, value in enumerate(labels):
        y[i] = wrap(value)
    np.testing.assert_array_equal(pm1(y), [-1, 1, -1, 1])
    y[0] = wrap(labels[0] if box == "masked_finite" else np.timedelta64("NaT", "ns"), missing=True)
    with pytest.raises(ValueError, match="NaT"):
        pm1(y)
    with pytest.raises(ValueError, match="NaT"):
        kern_md(X @ X.T, y)


def test_box_with_its_own_membership_keeps_its_class_ids():
    class Label(np.ndarray):
        def __new__(cls, key):
            label = np.array("NaT", dtype="datetime64[ns]").view(cls)
            label.key = key
            return label

        def __eq__(self, other):
            return self.key == other.key

        def __ne__(self, other):
            return self.key != other.key

        def __lt__(self, other):
            return self.key < other.key

        def __gt__(self, other):
            return self.key > other.key

    y = np.empty(4, dtype=object)
    for i, key in enumerate([0, 1, 0, 1]):
        y[i] = Label(key)
    np.testing.assert_array_equal(pm1(y), [-1, 1, -1, 1])
    np.testing.assert_array_equal(kern_md(X @ X.T, y)[0], [-.5, .5, -.5, .5])


@pytest.mark.parametrize("y", [
    np.array(["NaT", "other"] * 2, dtype=object),
    np.array([np.iinfo(np.int64).min, 1] * 2, dtype=np.int64),
])
def test_non_temporal_labels_keep_their_class_membership(y):
    np.testing.assert_array_equal(KernMD().fit(X, y).predict(X), y)


@pytest.mark.parametrize("kind", ["pandas", "pandas_new", "datetime", "timedelta"])
def test_string_dtype_with_temporal_missing_sentinel(kind):
    string_dtype = getattr(np.dtypes, "StringDType", None)
    if string_dtype is None:
        pytest.skip("StringDType requires NumPy 2")
    pd = pytest.importorskip("pandas")
    sentinel = {"pandas": pd.NaT, "pandas_new": pd.api.typing.NaTType(), "datetime": np.datetime64("NaT", "ns"),
                "timedelta": np.timedelta64("NaT", "ns")}[kind]
    dtype = string_dtype(na_object=sentinel)
    valid = np.array(["NaT", "other"] * 2, dtype=dtype)
    model = KernMD().fit(X, valid)
    np.testing.assert_array_equal(model.predict(X), valid)
    assert np.isfinite(model.dual_coef_).all()
    missing = np.array([sentinel, "other"] * 2, dtype=dtype)
    with pytest.raises(ValueError, match="NaT"):
        model.fit(X, missing)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    with pytest.raises(ValueError, match="NaT"):
        pm1(missing)


def test_boxed_non_singleton_pandas_nat_cannot_define_a_class():
    pd = pytest.importorskip("pandas")
    missing = pd.api.typing.NaTType()
    assert missing is not pd.NaT and np.isnat(missing.asm8)
    time = pd.Timestamp("2000-01-01")
    y = np.array([missing, time, time, time], dtype=object)
    # Set the fill on an existing gap. Some pandas constructors first
    # infer timestamps and replace this distinct NaT object with year one.
    sparse = pd.arrays.SparseArray(np.array([0, time, time, time], dtype=object), fill_value=0)
    sparse.fill_value = missing
    assert sparse[0] is missing
    for labels in (y, sparse):
        with pytest.raises(ValueError, match="NaT"):
            kern_md(X @ X.T, labels)
        with pytest.raises(ValueError, match="NaT"):
            pm1(labels)


def test_reflexive_pandas_nat_subclass_can_define_a_boxed_class():
    pd = pytest.importorskip("pandas")

    class Label(pd.api.typing.NaTType):
        def __eq__(self, other):
            return self.key == other.key

        def __ne__(self, other):
            return self.key != other.key

        def __lt__(self, other):
            return self.key < other.key

        def __gt__(self, other):
            return self.key > other.key

    y = np.empty(4, dtype=object)
    for i, key in enumerate([0, 1, 0, 1]):
        y[i] = Label()
        y[i].key = key
    np.testing.assert_array_equal(pm1(y), [-1, 1, -1, 1])
    np.testing.assert_array_equal(kern_md(X @ X.T, y)[0], [-.5, .5, -.5, .5])


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
@pytest.mark.parametrize("carrier", ["native", "passive", "category"])
def test_non_singleton_nat_fill_in_temporal_sparse_storage(dtype, carrier):
    pd = pytest.importorskip("pandas")
    data = np.array([10, 20] * 2, dtype=dtype)
    sentinel = pd.api.typing.NaTType()

    class Passive(pd.arrays.SparseArray):
        pass

    def target(fill, missing):
        if carrier == "category":
            levels = pd.arrays.SparseArray(data[:2], fill_value=fill)
            y = pd.Categorical.from_codes([0, 1] * 2, categories=pd.Index(levels))
            sparse = y.categories.array
        else:
            cls = Passive if carrier == "passive" else pd.arrays.SparseArray
            y = sparse = cls(data, fill_value=fill)
        try:
            sparse.fill_value = missing
        except (TypeError, ValueError):
            pytest.skip("pandas rejects this NaT object in native temporal sparse storage")
        return y

    valid = target(np.array(0, dtype=dtype)[()], sentinel)
    model = KernMD().fit(X, valid)
    np.testing.assert_array_equal(model.classes_, data[:2])
    assert np.isfinite(model.dual_coef_).all()
    missing = target(data[0], sentinel)
    assert missing[0] is sentinel
    assert model.score(X, missing) == model.score(X, target(data[0], pd.NaT)) == .5
    monitored = KernGDWD(initialization="zero", stopping="validation", max_iter=2)
    with pytest.raises(ValueError, match="Validation labels must belong"):
        monitored.fit(X, valid, validation_data=(X, missing))
    with pytest.raises(ValueError, match="NaT"):
        model.fit(X, missing)


@pytest.mark.parametrize("category", [False, True])
def test_legacy_uint64_duration_nat_fill_must_be_selected(category):
    pd = pytest.importorskip("pandas")
    data = np.array([10, 20, 30] if category else [10, 20] * 2, dtype="uint64")
    sentinel = np.timedelta64("NaT", "ns")

    def target(select_fill):
        fill = 10 if select_fill or category else 0
        sparse = pd.arrays.SparseArray(data, fill_value=fill)
        if category:
            codes = [0, 1] * 2 if select_fill else [1, 2] * 2
            y = pd.Categorical.from_codes(codes, categories=pd.Index(sparse))
            sparse = y.categories.array
        else:
            y = sparse
        try:
            sparse.fill_value = sentinel
        except (TypeError, ValueError):
            pytest.skip("pandas rejects temporal fills in numeric sparse storage")
        return y

    valid = target(False)
    model = KernMD().fit(X, valid)
    np.testing.assert_array_equal(model.classes_, [20, 30] if category else [10, 20])
    assert np.isfinite(model.dual_coef_).all()
    with pytest.raises(ValueError, match="NaT"):
        model.fit(X, target(True))
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_occupied_temporal_missing_fill_in_legacy_numeric_sparse_target():
    pd = pytest.importorskip("pandas")
    y = pd.arrays.SparseArray(np.array([10, 20, 10, 20], dtype="int32"), fill_value=10)
    try:
        y.fill_value = np.datetime64("NaT", "ns")
    except (TypeError, ValueError):
        pytest.skip("pandas rejects temporal fills in numeric sparse storage")
    with pytest.raises(ValueError, match="NaT"):
        KernMD().fit(X, y)
