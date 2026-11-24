"""Sparse object targets preserve each observed NumPy temporal scalar's unit."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from dwd.gen_kern_dwd import KernGDWD
from dwd.kern_md import KernMD, kern_md
from dwd.utils import pm1


pd = pytest.importorskip("pandas")
X = np.array([[-1.], [1.]] * 2)
HIGH = 2**61 + 1  # Promoting this ns tick to ps can wrap onto 1000 ps.


def sparse_rows(values, *, fill=None, array_type=pd.arrays.SparseArray):
    # Create gaps before assigning their true fill. Comparing the IDs while
    # pandas constructs the sparse array can already discard a class.
    # The extra text row prevents automatic pandas date/duration inference.
    raw = np.array(["unused", *values], dtype=object)
    result = array_type(raw, dtype=pd.SparseDtype(object, "gap")).take(np.arange(1, len(raw)))
    if fill is not None:
        result.fill_value = fill
    return result


@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
@pytest.mark.parametrize("carrier", ["array", "series", "index", "frame"])
@pytest.mark.parametrize("storage", ["points", "gap", "category"])
def test_fit_helpers_score_and_monitor_preserve_mixed_units(scalar, carrier, storage):
    # Keep category levels distinct during pandas' own uniqueness check.
    # Direct sparse inputs also exercise the original collision.
    high = HIGH + 2 if storage == "category" else HIGH
    small, large = scalar(1000, "ps"), scalar(high, "ns")
    ids = np.array([1, high] * 2, dtype=large.dtype)

    def target(codes):
        if storage == "category":
            # The unused attosecond level cannot share the fitted grid.
            levels = sparse_rows(["unused", "gap", large, scalar(1, "as")], fill=small)
            y = pd.Categorical.from_codes(np.add(codes, 1), categories=pd.Index(levels))
        else:
            first = "gap" if storage == "gap" else small
            y = sparse_rows(([first, large][i] for i in codes),
                            fill=small if storage == "gap" else scalar("NaT", "as"))
        if carrier == "series":
            return pd.Series(y)
        if carrier == "index":
            return pd.Index(y)
        if carrier == "frame":
            return pd.Series(y).to_frame()
        return y

    y = target([0, 1] * 2)
    fitted = KernMD().fit(X, y)
    assert fitted.classes_.dtype == ids.dtype
    assert fitted.classes_.view("i8").tolist() == [1, high]
    assert fitted.predict(X).view("i8").tolist() == [1, high] * 2
    np.testing.assert_array_equal(pm1(y), [-1, 1] * 2)
    if carrier != "frame":
        np.testing.assert_array_equal(kern_md(X @ X.T, y)[0], [-.5, .5] * 2)
    model = KernMD().fit(X, ids)
    assert model.score(X, target([0, 0, 0, 1]), sample_weight=[1, 2, 4, 8]) == 13 / 15
    monitored = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    monitored.fit(X, ids, validation_data=(X, y))
    assert [entry["score"] for entry in monitored.validation_history_] == [1., 1.]


@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
def test_three_temporal_classes_cannot_become_two(scalar):
    y = sparse_rows(["gap", scalar(HIGH, "ns"), scalar(0, "ns")] * 2,
                    fill=scalar(1000, "ps"))
    with pytest.raises(ValueError, match="exactly two classes"):
        KernMD().fit(np.arange(6.)[:, None], y)


@pytest.mark.parametrize("carrier", ["array", "series", "index", "frame", "category", "inherited"])
@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
def test_held_temporal_rows_without_common_dtype_use_exact_tick_equality(scalar, carrier):
    # One picosecond and HIGH nanoseconds have no common native dtype.
    # One of the true HIGH observations matches; the other three differ.
    class Inherited(pd.arrays.SparseArray):
        pass

    large, small = scalar(HIGH, "ns"), scalar(1, "ps")
    if carrier == "category":
        levels = sparse_rows(["unused", "gap", large], fill=small)
        y = pd.Categorical.from_codes([1, 2, 1, 1], categories=pd.Index(levels))
    else:
        array_type = Inherited if carrier == "inherited" else pd.arrays.SparseArray
        y = sparse_rows(["gap", large, "gap", "gap"], fill=small, array_type=array_type)
        if carrier == "series":
            y = pd.Series(y)
        elif carrier == "index":
            y = pd.Index(y)
        elif carrier == "frame":
            y = pd.Series(y).to_frame()
    ids = np.array([1, HIGH] * 2, dtype=scalar(0, "ns").dtype)
    with pytest.raises(ValueError, match="exact shared unit"):
        KernMD().fit(X, y)
    model = KernMD().fit(X, ids)
    assert model.score(X, y, sample_weight=[1, 2, 4, 8]) == 2 / 15
    monitored = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    with pytest.raises(ValueError, match="Validation labels must belong"):
        monitored.fit(X, ids, validation_data=(X, y))


@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
def test_selected_object_projection_supplies_its_own_temporal_rows(scalar):
    small, large = scalar(1000, "ps"), scalar(HIGH, "ns")
    reads = []

    class Projected(pd.arrays.SparseArray):
        def __array__(self, dtype=None, copy=None):
            reads.append(True)
            return np.asarray([large, small] * 2, dtype=object)

    labels = sparse_rows(["gap", large] * 2, fill=small, array_type=Projected)
    fitted = KernMD().fit(X, labels)
    assert fitted.predict(X).view("i8").tolist() == [HIGH, 1] * 2
    assert reads == [True]


@pytest.mark.parametrize("scalar,box", [(np.datetime64, pd.Timestamp), (np.timedelta64, pd.Timedelta)])
def test_ordinary_pandas_temporal_fill_keeps_its_unit_and_missing_policy(scalar, box):
    small, large = scalar(1000, "ps"), box(scalar(HIGH, "ns"))
    y = sparse_rows([small, "gap"] * 2, fill=large)
    ids = np.array([1, HIGH] * 2, dtype=scalar(0, "ns").dtype)
    fitted = KernMD().fit(X, y)
    np.testing.assert_array_equal(fitted.predict(X).view("i8"), ids.view("i8"))
    truth = sparse_rows([small, "gap", small, large], fill=pd.NaT)
    assert fitted.score(X, truth, sample_weight=[1, 2, 4, 8]) == 13 / 15
    with pytest.raises(ValueError, match="NaT"):
        KernMD().fit(X, truth)


@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
@pytest.mark.parametrize("gap", [True, False])
def test_inherited_sparse_object_category_preserves_selected_temporal_levels(scalar, gap):
    class Inherited(pd.arrays.SparseArray):
        pass

    small, large = scalar(1000, "ps"), scalar(HIGH + 2, "ns")
    levels = sparse_rows(["unused", "gap" if gap else small, large],
                         fill=small if gap else None, array_type=Inherited)
    y = pd.Categorical.from_codes([1, 2] * 2, categories=pd.Index(levels))
    fitted = KernMD().fit(X, y)
    assert fitted.classes_.dtype == large.dtype
    assert fitted.predict(X).view("i8").tolist() == [1, HIGH + 2] * 2


@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
def test_missing_category_code_cannot_become_a_finite_temporal_class(scalar):
    small, large = scalar(1000, "ps"), scalar(HIGH + 2, "ns")
    levels = sparse_rows(["unused", "gap", large], fill=small)
    missing = pd.Categorical.from_codes([-1, 2, 1, 2], categories=pd.Index(levels))
    model = KernMD().fit(X, np.array([1, HIGH + 2] * 2, dtype=large.dtype))
    assert model.score(X, missing, sample_weight=[1, 2, 4, 8]) == 14 / 15
    with pytest.raises(ValueError, match="NaT"):
        KernMD().fit(X, missing)


@pytest.mark.parametrize("scalar,box,python_small", [
    (np.datetime64, pd.Timestamp, datetime(1970, 1, 1, microsecond=1)),
    (np.timedelta64, pd.Timedelta, timedelta(microseconds=1)),
])
@pytest.mark.parametrize("spelling", ["pandas", "python_numpy"])
@pytest.mark.parametrize("category", [False, True])
def test_pandas_and_python_observations_keep_shared_microsecond_ticks(scalar, box, python_small, spelling, category):
    high = scalar(HIGH, "us")
    small, large = (box(scalar(1000, "ns")), box(high)) if spelling == "pandas" else (python_small, high)
    ids = np.array([1, HIGH] * 2, dtype=high.dtype)

    def labels(codes, fill=small):
        if category:
            levels = sparse_rows(["unused", "gap", large], fill=fill)
            return pd.Categorical.from_codes(np.add(codes, 1), categories=pd.Index(levels))
        return sparse_rows((large if i else "gap" for i in codes), fill=fill)

    y = labels([0, 1] * 2)
    fitted = KernMD().fit(X, y)
    assert fitted.classes_.dtype == ids.dtype
    np.testing.assert_array_equal(fitted.predict(X).view("i8"), ids.view("i8"))
    independent = KernMD().fit(X, ids)
    assert independent.score(X, y) == 1.
    assert independent.score(X, labels([0] * 4), sample_weight=[1, 2, 4, 8]) == 5 / 15
    assert independent.score(X, sparse_rows(["gap"] * 4, fill=pd.NaT)) == 0.
    monitored = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    monitored.fit(X, ids, validation_data=(X, y))
    assert [entry["score"] for entry in monitored.validation_history_] == [1., 1.]


def test_python_duration_outside_microsecond_range_does_not_wrap():
    huge = timedelta(days=300_000_000)
    y = sparse_rows(["gap", np.timedelta64(1, "D")] * 2, fill=huge)
    expected = np.array([300_000_000, 1] * 2, dtype="timedelta64[D]")
    model = KernMD().fit(X, y)
    np.testing.assert_array_equal(model.predict(X), expected)
    unrepresentable = sparse_rows(["gap", np.timedelta64(1, "us")] * 2, fill=timedelta.max)
    with pytest.raises(ValueError, match="exact shared unit"):
        KernMD().fit(X, unrepresentable)
    assert model.score(X, unrepresentable) == 0.


def test_native_sparse_storage_preserves_a_large_python_duration_fill():
    huge = timedelta(days=300_000_000)
    raw = np.array([0, 1] * 2, dtype="timedelta64[ns]")
    y = pd.arrays.SparseArray(raw, fill_value=raw[0])
    try:
        y.fill_value = huge
    except (TypeError, ValueError):
        pytest.skip("pandas rejects this fill in a native nanosecond array")
    assert y[0] is huge
    classes = np.array([300_000_000, 1] * 2, dtype="timedelta64[D]")
    assert KernMD().fit(X, classes).score(X, y) == .5


@pytest.mark.parametrize("levels", [
    [datetime(2000, 1, 1), datetime(2000, 1, 2)],
    [timedelta(seconds=1), timedelta(seconds=2)],
    [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-02")],
    [pd.Timedelta(seconds=1), pd.Timedelta(seconds=2)],
    [np.datetime64(1, "us"), np.datetime64(2, "us")],
    [np.timedelta64(1, "us"), np.timedelta64(2, "us")],
])
@pytest.mark.parametrize("carrier", ["list", "boxed", "series"])
def test_ordinary_temporal_targets_keep_their_fitted_representation(levels, carrier):
    values = levels * 2
    if carrier == "boxed":
        y = np.fromiter(values, dtype=object)
    elif carrier == "series":
        y = pd.Series(values, dtype=object)
    else:
        y = values
    model = KernMD().fit(X, y)
    native_list = carrier == "list" and type(levels[0]) in (np.datetime64, np.timedelta64)
    expected_dtype = levels[0].dtype if native_list else np.dtype(object)
    for result in (model.classes_, model.predict(X)):
        assert result.dtype == expected_dtype
        assert all(type(v) is type(levels[0]) for v in result)


@pytest.mark.parametrize("scalar,box", [(np.datetime64, pd.Timestamp), (np.timedelta64, pd.Timedelta)])
def test_sparse_iterator_uses_its_selected_pandas_temporal_rows(scalar, box):
    large, small = box(scalar(HIGH, "us")), box(scalar(1000, "ns"))
    calls = []

    class Iterated(pd.arrays.SparseArray):
        @property
        def __array__(self):
            raise AttributeError("use the iterator")

        def __iter__(self):
            calls.append(1)
            return iter([large, small] * 2)

    labels = Iterated([0, 1] * 2, fill_value=0)
    model = KernMD().fit(X, labels)
    assert model.predict(X).view("i8").tolist() == [HIGH, 1] * 2
    assert calls == [1]


@pytest.mark.parametrize("outer", [pd.Series, pd.Index, pd.DataFrame])
@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
@pytest.mark.parametrize("mixed", [True, False])
def test_outer_pandas_projection_recovers_only_differing_temporal_units(outer, scalar, mixed):
    small, large = scalar(1000, "ps") if mixed else scalar(1, "ns"), scalar(HIGH, "ns")
    storage = sparse_rows(["gap", large] * 2, fill=small)
    selected = np.fromiter([large, small] * 2, dtype=object)
    calls = []

    class Projected(outer):
        def __array__(self, dtype=None, copy=None):
            calls.append(1)
            return selected[:, None] if outer is pd.DataFrame else selected

    if outer is pd.Index:
        y = Projected._simple_new(storage)
    elif outer is pd.DataFrame:
        y = Projected({"labels": storage})
    else:
        y = Projected(storage)
    model = KernMD().fit(X, y)
    assert calls == [1]
    prediction = model.predict(X)
    if mixed:
        assert prediction.dtype == large.dtype
        assert prediction.view("i8").tolist() == [HIGH, 1] * 2
        native = np.array([HIGH, 1] * 2, dtype=large.dtype)
        calls.clear()
        independent = KernMD().fit(X, native)
        assert independent.score(X[[1, 1, 2, 3]], y, sample_weight=[1, 2, 4, 8]) == 14 / 15
        assert calls == [1]
    else:
        assert model.classes_.dtype == prediction.dtype == object
        assert all(type(v) is scalar and v.dtype == large.dtype for v in prediction)
        assert [int(v.view("i8")) for v in prediction] == [HIGH, 1] * 2


@pytest.mark.parametrize("placement", ["points", "gap", "category", "callback", "wrapped_callback", "level_callback"])
@pytest.mark.parametrize("spelling", ["python", "pandas", "numpy", "mixed"])
def test_sparse_temporal_representation_is_independent_of_callback_placement(placement, spelling):
    small, large = {
        "python": (timedelta(microseconds=1), timedelta(microseconds=3)),
        "pandas": (pd.Timedelta(1, "us"), pd.Timedelta(3, "us")),
        "numpy": (np.timedelta64(1, "us"), np.timedelta64(3, "us")),
        "mixed": (np.timedelta64(1000, "ns"), np.timedelta64(HIGH + 2, "us")),
    }[spelling]
    values = [large, small] * 2
    cells = np.fromiter(values, dtype=object)
    ticks = [HIGH + 2 if spelling == "mixed" else 3, 1] * 2
    calls = []

    class Projected(pd.arrays.SparseArray):
        def __array__(self, dtype=None, copy=None):
            calls.append(1)
            return cells

    class Levels(pd.arrays.SparseArray):
        def take(self, indices, **kwargs):
            calls.append(1)
            return np.fromiter(([small, large][i] for i in indices), dtype=object)

    if placement == "points":
        y = sparse_rows(values)
    elif placement == "gap":
        y = sparse_rows([large, "gap"] * 2, fill=small)
    elif placement == "category":
        # Use spare text levels to prevent pandas' temporal type inference.
        levels = sparse_rows(["unused", small, large])
        y = pd.Categorical.from_codes([2, 1] * 2, categories=pd.Index(levels))
    elif placement == "level_callback":
        levels = Levels([1, 2], fill_value=0)
        y = pd.Categorical.from_codes([1, 0] * 2, categories=pd.Index(levels))
    else:
        extension = Projected([0, 1] * 2, fill_value=0)
        y = pd.Series(extension) if placement == "wrapped_callback" else extension
    calls.clear()
    model = KernMD().fit(X, y)
    assert len(calls) == int("callback" in placement)
    prediction = model.predict(X)
    if spelling == "mixed":
        assert prediction.dtype == np.dtype("timedelta64[us]")
        assert prediction.view("i8").tolist() == ticks
    else:
        assert model.classes_.dtype == prediction.dtype == object
        assert all(type(v) is type(small) for v in prediction)
        np.testing.assert_array_equal(prediction, cells)
    for fitted in (model, KernMD().fit(X, np.array(ticks, dtype="timedelta64[us]"))):
        calls.clear()
        assert fitted.score(X[[1, 1, 2, 3]], y, sample_weight=[1, 2, 4, 8]) == 14 / 15
        assert len(calls) == int("callback" in placement)
    monitored = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    monitored.fit(X, y, validation_data=(X, y))
    assert [entry["score"] for entry in monitored.validation_history_] == [1., 1.]


@pytest.mark.parametrize("scalar,box", [(np.datetime64, pd.Timestamp), (np.timedelta64, pd.Timedelta)])
@pytest.mark.parametrize("placement", ["gap", "callback"])
@pytest.mark.parametrize("carrier", ["array", "series", "index", "frame"])
def test_native_held_rows_match_retained_temporal_representatives(scalar, box, placement, carrier):
    small, large = box(scalar(1, "us")), box(scalar(3, "us"))
    cells = np.fromiter([small, large] * 2, dtype=object)
    calls = []

    class Projected(pd.arrays.SparseArray):
        def __array__(self, dtype=None, copy=None):
            calls.append(1)
            return cells

    training = (sparse_rows(["gap", large] * 2, fill=small) if placement == "gap"
                else Projected([0, 1] * 2, fill_value=0))
    native = np.array([1000, 3000] * 2, dtype=scalar(0, "ns").dtype)
    wrap = {"array": lambda a: a, "series": pd.Series, "index": pd.Index,
            "frame": lambda a: pd.DataFrame({"labels": a})}[carrier]
    held = wrap(native)
    model = KernMD().fit(X, training)
    assert model.predict(X).dtype == object and all(type(v) is box for v in model.classes_)
    assert model.score(X[[1, 1, 2, 3]], held, sample_weight=[1, 2, 4, 8]) == 14 / 15
    assert len(calls) == int(placement == "callback")
    calls.clear()
    monitored = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    monitored.fit(X, training, validation_data=(X, held))
    assert [entry["score"] for entry in monitored.validation_history_] == [1., 1.]
    assert len(calls) == int(placement == "callback")
    native[0] = scalar(1001, "ns")
    with pytest.raises(ValueError, match="Validation labels must belong"):
        monitored.fit(X, training, validation_data=(X, wrap(native)))
