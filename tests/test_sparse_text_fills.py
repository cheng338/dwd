"""Numeric sparse points retain their identities beside occupied text fills."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from dwd.gen_kern_dwd import KernGDWD
from dwd._validation import exact_sparse_uint64_target
from dwd.kern_md import KernMD, kern_md
from dwd.utils import pm1

pd = pytest.importorskip("pandas")
X = np.array([[-4.], [3.], [-1.], [1.], [-3.], [2.]])
NUMBERS = [1, 2, 1, 2, 1, 2]
WEIGHTS = [2, 1, 9, 2, 1, 1]


def text_fill(rows, fill, *, cls=None, dtype=np.int64):
    cls = pd.arrays.SparseArray if cls is None else cls
    sparse = cls(np.array(rows, dtype=dtype), fill_value=dtype(0))
    try:
        sparse.fill_value = fill
    except (TypeError, ValueError):
        pytest.skip("pandas rejects a text fill on numeric sparse storage")
    return sparse


def wrap(sparse, carrier):
    if carrier == "series":
        return pd.Series(sparse)
    if carrier == "index":
        return pd.Index(sparse)
    if carrier == "frame":
        return pd.DataFrame({"label": sparse})
    return sparse


def check_dense_accuracy(model, sparse, dense, expected):
    try:
        score = accuracy_score(dense, model.predict(X), sample_weight=WEIGHTS)
    except (ValueError, TypeError) as error:
        # sklearn's supported label combinations depend on its version.
        with pytest.raises(type(error)) as received:
            model.score(X, sparse, sample_weight=WEIGHTS)
        assert str(received.value) == str(error)
    else:
        assert model.score(X, sparse, sample_weight=WEIGHTS) == score == expected


@pytest.mark.parametrize("carrier", ["array", "series", "index", "frame"])
@pytest.mark.parametrize("fill", ["1", b"1"])
def test_occupied_text_fill_fit_score_and_monitoring(carrier, fill):
    sparse = text_fill([1, 2, 0, 2, 1, 2], fill)
    held = wrap(sparse, carrier)
    assert list(sparse) == [1, 2, fill, 2, 1, 2]
    with pytest.raises(ValueError, match="mixes numeric and text"):
        KernMD().fit(X, held)
    strings = [fill, "2" if isinstance(fill, str) else b"2"] * 3
    for train, expected in ((NUMBERS, 7), (strings, 9)):
        model = KernMD().fit(X, train)
        predictions = model.predict(X)
        assert sum(w for w, p, t in zip(WEIGHTS, predictions, sparse) if p == t) == expected
        assert model.score(X, held, sample_weight=WEIGHTS) == expected / 16
        with pytest.raises(ValueError, match="Validation labels must belong"):
            KernGDWD(initialization="zero", max_iter=2, stopping="validation").fit(
                X, train, validation_data=(X, held))


def test_low_level_encoding_does_not_merge_text_with_a_point():
    held = text_fill([1, 2, 0, 2, 1, 2], "1")
    with pytest.raises(ValueError, match="mixes numeric and text"):
        pm1(held)
    with pytest.raises(ValueError, match="mixes numeric and text"):
        kern_md(X @ X.T, held)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_unoccupied_text_fill_and_all_text_observations(dtype):
    for carrier in ("array", "series", "index", "frame"):
        numeric = wrap(text_fill(NUMBERS, "1", dtype=dtype), carrier)
        model = KernMD().fit(X, numeric)
        np.testing.assert_array_equal(model.predict(X), NUMBERS)
        assert model.score(X, numeric) == 1.
        strings = KernMD().fit(X, ["1", "2"] * 3)
        all_text = wrap(text_fill([0] * 6, "1", dtype=dtype), carrier)
        assert strings.score(X, all_text, sample_weight=WEIGHTS) == 12 / 16
        check_dense_accuracy(model, all_text, np.full(len(X), "1"), 0.)


@pytest.mark.parametrize("fill,other", [("a\0", "b"), (b"a\0", b"b")])
def test_text_only_fill_keeps_its_trailing_nul(fill, other):
    held = text_fill([0] * 6, fill)
    exact = np.asarray([fill, other] * 3, dtype=object)
    fitted = KernMD().fit(X, exact)
    assert fitted.predict(X).tolist() == exact.tolist()
    dense = np.asarray([fill] * len(X), dtype=object)
    assert exact_sparse_uint64_target(held, mode="observations").tolist() == dense.tolist()
    check_dense_accuracy(fitted, held, dense, 12 / 16)


@pytest.mark.parametrize("passive", [False, True])
def test_only_observed_sparse_category_levels_contribute_labels(passive):
    class Levels(pd.arrays.SparseArray):
        pass

    # Construct a unique integral catalogue before replacing the fill. Even
    # after replacement the original scalar levels 1, 2 and '1' are distinct.
    levels = (Levels if passive else pd.arrays.SparseArray)(np.array([1, 2, 0]), fill_value=np.int64(0))
    index = pd.Index(levels)
    dtype = pd.CategoricalDtype(index)
    try:
        levels.fill_value = "1"
    except (TypeError, ValueError):
        pytest.skip("pandas rejects a text fill on numeric sparse storage")
    held = pd.Categorical.from_codes([0, 1, 2, 1, 0, 1], dtype=dtype)
    numeric = pd.Categorical.from_codes([0, 1, 0, 1, 0, 1], dtype=dtype)
    for carrier in ("array", "series", "index", "frame"):
        model = KernMD().fit(X, wrap(numeric, carrier))
        np.testing.assert_array_equal(model.predict(X), NUMBERS)
        with pytest.raises(ValueError, match="mixes numeric and text"):
            KernMD().fit(X, wrap(held, carrier))
        assert model.score(X, wrap(held, carrier), sample_weight=WEIGHTS) == 7 / 16


def test_active_sparse_array_projection_is_authoritative_and_runs_once():
    calls = []

    class Projection(pd.arrays.SparseArray):
        def __array__(self, dtype=None, copy=None):
            calls.append(1)
            return np.asarray(["1", "2"] * 3, dtype=dtype)

    sparse = text_fill([1, 2, 0, 2, 1, 2], "1", cls=Projection)
    model = KernMD().fit(X, sparse)
    assert model.classes_.tolist() == ["1", "2"] and len(calls) == 1
    calls.clear()
    assert model.score(X, sparse, sample_weight=WEIGHTS) == 1. and len(calls) == 1


@pytest.mark.parametrize("carrier", ["array", "series", "index", "frame"])
def test_live_text_fill_is_read_only_by_the_selected_conversion(carrier):
    reads = []

    class Observed(pd.arrays.SparseArray):
        @property
        def fill_value(self):
            reads.append(1)
            return "1"

    sparse = Observed(np.array([1, 2, 0, 2, 1, 2]), fill_value=np.int64(0))
    held = wrap(sparse, carrier)
    reads.clear()
    np.asarray(held)
    calls = reads.copy()
    assert calls
    reads.clear()
    model = KernMD().fit(X, NUMBERS)
    assert model.score(X, held, sample_weight=WEIGHTS) == 7 / 16
    assert reads == calls


def test_dynamic_frame_refuses_ambiguous_projection_and_accepts_explicit_one():
    class Dynamic(pd.DataFrame):
        def __getattribute__(self, name):
            return super().__getattribute__(name)

    class Explicit(Dynamic):
        def __array__(self, dtype=None, copy=None):
            return np.asarray(["1", "2"] * 3, dtype=dtype).reshape(-1, 1)

    sparse = text_fill([1, 2, 0, 2, 1, 2], "1")
    ambiguous = Dynamic({"target": sparse})
    raw = list(sparse)
    if list(np.asarray(ambiguous).ravel()) != raw:
        with pytest.raises(ValueError, match="changes a stored class ID"):
            KernMD().fit(X, ambiguous)
    model = KernMD().fit(X, Explicit({"target": sparse}))
    assert model.classes_.tolist() == ["1", "2"]


def test_weight_length_and_continuous_observations_still_reject():
    sparse = text_fill([1, 2, 0, 2, 1, 2], "1")
    model = KernMD().fit(X, NUMBERS)
    with pytest.raises(ValueError):
        model.score(X, sparse, sample_weight=[1, 2])
    fractional = text_fill([1.5, 2, 0, 2, 1, 2], "1", dtype=np.float64)
    with pytest.raises(ValueError):
        model.score(X, fractional)


@pytest.mark.parametrize("carrier", ["array", "series", "index", "frame"])
def test_selected_iteration_keeps_its_original_scalar_rows(carrier):
    reads = []

    class Iterated(pd.arrays.SparseArray):
        def __getattribute__(self, name):
            if name in ("__array__", "__array_interface__", "__array_struct__"):
                raise AttributeError(name)
            return super().__getattribute__(name)

        def __iter__(self):
            reads.append(1)
            return iter([np.int64(1), np.int64(2), "1", np.int64(2), np.int64(1), np.int64(2)])

    y = wrap(Iterated(np.asarray(NUMBERS), fill_value=0), carrier)
    reads.clear()
    with pytest.raises(ValueError, match="mixes numeric and text"):
        KernMD().fit(X, y)
    assert reads == [1]
    for labels, numerator in ((NUMBERS, 7), (["1", "2"] * 3, 9)):
        model = KernMD().fit(X, labels)
        reads.clear()
        assert model.score(X, y, sample_weight=WEIGHTS) == numerator / 16
        assert reads == [1]


@pytest.mark.parametrize("container", [pd.Series, pd.Index])
def test_outer_vector_can_select_sparse_iteration(container):
    class Vector(container):
        @property
        def __array__(self):
            raise AttributeError("use the vector iterator")

    storage = text_fill([1, 2, 0, 2, 1, 2], "1")
    y = Vector(storage) if container is pd.Series else Vector._simple_new(storage)
    assert list(y) == [1, 2, "1", 2, 1, 2]
    with pytest.raises(ValueError, match="mixes numeric and text"):
        KernMD().fit(X, y)
    for labels, numerator in ((NUMBERS, 7), (["1", "2"] * 3, 9)):
        assert KernMD().fit(X, labels).score(X, y, sample_weight=WEIGHTS) == numerator / 16


@pytest.mark.parametrize("carrier", ["array", "series", "index", "frame"])
def test_text_fill_cannot_hide_complex_training_or_score_labels(carrier):
    y = wrap(text_fill([1j, 2j, 0, 2j, 1j, 2j], "1j", dtype=np.complex128), carrier)
    with pytest.raises(ValueError, match="numeric and text|complex"):
        KernMD().fit(X, y)
    with pytest.raises(ValueError, match="complex"):
        KernMD().fit(X, ["1j", "2j"] * 3).score(X, y)
