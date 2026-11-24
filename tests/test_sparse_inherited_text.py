"""Paired accuracy for inherited text fills keeps each original operand."""

from types import SimpleNamespace

import numpy as np
import pytest

from dwd._validation import ExactLabelClassifierMixin, _sparse_real_target
from dwd.kern_md import KernMD

pd = pytest.importorskip("pandas")
X = np.array([[-3.], [3.]] * 3)
WEIGHTS = [7, 1, 4, 2, 3, 5]
NUMBERS = [1, 2] * 3


class StrID(str):
    pass


class BytesID(bytes):
    pass


def test_passive_fill_inspection_does_not_call_metadata_keys():
    class Metadata(dict):
        def keys(self):
            raise AssertionError("unexpected metadata callback")

    fill = StrID("red")
    fill.__dict__ = Metadata(note="ordinary label")
    rows = _sparse_real_target(compressed_rows(fill), pd, "score")
    assert rows[2] is fill
    assert KernMD().fit(X, NUMBERS).score(X, rows, sample_weight=WEIGHTS) == 18 / 22


@pytest.mark.parametrize("carrier", ["array", "series", "index", "frame"])
@pytest.mark.parametrize("labels,correct", [(NUMBERS, 18), (["red", "blue"] * 3, 4)])
def test_inherited_str_fill_scores_original_rows(carrier, labels, correct):
    fill = StrID("red")
    try:
        dtype = pd.SparseDtype(np.int64, fill)
        sparse = pd.arrays.SparseArray([1, 2, fill, 2, 1, 2], dtype=dtype)
    except (TypeError, ValueError):
        pytest.skip("pandas rejects this mixed sparse dtype at construction")
    held = {"array": lambda: sparse, "series": lambda: pd.Series(sparse),
            "index": lambda: pd.Index(sparse), "frame": lambda: pd.DataFrame({"y": sparse})}[carrier]()
    model = KernMD().fit(X, labels)
    assert sparse[2] is fill
    assert sum(w for i, (p, w) in enumerate(zip(model.predict(X), WEIGHTS)) if sparse[i] == p) == correct
    assert model.score(X, held, sample_weight=WEIGHTS) == correct / sum(WEIGHTS)


def compressed_rows(fill, *, all_fill=False):
    # The sparse layout itself is valid on both pandas lines. Supply the
    # selected fill separately on releases that forbid constructing the dtype.
    values = [0] * len(X) if all_fill else [1, 2, 0, 2, 1, 2]
    native = pd.arrays.SparseArray(np.array(values, dtype=np.int64), fill_value=0)
    return SimpleNamespace(sp_index=native.sp_index, sp_values=native.sp_values,
                           dtype=native.dtype, shape=native.shape, fill_value=fill)


class Predictions(ExactLabelClassifierMixin):
    def __init__(self, labels):
        self.labels = labels

    def predict(self, X):
        return self.labels


@pytest.mark.parametrize("fill", [StrID("caf\u00e9\0"), BytesID(b"\xff\0"), BytesID(b"1")])
@pytest.mark.parametrize("predict", ["text", "python_int", "numpy_int"])
@pytest.mark.parametrize("all_fill", [False, True])
def test_inherited_text_keeps_original_comparison_operands(fill, predict, all_fill):
    storage = compressed_rows(fill, all_fill=all_fill)
    rows = _sparse_real_target(storage, pd, "score")
    assert rows is not None and rows[2] is fill
    if predict == "text":
        alternative = "other" if isinstance(fill, str) else b"other"
        predictions = np.asarray([fill, alternative] * 3, dtype=object)
    else:
        predictions = np.asarray(NUMBERS, dtype=object if predict == "python_int" else np.int64)
    source = [fill] * len(X) if all_fill else [np.int64(1), np.int64(2), fill, np.int64(2), np.int64(1), np.int64(2)]
    model = Predictions(predictions)
    try:
        matched = sum(w for a, b, w in zip(source, predictions, WEIGHTS) if a == b)
    except (ValueError, TypeError) as error:
        with pytest.raises(type(error)) as actual:
            model.score(X, rows, sample_weight=WEIGHTS)
        assert str(actual.value) == str(error)
    else:
        assert model.score(X, rows, sample_weight=WEIGHTS) == matched / sum(WEIGHTS)


def test_comparison_and_conversion_overrides_keep_their_existing_path():
    class Compared(str):
        def __eq__(self, other):
            raise AssertionError("comparison replayed")

    class Converted(str):
        def __str__(self):
            raise AssertionError("conversion replayed")

    class ArrayProvider(str):
        @property
        def __array_interface__(self):
            raise AssertionError("protocol replayed")

    class ProvidesArray:
        __array__ = lambda self: np.array(1)

    class Mixed(str, ProvidesArray):
        pass

    per_instance = StrID("red")
    per_instance.__array__ = lambda: np.array(1)
    for fill in (Compared("red"), Converted("red"), ArrayProvider("red"), Mixed("red"), per_instance):
        assert _sparse_real_target(compressed_rows(fill), pd, "score") is None


def test_slotted_subclass_and_inherited_numpy_scalars_keep_their_values():
    class Slotted(str):
        __slots__ = ()
        __eq__ = str.__eq__
        __hash__ = str.__hash__

    class UnicodeScalar(np.str_):
        pass

    class BytesScalar(np.bytes_):
        pass

    for fill in (Slotted("red"), UnicodeScalar("red"), BytesScalar(b"red")):
        assert _sparse_real_target(compressed_rows(fill), pd, "score")[2] is fill


@pytest.mark.parametrize("fill", [StrID("red"), BytesID(b"1")])
@pytest.mark.parametrize("all_fill", [False, True])
def test_selected_iterator_runs_once_and_keeps_its_scalars(fill, all_fill):
    calls = []
    selected = [fill] * len(X) if all_fill else [np.int64(1), np.int64(2), fill, np.int64(2), np.int64(1), np.int64(2)]

    class Iterated(pd.arrays.SparseArray):
        def __getattribute__(self, name):
            if name in ("__array__", "__array_interface__", "__array_struct__"):
                raise AttributeError(name)
            return super().__getattribute__(name)

        def __iter__(self):
            calls.append(1)
            return iter(selected)

    carrier = Iterated(np.asarray(NUMBERS), fill_value=0)
    for dtype in (object, np.int64):
        predictions = np.asarray(NUMBERS, dtype=dtype)
        matched = sum(w for a, b, w in zip(selected, predictions, WEIGHTS) if a == b)
        calls.clear()
        assert Predictions(predictions).score(X, carrier, sample_weight=WEIGHTS) == matched / sum(WEIGHTS)
        assert calls == [1]


@pytest.mark.parametrize("passive_levels", [False, True])
def test_selected_sparse_categorical_levels_keep_inherited_text(passive_levels):
    class Levels(pd.arrays.SparseArray):
        pass

    cls = Levels if passive_levels else pd.arrays.SparseArray
    levels = cls(np.array([1, 2, 0], dtype=np.int64), fill_value=np.int64(0))
    catalog = pd.CategoricalDtype(pd.Index(levels))
    try:
        levels.fill_value = StrID("red")
    except (TypeError, ValueError):
        pytest.skip("pandas rejects mixed sparse catalogue storage")
    for codes, expected in (([0, 1, 2, 1, 0, 1], 18), ([0, 1, 0, 1, 0, 1], 22)):
        y = pd.Categorical.from_codes(codes, dtype=catalog)
        assert KernMD().fit(X, NUMBERS).score(X, pd.Series(y), sample_weight=WEIGHTS) == expected / sum(WEIGHTS)


def test_explicit_sparse_conversion_takes_precedence():
    calls = []

    class Projected(pd.arrays.SparseArray):
        def __array__(self, dtype=None, copy=None):
            calls.append(1)
            return np.asarray(NUMBERS, dtype=dtype)

    storage = Projected(np.array([1, 2, 0, 2, 1, 2]), fill_value=0)
    try:
        storage.fill_value = StrID("red")
    except (TypeError, ValueError):
        pytest.skip("pandas rejects a mixed sparse dtype")
    model = KernMD().fit(X, NUMBERS)
    calls.clear()
    assert model.score(X, storage, sample_weight=WEIGHTS) == 1.
    assert calls == [1]
