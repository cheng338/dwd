"""Public regressions for numeric class IDs inside zero-dimensional object arrays."""

from collections import UserList, deque
from collections.abc import Sequence
from contextlib import nullcontext
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from sklearn.exceptions import DataConversionWarning

from dwd.gen_kern_dwd import KernGDWD
from dwd.kern_md import KernMD
from dwd.utils import pm1


def object_box(value, depth=1):
    for _ in range(depth):
        box = np.empty((), dtype=object)
        box[()] = value
        value = box
    return value


@pytest.mark.parametrize("layout,depth,number", [
    ("series", 1, np.uint64), ("frame", 2, np.uint64),
    ("list", 2, int), ("array", 1, int),
])
def test_boxed_scalar_ids_fit_score_and_monitor(layout, depth, number):
    base = 2**53
    high = object_box(number(base + 1), depth)

    def target(rows):
        if layout == "list":
            return rows
        values = np.fromiter(rows, dtype=object)
        if layout == "array":
            return values
        series = pytest.importorskip("pandas").Series(values, dtype=object)
        return series.to_frame() if layout == "frame" else series

    X = np.array([[-2.], [2.], [-1.], [1.]])
    binary = target([float(base), high] * 2)
    known = target([float(base), high])
    unknown = target([float(base), object_box(number(base + 2), depth)])
    three = target([float(base), high, base + 4, high])
    monitor = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    with pytest.warns(DataConversionWarning) if layout == "frame" else nullcontext():
        monitor.fit(X, binary, validation_data=(X[:2], known))
    assert list(map(int, monitor.classes_)) == [base, base + 1]
    assert list(map(int, monitor.predict(X))) == [base, base + 1] * 2
    assert [r["score"] for r in monitor.validation_history_] == [1., 1.]
    assert monitor.score(X[:2], known) == 1.
    assert monitor.score(X[:2], unknown, sample_weight=[1, 3]) == .25
    assert pm1(binary).tolist() == [-1, 1] * 2
    with pytest.raises(ValueError, match="exactly two classes"):
        KernMD().fit(X, three)
    with pytest.raises(ValueError, match="must have binary labels"):
        pm1(three)
    with pytest.raises(ValueError, match="Validation labels must belong"):
        monitor.fit(X, binary, validation_data=(X[:2], unknown))


def test_boxed_custom_numeric_observation_retains_its_scalar_comparison():
    class SameType(int):
        def __eq__(self, other):
            return type(other) is SameType and int.__eq__(self, other)

        def __ne__(self, other):
            return not self == other

    X = np.array([[-2.], [2.], [-1.], [1.]])
    model = KernMD().fit(X, [10, 20] * 2)
    truth = np.fromiter([object_box(SameType(10), 2), object_box(20)], dtype=object)
    assert model.score(X[:2], truth, sample_weight=[3, 1]) == .25
    with pytest.raises(ValueError, match="Validation labels must belong"):
        KernGDWD(stopping="validation", initialization="zero", max_iter=2).fit(
            X, [10, 20] * 2, validation_data=(X[:2], truth))


@pytest.mark.parametrize("depth", [1, 2])
def test_cyclic_scalar_box_is_rejected_before_label_comparison(depth):
    root = np.empty((), dtype=object)
    root[()] = root if depth == 1 else object_box(root)
    y = np.fromiter([1, root], dtype=object)
    with pytest.raises(ValueError, match="cyclic scalar array"):
        KernMD().fit([[-1.], [1.]], y)


@pytest.mark.parametrize("sequence", [UserList, deque])
@pytest.mark.parametrize("column", [False, True])
def test_standard_sequence_preserves_adjacent_ids_in_fit_score_and_monitoring(sequence, column):
    X = np.array([[-2.], [2.], [-1.], [1.]])
    low, high = np.float64(2**53), np.uint64(2**53 + 1)

    def target(labels):
        return sequence(sequence([value]) for value in labels) if column else sequence(labels)

    labels = [low, high] * 2
    y = target(labels)
    with pytest.warns(DataConversionWarning) if column else nullcontext():
        model = KernMD().fit(X, y)
    assert list(map(int, model.classes_)) == [int(low), int(high)]
    assert list(map(int, model.predict(X))) == list(map(int, labels))
    assert model.score(X, y) == 1.
    assert model.score(X, target([high, high, low, high]), sample_weight=[3, 1, 1, 1]) == .5

    # Fit independently of the held carrier so loss during training cannot
    # hide an incorrect validation label partition.
    native = np.array(list(map(int, labels)), dtype=np.uint64)
    monitor = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    with pytest.warns(DataConversionWarning) if column else nullcontext():
        monitor.fit(X, native, validation_data=(X, y))
    assert monitor.validation_history_
    assert all(record["score"] == 1. for record in monitor.validation_history_)


@pytest.mark.parametrize("sequence", [UserList, deque])
def test_standard_sequence_preserves_third_exact_class(sequence):
    from dwd._validation import exact_sparse_uint64_target

    levels = [np.uint64(0), np.float64(2**53), np.uint64(2**53 + 1)]
    rows = sequence(levels * 2)
    exact = exact_sparse_uint64_target(rows)
    assert list(map(int, np.unique(exact))) == list(map(int, levels))
    with pytest.raises(ValueError, match="exactly two classes"):
        KernMD().fit(np.eye(6), rows)


@pytest.mark.parametrize("sequence", [UserList, deque])
def test_standard_sequence_subclass_uses_declared_array_once(sequence):
    class Projected(sequence):
        calls = 0

        def __array__(self, dtype=None, copy=None):
            self.calls += 1
            return np.array([10, 20] * 2, dtype=dtype)

        def __iter__(self):
            raise AssertionError("the declared array supplies the labels")

    y = Projected([4, 3] * 2)
    model = KernMD().fit(np.eye(4), y)
    assert y.calls == 1
    assert model.classes_.tolist() == [10, 20]


class _ReadOnlyLabels(Sequence):
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


def test_read_only_sequences_keep_exact_ids_and_respect_array_providers():
    from abc import ABCMeta

    class UnhashableMeta(ABCMeta):
        __hash__ = None

    class UnhashableLabels(_ReadOnlyLabels, metaclass=UnhashableMeta):
        pass

    class Registered:
        __init__ = _ReadOnlyLabels.__init__
        __len__ = _ReadOnlyLabels.__len__
        __getitem__ = _ReadOnlyLabels.__getitem__

    Sequence.register(Registered)
    X = np.array([[-2.], [2.], [-1.], [1.]])
    expected = [2**53, 2**53 + 1] * 2
    rows = [np.float64(expected[0]), np.uint64(expected[1])] * 2
    for carrier in (_ReadOnlyLabels, UnhashableLabels, Registered):
        y = carrier(rows)
        model = KernMD().fit(X, y)
        assert list(map(int, model.classes_)) == expected[:2]
        assert list(map(int, model.predict(X))) == expected
        assert model.score(X, carrier(reversed(rows))) == 0.
        with pytest.raises(ValueError, match="exactly two classes"):
            KernMD().fit(np.eye(3), carrier([0, *rows[:2]]))
        monitor = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
        monitor.fit(X, np.array(expected, dtype=np.uint64), validation_data=(X, y))
        assert monitor.validation_history_ and all(r["score"] == 1. for r in monitor.validation_history_)

    class Provider(_ReadOnlyLabels):
        calls = 0

        def __array__(self, dtype=None, copy=None):
            self.calls += 1
            return np.array(expected, dtype=np.uint64 if dtype is None else dtype)

        def __iter__(self):
            raise AssertionError("the declared array supplies the labels")

    y = Provider([4, 3] * 2)
    assert list(map(int, KernMD().fit(X, y).classes_)) == expected[:2]
    assert y.calls == 1


def test_sequence_admission_preserves_buffer_text_mapping_and_range_representations():
    from array import array
    from collections import UserDict
    from dwd._validation import exact_sparse_uint64_target

    class Buffer(array):
        def __iter__(self):
            raise AssertionError("the buffer supplies the labels")

    for source in (Buffer("Q", [5, 6]), memoryview(b"AB"), bytearray(b"AB"), b"AB",
                   "AB", range(-4, 4), UserDict({"one": 1, "two": 2})):
        native = np.asarray(source)
        exact = np.asarray(exact_sparse_uint64_target(source))
        assert exact.dtype == native.dtype and exact.shape == native.shape
        np.testing.assert_array_equal(exact, native)

    step = 2**63 - 1
    for source in (range(0, 3 * step, step), range(2 * step, -1, -step),
                   range(-1, 3 * step - 1, step)):
        exact = exact_sparse_uint64_target(source)
        assert list(map(int, exact)) == list(source)


@pytest.mark.parametrize("parent", [Fraction, Decimal])
def test_mutating_numeric_box_cannot_change_fitted_class_ids(parent):
    class Label(parent):
        pass

    original = object_box(Label(11))
    observations = np.fromiter([original, Label(20)]*2, dtype=object)
    X = np.array([[-2.], [2.], [-1.], [1.]])
    fitted = KernMD().fit(X, observations)
    original[()] = Label(99)
    assert list(map(int, fitted.classes_)) == [11, 20]
    assert list(map(int, fitted.predict(X))) == [11, 20]*2


@pytest.mark.parametrize("value", [float("inf"), np.float64("-inf"), Decimal("Infinity")])
@pytest.mark.parametrize("boxed", [True, False])
def test_nonfinite_object_labels_are_rejected(value, boxed):
    rows = [type(value)(0), value]*2
    if boxed:
        rows = [object_box(v) for v in rows]
    target = pytest.importorskip("pandas").Series(rows, dtype=object)
    with pytest.raises(ValueError, match="NaN or infinity"):
        KernMD().fit(np.eye(4), target)


@pytest.mark.parametrize("value, text", [(float("inf"), "red"), (Decimal("Infinity"), "red"),
                                         (np.float64("inf"), None)])
def test_mixed_text_target_keeps_caller_validation(value, text):
    from dwd._validation import exact_sparse_uint64_target

    series = pytest.importorskip("pandas").Series([value, text]*2, dtype=object)
    assert exact_sparse_uint64_target(series) is series
    with pytest.raises(TypeError):
        KernMD().fit(np.eye(4), series)


def test_indexed_labels_need_no_abc_registration():
    class Indexed:
        __init__ = _ReadOnlyLabels.__init__
        __len__ = _ReadOnlyLabels.__len__
        __getitem__ = _ReadOnlyLabels.__getitem__

    low, high = np.float64(2**53), np.uint64(2**53 + 1)
    labels = [low, high] * 2
    X = np.array([[-2.], [2.], [-1.], [1.]])
    for rows in (Indexed(labels), [Indexed([value]) for value in labels]):
        column = isinstance(rows, list)
        with pytest.warns(DataConversionWarning) if column else nullcontext():
            model = KernMD().fit(X, rows)
        assert list(map(int, model.predict(X))) == list(map(int, labels))
        assert model.score(X, Indexed(list(reversed(labels)))) == 0.
    with pytest.raises(ValueError, match="exactly two classes"):
        KernMD().fit(np.eye(3), Indexed([0, low, high]))
    native = np.array(list(map(int, labels)), dtype=np.uint64)
    monitor = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    monitor.fit(X, native, validation_data=(X, Indexed(labels)))
    assert monitor.validation_history_ and all(r["score"] == 1. for r in monitor.validation_history_)

    class Projected(Indexed):
        calls = 0

        def __array__(self, dtype=None, copy=None):
            self.calls += 1
            return np.asarray(native, dtype=dtype)

        def __len__(self):
            raise AssertionError("the declared array supplies the vector")

    y = Projected([3, 4] * 2)
    assert list(map(int, KernMD().fit(X, y).classes_)) == [int(low), int(high)]
    assert y.calls == 1


def test_indexed_admission_keeps_objects_without_positional_lengths():
    from scipy.sparse import csr_matrix
    from dwd._validation import exact_sparse_uint64_target

    class Scalar:
        def __len__(self):
            raise TypeError("no positional length")

        def __getitem__(self, index):
            raise AssertionError("scalar has no observations")

    class Keyed:
        def keys(self):
            return ("A", "B")

        def __iter__(self):
            return iter(self.keys())

        def __len__(self):
            return 2

        def __getitem__(self, key):
            return {"A": 1, "B": 2}[key]

    for value in (Scalar(), csr_matrix(np.eye(2))):
        assert exact_sparse_uint64_target(value) is value
    keys = Keyed()
    np.testing.assert_array_equal(np.asarray(exact_sparse_uint64_target(keys)), np.asarray(keys))
