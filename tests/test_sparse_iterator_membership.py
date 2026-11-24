"""Public class membership uses the rows supplied by a selected sparse iterator."""

from collections import namedtuple
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from dwd.gen_kern_dwd import KernGDWD
from dwd.kern_md import KernMD

pd = pytest.importorskip("pandas")

X = np.array([[-3.], [3.], [-2.], [2.], [-1.], [1.]])
NUMBERS = np.array([1, 2] * 3, dtype=np.int16)
STRINGS = ["1", "2"] * 3
WEIGHTS = [2, 13, 5, 11, 19, 3]


class PassiveStr(str):
    pass


class PassiveInteger(np.int16):
    pass


class PassiveUnsigned(np.uint64):
    pass


class PassiveInt(int):
    pass


class UnhashableClass(type):
    __hash__ = None


class PassiveBytes(bytes):
    pass


class UnhashableBytes(bytes, metaclass=UnhashableClass):
    pass


class PassiveFraction(Fraction):
    pass


class PassiveDecimal(Decimal):
    pass


class PassiveList(list):
    pass


LabelRow = namedtuple("LabelRow", ["label"])


class ComparedStr(str):
    def __eq__(self, other):
        raise AssertionError("the array projection did not compare its input scalars")

    __hash__ = str.__hash__


def mixed_rows(text):
    rows = list(NUMBERS)
    rows[2] = text
    return rows


def iterated(rows, *, outer=False):
    calls = []

    # Both dependency lines support this numeric backing store. The selected
    # iterator, on the vector or on its extension, supplies the public labels.
    class Selected(pd.Series if outer else pd.arrays.SparseArray):
        @property
        def __array__(self):
            raise AttributeError("use the row iterator")

        def __iter__(self):
            calls.append("iter")
            return iter(rows)

    data = pd.arrays.SparseArray(NUMBERS, fill_value=0)
    target = Selected(data)
    return target, calls


def monitored_scores(train, held):
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, train, validation_data=(X, held))
    return [step["score"] for step in model.validation_history_]


@pytest.mark.parametrize("text", ["1", PassiveStr("1"), b"1"],
                         ids=["builtin_str", "passive_str", "builtin_bytes"])
def test_mixed_iterator_refuses_binary_fit(text):
    rows = mixed_rows(text)
    assert all(text != point and point != text for point in NUMBERS[:2])
    target, calls = iterated(rows)
    with pytest.raises(ValueError, match="mixes numeric and text"):
        KernMD().fit(X, target)
    assert calls == ["iter"]


@pytest.mark.parametrize("train", [STRINGS, NUMBERS], ids=["text", "numeric"])
def test_mixed_iterator_validates_original_rows(train):
    rows = mixed_rows(PassiveStr("1"))
    membership = [sum(row == cls for cls in train[:2]) for row in rows]
    assert membership == ([0, 0, 1, 0, 0, 0] if train is STRINGS else [1, 1, 0, 1, 1, 1])
    target, calls = iterated(rows)
    with pytest.raises(ValueError, match="Validation labels must belong"):
        monitored_scores(train, target)
    assert calls == ["iter"]


def test_outer_sparse_iterator_preserves_mixed_rows():
    target, calls = iterated(mixed_rows(PassiveStr("1")), outer=True)
    with pytest.raises(ValueError, match="mixes numeric and text"):
        KernMD().fit(X, target)
    assert calls == ["iter"]
    calls.clear()
    with pytest.raises(ValueError, match="Validation labels must belong"):
        monitored_scores(STRINGS, target)
    assert calls == ["iter"]


def test_inherited_integer_and_text_remain_distinct():
    rows = [PassiveInteger(n) for n in NUMBERS]
    rows[2] = PassiveStr("1")
    levels = rows[:3]
    assert [[bool(a == b) for b in levels] for a in levels] == [
        [True, False, False], [False, True, False], [False, False, True]]
    target, calls = iterated(rows)
    with pytest.raises(ValueError, match="mixes numeric and text"):
        KernMD().fit(X, target)
    assert calls == ["iter"]
    calls.clear()
    with pytest.raises(ValueError, match="Validation labels must belong"):
        monitored_scores(STRINGS, target)
    assert calls == ["iter"]


@pytest.mark.parametrize("first,second,other", [
    ("1", "1\0", "2"),
    (PassiveStr("1"), PassiveStr("1\0"), "2"),
    (b"a", b"a\0", b"b"),
    (PassiveBytes(b"1"), b"1\0", b"2"),
], ids=["builtin_str", "passive_str", "builtin_bytes", "passive_and_builtin_bytes"])
def test_text_iterator_distinguishes_a_trailing_nul(first, second, other):
    rows = [first, second] * 3
    assert rows[0] != rows[1] and rows[1] != rows[0]
    target, calls = iterated(rows)

    # A caller can explicitly choose the fixed-width array. Its visible
    # scalars have lost the terminator and therefore represent only one class.
    projected = np.asarray(target)
    assert projected.tolist() == [first] * len(X)
    assert calls == ["iter"]
    with pytest.raises(ValueError, match="exactly two classes"):
        KernMD().fit(X, projected)

    calls.clear()
    model = KernMD().fit(X, target)
    assert list(model.classes_) == rows[:2]
    assert list(model.predict(X)) == rows
    assert calls == ["iter"]
    calls.clear()
    exact = np.fromiter(rows, dtype=object)
    assert monitored_scores(exact, target) == [1., 1.]
    assert calls == ["iter"]
    calls.clear()
    with pytest.raises(ValueError, match="Validation labels must belong"):
        monitored_scores([first, other] * 3, target)
    assert calls == ["iter"]


def test_mixed_bytes_and_unicode_iterator_matches_original_domains():
    rows = [b"a", "b"] * 3
    text = ["a", "b"] * 3
    assert [a == b for a, b in zip(rows, text)] == [False, True] * 3
    target, calls = iterated(rows)
    projected = np.asarray(target)
    assert projected.tolist() == text and calls == ["iter"]
    assert KernMD().fit(X, projected).classes_.tolist() == text[:2]

    exact = np.fromiter(rows, dtype=object)
    for held in (exact, target):
        calls.clear()
        with pytest.raises((TypeError, ValueError)):
            KernMD().fit(X, held)
        assert calls == (["iter"] if held is target else [])
        calls.clear()
        with pytest.raises(ValueError, match="Validation labels must belong"):
            monitored_scores(text, held)
        assert calls == (["iter"] if held is target else [])


def test_inherited_bytes_with_trailing_nul_keeps_its_class():
    rows = [b"1", PassiveBytes(b"1\0")] * 3
    levels = [b"1", b"1\0"]
    assert [[row == cls for cls in levels] for row in rows] == [[True, False], [False, True]] * 3
    target, calls = iterated(rows)
    projected = np.asarray(target)
    assert list(projected) == [b"1"] * len(X) and calls == ["iter"]
    with pytest.raises(ValueError, match="exactly two classes"):
        KernMD().fit(X, projected)

    calls.clear()
    model = KernMD().fit(X, target)
    assert list(model.classes_) == levels
    assert list(model.predict(X)) == rows
    assert calls == ["iter"]
    calls.clear()
    train = np.fromiter(levels * 3, dtype=object)
    assert monitored_scores(train, target) == [1., 1.]
    assert calls == ["iter"]
    calls.clear()
    with pytest.raises(ValueError, match="Validation labels must belong"):
        monitored_scores([b"1", b"2"] * 3, target)
    assert calls == ["iter"]


def test_nonnumeric_inherited_bytes_keeps_conversion_error():
    target, calls = iterated([PassiveBytes(b"a"), PassiveBytes(b"a\0")] * 3)
    with pytest.raises(ValueError, match="invalid literal for int") as conversion_error:
        np.asarray(target)
    assert calls == ["iter"]
    calls.clear()
    with pytest.raises(ValueError) as fit_error:
        KernMD().fit(X, target)
    assert str(fit_error.value) == str(conversion_error.value)
    assert calls == ["iter"]


@pytest.mark.parametrize("train", [STRINGS, NUMBERS], ids=["text", "numeric"])
def test_mixed_iterator_accuracy_uses_original_pairs(train):
    rows = mixed_rows(PassiveStr("1"))
    target, calls = iterated(rows)
    model = KernMD().fit(X, train)
    matched = sum(w for w, row, pred in zip(WEIGHTS, rows, model.predict(X)) if row == pred)
    assert matched == (5 if train is STRINGS else 48)
    assert model.score(X, target, sample_weight=WEIGHTS) == matched / sum(WEIGHTS)
    assert calls == ["iter"]


@pytest.mark.parametrize("rows,train", [
    (list(NUMBERS), NUMBERS),
    (STRINGS, STRINGS),
    ([PassiveStr(s) for s in STRINGS], STRINGS),
], ids=["numbers", "builtin_text", "passive_text"])
def test_single_family_iterator_fits_and_validates(rows, train):
    target, calls = iterated(rows)
    model = KernMD().fit(X, target)
    assert model.classes_.tolist() == list(train[:2])
    assert model.predict(X).tolist() == list(train)
    assert calls == ["iter"]
    calls.clear()
    assert monitored_scores(train, target) == [1., 1.]
    assert calls == ["iter"]


def test_explicit_array_takes_precedence_over_mixed_iterator():
    calls = []

    class Projected(pd.arrays.SparseArray):
        def __array__(self, dtype=None, copy=None):
            calls.append("array")
            return np.asarray(STRINGS, dtype=dtype)

        def __iter__(self):
            raise AssertionError("explicit array already supplied the labels")

    target = Projected(NUMBERS, fill_value=0)
    assert KernMD().fit(X, target).classes_.tolist() == STRINGS[:2]
    assert calls == ["array"]
    calls.clear()
    assert monitored_scores(STRINGS, target) == [1., 1.]
    assert calls == ["array"]


def test_active_comparator_keeps_selected_conversion():
    target, calls = iterated(mixed_rows(ComparedStr("1")))
    assert KernMD().fit(X, target).classes_.tolist() == STRINGS[:2]
    assert calls == ["iter"]
    calls.clear()
    assert monitored_scores(STRINGS, target) == [1., 1.]
    assert calls == ["iter"]


def test_inherited_bytes_membership_uses_numeric_class_operands():
    # NumPy's integer comparison with this subclass differs from Python int's.
    # The monitor must agree with the fitted scalar representatives.
    rows = mixed_rows(PassiveBytes(b"1"))
    representatives = KernMD().fit(X, NUMBERS).classes_
    assert rows[2] != 1
    assert [sum(row == cls for cls in representatives) for row in rows] == [1] * len(X)
    target, calls = iterated(rows)
    assert monitored_scores(NUMBERS, target) == [1., 1.]
    assert calls == ["iter"]


def test_validation_reads_held_labels_without_testing_container_truth():
    class Labels(list):
        def __bool__(self):
            raise AssertionError("validation tested the truth of the label container")

    held = Labels(mixed_rows(PassiveBytes(b"1")))
    assert [sum(row == cls for cls in NUMBERS[:2]) for row in held] == [1] * len(X)
    assert monitored_scores(NUMBERS, held) == [1., 1.]


def check_original_mixed_rows(rows, text_train, text_index, *, column=None):
    # Independently count matches against the actual scalar representatives.
    # Every numeric row is outside the text domain, and vice versa.
    target, calls = iterated(rows if column is None else column)
    projected = np.asarray(target)
    if column is not None:
        assert projected.shape == (len(rows), 1)
        projected = projected.reshape(-1)
    assert list(projected) == list(text_train)
    assert calls == ["iter"]
    assert list(KernMD().fit(X, projected).predict(X)) == list(text_train)
    assert monitored_scores(text_train, projected) == [1., 1.]

    calls.clear()
    with pytest.raises((TypeError, ValueError)):
        KernMD().fit(X, target)
    assert calls == ["iter"]
    for train in (text_train, NUMBERS):
        model = KernMD().fit(X, train)
        counts = [sum(bool(row == cls) for cls in model.classes_) for row in rows]
        text_rows = [i == text_index for i in range(len(rows))]
        expected = text_rows if train is text_train else [not flag for flag in text_rows]
        assert counts == expected
        calls.clear()
        with pytest.raises(ValueError, match="Validation labels must belong"):
            monitored_scores(train, target)
        assert calls == ["iter"]

        matched = sum(w for w, row, pred in zip(WEIGHTS, rows, model.predict(X)) if row == pred)
        assert matched == sum(w for w, same in zip(WEIGHTS, expected) if same)
        calls.clear()
        assert model.score(X, target, sample_weight=WEIGHTS) == matched / sum(WEIGHTS)
        assert calls == ["iter"]


@pytest.mark.parametrize("number", [PassiveInteger, PassiveUnsigned, PassiveInt])
@pytest.mark.parametrize("text", [b"1", np.bytes_(b"1")], ids=["bytes", "numpy_bytes"])
def test_passive_integer_rows_and_byte_labels(number, text):
    rows = [number(n) for n in NUMBERS]
    rows[2] = text
    check_original_mixed_rows(rows, [b"1", b"2"] * 3, 2)


@pytest.mark.parametrize("text_index", [0, 2], ids=["text_first", "array_first"])
def test_zero_dimensional_integer_rows_and_passive_text(text_index):
    rows = [np.array(n) for n in NUMBERS]
    rows[text_index] = PassiveStr("1")
    check_original_mixed_rows(rows, STRINGS, text_index)


@pytest.mark.parametrize("wrap", [LabelRow, lambda value: PassiveList([value])],
                         ids=["namedtuple", "passive_list"])
def test_singleton_row_subclasses_and_passive_text(wrap):
    rows = mixed_rows(PassiveStr("1"))
    check_original_mixed_rows(rows, STRINGS, 2, column=[wrap(row) for row in rows])


@pytest.mark.parametrize("number", [PassiveFraction, PassiveDecimal])
@pytest.mark.parametrize("train,matched", [
    (STRINGS, 5),
    (np.fromiter([1, 2] * 3, dtype=object), 48),
], ids=["text", "python_integers"])
def test_passive_exact_numbers_and_text_use_aligned_accuracy(number, train, matched):
    rows = [number(int(n)) for n in NUMBERS]
    rows[2] = PassiveStr("1")
    target, calls = iterated(rows)
    model = KernMD().fit(X, train)
    pairs = zip(WEIGHTS, rows, model.predict(X))
    assert sum(w for w, row, pred in pairs if row == pred) == matched
    assert model.score(X, target, sample_weight=WEIGHTS) == matched / sum(WEIGHTS)
    assert calls == ["iter"]


@pytest.mark.parametrize("operation,train,matched", [
    ("fit", STRINGS, None),
    ("monitor", STRINGS, None),
    ("score", STRINGS, 5),
    ("score", np.fromiter([1, 2] * 3, dtype=object), 48),
], ids=["fit", "monitor", "text_score", "numeric_score"])
def test_unhashable_integer_class_in_selected_rows(operation, train, matched):
    class Integer(int, metaclass=UnhashableClass):
        pass

    rows = [Integer(int(n)) for n in NUMBERS]
    rows[2] = PassiveStr("1")
    target, calls = iterated(rows)
    model = KernMD().fit(X, train)
    if operation == "fit":
        run = lambda: KernMD().fit(X, target)
    elif operation == "monitor":
        run = lambda: monitored_scores(train, target)
    else:
        pairs = zip(WEIGHTS, rows, model.predict(X))
        assert sum(w for w, row, pred in pairs if row == pred) == matched
        run = lambda: model.score(X, target, sample_weight=WEIGHTS)

    try:
        projected = np.asarray(target)
        # Read the projected labels here as some NumPy versions surface the
        # conversion error only when the returned array is first used.
        projected_rows = projected.tolist()
    except (SystemError, TypeError) as error:
        cause = error.__cause__ or error
        assert isinstance(cause, TypeError) and "unhashable type" in str(cause)
        assert calls == ["iter"]
        calls.clear()
        with pytest.raises(type(error)) as selected:
            run()
        selected_cause = selected.value.__cause__ or selected.value
        assert type(selected_cause) is type(cause) and selected_cause.args == cause.args
        assert calls == ["iter"]
        return

    assert projected_rows == STRINGS and calls == ["iter"]
    assert list(KernMD().fit(X, projected).classes_) == STRINGS[:2]
    assert monitored_scores(STRINGS, projected) == [1., 1.]
    calls.clear()
    if operation == "fit":
        with pytest.raises(ValueError, match="mixes numeric and text"):
            run()
    elif operation == "monitor":
        assert [sum(row == cls for cls in model.classes_) for row in rows] == [0, 0, 1, 0, 0, 0]
        with pytest.raises(ValueError, match="Validation labels must belong"):
            run()
    else:
        assert run() == matched / sum(WEIGHTS)
    assert calls == ["iter"]


def test_unhashable_decimal_class_in_paired_accuracy():
    class Amount(Decimal, metaclass=UnhashableClass):
        pass

    rows = [Amount(int(n)) for n in NUMBERS]
    rows[2] = PassiveStr("1")
    target, calls = iterated(rows)
    projected = np.asarray(target)
    assert projected.dtype == object and all(a is b for a, b in zip(projected, rows))
    assert calls == ["iter"]
    calls.clear()
    model = KernMD().fit(X, STRINGS)
    assert sum(w for w, a, b in zip(WEIGHTS, rows, model.predict(X)) if a == b) == 5
    assert model.score(X, target, sample_weight=WEIGHTS) == 5 / sum(WEIGHTS)
    assert calls == ["iter"]


def test_unhashable_integer_label_keeps_adjacent_wide_classes():
    class Integer(int, metaclass=UnhashableClass):
        pass

    first = 2**53
    levels = [first, first + 1]
    rows = [Integer(first), np.uint64(first + 1)] * 3
    train = np.array(levels * 3, dtype=np.uint64)
    model = KernMD().fit(X, train)
    target, calls = iterated(rows)
    try:
        projected = np.asarray(target)
        visible = projected.tolist()
    except (TypeError, OverflowError, SystemError) as error:
        cause = error.__cause__ or error
        assert calls == ["iter"]
        for run in (lambda: KernMD().fit(X, target), lambda: monitored_scores(train, target),
                    lambda: model.score(X, target, sample_weight=WEIGHTS)):
            calls.clear()
            with pytest.raises(type(error)) as selected:
                run()
            selected_cause = selected.value.__cause__ or selected.value
            assert type(selected_cause) is type(cause) and selected_cause.args == cause.args
            assert calls == ["iter"]
        return

    assert calls == ["iter"]
    if len(set(visible)) == 1:
        with pytest.raises(ValueError, match="exactly two classes"):
            KernMD().fit(X, projected)
    exact = np.fromiter(rows, dtype=object)
    for labels in (exact, target):
        reads = ["iter"] if labels is target else []
        calls.clear()
        fitted = KernMD().fit(X, labels)
        assert list(map(int, fitted.classes_)) == levels
        assert list(map(int, fitted.predict(X))) == levels * 3
        assert calls == reads
        calls.clear()
        assert monitored_scores(train, labels) == [1., 1.]
        assert calls == reads
        calls.clear()
        assert model.score(X, labels, sample_weight=WEIGHTS) == 1.
        assert calls == reads


def check_fit_against_object_rows(rows, target):
    exact = KernMD().fit(X, np.fromiter(rows, dtype=object))
    selected = KernMD().fit(X, target)
    for actual, expected in ((selected.classes_, exact.classes_),
                             (selected.predict(X), exact.predict(X))):
        assert list(map(type, actual)) == list(map(type, expected))
        assert list(actual) == list(expected)


def check_byte_projection(rows, byte_type, *operations):
    reference, reads = iterated(rows)
    try:
        projected = np.asarray(reference)
        projected.tolist()  # Some conversion errors surface on first access.
    except (TypeError, SystemError) as native:
        assert byte_type is UnhashableBytes
        assert reads == ["iter"]
        cause = native.__cause__
        reason = cause if cause is not None else native
        assert isinstance(reason, TypeError) and "unhashable type" in str(reason)
        for operation in operations:
            target, calls = iterated(rows)
            with pytest.raises(type(native)) as selected:
                operation(target)
            actual = selected.value
            assert type(actual) is type(native) and actual.args == native.args
            assert type(actual.__cause__) is type(cause)
            if cause is not None:
                assert actual.__cause__.args == cause.args
            assert calls == ["iter"]
        return None

    assert reads == ["iter"]
    return projected


@pytest.mark.parametrize("byte_type", [UnhashableBytes, PassiveBytes])
def test_byte_label_with_integer_rows_fits_original_classes(byte_type):
    rows = list(NUMBERS)
    # Put the passive byte first so coercion changes the fitted class type.
    # Keep the unhashable byte after an integer to exercise NumPy's refusal.
    rows[0 if byte_type is PassiveBytes else 2] = byte_type(b"1")
    projected = check_byte_projection(rows, byte_type, lambda target: KernMD().fit(X, target))
    if projected is None:
        return
    assert projected.dtype.kind in "iu" and list(projected) == list(NUMBERS)
    assert list(KernMD().fit(X, projected).classes_) == list(NUMBERS[:2])
    target, calls = iterated(rows)
    check_fit_against_object_rows(rows, target)
    assert calls == ["iter"]


@pytest.mark.parametrize("train,matched", [
    (NUMBERS, 53),
    (np.fromiter([1, 2] * 3, dtype=object), 48),
], ids=["numpy_integers", "python_integers"])
@pytest.mark.parametrize("byte_type", [UnhashableBytes, PassiveBytes])
def test_byte_membership_uses_fitted_integer_operands(byte_type, train, matched):
    rows = mixed_rows(byte_type(b"1"))
    model = KernMD().fit(X, train)
    projected = check_byte_projection(
        rows, byte_type, lambda target: model.score(X, target, sample_weight=WEIGHTS),
        lambda target: monitored_scores(train, target))
    if projected is None:
        return
    assert monitored_scores(train, projected) == [1., 1.]
    membership = [sum(row == cls for cls in model.classes_) for row in rows]
    pairs = zip(WEIGHTS, rows, model.predict(X))
    assert sum(w for w, row, pred in pairs if row == pred) == matched
    target, calls = iterated(rows)
    assert model.score(X, target, sample_weight=WEIGHTS) == matched / sum(WEIGHTS)
    assert calls == ["iter"]
    calls.clear()
    if train is NUMBERS:
        assert membership == [1] * len(X)
        assert monitored_scores(train, target) == [1., 1.]
    else:
        assert membership == [1, 1, 0, 1, 1, 1]
        with pytest.raises(ValueError, match="Validation labels must belong"):
            monitored_scores(train, target)
    assert calls == ["iter"]


@pytest.mark.parametrize("wrap", [lambda label: (label,), LabelRow], ids=["tuple", "namedtuple"])
def test_singleton_byte_rows_keep_distinct_original_classes(wrap):
    rows = [UnhashableBytes(b"1"), UnhashableBytes(b"01")] * 3
    assert rows[0] != rows[1] and rows[1] != rows[0]
    target, calls = iterated([wrap(label) for label in rows])
    try:
        projected = np.asarray(target)
        projected_rows = projected.tolist()
    except (TypeError, SystemError) as error:
        # NumPy can reject the namedtuple row during its own conversion.
        cause = error.__cause__ or error
        assert wrap is LabelRow and isinstance(cause, TypeError) and "unhashable type" in str(cause)
        assert calls == ["iter"]
        calls.clear()
        with pytest.raises(type(error)) as selected:
            KernMD().fit(X, target)
        selected_cause = selected.value.__cause__ or selected.value
        assert type(selected_cause) is type(cause) and selected_cause.args == cause.args
        assert calls == ["iter"]
        return

    assert projected_rows == [[1]] * len(X) and calls == ["iter"]
    with pytest.raises(ValueError, match="exactly two classes"):
        KernMD().fit(X, projected.reshape(-1))
    calls.clear()
    check_fit_against_object_rows(rows, target)
    assert calls == ["iter"]
    calls.clear()
    assert monitored_scores([b"1", b"01"] * 3, target) == [1., 1.]
    assert calls == ["iter"]


def test_inherited_fraction_comparator_keeps_projected_score_behavior():
    compared = []

    class ComparedFraction(Fraction):
        def __eq__(self, other):
            compared.append(type(other))
            return super().__eq__(other)

        __hash__ = Fraction.__hash__

    class InheritedFraction(ComparedFraction):
        pass

    rows = [InheritedFraction(int(n)) for n in NUMBERS]
    rows[2] = PassiveStr("1")
    target, calls = iterated(rows)
    projected = np.asarray(target)
    assert calls == ["iter"]
    model = KernMD().fit(X, STRINGS)
    compared.clear()
    with pytest.raises(TypeError) as explicit_error:
        model.score(X, projected, sample_weight=WEIGHTS)
    explicit_reads = compared.copy()
    calls.clear()
    compared.clear()
    with pytest.raises(TypeError) as selected_error:
        model.score(X, target, sample_weight=WEIGHTS)
    assert str(selected_error.value) == str(explicit_error.value)
    assert compared == explicit_reads and calls == ["iter"]


@pytest.mark.parametrize("number", [PassiveInteger, PassiveInt])
@pytest.mark.parametrize("train,matched", [
    (NUMBERS, 53),
    (np.fromiter([1, 2] * 3, dtype=object), 48),
    ([b"1", b"2"] * 3, 5),
], ids=["numpy_integer_predictions", "python_integer_predictions", "bytes_predictions"])
def test_passive_bytes_accuracy_uses_predicted_scalar_operands(number, train, matched):
    rows = [number(n) for n in NUMBERS]
    rows[2] = PassiveBytes(b"1")
    target, calls = iterated(rows)
    model = KernMD().fit(X, train)
    pairs = zip(WEIGHTS, rows, model.predict(X))
    assert sum(w for w, row, pred in pairs if row == pred) == matched
    assert model.score(X, target, sample_weight=WEIGHTS) == matched / sum(WEIGHTS)
    assert calls == ["iter"]
