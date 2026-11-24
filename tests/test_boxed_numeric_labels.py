"""Boxed training labels retain the values stored in each observation."""

from contextlib import nullcontext
from fractions import Fraction
from functools import total_ordering
from sys import version_info

import numpy as np
import pytest
from sklearn.exceptions import DataConversionWarning

from dwd.gen_kern_dwd import KernGDWD
from dwd.kern_md import KernMD


def boxed(rows, layout):
    values = np.asarray(rows, dtype=object)
    if layout == "array":
        return values
    if layout == "column":
        return values[:, None]
    pd = pytest.importorskip("pandas")
    if layout == "index":
        return pd.Index(values, dtype=object)
    series = pd.Series(values, dtype=object)
    return series.to_frame() if layout == "frame" else series


@pytest.mark.parametrize("low, high, layout", [
    (np.float64(16), np.uint64(17), "array"),
    (np.float64(2**53), np.uint64(2**53 + 1), "array"),
    (np.float64(2**53), np.uint64(2**53 + 1), "series"),
    (np.int64(-(2**53 + 1)), np.float64(-(2**53)), "index"),
    (np.int64(2**63 - 1), np.uint64(2**63), "column"),
    (np.float64(2**63), np.uint64(2**63 + 1), "frame"),
], ids=["ordinary", "array", "series", "negative", "signed-unsigned", "unsigned-frame"])
def test_binary_fit_predict_and_score_keep_exact_ids(low, high, layout):
    X = np.array([[2.], [-2.], [-1.], [1.]])
    rows = [high, low, low, high]
    y = boxed(rows, layout)
    column = layout in ("column", "frame")
    with pytest.warns(DataConversionWarning) if column else nullcontext():
        model = KernMD().fit(X, y)

    # NumPy scalar equality can itself round the expected ID in an assertion.
    assert list(map(int, model.classes_)) == [int(low), int(high)]
    assert list(map(int, model.predict(X))) == list(map(int, rows))
    assert model.score(X, y) == 1.
    assert model.score(X, boxed([low, high, high, low], layout)) == 0.


@pytest.mark.parametrize("factory", [KernMD, KernGDWD])
@pytest.mark.parametrize("shift", [0, 1])
def test_binary_classifier_refuses_three_exact_classes(factory, shift):
    levels = [np.uint64(0), np.float64(2**53), np.uint64(2**53 + 1)]
    rows = levels[shift:] + levels[:shift]
    X = np.array([[-2.], [0.], [2.]] * 2)
    with pytest.raises(ValueError, match="exactly two classes"):
        factory().fit(X, boxed(rows * 2, "array"))


def test_numerically_equal_representations_still_form_one_class():
    X = np.array([[-2.], [2.], [-1.], [1.]])
    labels = [np.int64(16), np.uint64(17), np.float64(16), np.float32(17)]
    model = KernMD().fit(X, boxed(labels, "array"))
    assert list(map(int, model.classes_)) == [16, 17]
    assert list(map(int, model.predict(X))) == [16, 17, 16, 17]


def test_fractional_class_and_wide_integer_keep_their_exact_values():
    labels = [np.float64(0.125), np.uint64(2**53 + 1)]
    X = np.array([[-1.], [1.]])
    model = KernMD().fit(X, boxed(labels, "array"))
    for values in (model.classes_, model.predict(X)):
        assert values[0] == Fraction(1, 8)
        assert int(values[1]) == 2**53 + 1


def test_custom_class_label_survives_fit_and_prediction():
    class TaggedInt(int):
        pass

    tagged = TaggedInt(3)
    X = np.array([[-1.], [1.]])
    model = KernMD().fit(X, boxed([tagged, np.uint64(7)], "array"))
    assert model.classes_[0] is tagged
    assert model.predict(X)[0] is tagged
    with pytest.raises(ValueError, match="exactly two classes"):
        model.fit(np.array([[-1.], [0.], [1.]]),
                  boxed([tagged, np.float64(2**53), np.uint64(2**53 + 1)], "array"))


def test_nested_boxed_complex_label_is_rejected_before_sorting():
    from dwd.kern_md import kern_md

    scalar = np.empty((), dtype=object)
    scalar[()] = np.array(1 + 0j, dtype=object)
    y = np.empty(2, dtype=object)
    y[0], y[1] = -1, scalar
    with pytest.raises(ValueError, match="y must be real"):
        KernMD().fit(np.array([[-1.], [1.]]), y)
    with pytest.raises(ValueError, match="y must be real"):
        kern_md(np.eye(2), y)


@pytest.mark.parametrize("peer", [1, np.uint64(1)])
def test_boxed_missing_label_is_rejected_before_sorting(peer):
    pd = pytest.importorskip("pandas")
    X = np.array([[-1.], [1.]])
    for missing in (pd.NaT, np.datetime64("NaT", "ns"), float("nan"), np.float64("nan")):
        with pytest.raises(ValueError, match="NaN"):
            KernMD().fit(X, pd.Series([peer, missing], dtype=object))


def test_boxed_finite_label_keeps_its_custom_ordering_error():
    class NoOrder(np.uint64):
        def __lt__(self, other):
            raise TypeError("unsupported class ordering")

        def __gt__(self, other):
            raise TypeError("unsupported class ordering")

    with pytest.raises(TypeError, match="unsupported class ordering"):
        KernMD().fit(np.array([[-1.], [1.]]), boxed([np.uint64(1), NoOrder(2)], "array"))


@pytest.mark.parametrize("peer", [1, np.uint64(1)])
@pytest.mark.parametrize("other", [None, "a", np.str_("a"), b"a", np.bytes_("a")])
def test_text_or_none_with_an_ordinary_numeric_peer_reaches_caller_validation(peer, other):
    from sklearn.utils.multiclass import check_classification_targets
    from dwd._validation import exact_sparse_uint64_target

    y = boxed([peer, other], "array")
    with pytest.raises(ValueError, match="Unknown label type"):
        check_classification_targets(exact_sparse_uint64_target(y))
    with pytest.raises(TypeError):
        KernMD().fit(np.array([[-1.], [1.]]), y)


def test_text_subclass_can_supply_its_own_order_with_numeric_peers():
    class First(str):
        def __lt__(self, other):
            return True if isinstance(other, int) else super().__lt__(other)

        def __gt__(self, other):
            return False if isinstance(other, int) else super().__gt__(other)

    first, second = First("first"), 20
    y = boxed([first, second] * 2, "array")
    X = np.array([[-2.], [2.], [-1.], [1.]])
    model = KernMD().fit(X, y)
    assert model.classes_[0] is first
    assert model.predict(X).tolist() == [first, second] * 2


@pytest.mark.parametrize("kind", ["list", "masked", "categorical"])
def test_estimator_and_solver_use_same_class_order(kind):
    high = np.uint64(2**53 + 1)
    low = Fraction(3 * 2**53 + 1, 3) if kind == "list" else np.float64(2**53)
    y = [high, low, high, low]
    if kind == "masked":
        y = np.ma.array(y, dtype=object, mask=False)
    elif kind == "categorical":
        pd = pytest.importorskip("pandas")
        try:
            y = pd.Categorical.from_codes([0, 1, 0, 1], categories=pd.Index([high, low], dtype=object))
        except ValueError as exc:
            if "unique" not in str(exc):
                raise
            pytest.skip("pandas rejected the adjacent levels at construction")
    X = np.array([[2.], [-2.], [1.], [-1.]])
    model = KernMD().fit(X, y)
    expected = [Fraction(low), Fraction(int(high))]
    assert list(map(Fraction, model.classes_)) == expected
    assert list(map(Fraction, model.predict(X))) == expected[::-1] * 2


def test_custom_order_keeps_numpy_integer_operands():
    @total_ordering
    class Marker:
        def __lt__(self, other):
            if isinstance(other, np.integer):
                return True
            return False if other is self else NotImplemented

        def __eq__(self, other):
            return self is other

    marker = Marker()
    X = np.array([[2.], [-2.], [1.], [-1.]])
    labels = boxed([np.int32(2), marker, np.uint64(2), marker], "array")
    result = KernMD().fit(X, labels).predict(X)
    assert result[1] is result[3] is marker
    assert list(map(int, result[[0, 2]])) == [2, 2]
    with pytest.raises(ValueError, match="numeric class IDs compare inconsistently"):
        KernMD().fit(X, boxed([marker, np.float64(2**53), np.uint64(2**53 + 1), marker], "array"))


@pytest.mark.parametrize("boxed_float", [False, True])
def test_scalar_arrays_inside_object_vector_preserve_adjacent_classes(boxed_float):
    low = np.float64(2**53)
    high = np.array(2**53 + 1, dtype=np.uint64)
    if boxed_float:
        low = np.array(low)
    X = np.array([[-1.], [1.], [-2.], [2.]])
    labels = boxed([low, high, low, high], "array")
    model = KernMD().fit(X, labels)
    assert list(map(int, model.predict(X))) == [2**53, 2**53 + 1] * 2
    for indices in ([0, 2], [1, 3]):
        assert model.score(X[indices], labels[indices]) == 1.
    with pytest.raises(ValueError, match="exactly two classes"):
        model.fit(X, boxed([low, high, 0, high], "array"))


def test_unused_categorical_levels_do_not_affect_training():
    pd = pytest.importorskip("pandas")
    # Keep the unused IDs adjacent and unequal under mixed NumPy scalar comparison.
    levels = pd.Index(["unused", 1, np.float64(2**53 + 2), 0, np.uint64(2**53 + 3)], dtype=object)
    labels = pd.Categorical.from_codes([3, 1, 3, 1], categories=levels)
    X = np.array([[-2.], [2.], [-1.], [1.]])
    for y in (labels, labels.remove_unused_categories()):
        model = KernMD().fit(X, y)
        assert model.classes_.tolist() == [0, 1]
        assert model.predict(X).tolist() == [0, 1, 0, 1]


class InheritedUint(np.uint64):
    pass


class InheritedFloat(np.float64):
    pass


@pytest.mark.parametrize("float_type, uint_type", [
    (np.float64, InheritedUint), (InheritedFloat, np.uint64),
    (InheritedFloat, InheritedUint),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_boxed_inherited_numpy_ids_keep_groups_and_orientation(float_type, uint_type, reverse):
    base = 2**53
    low, high = float_type(base), uint_type(base + 1)
    rows = [low, high, low, high]
    X = np.array([[-2.], [2.], [-1.], [1.]])
    if reverse:
        rows, X = rows[::-1], X[::-1]
    y = boxed(rows, "array")
    model = KernMD().fit(X, y)
    assert list(map(int, model.classes_)) == [base, base + 1]
    assert list(map(int, model.predict(X))) == list(map(int, rows))
    assert model.score(X, y) == 1.

    # Homogeneous boxed targets can still disagree with a prediction by one.
    only_low = boxed([low] * len(X), "array")
    weights = np.array([1., 2., 4., 8.])
    correct = np.array([int(v) == base for v in rows])
    assert model.score(X[correct], only_low[correct]) == 1.
    assert model.score(X[~correct], only_low[~correct]) == 0.
    assert model.score(X, only_low, sample_weight=weights) == weights[correct].sum() / weights.sum()
    # Training must also group the same low ID across scalar types.
    y[np.flatnonzero(correct)[0]] = uint_type(base)
    model.fit(X, y)
    assert list(map(int, model.classes_)) == [base, base + 1]
    assert list(map(int, model.predict(X))) == list(map(int, rows))


@pytest.mark.parametrize("float_type, uint_type", [
    (np.float64, InheritedUint), (InheritedFloat, np.uint64),
    (InheritedFloat, InheritedUint),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_boxed_inherited_numpy_ids_reject_three_classes(float_type, uint_type, reverse):
    rows = [uint_type(0), float_type(2**53), uint_type(2**53 + 1)]
    y = boxed(rows[::-1] if reverse else rows, "array")
    with pytest.raises(ValueError, match="exactly two classes"):
        KernMD().fit(np.array([[-1.], [0.], [1.]]), y)


def test_numpy_subclass_with_custom_order_preserves_comparisons():
    class FirstUint(np.uint64):
        def __lt__(self, other):
            return not isinstance(other, FirstUint)

        def __gt__(self, other):
            return False

    first, second = FirstUint(10), np.uint64(1)
    X = np.array([[-2.], [2.], [-1.], [1.]])
    model = KernMD().fit(X, boxed([first, second, first, second], "array"))
    assert list(map(int, model.classes_)) == [10, 1]
    assert model.classes_[0] is first
    assert list(map(int, model.predict(X))) == [10, 1, 10, 1]


class Bucket(np.uint64):
    """Class labels compare by decade, across different represented values."""
    @staticmethod
    def key(value):
        return int(value) // 10

    def __lt__(self, other):
        return self.key(self) < self.key(other)

    def __le__(self, other):
        return self.key(self) <= self.key(other)

    def __eq__(self, other):
        return self.key(self) == self.key(other)

    def __ne__(self, other):
        return self.key(self) != self.key(other)

    def __gt__(self, other):
        return self.key(self) > self.key(other)

    def __ge__(self, other):
        return self.key(self) >= self.key(other)


class TaggedBucket(Bucket):
    @staticmethod
    def key(value):
        if isinstance(value, TaggedBucket):
            return (0, int(value) // 10)
        return (1, int(value))


@pytest.mark.parametrize("rows", [
    [Bucket(10), Bucket(20), Bucket(11), Bucket(21)],
    [TaggedBucket(10), np.uint64(10), TaggedBucket(11), np.uint64(10)],
], ids=["same-class-different-payload", "different-class-same-payload"])
def test_custom_scalar_equality_in_fitting_scoring_and_monitoring(rows):
    from dwd.kern_md import kern_md
    from dwd.utils import pm1

    X = np.array([[-2.], [2.], [-1.], [1.]])
    y = boxed(rows, "array")
    expected = [-1., 1., -1., 1.]
    np.testing.assert_array_equal(pm1(y), expected)
    alpha, offset = kern_md(X @ X.T, y)
    np.testing.assert_array_equal(alpha, np.array(expected) / 2)
    assert offset == 0

    md = KernMD().fit(X, y)
    gdwd = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    gdwd.fit(X, y, validation_data=(X, y))
    assert all(r["score"] == 1. for r in gdwd.validation_history_)
    for model in (md, gdwd):
        assert model.classes_[0] is rows[0]
        assert model.classes_[1] is rows[1]
        predictions = model.predict(X)
        assert all(p == v for p, v in zip(predictions, rows))
        assert model.score(X, y) == 1.
        assert model.score(X, y[[1, 0, 3, 2]]) == 0.
        assert model.score(X, y[[1, 1, 2, 3]], sample_weight=[3, 1, 1, 1]) == .5


def test_custom_class_distinguishes_single_plain_id_during_score_and_monitoring():
    X = np.array([[-2.], [2.], [-1.], [1.]])
    y = boxed([TaggedBucket(10), np.uint64(20)] * 2, "array")
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, y, validation_data=(X[:1], boxed([TaggedBucket(11)], "array")))
    assert model.score(X[:1], boxed([TaggedBucket(11)], "array")) == 1.
    assert model.score(X[:1], np.array([10], dtype=np.uint64)) == 0.
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, y, validation_data=(X[:1], np.array([10], dtype=np.uint64)))


class FamilyBucket(Bucket, InheritedUint):
    @staticmethod
    def key(value):
        return isinstance(value, InheritedUint), int(value) // 10


@pytest.mark.parametrize("carrier", ["array", "sparse", "categorical"])
def test_homogeneous_truth_preserves_subclass_recognized_by_training_class(carrier):
    X = np.array([[-2.], [2.], [-1.], [1.]])
    y = boxed(list(map(FamilyBucket, [10, 20, 11, 21])), "array")
    rows = [0, 2]
    value = InheritedUint(16)
    held = boxed([value, value], "array")
    if carrier != "array":
        pd = pytest.importorskip("pandas")
        if carrier == "sparse":
            held = pd.arrays.SparseArray(held, dtype=pd.SparseDtype(object, 0))
        else:
            levels = pd.Index([value], dtype=object)
            held = pd.Categorical.from_codes([0, 0], categories=levels)
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, y, validation_data=(X[rows], held))
    assert all(r["score"] == 1. for r in model.validation_history_)
    assert model.score(X[rows], held) == 1.
    plain = np.array([16, 16], dtype=np.int64)
    assert model.score(X[rows], plain) == 0.
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, y, validation_data=(X[rows], plain))


def test_homogeneous_predictions_preserve_subclass_recognized_by_truth():
    class FamilyEqual(np.uint64):
        def __eq__(self, other):
            return isinstance(other, (InheritedUint, FamilyEqual)) and int(self) == int(other)

        def __ne__(self, other):
            return not self == other

    X = np.array([[-2.], [2.], [-1.], [1.]])
    y = boxed([InheritedUint(5), FamilyEqual(7)] * 2, "array")
    truth = boxed([FamilyEqual(5), FamilyEqual(7)] * 2, "array")
    model = KernMD().fit(X, y)
    assert model.score(X, truth) == 1.
    assert model.score(X[::2], truth[::2]) == 1.


@pytest.mark.parametrize("upper", [InheritedUint(20), np.uint64(20), InheritedFloat(20)])
def test_exact_training_groups_keep_representatives_recognized_by_held_labels(upper):
    class Family(InheritedUint):
        def __eq__(self, other):
            return isinstance(other, InheritedUint) and int(self) == int(other)

        def __ne__(self, other):
            return not self == other

    X = np.array([[-2.], [2.], [-1.], [1.]])
    lower = InheritedUint(10)
    y = boxed([lower, upper] * 2, "array")
    held = boxed([Family(10), Family(20) if isinstance(upper, InheritedUint) else upper] * 2, "array")
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, y, validation_data=(X, held))
    assert model.classes_[0] is lower and model.classes_[1] is upper
    assert model.predict(X)[0] is lower
    assert model.score(X, held) == 1.
    assert model.score(X[::2], held[::2]) == 1.
    assert all(row["score"] == 1. for row in model.validation_history_)


def test_numpy_subclass_conversion_hooks_cannot_change_represented_ids():
    class Misleading(np.uint64):
        def item(self):
            return 0

        def __int__(self):
            return 0

    X = np.array([[-2.], [2.], [-1.], [1.]])
    high = Misleading(2**53 + 1)
    y = boxed([InheritedFloat(2**53), high] * 2, "array")
    model = KernMD().fit(X, y)
    assert list(map(int, model.classes_)) == [2**53, 2**53 + 1]
    assert model.score(X, y) == 1.
    assert model.score(X[1::2], boxed([Misleading(2**53)] * 2, "array")) == 0.


class TaggedConversion(int):
    __hash__ = None

    def __new__(cls, value, converted):
        result = super().__new__(cls, value)
        result.converted = converted
        return result

    def __int__(self):
        return self.converted


def test_fit_checks_each_row_when_comparison_uses_an_instance_conversion():
    low = InheritedUint(5)
    X = np.array([[2.], [-2.], [-1.], [1.]])
    y = boxed([TaggedConversion(7, 5), low, low, TaggedConversion(7, 7)], "array")
    model = KernMD().fit(X, y)
    key = lambda v: int.__int__(v) if isinstance(v, TaggedConversion) else int(v)
    assert list(map(key, model.classes_)) == [5, 7]
    assert list(map(key, model.predict(X))) == [7, 5, 5, 7]


class CustomOrderUint(np.uint64):
    def __lt__(self, other):
        return super().__lt__(other)


@pytest.mark.parametrize("rows", [
    [np.float64(2**53), np.uint64(2**53), np.uint64(2**53 + 1), CustomOrderUint(2**53)],
    [InheritedUint(5), InheritedUint(5), TaggedConversion(7, 7), TaggedConversion(7, 5), CustomOrderUint(5)],
], ids=["rounded-floating-peer", "instance-dependent-conversion"])
def test_mixed_custom_training_refuses_overlapping_actual_masks(rows):
    from dwd.kern_md import kern_md
    from dwd.utils import pm1

    y = boxed(rows, "array")
    X = np.arange(len(rows), dtype=float)[:, None]
    for fit in (lambda: KernMD().fit(X, y), lambda: kern_md(X @ X.T, y), lambda: pm1(y)):
        with pytest.raises(ValueError, match="numeric class IDs compare inconsistently"):
            fit()


class ExactCustomUint(np.uint64):
    def __eq__(self, other):
        return int(self) == int(other)

    def __ne__(self, other):
        return int(self) != int(other)

    def __lt__(self, other):
        return int(self) < int(other)

    def __gt__(self, other):
        return int(self) > int(other)


def test_mixed_held_observations_compare_against_fixed_training_representatives():
    b = 2**53
    X = np.array([[-2.], [2.], [-1.], [1.]])
    y = boxed([ExactCustomUint(b), np.uint64(b + 1)] * 2, "array")
    held = boxed([ExactCustomUint(b), np.float64(b), np.uint64(b + 1)], "array")
    query = X[[0, 2, 1]]
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, y, validation_data=(query, held))
    assert all(row["score"] == 1. for row in model.validation_history_)
    assert model.score(query, held) == 1.


def test_monitoring_refuses_held_observation_matching_both_classes():
    class Either(np.uint64):
        def __eq__(self, other):
            return int(other) in (10, 20)

        def __ne__(self, other):
            return not self == other

    X = np.array([[-2.], [2.], [-1.], [1.]])
    y = np.array([10, 20] * 2, dtype=np.uint64)
    held = boxed([np.uint64(10), Either(20)], "array")
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, y, validation_data=(X[:2], held))


class IntegerPeer(np.uint64):
    def __eq__(self, other):
        return isinstance(other, np.integer) and int(self) == int(other)

    def __ne__(self, other):
        return not self == other

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if ufunc in (np.equal, np.not_equal) and any(isinstance(arg, np.ndarray) for arg in args):
            return ufunc is np.not_equal
        return NotImplemented


@pytest.mark.parametrize("carrier", ["array", "series", "sparse"])
def test_wrapped_integer_compares_as_the_same_scalar_in_fit_score_and_validation(carrier):
    low, high = IntegerPeer(10), np.uint64(20)
    y = boxed([low, high, np.array(np.uint64(10)), high, low, high], "array")
    if carrier == "series":
        y = pytest.importorskip("pandas").Series(y, dtype=object)
    elif carrier == "sparse":
        pd = pytest.importorskip("pandas")
        y = pd.arrays.SparseArray(y, dtype=pd.SparseDtype(object, 0))
    X = np.array([[-3.], [3.], [-2.], [2.], [-1.], [1.]])
    monitor = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    monitor.fit(X, y, validation_data=(X, y))
    assert all(row["score"] == 1. for row in monitor.validation_history_)
    for model in (KernMD().fit(X, y), monitor):
        assert model.classes_[0] is low
        assert list(map(int, model.predict(X))) == [10, 20] * 3
        assert model.score(X, y) == 1.
        assert model.score(X[2:3], y[2:3]) == 1.


@pytest.mark.parametrize("carrier", ["sequence", "column-rows", "categorical"])
def test_original_singleton_wrapper_is_extracted_for_a_custom_peer(carrier):
    value = np.array(np.uint64(10))
    truth = [value]
    if carrier == "column-rows":
        truth = [np.array([10], dtype=np.uint64)]
    elif carrier == "categorical":
        pd = pytest.importorskip("pandas")
        try:
            truth = pd.Categorical.from_codes([0], categories=pd.Index(truth, dtype=object))
        except TypeError as exc:
            if "unhashable" not in str(exc):
                raise
            pytest.skip("pandas rejected the unhashable category at construction")
    X = np.array([[-2.], [2.], [-1.], [1.]])
    y = boxed([IntegerPeer(10), np.uint64(20)] * 2, "array")
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    with pytest.warns(DataConversionWarning) if carrier == "column-rows" else nullcontext():
        model.fit(X, y, validation_data=(X[:1], truth))
    assert all(row["score"] == 1. for row in model.validation_history_)
    assert model.score(X[:1], truth) == 1.
    assert model.score(X[:1], [10]) == 0.


def test_wrapped_prediction_is_extracted_for_a_custom_truth_operand():
    from dwd._validation import ExactLabelClassifierMixin

    class Predictor(ExactLabelClassifierMixin):
        def predict(self, X):
            return boxed([np.array(np.uint64(10)), np.array(np.uint64(30))], "array")

    truth = boxed([IntegerPeer(10), IntegerPeer(20)], "array")
    assert Predictor().score(np.zeros((2, 1)), truth, sample_weight=[3, 1]) == .75


def test_score_and_validation_of_custom_sequence_labels_are_independent_of_batch():
    class Different(np.uint64):
        def __eq__(self, other):
            return type(other) is Different and int(self) == int(other)

        def __ne__(self, other):
            return not self == other

    X = np.array([[-3.], [3.], [-2.], [2.], [-1.], [1.]])
    y = boxed([IntegerPeer(10), np.uint64(20)] * 3, "array")
    model = KernMD().fit(X, y)
    monitor = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    labels = [np.array(np.uint64(10)), Different(20)]
    for held in (labels, tuple(labels), boxed(labels, "array")):
        assert model.score(X[:2], held) == .5
        assert model.score(X[:1], held[:1]) == 1.
        assert model.score(X[1:2], held[1:]) == 0.
        with pytest.raises(ValueError, match="Validation labels must belong"):
            monitor.fit(X, y, validation_data=(X[:2], held))


class SameTypeInt(np.int64):
    def __eq__(self, other):
        return type(other) is SameTypeInt and int(self) == int(other)

    def __ne__(self, other):
        return not self == other


@pytest.mark.parametrize("sequence", [list, tuple])
@pytest.mark.parametrize("training", ["native", "normalized-boxed", "object"])
def test_held_custom_sequence_matches_boxed_truth_for_every_training_dtype(sequence, training):
    X = np.array([[-3.], [3.], [-2.], [2.], [-1.], [1.]])
    if training == "native":
        y = np.array([10, 20] * 3, dtype=np.int64)
    elif training == "normalized-boxed":
        y = boxed([np.array(np.uint64(10)), np.uint64(20)] * 3, "array")
    else:
        y = boxed([10, 20] * 3, "array")
    model = KernMD().fit(X, y)
    assert model.classes_.dtype.kind == ("O" if training == "object" else "i")
    held = sequence([np.array(np.uint64(10)), SameTypeInt(20)])
    assert model.score(X[:2], held) == model.score(X[:2], boxed(held, "array")) == .5
    assert model.score(X[1:2], held[-1:]) == 0.
    with pytest.raises(ValueError, match="Validation labels must belong"):
        KernGDWD(stopping="validation", initialization="zero", max_iter=2).fit(
            X, y, validation_data=(X[:2], held))


@pytest.mark.parametrize("sequence", [list, tuple])
@pytest.mark.parametrize("column", [False, True])
def test_custom_numeric_sequence_preserves_training_representatives(sequence, column):
    X = np.array([[-2.], [2.], [-1.], [1.]])
    low = SameTypeInt(10)
    labels = [low, np.int64(20)] * 2
    if column:
        labels = [[value] for value in labels]
    y = sequence(labels)
    with pytest.warns(DataConversionWarning) if column else nullcontext():
        model = KernMD().fit(X, y)
    assert model.classes_[0] is low
    assert model.predict(X)[0] is low
    assert model.score(X, y) == 1.
    assert model.score(X[:1], [np.int64(10)]) == 0.


def test_custom_sequence_with_native_and_object_array_rows_preserves_scalar_types():
    low = IntegerPeer(10)
    labels = [low, np.uint64(20), np.uint64(10), np.uint64(20), low, np.uint64(20)]
    y = tuple(np.array([v], dtype=object if v is low else np.uint64) for v in labels)
    X = np.array([[-3.], [3.], [-2.], [2.], [-1.], [1.]])
    with pytest.warns(DataConversionWarning):
        model = KernMD().fit(X, y)
    assert model.classes_[0] is low
    assert list(map(int, model.predict(X))) == [10, 20] * 3
    assert model.score(X, y) == 1.


def test_sequence_normalization_preserves_equivalent_native_classes_and_custom_groups():
    from dwd._validation import exact_sparse_uint64_target
    from sklearn.utils.multiclass import check_classification_targets

    class Pass(np.uint64):
        def __eq__(self, other):
            return np.uint64.__eq__(self, other)

    for cls in (int, np.uint64, InheritedUint, Pass):
        rows = [cls(10), cls(20), cls(30)] * 2
        prepared = exact_sparse_uint64_target(rows)
        assert prepared.dtype == np.asarray(rows).dtype
        check_classification_targets(prepared)
        np.testing.assert_array_equal(prepared, [10, 20, 30] * 2)
    rows = list(map(Bucket, [10, 20, 11, 21]))
    prepared = exact_sparse_uint64_target(rows)
    assert prepared.dtype == object and prepared[0] is rows[0]
    assert list(map(int, np.unique(prepared))) == [10, 20]


def test_native_projection_detects_extra_membership_in_a_third_class():
    from dwd._validation import exact_sparse_uint64_target

    class Alias(int):
        def __eq__(self, other):
            return int(self) == int(other) or (type(other) is not Alias and int(self) == 10 and int(other) == 20)

        def __ne__(self, other):
            return not self == other

    y = list(map(Alias, [10, 20, 30] * 2))
    prepared = exact_sparse_uint64_target(y)
    assert prepared.dtype == object and prepared[0] is y[0]


@pytest.mark.parametrize("refusal", [TypeError, ValueError, AttributeError, NotImplementedError])
def test_direct_sequence_grouping_keeps_classes_that_refuse_native_comparison(refusal):
    from dwd.kern_md import kern_md
    from dwd.utils import pm1

    class Strict(int):
        def __eq__(self, other):
            if type(other) is not Strict:
                raise refusal("operand outside the label domain")
            return int.__eq__(self, other)

        def __ne__(self, other):
            return not self == other

    y = list(map(Strict, [10, 20, 10, 20]))
    np.testing.assert_array_equal(pm1(y), [-1, 1, -1, 1])
    alpha, _ = kern_md(np.eye(4), y)
    np.testing.assert_array_equal(alpha, [-.5, .5, -.5, .5])


def test_sequence_grouping_propagates_errors_comparing_original_labels():
    from dwd.utils import pm1

    class Invalid(int):
        def __eq__(self, other):
            if type(other) is Invalid:
                raise ValueError("original labels cannot be compared")
            return super().__eq__(other)

        def __ne__(self, other):
            return not self == other

    with pytest.raises(ValueError, match="original labels cannot be compared"):
        pm1(list(map(Invalid, [10, 20, 10, 20])))


def test_large_sequence_ids_and_nonfinite_custom_float_keep_domain_checks():
    from dwd._validation import exact_sparse_uint64_target

    class EqFloat(float):
        def __eq__(self, other):
            return float.__eq__(self, other)

        def __float__(self):
            value = float.__float__(self)
            return value if np.isfinite(value) else 0.

    b = 2**53
    plain = exact_sparse_uint64_target([np.float64(b), np.uint64(b + 1)])
    assert list(map(int, plain)) == [b, b + 1]
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="NaN or infinity"):
            exact_sparse_uint64_target([EqFloat(1), EqFloat(value)])


def test_score_validates_a_custom_python_float_by_its_represented_value():
    class TypedFloat(float):
        def __eq__(self, other):
            return type(other) is TypedFloat and float.__eq__(self, other)

        def __ne__(self, other):
            return not self == other

    X = np.array([[-2.], [2.], [-1.], [1.]])
    y = boxed([TypedFloat(10), TypedFloat(20)] * 2, "array")
    model = KernMD().fit(X, y)
    assert model.score(X, y) == 1.
    assert model.score(X, y[::-1]) == 0.


@pytest.mark.parametrize("conversion", ["refused", "infinite"])
@pytest.mark.parametrize("layout", ["array", "series", "list", "tuple", "column"])
def test_finite_custom_floats_keep_their_stored_values_and_comparators(conversion, layout):
    from dwd.kern_md import kern_md
    from dwd.utils import pm1

    class Unit(float):
        def __eq__(self, other):
            return type(other) is Unit and float.__eq__(self, other)

        def __ne__(self, other):
            return not self == other

        def __float__(self):
            if conversion == "refused":
                raise TypeError("an explicit unit conversion is required")
            return float("inf")

    low, high = Unit(10), Unit(20)
    rows = [low, high] * 2
    if layout in ("array", "series"):
        y = boxed(rows, layout)
    elif layout == "column":
        y = [[v] for v in rows]
    else:
        y = tuple(rows) if layout == "tuple" else rows
    X = np.array([[-2.], [2.], [-1.], [1.]])
    monitor = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    with pytest.warns(DataConversionWarning) if layout == "column" else nullcontext():
        monitor.fit(X, y, validation_data=(X, y))
    assert monitor.classes_[0] is low and monitor.classes_[1] is high
    assert all(row["score"] == 1. for row in monitor.validation_history_)
    assert monitor.score(X, y) == 1.
    assert monitor.score(X[:2], [low, 20.]) == .5
    np.testing.assert_array_equal(pm1(y), [-1., 1., -1., 1.])
    if layout != "column":
        alpha, _ = kern_md(np.eye(4), y)
        np.testing.assert_array_equal(alpha, [-.5, .5, -.5, .5])


@pytest.mark.parametrize("base, hook", [(int, "__int__"), (float, "__float__"), (np.uint64, "__int__")])
@pytest.mark.parametrize("layout", ["array", "series", "list", "tuple"])
def test_inherited_order_uses_stored_ids_when_conversion_changes_values(base, hook, layout):
    class Clamped(base):
        pass
    setattr(Clamped, hook, lambda self: 0. if hook == "__float__" else 0)
    rows = [Clamped(i) for i in [10, 20, 10, 20]]
    y = boxed(rows, layout) if layout in ("array", "series") else tuple(rows) if layout == "tuple" else rows
    X = np.array([[-2.], [2.], [-1.], [1.]])
    model = KernMD().fit(X, y)
    assert model.score(X, y) == 1.
    assert model.score(X, [10, 20] * 2) == 1.
    assert model.score(X, [0] * 4) == 0.
    with pytest.raises(ValueError, match="exactly two classes"):
        model.fit(X, [Clamped(i) for i in [10, 20, 30, 20]])


@pytest.mark.parametrize("stored", [float("nan"), float("inf"), -float("inf")])
def test_inherited_float_conversion_cannot_hide_a_nonfinite_object_label(stored):
    class MaskedValue(float):
        def __float__(self):
            return 0.

    rows = [MaskedValue(10), MaskedValue(stored)] * 2
    for y in (rows, boxed(rows, "array"), [np.array([v], dtype=object) for v in rows]):
        with pytest.raises(ValueError, match="NaN or infinity"):
            KernMD().fit(np.eye(4), y)


def test_failed_nonnumeric_array_protocol_still_propagates():
    class Malformed:
        def __array__(self, *args, **kwargs):
            raise ValueError("malformed input array")

    class MalformedList(list):
        __array__ = Malformed.__array__

    class Refusing(float):
        def __float__(self):
            raise TypeError("no numeric cast")

    for y in ([Malformed(), Refusing(10)], MalformedList([Refusing(10), Refusing(20)])):
        with pytest.raises(ValueError, match="malformed input array"):
            KernMD().fit(np.eye(2), y)


def test_scalar_realness_leaves_an_unused_array_descriptor_unread():
    from dwd.kern_md import kern_md
    from dwd.utils import pm1

    class GramNumber(int):
        @property
        def __array_interface__(self):
            raise NotImplementedError("no external array representation")

    y = np.fromiter(map(GramNumber, [10, 20, 10, 20]), dtype=object, count=4)
    for value in y:
        assert not np.iscomplexobj(value)
    np.testing.assert_array_equal(pm1(y), [-1, 1, -1, 1])
    alpha, _ = kern_md(np.eye(4), y)
    np.testing.assert_array_equal(alpha, [-.5, .5, -.5, .5])
    assert KernMD().fit(np.eye(4), y).score(np.eye(4), y) == 1.


@pytest.mark.parametrize("protocol", ["dtype", "dispatch"])
@pytest.mark.parametrize("outcome", ["complex", "real", "unavailable"])
def test_realness_of_original_labels_is_checked_before_numeric_projection(protocol, outcome):
    from dwd.kern_md import kern_md
    from dwd.utils import pm1

    class Unit(int):
        def __int__(self):
            raise TypeError("no unit conversion")

    class Dtype(Unit):
        def __getattr__(self, name):
            if name != "dtype":
                raise AttributeError(name)
            if outcome == "unavailable":
                raise LookupError("label dtype unavailable")
            return np.dtype(complex if outcome == "complex" else float)

    class Dispatch(Unit):
        def __array_function__(self, func, types, args, kwargs):
            if func is not np.iscomplexobj:
                return NotImplemented
            if outcome == "unavailable":
                raise LookupError("label dtype unavailable")
            return outcome == "complex"

    cls = Dtype if protocol == "dtype" else Dispatch
    y = np.fromiter(map(cls, [10, 20, 10, 20]), dtype=object, count=4)
    X = np.eye(4)
    for fit in (lambda: pm1(y), lambda: kern_md(X, y), lambda: KernMD().fit(X, y)):
        if outcome == "complex":
            with pytest.raises(ValueError, match="y must be real"):
                fit()
        elif outcome == "unavailable":
            with pytest.raises(LookupError, match="label dtype unavailable"):
                fit()
        else:
            fit()


def test_sequence_recovery_preserves_the_first_container_protocol_error():
    class Explicit(list):
        calls = 0

        @property
        def __array_interface__(self):
            self.calls += 1
            raise LookupError(f"container protocol call {self.calls}")

    y = Explicit([1, 2])
    with pytest.raises(LookupError, match="container protocol call 1"):
        KernMD().fit(np.eye(2), y)
    assert y.calls == 1


class ProjectedLabels(list):
    def __init__(self, array):
        super().__init__([999] * len(array))
        self.array = array
        self.calls = 0

    def __array__(self, dtype=None, copy=None):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("label projection repeated")
        return np.asarray(self.array, dtype=dtype)

    def __iter__(self):
        raise RuntimeError("backing elements are not labels")


@pytest.mark.parametrize("rows", [False, True])
def test_explicit_label_arrays_are_read_once_for_fitting_scoring_and_monitoring(rows):
    from dwd.kern_md import kern_md
    from dwd.utils import pm1

    X = np.array([[-2.], [2.], [-1.], [1.]])
    ordinary = np.array([10, 20] * 2, dtype=np.uint64)
    typed = boxed([IntegerPeer(10), np.uint64(20)] * 2, "array")
    model = KernMD().fit(X, typed)

    def source():
        return ([ProjectedLabels(ordinary[i:i + 1]) for i in range(len(X))] if rows
                else ProjectedLabels(ordinary))

    with pytest.warns(DataConversionWarning) if rows else nullcontext():
        fitted = KernMD().fit(X, source())
    np.testing.assert_array_equal(fitted.predict(X), ordinary)
    assert model.score(X, source()) == 1.
    monitor = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    with pytest.warns(DataConversionWarning) if rows else nullcontext():
        monitor.fit(X, typed, validation_data=(X, source()))
    assert all(record["score"] == 1. for record in monitor.validation_history_)
    np.testing.assert_array_equal(pm1(source()), [-1., 1., -1., 1.])
    if not rows:
        alpha, _ = kern_md(np.eye(4), source())
        np.testing.assert_array_equal(alpha, [-.5, .5, -.5, .5])


def test_earlier_nested_projection_error_precedes_later_row_conversion():
    class Invalid:
        def __array__(self, *args, **kwargs):
            raise LookupError("original field unavailable")

    later = ProjectedLabels(np.array([20]))
    with pytest.raises(LookupError, match="original field unavailable"):
        KernMD().fit(np.eye(2), [[Invalid()], later])
    assert later.calls == 0


@pytest.mark.skipif(version_info < (3, 12), reason="Python-level buffer exports require Python 3.12")
def test_declared_label_protocols_follow_numpy_buffer_precedence():
    from dwd.utils import pm1

    class Buffered(list):
        def __buffer__(self, flags):
            return memoryview(np.array([10, 20, 10, 20]))

        def __array__(self, dtype=None, copy=None):
            return np.array([80, 70, 80, 70], dtype=dtype)

    y = Buffered([600, 500] * 2)
    reference = np.asarray(y)
    assert reference.tolist() == [10, 20] * 2
    np.testing.assert_array_equal(pm1(y), pm1(reference))
    model = KernMD().fit(np.eye(4), y)
    assert model.classes_.tolist() == [10, 20]


@pytest.mark.parametrize("refused", [False, True])
def test_score_validates_custom_operand_domain_independently_of_other_rows(refused):
    from dwd._validation import ExactLabelClassifierMixin

    class Domain(int):
        @property
        def dtype(self):
            if refused:
                raise LookupError("label domain unavailable")
            return np.dtype(complex)

    class Comparator(int):
        def __eq__(self, other):
            return int.__eq__(self, other)

    y = boxed([Domain(10), Comparator(20)] * 2, "array")
    ordinary = np.array([10, 20] * 2)
    X = np.eye(4)
    model = KernMD().fit(X, ordinary)

    class PredictDomain(ExactLabelClassifierMixin):
        def predict(self, X):
            return y[:len(X)]

    error = LookupError if refused else ValueError
    message = "label domain unavailable" if refused else "real"
    with pytest.raises(error, match=message):
        model.score(X, y)
    with pytest.raises(error, match=message):
        model.score(X[:1], y[:1])
    with pytest.raises(error, match=message):
        PredictDomain().score(X, ordinary)


@pytest.mark.parametrize("base, step", [(2**53, 4), (2**63, 4096)])
def test_conversion_override_partition_uses_exact_keys_for_every_stored_id(base, step):
    class MaskedUInt(np.uint64):
        def __int__(self):
            return 0

    X = np.array([[-2.], [2.], [-1.], [1.]])
    low, near = np.float64(base), MaskedUInt(base + 1)
    binary = boxed([low, near] * 2, "array")
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, binary, validation_data=(X, binary))
    assert list(map(int, model.classes_)) == [base, base + 1]
    assert model.score(X, binary) == 1.
    assert all(record["score"] == 1. for record in model.validation_history_)
    three = boxed([low, near, np.uint64(base + step)] * 2, "array")
    with pytest.raises(ValueError, match="exactly two classes"):
        KernMD().fit(np.arange(6.)[:, None], three)


@pytest.mark.parametrize("base, step", [(2**53, 4), (2**63, 4096)])
def test_auto_dwd_C_refuses_a_third_stored_id_when_conversion_rounds(base, step):
    pytest.importorskip("cvxpy")
    from dwd.socp_dwd import auto_dwd_C

    class MaskedUInt(np.uint64):
        def __int__(self):
            return 0

    y = boxed([np.float64(base), MaskedUInt(base + 1), np.uint64(base + step)] * 2, "array")
    with pytest.raises(ValueError, match="exactly two classes"):
        auto_dwd_C(np.arange(6.)[:, None], y)


def test_custom_score_keeps_exact_numeric_comparison_on_other_rows():
    X = np.array([[-2.], [2.], [-1.], [1.]])
    tag = TaggedBucket(10)
    y = boxed([tag, np.uint64(2**53 + 1)] * 2, "array")
    truth = boxed([tag, np.array(float(2**53))] * 2, "array")
    model = KernMD().fit(X, y)
    assert model.score(X, truth) == .5


def test_numpy_ufunc_override_remains_a_scalar_class_representative():
    class ScalarOnly(np.uint64):
        def __array_ufunc__(self, ufunc, method, *args, **kwargs):
            if any(isinstance(a, np.ndarray) and a.ndim for a in args):
                raise AssertionError("class comparison received a vector")
            return NotImplemented

    label = ScalarOnly(2)
    y = boxed([label, np.uint64(1), label, np.uint64(1)], "array")
    X = np.array([[2.], [-2.], [1.], [-1.]])
    model = KernMD().fit(X, y)
    assert model.classes_[1] is label
    assert model.predict(X)[0] is label
    assert model.score(X, y) == 1.


def test_auto_dwd_C_uses_custom_class_groups():
    pytest.importorskip("cvxpy")
    from dwd.socp_dwd import auto_dwd_C

    X = np.array([[-2.], [2.], [-1.], [1.]])
    y = boxed([TaggedBucket(10), np.uint64(10), TaggedBucket(11), np.uint64(10)], "array")
    assert auto_dwd_C(X, y) == auto_dwd_C(X, [-1, 1, -1, 1])


@pytest.mark.parametrize("three_classes", [False, True], ids=["binary", "three-classes"])
def test_direct_kern_md_preserves_pandas_sparse_object_ids(three_classes):
    pd = pytest.importorskip("pandas")
    from dwd.kern_md import kern_md

    low, high = np.float64(2**53), np.uint64(2**53 + 1)
    K = np.diag([1., 4., 9., 16.])
    # A third observed ID is the implicit fill; the adjacent pair stays boxed.
    rows = [high, low, 0 if three_classes else low, high]
    y = pd.arrays.SparseArray(boxed(rows, "array"), dtype=pd.SparseDtype(object, 0))
    if three_classes:
        with pytest.raises(ValueError, match="exactly two classes"):
            kern_md(K, y)
    else:
        alpha, intercept = kern_md(K, y)
        np.testing.assert_array_equal(alpha, [.5, -.5, -.5, .5])
        assert intercept == -.5
