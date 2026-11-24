"""Preserve the integer identities of pandas sparse classification targets."""
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from sklearn.exceptions import DataConversionWarning

from dwd._validation import exact_sparse_uint64_target
from dwd.cv import run_cv
from dwd.gen_dwd import GenDWD, GenDWDCV
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV
from dwd.kern_md import KernMD, kern_md
from dwd.utils import pm1

pd = pytest.importorskip("pandas")


def binary_data(ids=(2**53, 2**53 + 1)):
    labels = np.tile(np.array(ids, dtype=np.uint64), 12)
    X = np.tile([[-1.], [1.]], (12, 1))
    return X, labels


@pytest.mark.parametrize("container", [pd.Series, pd.Index, lambda a: a, lambda a: pd.DataFrame({"y": a})],
                         ids=["series", "index", "array", "frame"])
@pytest.mark.parametrize("fill", [0, np.uint64(0), 2**53, np.uint64(2**53),
                                  2**64 - 1, np.uint64(2**64 - 1), 17, 0.0,
                                  np.longdouble(0)])
def test_class_ids_are_read_from_declared_sparse_subtype(container, fill):
    labels = np.array([0, 2**53, 2**53 + 1, 2**64 - 1], dtype=np.uint64)
    # Construct using integer comparison so the source itself has exact IDs.
    a = pd.arrays.SparseArray(labels, fill_value=np.uint64(fill))
    a.fill_value = fill
    y = container(a)
    actual = exact_sparse_uint64_target(y)
    assert actual.dtype == np.uint64 and actual.shape == y.shape
    assert actual.ravel().tolist() == labels.tolist()
    assert a.fill_value is fill
    assert a.sp_values.tolist() == [int(i) for i in labels if int(i) != int(fill)]


@pytest.mark.parametrize("fill", [np.nan, -1, 1.5, 2**64])
def test_invalid_occupied_fill_never_becomes_a_uint64_label(fill):
    a = pd.arrays.SparseArray(np.array([0, 2**53 + 1], dtype=np.uint64), fill_value=0)
    try:
        a.fill_value = fill
    except (TypeError, ValueError, OverflowError):
        pytest.skip("pandas rejects this sparse dtype at construction")
    with pytest.raises(ValueError, match="representable as a uint64 class label"):
        KernMD().fit([[-1.], [1.]], pd.Series(a))
    b = pd.arrays.SparseArray(np.array([7, 2**53 + 1], dtype=np.uint64), fill_value=0)
    b.fill_value = fill
    assert exact_sparse_uint64_target(b).tolist() == [7, 2**53 + 1]


@pytest.mark.parametrize("estimator", [
    GenDWD(), KernMD(), KernGDWD(),
    GenDWDCV(lambd_vals=(1.,), q_vals=(1.,), cv=2),
    KernGDWDCV(lambd_vals=(1.,), q_vals=(1.,), cv=2),
], ids=lambda estimator: type(estimator).__name__)
def test_public_fits_keep_adjacent_sparse_classes(estimator):
    X, labels = binary_data()
    y = pd.Series(pd.arrays.SparseArray(labels, fill_value=2**53))
    model = estimator.fit(X, y)
    assert model.classes_.dtype == np.uint64
    assert model.classes_.tolist() == [2**53, 2**53 + 1]
    assert model.predict(X).tolist() == labels.tolist()
    assert model.score(X, y) == 1.


def test_run_cv_and_low_level_binary_encoding_keep_adjacent_ids():
    X, labels = binary_data()
    y = pd.Series(pd.arrays.SparseArray(labels, fill_value=2**53))
    assert pm1(y).tolist() == X[:, 0].tolist()
    alpha, offset = kern_md(X @ X.T, y)
    np.testing.assert_array_equal(alpha, X[:, 0] / 12)
    assert offset == 0.
    _, score, best, _, _ = run_cv(KernGDWD(), X, y, {"lambd": [1.]}, cv=2)
    assert score == 1. and best.predict(X).tolist() == labels.tolist()


def test_column_warning_preserved_and_score_compares_exact_label_ids():
    X, labels = binary_data((0, 2**53))
    y = pd.Series(pd.arrays.SparseArray(labels, fill_value=0))
    with pytest.warns(DataConversionWarning):
        model = KernMD().fit(X, y.to_frame())
    truth = pd.Series(pd.arrays.SparseArray(np.array([0, 2**53, 2**53 + 1], dtype=np.uint64),
                                            fill_value=0))
    query = np.array([[-1.], [1.], [1.]])
    assert model.predict(query).tolist() == [0, 2**53, 2**53]
    assert model.score(query, truth, sample_weight=[1, 1, 4]) == pytest.approx(1/3)
    # A model trained on valid integral float classes also distinguishes the adjacent ID.
    model.fit(X, labels.astype(float))
    assert model.score(query, truth) == pytest.approx(2/3)


def test_monitoring_rejects_adjacent_unknown_class():
    X, labels = binary_data((0, 2**53))
    invalid = pd.Series(pd.arrays.SparseArray(np.array([0, 2**53 + 1], dtype=np.uint64), fill_value=0))
    model = KernGDWD(stopping="validation", max_iter=2)
    for training in (labels, labels.astype(float)):
        with pytest.raises(ValueError, match="Validation labels must belong"):
            model.fit(X, training, validation_data=(X[:2], invalid))


def test_float_truth_does_not_round_a_large_uint64_prediction():
    X, labels = binary_data((0, 2**53 + 1))
    sparse = pd.Series(pd.arrays.SparseArray(labels, fill_value=0))
    model = KernMD().fit(X, sparse)
    assert model.predict(X[:2]).tolist() == [0, 2**53 + 1]
    assert model.score(X[:2], [0., float(2**53)], sample_weight=[1, 3]) == .25
    assert model.score(X[:2], [0., 2**53 + 1], sample_weight=[1, 3]) == 1.
    _, lower_labels = binary_data((0, 2**53))
    model.fit(X, pd.Series(pd.arrays.SparseArray(lower_labels, fill_value=0)))
    assert model.score(X[:2], [0., 2**53 + 1], sample_weight=[1, 3]) == .25


def test_non_native_uint64_target_preserves_the_original_class_ids():
    X, labels = binary_data((0, 2**53 + 1))
    sparse = pd.Series(pd.arrays.SparseArray(labels.astype(labels.dtype.newbyteorder("S")), fill_value=0))
    model = KernMD().fit(X, sparse)
    assert model.classes_.tolist() == [0, 2**53 + 1]
    assert model.predict(X).tolist() == labels.tolist()


def test_monitoring_reads_mixed_numeric_sequence_before_float_promotion():
    model = KernGDWD(stopping="validation", max_iter=2)
    X, labels = binary_data((0, 2**53 + 1))
    sparse = pd.Series(pd.arrays.SparseArray(labels, fill_value=0))
    model.fit(X, sparse, validation_data=(X[:2], [0., 2**53 + 1]))
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, sparse, validation_data=(X[:2], [0., float(2**53)]))


@pytest.mark.parametrize("fill", [0, np.uint64(0)])
def test_monitoring_sparse_unsigned_single_positive_class(fill):
    X, _ = binary_data()
    training = np.tile(np.array([-1, 0], dtype=np.int64), 12)
    validation = pd.Series(pd.arrays.SparseArray(np.zeros(2, dtype=np.uint64),
                                                 fill_value=fill))
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, training, validation_data=(X[[1, 1]], validation))
    assert [row["score"] for row in model.validation_history_] == [1., 1.]


@pytest.mark.parametrize("upper, train_dtype, unknown", [
    (2**53, np.int64, 2**53 + 1),
    (2**53 + 1, np.int64, 2**53),
    (2**53, np.float64, 2**53 + 1),
])
def test_monitoring_reads_signed_unsigned_sequence_before_inference(upper, train_dtype, unknown):
    X, _ = binary_data()
    training = np.tile(np.array([-1, upper], dtype=train_dtype), 12)
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    known = [np.int64(-1), np.uint64(upper)]
    model.fit(X, training, validation_data=(X[:2], known))
    assert [row["score"] for row in model.validation_history_] == [1., 1.]
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, training, validation_data=(X[:2], [np.int64(-1), np.uint64(unknown)]))


def test_monitoring_compares_boxed_float_with_sparse_integer_class_exactly():
    X, ids = binary_data((0, 2**53 + 1))
    training = pd.Series(pd.arrays.SparseArray(ids, fill_value=0))
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    known = pd.Series([np.float64(0), np.uint64(2**53 + 1)], dtype=object)
    model.fit(X, training, validation_data=(X[:2], known))
    assert [row["score"] for row in model.validation_history_] == [1., 1.]
    unknown = pd.Series([np.float64(2**53)], dtype=object)
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, training, validation_data=(X[:1], unknown))


@pytest.mark.parametrize("container", [
    lambda value: pd.Series([value], dtype="UInt64"),
    lambda value: pd.DataFrame({"y": pd.Series([value], dtype="UInt64")}),
    lambda value: np.array([np.uint64(value)], dtype=object),
    lambda value: np.array([np.int64(value)], dtype=object),
], ids=["nullable-series", "nullable-frame", "boxed-unsigned", "boxed-signed"])
def test_monitoring_compares_original_integer_scalar_with_float_class(container):
    X, _ = binary_data()
    training = np.tile([-1., float(2**53)], 12)
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, training, validation_data=(X[[1]], container(2**53)))
    assert [row["score"] for row in model.validation_history_] == [1., 1.]
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, training, validation_data=(X[[1]], container(2**53 + 1)))


@pytest.mark.parametrize("classes, valid, unknown", [
    (["a", "b"], "b", "c"),
    ([0.25, 0.75], np.longdouble(0.75),
     np.nextafter(np.longdouble(0.75), np.longdouble(1))),
    ([False, True], np.bool_(True), None),
])
def test_monitoring_preserves_other_boxed_classes(classes, valid, unknown):
    X, _ = binary_data()
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    training = np.tile(classes, 12)
    model.fit(X, training, validation_data=(X[[1]], np.array([valid], dtype=object)))
    assert [row["score"] for row in model.validation_history_] == [1., 1.]
    with pytest.raises(ValueError):
        model.fit(X, training, validation_data=(X[[1]], np.array([unknown], dtype=object)))


@pytest.mark.parametrize("train_dtype, held_dtype", [
    (np.int64, np.uint64), (np.uint64, np.int64),
    (">i8", ">u8"), (">u8", ">i8"),
])
def test_monitoring_encodes_adjacent_signed_and_unsigned_classes(train_dtype, held_dtype):
    X, ids = binary_data((2**53, 2**53 + 1))
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, ids.astype(train_dtype), validation_data=(X[:2], ids[:2].astype(held_dtype)))
    assert [row["score"] for row in model.validation_history_] == [1., 1.]


@pytest.mark.parametrize("classes, train_dtype, held_dtype, index", [
    ((0, 2**64 - 1), np.uint64, np.int8, 0),
    ((2**63 - 1, 2**63), np.uint64, np.int64, 0),
    ((-1, 255), np.int16, np.uint8, 1),
    ((2**64 - 2, 2**64 - 1), np.uint64, np.uint64, 1),
])
def test_monitoring_accepts_representable_class_at_bounds(classes, train_dtype, held_dtype, index):
    X, _ = binary_data()
    training = np.tile(np.array(classes, dtype=train_dtype), 12)
    validation = np.array([classes[index]], dtype=held_dtype)
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, training, validation_data=(X[[index]], validation))
    assert [row["score"] for row in model.validation_history_] == [1., 1.]


@pytest.mark.parametrize("classes, train_dtype, held_dtype, unknown", [
    ((0, 2**53), np.int64, np.uint64, 2**53 + 1),
    ((0, 2**53), np.uint64, np.int64, 2**53 + 1),
    ((-1, 0), np.int64, np.uint64, 2**64 - 1),
    ((0, 2**64 - 1), np.uint64, np.int64, -1),
    ((0, 2**63 - 1), np.int64, np.uint64, 2**63),
    ((0, 2**63), np.uint64, np.int64, 2**63 - 1),
    ((-1, 256), np.int16, np.uint8, 0),
])
def test_monitoring_rejects_unknown_signed_and_unsigned_ids(classes, train_dtype, held_dtype, unknown):
    X, _ = binary_data()
    training = np.tile(np.array(classes, dtype=train_dtype), 12)
    validation = np.array([unknown], dtype=held_dtype)
    with pytest.raises(ValueError, match="Validation labels must belong"):
        KernGDWD(stopping="validation", max_iter=2).fit(
            X, training, validation_data=(X[:1], validation))


def integer_target(values, storage, dtype):
    native = np.asarray(values, dtype=dtype)
    if storage == "nullable":
        return pd.array(native, dtype="UInt64" if dtype == np.uint64 else "Int64")
    # Store an unused level and reverse the occupied levels, so codes cannot
    # themselves be the labels. The Python zero fill reproduced the rounding.
    levels, codes = np.unique(native, return_inverse=True)
    categories = np.concatenate((levels[::-1], np.array([17], dtype=dtype)))
    if storage == "sparse-categories":
        categories = pd.arrays.SparseArray(categories, fill_value=0)
    return pd.Categorical.from_codes(len(levels) - 1 - codes, categories=pd.Index(categories))


@pytest.mark.parametrize("storage", ["nullable", "sparse-categories", "dense-categories"])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("container", [lambda a: a, pd.Series, pd.Index,
                                       lambda a: pd.DataFrame({"y": a})],
                         ids=["array", "series", "index", "frame"])
def test_integer_targets_preserve_training_monitoring_and_score(storage, dtype, container):
    low = -1 if dtype == np.int64 else 0
    high = 2**53 + 1
    X = np.tile([[-1.], [1.]], (8, 1))
    native = np.tile(np.array([low, high], dtype=dtype), 8)
    y = container(integer_target(native, storage, dtype))
    held = container(integer_target(native[:2], storage, dtype))
    unknown = container(integer_target([low, high - 1], storage, dtype))
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, y, validation_data=(X[:2], held))
    assert model.classes_.tolist() == [low, high]
    assert model.classes_.dtype == dtype
    assert model.predict(X).tolist() == native.tolist()
    assert [r["score"] for r in model.validation_history_] == [1., 1.]
    assert model.score(X[:2], held) == 1.
    assert model.score(X[:2], unknown, sample_weight=[1, 3]) == .25
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, y, validation_data=(X[:2], unknown))
    # Reverse the roles. A true unknown can otherwise round onto this class.
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, np.tile(np.array([low, high - 1], dtype=dtype), 8),
                  validation_data=(X[:2], held))


@pytest.mark.parametrize("storage", ["nullable", "sparse-categories"])
def test_missing_integer_target_never_becomes_a_class(storage):
    y = integer_target([0, 2**53 + 1], storage, np.uint64)
    if storage == "nullable":
        y[0] = pd.NA
    else:
        y = pd.Categorical.from_codes([-1, y.codes[1]], dtype=y.dtype)
    with pytest.raises(ValueError):
        KernMD().fit([[-1.], [1.]], y)
    X, native = binary_data((0, 2**53 + 1))
    with pytest.raises(ValueError):
        KernGDWD(stopping="validation").fit(X, native, validation_data=(X[:2], y))


def test_legacy_sparse_text_fill_does_not_stringify_integer_categories():
    high = 2**53 + 1
    sparse = pd.arrays.SparseArray(np.array([0, high], dtype=np.uint64), fill_value=0)
    try:
        sparse.fill_value = "other"
    except (ValueError, TypeError):
        pytest.skip("pandas rejects a text fill on integer sparse storage")
    held = pd.Categorical.from_codes([0, 1], categories=pd.Index(sparse))
    X = np.tile([[-1.], [1.]], (8, 1))
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, ["other", str(high)] * 8, validation_data=(X[:2], held))


@pytest.mark.parametrize("low, high", [
    (np.int64(-1), np.uint64(2**53 + 1)), (0., 2**53 + 1),
    (np.int64(0), np.uint64(2**64 - 1)),
])
def test_mixed_training_sequence_preserves_integer_labels(low, high):
    X = np.tile([[-1.], [1.]], (8, 1))
    training = [low, high] * 8
    known = [low, high]
    unknown = [low, int(high) - 1]
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, training, validation_data=(X[:2], known))
    assert model.classes_.tolist() == [int(low), int(high)]
    assert model.predict(X).tolist() == [int(low), int(high)] * 8
    assert [r["score"] for r in model.validation_history_] == [1., 1.]
    assert model.score(X[:2], known) == 1.
    assert model.score(X[:2], unknown, sample_weight=[1, 3]) == .25
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, training, validation_data=(X[:2], unknown))


@pytest.mark.parametrize("classes", [
    (np.int64(-1), np.uint64(2**64 - 1)), (.5, 2**53 + 1), (.5, np.uint64(2**64 - 1)),
])
def test_mixed_labels_across_native_dtypes_keep_scalar_ids(classes):
    X = np.tile([[-1.], [1.]], (8, 1))
    exact = [classes[0].item() if isinstance(classes[0], np.generic) else classes[0],
             int(classes[1])]
    held = list(classes)
    for training in (np.array(exact * 8, dtype=object), held * 8):
        model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
        model.fit(X, training, validation_data=(X[:2], held))
        assert model.classes_.tolist() == exact
        assert model.predict(X[:2]).tolist() == exact
        assert [row["score"] for row in model.validation_history_] == [1., 1.]
        with pytest.raises(ValueError, match="Validation labels must belong"):
            model.fit(X, training, validation_data=(X[:2], [classes[0], exact[1] - 1]))


def test_mixed_score_does_not_require_truth_to_share_a_native_dtype():
    X, ids = binary_data((0, 2**64 - 1))
    model = KernMD().fit(X, ids)
    assert model.score(X[:2], [np.int64(-1), np.uint64(2**64 - 1)], sample_weight=[1, 3]) == .75
    # sklearn accuracy still rejects non-integral and missing observations.
    for bad in ((.5, np.uint64(2**64 - 1)), (np.nan, np.uint64(2**64 - 1))):
        with pytest.raises(ValueError):
            model.score(X[:2], bad)


@pytest.mark.parametrize("high", [2**32, 2**53])
def test_float_inference_on_integral_sequence_does_not_change_held_target_type(high):
    X, ids = binary_data((0, high))
    model = KernMD().fit(X, ids.astype(float))
    assert model.score(X[:2], [np.int64(0), np.uint64(high)]) == 1.
    assert model.score(X[:2], [0., high + 1], sample_weight=[1, 3]) == .25


def test_numeric_sequences_reuse_inferred_arrays_and_preserve_ordinary_inputs():
    for classes in ((.25, .75), [0, 1], (0, 1.), [False, True]):
        dense = exact_sparse_uint64_target(classes)
        assert isinstance(dense, np.ndarray)
        assert dense.dtype == np.asarray(classes).dtype
        np.testing.assert_array_equal(dense, classes)
        assert exact_sparse_uint64_target(dense) is dense
    names = ("red", "blue")
    assert exact_sparse_uint64_target(names) is names


@pytest.mark.parametrize("high", [2**32, 2**53, 2**53 + 1, 2**63])
def test_float_sparse_categories_with_integer_fill_use_original_ids(high):
    levels = pd.arrays.SparseArray(np.array([0, 2], dtype=np.float32), fill_value=0)
    levels.fill_value = np.uint64(high)
    categories = pd.Index(levels)
    training = pd.Categorical.from_codes([1, 0] * 8, categories=categories)
    assert int(training[1]) == high
    single = pd.Categorical.from_codes([0], categories=categories)
    X = np.tile([[-1.], [1.]], (8, 1))
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, training, validation_data=(X[[1]], single))
    assert model.classes_.tolist() == [2, high]
    assert int(model.predict(X[[1]])[0]) == high
    assert [row["score"] for row in model.validation_history_] == [1., 1.]
    assert model.score(X, training) == 1.
    assert model.score(X[[1]], single) == 1.
    other = 2**53 if high == 2**53 + 1 else 2**53 + 1
    unknown = np.array([other], dtype=np.uint64)
    assert model.score(X[[1]], unknown) == 0.
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, training, validation_data=(X[[1]], unknown))


def test_signed_nullable_class_compares_exactly_with_integral_float_when_scoring():
    X = np.tile([[-1.], [1.]], (8, 1))
    exact = pd.Series(np.tile([-(2**53 + 1), 1], 8), dtype="Int64")
    floats = np.tile(np.array([-(2**53), 1.]), 8)
    assert KernMD().fit(X, exact).score(X, floats) == .5
    assert KernMD().fit(X, floats).score(X, exact) == .5


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_boxing_large_mixed_numeric_targets_preserves_finite_validation(bad):
    X, native = binary_data((0, 2**64 - 1))
    bad_labels = [bad, np.uint64(2**64 - 1)]
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    with pytest.raises(ValueError, match="NaN|inf"):
        model.fit(X, bad_labels * 12, validation_data=(X[:2], native[:2]))
    with pytest.raises(ValueError, match="NaN|inf"):
        model.fit(X, native, validation_data=(X[:2], bad_labels))
    with pytest.raises(ValueError, match="NaN|inf"):
        KernMD().fit(X, native).score(X[:2], bad_labels)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_zero_dimensional_numeric_ids_preserved_in_training_monitoring_and_score(dtype):
    high = 2**53 + 1
    X, exact = binary_data((0, high))
    boxed = [np.array(0.), np.array(high, dtype=dtype)]
    adjacent = [0., np.array(high - 1, dtype=dtype)]
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, boxed * 12, validation_data=(X[:2], boxed))
    assert model.classes_.tolist() == [0, high]
    assert model.predict(X).tolist() == exact.tolist()
    assert [r["score"] for r in model.validation_history_] == [1., 1.]
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, boxed * 12, validation_data=(X[:2], adjacent))
    assert KernMD().fit(X, exact).score(X[:2], boxed, sample_weight=[1, 3]) == 1.
    assert KernMD().fit(X, exact.astype(float)).score(X[:2], boxed, sample_weight=[1, 3]) == .25


def test_zero_dimensional_extended_float_keeps_original_precision_with_integer_id():
    high = 2**53 + 1
    X, _ = binary_data()
    lo = np.longdouble(.75)
    hi = np.array(high, dtype=np.uint64)
    classes = [np.array(lo), hi]
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, classes * 12, validation_data=(X[:2], classes))
    assert int(model.classes_[1]) == high
    assert [r["score"] for r in model.validation_history_] == [1., 1.]
    nearby = np.array(np.nextafter(lo, np.longdouble(1)))
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, classes * 12, validation_data=(X[:2], [nearby, hi]))


@pytest.mark.parametrize("number", [Decimal, Fraction])
@pytest.mark.parametrize("high", [3, 2**53 + 1])
def test_sparse_category_integral_decimal_and_fraction_fills(number, high):
    levels = pd.arrays.SparseArray(np.array([0, 2], dtype=np.float32), fill_value=0)
    try:
        levels.fill_value = number(high)
    except (ValueError, TypeError):
        pytest.skip("pandas rejects this sparse fill at construction")
    categories = pd.Index(levels)
    train = pd.Categorical.from_codes([1, 0] * 8, categories=categories)
    assert train[1] == number(high)
    held = pd.Categorical.from_codes([1, 0], categories=categories)
    X = np.tile([[-1.], [1.]], (8, 1))
    model = KernGDWD(stopping="validation", initialization="zero", max_iter=2)
    model.fit(X, train, validation_data=(X[:2], held))
    assert model.classes_.tolist() == [2, high]
    assert [r["score"] for r in model.validation_history_] == [1., 1.]
    assert model.score(X, train) == 1.
    unknown = np.array([2, high + 1], dtype=np.uint64)
    assert model.score(X[:2], unknown, sample_weight=[1, 3]) == .25
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, train, validation_data=(X[:2], unknown))
