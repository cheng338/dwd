"""KernMD preserves exact string class IDs through validation, fitting and scoring."""
from contextlib import nullcontext
from decimal import Decimal
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.utils.validation import check_is_fitted

from dwd.gen_dwd import GenDWD
from dwd.gen_kern_dwd import KernGDWD
from dwd.kern_md import KernMD


X = np.array([[-3.], [3.], [-2.], [2.], [-1.], [1.], [-0.5], [0.5]])
LABELS = ["q", "q\0"] * 4
requires_string_dtype = pytest.mark.skipif(
    not hasattr(np.dtypes, "StringDType"), reason="NumPy has no StringDType"
)


@requires_string_dtype
def test_string_classes_survive_fit_and_pickle():
    y = np.array(LABELS, dtype=np.dtypes.StringDType())
    model = KernMD().fit(X, y)
    assert model.classes_.dtype == y.dtype
    assert model.classes_.tolist() == ["q", "q\0"]
    assert model.predict(X).tolist() == LABELS
    assert model.score(X, y) == 1.0

    restored = pickle.loads(pickle.dumps(model))
    assert restored.classes_.dtype == y.dtype
    assert restored.predict(X[::-1]).tolist() == LABELS[::-1]


@requires_string_dtype
def test_string_target_validation_rejects_invalid_fit_inputs():
    y = np.array(LABELS, dtype=np.dtypes.StringDType())
    with pytest.raises(ValueError):
        KernMD().fit(X, y[:-1])
    nonfinite = X.copy()
    nonfinite[0] = np.nan
    with pytest.raises(ValueError):
        KernMD().fit(nonfinite, y)
    with pytest.raises(ValueError):
        KernMD().fit(X, np.column_stack((y, y)))


@pytest.mark.parametrize("dtype, rows", [
    (object, ["left", "right"] * 4), (str, ["left", "right"] * 4), (np.int64, [1, 2] * 4),
], ids=["object", "unicode", "int64"])
def test_ordinary_label_dtype_is_preserved(dtype, rows):
    y = np.array(rows, dtype=dtype)
    model = KernMD().fit(X, y)
    assert model.classes_.dtype == y.dtype
    assert model.predict(X).tolist() == rows


@requires_string_dtype
@pytest.mark.parametrize("missing", [float("nan"), np.float32("nan"), Decimal("NaN"),
                                     object(), None, "pandas.NA"],
                         ids=["nan", "numpy-nan", "decimal-nan", "object", "none", "pandas-na"])
def test_only_an_observed_string_sentinel_invalidates_fit(missing):
    if isinstance(missing, str):
        missing = pytest.importorskip("pandas").NA
    dtype = np.dtypes.StringDType(na_object=missing)
    model = KernMD().fit(X, np.array(LABELS, dtype=dtype))
    assert model.classes_.dtype == dtype
    assert model.predict(X).tolist() == LABELS
    with pytest.raises(ValueError, match="missing"):
        model.fit(X, np.array([missing, "right"] * 4, dtype=dtype))
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@requires_string_dtype
def test_string_valued_sentinel_is_a_class_name():
    dtype = np.dtypes.StringDType(na_object="NA")
    rows = ["NA", "right"] * 4
    model = KernMD().fit(X, np.array(rows, dtype=dtype))
    assert model.classes_.dtype == dtype
    assert model.classes_.tolist() == ["NA", "right"]
    assert model.predict(X).tolist() == rows


@requires_string_dtype
@pytest.mark.parametrize("estimator", ["KernMD", "GenDWD", "KernGDWD", "DWD", "SVM"])
def test_binary_fit_uses_complete_embedded_nul_labels(estimator):
    if estimator in ("DWD", "SVM"):
        cp = pytest.importorskip("cvxpy", exc_type=ImportError)
        if "CLARABEL" not in cp.installed_solvers():
            pytest.skip("optional CLARABEL solver unavailable")
        from dwd.socp_dwd import DWD
        from dwd.svm import SVM
        options = dict(solver_kws={"solver": "CLARABEL"})
        model = DWD(C="auto", **options) if estimator == "DWD" else SVM(C=.1, **options)
    else:
        model = {"KernMD": KernMD(), "GenDWD": GenDWD(stopping="fixed", max_iter=8),
                 "KernGDWD": KernGDWD(stopping="fixed", max_iter=8)}[estimator]
    rows = ["id\0B", "id\0A"] * 4
    y = np.array(rows, dtype=np.dtypes.StringDType(coerce=False, na_object="id\0A"))
    control = clone(model).fit(X, y.astype(object))
    model.fit(X, y)
    assert model.classes_.dtype == y.dtype
    assert model.classes_.tolist() == ["id\0A", "id\0B"]
    predictions = model.predict(X)
    assert predictions.dtype == y.dtype
    assert predictions.tolist() == control.predict(X).tolist() == rows
    held = y.copy()
    held[0] = "id\0C"
    weights = [7, 1, 1, 1, 1, 1, 1, 1]
    assert model.score(X, held, sample_weight=weights) == .5
    if estimator == "DWD":
        assert model.C_ == control.C_


@requires_string_dtype
@pytest.mark.parametrize("train_dtype, held_dtype", [("T", "T"), (object, "T"), ("T", object)])
def test_kernel_validation_uses_complete_embedded_nul_labels(train_dtype, held_dtype):
    rows = ["id\0B", "id\0A"] * 4
    train = np.array(rows, dtype=train_dtype)
    held = np.array(rows, dtype=held_dtype)
    model = KernGDWD(stopping="validation", max_iter=3)
    model.fit(X, train, validation_data=(X, held))
    assert all(row["score"] == 1. for row in model.validation_history_)
    held[0] = "id\0C"
    with pytest.raises(ValueError, match="Validation labels must belong"):
        model.fit(X, train, validation_data=(X, held))


@pytest.mark.parametrize("sequence", [list, tuple])
@pytest.mark.parametrize("column", [False, True])
def test_unicode_sequence_fit_and_score_preserve_trailing_nuls(sequence, column):
    def target(rows):
        return sequence(sequence([value]) for value in rows) if column else sequence(rows)

    with pytest.warns(DataConversionWarning) if column else nullcontext():
        model = KernMD().fit(X, target(LABELS))
    assert model.classes_.tolist() == ["q", "q\0"]
    assert model.predict(X).tolist() == LABELS
    assert model.score(X, target(LABELS)) == 1.
    assert model.score(X, target(["q\0", "q\0\0"] * 4)) == 0.
    ordinary = ["left", "right"] * 4
    with pytest.warns(DataConversionWarning) if column else nullcontext():
        model.fit(X, target(ordinary))
    assert model.classes_.dtype == np.asarray(ordinary).dtype


@requires_string_dtype
@pytest.mark.parametrize("sequence, column", [(list, False), (tuple, False), (list, True), (tuple, True)])
def test_native_string_fit_recognizes_sequence_observations(sequence, column):
    def target(rows):
        return sequence(sequence([value]) for value in rows) if column else sequence(rows)

    y = np.array(LABELS, dtype="T")
    model = KernGDWD(stopping="validation", max_iter=3)
    with pytest.warns(DataConversionWarning) if column else nullcontext():
        model.fit(X, y, validation_data=(X, target(y.tolist())))
    assert model.validation_history_ and all(row["score"] == 1. for row in model.validation_history_)
    assert model.score(X, target(y.tolist())) == 1.

    ordinary = np.array(["q", "r"] * 4, dtype="T")
    absent = target([value + "\0" for value in ordinary])
    model.fit(X, ordinary, validation_data=(X, ordinary))
    assert model.score(X, absent) == 0.
    with pytest.warns(DataConversionWarning) if column else nullcontext():
        with pytest.raises(ValueError, match="Validation labels must belong"):
            model.fit(X, ordinary, validation_data=(X, absent))


def _native_array_rows(values, column):
    rows = values[:, None]
    return list(rows) if column else tuple(row.reshape(()) for row in rows)


@requires_string_dtype
@pytest.mark.parametrize("column", [False, True], ids=["scalar-arrays", "singleton-rows"])
def test_native_array_rows_retain_training_and_observation_labels(column):
    rows = ["id\0A\0", "id\0B\0"] * 4
    dtype = np.dtypes.StringDType(coerce=False, na_object=None)
    y = np.array(rows, dtype=dtype)
    target = _native_array_rows(y, column)
    model = KernGDWD(stopping="validation", max_iter=3)
    with pytest.warns(DataConversionWarning) if column else nullcontext():
        model.fit(X, target, validation_data=(X, target))
    assert model.classes_.dtype == dtype
    assert model.classes_.tolist() == rows[:2]
    assert model.predict(X).tolist() == rows
    assert model.validation_history_ and all(row["score"] == 1. for row in model.validation_history_)
    assert model.score(X, target) == 1.

    held = y.copy()
    held[0] = "id\0A"
    absent = _native_array_rows(held, column)
    assert model.score(X, absent, sample_weight=[7, 1, 1, 1, 1, 1, 1, 1]) == .5
    with pytest.warns(DataConversionWarning) if column else nullcontext():
        with pytest.raises(ValueError, match="Validation labels must belong"):
            model.fit(X, y, validation_data=(X, absent))

    model = KernMD().fit(X, y)
    held[0] = None
    with pytest.warns(DataConversionWarning) if column else nullcontext():
        with pytest.raises(ValueError, match="missing"):
            model.fit(X, _native_array_rows(held, column))
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@requires_string_dtype
def test_native_array_rows_enforce_target_dimensions_and_sample_count():
    y = np.array(LABELS, dtype="T")
    with pytest.raises(ValueError):
        KernMD().fit(X, list(np.column_stack((y, y))))
    with pytest.raises(ValueError):
        KernMD().fit(X, _native_array_rows(y[:-1], False))


class InheritedString(str):
    pass


class InheritedNumpyString(np.str_):
    pass


@pytest.mark.parametrize("array_dtype", ["U", pytest.param("T", marks=requires_string_dtype)])
@pytest.mark.parametrize("scalar_type", [str, np.str_, InheritedString, InheritedNumpyString])
@pytest.mark.parametrize("column", [False, True])
@pytest.mark.parametrize("array_first", [False, True])
def test_mixed_text_rows_keep_observable_trailing_nuls(array_dtype, scalar_type, column, array_first):
    dtype = np.dtypes.StringDType(na_object=None, coerce=False) if array_dtype == "T" else str

    def target(value, scalar_factory=scalar_type):
        array = np.array("q", dtype=dtype)
        scalar = scalar_factory(value)
        pair = [array.reshape(1), [scalar]] if column else [array, scalar]
        return (pair if array_first else pair[::-1]) * 4

    y = target("q\0")
    expected = (["q", "q\0"] if array_first else ["q\0", "q"]) * 4
    model = KernGDWD(stopping="validation", max_iter=3)
    with pytest.warns(DataConversionWarning) if column else nullcontext():
        model.fit(X, y, validation_data=(X, y))
    if array_dtype == "T":
        # Some NumPy versions reject a passive str subclass beside a strict
        # native string box. The equivalent plain scalar determines the dtype.
        try:
            expected_dtype = np.asarray(y).dtype
        except ValueError as error:
            assert str(error) == "StringDType only allows string data when string coercion is disabled."
            expected_dtype = np.asarray(target("q\0", str)).dtype
        assert model.classes_.dtype == expected_dtype
        assert model.classes_.dtype.na_object is None
    assert model.classes_.tolist() == ["q", "q\0"]
    assert model.predict(X).tolist() == expected
    assert model.score(X, y) == 1.
    assert model.validation_history_ and all(row["score"] == 1. for row in model.validation_history_)

    absent = target("q\0\0")
    assert model.score(X, absent) == .5
    with pytest.warns(DataConversionWarning) if column else nullcontext():
        with pytest.raises(ValueError, match="Validation labels must belong"):
            model.fit(X, y, validation_data=(X, absent))
