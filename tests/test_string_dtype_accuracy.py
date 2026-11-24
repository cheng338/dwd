"""Public accuracy with labels whose NumPy dtype cannot carry metadata."""
import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from dwd.kern_md import KernMD


pytestmark = pytest.mark.skipif(
    not hasattr(np.dtypes, "StringDType"),
    reason="NumPy has no variable-width string dtype",
)


def text_target(rows, form, calls):
    if form == "object":
        return np.array(rows, dtype=object)
    strings = np.array(rows, dtype=np.dtypes.StringDType())
    if form == "numpy":
        return strings
    pd = pytest.importorskip("pandas")
    if form == "extension":
        return pd.arrays.NumpyExtensionArray(strings)
    base = pd.Series if form == "series" else pd.arrays.SparseArray

    class Projected(base):
        def __array__(self, dtype=None, copy=None):
            calls.append("array")
            return strings if dtype is None else strings.astype(dtype, copy=False)

        def __iter__(self):
            raise AssertionError("use the selected array projection")

    return Projected(pd.arrays.SparseArray(np.arange(len(rows)), fill_value=0))


@pytest.mark.parametrize("train,held", [
    ("numpy", "numpy"), ("object", "numpy"), ("numpy", "object"),
    ("object", "sparse"), ("sparse", "object"), ("series", "series"),
    ("extension", "extension"), ("object", "extension"), ("extension", "numpy"),
])
def test_variable_width_accuracy_preserves_rows_and_weights(train, held):
    x = np.array([[-2.], [2.], [-1.], [1.]])
    labels = ["q", "q\0", "q", "q\0"]
    train_calls, held_calls = [], []
    model = KernMD().fit(x, text_target(labels, train, train_calls))
    assert model.predict(x).tolist() == labels
    target = text_target(["q", "q", "q", "q\0"], held, held_calls)
    assert model.score(x, target, sample_weight=[2, 4, 1, 3]) == 0.6
    assert train_calls == (["array"] if train in {"sparse", "series"} else [])
    assert held_calls == (["array"] if held in {"sparse", "series"} else [])


@pytest.mark.parametrize("weights", [[1, 1], [0, 0, 0, 0]])
def test_variable_width_accuracy_keeps_weight_validation(weights):
    x = np.array([[-2.], [2.], [-1.], [1.]])
    y = np.array(["q", "q\0", "q", "q\0"], dtype=np.dtypes.StringDType())
    model = KernMD().fit(x, y)
    boxed = np.asarray(y, dtype=object)
    with pytest.raises((ValueError, ZeroDivisionError)) as control:
        accuracy_score(boxed, boxed, sample_weight=weights)
    with pytest.raises(type(control.value)) as actual:
        model.score(x, y, sample_weight=weights)
    assert str(actual.value) == str(control.value)


@pytest.mark.parametrize("missing", [None, np.nan])
def test_variable_width_accuracy_checks_missing_rows(missing):
    x = np.array([[-2.], [2.], [-1.], [1.]])
    y = np.array(["q", "q\0", "q", "q\0"], dtype=object)
    model = KernMD().fit(x, y)
    held = np.array(["q", missing, "q", "q\0"], dtype=np.dtypes.StringDType(na_object=missing))
    with pytest.raises((ValueError, TypeError)) as control:
        accuracy_score(np.asarray(held, dtype=object), model.predict(x))
    with pytest.raises(type(control.value)) as actual:
        model.score(x, held)
    assert str(actual.value) == str(control.value)
