"""Exact sparse feature values reaching the public shared-fold search."""
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from dwd.model_selection import fit_shared_fold_search


pytestmark = pytest.mark.usefixtures("search_parent_native_thread")


pd = pytest.importorskip("pandas")
_VALUES = [2**53 + 1, 0, 2**64 - 1, 0, 0, 2**63 + 1, 0, 2**53 + 3]


class _CheckSparseRows(BaseEstimator):
    def __init__(self, parity=False):
        self.parity = parity

    def _check(self, X):
        assert type(X) is pd.DataFrame
        assert X.columns.tolist() == ["signal", "signal"]
        assert X.columns.name == "columns" and X.index.name == "rows"
        assert X.attrs == {"unit": "count"}
        expected = [_VALUES[row // 2] for row in X.index]
        assert [int(value) for value in X.iloc[:, 0]] == expected
        assert X.iloc[:, 1].tolist() == [row * 3 for row in X.index]

    def fit(self, X, y):
        self._check(X)
        self.fit_frame_ = X
        return self

    def predict(self, X):
        self._check(X)
        return np.array([int(value) % 2 if self.parity else 0 for value in X.iloc[:, 0]])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class _FrameSplitter:
    def __init__(self, X):
        self.original_id = id(X)

    def get_n_splits(self, X, y, groups=None):
        assert id(X) == self.original_id
        return 2

    def split(self, X, y, groups=None):
        assert id(X) == self.original_id
        left, right = np.array([3, 2, 0, 1]), np.array([6, 4, 7, 5])
        yield left, right
        yield right, left


@pytest.mark.parametrize("backend,n_jobs,fill,byte_order", [("loky", 1, 0, "="), ("threading", 2, 0.0, "="),
                                                           ("loky", 2, np.int64(0), "="), ("loky", 1, 0, "S")])
def test_exact_sparse_feature_rows_and_original_parent_frame(backend, n_jobs, fill, byte_order):
    values = pd.arrays.SparseArray(np.array(_VALUES, dtype=np.dtype(np.uint64).newbyteorder(byte_order)), fill_value=fill)
    rows = 2 * np.arange(len(values))
    frame = pd.DataFrame({"a": values, "b": 3 * rows}, index=pd.Index(rows, name="rows"))
    frame.columns = pd.Index(["signal", "signal"], name="columns")
    frame.attrs["unit"] = "count"
    target = np.array([v % 2 for v in _VALUES])
    search = GridSearchCV(_CheckSparseRows(), {"parity": [False, True]},
                          n_jobs=n_jobs, cv=_FrameSplitter(frame), error_score="raise")
    options = {"inner_max_num_threads": 1} if backend == "loky" else {}
    with parallel_config(backend=backend, **options):
        assert fit_shared_fold_search(search, frame, target) is search
    np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.5, 1.])
    assert search.best_params_ == {"parity": True}
    assert search.best_estimator_.fit_frame_ is frame
    assert type(frame.iloc[:, 0].array.fill_value) is type(fill)
    np.testing.assert_array_equal(search.predict(frame), target)


class _TargetIdentitySplitter:
    def __init__(self, original):
        self.original = original
        self.calls = 0

    def get_n_splits(self, X, y, groups=None):
        return 1

    def split(self, X, y, groups=None):
        self.calls += 1
        left, right = np.array([0, 2]), np.array([1, 3])
        yield (left, right) if y is self.original else (right, left)


class _SelectByRow(BaseEstimator):
    def __init__(self, parity=0):
        self.parity = parity

    def fit(self, X, y):
        self.y_ = y
        return self

    def score(self, X, y):
        return np.mean(X[:, 0] % 2 == self.parity)


def test_search_rejects_only_lossy_target_before_custom_split():
    X = np.arange(4).reshape(-1, 1)
    values = np.array([0, 2**64 - 1, 0, 2**53 + 1], dtype=np.uint64)
    y = pd.Series(pd.arrays.SparseArray(values, fill_value=0), index=["b", "a", "a", "b"], name="label")
    cv = _TargetIdentitySplitter(y)
    search = GridSearchCV(_SelectByRow(), {"parity": [0, 1]}, cv=cv, error_score="raise")
    with pytest.raises(ValueError, match="loses class IDs"):
        fit_shared_fold_search(search, X, y)
    assert cv.calls == 0

    # An automatic replacement at fit entry would switch the winning candidate.
    dense = pd.Series(y.to_numpy(dtype=np.uint64, copy=True), index=y.index, name=y.name)
    cv.original = dense
    fit_shared_fold_search(search, X, dense)
    assert search.best_params_ == {"parity": 1} and search.best_estimator_.y_ is dense
    fit_shared_fold_search(search, X, dense.copy())
    assert search.best_params_ == {"parity": 0}

    for exact in [pd.Series(pd.arrays.SparseArray(values, fill_value=np.uint64(0))),
                  pd.Series(pd.arrays.SparseArray(np.array([0, 2**54] * 2, dtype=np.uint64), fill_value=0))]:
        cv.original = exact
        fit_shared_fold_search(search, X, exact)
        assert search.best_params_ == {"parity": 1} and search.best_estimator_.y_ is exact


class _FoldDtype(BaseEstimator):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def fit(self, X, y):
        return self

    def score(self, X, y):
        return float(str(X.iloc[:, 0].dtype) == self.dtype)


def test_representable_sparse_features_preserve_native_fold_dtype():
    values = pd.arrays.SparseArray(np.array([0, 2**54] * 2, dtype=np.uint64), fill_value=0)
    frame = pd.DataFrame({"value": values})
    rows = np.array([0, 1])
    native = str(frame.iloc[rows, 0].dtype)
    search = GridSearchCV(_FoldDtype(), {"dtype": ["unselected", native]}, cv=[(rows, rows + 2)],
                          n_jobs=1, error_score="raise")
    fit_shared_fold_search(search, frame, np.arange(4))
    assert search.best_params_ == {"dtype": native}


@pytest.mark.filterwarnings("ignore:Allowing arbitrary scalar fill_value")
@pytest.mark.filterwarnings("ignore:The 'sparse_index' parameter")
@pytest.mark.parametrize("fill", [-1, np.nan, np.complex128(0j)])
def test_mixed_fill_cannot_silently_round_integer_features(fill):
    try:
        dtype = pd.SparseDtype(np.uint64, fill)
    except ValueError:
        pytest.skip("this pandas version rejects this fill at construction")
    positions = pd.arrays.SparseArray([0, 1, 0, 1], fill_value=0).sp_index
    array = pd.arrays.SparseArray(np.array([2**53 + 1, 2**64 - 1], dtype=np.uint64),
                                  sparse_index=positions, dtype=dtype, fill_value=fill)
    frame = pd.DataFrame({"value": array})
    search = GridSearchCV(_SelectByRow(), {}, cv=2, n_jobs=1, error_score="raise")
    with pytest.raises(ValueError, match="sparse uint64 fill"):
        fit_shared_fold_search(search, frame, np.arange(4))


def test_extended_precision_target_uses_actual_platform_dtype():
    values = np.array([0, 2**53 + 1, 0, 2**64 - 1], dtype=np.uint64)
    y = pd.Series(pd.arrays.SparseArray(values, fill_value=np.longdouble(0)))
    X = np.arange(4).reshape(-1, 1)
    search = GridSearchCV(_SelectByRow(), {}, cv=_TargetIdentitySplitter(y), error_score="raise")
    converted = np.asarray(y)
    if [int(v) for v in converted] == [int(v) for v in values]:
        fit_shared_fold_search(search, X, y)
        assert search.best_estimator_.y_ is y
    else:
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(search, X, y)


class _ExactParity(BaseEstimator):
    def __init__(self, parity=False):
        self.parity = parity

    def fit(self, X, y):
        return self

    def score(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        else:
            X = X[:, 0]
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        return np.mean([(int(x) % 2 if self.parity else 0) == int(v) % 2 for x, v in zip(X, y)])


def _parity_search():
    return GridSearchCV(_ExactParity(), {"parity": [False, True]}, cv=2, n_jobs=1, error_score="raise")


def _retag_fill(data, fill):
    # Construct while the caller's integers are exact, then change only the
    # implicit fill. The stored values and positions are untouched.
    array = pd.arrays.SparseArray(np.array(data, dtype=np.uint64), fill_value=np.uint64(0))
    array.fill_value = fill
    return array


@pytest.mark.parametrize("gaps", [False, True])
def test_float_fill_must_not_absorb_adjacent_integer(gaps):
    high = 2**53
    data = [high + 1, 0 if gaps else high + 2] * 4
    source = _retag_fill(data, float(high))
    reference = np.array([high + 1, high if gaps else high + 2] * 4, dtype=np.uint64)
    assert [int(value) for value in source] == reference.tolist()
    frame = pd.DataFrame({"value": source})
    search = _parity_search()
    fit_shared_fold_search(search, frame, reference)
    np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.5, 1.])
    assert search.best_params_ == {"parity": True}
    for y in (frame.iloc[:, 0], frame):
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(_parity_search(), reference[:, None], y)
    if not gaps:
        for y in (source, pd.Index(source)):
            search = _parity_search()
            fit_shared_fold_search(search, reference[:, None], y)
            np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.5, 1.])


@pytest.mark.parametrize("offsets,lossy", [([0, 3], True), ([2, 3], False), ([0, 4], False)])
def test_default_conversion_after_recompressing_an_explicit_fill(offsets, lossy):
    high = 2**53
    data = [high + k for k in offsets] * 4
    y = pd.Series(_retag_fill(data, float(high)))
    assert not y.array.sp_index.ngaps
    X = np.array(data, dtype=np.uint64)[:, None]
    if lossy:
        selected = y.take([0, 1])
        assert [int(v) for v in selected] == data[:2] and int(np.asarray(selected)[1]) != data[1]
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(_parity_search(), X, y)
    else:
        search = _parity_search()
        fit_shared_fold_search(search, X, y)
        assert search.cv_results_["mean_test_score"][1] == 1.


@pytest.mark.filterwarnings("ignore:Allowing arbitrary scalar fill_value")
def test_collision_at_uint64_upper_bound():
    data = [2**64 - 1, 2**53 + 2] * 4
    try:
        source = _retag_fill(data, float(2**64))
    except ValueError:
        pytest.skip("this pandas version rejects this fill at construction")
    with pytest.raises(ValueError, match="sparse uint64 fill"):
        fit_shared_fold_search(_parity_search(), pd.DataFrame({"x": source}), data)


@pytest.mark.filterwarnings("ignore:Allowing arbitrary scalar fill_value")
@pytest.mark.parametrize("fill", [str(2**53 + 1), str(2**53 + 1).encode()])
def test_text_fill_cannot_turn_stored_integer_into_the_same_token(fill):
    try:
        source = _retag_fill([0, 2**53 + 1] * 4, fill)
    except (TypeError, ValueError):
        pytest.skip("this pandas version rejects a text fill at construction")
    frame = pd.DataFrame({"value": source})
    assert isinstance(source[1], np.unsignedinteger) and source[0] == fill
    sample = frame.take([0, 1]).iloc[:, 0]
    if isinstance(sample.iloc[1], (bytes, str)):
        with pytest.raises(ValueError, match="sparse uint64 fill"):
            fit_shared_fold_search(_parity_search(), frame, np.arange(8))
    if isinstance(np.asarray(source)[1], (bytes, str)) or isinstance(sample.iloc[1], (bytes, str)):
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(_parity_search(), np.arange(8)[:, None], frame.iloc[:, 0])


@pytest.mark.filterwarnings("ignore:Allowing arbitrary scalar fill_value")
@pytest.mark.parametrize("text", ["class", "classcode-" + "9" * 28])
@pytest.mark.parametrize("as_bytes", [False, True])
def test_all_gap_index_target_keeps_the_entire_label(text, as_bytes):
    fill = text.encode() if as_bytes else text
    try:
        values = _retag_fill([0] * 8, fill)
    except (ValueError, TypeError):
        pytest.skip("this pandas version rejects a text fill at construction")
    y = pd.Index(values)
    assert y[0] == fill
    # A recording estimator also accepts bytes on sklearn versions where
    # classifiers reject byte labels independently of their storage.
    search = GridSearchCV(_SelectByRow(), {}, cv=2, error_score="raise")
    X = np.arange(8)[:, None]
    if np.asarray(y)[0] != fill:
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(search, X, y)
    else:
        fit_shared_fold_search(search, X, y)
        assert np.asarray(search.best_estimator_.y_).tolist() == [fill] * len(y)


@pytest.mark.filterwarnings("ignore:Allowing arbitrary scalar fill_value")
@pytest.mark.parametrize("fill", [-1, 1.5, np.nan])
def test_all_gap_selection_preserves_the_fill_or_refuses(fill):
    try:
        source = _retag_fill([0] * 4, fill)
    except ValueError:
        pytest.skip("this pandas version rejects this fill at construction")
    frame = pd.DataFrame({"value": source})
    with np.errstate(invalid="ignore", over="ignore"):
        try:
            observed = frame.take([0, 1]).iloc[0, 0]
            exact = pd.isna(observed) and pd.isna(fill) or observed == fill
        except (OverflowError, ValueError):
            exact = False
    if exact:
        expected = str(frame.take([0, 1]).iloc[:, 0].dtype)
        search = GridSearchCV(_FoldDtype(expected), {}, cv=2, error_score="raise")
        fit_shared_fold_search(search, frame, np.arange(4))
        assert search.best_score_ == 1.
    else:
        with pytest.raises(ValueError, match="sparse uint64 fill"):
            fit_shared_fold_search(_parity_search(), frame, np.arange(4))
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(_parity_search(), np.arange(4)[:, None], frame.iloc[:, 0])


@pytest.mark.filterwarnings("ignore:data_manager option is deprecated")
def test_legacy_array_manager_positional_take_keeps_adjacent_integer():
    try:
        pd.get_option("mode.data_manager")
    except pd.errors.OptionError:
        pytest.skip("ArrayManager is unavailable")
    data = [2**53 + 1, 2**53 + 2] * 4
    with pd.option_context("mode.data_manager", "array"):
        frame = pd.DataFrame({"value": _retag_fill(data, float(2**53))})
        for X, y in [(frame, data), (np.array(data, dtype=np.uint64)[:, None], frame.iloc[:, 0])]:
            search = _parity_search()
            fit_shared_fold_search(search, X, y)
            np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.5, 1.])


class _PanelParity(BaseEstimator):
    def __init__(self, signal=False, position=0, convert=False):
        self.signal = signal
        self.position = position
        self.convert = convert

    def fit(self, X, y):
        self.y_ = y
        return self

    def score(self, X, y):
        assert y.columns.tolist() == ["label"] * y.shape[1] and y.columns.name == "outputs"
        assert y.index.name == "rows"
        labels = np.asarray(y)[:, self.position] if self.convert else y.iloc[:, self.position]
        predicted = X[:, 0] if self.signal else np.zeros(len(y), dtype=int)
        return np.mean([int(v) % 2 == pred for v, pred in zip(labels, predicted)])


class _PanelSplits:
    def __init__(self, original):
        self.original = original
        self.calls = 0

    def get_n_splits(self, X, y, groups=None):
        assert y is self.original
        return 2

    def split(self, X, y, groups=None):
        assert y is self.original
        self.calls += 1
        left, right = np.array([3, 2, 0, 1]), np.array([4, 6, 7, 5])
        yield left, right
        yield right, left


@pytest.mark.parametrize("case", ["row_take", "signed_mix", "float_mix", "signed_signal",
                                   "fold_dense", "fold_sparse", "exact_unsigned", "exact_float"])
@pytest.mark.parametrize("position", [0, 1])
def test_multioutput_sparse_target_winner_and_owning_workaround(case, position):
    labels = np.array([0, 2**53 + 1, 0, 2**64 - 1] * 2, dtype=np.uint64)
    if case == "signed_signal":
        labels = np.array([0, -(2**53 + 1), 0, 2**63 - 1] * 2, dtype=np.int64)
    if case == "exact_float":
        labels = np.array([2**54, 1] * 4, dtype=np.uint64)
    signal = (labels if case in ("fold_dense", "signed_signal") else
              pd.arrays.SparseArray(labels, fill_value=0 if case == "row_take" else np.uint64(0)))
    ordinary = np.array([0, 1] * 4, dtype=np.uint64)
    if case == "signed_mix":
        ordinary = ordinary.astype(np.int64)
    elif case in ("float_mix", "exact_float"):
        ordinary = ordinary.astype(float)
    elif case in ("fold_dense", "fold_sparse", "signed_signal"):
        ordinary = pd.arrays.SparseArray(ordinary, fill_value=0)
    columns = (ordinary, signal) if position else (signal, ordinary)
    y = pd.DataFrame(dict(enumerate(columns)), index=pd.Index([3, 2, 3, 7] * 2, name="rows"))
    y.columns = pd.Index(["label", "label"], name="outputs")
    X = np.array([int(v) % 2 for v in labels])[:, None]
    convert = case != "row_take"
    for owning in (False, True):
        if owning:
            # Read each column's exact scalars. A dense uint64 column next
            # to an int64 or float64 column still promotes during conversion.
            data = np.empty(y.shape, dtype=object)
            for i in range(y.shape[1]):
                column = y.iloc[:, i].array
                data[:, i] = [column[row] for row in range(len(column))]
            target = pd.DataFrame(data, index=y.index, columns=y.columns, dtype=object)
        else:
            target = y
        cv = _PanelSplits(target)
        search = GridSearchCV(_PanelParity(position=position, convert=convert),
                              {"signal": [False, True]}, cv=cv, n_jobs=1, error_score="raise")
        native = GridSearchCV(_PanelParity(position=position, convert=convert),
                              {"signal": [False, True]}, cv=_PanelSplits(target), n_jobs=1, error_score="raise")
        native.fit(X, target)
        if native.best_params_ == {"signal": False}:
            assert not owning
            np.testing.assert_array_equal(native.cv_results_["mean_test_score"], [1., .5])
            with pytest.raises(ValueError, match="loses class IDs"):
                fit_shared_fold_search(search, X, target)
            assert cv.calls == 0
        else:
            fit_shared_fold_search(search, X, target)
            assert cv.calls == 1 and search.best_estimator_.y_ is target
            np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.5, 1.])
            assert search.best_params_ == {"signal": True}


def test_multicolumn_target_uses_frame_conversion_after_recompression():
    # Default conversion of the isolated first Series loses the odd integer
    # after take recompresses the explicit fill. This frame uses uint64 for
    # both columns, so its default conversion can preserve both values.
    high = 2**53
    labels = np.array([high, high + 3] * 4, dtype=np.uint64)
    y = pd.DataFrame({0: _retag_fill(labels, float(high)), 1: labels.copy()},
                     index=pd.Index(range(8), name="rows"))
    y.columns = pd.Index(["label", "label"], name="outputs")
    X = np.array([int(v) % 2 for v in labels])[:, None]
    search = GridSearchCV(_PanelParity(convert=True), {"signal": [False, True]},
                          cv=_PanelSplits(y), n_jobs=1, error_score="raise")
    fit_shared_fold_search(search, X, y)
    np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.5, 1.])
    assert search.best_estimator_.y_ is y


@pytest.mark.filterwarnings("ignore:Allowing arbitrary scalar fill_value")
@pytest.mark.parametrize("high_on_stored_rows", [False, True])
def test_fold_can_drop_the_gaps_that_would_select_an_object_frame(high_on_stored_rows):
    from decimal import Decimal

    marker = pd.arrays.SparseArray(np.array([1, 0] * 4, dtype=np.uint8), fill_value=np.uint8(0))
    try:
        marker.fill_value = Decimal(0)
    except (TypeError, ValueError):
        pytest.skip("this pandas version rejects the Decimal fill at construction")
    high = 2**53 + 1
    labels = np.array(([high, 0] if high_on_stored_rows else [0, high]) * 4, dtype=np.uint64)
    other = pd.arrays.SparseArray(np.array([0, 1] * 4, dtype=np.uint64), fill_value=0)
    target = pd.DataFrame({0: labels, 1: marker, 2: other}, index=pd.Index(range(8), name="rows"))
    target.columns = pd.Index(["label"] * 3, name="outputs")
    even, odd = np.arange(0, 8, 2), np.arange(1, 8, 2)
    X = np.array([int(v) % 2 for v in labels])[:, None]
    search = GridSearchCV(_PanelParity(convert=True), {"signal": [False, True]},
                          cv=[(even, odd), (odd, even)], n_jobs=1, error_score="raise")
    if any(int(actual) != int(expected) for actual, expected in zip(np.asarray(target.take(even))[:, 0], labels[even])):
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(search, X, target)
    else:
        fit_shared_fold_search(search, X, target)
        np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.5, 1.])
        assert search.best_estimator_.y_ is target


@pytest.mark.filterwarnings("ignore:Allowing arbitrary scalar fill_value")
@pytest.mark.parametrize("case", ["category_signed", "category_unsigned", "category_unused",
                                   "nullable_signed", "nullable_unsigned", "sparse_signed",
                                   "signed_fill_collision", "categorical_neighbor", "nullable_neighbor",
                                   "integer_fill_float_payload"])
def test_extension_target_columns_compare_observed_labels_through_native_conversion(case):
    high = 2**53 + 1
    mask = np.array([0, 2] * 4, dtype=np.uint64)
    ordinary = pd.arrays.SparseArray(mask, fill_value=np.uint64(0))
    columns = [ordinary]
    if case.startswith("category_"):
        typ = np.int64 if case == "category_signed" else np.uint64
        odd = -high if typ is np.int64 else high
        categories = np.array([0, 1, odd], dtype=typ)
        labels = pd.Categorical.from_codes([0, 1 if case == "category_unused" else 2] * 4, categories=categories)
        columns[0] = pd.arrays.SparseArray(mask, fill_value=0)
    elif case in ("nullable_signed", "nullable_unsigned"):
        labels = pd.array([0, -high if case == "nullable_signed" else high] * 4,
                          dtype="Int64" if case == "nullable_signed" else "UInt64")
    elif case in ("sparse_signed", "signed_fill_collision"):
        labels = pd.arrays.SparseArray(np.array([0, -high] * 4, dtype=np.int64), fill_value=np.int64(0))
        labels.fill_value = -float(2**53) if case == "signed_fill_collision" else 0.
        columns.append(pd.Categorical(["a", "b"] * 4))
    elif case == "integer_fill_float_payload":
        labels = pd.arrays.SparseArray(np.array([0., 2.] * 4), fill_value=0.)
        try:
            labels.fill_value = np.uint64(high)
        except ValueError:
            pytest.skip("this pandas version rejects the integer fill at construction")
    else:
        labels = pd.arrays.SparseArray(np.array([0, high] * 4, dtype=np.uint64), fill_value=0)
        columns.append(pd.Categorical(["a", "b"] * 4) if case == "categorical_neighbor" else
                       pd.array(mask, dtype="UInt8"))
    output = len(columns)
    columns.append(labels)
    target = pd.DataFrame(dict(enumerate(columns)), index=pd.Index([5, 7, 5, 2] * 2, name="rows"))
    target.columns = pd.Index(["label"] * len(columns), name="outputs")
    reference = [int(labels[i]) for i in range(len(labels))]
    X = np.array([v % 2 for v in reference])[:, None]
    rows = np.arange(len(labels))[::-1]
    # Scalar indexing reads the exact logical value even when iteration or
    # conversion of a categorical column uses an intermediate dense array.
    batches = [(target, range(len(labels))), (target.take(rows), rows)]
    lossy = any(int(v) != reference[r] for frame, order in batches
                for i, r in enumerate(order) for v in (frame.iat[i, output], np.asarray(frame)[i, output]))
    cv = _PanelSplits(target)
    search = GridSearchCV(_PanelParity(position=output, convert=True), {"signal": [False, True]},
                          cv=cv, n_jobs=1, error_score="raise")
    if lossy:
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(search, X, target)
        assert cv.calls == 0
    else:
        fit_shared_fold_search(search, X, target)
        np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.5, 1.])
        assert search.best_params_ == {"signal": True} and search.best_estimator_.y_ is target


class _MatchOutput(BaseEstimator):
    def __init__(self, signal=False, label=None):
        self.signal = signal
        self.label = label

    def fit(self, X, y):
        self.y_ = y
        return self

    def score(self, X, y):
        observed = np.asarray(y).reshape(len(y), -1)[:, -1]
        predicted = X[:, 0] if self.signal else np.zeros(len(y), dtype=int)
        return np.mean([int(_label_key(v) == self.label) == p for v, p in zip(observed, predicted)])


def _label_key(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.inexact) and not value.imag and np.isfinite(value) and value.real == np.trunc(value.real):
        return int(value.real)
    return value.item() if isinstance(value, np.generic) else value


@pytest.mark.parametrize("marker,fill,high,used,missing", [
    ("sparse", np.uint64(0), 3, True, False),
    ("sparse", np.uint64(0), 2**53 + 1, True, False),
    ("dense", 0, 2**53 + 1, True, False),
    ("dense", 0, 2**64 - 1, True, True),
    ("dense", 0, 2**53 + 1, False, False),
    ("dense", 0, 2**53 + 1, False, True),
])
def test_sparse_categorical_target_uses_observed_levels_and_frame_conversion(marker, fill, high, used, missing):
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint64)
    levels = pd.Index(pd.arrays.SparseArray(np.array([0, 1, high], dtype=np.uint64), fill_value=fill))
    codes = bits.astype(int) * (2 if used else 1)
    if missing:
        codes[[0, 6]] = -1
    labels = pd.Categorical.from_codes(codes, categories=levels)
    first = pd.arrays.SparseArray(bits, fill_value=np.uint64(0)) if marker == "sparse" else bits
    y = pd.DataFrame({0: first, 1: labels}, index=pd.Index([8, 9, 8, 7] * 2, name="rows"))
    y.columns = pd.Index(["label", "label"], name="outputs")
    # Build the owning control directly from positional scalars. Explicit
    # object conversion of the whole categorical can already round an ID.
    control = pd.DataFrame([[y.iat[i, 0], y.iat[i, 1]] for i in range(len(y))],
                           index=y.index, columns=y.columns, dtype=object)
    def search(target):
        return GridSearchCV(_MatchOutput(label=high if used else 1), {"signal": [False, True]},
                            cv=_PanelSplits(target), n_jobs=1, error_score="raise")

    exact = fit_shared_fold_search(search(control), bits[:, None], control)
    np.testing.assert_array_equal(exact.cv_results_["mean_test_score"], [.5, 1.])
    assert exact.best_params_ == {"signal": True} and exact.best_estimator_.y_ is control
    native = search(y).fit(bits[:, None], y)
    checked = search(y)
    expected = [_label_key(labels[i]) for i in range(len(labels))]
    with np.errstate(invalid="ignore"):
        lossy = any(_label_key(v) != expected[row] for rows in (np.arange(len(y)), np.arange(len(y))[::-1])
                    for row, v in zip(rows, np.asarray(y.take(rows))[:, 1]))
    if lossy:
        if native.best_params_ == {"signal": False}:
            np.testing.assert_array_equal(native.cv_results_["mean_test_score"], [1., .5])
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(checked, bits[:, None], y)
        assert checked.cv.calls == 0
    else:
        fit_shared_fold_search(checked, bits[:, None], y)
        np.testing.assert_array_equal(checked.cv_results_["mean_test_score"], exact.cv_results_["mean_test_score"])
        assert checked.best_params_ == {"signal": True} and checked.best_estimator_.y_ is y


@pytest.mark.parametrize("carrier", [pd.Categorical, pd.CategoricalIndex, pd.Series, pd.DataFrame])
@pytest.mark.parametrize("fill,used,missing", [(0, True, False), (np.uint64(0), True, False),
                                              (0, False, True), (np.uint64(0), True, True)])
def test_single_output_sparse_categorical_uses_the_carriers_native_conversion(carrier, fill, used, missing):
    high = 2**53 + 1
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    levels = pd.Index(pd.arrays.SparseArray(np.array([0, 1, high], dtype=np.uint64), fill_value=fill))
    codes = bits * (2 if used else 1)
    if missing:
        codes[[0, 6]] = -1
    cat = pd.Categorical.from_codes(codes, categories=levels)
    target = pd.DataFrame({"label": cat}) if carrier is pd.DataFrame else carrier(cat)
    search = GridSearchCV(_MatchOutput(label=high if used else 1), {"signal": [False, True]},
                          cv=_PanelSplits(target), n_jobs=1, error_score="raise")
    native = GridSearchCV(search.estimator, search.param_grid, cv=_PanelSplits(target),
                          n_jobs=1, error_score="raise").fit(bits[:, None], target)
    expected = [_label_key(cat[i]) for i in range(len(cat))]
    with np.errstate(invalid="ignore"):
        lossy = any(_label_key(v) != expected[row] for rows in (np.arange(len(cat)), np.arange(len(cat))[::-1])
                    for row, v in zip(rows, np.asarray(target.take(rows)).reshape(-1)))
    if lossy:
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(search, bits[:, None], target)
        assert search.cv.calls == 0
    else:
        fit_shared_fold_search(search, bits[:, None], target)
        np.testing.assert_array_equal(search.cv_results_["mean_test_score"], native.cv_results_["mean_test_score"])
        np.testing.assert_array_equal(native.cv_results_["mean_test_score"], [.5, 1.])
        assert search.best_params_ == {"signal": True} and search.best_estimator_.y_ is target


@pytest.mark.filterwarnings("ignore:Allowing arbitrary scalar fill_value")
@pytest.mark.filterwarnings("ignore:Concatenating sparse arrays with multiple fill values")
@pytest.mark.parametrize("layout", ["vector", "dense_marker", "sparse_category_marker", "longdouble", "clongdouble"])
@pytest.mark.parametrize("dtype,stored,fill,signal_code", [
    (np.float32, 2, np.uint64(2**53 + 1), 0),
    (np.float32, 2, np.uint64(3), 0),
    (np.float32, 2, float(2**54 + 4), 0),
    (np.float32, 2, Fraction(2**53 + 1), 0),
    (np.float32, 2, Decimal(2**53 + 1), 0),
    (np.float32, 2, Fraction(3), 0),
    (np.float32, 2, Decimal(3), 0),
    (np.int64, -(2**53 + 1), 0., 1),
    (np.int64, -(2**53 + 1), np.int64(0), 1),
])
def test_integer_class_id_in_sparse_category_point_or_fill(layout, dtype, stored, fill, signal_code):
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    levels = pd.arrays.SparseArray(np.array([0, stored], dtype=dtype), fill_value=dtype(0))
    try:
        levels.fill_value = fill
    except (TypeError, ValueError):
        pytest.skip("this pandas version rejects the mixed fill at construction")
    category = pd.Categorical.from_codes(bits if signal_code else 1 - bits, categories=pd.Index(levels))
    label = int(stored if signal_code else fill)
    expected = [_label_key(category[i]) for i in range(len(bits))]
    assert [int(v == label) for v in expected] == bits.tolist()
    if layout == "vector":
        y = pd.Series(category)
        owning = pd.Series(expected, dtype=object)
    else:
        marker = bits
        if layout == "sparse_category_marker":
            levels = pd.arrays.SparseArray(np.array([0, 5, 6], dtype=np.uint64), fill_value=np.uint64(0))
            marker = pd.Categorical.from_codes(bits, categories=pd.Index(levels))
        elif layout in ("longdouble", "clongdouble"):
            marker = bits.astype(layout)
        y = pd.DataFrame({"marker": marker, "label": category})
        owning = pd.DataFrame([[y.iat[i, 0], expected[i]] for i in range(len(bits))], columns=y.columns, dtype=object)

    def search(target):
        return GridSearchCV(_MatchOutput(label=label), {"signal": [False, True]},
                            cv=_PanelSplits(target), n_jobs=1, error_score="raise")

    native = search(y).fit(bits[:, None], y)
    checked = search(y)
    exact = fit_shared_fold_search(search(owning), bits[:, None], owning)
    np.testing.assert_array_equal(exact.cv_results_["mean_test_score"], [.5, 1.])
    assert exact.best_params_ == {"signal": True} and exact.best_estimator_.y_ is owning
    selections = (np.arange(len(bits))[::-1], np.flatnonzero(bits), np.flatnonzero(1 - bits))
    loses = any(_label_key(v) != expected[row] for rows in selections
                for row, v in zip(rows, np.asarray(y.take(rows)).reshape(len(rows), -1)[:, -1]))
    if loses:
        if native.best_params_ == {"signal": False}:
            np.testing.assert_array_equal(native.cv_results_["mean_test_score"], [1., .5])
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(checked, bits[:, None], y)
        assert checked.cv.calls == 0
    else:
        fit_shared_fold_search(checked, bits[:, None], y)
        np.testing.assert_array_equal(checked.cv_results_["mean_test_score"], [.5, 1.])
        assert checked.best_params_ == {"signal": True} and checked.best_estimator_.y_ is y


@pytest.mark.filterwarnings("ignore:Allowing arbitrary scalar fill_value")
@pytest.mark.parametrize("dtype,stored,fill,label", [
    (np.float32, 2., float(2**54 + 4), float(2**54 + 4)),
    (np.float32, 2., float(2**54), float(2**54)),
    (np.complex64, 2 + 1j, complex(2**54 + 4, 1), complex(2**54 + 4, 1)),
    (np.float64, 1., "1.0", 1.),
    (np.bool_, True, "True", True),
])
def test_sparse_companion_payload_and_fill_keep_distinct_labels(dtype, stored, fill, label):
    marker = pd.arrays.SparseArray(np.array([0, 1] * 4, dtype=np.uint64), fill_value=np.uint64(0))
    values = pd.arrays.SparseArray(np.array([0, stored] * 4, dtype=dtype), fill_value=dtype(0))
    try:
        values.fill_value = fill
    except (TypeError, ValueError):
        pytest.skip("this pandas version rejects the mixed fill at construction")
    target = pd.DataFrame({0: marker, 1: pd.Categorical(["a", "b"] * 4), 2: values})
    X = np.array([int((v.item() if isinstance(v, np.generic) else v) == label)
                  for v in (values[i] for i in range(len(values)))])[:, None]
    assert X.sum() == 4
    cv = _PanelSplits(target)
    def search(splitter):
        return GridSearchCV(_MatchOutput(label=label), {"signal": [False, True]},
                            cv=splitter, n_jobs=1, error_score="raise")
    native = search(_PanelSplits(target)).fit(X, target)
    guarded = search(cv)
    if native.best_params_ == {"signal": False}:
        np.testing.assert_array_equal(native.cv_results_["mean_test_score"], [1., .5])
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(guarded, X, target)
        assert cv.calls == 0
    else:
        fit_shared_fold_search(guarded, X, target)
        np.testing.assert_array_equal(guarded.cv_results_["mean_test_score"], [.5, 1.])
        assert guarded.best_params_ == {"signal": True} and guarded.best_estimator_.y_ is target


@pytest.mark.parametrize("case", ["native", "python_fill", "narrow_fill", "swapped", "signed", "nullable"])
def test_only_exact_native_uint64_frames_bypass_value_probes(case, monkeypatch):
    import dwd.model_selection as selection

    values = np.array([0, 2**53 + 1] * 4, dtype=np.uint64)
    sparse = pd.arrays.SparseArray(values, fill_value=np.uint64(0))
    second = values.copy()
    if case in ("python_fill", "narrow_fill"):
        sparse.fill_value = 0 if case == "python_fill" else np.uint8(0)
    elif case == "swapped":
        sparse = pd.arrays.SparseArray(values.astype(values.dtype.newbyteorder("S")), fill_value=np.uint64(0))
    elif case == "signed":
        second = second.astype(np.int64)
    elif case == "nullable":
        second = pd.array(second, dtype="UInt64")
    target = pd.DataFrame({0: sparse, 1: second, 2: values.copy()})
    called = []
    probe = selection._frame_numeric_probe
    def recording_probe(array, pd):
        called.append(array.dtype)
        return probe(array, pd)
    monkeypatch.setattr(selection, "_frame_numeric_probe", recording_probe)
    loses = selection._multioutput_sparse_target_loses(target, pd)
    assert bool(called) == (case != "native")
    if case == "native":
        assert not loses
        cv = _PanelSplits(target)
        search = GridSearchCV(_MatchOutput(label=2**53 + 1), {"signal": [False, True]},
                              cv=cv, n_jobs=1, error_score="raise")
        fit_shared_fold_search(search, np.array([0, 1] * 4)[:, None], target)
        np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.5, 1.])
        assert not called and search.best_estimator_.y_ is target


@pytest.mark.filterwarnings("ignore:Allowing arbitrary scalar fill_value")
@pytest.mark.filterwarnings("ignore:Concatenating sparse arrays with multiple fill values")
@pytest.mark.parametrize("dtype,point,fill", [
    (np.float32, 2**54, float(2**54 + 4)),
    (np.float32, -(2**54), float(-(2**54 + 4))),
    (np.float32, 2**54, float(2**54 + 2**31)),
    (np.float16, 2048, 2048.5),
    (np.float16, 2048, 2050.0),
    (np.complex64, complex(2**54, 3), complex(2**54 + 4, 3)),
    (np.complex64, complex(2**54, 3), complex(2**54 + 2**31, 3)),
])
def test_stored_implicit_and_mixed_target_folds(dtype, point, fill):
    bits = np.array([0, 1] * 4)
    stored_rows, gap_rows = np.flatnonzero(bits), np.flatnonzero(1 - bits)
    values = pd.arrays.SparseArray(np.array([0, point] * 4, dtype=dtype), fill_value=dtype(0))
    try:
        values.fill_value = fill
    except (TypeError, ValueError):
        pytest.skip("this pandas version rejects the mixed fill at construction")
    target = pd.DataFrame({0: pd.arrays.SparseArray(bits.astype("u8"), fill_value=np.uint64(0)), 1: values})
    def scalar(value):
        return value.item() if isinstance(value, np.generic) else value

    expected = [scalar(values[i]) for i in range(len(values))]
    assert [int(value == point) for value in expected] == bits.tolist()
    # Mixed, point-only and gap-only row selection can dispatch to different
    # pandas casts even on one frame. Compare the scalar and the matrix that
    # an estimator would receive with the exact label at that original row.
    mixed = np.arange(len(bits))[::-1]
    lossy = False
    for rows in (mixed, stored_rows, gap_rows):
        batch = target.take(rows)
        column = np.asarray(batch)[:, 1]
        for i, row in enumerate(rows):
            if scalar(batch.iat[i, 1]) != expected[row] or scalar(column[i]) != expected[row]:
                lossy = True
    folds = [(stored_rows, gap_rows), (gap_rows, stored_rows)]
    def search():
        return GridSearchCV(_MatchOutput(label=point), {"signal": [False, True]},
                            cv=folds, n_jobs=1, error_score="raise")

    owning = pd.DataFrame({0: bits.tolist(), 1: expected}, dtype=object)
    fitted = fit_shared_fold_search(search(), bits[:, None], owning)
    np.testing.assert_array_equal(fitted.cv_results_["mean_test_score"], [.5, 1.])
    assert fitted.best_params_ == {"signal": True}
    if lossy:
        native = search().fit(bits[:, None], target)
        if native.best_params_ == {"signal": False}:
            np.testing.assert_array_equal(native.cv_results_["mean_test_score"], [1., .5])
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(search(), bits[:, None], target)
    else:
        fitted = fit_shared_fold_search(search(), bits[:, None], target)
        np.testing.assert_array_equal(fitted.cv_results_["mean_test_score"], [.5, 1.])
        assert fitted.best_estimator_.y_ is target


def _sparse_object_category(levels, codes, fill=np.nan):
    dictionary = pd.arrays.SparseArray(np.array(levels, dtype=object), dtype=pd.SparseDtype(object, fill))
    return pd.Categorical.from_codes(codes, categories=pd.Index(dictionary))


@pytest.mark.filterwarnings("ignore:Allowing arbitrary scalar fill_value")
@pytest.mark.parametrize("companion", ["signed", "wide_fill"])
@pytest.mark.parametrize("layout", ["category_first", "category_last", "mixed_category", "dense_object",
                                    "single_frame", "series"])
def test_sparse_companion_rankings_with_object_category_and_ablations(companion, layout):
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    X = bits[:, None]
    high = 2**53 + 1
    if companion == "signed":
        label, other = -high, 0
        signal = pd.arrays.SparseArray(np.where(bits, label, other).astype("i8"), fill_value=0.)
    else:
        label, other = 2**54 + 4, 2
        signal = pd.arrays.SparseArray(np.where(bits, 0, other).astype("f4"), fill_value=0.)
        signal.fill_value = float(label)
    levels = ["zero" if layout == "mixed_category" else np.uint64(0), np.uint64(high)]
    marker = _sparse_object_category(levels, bits)
    columns = {"category": marker, "signal": signal}
    if layout == "category_last":
        columns = {"signal": signal, "category": marker}
    elif layout == "dense_object":
        columns["category"] = np.array([levels[b] for b in bits], dtype=object)
    elif layout in ("single_frame", "series"):
        del columns["category"]
    y = pd.DataFrame(columns)
    expected = {"category": [_label_key(levels[b]) for b in bits], "signal": [label if b else other for b in bits]}
    owning = pd.DataFrame({key: expected[key] for key in columns}, dtype=object)
    scored_label = high if layout == "category_last" else label

    def search(target):
        return GridSearchCV(_MatchOutput(label=scored_label), {"signal": [False, True]},
                            cv=_PanelSplits(target), n_jobs=1, error_score="raise")

    # A caller can replace the affected column with owning integers, or
    # construct an owning object frame from each original scalar.
    integer_signal = np.array(expected["signal"], dtype="i8" if companion == "signed" else "u8")
    dense = y.assign(signal=integer_signal)
    if layout == "series":
        y, owning, dense = y["signal"], owning["signal"], dense["signal"]
    for exact in (owning, dense):
        fitted = fit_shared_fold_search(search(exact), X, exact)
        np.testing.assert_array_equal(fitted.cv_results_["mean_test_score"], [.5, 1.])
        assert fitted.best_params_ == {"signal": True} and fitted.best_estimator_.y_ is exact

    # pandas versions differ on both witnesses. Compare every output, even
    # when a correct score on the last column conceals damage to the first.
    rows = np.arange(len(y))
    batches = [(y, rows)]
    for subset in (rows[::-1], np.flatnonzero(bits), np.flatnonzero(1 - bits)):
        batches.append((y.take(subset), subset))
    loses = False
    for batch, order in batches:
        matrix = np.asarray(batch).reshape(len(batch), -1)
        for j, column in enumerate(columns):
            selected = batch if layout == "series" else batch.iloc[:, j]
            for i, row in enumerate(order):
                reference = expected[column][row]
                loses |= _label_key(selected.iat[i]) != reference or _label_key(matrix[i, j]) != reference

    # Both test folds contain both labels. When the signal class disappears
    # on conversion, the constant candidate overtakes the signal candidate.
    converted = np.asarray(y.take(rows[::-1])).reshape(len(y), -1)
    matches = [_label_key(v) == scored_label for v in converted[:, -1]]
    native = search(y).fit(X, y)
    scores = [.5, 1.] if any(matches) else [1., .5]
    np.testing.assert_array_equal(native.cv_results_["mean_test_score"], scores)
    assert native.best_params_ == {"signal": any(matches)}
    checked = search(y)
    if loses:
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(checked, X, y)
        assert checked.cv.calls == 0
    else:
        fit_shared_fold_search(checked, X, y)
        np.testing.assert_array_equal(checked.cv_results_["mean_test_score"], [.5, 1.])
        assert checked.best_estimator_.y_ is y


@pytest.mark.parametrize("levels,fill,codes", [
    ([np.uint64(2**53 + 1), "class"], np.nan, [1, 1]),
    (["left", "right"], np.nan, [0, 1]),
    ([], np.nan, [-1, -1]),
    ([np.uint64(2**53 + 1), "unobserved"], np.uint64(2**53 + 1), [0, -1]),
    (["implicit", np.uint64(2**53 + 1)], "implicit", [0, 1]),
])
def test_exact_sparse_object_categories_preserve_target(levels, fill, codes):
    bits = np.array([0, 1] * 4)
    marker = _sparse_object_category(levels, codes * 4, fill)
    y = pd.DataFrame({"marker": marker, "signal": bits})
    # Include unused and absent dictionary entries, a missing code, and
    # labels stored as points or as the dictionary's implicit fill.
    expected = [None if code < 0 else _label_key(levels[code]) for code in codes * 4]
    assert [_label_key(value) for value in np.asarray(y)[:, 0]] == expected
    search = GridSearchCV(_MatchOutput(label=1), {"signal": [False, True]},
                          cv=_PanelSplits(y), n_jobs=1, error_score="raise")
    fit_shared_fold_search(search, bits[:, None], y)
    np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.5, 1.])
    assert search.best_params_ == {"signal": True} and search.best_estimator_.y_ is y


@pytest.mark.parametrize("temporal", [pd.Timestamp, pd.Timedelta])
def test_temporal_only_fold_cannot_become_an_integer_class(temporal):
    high = 2**53 + 1
    category = _sparse_object_category([temporal(high, unit="ns"), np.uint64(high)],
                                       [0, -1, 0, -1, 1, -1, 1, -1])
    target = pd.Series(category)
    owning = pd.Series([target.iat[i] for i in range(len(target))], dtype=object)
    X = np.array([1, 0, 1, 0, 0, 0, 0, 0])[:, None]

    class DetectInteger(_MatchOutput):
        def score(self, X, y):
            flags = [isinstance(v, (int, np.integer)) and not isinstance(v, np.timedelta64)
                     and int(v) == self.label for v in np.asarray(y)]
            predictions = X[:, 0] if self.signal else np.zeros(len(y), dtype=int)
            return np.mean(flags == predictions)

    def search(y):
        return GridSearchCV(DetectInteger(label=high), {"signal": [False, True]},
                            cv=_PanelSplits(y), n_jobs=1, error_score="raise")

    exact = fit_shared_fold_search(search(owning), X, owning)
    np.testing.assert_array_equal(exact.cv_results_["mean_test_score"], [.75, .5])
    native = search(target).fit(X, target)
    checked = search(target)
    # The first fold contains only temporal and missing levels. On some
    # pandas versions its timestamps or durations emerge as integer nanoseconds.
    if any(isinstance(v, (int, np.integer)) and not isinstance(v, np.timedelta64)
           for v in np.asarray(target.take([3, 2, 0, 1]))):
        np.testing.assert_array_equal(native.cv_results_["mean_test_score"], [.5, .75])
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(checked, X, target)
        assert checked.cv.calls == 0
    else:
        fit_shared_fold_search(checked, X, target)
        np.testing.assert_array_equal(checked.cv_results_["mean_test_score"], [.75, .5])
        assert checked.best_params_ == {"signal": False} and checked.best_estimator_.y_ is target


class _MatchIntervalEndpoint(_MatchOutput):
    def score(self, X, y):
        side, endpoint = self.label
        observed = np.asarray(y).reshape(len(y), -1)[:, -1]
        # NumPy's mixed int/float comparison can round the reference integer.
        flags = [isinstance(v, pd.Interval) and int(getattr(v, side)) == endpoint for v in observed]
        predicted = X[:, 0] if self.signal else np.zeros(len(y), dtype=int)
        return np.mean(flags == predicted)


@pytest.mark.parametrize("high", [3, 2**53 + 1])
@pytest.mark.parametrize("mixed_closure", [False, True])
def test_interval_only_fold_keeps_exact_numeric_endpoints(high, mixed_closure):
    levels = [pd.Interval(high, high + 2), pd.Interval(.25, .5), np.uint64(high)]
    codes = [0, 1, 0, 1, 2, 1, 2, 1]
    if mixed_closure:
        levels.append(pd.Interval(6, 8, closed="left"))
        codes[5] = 3
    target = pd.Series(_sparse_object_category(levels, codes))
    owning = pd.Series([target.iat[i] for i in range(len(target))], dtype=object)
    X = np.array([1, 0, 1, 0, 0, 0, 0, 0])[:, None]

    def search(y):
        return GridSearchCV(_MatchIntervalEndpoint(label=("left", high)), {"signal": [False, True]},
                            cv=_PanelSplits(y), n_jobs=1, error_score="raise")

    exact = fit_shared_fold_search(search(owning), X, owning)
    np.testing.assert_array_equal(exact.cv_results_["mean_test_score"], [.75, 1.])
    native = search(target).fit(X, target)
    checked = search(target)
    # Inspect the endpoints themselves. Mixed int/float interval comparison
    # can also round the reference integer used in ordinary equality.
    first = np.asarray(target.take([0, 1, 2, 3]))[0]
    if int(first.left) != high or int(first.right) != high + 2:
        np.testing.assert_array_equal(native.cv_results_["mean_test_score"], [1., .75])
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(checked, X, target)
        assert checked.cv.calls == 0
    else:
        fit_shared_fold_search(checked, X, target)
        np.testing.assert_array_equal(checked.cv_results_["mean_test_score"], [.75, 1.])
        assert checked.best_params_ == {"signal": True} and checked.best_estimator_.y_ is target


# Halfway between the largest finite binary64 value and 2**1024.
_FLOAT_OVERFLOW_MIDPOINT = 2**1024 - 2**970


@pytest.mark.parametrize("first,second,side", [
    pytest.param(pd.Interval(0, 2**64 + 1), pd.Interval(1, 2**1024), "right", id="oversized-right"),
    pytest.param(pd.Interval(-(2**64 + 1), 0), pd.Interval(-(2**1024), 1), "left", id="oversized-left"),
    pytest.param(pd.Interval(0, 2**64 + 1), pd.Interval(1, _FLOAT_OVERFLOW_MIDPOINT - 1),
                 "right", id="below-overflow-midpoint"),
    pytest.param(pd.Interval(0, 2**64 + 1), pd.Interval(1, _FLOAT_OVERFLOW_MIDPOINT),
                 "right", id="at-overflow-midpoint"),
    pytest.param(pd.Interval(np.uint64(2**63 + 1), np.uint64(2**63 + 7)),
                 pd.Interval(np.uint64(2**64 - 5), np.uint64(2**64 - 1)), "right", id="ordinary-uint64"),
])
def test_sparse_interval_integer_overflow_before_split(first, second, side):
    labels = ["marker", first, second]
    target = pd.Series(pd.arrays.SparseArray(np.array(labels, dtype=object),
                                             dtype=pd.SparseDtype(object, np.nan)))
    owning = pd.Series([target.iat[i] for i in range(len(target))], dtype=object)
    X = np.array([0, 0, 1])[:, None]
    folds = [([0, 1], [2]), ([2, 0], [1])]

    def key(interval):
        return int(interval.left), int(interval.right), interval.closed

    def exact_take(rows):
        selected = target.take(rows)
        expected = [key(labels[i]) for i in rows]
        return [key(v) for v in selected] == expected and [key(v) for v in np.asarray(selected)] == expected

    assert [key(owning.iat[i]) for i in (1, 2)] == [key(first), key(second)]
    # The text marker keeps the source boxed. Pandas may infer an interval
    # dtype for an interval-only take, even for a single selected row.
    try:
        exact = all(exact_take(rows) for rows in ([2], [1], [1, 2], [2, 1]))
    except (OverflowError, TypeError, ValueError):
        exact = False

    for y in (owning, target):
        search = GridSearchCV(_MatchIntervalEndpoint(label=(side, int(getattr(second, side)))),
                              {"signal": [False, True]}, cv=_OrderedFolds(y, folds), n_jobs=1, error_score="raise")
        if y is target and not exact:
            def unexpected_count(*args, **kwargs):
                pytest.fail("get_n_splits ran before the invalid sparse target was rejected")

            search.cv.get_n_splits = unexpected_count
            with pytest.raises(ValueError, match="loses class IDs"):
                fit_shared_fold_search(search, X, y)
            assert search.cv.calls == 0
        else:
            fit_shared_fold_search(search, X, y)
            np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.5, 1.])
            assert search.best_params_ == {"signal": True} and search.best_estimator_.y_ is y


@pytest.mark.parametrize("side,endpoint,other,fill", [
    pytest.param("left", 2**53 + 1, pd.Interval(-10., 10.), np.nan, id="stored-text"),
    pytest.param("right", 2**64 - 1, pd.Interval(-10., 10.), "x", id="text-fill-upper-bound"),
    pytest.param("left", 2**53 + 1, np.nan, np.nan, id="missing-gap"),
    pytest.param("left", 2**53 + 1, np.nan, "x", id="stored-missing"),
    pytest.param("left", 2**53, pd.Interval(-10., 10.), np.nan, id="exact-neighbor"),
    pytest.param("left", 2**53 + 1, pd.Interval(-10., 10., closed="left"), np.nan, id="different-closures"),
    pytest.param("left", 2**53 + 1, pd.Interval(np.uint64(1), np.uint64(10)), np.nan, id="all-unsigned"),
])
def test_sparse_interval_companion_preserves_fold_ranking(side, endpoint, other, fill):
    left = np.uint64(endpoint if side == "left" else 2**53)
    right = np.uint64(endpoint if side == "right" else 2**53 + 4)
    marked = pd.Interval(left, right)
    labels = [marked, other, marked, other, other, other, other, "x"]
    intervals = pd.arrays.SparseArray(np.array(labels, dtype=object), dtype=pd.SparseDtype(object, fill))
    _check_interval_companion_ranking(intervals, side, endpoint)


def test_sparse_interval_companion_can_promote_to_float32():
    high = 2**24 + 1
    marked = pd.Interval(np.uint32(high), np.uint32(high + 3))
    other = pd.Interval(np.float32(-2), np.float32(3))
    labels = [marked, other, marked, other, other, other, other, "x"]
    values = pd.arrays.SparseArray(np.array(labels, dtype=object), dtype=pd.SparseDtype(object, np.nan))
    _check_interval_companion_ranking(values, "left", high)


def test_sparse_interval_companion_accepts_unhashable_timezone():
    from dateutil.tz import tzoffset

    start = pd.Timestamp("2020-01-01", tz=tzoffset("local", 1800))
    interval = pd.Interval(start, start + pd.Timedelta(days=1))
    labels = [interval, "x", interval, "x", interval, "x", interval, "x"]
    target = pd.Series(pd.arrays.SparseArray(np.array(labels, dtype=object), dtype=pd.SparseDtype(object, np.nan)))
    X = np.array([False, True] * 4)[:, None]
    search = GridSearchCV(_MatchOutput(label="x"), {"signal": [False, True]},
                          cv=_PanelSplits(target), n_jobs=1, error_score="raise")
    fit_shared_fold_search(search, X, target)
    np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.5, 1.])
    assert search.best_params_ == {"signal": True} and search.best_estimator_.y_ is target


@pytest.mark.parametrize("endpoint_type", [float, np.uint64])
@pytest.mark.parametrize("bin_type", ["floating", "integer-left", "integer-right"])
def test_fractional_interval_bins_use_bounded_fold_checks(monkeypatch, endpoint_type, bin_type):
    # The marked label is exactly representable even after promotion to float.
    high = 2**53 if endpoint_type is np.uint64 else 2**50
    marked = pd.Interval(endpoint_type(high), endpoint_type(high + 4))
    bits = np.arange(256) % 2
    def bin_label(i):
        if bin_type == "integer-left":
            return pd.Interval(np.uint64(i), i + .75)
        if bin_type == "integer-right":
            return pd.Interval(i + .25, np.uint64(i + 1))
        return pd.Interval(-i - .75, -i - .25)

    labels = [marked if bit else bin_label(i) for i, bit in enumerate(bits)]
    labels[0] = "other"
    target = pd.Series(pd.arrays.SparseArray(np.array(labels, dtype=object),
                                             dtype=pd.SparseDtype(object, np.nan)))
    splitter = _PanelSplits(target)
    take = pd.arrays.SparseArray.take
    checked = []

    def count_take(array, indices, *args, **kwargs):
        if not splitter.calls:
            checked.append(len(indices))
        return take(array, indices, *args, **kwargs)

    monkeypatch.setattr(pd.arrays.SparseArray, "take", count_take)
    search = GridSearchCV(_MatchOutput(label=marked), {"signal": [False, True]}, cv=splitter,
                          n_jobs=1, error_score="raise", return_train_score=True)
    fit_shared_fold_search(search, bits[:, None], target)
    np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.5, 1.])
    np.testing.assert_array_equal(search.cv_results_["mean_train_score"], [.5, 1.])
    assert search.best_params_ == {"signal": True} and search.best_estimator_.y_ is target
    assert max(checked, default=0) < 32


@pytest.mark.parametrize("fill_type", [np.uint64, float], ids=["integer-fill", "floating-fill"])
def test_recompression_compares_interval_fill_endpoints_exactly(fill_type):
    high = 2**53
    marked = pd.Interval(np.uint64(high + 1), np.uint64(high + 4))
    other = pd.Interval(np.uint64(1), np.uint64(10))
    labels = [marked, other, marked, other, other, other, np.nan, "x"]
    intervals = pd.arrays.SparseArray(np.array(labels, dtype=object), dtype=pd.SparseDtype(object, np.nan))
    # Install the fill after construction so the marked interval starts exact.
    # Gathering that interval alone can recompress it into the adjacent float.
    intervals.fill_value = pd.Interval(fill_type(high), fill_type(high + 4))
    _check_interval_companion_ranking(intervals, "left", high + 1)


def _check_interval_companion_ranking(intervals, side, endpoint):
    bits = np.array([1, 0, 1, 0, 0, 0, 0, 0])
    X = bits[:, None]
    category = _sparse_object_category([np.uint64(5), np.uint64(2**53 + 7)], bits)
    target = pd.DataFrame({"ids": category, "interval": intervals})
    owning = pd.DataFrame([[target.iat[i, j] for j in range(2)] for i in range(len(target))], dtype=object)
    assert int(getattr(np.asarray(target)[0, 1], side)) == endpoint

    def grid(y):
        return GridSearchCV(_MatchIntervalEndpoint(label=(side, endpoint)), {"signal": [False, True]},
                            cv=_PanelSplits(y), n_jobs=1, return_train_score=True, error_score="raise")

    def check_scores(search, y, retained):
        # Only the fold containing the marked rows can distinguish the candidates.
        fold_scores = [[1., 1.], [.5, 1.] if retained else [1., .5]]
        for i in range(2):
            np.testing.assert_array_equal(search.cv_results_[f"split{i}_test_score"], fold_scores[i])
            np.testing.assert_array_equal(search.cv_results_[f"split{i}_train_score"], fold_scores[1 - i])
        np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.75, 1.] if retained else [1., .75])
        assert search.best_params_ == {"signal": retained}
        assert search.cv.calls == 1 and search.best_estimator_.y_ is y

    check_scores(fit_shared_fold_search(grid(owning), X, owning), owning, True)
    selected = target.take([3, 2, 0, 1])
    retained = int(getattr(np.asarray(selected)[2, 1], side)) == endpoint
    check_scores(grid(target).fit(X, target), target, retained)
    checked = grid(target)
    if retained:
        check_scores(fit_shared_fold_search(checked, X, target), target, True)
    else:
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(checked, X, target)
        assert checked.cv.calls == 0


def _matches_temporal_value(value, expected):
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        value = value.asm8
    if isinstance(expected, str):
        return isinstance(value, np.datetime64) and np.datetime_as_string(value, unit="ps") == expected
    if not isinstance(value, np.timedelta64):
        return False
    unit, count = expected
    # Calendar durations have no fixed length in seconds. Each reference
    # unit below can express the corresponding original and selected values.
    if (np.datetime_data(value.dtype)[0] in ("Y", "M")) != (unit == "M"):
        return False
    return int(value.astype(f"timedelta64[{unit}]").astype(np.int64)) == count


class _MatchTemporal(_MatchOutput):
    def score(self, X, y):
        observed = [_matches_temporal_value(value, self.label) for value in np.asarray(y)]
        predicted = X[:, 0] if self.signal else np.zeros(len(y))
        return np.mean(observed == predicted)


@pytest.mark.parametrize("value,expected", [
    (np.datetime64(2342342345, "ps"), "1970-01-01T00:00:00.002342342345"),
    (np.datetime64(2342342, "ns"), "1970-01-01T00:00:00.002342342000"),
    (np.datetime64(24000, "ps"), "1970-01-01T00:00:00.000000024000"),
    (np.datetime64(-7, "3ns"), "1969-12-31T23:59:59.999999979000"),
    (np.datetime64(-3, "2D"), "1969-12-26T00:00:00.000000000000"),
    (np.datetime64(-3, "3M"), "1969-04-01T00:00:00.000000000000"),
    (np.datetime64(-3, "M"), "1969-10-01T00:00:00.000000000000"),
    (np.timedelta64(2342342345, "ps"), ("ps", 2342342345)),
    (np.timedelta64(-7, "3ns"), ("ps", -21000)),
    (np.timedelta64(-3, "2D"), ("ns", -6 * 86400 * 10**9)),
    (np.timedelta64(-3, "3M"), ("M", -9)),
    (np.timedelta64(3, "2Y"), ("M", 72)),
])
def test_temporal_only_folds_keep_the_original_instant_or_duration(value, expected):
    high = np.uint64(2**53 + 1)
    try:
        target = pd.Series(_sparse_object_category([value, high], [0] * 4 + [1] * 4))
    except ValueError as error:
        # Starting with the unsigned level preserves the mixed dictionary
        # when pandas tries to infer a temporal dtype from its first entry.
        if "from_unit must be <= to_unit" not in str(error) and "containing a multiplier" not in str(error):
            raise
        target = pd.Series(_sparse_object_category([high, value], [1] * 4 + [0] * 4))
    X = np.array([1] * 4 + [0] * 4)[:, None]
    owning = pd.Series([target.iat[i] for i in range(len(target))], dtype=object)

    def search(y):
        return GridSearchCV(_MatchTemporal(label=expected), {"signal": [False, True]},
                            cv=_PanelSplits(y), n_jobs=1, error_score="raise")

    assert _matches_temporal_value(target.iat[0], expected)
    exact = fit_shared_fold_search(search(owning), X, owning)
    np.testing.assert_array_equal(exact.cv_results_["mean_test_score"], [.5, 1.])
    assert exact.best_params_ == {"signal": True} and exact.best_estimator_.y_ is owning

    # Scalar lookup can still return the original level after row selection
    # while NumPy conversion of this all-temporal fold changes its label.
    temporal_fold = target.take([3, 2, 0, 1])
    assert all(_matches_temporal_value(temporal_fold.iat[i], expected) for i in range(4))
    try:
        converted = np.asarray(temporal_fold)
    except (ValueError, TypeError, OverflowError, NotImplementedError) as error:
        with pytest.raises(type(error)):
            search(target).fit(X, target)
        retained = False
    else:
        retained = all(_matches_temporal_value(v, expected) for v in converted)
        native = search(target).fit(X, target)
        np.testing.assert_array_equal(native.cv_results_["mean_test_score"], [.5, 1.] if retained else [1., .5])
        assert native.best_params_ == {"signal": retained}
    checked = search(target)
    if retained:
        fit_shared_fold_search(checked, X, target)
        np.testing.assert_array_equal(checked.cv_results_["mean_test_score"], [.5, 1.])
        assert checked.best_params_ == {"signal": True} and checked.best_estimator_.y_ is target
    else:
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(checked, X, target)
        assert checked.cv.calls == 0


def test_distinct_steps_of_one_duration_unit_need_linear_row_selections(monkeypatch):
    take = pd.Categorical.take
    selections = 0

    def count_take(self, *args, **kwargs):
        nonlocal selections
        selections += 1
        return take(self, *args, **kwargs)

    monkeypatch.setattr(pd.Categorical, "take", count_take)
    counts = []
    for size in (8, 32):
        levels = [np.uint64(2**53 + 1)] + [np.timedelta64(k + 1, f"{k}ns") for k in range(1, size + 1)]
        y = pd.Series(_sparse_object_category(levels, np.arange(len(levels))))
        X = (np.arange(len(y)) == size)[:, None]
        first, last = np.array_split(np.arange(len(y)), 2)
        label = ("ns", size * (size + 1))

        def search():
            return GridSearchCV(_MatchTemporal(label=label), {"signal": [False, True]},
                                cv=[(first, last), (last, first)], n_jobs=1, error_score="raise")

        # The last fold contains the only match. Some pandas releases drop
        # its unit multiplier, so even the constant candidate beats signal.
        retained = _matches_temporal_value(np.asarray(y.take(last))[-1], label)
        one_error = 1. - .5 / len(last)
        native = search().fit(X, y)
        np.testing.assert_array_equal(native.cv_results_["mean_test_score"],
                                      [one_error, 1.] if retained else [1., one_error])
        assert native.best_params_ == {"signal": retained}
        checked = search()
        selections = 0
        if retained:
            fit_shared_fold_search(checked, X, y)
            assert checked.best_params_ == {"signal": True} and checked.best_estimator_.y_ is y
        else:
            with pytest.raises(ValueError, match="loses class IDs"):
                fit_shared_fold_search(checked, X, y)
        counts.append(selections)
    # Four times as many distinct multipliers should not trigger a take for
    # every pair of multipliers. Count the actual takes, including CV folds.
    assert counts[1] <= 4 * counts[0], counts


class _MatchNativeTemporal(_MatchTemporal):
    def score(self, X, y):
        columns = np.asarray(y).reshape(len(y), -1)
        if columns.shape[1] == 2:
            assert [int(v) for v in columns[:, 1]] == [2**53 + 1 if bit else 0 for bit in X[:, 0]]
        return super().score(X, columns[:, 0])


@pytest.mark.filterwarnings("ignore:Allowing arbitrary scalar fill_value")
@pytest.mark.filterwarnings("ignore:Concatenating sparse arrays with multiple fill values")
@pytest.mark.parametrize("kind,case,layout", [
    ("M", "stepped", "categorical_frame"), ("M", "day", "categorical_frame"),
    ("M", "picosecond", "categorical_frame"), ("M", "nanosecond", "categorical_frame"),
    ("M", "unused_fill", "categorical_frame"),
    ("M", "picosecond", "categorical_series"), ("M", "nanosecond", "categorical_series"),
    ("M", "picosecond", "sparse_frame"), ("M", "nanosecond", "sparse_frame"),
    ("M", "picosecond", "sparse_series"), ("M", "nanosecond", "sparse_series"),
    ("m", "stepped", "sparse_frame"), ("m", "stepped", "categorical_frame"),
    ("m", "picosecond", "sparse_series"), ("m", "nanosecond", "sparse_series"),
    ("M", "object_picosecond", "sparse_frame"), ("M", "object_timestamp", "sparse_frame"),
])
def test_native_sparse_temporal_fill_preserves_search_ranking(kind, case, layout):
    bits = np.array([1, 0] * 4)
    X = bits[:, None]
    categorical = layout.startswith("categorical")
    repeats = 1 if categorical else 4
    scalar = np.datetime64 if kind == "M" else np.timedelta64
    codes = 1 - bits
    if case in ("stepped", "day"):
        unit = "2D" if case == "stepped" else "D"
        ticks = np.array([2 if case == "stepped" else 4, 0] * repeats, dtype=f"{kind}8[{unit}]")
        # A four-day fill also constructs the January 5 case on releases
        # that accept two-day storage but reject a fill expressed in 2D.
        try:
            values = pd.arrays.SparseArray(ticks, fill_value=scalar(4, "D"))
        except ValueError as error:
            if case != "stepped" or "containing a multiplier" not in str(error):
                raise
            pytest.skip(str(error))
        expected = "1970-01-05T00:00:00.000000000000" if kind == "M" else ("D", 4)
        if case == "stepped" and kind == "M":
            try:
                values[1]
            except ValueError as error:
                if "containing a multiplier" not in str(error):
                    raise
                # The stored scalar cannot be boxed here. The dictionary gap
                # and a missing code still reproduce the January 5/3 ranking.
                codes = bits - 1
    else:
        if case.startswith("object_"):
            pair = [0, np.datetime64(1, "ns")]
            if case == "object_timestamp":
                pair.reverse()  # A stored instant can become integer 1 in the fold.
            values = pd.arrays.SparseArray(np.array(pair * repeats, dtype=object), dtype=pd.SparseDtype(object, 0))
        else:
            ticks = np.array(([0, 1, 2] if case == "unused_fill" else [0, 1]) * repeats, dtype=f"{kind}8[ns]")
            values = pd.arrays.SparseArray(ticks, fill_value=ticks[0])
        count = 2_001_000 if case == "nanosecond" else 2_000_001
        fill = scalar(2001, "ns") if case == "nanosecond" else scalar(count, "ps")
        try:
            values.fill_value = pd.Timestamp(2001) if case == "object_timestamp" else fill
        except ValueError as error:
            if (kind, case) != ("m", "picosecond") or "fill_value must be a valid value" not in str(error):
                raise
            pytest.skip(str(error))
        if case == "unused_fill":
            codes = 2 - bits
        if case in ("object_timestamp", "unused_fill"):
            count = 1000  # Score the stored 1 ns.
        expected = f"1970-01-01T00:00:00.{count:012d}" if kind == "M" else ("ps", count)

    temporal = pd.Categorical.from_codes(codes, categories=pd.Index(values)) if categorical else values
    owning_times = [temporal[i] for i in range(len(bits))]
    assert [_matches_temporal_value(stamp, expected) for stamp in owning_times] == bits.tolist()
    if layout.endswith("frame"):
        high = np.uint64(2**53 + 1)
        ids = np.where(bits, high, np.uint64(0))
        companion = (_sparse_object_category([np.uint64(0), high], bits) if categorical
                     else pd.arrays.SparseArray(ids, fill_value=np.uint64(0)))
        target = pd.DataFrame({"time": temporal, "id": companion})
        owning = pd.DataFrame(list(zip(owning_times, ids)), columns=target.columns, dtype=object)
    else:
        target = pd.Series(temporal)
        owning = pd.Series(owning_times, dtype=object)

    def search(y):
        return GridSearchCV(_MatchNativeTemporal(label=expected), {"signal": [False, True]},
                            cv=_PanelSplits(y), n_jobs=1, error_score="raise")

    exact = fit_shared_fold_search(search(owning), X, owning)
    np.testing.assert_array_equal(exact.cv_results_["mean_test_score"], [.5, 1.])
    assert exact.best_params_ == {"signal": True} and exact.best_estimator_.y_ is owning
    converted = np.asarray(target.take([1, 0])).reshape(2, -1)[1, 0]
    retained = _matches_temporal_value(converted, expected)
    parent_retained = _matches_temporal_value(np.asarray(target).reshape(len(bits), -1)[0, 0], expected)
    if case in ("day", "nanosecond", "unused_fill"):
        assert retained and parent_retained
    if case == "stepped" and not retained:
        changed = "1970-01-03T00:00:00.000000000000" if kind == "M" else ("D", 2)
        assert _matches_temporal_value(converted, changed)
    native = search(target).fit(X, target)
    np.testing.assert_array_equal(native.cv_results_["mean_test_score"], [.5, 1.] if retained else [1., .5])
    assert native.best_params_ == {"signal": retained} and native.best_estimator_.y_ is target
    checked = search(target)
    if retained and parent_retained:
        fit_shared_fold_search(checked, X, target)
        np.testing.assert_array_equal(checked.cv_results_["mean_test_score"], [.5, 1.])
        assert checked.best_params_ == {"signal": True} and checked.best_estimator_.y_ is target
    else:
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(checked, X, target)
        assert checked.cv.calls == 0


class _OrderedFolds(_PanelSplits):
    def __init__(self, target, folds):
        super().__init__(target)
        self.folds = folds

    def split(self, X, y, groups=None):
        assert y is self.original
        self.calls += 1
        yield from self.folds


class _MatchLastInstant(_MatchTemporal):
    def score(self, X, y):
        return super().score(X, np.asarray(y).reshape(len(y), -1)[:, -1])


def _instant_grid(target, label, folds):
    return GridSearchCV(_MatchLastInstant(label=label), {"signal": [False, True]},
                        cv=_OrderedFolds(target, folds), n_jobs=1, error_score="raise")


def _check_instant_ranking(search, target, retained):
    np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [.5, 1.] if retained else [1., .5])
    assert search.best_params_ == {"signal": retained} and search.best_estimator_.y_ is target


@pytest.mark.parametrize("categorical", [False, True])
def test_sparse_object_fold_order_can_change_a_stored_date(categorical):
    raw = ["x", np.datetime64(1, "2D"), np.datetime64(4, "D")]
    if categorical:
        target = pd.Series(_sparse_object_category(raw, [0, 1, 2] * 2))
    else:
        target = pd.Series(pd.arrays.SparseArray(np.array(raw * 2, dtype=object), dtype=pd.SparseDtype(object, np.nan)))
    owning = pd.Series([target.iat[i] for i in range(len(target))], dtype=object)
    X = np.array([0, 1, 0] * 2)[:, None]
    label = "1970-01-03T00:00:00.000000000000"
    assert _matches_temporal_value(target.iat[1], label)
    # With the ordinary date first, pandas can interpret the single 2D tick
    # as one day. The exact object parent still has the January 3 scalar.
    rows = np.array([1, 2])
    orders = (rows, rows[::-1])
    retained = []
    for order in orders:
        stamp = np.asarray(target.take(order))[np.flatnonzero(order == 1)[0]]
        retained.append(_matches_temporal_value(stamp, label))
        if not retained[-1]:
            assert _matches_temporal_value(stamp, "1970-01-02T00:00:00.000000000000")
    for order, exact in zip(orders, retained):
        folds = [([0, 1, 2, 3], order + 3), ([3, 4, 5, 0], order)]
        control = fit_shared_fold_search(_instant_grid(owning, label, folds), X, owning)
        _check_instant_ranking(control, owning, True)
        native = _instant_grid(target, label, folds).fit(X, target)
        _check_instant_ranking(native, target, exact)
        checked = _instant_grid(target, label, folds)
        if all(retained):
            _check_instant_ranking(fit_shared_fold_search(checked, X, target), target, True)
        else:
            with pytest.raises(ValueError, match="loses class IDs"):
                fit_shared_fold_search(checked, X, target)
            assert checked.cv.calls == 0


@pytest.mark.parametrize("year,layout", [
    (32000, "overlap"), (2000, "overlap"), (32000, "disjoint"), (32000, "python_date"), (2100, "python_date")
])
def test_sparse_frame_temporal_promotion_depends_on_common_rows(year, layout):
    def sparse(points):
        return pd.arrays.SparseArray(np.array(points, dtype=object), dtype=pd.SparseDtype(object, np.nan))

    a = np.datetime64(1001, "ns")
    if layout == "python_date":
        b, day = np.datetime64(1, "s"), pd.Timestamp("2020-03-01").date()
        fine, coarse = [b, a, day, a, "x"], [b, b, day, "x", b]
        folded = [0, 1]
        witnesses = ([0, 1, 2], [0, 1, 3], [0, 1, 4])
        folds = [([2, 3, 4], folded), ([2, 3, 4], folded[::-1])]
    else:
        b = np.datetime64("2020-01-01", "s")
        fine = [a, a, "x", "y"]
        coarse = [b, "s", b, "v"] if layout == "overlap" else ["s", "v", b, b]
        folded = [0]
        witnesses = ([0, 1], [0, 2])
        folds = [([1, 2, 3], folded), ([0, 1, 3], [2])]
    dates = np.array([f"{year}-01-01"] + ["1999-01-01"] * (len(fine) - 1), dtype="M8[s]")
    target = pd.DataFrame({"fine": sparse(fine), "coarse": sparse(coarse), "date": pd.Categorical(dates)})
    owning = pd.DataFrame([[target.iat[i, j] for j in range(3)] for i in range(len(target))],
                          columns=target.columns, dtype=object)
    X = (np.arange(len(target)) == 0)[:, None]
    label = f"{year}-01-01T00:00:00.000000000000"
    for rows in witnesses:
        assert _matches_temporal_value(np.asarray(target.take(rows))[0, 2], label)
    _check_instant_ranking(fit_shared_fold_search(_instant_grid(owning, label, folds), X, owning), owning, True)
    native = _instant_grid(target, label, folds)
    try:
        stamp = np.asarray(target.take(folded))[0, 2]
    except OverflowError:
        with pytest.raises(OverflowError):
            native.fit(X, target)
        exact = False
    else:
        exact = _matches_temporal_value(stamp, label)
        if not exact:
            assert _matches_temporal_value(stamp, "2187-09-28T21:37:20.812867584000")
        _check_instant_ranking(native.fit(X, target), target, exact)
    if year != 32000 or layout == "disjoint":
        assert exact
    checked = _instant_grid(target, label, folds)
    if exact:
        _check_instant_ranking(fit_shared_fold_search(checked, X, target), target, True)
    else:
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(checked, X, target)
        assert checked.cv.calls == 0


@pytest.mark.filterwarnings("ignore:Allowing arbitrary scalar fill_value")
@pytest.mark.parametrize("dtype,fill,index", [
    ("V2", "temporal", False), ("V2", "temporal", True), ("V2", "unused", False),
    ("u8", "temporal", False), ("u8", "temporal", True), ("u8", "unused", False),
    ("record", "temporal", False), ("object", "native", False), ("u8", "native", False),
])
def test_temporal_fill_on_sparse_non_temporal_vector(dtype, fill, index):
    if dtype == "record":
        data = np.array([(1,), (2,)], dtype=[("a", "i8")]).view(np.recarray)
    else:
        data = np.array(([0, 2**53 + 1] if dtype == "u8" else [b"a", b"b"]) * 4, dtype=dtype)
    original_fill = np.array(7 if dtype == "u8" else b"z", dtype=dtype)[()] if fill == "unused" else data[0]
    try:
        values = pd.arrays.SparseArray(data, fill_value=original_fill)
        if fill != "native":
            values.fill_value = np.datetime64(2_000_001, "ps")
    except ValueError as error:
        if fill == "native" or "fill_value must be a valid value for the SparseDtype.subtype" not in str(error):
            raise
        pytest.skip(f"pandas rejects the {dtype} sparse dtype or its temporal fill: {error}")
    if dtype == "record":
        target = pd.Categorical.from_codes([0, 1] * 4, categories=pd.Index(values))
    else:
        target = pd.Index(values) if index else values
    owning = pd.Series([target[i] for i in range(len(target))], dtype=object)
    X = np.array([1, 0] * 4)[:, None]
    if fill == "temporal":
        label = "1970-01-01T00:00:00.000002000001"
        assert _matches_temporal_value(target[0], label)
        assert values.sp_index.ngaps == len(data) // 2
        estimator = _MatchTemporal(label=label)
    else:
        estimator = _MatchOutput(label=_label_key(data[0]))
        if fill == "unused":
            assert values.sp_index.ngaps == 0

    def grid(y):
        return GridSearchCV(estimator, {"signal": [False, True]}, cv=_PanelSplits(y), error_score="raise")

    _check_instant_ranking(fit_shared_fold_search(grid(owning), X, owning), owning, True)
    try:
        observed = np.asarray(target.take([1, 0]))[1]
    except TypeError as error:
        # Some pandas versions cannot promote the record and datetime
        # dtypes when selecting levels from the sparse dictionary.
        assert dtype == "record"
        with pytest.raises(type(error)):
            grid(target).fit(X, target)
        retained = False
    else:
        retained = fill != "temporal" or _matches_temporal_value(observed, label)
        if not retained:
            assert isinstance(observed, (int, np.integer)) and int(observed) == 2_000_001
        _check_instant_ranking(grid(target).fit(X, target), target, retained)
    checked = grid(target)
    if retained:
        _check_instant_ranking(fit_shared_fold_search(checked, X, target), target, True)
    else:
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(checked, X, target)
        assert checked.cv.calls == 0


@pytest.mark.parametrize("year", [2500, 2100])
def test_full_order_take_can_bypass_sparse_fold_datetime_conversion(year):
    # Every sparse value has the same resolution. A full take in original
    # order can leave it boxed even when a real fold infers nanoseconds.
    fine = pd.arrays.SparseArray(np.array([np.datetime64(1001, "ns")] * 4, dtype=object),
                                 dtype=pd.SparseDtype(object, np.nan))
    dates = np.array([f"{year}-01-01", "1999-01-01"] * 2, dtype="M8[s]")
    target = pd.DataFrame({"fine": fine, "date": pd.Categorical(dates)})
    owning = pd.DataFrame([[target.iat[i, 0], target.iat[i, 1]] for i in range(4)], dtype=object)
    X = np.array([1, 0] * 2)[:, None]
    label = f"{year}-01-01T00:00:00.000000000000"
    full = target.take([0, 1, 2, 3])
    if full.dtypes.iloc[0] == fine.dtype:
        assert _matches_temporal_value(np.asarray(full)[0, 1], label)
    folds = [([2, 3], [1, 0]), ([0, 1], [3, 2])]
    _check_instant_ranking(fit_shared_fold_search(_instant_grid(owning, label, folds), X, owning), owning, True)
    native = _instant_grid(target, label, folds)
    try:
        stamp = np.asarray(target.take([3, 2, 1, 0]))[-1, 1]
    except OverflowError:
        with pytest.raises(OverflowError):
            native.fit(X, target)
        exact = False
    else:
        exact = _matches_temporal_value(stamp, label)
        if not exact:
            assert _matches_temporal_value(stamp, "1915-06-14T00:25:26.290448384000")
        _check_instant_ranking(native.fit(X, target), target, exact)
    if year == 2100:
        assert exact
    checked = _instant_grid(target, label, folds)
    if exact:
        _check_instant_ranking(fit_shared_fold_search(checked, X, target), target, True)
    else:
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(checked, X, target)
        assert checked.cv.calls == 0


@pytest.mark.parametrize("representation", ["aligned", "nanoseconds", "timestamp", "milliseconds", "far_companion"])
def test_sparse_temporal_frame_limits_gathers_for_distinct_row_patterns(monkeypatch, representation):
    width = 8
    indices = np.arange(2**width)
    coarse = np.datetime64(1, "s")
    fine = np.datetime64(2_000_000_000 if representation == "aligned" else 1001, "ns")
    if representation == "milliseconds":
        coarse, fine = np.datetime64("2500-01-01", "s"), np.datetime64("2501-01-01", "ms")
    fine_label = str(np.datetime_as_string(fine, unit="ps"))
    coarse_label = str(np.datetime_as_string(coarse, unit="ps"))
    if representation == "timestamp":
        fine = pd.Timestamp(fine)
    columns = {}
    for j in range(width):
        points = [fine if row & (1 << j) else coarse for row in indices] + ["x"]
        columns[j] = pd.arrays.SparseArray(np.array(points, dtype=object), dtype=pd.SparseDtype(object, np.nan))
    target = pd.DataFrame(columns)
    label = fine_label
    bits = (indices >> (width - 1)) & 1
    if representation == "far_companion":
        # The same sparse columns can force an out-of-range seconds
        # companion into nanoseconds when the fold omits the string row.
        target["date"] = pd.Categorical(np.array(["1999", "2500"] * 128 + ["1999"], dtype="M8[s]"))
        label = "2500-01-01T00:00:00.000000000000"
        bits = indices & 1
    X = np.append(bits, 0)[:, None]
    even, odd = indices[::2], indices[1::2][::-1]
    folds = [(even, odd), (odd, even)]
    native = _instant_grid(target, label, folds)
    try:
        stamp = np.asarray(target.take(odd))[0, -1]
    except OverflowError:
        with pytest.raises(OverflowError):
            native.fit(X, target)
        exact = False
    else:
        exact = _matches_temporal_value(stamp, label)
        _check_instant_ranking(native.fit(X, target), target, exact)
    if representation != "far_companion":
        assert exact
        rows = [255, 170, 43, 0]
        observed = np.asarray(target.take(rows))
        for i, row in enumerate(rows):
            for j in range(width):
                expected = fine_label if row & (1 << j) else coarse_label
                assert _matches_temporal_value(observed[i, j], expected)

    native_take = pd.DataFrame.take
    gathers = 0

    def counted_take(self, *args, **kwargs):
        nonlocal gathers
        gathers += 1
        if gathers > 8 * width + 16:
            pytest.fail(f"{gathers} DataFrame.take calls for {width} columns and {len(target)} rows")
        return native_take(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "take", counted_take)
    checked = _instant_grid(target, label, folds)
    if exact:
        _check_instant_ranking(fit_shared_fold_search(checked, X, target), target, True)
    else:
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(checked, X, target)
        assert checked.cv.calls == 0


class _MatchDurationColumn(_MatchOutput):
    def score(self, X, y):
        from datetime import timedelta

        observed = np.asarray(y)[:, 0]
        duration = [isinstance(v, (timedelta, np.timedelta64)) and not pd.isna(v) for v in observed]
        return np.mean(duration == (X[:, 0] if self.signal else np.zeros(len(X))))


@pytest.mark.filterwarnings("ignore:Concatenating sparse arrays with multiple fill values")
def test_missing_row_can_change_a_boundary_duration_into_integer_ticks():
    def sparse(values, fill):
        return pd.arrays.SparseArray(np.array(values, dtype=object), dtype=pd.SparseDtype(object, fill))

    limit = (2**63 - 1) // 10**9
    seconds = np.timedelta64(-limit, "s")
    beyond = np.timedelta64(-limit - 1, "s")
    ns_limit = np.timedelta64(-(2**63 - 1), "ns")
    # Same seconds representation on either side of the nanosecond range.
    # The literal string "NaT" in the companion keeps this fold boxed.
    y = pd.DataFrame({0: sparse([None, None, beyond, seconds] * 2, np.nan),
                      1: sparse(["NaT", ns_limit, ns_limit, ns_limit] * 2, np.timedelta64(-limit + 1, "s"))})
    owning = pd.DataFrame([[y.iat[i, 0], y.iat[i, 1]] for i in range(len(y))], dtype=object)
    X = np.array([0, 0, 1, 1] * 2)[:, None]
    folds = [([1, 2, 4, 5, 6, 7], [0, 3]), ([0, 1, 2, 3, 5, 6], [4, 7])]

    def grid(target):
        return GridSearchCV(_MatchDurationColumn(), {"signal": [False, True]},
                            cv=_OrderedFolds(target, folds), n_jobs=1, error_score="raise")

    _check_instant_ranking(fit_shared_fold_search(grid(owning), X, owning), owning, True)
    observed = np.asarray(y.take([0, 3]))[1, 0]
    integer = isinstance(observed, (int, np.integer)) and not isinstance(observed, np.timedelta64)
    if integer:
        assert int(observed) == -limit * 10**9
    _check_instant_ranking(grid(y).fit(X, y), y, not integer)
    # Nanosecond conversion can wrap the singleton silently or raise.
    # Convert to exact whole seconds before comparing, so the reference
    # itself never needs to fit in an int64 nanosecond count.
    from datetime import timedelta
    try:
        singleton = np.asarray(y.take([2]))[0, 0]
        same_duration = isinstance(singleton, (timedelta, np.timedelta64)) and (
            int(pd.Timedelta(singleton).as_unit("s", round_ok=False).asm8.astype("i8")) == -limit - 1
        )
    except (ValueError, OverflowError):
        same_duration = False
    lossy = integer or not same_duration
    checked = grid(y)
    if lossy:
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(checked, X, y)
        assert checked.cv.calls == 0
    else:
        _check_instant_ranking(fit_shared_fold_search(checked, X, y), y, True)


@pytest.mark.filterwarnings("ignore:Concatenating sparse arrays with multiple fill values")
def test_joint_duration_rows_can_overflow_a_categorical_companion():
    limit = (2**63 - 1) // 10**6  # Whole milliseconds that fit in nanoseconds.
    first = np.array([np.timedelta64(-limit, "ms"), pd.NaT, np.timedelta64(-limit - 1, "ms"), pd.NaT], dtype=object)
    # An unused fill avoids overflow while comparing the source values at construction.
    points = pd.arrays.SparseArray(first, dtype=pd.SparseDtype(object, "unused"))
    points.fill_value = np.timedelta64(-(2**63 - 1), "ns")
    coarse, fine = pd.Timedelta(-limit + 1, "ms"), pd.Timedelta(1, "ns")
    other = pd.arrays.SparseArray(np.array([coarse, coarse, fine, fine], dtype=object),
                                  dtype=pd.SparseDtype(object, np.nan))
    durations = pd.Categorical(np.array([1, -limit - 1, 0, -limit + 1], dtype="m8[ms]"))
    y = pd.DataFrame({0: points, 1: other, 2: durations})
    owning = pd.DataFrame([[y.iat[i, j] for j in range(3)] for i in range(4)], dtype=object)
    X = np.array([0, 1, 0, 0])[:, None]
    folds = [([3], [0, 1]), ([3], [1, 0])]
    label = ("ms", -limit - 1)
    assert _matches_temporal_value(y.iat[1, 2], label)
    _check_instant_ranking(fit_shared_fold_search(_instant_grid(owning, label, folds), X, owning), owning, True)
    native = _instant_grid(y, label, folds)
    try:
        value = np.asarray(y.take([0, 1]))[1, 2]
    except (OverflowError, ValueError) as error:
        with pytest.raises(type(error)):
            native.fit(X, y)
        exact = False
    else:
        exact = _matches_temporal_value(value, label)
        if not exact:
            assert _matches_temporal_value(value, ("ns", 2**64 - (limit + 1) * 10**6))
        _check_instant_ranking(native.fit(X, y), y, exact)
    checked = _instant_grid(y, label, folds)
    if exact:
        _check_instant_ranking(fit_shared_fold_search(checked, X, y), y, True)
    else:
        with pytest.raises(ValueError, match="loses class IDs"):
            fit_shared_fold_search(checked, X, y)
        assert checked.cv.calls == 0


@pytest.mark.filterwarnings("ignore:Concatenating sparse arrays with multiple fill values")
@pytest.mark.parametrize("layout", ["repeated", "distinct", "observed_fill"])
def test_repeated_sparse_columns_share_only_equivalent_fold_checks(monkeypatch, layout):
    coarse, fine = np.datetime64(1, "s"), np.datetime64(1001, "ns")
    points = [coarse, fine, "x", fine]

    def sparse(group, fill=np.nan):
        return pd.arrays.SparseArray(np.array(group * 8, dtype=object), dtype=pd.SparseDtype(object, fill))

    columns = [sparse(points) for _ in range(16)]
    label = str(np.datetime_as_string(fine, unit="ps"))
    if layout == "distinct":
        fresh, boxed = np.datetime64(1001, "ns"), pd.Timestamp(fine)
        columns.extend([sparse([fine, coarse, "x", fine]), sparse([coarse, fresh, "x", fresh]),
                        sparse([coarse, boxed, "x", boxed]), sparse(points, float("nan"))])
    if layout == "observed_fill":
        last = sparse(["x", coarse, "x", fine], "x")
        last.fill_value = np.datetime64(2_000_001, "ps")
        columns.append(last)
        label = str(np.datetime_as_string(last.fill_value, unit="ps"))
    y = pd.DataFrame(dict(enumerate(columns)))
    X = np.array([_matches_temporal_value(y.iat[i, -1], label) for i in range(len(y))])[:, None]
    left, right = np.arange(16)[::-1], np.arange(16, 32)[::-1]
    folds = [(left, right), (right, left)]
    observed = np.asarray(y.take(left))[:, -1]
    retained = np.array_equal([_matches_temporal_value(v, label) for v in observed], X[left, 0])
    _check_instant_ranking(_instant_grid(y, label, folds).fit(X, y), y, retained)
    checked = _instant_grid(y, label, folds)
    if layout == "observed_fill":
        owning = pd.DataFrame([[y.iat[i, j] for j in range(y.shape[1])] for i in range(len(y))], dtype=object)
        _check_instant_ranking(fit_shared_fold_search(_instant_grid(owning, label, folds), X, owning), owning, True)
        if not retained or not _matches_temporal_value(np.asarray(y)[0, -1], label):
            with pytest.raises(ValueError, match="loses class IDs"):
                fit_shared_fold_search(checked, X, y)
            assert checked.cv.calls == 0
            return
    assert retained

    # Count native sparse gathers before the splitter runs. Gathers during
    # estimator fitting and scoring naturally grow with the output count.
    def representation(array):
        return tuple(dict.fromkeys(id(v) for v in array.sp_values)), id(array.fill_value)

    expected = {representation(array) for array in columns}
    native_take = pd.arrays.SparseArray.take
    gathers = []

    def counted_take(array, *args, **kwargs):
        if not checked.cv.calls:
            gathers.append(representation(array))
        return native_take(array, *args, **kwargs)

    monkeypatch.setattr(pd.arrays.SparseArray, "take", counted_take)
    _check_instant_ranking(fit_shared_fold_search(checked, X, y), y, True)
    if gathers:
        assert expected.issubset(gathers)
    assert len(gathers) <= 12 * len(expected), (len(gathers), len(expected))
