"""F-7 public score/label safety, unchanged finite arithmetic, and batch casts.

The same expectations apply to the baseline and every candidate. No expected
failure is conditional on the implementation being evaluated.
"""
import warnings

import numpy as np
import pytest
from legacy_solver_fixtures import load_legacy_case
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator

from dwd.gen_kern_dwd import KernGDWD
from dwd.kern_md import KernMD
from dwd.kernel_utils import KernelClfMixin


def _dense_kernel(train, query):
    return train @ query.T


def _csr_kernel(train, query):
    return csr_matrix(train @ query.T)


class _FittedKernelClassifier(KernelClfMixin, BaseEstimator):
    """Finite fitted state isolates public prediction from fit convergence."""

    def __init__(self, kernel="precomputed"):
        self.kernel = kernel
        self.kernel_kws = None
        self._Xfit = np.eye(2)
        self.dual_coef_ = np.array([[1., -0.5]])
        self.intercept_ = np.array([0.])
        self.classes_ = np.array(["negative", "positive"])
        self.prediction_batch_size = None


class _ScoreHookClassifier(_FittedKernelClassifier):
    def __init__(self, bad, columns):
        super().__init__()
        self.bad = bad
        self.columns = columns
        if columns == 3:
            self.dual_coef_ = np.array([[1., 0.], [0., 1.], [-1., -1.]])
            self.intercept_ = np.zeros(3)
            self.classes_ = np.array(["first", "second", "third"])

    def _kernel_decision_product(self, kernel):
        values = super()._kernel_decision_product(kernel)
        if self.bad is not None:
            values[:, -1] = self.bad
        return values


def _arrays(model):
    return (model._Xfit, model.dual_coef_, model.intercept_, model.classes_)


def _save(array):
    if issparse(array):
        return (array.shape, array.data.copy(), array.indices.copy(), array.indptr.copy())
    return array.copy()


def _unchanged(array, before):
    if issparse(array):
        assert array.shape == before[0]
        for actual, expected in zip((array.data, array.indices, array.indptr), before[1:]):
            assert actual.dtype == expected.dtype
            assert_array_equal(actual, expected)
    else:
        assert array.dtype == before.dtype
        assert_array_equal(array, before)


def _assert_inputs_unchanged(model, query, before):
    for actual, saved in zip((*_arrays(model), query), before):
        _unchanged(actual, saved)


@pytest.mark.parametrize("kind", ["mean_difference", "legacy_dwd"])
@pytest.mark.parametrize("batch", [None, 1, 3])
@pytest.mark.parametrize("method", ["decision_function", "predict"])
def test_finite_extreme_polynomial_queries_raise_and_model_remains_usable(kind, batch, method):
    rng = np.random.RandomState(13)
    train = rng.normal(size=(8, 2))
    labels = np.tile([0, 1], 4)
    if kind == "mean_difference":
        model = KernMD(kernel="poly", kernel_kws={"degree": 4})
    else:
        model = load_legacy_case('polynomial')['model']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if kind == 'mean_difference':
            model.fit(train, labels)
    model.prediction_batch_size = batch
    ordinary = train[:5].copy()
    expected_scores = model.decision_function(ordinary)
    expected_labels = model.predict(ordinary)
    query = ordinary * 1e100
    assert np.isfinite(query).all()
    before = tuple(_save(a) for a in (*_arrays(model), query))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="Nonfinite"):
            getattr(model, method)(query)
    _assert_inputs_unchanged(model, query, before)
    assert_array_equal(model.decision_function(ordinary), expected_scores)
    assert_array_equal(model.predict(ordinary), expected_labels)


_ROUTES = [
    ("precomputed", False), ("precomputed", True),
    ("linear", False), ("linear", True),
    (_dense_kernel, False), (_csr_kernel, False),
]


@pytest.mark.parametrize("kernel,sparse", _ROUTES,
                         ids=["precomputed-dense", "precomputed-csr", "linear-dense",
                              "linear-csr", "callable-dense", "callable-csr-output"])
@pytest.mark.parametrize("batch", [None, 1, 2, 10])
def test_finite_scores_labels_shapes_and_sources_are_unchanged(kernel, sparse, batch):
    model = _FittedKernelClassifier(kernel)
    # Dyadic operands make the intended old operation exact in these cases.
    dense = np.array([[0., 0.], [1., 2.], [2., 1.], [-2., 1.], [.5, -.5]])
    query = csr_matrix(dense) if sparse else dense.copy()
    expected = dense @ model.dual_coef_[0] + model.intercept_[0]
    before = tuple(_save(a) for a in (*_arrays(model), query))
    model.prediction_batch_size = batch
    actual = model.decision_function(query)
    assert actual.shape == (len(dense),)
    assert_array_equal(actual, expected)
    assert_array_equal(model.predict(query), model.classes_[(expected > 0).astype(int)])
    # The explicit override has the same result and does not change the setting.
    assert_array_equal(model.decision_function(query, prediction_batch_size=None), expected)
    assert_array_equal(model.decision_function(query, prediction_batch_size=2), expected)
    assert model.prediction_batch_size == batch
    _assert_inputs_unchanged(model, query, before)


def _overflow_case(kind):
    model = _FittedKernelClassifier()
    largest = np.finfo(float).max
    if kind == "dot_positive":
        model.dual_coef_ = np.array([[2., 0.]])
        row = [largest, 0.]
    elif kind == "dot_negative":
        model.dual_coef_ = np.array([[2., 0.]])
        row = [-largest, 0.]
    elif kind == "dot_nan":
        model.dual_coef_ = np.array([[2., -2.]])
        row = [largest, largest]
    elif kind == "add_positive":
        model.dual_coef_ = np.array([[1., 0.]])
        model.intercept_ = np.array([largest])
        row = [largest, 0.]
    else:
        assert kind == "add_negative"
        model.dual_coef_ = np.array([[1., 0.]])
        model.intercept_ = np.array([-largest])
        row = [-largest, 0.]
    # Include valid rows before the bad final row, exercising a final partial batch.
    return model, np.array([[0., 0.], [0., 0.], row])


@pytest.mark.parametrize("kind", ["dot_positive", "dot_negative", "dot_nan",
                                  "add_positive", "add_negative"])
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "csr"])
@pytest.mark.parametrize("batch", [None, 2])
def test_finite_operands_with_nonfinite_dot_or_add_never_produce_public_results(kind, sparse, batch):
    model, dense = _overflow_case(kind)
    query = csr_matrix(dense) if sparse else dense.copy()
    model.prediction_batch_size = batch
    assert np.isfinite(dense).all()
    assert np.isfinite(model.dual_coef_).all()
    assert np.isfinite(model.intercept_).all()
    before = tuple(_save(a) for a in (*_arrays(model), query))
    with np.errstate(over="ignore", invalid="ignore"):
        for method in (model.decision_function, model.predict):
            with pytest.raises(FloatingPointError, match="Nonfinite"):
                method(query)
    _assert_inputs_unchanged(model, query, before)
    safe = np.zeros((3, 2))
    expected = np.full(3, model.intercept_[0])
    assert_array_equal(model.decision_function(safe), expected)
    assert_array_equal(model.predict(safe), model.classes_[(expected > 0).astype(int)])


@pytest.mark.parametrize("mode", ["ignore", "raise"])
@pytest.mark.parametrize("batch", [None, 2])
@pytest.mark.parametrize("method", ["decision_function", "predict"])
def test_caller_error_state_is_restored_on_prediction_failure(mode, batch, method):
    model, query = _overflow_case("add_positive")
    model.prediction_batch_size = batch
    original = np.geterr().copy()
    with np.errstate(over=mode, invalid=mode):
        inside = np.geterr().copy()
        with pytest.raises(FloatingPointError):
            getattr(model, method)(query)
        assert np.geterr() == inside
    assert np.geterr() == original


@pytest.mark.parametrize("batch", [None, 2])
def test_overflow_callback_remains_observable_before_public_rejection(batch):
    model, query = _overflow_case("add_positive")
    model.prediction_batch_size = batch
    original_errors = np.geterr().copy()
    original_callback = np.geterrcall()
    calls = []

    def record(error, flag):
        calls.append((error, flag))

    np.seterrcall(record)
    try:
        with np.errstate(over="call", invalid="call"):
            inside = np.geterr().copy()
            with pytest.raises(FloatingPointError, match="Nonfinite"):
                model.decision_function(query)
            assert np.geterr() == inside
            assert np.geterrcall() is record
            assert any("overflow" in error and flag & 2 for error, flag in calls)
    finally:
        np.seterrcall(original_callback)
    assert np.geterr() == original_errors
    assert np.geterrcall() is original_callback


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "positive-inf", "negative-inf"])
@pytest.mark.parametrize("columns", [1, 3], ids=["binary", "multicolumn"])
@pytest.mark.parametrize("batch", [None, 2])
def test_nonfinite_score_hooks_cannot_escape_public_decisions_or_labels(bad, columns, batch):
    model = _ScoreHookClassifier(bad, columns)
    model.prediction_batch_size = batch
    query = np.array([[1., 0.], [0., 1.], [-1., -1.]])
    before = tuple(_save(a) for a in (*_arrays(model), query))
    for method in (model.decision_function, model.predict):
        with pytest.raises(FloatingPointError, match="Nonfinite"):
            method(query)
    _assert_inputs_unchanged(model, query, before)
    model.bad = None
    expected = query @ model.dual_coef_.T + model.intercept_
    actual = model.decision_function(query)
    assert_array_equal(actual, expected.ravel() if columns == 1 else expected)
    indices = ((expected[:, 0] > 0).astype(int) if columns == 1 else expected.argmax(axis=1))
    assert_array_equal(model.predict(query), model.classes_[indices])


_WIDER_LONGDOUBLE = np.finfo(np.longdouble).maxexp > np.finfo(np.float64).maxexp


def _wide_query():
    huge = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)
    query = np.zeros((3, 2), dtype=np.longdouble)
    query[:, 0] = [huge / 4, huge / 2, huge]
    assert np.isfinite(query).all()
    model = _FittedKernelClassifier()
    model.dual_coef_ = np.array([[1., 0.]])
    return model, query


@pytest.mark.skipif(not _WIDER_LONGDOUBLE, reason="longdouble has no wider exponent range on this platform")
def test_finite_wide_unbatched_scores_are_not_rounded_or_refused():
    model, query = _wide_query()
    scores = model.decision_function(query)
    assert scores.dtype == np.dtype(np.longdouble)
    assert np.isfinite(scores).all()
    assert scores[-1] > np.longdouble(np.finfo(np.float64).max)
    assert_array_equal(scores, query[:, 0])
    assert_array_equal(model.predict(query), np.full(3, "positive"))


@pytest.mark.skipif(not _WIDER_LONGDOUBLE, reason="longdouble has no wider exponent range on this platform")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("method", ["decision_function", "predict"])
def test_finite_wide_scores_overflowing_the_batch_output_are_rejected(batch, method):
    model, query = _wide_query()
    # The real ordinary product is finite. The failure occurs later when the
    # existing batching array converts its output to float64.
    with np.errstate(over="ignore", invalid="ignore"):
        raw = model._kernel_decision_product(query.T)
        assert raw.dtype == np.dtype(np.longdouble)
        assert np.isfinite(raw).all()
        assert_array_equal(raw[:, 0], query[:, 0])
        model.prediction_batch_size = batch
        before = tuple(_save(a) for a in (*_arrays(model), query))
        with pytest.raises(FloatingPointError, match="Nonfinite"):
            getattr(model, method)(query)
    _assert_inputs_unchanged(model, query, before)
    model.prediction_batch_size = None
    assert_array_equal(model.decision_function(query), query[:, 0])


@pytest.mark.parametrize("precision,affine", [("compensated", "standard"),
                                              ("adaptive", "standard"),
                                              ("compensated", "joint"),
                                              ("adaptive", "joint")])
@pytest.mark.parametrize("batch", [None, 2])
def test_existing_accurate_and_joint_routes_keep_exact_finite_results(precision, affine, batch):
    model = _FittedKernelClassifier()
    model.prediction_precision_ = precision
    model.affine_computation_ = affine
    model.prediction_batch_size = batch
    model.intercept_ = np.array([.125])
    query = np.array([[1., 2.], [0., 0.], [1., -1.], [-1., 1.], [.5, .25]])
    expected = query @ model.dual_coef_[0] + model.intercept_[0]
    before = tuple(_save(a) for a in (*_arrays(model), query))
    with np.errstate(under="ignore"):
        assert_array_equal(model.decision_function(query), expected)
        assert_array_equal(model.predict(query), model.classes_[(expected > 0).astype(int)])
    _assert_inputs_unchanged(model, query, before)


@pytest.mark.parametrize("batch", [None, 1, 2])
def test_largest_finite_and_zero_scores_remain_valid(batch):
    model = _FittedKernelClassifier()
    model.dual_coef_ = np.array([[1., 0.]])
    model.prediction_batch_size = batch
    largest = np.finfo(float).max
    query = np.array([[largest, 0.], [-largest, 0.], [0., 0.]])
    expected = query[:, 0]
    assert_array_equal(model.decision_function(query), expected)
    assert_array_equal(model.predict(query), ["positive", "negative", "negative"])
