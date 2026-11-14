"""RBF numerical geometry, caller policy, and historical-model regressions.

No all-raise solver fit is required: the separate F4 nextafter issue is outside
this test's scope. Construction-only tests do not claim fitted convergence.
"""
from collections import OrderedDict, UserDict
from decimal import Decimal, localcontext
from fractions import Fraction
import pickle
import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import rbf_kernel

from dwd.gen_kern_dwd import KernGDWD, get_K_eig
from dwd import kernel_utils


DEFAULT = dict(divide="warn", over="warn", invalid="warn", under="ignore")
DELTA = np.array([[0., 0.], [1., 0.], [0., 2.], [1., 2.]])
LABELS = np.array([-1, 1, -1, 1])


@pytest.fixture(autouse=True)
def restore_policy_and_handler():
    before, handler = np.geterr(), np.geterrcall()
    np.seterr(**DEFAULT)
    np.seterrcall(None)
    try:
        yield
    finally:
        np.seterr(**before)
        np.seterrcall(handler)


def dense(value):
    return value.toarray() if hasattr(value, "toarray") else np.asarray(value)


def represented(value, kind):
    return csr_matrix(value) if kind == "csr" else np.array(value, dtype=np.float64)


def model(kws=None):
    return KernGDWD(kernel="rbf", kernel_kws={"gamma": .5} if kws is None else kws,
                   backend="spectral", lambd=.13, q=1,
                   initialization="zero", stopping="fixed", max_iter=3, random_state=7)


def construct(features, kws):
    """Exercise the real training-kernel boundary, separately from the solver."""
    estimator = model(kws)
    estimator._Xfit = features.copy()
    value = estimator._compute_training_kernel(estimator._Xfit)
    return estimator, value


def oracle(left, right, gamma):
    """Exact represented binary64 distances, then a 100-digit Decimal exp."""
    left, right = dense(left), dense(right)
    scale = Fraction.from_float(float(gamma))
    output = np.empty((len(left), len(right)))
    with localcontext() as context:
        context.prec = 100
        for i, x in enumerate(left):
            for j, y in enumerate(right):
                distance = sum(((Fraction.from_float(float(a)) - Fraction.from_float(float(b))) ** 2
                                for a, b in zip(x, y)), Fraction())
                exponent = scale * distance
                output[i, j] = float((-(Decimal(exponent.numerator) / Decimal(exponent.denominator))).exp())
    return output


def assert_geometry(actual, expected):
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=4e-15, atol=1e-15)


@pytest.mark.parametrize("kind", ("dense", "csr"))
@pytest.mark.parametrize("offset", (0., float(2**40)))
@pytest.mark.parametrize("dtype", (float, np.float16, np.float32, np.float64))
@pytest.mark.parametrize("mapping", (dict, OrderedDict))
@pytest.mark.parametrize("under", ("ignore", "warn"))
def test_equal_half_gamma_representations_keep_the_same_geometry(kind, offset, dtype, mapping, under):
    X = represented(DELTA + offset, kind)
    original = dense(X).copy()
    kws, policy = mapping(gamma=dtype(.5)), dict(DEFAULT, under=under)
    expected = oracle(X, X, .5)
    with np.errstate(**policy), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        estimator, actual = construct(X, kws)
        cross = estimator._compute_kernel(X.copy())
        assert np.geterr() == policy and np.geterrcall() is None
    assert estimator.kernel_computation_ == "guarded_norm_sum_rel_v2"
    np.testing.assert_array_equal(dense(X), original)
    assert not seen  # This exact dyadic fixture has no actual overflow/underflow.
    assert_geometry(actual, expected)
    assert_geometry(cross, expected)
    np.testing.assert_array_equal(np.diag(actual), np.ones(X.shape[0]))


class Sink:
    def __init__(self, fail=False):
        self.events, self.fail = [], fail

    def _record(self, event):
        self.events.append(event)
        if self.fail:
            raise RuntimeError("F16 handler sentinel")

    def __call__(self, description, flag):
        self._record((description, flag))

    def write(self, description):
        self._record(description)


@pytest.mark.parametrize("under", ("warn", "raise", "call", "log", "print"))
@pytest.mark.parametrize("kind", ("dense", "csr"))
def test_nonordinary_policy_does_not_silently_select_wrong_no_event_geometry(under, kind):
    X, sink = represented(DELTA + 2**40, kind), Sink()
    np.seterrcall(sink)
    policy = dict(DEFAULT, under=under)
    with np.errstate(**policy):
        estimator, actual = construct(X, {"gamma": .5})
        assert np.geterr() == policy and np.geterrcall() is sink
    reference = oracle(X, X, .5)
    assert estimator.kernel_computation_ == "guarded_norm_sum_rel_v2"
    assert_geometry(actual, reference)
    assert sink.events == []


@pytest.mark.parametrize("kind", ("dense", "csr"))
def test_float32_point_one_preserves_its_value_not_the_python_literal(kind):
    X = represented(DELTA + 2**40, kind)
    scalar = np.float32(.1)
    first, K32 = construct(X, {"gamma": scalar})
    _, K64 = construct(X, {"gamma": .1})
    reference32, reference64 = oracle(X, X, float(scalar)), oracle(X, X, .1)
    assert np.max(np.abs(reference32 - reference64)) > 1e-10
    assert first.kernel_kws["gamma"] is scalar
    assert_geometry(K32, reference32)
    assert_geometry(K64, reference64)


@pytest.mark.parametrize("dtype", (np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.uint16, np.uint32, np.uint64))
def test_safe_ordinary_integer_types_and_bounded_historical_behavior(dtype):
    estimator = model({"gamma": dtype(1)})
    assert estimator._guarded_rbf_eligible() is True
    _, actual = construct(DELTA + 2**40, estimator.kernel_kws)
    assert_geometry(actual, oracle(DELTA, DELTA, 1.))


def test_new_uint64_requires_losslessness_while_old_int64_rounding_is_preserved():
    unsafe_unsigned, historical = np.uint64(2**53 + 1), np.int64(2**53 + 1)
    assert int(float(unsafe_unsigned)) != int(unsafe_unsigned)
    assert model({"gamma": unsafe_unsigned})._guarded_rbf_eligible() is False
    assert model({"gamma": historical})._guarded_rbf_eligible() is True
    safe = model({"gamma": np.uint64(2**53)})
    assert safe._guarded_rbf_eligible() is True


@pytest.mark.parametrize("increment", (False, True))
def test_longdouble_is_not_silently_narrowed_before_fallback(increment, monkeypatch):
    gamma = np.longdouble(.5)
    if increment:
        gamma = gamma + np.ldexp(np.longdouble(1), -60)
    estimator = model({"gamma": gamma})
    admitted_alias = type(gamma) in (float, np.float64)
    assert estimator._guarded_rbf_eligible() is admitted_alias
    if np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant and increment:
        assert gamma != np.longdouble(float(gamma))
    seen = []

    def original_route(X, Y, metric, **kwargs):
        seen.append(kwargs["gamma"])
        return np.eye(len(X))

    monkeypatch.setattr(kernel_utils, "pairwise_kernels", original_route)
    estimator._Xfit = DELTA.copy()
    estimator._compute_training_kernel(estimator._Xfit)
    assert len(seen) == 1 and seen[0] is gamma


class HostileDict(dict):
    def get(self, *args, **kwargs):
        raise AssertionError("Custom mapping get must not run during eligibility")


class HostileOrderedDict(OrderedDict):
    def get(self, *args, **kwargs):
        raise AssertionError("Custom OrderedDict get must not run during eligibility")


class HostileFloat(float):
    def __float__(self):
        raise AssertionError("Custom float coercion must not run")


@pytest.mark.parametrize("kws", (HostileDict(gamma=.5), HostileOrderedDict(gamma=.5),
                                  UserDict(gamma=.5), {"gamma": HostileFloat(.5)}))
def test_custom_mappings_and_scalar_overrides_remain_ineligible(kws):
    assert model(kws)._guarded_rbf_eligible() is False


@pytest.mark.parametrize("gamma", (-.5, np.nan, np.inf, -np.inf, complex(.5), np.float32(np.nan)))
def test_invalid_or_nonreal_gamma_never_enters_guarded_construction(gamma):
    assert model({"gamma": gamma})._guarded_rbf_eligible() is False


@pytest.mark.parametrize("gamma", (None, 0., True))
def test_existing_default_zero_and_bool_gamma_remain_supported(gamma):
    estimator = model({"gamma": gamma})
    assert estimator._guarded_rbf_eligible() is True
    _, actual = construct(DELTA, estimator.kernel_kws)
    assert_geometry(actual, oracle(DELTA, DELTA, .5 if gamma is None else float(gamma)))


@pytest.mark.parametrize("mode", ("ignore", "raise", "call", "log", "print"))
def test_actual_exponential_underflow_obeys_caller_policy(mode, capfd):
    X, sink = np.array([[0.], [40.]]), Sink()
    np.seterrcall(sink)
    policy = dict(DEFAULT, under=mode)
    propagates = mode != "ignore"
    with np.errstate(**policy):
        if propagates and mode == "raise":
            with pytest.raises(FloatingPointError, match="underflow"):
                construct(X, {"gamma": .5})
        else:
            _, actual = construct(X, {"gamma": .5})
            np.testing.assert_array_equal(actual, np.eye(2))
        assert np.geterr() == policy and np.geterrcall() is sink
    output = capfd.readouterr()
    assert bool(sink.events) is (propagates and mode in ("call", "log"))
    assert ("underflow" in output.err.lower()) is (propagates and mode == "print")


def test_caller_underflow_warning_contract():
    with np.errstate(under="warn"), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        construct(np.array([[0.], [40.]]), {"gamma": .5})
    assert any(issubclass(item.category, RuntimeWarning) and "underflow" in str(item.message).lower()
               for item in seen), "Real exponential underflow was silently masked"


@pytest.mark.parametrize("mode", ("call", "log"))
def test_active_handler_exception_is_not_swallowed(mode):
    sink = Sink(fail=True)
    np.seterrcall(sink)
    with np.errstate(under=mode):
        with pytest.raises(RuntimeError, match="F16 handler sentinel"):
            construct(np.array([[0.], [40.]]), {"gamma": .5})
        assert sink.events
        assert np.geterr()["under"] == mode and np.geterrcall() is sink


@pytest.mark.parametrize("route", ("sklearn", "norm_sum"))
def test_saved_construction_routes_retain_their_declared_error_policy(route):
    estimator = model()
    estimator._Xfit = np.array([[0.], [40.]])
    estimator.kernel_computation_ = "guarded_" + route + "_rel_v2"
    with np.errstate(under="warn"), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        actual = estimator._compute_kernel(estimator._Xfit)
    np.testing.assert_array_equal(actual, np.eye(2))
    underflow = any("underflow" in str(item.message).lower() for item in seen)
    assert underflow is True


@pytest.mark.parametrize("kind", ("dense", "csr"))
@pytest.mark.parametrize("values,gamma", [
    ([[0.], [1e155], [2e155]], 1e-308),
    ([[0.], [1e-160], [2e-160]], 1e308),
    ([[-1e308], [1e308]], 0.),
])
def test_existing_extreme_range_geometry_under_ordinary_policy(kind, values, gamma):
    X = represented(values, kind)
    with np.errstate(all="ignore"):
        _, actual = construct(X, {"gamma": gamma})
    assert_geometry(actual, oracle(X, X, gamma))


@pytest.mark.parametrize("kind", ("dense", "csr"))
@pytest.mark.parametrize("change", ("float32", "ordered", "underwarn"))
def test_real_fit_and_continuous_prediction_follow_corrected_geometry(kind, change):
    X = represented(DELTA + 2**40, kind)
    Q = represented(np.vstack((DELTA + [.25, .125], [[.5, 1.]])) + 2**40, kind)
    control = model().fit(X, LABELS)
    kws = OrderedDict(gamma=.5) if change == "ordered" else {"gamma": np.float32(.5) if change == "float32" else .5}
    policy = dict(DEFAULT, under="warn" if change == "underwarn" else "ignore")
    with np.errstate(**policy):
        actual = model(kws).fit(X, LABELS)
        K = actual._compute_kernel(actual._Xfit)
        query_K = actual._compute_kernel(Q)
        scores = actual.decision_function(Q)
    assert np.isfinite(scores).all()
    assert actual.kernel_computation_ == "guarded_norm_sum_rel_v2"
    assert_geometry(K, oracle(X, X, .5))
    assert_geometry(query_K, oracle(X, Q, .5))
    np.testing.assert_allclose(scores, control.decision_function(Q), rtol=5e-8, atol=5e-10)
    np.testing.assert_array_equal(actual.predict(Q), control.predict(Q))


@pytest.mark.parametrize("change", ("gamma_type", "mapping_type"))
@pytest.mark.parametrize("reverse", (False, True))
def test_concrete_parameter_type_changes_still_invalidate_cv_cache(change, reverse):
    old = {"gamma": .5}
    new = {"gamma": np.float32(.5)} if change == "gamma_type" else OrderedDict(gamma=.5)
    if reverse:
        old, new = new, old
    estimator = model(old).cv_init(DELTA)
    estimator.set_params(kernel_kws=new)
    assert estimator._cv_cache_matches(DELTA) is False


@pytest.mark.parametrize("historical", ("unversioned", "versioned-legacy", "versioned-guarded"))
def test_historical_cv_geometry_requires_new_construction_but_not_old_prediction_rewriting(historical):
    X = DELTA + 2**40
    estimator = model().cv_init(X)
    guarded = historical == "versioned-guarded"
    old_K = oracle(X, X, .5) if guarded else rbf_kernel(X, X, gamma=.5)
    estimator._cv_K = old_K.copy()
    estimator._K_eig = get_K_eig(old_K)
    estimator._cv_kernel_computation = "guarded_sklearn_rel_v1" if guarded else "sklearn"
    if historical == "unversioned":
        del estimator._cv_rbf_generation
    else:
        estimator._cv_rbf_generation = ("relative_guard_v1", estimator._cv_kernel_computation)
    before = estimator._cv_K.copy()
    assert estimator._cv_cache_matches(X) is False
    np.testing.assert_array_equal(estimator._cv_K, before)
    estimator.fit(X, LABELS)
    control = model().fit(X, LABELS)
    assert estimator.kernel_computation_ == "guarded_norm_sum_rel_v2"
    assert_geometry(estimator._compute_kernel(X.copy()), oracle(X, X, .5))
    np.testing.assert_allclose(estimator.decision_function(X.copy()), control.decision_function(X.copy()),
                               rtol=5e-8, atol=5e-10)


@pytest.mark.parametrize("route", ("guarded_sklearn_rel_v1", "guarded_norm_sum_rel_v1"))
def test_old_fitted_v1_prediction_keeps_its_owned_geometry_and_masking_after_pickle(route):
    fitted = model().fit(DELTA, LABELS)
    fitted.kernel_computation_ = route
    restored = pickle.loads(pickle.dumps(fitted))
    query = np.array([[40., 0.]])
    with np.errstate(under="warn"), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        actual = restored._compute_kernel(query)
        prediction = restored.decision_function(query)
    np.testing.assert_array_equal(actual, np.zeros((4, 1)))
    assert np.isfinite(prediction).all() and not seen
    assert restored.kernel_computation_ == route


@pytest.mark.parametrize("kind", ("dense", "csr"))
@pytest.mark.parametrize("under", ("ignore", "warn"))
def test_real_fitted_batch_prediction_preserves_geometry_scores_and_caller_policy(kind, under):
    from dwd._prediction_norms import prepare_norm_kernel

    X = represented(DELTA + 2**40, kind)
    Q = represented(np.vstack((DELTA + [.25, .125], [[40., 40.]])) + 2**40, kind)
    fitted = model().fit(X, LABELS)
    assert fitted.kernel_computation_ == "guarded_norm_sum_rel_v2"
    policy = dict(DEFAULT, under=under)
    captured = []
    with np.errstate(**policy):
        prepared = prepare_norm_kernel(fitted, Q, Q)
        assert callable(prepared) if under == "ignore" else prepared is None
        for batch in (None, 2):
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                scores = fitted.decision_function(Q, prediction_batch_size=batch)
            captured.append((scores, seen))
            assert np.geterr() == policy and np.geterrcall() is None
    unbatched, batched = captured[0][0], captured[1][0]
    assert np.isfinite(unbatched).all() and np.isfinite(batched).all()
    np.testing.assert_allclose(batched, unbatched, rtol=5e-8, atol=5e-10)
    np.testing.assert_array_equal(batched >= 0., unbatched >= 0.)
    # Compare real fitted coefficients against an independently computed kernel;
    # equality of two equally wrong batch routes would not establish geometry.
    expected_scores = np.asarray(fitted._kernel_decision_product(oracle(X, Q, .5))).ravel()
    np.testing.assert_allclose(unbatched, expected_scores, rtol=5e-8, atol=5e-10)
    # Arithmetic-policy signalling is tested at the construction-only boundary
    # in test_rbf_repaired_error_policy.py. Score reduction has a separate nextafter
    # underflow issue (F4), so its warnings cannot establish exponent behaviour.
