"""Repaired-RBF policy boundaries; construction only, no solver.

The private opt-in has a deliberately narrower contract than changing the
public direct evaluator. NumPy multiplication overflow is observable. Exact
Fraction arithmetic creates no NumPy overflow event; saturating its conversion
allows the subsequent real scalar exponential to report actual underflow.
Exact event multiplicity/order and bitwise equality of libm/NumPy exp are not
promised. Historical/default arithmetic remains unchanged.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from dwd._direct_rbf import accurate_rbf, _rbf_from_validated_features
from dwd._guarded_rbf import guarded_rbf
from dwd.gen_kern_dwd import KernGDWD


DEFAULT = dict(divide="warn", over="warn", invalid="warn", under="ignore")


@pytest.fixture(autouse=True)
def restore_error_policy():
    previous, callback = np.geterr(), np.geterrcall()
    np.seterr(**DEFAULT)
    np.seterrcall(None)
    try:
        yield
    finally:
        np.seterr(**previous)
        np.seterrcall(callback)


def represented(value, kind):
    value = np.asarray(value, dtype=np.float64)
    return csr_matrix(value) if kind == "csr" else value


class Sink:
    def __init__(self, fail=False):
        self.events, self.fail = [], fail

    def record(self, event):
        self.events.append(event)
        if self.fail:
            raise RuntimeError("repaired exponent callback sentinel")

    def __call__(self, description, flag):
        self.record((description, flag))

    def write(self, description):
        self.record(description)


def warning_has(records, word):
    return any(issubclass(item.category, RuntimeWarning)
               and word in str(item.message).lower() for item in records)


def expected_underflow_effects(records, sink, mode):
    assert warning_has(records, "underflow") is (mode == "warn")
    assert any("underflow" in str(event).lower() for event in sink.events) is (mode in ("call", "log"))


def exact_kernel(X, Y, gamma):
    X = X.toarray() if hasattr(X, "toarray") else X
    Y = Y.toarray() if hasattr(Y, "toarray") else Y
    result = np.empty((len(X), len(Y)), dtype=np.float64)
    with localcontext() as context:
        context.prec = 100
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                squared = sum(((Fraction.from_float(float(a)) - Fraction.from_float(float(b))) ** 2
                               for a, b in zip(x, y)), Fraction())
                exponent = Fraction.from_float(float(gamma)) * squared
                result[i, j] = float((-(Decimal(exponent.numerator) / Decimal(exponent.denominator))).exp())
    return result


@pytest.mark.parametrize("kind", ("dense", "csr"))
@pytest.mark.parametrize("relationship", ("same", "cross"))
@pytest.mark.parametrize("construction", ("sklearn", "norm_sum"))
@pytest.mark.parametrize("under", ("warn", "raise", "call", "log"))
def test_guarded_opt_in_repaired_underflow_same_and_cross(kind, relationship, construction, under):
    if relationship == "same":
        X = represented([[2**40], [2**40 + 1]], kind)
        Y, expected = X, np.eye(2)
    else:
        X, Y = represented([[2**40]], kind), represented([[2**40 + 1]], kind)
        expected = np.zeros((1, 1))
    sink = Sink()
    np.seterrcall(sink)
    policy = dict(DEFAULT, under=under)
    with np.errstate(**policy), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if under == "raise":
            with pytest.raises(FloatingPointError, match="underflow"):
                guarded_rbf(X, Y, gamma=1000., mode=construction,
                            self_kernel=relationship == "same", preserve_errors=True)
        else:
            actual = guarded_rbf(X, Y, gamma=1000., mode=construction,
                                 self_kernel=relationship == "same", preserve_errors=True)
            np.testing.assert_array_equal(actual, expected)
        assert np.geterr() == policy and np.geterrcall() is sink
    expected_underflow_effects(seen, sink, under)


@pytest.mark.parametrize("kind", ("dense", "csr"))
@pytest.mark.parametrize("under", ("call", "log"))
def test_repaired_callback_exceptions_propagate(kind, under):
    X, Y = represented([[0.]], kind), represented([[1.]], kind)
    sink = Sink(fail=True)
    np.seterrcall(sink)
    with np.errstate(under=under):
        with pytest.raises(RuntimeError, match="repaired exponent callback sentinel"):
            _rbf_from_validated_features(X, Y, 1000., 2, preserve_errors=True)
        assert np.geterr()["under"] == under and np.geterrcall() is sink
    assert sink.events


@pytest.mark.parametrize("kind", ("dense", "csr"))
@pytest.mark.parametrize("over", ("ignore", "raise"))
@pytest.mark.parametrize("under", ("warn", "raise"))
def test_real_numpy_product_overflow_precedes_repaired_exponential(kind, over, under):
    X, Y = represented([[0.]], kind), represented([[2.]], kind)
    policy = dict(DEFAULT, over=over, under=under)
    expected_error = "overflow" if over == "raise" else "underflow" if under == "raise" else None
    with np.errstate(**policy), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if expected_error:
            with pytest.raises(FloatingPointError, match=expected_error):
                _rbf_from_validated_features(X, Y, 1e308, 2, preserve_errors=True)
        else:
            actual = _rbf_from_validated_features(X, Y, 1e308, 2, preserve_errors=True)
            np.testing.assert_array_equal(actual, [[0.]])
        assert np.geterr() == policy
    assert warning_has(seen, "underflow") is (over == "ignore" and under == "warn")
    assert not warning_has(seen, "overflow")


@pytest.mark.parametrize("kind", ("dense", "csr"))
@pytest.mark.parametrize("over", ("ignore", "raise"))
@pytest.mark.parametrize("under", ("warn", "raise"))
def test_exact_fraction_conversion_saturation_has_no_invented_overflow_event(kind, over, under):
    X, Y = represented([[-1e308]], kind), represented([[1e308]], kind)
    policy = dict(DEFAULT, over=over, under=under)
    with np.errstate(**policy), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if under == "raise":
            with pytest.raises(FloatingPointError, match="underflow"):
                _rbf_from_validated_features(X, Y, 1., 2, preserve_errors=True)
        else:
            actual = _rbf_from_validated_features(X, Y, 1., 2, preserve_errors=True)
            np.testing.assert_array_equal(actual, [[0.]])
        assert np.geterr() == policy
    assert not warning_has(seen, "overflow")
    assert warning_has(seen, "underflow") is (under == "warn")


@pytest.mark.parametrize("kind", ("dense", "csr"))
@pytest.mark.parametrize("coordinate,gamma", ((1e155, 1e-308), (1e-160, 1e308)))
def test_fraction_recovery_of_representable_kernel_does_not_report_incidental_distance_range(kind, coordinate, gamma):
    X, Y = represented([[0.]], kind), represented([[coordinate]], kind)
    with np.errstate(all="raise"):
        actual = _rbf_from_validated_features(X, Y, gamma, 2, preserve_errors=True)
        historical = _rbf_from_validated_features(X, Y, gamma, 2)
    # Both old and revised exceptional routes round the exact exponent to
    # binary64 before exp. An oracle that omits that rounding confounds two
    # different errors, notably when exponent rounding near 100 is amplified.
    exponent = Fraction.from_float(gamma) * Fraction.from_float(coordinate) ** 2
    represented_exponent = float(exponent)
    exponent_error = abs(Fraction.from_float(represented_exponent) - exponent)
    with localcontext() as context:
        context.prec = 120
        exact_exponent = Decimal(exponent.numerator) / Decimal(exponent.denominator)
        exact_value = (-exact_exponent).exp()
        represented_value = (-Decimal.from_float(represented_exponent)).exp()
        reference = np.array([[float(represented_value)]])
        np.testing.assert_array_max_ulp(actual, reference, maxulp=4)
        np.testing.assert_array_max_ulp(historical, reference, maxulp=4)
        np.testing.assert_array_max_ulp(actual, historical, maxulp=4)
        # This directly checks the old retained arithmetic, without claiming
        # correctly-rounded exp of the unrounded exact Fraction exponent.
        assert historical[0, 0] == math.exp(-represented_exponent)

        # Let e be the exact exponent and er its binary64 rounding. Then
        # |exp(-er)-exp(-e)| <= exp(-e)*(exp(|er-e|)-1).
        # Add only the retained four-ULP exp allowance and the half-ULP rounding
        # of the independently evaluated Decimal reference to binary64.
        delta = Decimal(exponent_error.numerator) / Decimal(exponent_error.denominator)
        exponent_budget = exact_value * (delta.exp() - Decimal(1))
        reference_ulp = math.ulp(float(reference[0, 0]))
        for result in (float(actual[0, 0]), float(historical[0, 0])):
            exp_budget = (Decimal(4) * Decimal.from_float(max(math.ulp(result), reference_ulp))
                          + Decimal.from_float(reference_ulp) / Decimal(2))
            exact_oracle_error = abs(Decimal.from_float(result) - exact_value)
            assert exact_oracle_error <= exponent_budget + exp_budget, (
                f"Exact-oracle error {exact_oracle_error}; exponent rounding "
                f"{delta}; derived exponent budget {exponent_budget}; "
                f"retained exp/rounding budget {exp_budget}")


@pytest.mark.parametrize("kind", ("dense", "csr"))
@pytest.mark.parametrize("range_case", ("normal-product-overflow", "fraction-conversion-overflow"))
@pytest.mark.parametrize("entry", ("public", "private-default", "private-false"))
def test_public_direct_and_private_default_preserve_historical_policy(kind, range_case, entry):
    values, gamma = (([[0.]], [[2.]]), 1e308) if range_case == "normal-product-overflow" else (([[-1e308]], [[1e308]]), 1.)
    X, Y = represented(values[0], kind), represented(values[1], kind)
    with np.errstate(all="raise"), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if entry == "public":
            actual = accurate_rbf(X, Y, gamma=gamma)
        elif entry == "private-default":
            actual = _rbf_from_validated_features(X, Y, gamma, 2)
        else:
            actual = _rbf_from_validated_features(X, Y, gamma, 2, preserve_errors=False)
    np.testing.assert_array_equal(actual, [[0.]])
    assert not seen


@pytest.mark.parametrize("kind", ("dense", "csr"))
@pytest.mark.parametrize("construction", ("sklearn", "norm_sum"))
@pytest.mark.parametrize("under", ("warn", "raise", "call", "log"))
def test_existing_v1_prediction_repair_remains_masked(kind, construction, under):
    X = represented([[2**40], [2**40 + 1]], kind)
    estimator = KernGDWD(kernel="rbf", kernel_kws={"gamma": 1000.})
    estimator._Xfit = X
    estimator.kernel_computation_ = "guarded_" + construction + "_rel_v1"
    sink = Sink()
    np.seterrcall(sink)
    with np.errstate(under=under), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        actual = estimator._compute_kernel(X.copy())
        assert np.geterr()["under"] == under and np.geterrcall() is sink
    np.testing.assert_array_equal(actual, np.eye(2))
    assert not seen and not sink.events


@pytest.mark.parametrize("kind", ("dense", "csr"))
@pytest.mark.parametrize("construction", ("sklearn", "norm_sum"))
def test_repaired_values_remain_close_across_no_event_policies(kind, construction):
    X = represented(np.array([[0., 0.], [.25, .5], [1., 2.]]) + 2**40, kind)
    Y = represented(np.array([[.125, .25], [.5, 1.]]) + 2**40, kind)
    values = []
    for policy in (DEFAULT, dict(DEFAULT, under="warn"), dict(DEFAULT, under="raise")):
        with np.errstate(**policy), warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            values.append(guarded_rbf(X, Y, gamma=.125, mode=construction, preserve_errors=True))
            assert np.geterr() == policy
        assert not seen
    expected = exact_kernel(X, Y, .125)
    for actual in values:
        np.testing.assert_array_max_ulp(actual, expected, maxulp=4)
        np.testing.assert_array_max_ulp(actual, values[0], maxulp=4)
    # Existing ordinary arithmetic must stay byte-identical to the old default.
    with np.errstate(**DEFAULT):
        historical = guarded_rbf(X, Y, gamma=.125, mode=construction)
    np.testing.assert_array_equal(values[0], historical)


@pytest.mark.parametrize("kind", ("dense", "csr"))
@pytest.mark.parametrize("shortcut", ("zero-gamma", "identical-extreme"))
def test_exact_constant_kernel_shortcuts_do_not_invent_events(kind, shortcut):
    X = represented([[1e308]], kind)
    Y = represented([[-1e308] if shortcut == "zero-gamma" else [1e308]], kind)
    gamma = 0. if shortcut == "zero-gamma" else 1e308
    with np.errstate(all="raise"):
        actual = _rbf_from_validated_features(X, Y, gamma, 2, preserve_errors=True)
    np.testing.assert_array_equal(actual, [[1.]])
