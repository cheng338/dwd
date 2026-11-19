"""Stored sparse entries are interpreted in binary64 before aggregation.

Independent Fraction oracles cover duplicate cancellation, public prediction,
caller immutability and the ordinary, subnormal and extended-precision paths.
Fixed fitted states isolate prediction; these tests do not validate training.
"""
from fractions import Fraction

import numpy as np
import pytest
from scipy.sparse import csr_array, csr_matrix, issparse


@pytest.fixture(autouse=True)
def _explicit_numpy_policy():
    # Keep floating-point policy local to each test and restore it afterward.
    # Outward nextafter steps may underflow; strict-underflow policy is outside
    # these working-representation regressions.
    with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
        yield


_DTYPES = [np.int64, np.float32, np.float64, np.longdouble]
_KINDS = [csr_matrix, csr_array]
_ROUTES = ["kernel_adaptive", "kernel_compensated", "affine_adaptive", "affine_compensated"]


def _exact_binary64(K, alpha, b=0.0):
    """Fraction(float(each original entry))*Fraction(float(coefficient))."""
    answers = []
    for row in range(K.shape[0]):
        if issparse(K):
            start, stop = K.indptr[row:row + 2]
            terms = zip(K.data[start:stop], alpha[K.indices[start:stop]])
        else:
            terms = zip(K[row], alpha)
        value = Fraction(float(b))
        for item, coefficient in terms:
            value += Fraction(float(item)) * Fraction(float(coefficient))
        answers.append(value)
    return answers


def _snapshot(K):
    """Include bytes, storage identity, writable state and cached CSR flags."""
    if issparse(K):
        # Populate the two public caches before the call being checked.
        bool(K.has_sorted_indices)
        bool(K.has_canonical_format)
        arrays = (K.data, K.indices, K.indptr)
        flags = (K.has_sorted_indices, K.has_canonical_format,
                 K.__dict__.get("_has_sorted_indices"),
                 K.__dict__.get("_has_canonical_format"))
    else:
        arrays, flags = (K,), None
    return (K.shape, tuple((str(a.dtype), a.shape, a.strides,
                           a.__array_interface__["data"][0], a.flags.writeable,
                           a.tobytes(order="A")) for a in arrays), flags)


def _read_only(K):
    for array in ((K.data, K.indices, K.indptr) if issparse(K) else (K,)):
        array.flags.writeable = False
    return K


def _duplicate_case(dtype, kind, *, rows=257, sorted_indices=False):
    # Three repeated entries at column 2, separated by explicitly stored zeros.
    # Binary64 arithmetic can lose the unit term, while a converted/coalesced
    # signed block can erase the magnitude that should force refinement.
    power = 60 if np.dtype(dtype) != np.dtype(np.longdouble) else 70
    if np.dtype(dtype) == np.dtype(np.longdouble) and np.finfo(np.longdouble).nmant <= 52:
        power = 60
    A = 2 ** power
    data, indices, indptr = [], [], [0]
    for i in range(rows):
        sign = 1 if i % 3 != 1 else -1
        unit = 0 if i % 3 == 2 else sign
        pairs = [(2, sign * A), (0, 0), (2, unit), (1, 0), (2, -sign * A), (0, 0)]
        if sorted_indices:
            pairs.sort(key=lambda item: item[0])  # stable, duplicates remain.
        indices.extend(column for column, _ in pairs)
        data.extend(value for _, value in pairs)
        indptr.append(len(data))
    K = kind((np.asarray(data, dtype=dtype), np.asarray(indices, dtype=np.int32),
              np.asarray(indptr, dtype=np.int32)), shape=(rows, 3))
    return K, np.ones(3), -0.5


def _ordinary_case(dtype, kind, rows=131):
    dense = np.asarray([[i % 5 - 2, (i * 3) % 7 - 3, 1] for i in range(rows)], dtype=dtype)
    return kind(dense), np.array([0.5, -0.25, 2.0]), 0.125


def _invoke(route, K, alpha, b):
    from dwd._kernel_scores import adaptive_kernel_matvec, compensated_kernel_matvec
    from dwd._affine_scores import affine_scores
    if route == "kernel_adaptive":
        return adaptive_kernel_matvec(K, alpha)
    if route == "kernel_compensated":
        return compensated_kernel_matvec(K, alpha)
    return affine_scores(K, alpha, b, policy=route.removeprefix("affine_"))


def _checked_call(route, K, alpha, b, *, exact=False):
    before, alpha_before = _snapshot(K), _snapshot(alpha)
    expected = _exact_binary64(K, alpha, b if route.startswith("affine_") else 0.0)
    try:
        values = np.asarray(_invoke(route, K, alpha, b))
    finally:
        assert _snapshot(K) == before, "Caller kernel buffers/flags changed"
        assert _snapshot(alpha) == alpha_before, "Caller coefficient buffer changed"
    assert values.shape == (K.shape[0],)
    assert values.dtype == np.dtype(np.float64)
    assert np.isfinite(values).all()
    rounded = np.asarray([float(value) for value in expected])
    if exact or route.endswith("compensated"):
        np.testing.assert_array_equal(values, rounded)
    else:
        for observed, wanted in zip(values, expected):
            error = abs(Fraction(float(observed)) - wanted)
            limit = Fraction(float(5e-7 * max(1.0, abs(float(observed)))))
            assert error <= limit, (observed, wanted, error, limit)
    if route.startswith("affine_"):
        for observed, wanted in zip(values, expected):
            # A nonzero exactly representable target must have its exact sign;
            # the zero tie rule must not be turned into a positive label.
            if float(wanted) != 0.0:
                assert np.sign(observed) == (1 if wanted > 0 else -1)
            elif wanted == 0:
                assert observed == 0.0
    return values


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("sorted_indices", [False, True])
@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("readonly", [False, True])
def test_duplicate_cancellation(dtype, kind, sorted_indices, route, readonly):
    K, alpha, b = _duplicate_case(dtype, kind, sorted_indices=sorted_indices)
    if readonly:
        _read_only(K)
        _read_only(alpha)
    _checked_call(route, K, alpha, b, exact=True)


@pytest.mark.parametrize("rows", [1, 127, 128, 129, 255, 256, 257])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", _ROUTES)
def test_128_row_block_boundaries(rows, kind, route):
    K, alpha, b = _duplicate_case(np.float32, kind, rows=rows)
    _checked_call(route, K, alpha, b, exact=True)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", _ROUTES)
def test_canonical_ordinary_sparse(dtype, kind, route):
    K, alpha, b = _ordinary_case(dtype, kind)
    assert K.has_canonical_format
    _checked_call(route, K, alpha, b, exact=True)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("route", _ROUTES)
def test_stale_flags_are_not_trusted_or_changed(dtype, route):
    K, alpha, b = _duplicate_case(dtype, csr_matrix, rows=3)
    K.has_sorted_indices = True
    K.has_canonical_format = True  # deliberately stale cache, buffers unchanged.
    _checked_call(route, _read_only(K), alpha, b, exact=True)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("route", _ROUTES)
def test_dense_float64_regressions(layout, route):
    K = np.tile([2.0**60, 1.0, -2.0**60], (257, 1))
    if layout == "F":
        K = np.asfortranarray(K)
    elif layout == "strided":
        buffer = np.zeros((257, 6))
        buffer[:, ::2] = K
        K = buffer[:, ::2]
    _checked_call(route, _read_only(K), np.ones(3), -0.5, exact=True)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("batch", [None, 1, 128])
def test_public_callable_prediction_with_fixed_fitted_state(dtype, kind, route, batch):
    """Exercise public prediction, with fixed model state to isolate scoring."""
    from dwd.gen_kern_dwd import KernGDWD
    Q, alpha, b = _duplicate_case(dtype, kind, rows=129)
    before = _snapshot(Q)
    returned = []
    def kernel(left, right):
        ids = np.asarray(right[:, 0], dtype=np.int64)
        result = Q[ids].T.tocsr()
        _read_only(result)
        returned.append((result, _snapshot(result)))
        return result
    model = KernGDWD(kernel=kernel, prediction_batch_size=batch)
    # This is a deliberately constructed fitted scoring fixture, not a claim
    # that training generated these coefficients or that a fit was validated.
    model._Xfit = np.arange(3.0).reshape(3, 1)
    model.n_features_in_ = 1
    model.kernel_ = kernel
    model.kernel_kws_ = {}
    model.dual_coef_ = alpha.reshape(1, -1)
    model.intercept_ = np.array([b if route.startswith("affine_") else 0.0])
    model.classes_ = np.array(["negative", "positive"])
    model.prediction_precision_ = route.split("_")[1]
    model.affine_computation_ = "joint" if route.startswith("affine_") else "standard"
    query = np.arange(129.0).reshape(129, 1)
    query_before = _snapshot(query)
    expected = _exact_binary64(Q, alpha, float(model.intercept_[0]))
    scores = model.decision_function(query)
    np.testing.assert_array_equal(scores, [float(value) for value in expected])
    expected_labels = np.asarray(["positive" if value > 0 else "negative" for value in expected])
    np.testing.assert_array_equal(model.predict(query), expected_labels)
    assert _snapshot(Q) == before
    assert _snapshot(query) == query_before
    assert all(_snapshot(item) == saved for item, saved in returned)


@pytest.mark.parametrize("case", ["extra_fraction", "round_down_below_binary64", "round_up_subnormal"])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", _ROUTES)
def test_genuine_extended_binary64_working_representation(case, kind, route):
    info = np.finfo(np.longdouble)
    if info.nmant <= np.finfo(float).nmant or info.minexp >= np.finfo(float).minexp:
        pytest.skip("Requires genuinely extended longdouble precision and exponent range")
    if case == "extra_fraction":
        value = np.longdouble(1) + np.ldexp(np.longdouble(1), -60)
        alpha, b = np.array([1.0]), -1.0
    elif case == "round_down_below_binary64":
        value = np.ldexp(np.longdouble(1), -1075)
        alpha, b = np.array([2.0**1023]), -(2.0**-53)
    else:
        value = np.ldexp(np.longdouble(3), -1076)
        alpha, b = np.array([2.0**1023]), -7.0 * 2.0**-54
    K = kind((np.array([value], dtype=np.longdouble), [0], [0, 1]), shape=(1, 1))
    # Adaptive matvec retains its documented absolute error allowance; affine
    # additionally must preserve the exact represented-input sign. Compensated
    # results are compared to the rounded rational oracle.
    _checked_call(route, _read_only(K), alpha, b)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", _ROUTES)
def test_extended_entry_outside_binary64_range_rejects(sign, kind, route):
    if np.finfo(np.longdouble).maxexp <= np.finfo(float).maxexp:
        pytest.skip("Requires longdouble exponent range beyond binary64")
    value = sign * np.ldexp(np.longdouble(1), 1100)
    K = kind((np.array([value], dtype=np.longdouble), [0], [0, 1]), shape=(1, 1))
    before = _snapshot(K)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises((FloatingPointError, ValueError, OverflowError)):
            _invoke(route, K, np.ones(1), 0.0)
    assert _snapshot(K) == before


@pytest.mark.parametrize("case", ["smallest", "opposite", "normal_boundary"])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", _ROUTES)
def test_binary64_subnormal_products(case, kind, route):
    eta, tiny = np.finfo(float).smallest_subnormal, np.finfo(float).tiny
    row = {"smallest": [eta], "opposite": [eta, -eta], "normal_boundary": [tiny, -tiny, eta]}[case]
    K = kind((np.array(row), np.arange(len(row)), [0, len(row)]), shape=(1, len(row)))
    _checked_call(route, K, np.ones(len(row)), 0.0, exact=True)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1.0j])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", _ROUTES)
def test_nonfinite_or_complex_kernel_rejects(bad, kind, route):
    K = kind(np.array([[bad]]))
    before = _snapshot(K)
    with np.errstate(all="ignore"):
        with pytest.raises((FloatingPointError, ValueError, OverflowError)):
            _invoke(route, K, np.ones(1), 0.0)
    assert _snapshot(K) == before


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1.0j])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", _ROUTES)
def test_nonfinite_or_complex_coefficient_rejects(bad, kind, route):
    K, alpha = kind(np.ones((1, 1))), np.array([bad])
    before, alpha_before = _snapshot(K), _snapshot(alpha)
    with np.errstate(all="ignore"):
        with pytest.raises((FloatingPointError, ValueError, OverflowError)):
            _invoke(route, K, alpha, 0.0)
    assert _snapshot(K) == before
    assert _snapshot(alpha) == alpha_before


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1.0j])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", ["affine_adaptive", "affine_compensated"])
def test_nonfinite_or_complex_intercept_rejects(bad, kind, route):
    K = kind(np.ones((1, 1)))
    before = _snapshot(K)
    with pytest.raises((FloatingPointError, ValueError, OverflowError)):
        _invoke(route, K, np.ones(1), bad)
    assert _snapshot(K) == before


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", _ROUTES)
def test_unrepresentable_output_rejects(kind, route):
    K = kind(np.ones((1, 2)))
    before = _snapshot(K)
    with pytest.raises((FloatingPointError, ValueError, OverflowError)):
        _invoke(route, K, np.full(2, np.finfo(float).max), 0.0)
    assert _snapshot(K) == before
