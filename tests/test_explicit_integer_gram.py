"""No.13: supported explicit systems must not use integer/bool arithmetic.

The system oracle accumulates Python integers, independently of NumPy dot/sum.
Every observed factor construction delegates to the real factor. Extreme inputs
may legitimately refuse after construction, but the constructed system itself
must still match the independent oracle. Ordinary fixtures must actually fit.
Schur validates the complete input at float64; public hooks must see that same
validated representation. The retired legacy system-only dtype policy is not
an available public computation route.
"""
from fractions import Fraction

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError

import dwd._explicit_factor as factors
import dwd.gen_dwd as gdwd


EPS = np.finfo(np.float64).eps
REFUSAL = (ValueError, np.linalg.LinAlgError, FloatingPointError)
DTYPES = (np.bool_, np.int8, np.uint8, np.int16, np.uint16,
          np.int32, np.uint32, np.int64, np.uint64)


@pytest.fixture(autouse=True)
def numerical_policy():
    # Separate nextafter/underflow-policy findings are outside this repair.
    with np.errstate(divide='warn', over='warn', invalid='warn', under='ignore'):
        yield


def ordinary(dtype=np.int8):
    rows = np.arange(16)
    bits = np.column_stack([(rows >> bit) & 1 for bit in (0, 1, 2)])
    kind = np.dtype(dtype).kind
    X = (bits if kind == 'b' else 16 * bits if kind == 'u'
         else 16 * bits - 8).astype(dtype)
    y = np.where(bits[:, 0], 1, -1)
    return X, y


def snapshot(X):
    arrays = (X.data, X.indices, X.indptr) if sparse.issparse(X) else (X,)
    return [(value.copy(), value.dtype, value.shape, value.strides,
             value.flags.writeable) for value in arrays]


def unchanged(X, before):
    arrays = (X.data, X.indices, X.indptr) if sparse.issparse(X) else (X,)
    for value, (saved, dtype, shape, strides, writable) in zip(arrays, before):
        assert_array_equal(value, saved)
        assert value.dtype == dtype and value.shape == shape
        assert value.strides == strides and value.flags.writeable == writable


def integer_system(X, penalty=0.5):
    rows = [[1] + [int(v) for v in row] for row in np.asarray(X)]
    size = len(rows[0])
    # q=1 gives M=4; all selected penalty shifts are dyadic.
    shift = Fraction(2 * len(rows), 4) * Fraction(penalty)
    return [[Fraction(sum(row[j] * row[k] for row in rows))
             + (shift if j == k and j else 0)
             for k in range(size)] for j in range(size)]


def floating_system_exact(X, penalty=0.5):
    rows = [[Fraction(1)] + [Fraction(float(v)) for v in row]
            for row in np.asarray(X)]
    size = len(rows[0])
    shift = Fraction(2 * len(rows), 4) * Fraction(penalty)
    return [[sum((row[j] * row[k] for row in rows), Fraction())
             + (shift if j == k and j else 0)
             for k in range(size)] for j in range(size)]


def solve_fraction(matrix, rhs):
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for j in range(len(rows)):
        pivot = next(i for i in range(j, len(rows)) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        divisor = rows[j][j]
        rows[j] = [v / divisor for v in rows[j]]
        for i in range(len(rows)):
            if i != j:
                multiple = rows[i][j]
                rows[i] = [a - multiple * b for a, b in zip(rows[i], rows[j])]
    return np.array([float(row[-1]) for row in rows])


def solve(X, y, *, steps=3, mode='schur', implicit=False, penalty=0.5):
    return gdwd.solve_gen_dwd(X, y, penalty, q=1,
                             implicit_P=implicit, initialization='zero',
                             stopping='fixed', max_iter=steps)


def capture_factor(monkeypatch):
    original = factors.ExplicitFactor
    captured = []

    def observed(P):
        captured.append(np.asarray(P).copy())
        return original(P)

    monkeypatch.setattr(factors, 'ExplicitFactor', observed)
    return captured


def assert_results_close(actual, expected):
    for a, b in zip(actual, expected):
        assert_allclose(a, b, rtol=128 * EPS, atol=128 * EPS)


@pytest.mark.parametrize('dtype', DTYPES)
def test_integer_and_boolean_system_matches_exact_arithmetic(monkeypatch, dtype):
    X, y = ordinary(dtype)
    X.flags.writeable = False
    before = snapshot(X)
    captured = capture_factor(monkeypatch)
    solve(X, y, steps=0)
    assert len(captured) == 1
    assert_array_equal(captured[0], np.asarray(integer_system(X), dtype=float))
    unchanged(X, before)


@pytest.mark.parametrize('dtype', (np.int8, np.uint8, np.int16, np.uint16,
                                  np.int32, np.uint32, np.int64, np.uint64))
def test_overflow_scale_system_before_factor_outcome(monkeypatch, dtype):
    # Eight orthogonal sign/zero rows. Powers of two are exactly representable
    # in binary64 even for the wide integer cases; Gram products overflow the
    # original integer dtype. A later factor refusal is not a construction pass.
    bits = np.iinfo(dtype).bits
    scale = 1 << ((bits + 1) // 2)
    if np.dtype(dtype).kind == 'u':
        X = np.array([[0], [scale]] * 4, dtype=dtype)
    else:
        X = np.array([[-scale], [scale]] * 4, dtype=dtype)
    y = np.tile([-1, 1], 4)
    captured = capture_factor(monkeypatch)
    try:
        solve(X, y, steps=0)
    except REFUSAL:
        pass
    assert len(captured) == 1
    assert_array_equal(captured[0], np.asarray(integer_system(X), dtype=float))


@pytest.mark.parametrize('dtype,power', [(np.int64, 62), (np.uint64, 63)])
def test_column_sum_overflow_is_prevented_too(monkeypatch, dtype, power):
    X = np.array([[0], [1 << power]] * 4, dtype=dtype)
    y = np.tile([-1, 1], 4)
    before = snapshot(X)
    captured = capture_factor(monkeypatch)
    try:
        solve(X, y, steps=0)
    except REFUSAL:
        pass
    assert len(captured) == 1
    expected = np.asarray(integer_system(X), dtype=float)
    assert expected[0, 1] == float(4 * (1 << power))
    assert_array_equal(captured[0], expected)
    unchanged(X, before)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('steps', (1, 5))
def test_ordinary_public_solve_matches_float64_input(dtype, steps):
    X, y = ordinary(dtype)
    before = snapshot(X)
    actual = solve(X, y, steps=steps)
    expected = solve(X.astype(np.float64), y, steps=steps)
    assert len(actual[2]) == steps + 1
    assert_results_close(actual, expected)
    assert np.all(np.diff(actual[2]) <= 128 * EPS)
    unchanged(X, before)


@pytest.mark.parametrize('dtype', (np.bool_, np.int8, np.uint8, np.int64))
def test_first_step_satisfies_independent_fraction_equation(dtype):
    X, y = ordinary(dtype)
    n, p = X.shape
    # At the zero iterate V'(0)=-1. P delta=(n/M)gradient with M=4.
    rows = [[1] + [int(v) for v in row] for row in X]
    rhs = [-Fraction(sum(row[j] * int(label) for row, label in zip(rows, y)), 4)
           for j in range(p + 1)]
    expected_step = solve_fraction(integer_system(X), rhs)
    actual = solve(X, y, steps=1)
    assert_allclose(-np.r_[actual[1], actual[0]], expected_step,
                    rtol=128 * EPS, atol=128 * EPS)


@pytest.mark.parametrize('layout', ('C', 'F', 'strided', 'readonly', 'list'))
def test_estimator_input_layout_and_caller_ownership(layout):
    X, y = ordinary(np.int16)
    if layout == 'F':
        X = np.asfortranarray(X)
    elif layout == 'strided':
        base = np.zeros((len(X), X.shape[1] * 2), dtype=X.dtype)
        base[:, ::2] = X
        X = base[:, ::2]
    elif layout == 'readonly':
        X.flags.writeable = False
    before = snapshot(X)
    supplied = X.tolist() if layout == 'list' else X
    options = dict(implicit_P=False, initialization='zero',
                   stopping='fixed', max_iter=4, lambd=0.5)
    actual = gdwd.GenDWD(**options).fit(supplied, y)
    expected = gdwd.GenDWD(**options).fit(X.astype(float), y)
    assert_allclose(actual.coef_, expected.coef_, rtol=128 * EPS, atol=128 * EPS)
    assert_allclose(actual.intercept_, expected.intercept_, rtol=128 * EPS, atol=128 * EPS)
    assert_allclose(actual.objective_history_, expected.objective_history_,
                    rtol=128 * EPS, atol=128 * EPS)
    assert_allclose(actual.gradient_norm_, expected.gradient_norm_,
                    rtol=128 * EPS, atol=128 * EPS)
    assert_array_equal(actual.predict(X), expected.predict(X))
    unchanged(X, before)


@pytest.mark.parametrize('mode,implicit', [('schur', True), ('schur', False)])
def test_implicit_and_corrected_integer_controls(mode, implicit):
    X, y = ordinary(np.int8)
    assert_results_close(solve(X, y, mode=mode, implicit=implicit),
                         solve(X.astype(float), y, mode=mode, implicit=implicit))


@pytest.mark.parametrize('representation', ('csr', 'csc', 'coo'))
def test_sparse_integer_control_and_ownership(representation):
    dense, y = ordinary(np.int16)
    X = getattr(sparse, representation + '_matrix')(dense)
    # Snapshot format-specific buffers, including the unsorted COO coordinates.
    names = ('data', 'row', 'col') if representation == 'coo' else ('data', 'indices', 'indptr')
    saved = {name: getattr(X, name).copy() for name in names}
    result = solve(X, y)
    assert_results_close(result, solve(dense.astype(float), y))
    for name, value in saved.items():
        assert_array_equal(getattr(X, name), value)


@pytest.mark.parametrize('dtype', (np.float32, np.float64, np.longdouble))
def test_supported_floating_system_uses_float64_validation(monkeypatch, dtype):
    X = (np.arange(24).reshape(8, 3) / 37.0 + np.eye(8, 3) / 11).astype(dtype)
    y = np.tile([-1, 1], 4)
    before = snapshot(X)
    captured = capture_factor(monkeypatch)
    solve(X, y, steps=0)
    working = X.astype(np.float64)
    sums = working.sum(axis=0).reshape(-1, 1)
    expected = np.block([[np.array([[len(X)]]), sums.T],
                         [sums, working.T.dot(working) + (len(X) / 4) * np.eye(X.shape[1])]])
    assert len(captured) == 1
    assert_array_equal(captured[0], expected)
    assert captured[0].dtype == np.float64
    assert_allclose(captured[0], np.asarray(floating_system_exact(working), dtype=float),
                    rtol=8 * EPS, atol=8 * EPS)
    if dtype == np.float32:
        # Retain a case that distinguishes whole-input float64 validation
        # from the retired float32 Gram arithmetic.
        assert not np.array_equal(X.T.dot(X), working.T.dot(working))
    unchanged(X, before)


@pytest.mark.parametrize('dtype', (np.bool_, np.int8, np.uint8))
def test_custom_explicit_step_receives_supported_validated_float64_input(monkeypatch, dtype):
    X, y = ordinary(dtype)
    before = snapshot(X)
    original = gdwd.get_step_explicit
    seen = []

    def observed(features, *args, **kwargs):
        seen.append(features.dtype)
        return original(features, *args, **kwargs)

    monkeypatch.setattr(gdwd, 'get_step_explicit', observed)
    result = solve(X, y, steps=2)
    assert len(result[2]) == 3
    assert seen == [np.dtype(np.float64)] * 2
    unchanged(X, before)


@pytest.mark.parametrize('refit', (False, True))
@pytest.mark.parametrize('exception_type', (RuntimeError, KeyboardInterrupt))
def test_factor_failure_preserves_exception_and_clears_fitted_state(monkeypatch, refit, exception_type):
    X, y = ordinary(np.int8)
    model = gdwd.GenDWD(implicit_P=False, initialization='zero', max_iter=1)
    if refit:
        model.fit(X, y)
    failure = exception_type('deliberate factor failure')

    def refuse(P):
        raise failure

    monkeypatch.setattr(factors, 'ExplicitFactor', refuse)
    with pytest.raises(exception_type) as caught:
        model.fit(X, y)
    assert caught.value is failure
    assert not any(name.endswith('_') and not name.startswith('_') for name in vars(model))
    with pytest.raises(NotFittedError):
        model.predict(X)


@pytest.mark.parametrize('bad', ('complex', 'nonfinite', 'oneclass'))
def test_invalid_refit_still_cleans_learned_state(bad):
    X, y = ordinary(np.int8)
    model = gdwd.GenDWD(implicit_P=False, initialization='zero', max_iter=1).fit(X, y)
    invalid, labels = X, y
    if bad == 'complex':
        invalid = X.astype(complex) + 1j
    elif bad == 'nonfinite':
        invalid = X.astype(float)
        invalid[0, 0] = np.inf
    else:
        labels = np.ones(len(y))
    with pytest.raises(ValueError):
        model.fit(invalid, labels)
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_tiny_internal_cv_matches_explicit_float_input():
    X, y = ordinary(np.int8)
    options = dict(lambd_vals=[0.5, 1.0], q_vals=[1.0], cv=2,
                   implicit_P=False, initialization='zero',
                   stopping='fixed', max_iter=2)
    actual = gdwd.GenDWDCV(**options).fit(X, y)
    expected = gdwd.GenDWDCV(**options).fit(X.astype(float), y)
    assert actual.best_params_ == expected.best_params_
    assert actual.best_score_ == expected.best_score_
    assert_allclose(actual.coef_, expected.coef_, rtol=128 * EPS, atol=128 * EPS)
    assert_allclose(actual.intercept_, expected.intercept_, rtol=128 * EPS, atol=128 * EPS)
    assert_array_equal(actual.predict(X), expected.predict(X))
