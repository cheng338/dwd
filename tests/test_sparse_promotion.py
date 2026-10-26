"""Promote literal sparse entries before format conversion can sum them."""
from fractions import Fraction
import math

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import sparse

from dwd._sparse_features import promote_sparse_features
from dwd.cv import run_cv
from dwd.gen_dwd import GenDWD, GenDWDCV, _augmented_gram, solve_gen_dwd
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV


LAYOUTS = ("coo_matrix", "coo_array", "csr_matrix", "csr_array",
           "csc_matrix", "csc_array")
DTYPES = (np.int8, np.float32)
Y = np.tile([-1, 1], 4)
OPTIONS = dict(lambd=.1, max_iter=2, stopping="fixed",
               initialization="zero", random_state=7)
KERNELS = {
    "linear": {}, "rbf": {"gamma": .001},
    "poly": {"gamma": .001, "degree": 2, "coef0": 1.},
    "polynomial": {"gamma": .001, "degree": 2, "coef0": 1.},
    "cosine": {}, "laplacian": {"gamma": .001},
}


def raw_features(dtype, layout, *, readonly=True, stale=False):
    # int8 addition would wrap 200 to -56; float32 addition would lose 1.
    prefix = [100, 100] if np.dtype(dtype).kind in "iu" else [2**24, 1, -2**24]
    data = np.array(prefix + [1, 2, 3, 4, 5, 6, 7], dtype=dtype)
    # Some supported sklearn pairwise routines require int32 sparse indices.
    # Index-width compatibility is separate from feature-value promotion.
    rows = np.array([0] * len(prefix) + list(range(1, 8)), dtype=np.int32)
    cols = np.array([0] * len(prefix) + [1, 2, 0, 1, 2, 0, 1], dtype=np.int32)
    # Independent rational oracle: never convert the raw sparse object to
    # dense or use production normalization to construct expected values.
    exact = [[Fraction() for _ in range(3)] for _ in range(8)]
    for row, col, value in zip(rows, cols, data):
        exact[row][col] += Fraction(float(value))
    expected = np.array([[float(value) for value in row] for row in exact])
    cls = getattr(sparse, layout)
    if layout.startswith("coo"):
        raw = cls((data, (rows, cols)), shape=(8, 3))
    else:
        major, minor, size = (rows, cols, 8) if layout.startswith("csr") else (cols, rows, 3)
        order = np.argsort(major, kind="stable")
        indptr = np.asarray(np.r_[0, np.cumsum(np.bincount(major, minlength=size))], dtype=np.int32)
        raw = cls((data[order], minor[order], indptr), shape=(8, 3))
    if stale:
        raw.has_canonical_format = True
        if raw.format != "coo":
            raw.has_sorted_indices = True
    if readonly:
        for value in buffers(raw):
            value.flags.writeable = False
    return raw, expected


def buffers(raw):
    return ((raw.data, raw.row, raw.col) if raw.format == "coo"
            else (raw.data, raw.indices, raw.indptr))


def snapshot(raw):
    metadata = {key: value for key, value in raw.__dict__.items()
                if key in ("has_canonical_format", "_has_canonical_format", "_has_sorted_indices")}
    return [(value.copy(), value.flags.writeable) for value in buffers(raw)], metadata


def assert_unchanged(raw, before):
    for actual, (expected, writeable) in zip(buffers(raw), before[0]):
        assert_array_equal(actual, expected)
        assert actual.flags.writeable == writeable
    assert snapshot(raw)[1] == before[1]


def literal_kernel(x, y, name, kws):
    result = np.empty((len(x), len(y)))
    for i, row in enumerate(x):
        for j, other in enumerate(y):
            dot = math.fsum(float(a) * float(b) for a, b in zip(row, other))
            if name == "linear":
                value = dot
            elif name in ("poly", "polynomial"):
                value = (kws["gamma"] * dot + kws["coef0"]) ** kws["degree"]
            elif name == "cosine":
                left = math.sqrt(math.fsum(float(a)**2 for a in row))
                right = math.sqrt(math.fsum(float(b)**2 for b in other))
                value = dot / (left * right) if left and right else 0.
            elif name == "rbf":
                value = math.exp(-kws["gamma"] * math.fsum((float(a)-float(b))**2
                                                         for a, b in zip(row, other)))
            else:
                value = math.exp(-kws["gamma"] * math.fsum(abs(float(a)-float(b))
                                                         for a, b in zip(row, other)))
            result[i, j] = value
    return result


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_promotion_preserves_raw_storage_and_ignores_stale_flags(layout, dtype):
    raw, expected = raw_features(dtype, layout, stale=True)
    before = snapshot(raw)
    promoted = promote_sparse_features(raw, force_float64=True)
    assert promoted.dtype == np.float64
    assert_array_equal(promoted.toarray(), expected)
    assert not np.shares_memory(promoted.data, raw.data)
    assert_unchanged(raw, before)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_linear_fit_lowlevel_and_cv_preparation(layout, dtype):
    raw, dense = raw_features(dtype, layout)
    before = snapshot(raw)
    fitted = GenDWD(**OPTIONS).fit(raw, Y)
    control = GenDWD(**OPTIONS).fit(dense, Y)
    assert_allclose(fitted.coef_, control.coef_, rtol=1e-11, atol=1e-13)
    assert_allclose(fitted.intercept_, control.intercept_, rtol=1e-11, atol=1e-13)
    assert_allclose(control.decision_function(raw), control.decision_function(dense),
                    rtol=1e-11, atol=1e-13)
    assert_array_equal(_augmented_gram(raw), _augmented_gram(dense))
    prepared = GenDWD(**OPTIONS).cv_init(raw)
    assert_array_equal(prepared._P0_X.toarray(), dense)
    first, second = solve_gen_dwd(raw, Y, **OPTIONS), solve_gen_dwd(dense, Y, **OPTIONS)
    for actual, expected in zip(first[:3], second[:3]):
        assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)
    assert_unchanged(raw, before)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("layout", ("coo_matrix", "csr_array", "csc_matrix"))
@pytest.mark.parametrize("dtype", DTYPES)
def test_named_kernel_public_boundaries(kernel, layout, dtype):
    raw, dense = raw_features(dtype, layout)
    before = snapshot(raw)
    options = dict(OPTIONS, kernel=kernel, kernel_kws=KERNELS[kernel])
    fitted = KernGDWD(**options).fit(raw, Y)
    assert_array_equal(fitted._Xfit.toarray(), dense)
    expected = literal_kernel(dense, dense, kernel, KERNELS[kernel])
    assert_allclose(fitted._compute_kernel(fitted._Xfit), expected, rtol=2e-13, atol=2e-13)
    control = KernGDWD(**options).fit(dense, Y)
    # Fit products may round differently between dense and sparse BLAS.
    assert_allclose(fitted.decision_function(dense), control.decision_function(dense),
                    rtol=2e-8, atol=2e-10)
    for batch_size in (None, 3):
        assert_allclose(control.decision_function(raw, prediction_batch_size=batch_size),
                        control.decision_function(dense), rtol=2e-12, atol=2e-12)
    gram, labels = control._prepare_validation((raw, Y))
    assert_allclose(gram, expected.T, rtol=2e-13, atol=2e-13)
    assert_array_equal(labels, Y)
    prepared = KernGDWD(**options).cv_init(raw)
    assert_array_equal(prepared._Xfit.toarray(), dense)
    assert_unchanged(raw, before)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("factory,estimator", ((GenDWDCV, GenDWD), (KernGDWDCV, KernGDWD)))
def test_outer_cv_boundaries(dtype, factory, estimator):
    raw, dense = raw_features(dtype, "coo_array")
    before = snapshot(raw)
    options = dict(lambd_vals=[.1], q_vals=[1.], cv=2, max_iter=2,
                   stopping="fixed", initialization="zero", random_state=7)
    first, second = factory(**options).fit(raw, Y), factory(**options).fit(dense, Y)
    assert_allclose(first.decision_function(dense), second.decision_function(dense),
                    rtol=2e-8, atol=2e-10)
    first = run_cv(estimator(**OPTIONS), raw, Y, {"lambd": [.1]}, cv=2)
    second = run_cv(estimator(**OPTIONS), dense, Y, {"lambd": [.1]}, cv=2)
    assert first[0] == second[0]
    assert_allclose(first[2].decision_function(dense), second[2].decision_function(dense),
                    rtol=2e-8, atol=2e-10)
    assert_unchanged(raw, before)


def test_callable_kernel_sees_promoted_features():
    raw, dense = raw_features(np.int8, "coo_matrix")
    calls = []

    def kernel(left, right):
        left = left.toarray() if sparse.issparse(left) else np.asarray(left)
        right = right.toarray() if sparse.issparse(right) else np.asarray(right)
        calls.append((left.copy(), right.copy()))
        return left @ right.T

    fitted = KernGDWD(kernel=kernel, **OPTIONS).fit(raw, Y)
    fitted.decision_function(raw)
    assert len(calls) >= 2
    for left, right in calls:
        assert_array_equal(left, dense)
        assert_array_equal(right, dense)


def test_fast_paths_and_legacy_float32_policy():
    dense = np.arange(24., dtype=np.float64).reshape(8, 3)
    for value in (dense, sparse.csr_matrix(dense), sparse.coo_array(dense)):
        assert promote_sparse_features(value, force_float64=True) is value
    raw, _ = raw_features(np.float32, "coo_matrix")
    assert promote_sparse_features(raw) is raw
    assert GenDWD(solver_mode="legacy").cv_init(raw)._P0_X.dtype == np.float32


@pytest.mark.parametrize("estimator", (GenDWD, KernGDWD))
@pytest.mark.parametrize("bad", (complex(1, 0), complex(1, 1), np.nan, np.inf))
def test_refusals_and_failed_refit_cleanup(estimator, bad):
    raw, dense = raw_features(np.float32, "coo_matrix", readonly=False)
    if isinstance(bad, complex):
        raw = raw.astype(np.complex128)
    raw.data[0] = bad
    before = snapshot(raw)
    fitted = estimator(**OPTIONS).fit(dense, Y)
    with pytest.raises((TypeError, ValueError)):
        fitted.decision_function(raw)
    with pytest.raises((TypeError, ValueError)):
        fitted.fit(raw, Y)
    assert not hasattr(fitted, "coef_")
    assert not hasattr(fitted, "dual_coef_")
    assert_unchanged(raw, before)


@pytest.mark.parametrize("kernel", ("chi2", "additive_chi2", "not_a_kernel"))
def test_unsupported_sparse_kernel_refusals(kernel):
    raw, _ = raw_features(np.int8, "coo_matrix")
    before = snapshot(raw)
    with pytest.raises((TypeError, ValueError)):
        KernGDWD(kernel=kernel, **OPTIONS).fit(raw, Y)
    assert_unchanged(raw, before)
