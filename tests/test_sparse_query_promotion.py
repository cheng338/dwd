"""Sparse query promotion before any lossy duplicate aggregation.

The source snapshot is selected by the validation runner. No production
normalizer is used to construct the represented-coordinate oracle. These
tests require only the named-kernel working-dtype policy, not exact summation
of arbitrary floating-point coordinates or a new callable/precomputed policy.
"""
from fractions import Fraction
import math

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import sparse

from dwd.gen_kern_dwd import KernGDWD
from dwd.kern_md import KernMD


LAYOUTS = ("csr_matrix", "csr_array", "coo_matrix", "coo_array",
           "csc_matrix", "csc_array")
KERNELS = {
    "rbf": {"gamma": 1.},
    "linear": {},
    "poly": {"gamma": .25, "degree": 2, "coef0": .5},
    "sigmoid": {"gamma": .25, "coef0": .1},
    "cosine": {},
}
TRAIN = np.array([[0., 0.], [.5, 1.], [1., 0.], [2., 1.]])
LABELS = np.array([0, 0, 1, 1])


@pytest.fixture(autouse=True)
def _numpy_policy():
    with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
        yield


def _buffers(value):
    return ((value.data, value.row, value.col) if value.format == "coo"
            else (value.data, value.indices, value.indptr))


def _snapshot(value):
    metadata = {key: item for key, item in value.__dict__.items()
                if key in ("has_canonical_format", "_has_canonical_format",
                           "_has_sorted_indices")}
    return (value.shape, tuple((str(a.dtype), a.shape, a.strides,
                               a.flags.writeable, a.__array_interface__["data"][0],
                               a.tobytes(order="A")) for a in _buffers(value)),
            metadata)


def _entries(layout, entries, shape, dtype=np.float32, *, readonly=True,
             stale=False):
    rows = np.array([row for row, _, _ in entries], dtype=np.int32)
    cols = np.array([col for _, col, _ in entries], dtype=np.int32)
    data = np.array([value for _, _, value in entries], dtype=dtype)
    # Accumulate the original represented entries, never raw.toarray().
    exact = [[Fraction() for _ in range(shape[1])] for _ in range(shape[0])]
    for row, col, value in zip(rows, cols, data):
        exact[int(row)][int(col)] += Fraction(float(value))
    dense = np.array([[float(value) for value in row] for row in exact])
    cls = getattr(sparse, layout)
    if layout.startswith("coo"):
        raw = cls((data, (rows, cols)), shape=shape)
    else:
        major, minor, size = ((rows, cols, shape[0]) if layout.startswith("csr")
                              else (cols, rows, shape[1]))
        order = np.argsort(major, kind="stable")
        indptr = np.asarray(np.r_[0, np.cumsum(np.bincount(major, minlength=size))],
                            dtype=np.int32)
        raw = cls((data[order], minor[order], indptr), shape=shape)
    if stale:
        raw.has_canonical_format = True
        if raw.format != "coo":
            raw.has_sorted_indices = True
    if readonly:
        for buffer in _buffers(raw):
            buffer.flags.writeable = False
    return raw, dense


def _cancellation_query(layout, *, stale=False):
    return _entries(layout, [
        (0, 0, 2.**24), (0, 0, 1.), (0, 0, -2.**24),
        (1, 1, .5), (1, 0, 2.**24), (1, 0, -1.), (1, 0, -2.**24),
        (2, 1, 0.),  # an explicit zero, followed by a genuinely empty row
        (4, 0, .5), (4, 1, 1.),
    ], (5, 2), stale=stale)


def _literal_kernel(left, right, kernel, kws):
    result = np.empty((len(left), len(right)))
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            dot = math.fsum(float(x) * float(y) for x, y in zip(a, b))
            if kernel == "linear":
                value = dot
            elif kernel == "poly":
                value = (kws["gamma"] * dot + kws["coef0"]) ** kws["degree"]
            elif kernel == "sigmoid":
                value = math.tanh(kws["gamma"] * dot + kws["coef0"])
            elif kernel == "rbf":
                value = math.exp(-kws["gamma"] * math.fsum(
                    (float(x) - float(y))**2 for x, y in zip(a, b)))
            elif kernel == "laplacian":
                value = math.exp(-kws["gamma"] * math.fsum(
                    abs(float(x) - float(y)) for x, y in zip(a, b)))
            else:
                an = math.sqrt(math.fsum(float(x)**2 for x in a))
                bn = math.sqrt(math.fsum(float(x)**2 for x in b))
                value = dot / (an * bn) if an and bn else 0.
            result[i, j] = value
    return result


def _assert_public_predictions(model, raw, expected, *, atol=2e-12):
    before = _snapshot(raw)
    coefficients, intercept = model.dual_coef_.copy(), model.intercept_.copy()
    for batch in (None, 1, 3, raw.shape[0] + 1):
        model.prediction_batch_size = batch
        assert_allclose(model.decision_function(raw), expected, rtol=2e-12, atol=atol)
        assert_array_equal(model.predict(raw), model.classes_[(expected > 0).astype(int)])
    # Explicit override must use the same promotion boundary as the setting.
    assert_allclose(model.decision_function(raw, prediction_batch_size=2),
                    expected, rtol=2e-12, atol=atol)
    assert_array_equal(model.dual_coef_, coefficients)
    assert_array_equal(model.intercept_, intercept)
    assert _snapshot(raw) == before


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("kernel", tuple(KERNELS))
def test_float64_kernmd_duplicate_queries_match_represented_coordinates(layout, kernel):
    raw, dense = _cancellation_query(layout)
    assert_array_equal(dense[:, 0], [1., -1., 0., 0., .5])
    model = KernMD(kernel=kernel, kernel_kws=KERNELS[kernel]).fit(TRAIN.copy(), LABELS)
    gram = _literal_kernel(TRAIN, dense, kernel, KERNELS[kernel])
    expected = (gram.T @ model.dual_coef_.T + model.intercept_).ravel()
    assert_allclose(model.decision_function(dense), expected, rtol=2e-12, atol=2e-12)
    _assert_public_predictions(model, raw, expected)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_original_one_feature_rbf_label_flip_is_prevented(layout):
    model = KernMD(kernel="rbf", kernel_kws={"gamma": 1.}).fit(
        np.array([[0.], [.5], [1.], [2.]]), LABELS)
    raw, dense = _entries(layout, [(0, 0, 2.**24), (0, 0, 1.), (0, 0, -2.**24)], (1, 1))
    expected = model.decision_function(dense)
    assert expected[0] > 0
    assert model.decision_function(np.array([[0.]]))[0] < 0
    _assert_public_predictions(model, raw, expected)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("kernel", ["rbf", "linear"])
def test_stale_flags_and_readonly_storage_do_not_change_query_meaning(layout, kernel):
    raw, dense = _cancellation_query(layout, stale=True)
    model = KernMD(kernel=kernel, kernel_kws=KERNELS[kernel]).fit(TRAIN.copy(), LABELS)
    _assert_public_predictions(model, raw, model.decision_function(dense))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_mixed_float64_laplacian_query_preserves_original_narrow_storage(layout):
    # Promotion owns the sparse copy. This does not assert a broader fix for
    # the separate original-float64 Laplacian caller-mutation finding.
    kws = {"gamma": .5}
    model = KernMD(kernel="laplacian", kernel_kws=kws).fit(TRAIN.copy(), LABELS)
    raw, dense = _cancellation_query(layout)
    gram = _literal_kernel(TRAIN, dense, "laplacian", kws)
    expected = (gram.T @ model.dual_coef_.T + model.intercept_).ravel()
    assert_allclose(model.decision_function(dense), expected, rtol=2e-12, atol=2e-12)
    _assert_public_predictions(model, raw, expected)


@pytest.mark.parametrize("layout", ["csr_matrix", "coo_array", "csc_matrix"])
@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_narrow_intermediate_overflow_with_finite_float64_coordinate(layout, kernel):
    high = float(np.finfo(np.float32).max)
    raw, dense = _entries(layout, [
        (0, 0, high), (0, 0, high), (0, 0, -high),
        (1, 0, -high), (1, 0, -high), (1, 0, high),
        (2, 0, 1.),
    ], (3, 2))
    assert np.isfinite(raw.data).all()
    assert_array_equal(dense[:, 0], [high, -high, 1.])
    model = KernMD(kernel=kernel, kernel_kws=KERNELS[kernel]).fit(TRAIN.copy(), LABELS)
    gram = _literal_kernel(TRAIN, dense, kernel, KERNELS[kernel])
    expected = (gram.T @ model.dual_coef_.T + model.intercept_).ravel()
    assert np.isfinite(expected).all()
    _assert_public_predictions(model, raw, expected)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("backend", ["spectral", "auto"])
def test_actual_kerngdwd_fits_share_correct_named_query_boundary(layout, backend):
    # Full-rank RBF Gram avoids the separate historical singular-solver issue.
    options = dict(kernel="rbf", kernel_kws={"gamma": 1.}, backend=backend,
                   lambd=.2, max_iter=2, stopping="fixed", initialization="zero",
                   random_state=7, acceleration=None, affine_computation="standard",
                   residual_check_order="refinement_first")
    raw, dense = _cancellation_query(layout)
    model = KernGDWD(**options).fit(TRAIN.copy(), LABELS)
    expected = model.decision_function(dense)
    _assert_public_predictions(model, raw, expected)


@pytest.mark.parametrize("basis_dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("layout", ["csr_matrix", "coo_array"])
@pytest.mark.parametrize("kernel", ["rbf", "linear", "poly", "sigmoid", "cosine"])
def test_normal_query_working_dtype_and_outputs_are_preserved(basis_dtype, layout, kernel):
    model = KernMD(kernel=kernel, kernel_kws=KERNELS[kernel]).fit(
        np.eye(4, dtype=basis_dtype), LABELS)
    query = np.array([[.25, .5, 0., 0.], [1., 0., -.25, .5], [0., 0., 0., 0.]],
                     dtype=np.float32)
    raw = getattr(sparse, layout)(query)
    before = _snapshot(raw)
    observed = []
    original = model._compute_kernel
    def observe(value):
        result = original(value)
        observed.append((value.dtype, result.dtype))
        return result
    model._compute_kernel = observe
    expected = model.decision_function(query)
    observed.clear()
    actual = model.decision_function(raw)
    assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    wanted = np.dtype(np.float32 if basis_dtype == np.float32 else np.float64)
    assert observed and all(result_dtype == wanted for _, result_dtype in observed)
    if basis_dtype == np.float32:
        assert all(input_dtype == np.dtype(np.float32) for input_dtype, _ in observed)
    assert _snapshot(raw) == before


@pytest.mark.parametrize("kernel", tuple(KERNELS))
def test_dense_float32_queries_keep_their_existing_input_representation(kernel):
    model = KernMD(kernel=kernel, kernel_kws=KERNELS[kernel]).fit(TRAIN.copy(), LABELS)
    query = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    expected = model.decision_function(query)
    seen = []
    original = model._compute_kernel
    def observe(value):
        seen.append((value.dtype, value is query))
        return original(value)
    model._compute_kernel = observe
    assert_array_equal(model.decision_function(query), expected)
    assert seen == [(np.dtype(np.float32), True)]


@pytest.mark.parametrize("kind", ["function", "str_subclass"])
def test_callable_kernel_keeps_explicit_float32_sparse_storage_contract(kind):
    seen = []
    def callback(left, right):
        if sparse.issparse(right):
            seen.append((right.dtype, right.nnz, right.format))
            # The callback explicitly chooses its own working arithmetic.
            right = right.astype(np.float64).toarray()
        return np.asarray(left) @ np.asarray(right).T
    class StringCallback(str):
        def __call__(self, left, right):
            return callback(left, right)
    kernel = callback if kind == "function" else StringCallback("linear")
    model = KernMD(kernel=kernel).fit(TRAIN.copy(), LABELS)
    raw, dense = _cancellation_query("csr_matrix")
    before = _snapshot(raw)
    expected = model.decision_function(dense)
    assert_allclose(model.decision_function(raw), expected, rtol=0, atol=0)
    assert seen == [(np.dtype(np.float32), raw.nnz, "csr")]
    assert _snapshot(raw) == before


@pytest.mark.parametrize("layout", ["csr_matrix", "coo_array"])
def test_precomputed_kernel_keeps_existing_matrix_dtype_contract(layout):
    model = KernMD(kernel="precomputed").fit(np.eye(4), LABELS)
    dense = np.array([[.25, .5, 0., 1.], [1., 0., -.5, .25]], dtype=np.float32)
    raw = getattr(sparse, layout)(dense)
    before = _snapshot(raw)
    seen = []
    original = model._compute_kernel
    def observe(value):
        seen.append(value.dtype)
        return original(value)
    model._compute_kernel = observe
    assert_array_equal(model.decision_function(raw), model.decision_function(dense))
    assert seen == [np.dtype(np.float32), np.dtype(np.float32)]
    assert _snapshot(raw) == before


@pytest.mark.parametrize("layout", ["csr_matrix", "coo_array"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1.j])
def test_invalid_sparse_queries_still_refuse_without_buffer_changes(layout, bad):
    dtype = np.complex128 if isinstance(bad, complex) else np.float32
    # Build directly: the coordinate oracle is intentionally only finite-real.
    raw = getattr(sparse, layout)(np.array([[bad, 0.]], dtype=dtype))
    if raw.format != "coo":
        bool(raw.has_canonical_format)
        bool(raw.has_sorted_indices)
    before = _snapshot(raw)
    model = KernMD(kernel="rbf").fit(TRAIN.copy(), LABELS)
    with pytest.raises((ValueError, TypeError, FloatingPointError)):
        model.decision_function(raw)
    assert _snapshot(raw) == before


def test_sparse_training_remains_explicitly_unsupported_for_kernmd():
    with pytest.raises((TypeError, ValueError)):
        KernMD(kernel="rbf").fit(sparse.csr_matrix(TRAIN), LABELS)


def test_callable_string_cosine_preserves_existing_special_case_promotion():
    # Compatibility oddity: the old public boundary special-cases equality
    # with "cosine" before the kernel dispatcher selects a callable. Preserve
    # that existing behavior here; do not generalize it to other callables.
    observed = []
    class StringCallback(str):
        def __call__(self, left, right):
            if sparse.issparse(right):
                observed.append(right.dtype)
                right = right.toarray()
            return np.asarray(left) @ np.asarray(right).T
    model = KernMD(kernel=StringCallback("cosine")).fit(TRAIN.copy(), LABELS)
    raw, dense = _cancellation_query("csr_matrix")
    before = _snapshot(raw)
    expected = model.decision_function(dense)
    assert_allclose(model.decision_function(raw), expected, rtol=0, atol=0)
    assert observed == [np.dtype(np.float64)]
    assert _snapshot(raw) == before
