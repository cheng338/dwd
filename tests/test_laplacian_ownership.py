"""Real public Laplacian predictions preserve caller-owned sparse storage.

Exact dyadic fixtures isolate ownership from independent duplicate-summation
and solver-rounding questions. No historical duplicate training basis is
presented as a valid fitted model.
"""
from fractions import Fraction
import math
import pickle
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import sparse

from dwd.gen_kern_dwd import KernGDWD
from dwd.kern_md import KernMD
import dwd.kernel_utils as kernel_module
import dwd._sparse_features as sparse_features


CSR = ("csr_matrix", "csr_array")
CONVERSIONS = ("csc_matrix", "csc_array", "coo_matrix", "coo_array")
TRAIN = np.array([[0., .25, -.5], [.5, 1., .25], [1., -.25, 0.],
                  [2., 1., .5], [-1., .5, 1.], [.25, -1., 2.],
                  [1.5, .25, -1.], [-.5, 2., .75]])
LABELS = np.array([-1, -1, 1, 1, -1, 1, 1, -1])
QUERY = np.array([[1., -.25, .5], [-.5, .75, 0.], [0., 0., 0.],
                  [0., 0., 0.], [.25, 1., -.5]])
GAMMA = .375


@pytest.fixture(autouse=True)
def restore_numpy_policy():
    # Strict-underflow solver scoring is the separate F4 issue.
    with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
        yield


def buffers(value):
    return ((value.data, value.row, value.col) if value.format == "coo"
            else (value.data, value.indices, value.indptr))


def snapshot(value):
    # Reading cached-flag properties could itself mutate the object under test.
    metadata = tuple(sorted((name, item) for name, item in vars(value).items()
                            if name in ("has_canonical_format", "_has_canonical_format",
                                        "_has_sorted_indices")))
    return (value.shape, value.format, metadata,
            tuple((id(a), str(a.dtype), a.shape, a.strides, a.flags.writeable,
                   a.__array_interface__["data"][0], a.tobytes(order="A"))
                  for a in buffers(value)))


def represented_csr_values(value):
    """Read the original storage directly without canonicalization or flags."""
    result = []
    for start, stop in zip(value.indptr[:-1], value.indptr[1:]):
        row = [Fraction() for _ in range(value.shape[1])]
        for column, item in zip(value.indices[start:stop], value.data[start:stop]):
            row[int(column)] += Fraction(float(item))
        result.append([float(item) for item in row])
    return np.array(result)


def from_entries(layout, entries, shape, *, dtype=np.float64, readonly=False, stale=False):
    rows = np.array([row for row, _, _ in entries], dtype=np.int32)
    cols = np.array([col for _, col, _ in entries], dtype=np.int32)
    values = np.array([value for _, _, value in entries], dtype=dtype)
    represented = [[Fraction() for _ in range(shape[1])] for _ in range(shape[0])]
    for row, col, value in zip(rows, cols, values):
        represented[int(row)][int(col)] += Fraction(float(value))
    dense = np.array([[float(value) for value in row] for row in represented])
    constructor = getattr(sparse, layout)
    if layout.startswith("coo"):
        raw = constructor((values, (rows, cols)), shape=shape)
    else:
        major, minor, size = ((rows, cols, shape[0]) if layout.startswith("csr")
                              else (cols, rows, shape[1]))
        order = np.argsort(major, kind="stable")
        indptr = np.asarray(np.r_[0, np.cumsum(np.bincount(major, minlength=size))],
                            dtype=np.int32)
        raw = constructor((values[order], minor[order], indptr), shape=shape)
    if stale:
        raw.has_canonical_format = True
        if raw.format != "coo":
            raw.has_sorted_indices = True
    if readonly:
        for array in buffers(raw):
            array.flags.writeable = False
    return raw, dense


def represented_sparse(dense, layout="csr_matrix", kind="unsorted", **options):
    entries = []
    for row, values in enumerate(dense):
        columns = list(np.flatnonzero(values))
        if kind == "unsorted":
            columns.reverse()
        for column in columns:
            value = float(values[column])
            if kind == "duplicates":
                entries.extend([(row, column, value / 2), (row, column, value / 2)])
            else:
                entries.append((row, column, value))
    # Include one explicit zero and leave the next row genuinely empty.
    if dense is QUERY:
        entries.append((2, dense.shape[1] - 1, 0.))
    return from_entries(layout, entries, dense.shape, **options)


def fitted(route="spectral", *, dtype=np.float64, **options):
    if route == "md":
        return KernMD(kernel="laplacian", kernel_kws={"gamma": GAMMA}).fit(
            TRAIN.astype(dtype), LABELS)
    settings = dict(kernel="laplacian", kernel_kws={"gamma": GAMMA},
                    backend=route, lambd=.2,
                    max_iter=3, stopping="fixed", initialization="zero",
                    acceleration=None, random_state=7, affine_computation="standard",
                    residual_check_order="refinement_first")
    settings.update(options)
    return KernGDWD(**settings).fit(TRAIN.astype(dtype), LABELS)


def literal_laplacian(left, right, gamma=GAMMA):
    return np.array([[math.exp(-gamma * math.fsum(abs(float(a) - float(b))
                                                for a, b in zip(x, y)))
                      for y in right] for x in left])


def expected_scores(model, dense, *, basis=TRAIN):
    return (literal_laplacian(basis, dense).T @ model.dual_coef_.T
            + model.intercept_).ravel()


def exercise_prediction(model, raw, dense, *, batch=None):
    before = snapshot(raw)
    coefficients, intercept = model.dual_coef_.copy(), model.intercept_.copy()
    expected = expected_scores(model, dense)
    reference = model.decision_function(dense, prediction_batch_size=batch)
    assert_allclose(reference, expected, rtol=2e-11, atol=2e-12)
    actual = model.decision_function(raw, prediction_batch_size=batch)
    assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
    assert snapshot(raw) == before
    # predict uses the estimator setting, so set it explicitly and restore.
    previous = getattr(model, "prediction_batch_size", None)
    model.prediction_batch_size = batch
    try:
        assert_array_equal(model.predict(raw), model.classes_[(actual > 0).astype(int)])
    finally:
        model.prediction_batch_size = previous
    assert snapshot(raw) == before
    assert_array_equal(model.dual_coef_, coefficients)
    assert_array_equal(model.intercept_, intercept)


@pytest.mark.parametrize("layout", CSR)
@pytest.mark.parametrize("kind", ("unsorted", "duplicates"))
@pytest.mark.parametrize("readonly", (False, True))
def test_original_float64_csr_public_prediction_ownership(layout, kind, readonly):
    raw, dense = represented_sparse(QUERY, layout, kind, readonly=readonly)
    exercise_prediction(fitted(), raw, dense)


@pytest.mark.parametrize("route", ("spectral", "cholesky", "md"))
@pytest.mark.parametrize("batch", (None, 1, 2, 5, 8))
def test_actual_predictors_and_batch_boundaries(route, batch):
    raw, dense = represented_sparse(QUERY)
    exercise_prediction(fitted(route), raw, dense, batch=batch)


@pytest.mark.parametrize("layout", CONVERSIONS)
@pytest.mark.parametrize("readonly", (False, True))
def test_sparse_format_conversion_preserves_original_storage(layout, readonly):
    raw, dense = represented_sparse(QUERY, layout, "duplicates", readonly=readonly)
    exercise_prediction(fitted("md"), raw, dense)


@pytest.mark.parametrize("layout", CSR)
@pytest.mark.parametrize("kind", ("unsorted", "duplicates"))
def test_stale_flags_are_not_trusted_and_remain_unchanged(layout, kind):
    raw, dense = represented_sparse(QUERY, layout, kind, stale=True)
    exercise_prediction(fitted("md"), raw, dense)


@pytest.mark.parametrize("layout", CSR)
def test_shared_index_sibling_does_not_change_meaning(layout):
    raw, dense = represented_sparse(QUERY, layout)
    sibling = getattr(sparse, layout)((raw.data.copy() * 2 + .125,
                                      raw.indices, raw.indptr), shape=raw.shape, copy=False)
    assert np.shares_memory(raw.indices, sibling.indices)
    assert np.shares_memory(raw.indptr, sibling.indptr)
    before = snapshot(sibling)
    represented_before = represented_csr_values(sibling)
    exercise_prediction(fitted("md"), raw, dense)
    assert snapshot(sibling) == before
    assert_array_equal(represented_csr_values(sibling), represented_before)


@pytest.mark.parametrize("layout", CSR)
def test_matching_float32_operands_do_not_hide_mutation_behind_promotion(layout):
    raw, dense = represented_sparse(QUERY, layout, dtype=np.float32)
    exercise_prediction(fitted("md", dtype=np.float32), raw, dense)


@pytest.mark.parametrize("layout", CSR)
def test_canonical_readonly_storage_needs_no_inplace_work(layout):
    raw, dense = represented_sparse(QUERY, layout, "canonical", readonly=True)
    exercise_prediction(fitted("md"), raw, dense)


def test_exact_cancellation_duplicates_keep_represented_coordinate():
    raw, dense = from_entries("csr_matrix", [
        (0, 0, 2.**40), (0, 0, .5), (0, 0, -2.**40),
        (1, 1, -2.**40), (1, 1, -.25), (1, 1, 2.**40),
    ], (2, 3))
    assert_array_equal(dense, [[.5, 0., 0.], [0., -.25, 0.]])
    exercise_prediction(fitted("md"), raw, dense)


def test_duplicate_overflow_keeps_upstream_arithmetic_on_owned_storage():
    # Ownership is the requested change. Do not silently replace upstream
    # aggregation with a new finiteness policy or claim exact extreme sums.
    high = np.finfo(float).max
    raw = sparse.csr_matrix(([high, high], [0, 0], [0, 2]), shape=(1, 3))
    before = snapshot(raw)
    model = fitted("md")
    with warnings.catch_warnings(record=True) as reference_warnings:
        warnings.simplefilter("always")
        reference = kernel_module.pairwise_kernels(
            X=model._Xfit, Y=raw.copy(), metric="laplacian", gamma=GAMMA)
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual = model._compute_kernel(raw)
    assert_array_equal(actual, reference)
    assert [(w.category, str(w.message)) for w in actual_warnings] == [
        (w.category, str(w.message)) for w in reference_warnings]
    assert snapshot(raw) == before


@pytest.mark.parametrize("route", ("spectral", "cholesky"))
def test_training_and_cv_preparation_keep_owned_sparse_basis(route):
    raw, dense = represented_sparse(TRAIN, kind="duplicates", readonly=True)
    before = snapshot(raw)
    model = fitted(route)
    settings = model.get_params()
    cached = KernGDWD(**settings).cv_init(raw)
    assert snapshot(raw) == before
    actual = cached.fit(raw, LABELS)
    fresh = KernGDWD(**settings).fit(dense, LABELS)
    assert snapshot(raw) == before
    assert_allclose(actual.decision_function(QUERY), fresh.decision_function(QUERY),
                    rtol=2e-10, atol=2e-11)
    assert not np.shares_memory(actual._Xfit.data, raw.data)


@pytest.mark.parametrize("readonly", (False, True))
def test_real_validation_data_obeys_same_storage_boundary(readonly):
    raw, dense = represented_sparse(QUERY, kind="duplicates", readonly=readonly)
    labels = np.array([-1, 1, -1, 1, 1])
    before = snapshot(raw)
    options = dict(kernel="laplacian", kernel_kws={"gamma": GAMMA}, lambd=.2,
                   max_iter=3, stopping="validation", check_interval=1, patience=2,
                   initialization="zero", backend="spectral", acceleration=None,
                   random_state=7, affine_computation="standard")
    reference = KernGDWD(**options).fit(TRAIN.copy(), LABELS,
                                      validation_data=(dense, labels))
    actual = KernGDWD(**options).fit(TRAIN.copy(), LABELS, validation_data=(raw, labels))
    assert snapshot(raw) == before
    assert actual.returned_iteration_ == reference.returned_iteration_
    assert actual.n_iter_ == reference.n_iter_
    assert_allclose(actual.decision_function(dense), reference.decision_function(dense),
                    rtol=2e-10, atol=2e-11)
    assert actual.validation_history_ == reference.validation_history_


def test_old_missing_fitted_specification_with_valid_unique_sparse_basis():
    model = fitted()
    raw_basis, dense_basis = represented_sparse(TRAIN)
    assert_array_equal(dense_basis, TRAIN)
    # Equivalent unique coordinates preserve the actual fitted Gram. This is
    # an explicit compatibility fixture, not a historical duplicate-basis claim.
    model._Xfit = raw_basis
    del model.kernel_
    del model.kernel_kws_
    model = pickle.loads(pickle.dumps(model))
    before = snapshot(model._Xfit)
    actual = model.decision_function(QUERY)
    assert_allclose(actual, expected_scores(model, QUERY), rtol=2e-11, atol=2e-12)
    assert snapshot(model._Xfit) == before


@pytest.mark.parametrize("kind", ("canonical", "unsorted"))
@pytest.mark.parametrize("self_kernel", (False, True))
def test_both_operands_and_self_identity_reach_real_pairwise_kernel(monkeypatch, kind, self_kernel):
    model = fitted()
    basis, _ = represented_sparse(TRAIN, kind=kind)
    query, dense = represented_sparse(QUERY, kind=kind)
    model._Xfit = basis
    before_basis, before_query = snapshot(basis), snapshot(query)
    seen = []
    original = kernel_module.pairwise_kernels
    def observe(*args, **kwargs):
        seen.append((kwargs["X"], kwargs["Y"]))
        return original(*args, **kwargs)
    monkeypatch.setattr(kernel_module, "pairwise_kernels", observe)
    actual = model._compute_kernel(basis if self_kernel else query)
    assert_allclose(actual, literal_laplacian(TRAIN, TRAIN if self_kernel else dense),
                    rtol=2e-14, atol=2e-15)
    assert len(seen) == 1
    left, right = seen[0]
    if self_kernel:
        assert left is right
    copy_expected = kind != "canonical"
    assert np.shares_memory(left.data, basis.data) is (not copy_expected)
    if not self_kernel:
        assert np.shares_memory(right.data, query.data) is (not copy_expected)
    assert snapshot(basis) == before_basis
    if not self_kernel:
        assert snapshot(query) == before_query


@pytest.mark.parametrize("kind", ("canonical", "unsorted", "duplicates"))
@pytest.mark.parametrize("layout", CSR)
def test_laplacian_preparation_changes_ownership_not_arithmetic(layout, kind):
    raw, _ = represented_sparse(QUERY, layout, kind, stale=True)
    before = snapshot(raw)
    prepared = sparse_features.prepare_laplacian_features(raw)
    assert snapshot(raw) == before
    copy_expected = kind != "canonical"
    for original, result in zip(buffers(raw), buffers(prepared)):
        assert_array_equal(result, original)
        assert np.shares_memory(result, original) is (not copy_expected)
    dense = QUERY.copy()
    assert sparse_features.prepare_laplacian_features(dense) is dense
    for conversion in (sparse.csc_matrix(QUERY), sparse.coo_array(QUERY)):
        assert sparse_features.prepare_laplacian_features(conversion) is conversion


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf, 1j))
def test_invalid_queries_refuse_without_mutating_storage(bad):
    dtype = complex if isinstance(bad, complex) else float
    raw = sparse.csr_matrix(np.array([[bad, 0., 0.]], dtype=dtype))
    before = snapshot(raw)
    if isinstance(bad, complex):
        with pytest.raises(ValueError, match="Complex data not supported"):
            fitted("md").decision_function(raw)
        after = snapshot(raw)
        # sklearn formats the sparse object in its complex-input refusal;
        # SciPy may populate cached flags while formatting. These refused
        # inputs never reach Laplacian. Shape, format and every buffer property
        # remain protected; do not claim metadata immutability for this case.
        assert (after[0], after[1], after[3]) == (before[0], before[1], before[3])
        return
    with pytest.raises((ValueError, TypeError)):
        fitted("md").decision_function(raw)
    assert snapshot(raw) == before


@pytest.mark.parametrize("batch", (0, -1, True, 1.5))
def test_invalid_batch_controls_refuse_before_kernel_mutation(batch):
    raw, _ = represented_sparse(QUERY)
    before = snapshot(raw)
    with pytest.raises(ValueError, match="prediction_batch_size"):
        fitted().decision_function(raw, prediction_batch_size=batch)
    assert snapshot(raw) == before


def test_exception_after_real_kernel_construction_preserves_caller(monkeypatch):
    model = fitted()
    raw, _ = represented_sparse(QUERY)
    before = snapshot(raw)
    original = kernel_module.pairwise_kernels
    def failing(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("IR1 after kernel sentinel")
    monkeypatch.setattr(kernel_module, "pairwise_kernels", failing)
    with pytest.raises(RuntimeError, match="IR1 after kernel sentinel"):
        model.decision_function(raw)
    assert snapshot(raw) == before


@pytest.mark.parametrize("kernel,kws", (("linear", {}), ("poly", {"gamma": .25, "degree": 2}),
                                       ("sigmoid", {"gamma": .25}), ("rbf", {"gamma": .375}),
                                       ("cosine", {})))
def test_other_named_kernels_keep_predictions_and_query_storage(kernel, kws):
    model = KernMD(kernel=kernel, kernel_kws=kws).fit(TRAIN.copy(), LABELS)
    raw, dense = represented_sparse(QUERY, kind="duplicates", readonly=True)
    before = snapshot(raw)
    assert_allclose(model.decision_function(raw), model.decision_function(dense),
                    rtol=2e-11, atol=2e-12)
    assert snapshot(raw) == before


def test_callable_and_precomputed_boundaries_are_not_canonicalized():
    seen = []
    def callback(left, right):
        if sparse.issparse(right):
            seen.append(right)
            right = right.toarray()
        return np.asarray(left) @ np.asarray(right).T
    model = KernMD(kernel=callback).fit(TRAIN.copy(), LABELS)
    raw, dense = represented_sparse(QUERY, kind="duplicates")
    before = snapshot(raw)
    assert_allclose(model.decision_function(raw), model.decision_function(dense),
                    rtol=2e-14, atol=2e-15)
    assert len(seen) == 1 and seen[0] is raw and snapshot(raw) == before
    precomputed = KernMD(kernel="precomputed").fit(np.eye(4), [-1, -1, 1, 1])
    pre, dense_pre = represented_sparse(np.array([[.5, 1., 0., .25], [1., -.25, .5, 0.]]),
                                        kind="duplicates")
    before_pre = snapshot(pre)
    assert_allclose(precomputed.decision_function(pre), precomputed.decision_function(dense_pre),
                    rtol=0, atol=0)
    assert snapshot(pre) == before_pre


def test_kernel_md_sparse_training_is_refused():
    with pytest.raises((ValueError, TypeError)):
        KernMD(kernel="laplacian").fit(sparse.csr_matrix(TRAIN), LABELS)
