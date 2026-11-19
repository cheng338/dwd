"""Sparse integer coordinates are summed before one floating conversion."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse


FORMATS = ("coo", "csr", "csc", "bsr", "lil", "dok", "dia")
DUPLICATE_FORMATS = FORMATS[:5]
KINDS = ("matrix", "array")


@pytest.mark.parametrize("fmt", ["coo", "csr", "csc", "bsr", "dok", "dia"])
@pytest.mark.parametrize("kind", KINDS)
def test_verified_unique_linear_query_avoids_unnecessary_float_copy(fmt, kind):
    from dwd._integer_sparse import prepare_integer_linear_query
    dense = np.diag(np.asarray([2**53 + 1, 7, -3, 0], dtype=np.int64))
    value = getattr(sparse, fmt + "_" + kind)(dense)
    _readonly(value)
    before = _state(value)
    assert prepare_integer_linear_query(value) is value
    assert _state(value) == before
INTEGER_CASES = (
    ("signed_cancellation", np.int64, (2**53 + 1, -(2**53))),
    ("signed_overflow", np.int64, (2**63 - 1, 2**63 - 1)),
    ("unsigned_overflow", np.uint64, (2**64 - 1, 1)),
)


@pytest.fixture(scope="module")
def admission():
    from dwd import _sparse_features, linear_model
    return SimpleNamespace(promote_sparse_features=_sparse_features.promote_sparse_features,
                           linear_model=linear_model)


def _array_state(value):
    return (value.dtype.str, value.shape, value.strides,
            value.flags.writeable, value.__array_interface__["data"][0],
            value.tobytes(order="A"))


def _scalar_state(value):
    converted = np.asarray(value)
    return (converted.dtype.str, converted.tobytes())


def _numeric_buffers(value):
    if value.format == "coo":
        return (value.data, *value.coords)
    if value.format in ("csr", "csc", "bsr"):
        return (value.data, value.indices, value.indptr)
    if value.format == "dia":
        return (value.data, value.offsets)
    return ()


def _state(value):
    flags = tuple(sorted((key, item) for key, item in value.__dict__.items()
                         if key in ("has_canonical_format", "_has_canonical_format",
                                    "_has_sorted_indices")))
    if value.format == "lil":
        payload = (_array_state(value.rows), _array_state(value.data),
                   tuple((id(row), tuple(row)) for row in value.rows),
                   tuple((id(row), tuple(_scalar_state(item) for item in row))
                         for row in value.data))
    elif value.format == "dok":
        payload = tuple(sorted((key, _scalar_state(item)) for key, item in value.items()))
    else:
        payload = tuple(_array_state(item) for item in _numeric_buffers(value))
    return (value.shape, value.dtype.str, flags, payload)


def _readonly(value):
    for item in _numeric_buffers(value):
        item.flags.writeable = False
    # SciPy's LIL conversion obtains a writable object-array memoryview even
    # though it does not alter the row lists. Supported LIL fixtures retain
    # writable outer arrays; _state still checks their bytes and every list.
    # The old readonly refusal is checked separately below.


def _width(value, dtype):
    if value.format == "coo":
        value.coords = tuple(item.astype(dtype) for item in value.coords)
    elif value.format in ("csr", "csc", "bsr"):
        value.indices = value.indices.astype(dtype)
        value.indptr = value.indptr.astype(dtype)
    elif value.format == "dia":
        value.offsets = value.offsets.astype(dtype)


def _stored_integer_literals(value):
    """Read storage only: never convert format or sum in the source dtype."""
    literals = []
    if value.format == "coo":
        literals = [(int(r), int(c), int(v))
                    for r, c, v in zip(*value.coords, value.data, strict=True)]
    elif value.format in ("csr", "csc"):
        for major in range(len(value.indptr) - 1):
            for offset in range(int(value.indptr[major]), int(value.indptr[major + 1])):
                minor = int(value.indices[offset])
                row, col = (major, minor) if value.format == "csr" else (minor, major)
                literals.append((row, col, int(value.data[offset])))
    elif value.format == "bsr":
        nr, nc = value.blocksize
        for major in range(len(value.indptr) - 1):
            for offset in range(int(value.indptr[major]), int(value.indptr[major + 1])):
                minor = int(value.indices[offset])
                for r in range(nr):
                    for c in range(nc):
                        literals.append((major * nr + r, minor * nc + c,
                                         int(value.data[offset, r, c])))
    elif value.format == "lil":
        for row, (columns, values) in enumerate(zip(value.rows, value.data, strict=True)):
            literals.extend((row, int(col), int(item))
                            for col, item in zip(columns, values, strict=True))
    elif value.format == "dok":
        literals = [(int(r), int(c), int(item)) for (r, c), item in value.items()]
    else:
        assert value.format == "dia"
        for offset, diagonal in zip(value.offsets, value.data, strict=True):
            for col, item in enumerate(diagonal):
                row = col - int(offset)
                if 0 <= row < value.shape[0] and col < value.shape[1]:
                    literals.append((row, col, int(item)))
    return literals


def _integer_oracle(literals, shape):
    totals = {}
    for row, col, value in literals:
        totals[row, col] = totals.get((row, col), 0) + value
    result = np.zeros(shape, dtype=np.float64)
    for index, total in totals.items():
        result[index] = float(total)
    return result


def _fixture(fmt, kind, dtype, prefix, *, unique=False, stale=False,
             index_dtype=None, empty=False):
    cls = getattr(sparse, fmt + "_" + kind)
    dtype = np.dtype(dtype)
    if empty:
        value = (cls((6, 6), dtype=dtype, blocksize=(2, 2)) if fmt == "bsr"
                 else cls((4, 4), dtype=dtype))
    elif fmt == "bsr":
        # Two-dimensional blocks are essential: grouping scalar COO entries
        # would miss blocksize, elementwise accumulation and zero-block checks.
        blocks = [np.array([[item, 1], [0, item]], dtype=dtype) for item in prefix]
        columns = list(range(len(prefix))) if unique else [1] * len(prefix)
        if unique:
            columns.reverse()  # Valid unsorted unique block coordinates.
        blocks += [np.array([[2, 0], [0, 3]], dtype=dtype), np.zeros((2, 2), dtype=dtype)]
        indices = np.asarray(columns + [1, 0], dtype=np.int32)
        indptr = np.asarray([0, len(prefix), len(prefix) + 1, len(prefix) + 2], dtype=np.int32)
        value = cls((np.asarray(blocks, dtype=dtype), indices, indptr), shape=(6, 6))
    elif fmt == "dia":
        # Valid unique offsets. First entry on offset+1 is padding, not data.
        data = np.asarray([[prefix[0], 0, 3, 0], [7, prefix[-1], 0, 2]], dtype=dtype)
        value = cls((data, np.asarray([0, 1], dtype=np.int32)), shape=(4, 4))
    else:
        entries = [(0, 3, 2)]
        entries += [(0, j if unique else 1, item) for j, item in enumerate(prefix)]
        entries += [(1, 0, 3), (3, 2, 0)]
        rows = np.asarray([item[0] for item in entries], dtype=np.int32)
        cols = np.asarray([item[1] for item in entries], dtype=np.int32)
        data = np.asarray([item[2] for item in entries], dtype=dtype)
        if fmt == "coo":
            value = cls((data, (rows, cols)), shape=(4, 4))
        elif fmt in ("csr", "csc"):
            major, minor = (rows, cols) if fmt == "csr" else (cols, rows)
            order = np.argsort(major, kind="stable")
            indptr = np.asarray(np.r_[0, np.cumsum(np.bincount(major, minlength=4))], dtype=np.int32)
            value = cls((data[order], minor[order], indptr), shape=(4, 4))
        elif fmt == "lil":
            value = cls((4, 4), dtype=dtype)
            for row, col, item in zip(rows, cols, data, strict=True):
                value.rows[int(row)].append(int(col))
                value.data[int(row)].append(item)
        else:
            assert fmt == "dok" and unique
            value = cls((4, 4), dtype=dtype)
            for row, col, item in zip(rows, cols, data, strict=True):
                value[int(row), int(col)] = item
    if index_dtype is not None:
        _width(value, index_dtype)
    if stale and fmt in ("coo", "csr", "csc", "bsr"):
        value.has_canonical_format = True
        if fmt != "coo":
            value.has_sorted_indices = True
    _readonly(value)
    return value


def _assert_exact(admission, value):
    before = _state(value)
    expected = _integer_oracle(_stored_integer_literals(value), value.shape)
    result = admission.promote_sparse_features(value)
    assert type(result) is type(value)
    assert result.dtype == np.dtype(np.float64) and result.format == value.format
    np.testing.assert_array_equal(result.toarray(), expected)
    assert _state(value) == before
    if value.format == "bsr":
        assert result.blocksize == value.blocksize == (2, 2)
        if value.nnz:
            assert np.any(np.all(result.data == 0, axis=(1, 2)))
    if value.format in ("coo", "csr", "csc", "bsr", "dia"):
        # SciPy can downcast structural indices in new matrix containers.
        # Exact numeric coordinates and unchanged caller storage are required;
        # an additional same-width output promise is deliberately not made.
        for output in _numeric_buffers(result)[1:]:
            assert output.dtype.kind == "i" and output.dtype.itemsize in (4, 8)
    for original in _numeric_buffers(value):
        for output in _numeric_buffers(result):
            assert not np.shares_memory(original, output)
    return result, expected


@pytest.mark.parametrize("fmt", DUPLICATE_FORMATS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case,dtype,prefix", INTEGER_CASES, ids=[case[0] for case in INTEGER_CASES])
def test_duplicate_integer_coordinates_are_exact(admission, fmt, kind, case, dtype, prefix):
    value = _fixture(fmt, kind, dtype, prefix, stale=True)
    _, expected = _assert_exact(admission, value)
    location = (0, 2) if fmt == "bsr" else (0, 1)
    assert expected[location] == float(sum(int(item) for item in prefix))


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype,large", [(np.int64, 2**53 + 1), (np.uint64, 2**64 - 1)])
def test_unique_large_integer_has_only_final_scalar_rounding(admission, fmt, kind, dtype, large):
    value = _fixture(fmt, kind, dtype, (large,), unique=True)
    _assert_exact(admission, value)


@pytest.mark.parametrize("fmt", DUPLICATE_FORMATS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype,prefix", [(np.int8, (100, 100)), (np.uint8, (250, 10))])
def test_narrow_integer_overflow_protection_remains(admission, fmt, kind, dtype, prefix):
    value = _fixture(fmt, kind, dtype, prefix, stale=True)
    _assert_exact(admission, value)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("kind", KINDS)
def test_empty_integer_storage(admission, fmt, kind):
    _assert_exact(admission, _fixture(fmt, kind, np.int64, (), empty=True))


@pytest.mark.parametrize("fmt", DUPLICATE_FORMATS)
@pytest.mark.parametrize("kind", KINDS)
def test_unsorted_unique_integer_storage(admission, fmt, kind):
    value = _fixture(fmt, kind, np.int64, (2**53 + 1, 7), unique=True, stale=True)
    _assert_exact(admission, value)


@pytest.mark.parametrize("fmt", ["coo", "csr", "bsr"])
@pytest.mark.parametrize("kind", KINDS)
def test_non_native_endian_integer_cancellation(admission, fmt, kind, record_property):
    dtype = np.dtype(np.int64).newbyteorder("S")
    if fmt in ("coo", "bsr"):
        # These actual SciPy constructors reject the dtype before promotion.
        # Record that dependency boundary, not an exact-summation success.
        with pytest.raises(ValueError, match=r"scipy\.sparse does not support dtype") as observed:
            _fixture(fmt, kind, dtype, (2**53 + 1, -(2**53)), stale=True)
        record_property("scope", "dependency constructor refusal; candidate not called")
        record_property("dtype", dtype.str)
        record_property("refusal", str(observed.value))
        return
    value = _fixture(fmt, kind, dtype, (2**53 + 1, -(2**53)), stale=True)
    assert not value.dtype.isnative
    _assert_exact(admission, value)
    record_property("scope", "accepted CSR storage; candidate exact-coordinate check")


READONLY_LIL = (
    ("exact_integer", np.int64, (2**53 + 1, -(2**53)), False),
    ("narrow_integer", np.int8, (100, 100), False),
    ("float32_forced", np.float32, (2**24, 1, -(2**24)), False),
    ("boolean", np.bool_, (True, True), False),
    ("empty_integer", np.int64, (), True),
)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case,dtype,prefix,empty", READONLY_LIL,
                         ids=[case[0] for case in READONLY_LIL])
def test_readonly_lil_outer_arrays_retain_existing_refusal(
        admission, kind, case, dtype, prefix, empty, record_property):
    # This direct dependency operation is the prior cast-first LIL boundary.
    # It is used only to verify refusal parity, not as an integer-sum oracle.
    def previous_cast(value, *, force_float64):
        assert force_float64
        return value.astype(np.float64)
    messages = []
    for promote in (previous_cast, admission.promote_sparse_features):
        value = _fixture("lil", kind, dtype, prefix, empty=empty)
        value.rows.flags.writeable = False
        value.data.flags.writeable = False
        before = _state(value)
        with pytest.raises(ValueError, match="buffer source array is read-only") as observed:
            promote(value, force_float64=True)
        assert _state(value) == before
        messages.append(str(observed.value))
    assert messages[0] == messages[1]
    record_property("scope", "unchanged baseline/candidate dependency refusal")
    record_property("refusal", messages[0])


@pytest.mark.parametrize("fmt", ["coo", "csr", "csc", "bsr"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_exact_groups_accept_both_structural_index_widths(admission, fmt, kind, index_dtype):
    value = _fixture(fmt, kind, np.int64, (2**53 + 1, -(2**53)),
                     stale=True, index_dtype=index_dtype)
    _assert_exact(admission, value)


NONINTEGER = (
    ("float32_native", np.float32, False, (2**24, 1, -(2**24))),
    ("float32_forced", np.float32, True, (2**24, 1, -(2**24))),
    ("float64", np.float64, False, (0.5, 0.25)),
    ("boolean", np.bool_, False, (True, True)),
    ("complex", np.complex128, True, (1 + 2j, 3 - 1j)),
)


@pytest.mark.parametrize("fmt", ["coo", "csr", "lil"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case,dtype,force,prefix", NONINTEGER, ids=[case[0] for case in NONINTEGER])
def test_noninteger_routes_match_existing_policy(admission, fmt, kind, case, dtype, force, prefix):
    original = _fixture(fmt, kind, dtype, prefix)
    before = _state(original)
    actual = admission.promote_sparse_features(original, force_float64=force)
    assert type(actual) is type(original)
    if case in ("float32_native", "float64", "complex"):
        assert actual is original
        assert actual.dtype == np.dtype(dtype)
    else:
        assert actual is not original and actual.dtype == np.dtype(np.float64)
        expected = np.zeros((4, 4), dtype=np.float64)
        if case == "float32_forced":
            # Exactly represented binary64 sum 2**24 + 1 - 2**24 is one.
            expected[0, 1], expected[0, 3], expected[1, 0] = 1., 2., 3.
        else:
            assert case == "boolean"
            # The two stored True entries count as two, not logical OR.
            expected[0, 1], expected[0, 3], expected[1, 0] = 2., 1., 1.
        np.testing.assert_array_equal(actual.toarray(), expected)
    assert _state(original) == before


@pytest.mark.parametrize("case", ["signed", "unsigned", "float", "boolean", "complex", "dense"])
def test_linear_query_hook_changes_only_sparse_integer_representation(admission, monkeypatch, case):
    from sklearn.base import BaseEstimator
    class FittedProbe(admission.linear_model.LinearClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            return self
    class AdmissionObserved(Exception):
        pass
    def admitted_input(value):
        captured = []
        def observe(array, *args, **kwargs):
            captured.append(array)
            raise AdmissionObserved
        fitted = FittedProbe()
        fitted.coef_ = np.zeros((1, value.shape[1]))
        fitted.intercept_ = np.zeros(1)
        fitted.classes_ = np.array([-1, 1])
        with monkeypatch.context() as patch:
            patch.setattr(admission.linear_model, "check_array", observe)
            with pytest.raises(AdmissionObserved):
                fitted.decision_function(value)
        assert len(captured) == 1
        return captured[0]
    if case == "dense":
        value = np.array([[2**53 + 1]], dtype=np.int64)
        value.flags.writeable = False
        assert admitted_input(value) is value
        return
    dtype, prefix = {
        "signed": (np.int64, (2**53 + 1, -(2**53))),
        "unsigned": (np.uint64, (2**64 - 1, 1)),
        "float": (np.float64, (0.5, 0.25)),
        "boolean": (np.bool_, (True, True)),
        "complex": (np.complex128, (1j, 2j)),
    }[case]
    value = _fixture("csr", "array", dtype, prefix, stale=True)
    before = _state(value)
    actual = admitted_input(value)
    if case in ("signed", "unsigned"):
        expected = _integer_oracle(_stored_integer_literals(value), value.shape)
        np.testing.assert_array_equal(actual.toarray(), expected)
        assert actual is not value
    else:
        assert actual is value
    assert _state(value) == before


def test_wide_sparse_input_does_not_require_dense_feature_storage(admission, monkeypatch):
    value = sparse.coo_array((np.array([2**53 + 1, -(2**53)], dtype=np.int64),
                              (np.array([0, 0]), np.array([9_999_999, 9_999_999]))),
                             shape=(3, 10_000_000))
    _readonly(value)
    before = _state(value)
    def forbidden(*args, **kwargs):
        raise AssertionError("Feature storage was densified")
    with monkeypatch.context() as patch:
        for fmt in FORMATS:
            for kind in KINDS:
                cls = getattr(sparse, fmt + "_" + kind)
                patch.setattr(cls, "toarray", forbidden)
                patch.setattr(cls, "todense", forbidden)
        actual = admission.promote_sparse_features(value)
    assert actual.shape == value.shape and actual.nnz == 1
    np.testing.assert_array_equal(actual.data, [1.0])
    np.testing.assert_array_equal(actual.coords[0], [0])
    np.testing.assert_array_equal(actual.coords[1], [9_999_999])
    assert _state(value) == before
