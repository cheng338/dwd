"""KernelScaler's copy option controls ownership, not transactional recovery.

Default copying protects caller storage even on arithmetic failure. An in-place
failure has no rollback promise; these tests deliberately require neither an
unchanged input nor a particular mutation after that failure.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dwd.kernel_utils import KernelScaler


DTYPES = (np.float16, np.float32, np.float64)


def _array_state(value):
    return (value.dtype.str, value.shape, value.strides,
            value.flags.writeable, value.__array_interface__["data"][0],
            value.tobytes(order="A"))


def _fitted_state(scaler):
    return tuple((name, (id(value), _array_state(value)) if isinstance(value, np.ndarray)
                  else value) for name, value in sorted(scaler.__dict__.items()))


def _ordinary_kernel(dtype):
    # Positive definite, with exact power-of-two fitted factors (2, 4).
    return np.asarray([[0.5, 0.125], [0.125, 0.125]], dtype=dtype)


def _owned_input(matrix, layout):
    if layout == "c":
        owner = np.array(matrix, order="C", copy=True)
        value = owner
    else:
        assert layout == "strided"
        owner = np.full((4, 4), 7, dtype=matrix.dtype)
        value = owner[::2, ::2]
        value[:] = matrix
    return value, owner


def _overflow_fixture(dtype, stage):
    maximum = dtype(np.finfo(dtype).max)
    if stage == "first":
        factors = np.asarray([4, 2], dtype=dtype)
        diagonal = np.asarray([0.125, 0.5], dtype=dtype)
        large = np.ldexp(maximum, -1)
    else:
        assert stage == "second"
        factors = np.asarray([2, 2], dtype=dtype)
        diagonal = np.asarray([0.5, 0.5], dtype=dtype)
        large = dtype(np.ldexp(maximum, -2) * dtype(1.5))
    # This is a valid diagonal PSD transform matrix with a different diagonal
    # from the training matrix that supplies the fitted scales.
    matrix = np.diag(np.asarray([large, 0.5], dtype=dtype))
    assert np.isfinite(matrix).all()
    first = matrix.copy()
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        first *= factors[None, :]
        second = first.copy()
        second *= factors[:, None]
    assert bool(np.isfinite(first).all()) == (stage == "second")
    assert not np.isfinite(second).all()
    return KernelScaler().fit(np.diag(diagonal)), matrix


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("stage", ["first", "second"])
@pytest.mark.parametrize("layout", ["c", "strided"])
def test_default_copy_preserves_input_and_parent_on_overflow(dtype, stage, layout):
    scaler, matrix = _overflow_fixture(dtype, stage)
    value, owner = _owned_input(matrix, layout)
    before, value_before = _array_state(owner), _array_state(value)
    learned = _fitted_state(scaler)
    # Omission exercises the public default, not only explicit copy=True.
    with pytest.raises(FloatingPointError, match="nonfinite transformed values"):
        scaler.transform(value)
    assert _array_state(owner) == before
    assert _array_state(value) == value_before
    assert _fitted_state(scaler) == learned


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("stage", ["first", "second"])
def test_inplace_overflow_refuses_and_preserves_learned_state(dtype, stage):
    scaler, value = _overflow_fixture(dtype, stage)
    learned = _fitted_state(scaler)
    with pytest.raises(FloatingPointError, match="nonfinite transformed values"):
        scaler.transform(value, copy=False)
    assert _fitted_state(scaler) == learned
    # No assertion about value after failure: rollback is not guaranteed,
    # and the contract does not require the implementation to corrupt it.


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("outcome", ["success", "overflow"])
def test_default_copy_preserves_readonly_view_and_parent(dtype, outcome):
    if outcome == "success":
        matrix = _ordinary_kernel(dtype)
        scaler = KernelScaler().fit(matrix)
    else:
        scaler, matrix = _overflow_fixture(dtype, "second")
    value, owner = _owned_input(matrix, "strided")
    value.flags.writeable = False
    before, value_before = _array_state(owner), _array_state(value)
    learned = _fitted_state(scaler)
    if outcome == "success":
        result = scaler.transform(value)
        assert not np.shares_memory(result, owner)
        assert result.dtype == np.dtype(dtype)
        assert_array_equal(result, np.asarray([[2, 1], [1, 2]], dtype=dtype))
    else:
        with pytest.raises(FloatingPointError, match="nonfinite transformed values"):
            scaler.transform(value)
    assert _array_state(owner) == before
    assert _array_state(value) == value_before
    assert _fitted_state(scaler) == learned


@pytest.mark.parametrize("dtype", DTYPES)
def test_successful_inplace_nonoverlapping_view_updates_only_its_parent_region(dtype):
    matrix = _ordinary_kernel(dtype)
    scaler = KernelScaler().fit(matrix)
    learned = _fitted_state(scaler)
    value, owner = _owned_input(matrix, "strided")
    expected = np.asarray([[2, 1], [1, 2]], dtype=dtype)
    expected_owner = owner.copy()
    expected_owner[::2, ::2] = expected
    result = scaler.transform(value, copy=False)
    assert result is value and np.shares_memory(result, owner)
    assert result.dtype == np.dtype(dtype)
    assert_array_equal(result, expected)
    assert_array_equal(owner, expected_owner)
    assert _fitted_state(scaler) == learned


def test_copy_false_integer_conversion_may_allocate():
    value = np.asarray([[2, 2], [2, 8]], dtype=np.int64)
    scaler = KernelScaler().fit(value)
    before, learned = _array_state(value), _fitted_state(scaler)
    result = scaler.transform(value, copy=False)
    assert result.dtype == np.dtype(np.float64)
    assert not np.shares_memory(result, value)
    assert_array_equal(result, [[2, 1], [1, 2]])
    assert _array_state(value) == before
    assert _fitted_state(scaler) == learned
