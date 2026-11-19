"""Installed native ordered-sum arithmetic, buffer, and atomic-refusal tests.

These exercise the package entry directly, without a prototype adapter or
workspace paths. The exponent ladder stresses valid partials; it does not
claim to force the fixed 64-partial overflow branch for finite binary64.
"""
from array import array
from fractions import Fraction
import math
import platform
import struct
import sys
import importlib.util

import numpy as np
import pytest
from _arithmetic_mode_test_support import changed_arithmetic

# A portable wheel intentionally omits this extension. A present artifact that
# cannot import is an error, not a portable-build skip. Native CI separately
# requires an actual installed-wheel C-entry witness.
if importlib.util.find_spec('dwd._ordered_reduction_accel') is None:
    pytest.skip('Portable build has no ordered-gradient binary.',
                allow_module_level=True)
from dwd import _ordered_reduction_accel as core

pytestmark = pytest.mark.skipif(
    sys.implementation.name != 'cpython'
    or sys.platform not in ('win32', 'linux')
    or platform.machine().lower() not in ('amd64', 'x86_64')
    or sys.maxsize <= 2**32,
    reason='Native ordered-gradient qualification requires Windows/Linux x86-64 CPython.',
)

OK, ARGUMENT, MODE, NONFINITE, MEMORY, CAPACITY, OVERFLOW = range(7)


def word(value):
    return struct.pack('=d', value)


def run_exact(rows):
    packed = np.array(rows, dtype=np.float64, order='C', copy=True)
    before = packed.tobytes()
    output = np.full(len(packed), -919.25)
    assert core.reduce_rows(packed, output) == OK
    expected = np.array([math.fsum(row.tolist()) for row in packed])
    assert output.tobytes() == expected.tobytes()
    assert packed.tobytes() == before
    return output


TINY = float.fromhex('0x0.0000000000001p-1022')
MAX = sys.float_info.max
LITERAL_CASES = [
    ([0.0, -0.0, 0.0], 0.0),
    ([-0.0, -0.0, -0.0], 0.0),
    ([1.0, -1.0, -0.0], 0.0),
    ([1e16, 1.0, -1e16], 1.0),
    ([2.0**53, 1.0, -(2.0**53)], 1.0),
    ([1.0, 2.0**-53, 0.0], 1.0),
    ([1.0, 2.0**-53, 2.0**-54], math.nextafter(1.0, math.inf)),
    ([1.0, 2.0**-53, -(2.0**-54)], 1.0),
    ([math.nextafter(1.0, math.inf), 2.0**-53, 0.0],
     math.nextafter(math.nextafter(1.0, math.inf), math.inf)),
    ([-1.0, -(2.0**-53), -(2.0**-54)], math.nextafter(-1.0, -math.inf)),
    ([TINY, TINY, -TINY], TINY),
    ([sys.float_info.min, -sys.float_info.min, TINY], TINY),
    ([MAX, -MAX, TINY], TINY),
    ([2.0**1000, 2.0**-1000, -(2.0**1000)], 2.0**-1000),
]


@pytest.mark.parametrize('values,expected', LITERAL_CASES)
def test_known_literals_and_reference(values, expected):
    result = run_exact([values])
    assert word(result[0]) == word(expected)


@pytest.mark.parametrize('length', [1, 2, 6, 63, 64, 65, 127, 768, 12288])
def test_cancellation_and_binary64_rounding_against_exact_rational(length):
    # Multiplication by powers of two is exact; exact-rational conversion is
    # a second oracle, independent of the partial-sum implementation.
    values = [math.ldexp(float((i*37) % 127-63), (i*31) % 101-80)
              for i in range(length)]
    exact = float(sum((Fraction.from_float(v) for v in values), Fraction()))
    actual = run_exact([values])[0]
    assert word(actual) == word(exact)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('negate', [False, True])
def test_full_exponent_ladder_stress_without_false_capacity_claim(reverse, negate):
    values = [math.ldexp(1.0, exponent) for exponent in range(-1074, 1001, 53)]
    if reverse:
        values.reverse()
    if negate:
        values = [-value for value in values]
    run_exact([values])


@pytest.mark.parametrize('shape', [(1, 12288), (64, 192), (64, 1), (16, 768), (4, 3072)])
def test_valid_boundary_shapes(shape):
    packed = ((np.arange(math.prod(shape), dtype=np.int64) % 31)-15).astype(float)
    packed = packed.reshape(shape)/128.0
    run_exact(packed)


def test_readonly_input_and_native_memoryview_buffers():
    packed = np.array([[1e16, 1.0, -1e16], [2.0, -1.0, -0.0]])
    packed.setflags(write=False)
    output = np.empty(2)
    assert core.reduce_rows(memoryview(packed), memoryview(output)) == OK
    assert output.tobytes() == np.array([1.0, 1.0]).tobytes()


def test_standard_library_native_double_buffers():
    data = array('d', [1e16, 1.0, -1e16, 2.0, 3.0, 4.0])
    packed = memoryview(data).cast('B').cast('d', shape=[2, 3])
    output = array('d', [-7.0, -7.0])
    assert core.reduce_rows(packed, output) == OK
    assert output.tobytes() == array('d', [1.0, 9.0]).tobytes()


@pytest.mark.parametrize('value', [math.inf, -math.inf, math.nan])
@pytest.mark.parametrize('row', [0, 1, 2])
def test_nonfinite_any_row_is_atomic(value, row):
    packed = np.ones((3, 6))
    packed[row, -1] = value
    output = np.array([-0.0, 123.25, -919.25])
    before = output.tobytes()
    assert core.reduce_rows(packed, output) == NONFINITE
    assert output.tobytes() == before


@pytest.mark.parametrize('values', [[MAX, MAX, -MAX], [-MAX, -MAX, MAX], [MAX, MAX, 0.0]])
def test_intermediate_overflow_declines_without_committing_prior_rows(values):
    packed = np.array([[1.0, 2.0, 3.0], values])
    output = np.array([-0.0, 987.25])
    before = output.tobytes()
    with pytest.raises(OverflowError):
        math.fsum(values)
    assert core.reduce_rows(packed, output) == OVERFLOW
    assert output.tobytes() == before


def malformed_cases():
    opposite = '>' if sys.byteorder == 'little' else '<'
    yield np.ones((2, 3), dtype=np.float32), np.ones(2), 'float32-input'
    yield np.ones((2, 3), dtype=np.int64), np.ones(2), 'int-input'
    yield np.ones((2, 3), dtype=opposite+'f8'), np.ones(2), 'opposite-endian'
    yield np.asfortranarray(np.ones((2, 3))), np.ones(2), 'fortran'
    yield np.ones((2, 6))[:, ::2], np.ones(2), 'strided'
    yield np.ones((2, 3))[:, ::-1], np.ones(2), 'negative-stride'
    yield np.ones(6), np.ones(2), 'input-rank1'
    yield np.ones((1, 2, 3)), np.ones(2), 'input-rank3'
    yield np.ones((0, 3)), np.ones(0), 'empty-rows'
    yield np.ones((2, 0)), np.ones(2), 'empty-width'
    yield np.ones((65, 1)), np.ones(65), 'rows-over-cap'
    yield np.ones((1, 12289)), np.ones(1), 'entries-over-cap'
    yield np.ones((64, 193)), np.ones(64), 'combined-entries-over-cap'
    yield np.ones((2, 3)), np.ones(3), 'wrong-output-length'
    yield np.ones((2, 3)), np.ones((2, 1)), 'output-rank2'
    yield np.ones((2, 3)), np.ones(2, dtype=np.float32), 'output-float32'
    yield np.ones((2, 3)), np.ones(4)[::2], 'output-strided'
    yield np.ones((2, 3)), np.ones(2)[::-1], 'output-negative-stride'
    readonly = np.ones(2)
    readonly.setflags(write=False)
    yield np.ones((2, 3)), readonly, 'output-readonly'
    unaligned_in = np.ndarray((2, 3), dtype=float, buffer=bytearray(49), offset=1)
    yield unaligned_in, np.ones(2), 'input-unaligned'
    unaligned_out = np.ndarray((2,), dtype=float, buffer=bytearray(17), offset=1)
    yield np.ones((2, 3)), unaligned_out, 'output-unaligned'


@pytest.mark.parametrize('packed,output,label', list(malformed_cases()), ids=lambda x: x if isinstance(x, str) else None)
def test_malformed_buffers_refuse_without_writes(packed, output, label):
    before = output.tobytes()
    input_before = packed.tobytes()
    assert core.reduce_rows(packed, output) == ARGUMENT, label
    assert output.tobytes() == before
    assert packed.tobytes() == input_before


def test_input_output_overlap_is_refused_atomically():
    storage = np.arange(16, dtype=float)
    packed = storage[:12].reshape(2, 6)
    output = storage[4:6]
    before = storage.tobytes()
    assert core.reduce_rows(packed, output) == ARGUMENT
    assert storage.tobytes() == before


def test_adjacent_disjoint_views_allowed_with_stable_storage():
    storage = np.array([1., 2., 3., 4., 5., 6., -9., -9.])
    assert core.reduce_rows(storage[:6].reshape(2, 3), storage[6:]) == OK
    assert storage[6:].tobytes() == np.array([6., 15.]).tobytes()


@pytest.mark.parametrize('bad', [None, object(), [1.0, 2.0]])
def test_nonexporter_exception_leaves_output_unchanged(bad):
    output = np.array([-9.25])
    before = output.tobytes()
    with pytest.raises(TypeError):
        core.reduce_rows(bad, output)
    assert output.tobytes() == before


@pytest.mark.skipif(sys.version_info < (3, 12), reason='Python-level buffer hook introduced in 3.12')
def test_buffer_exporter_exception_propagates_exact_object():
    error = RuntimeError('test-exporter-refusal')
    class RefusingExporter:
        def __buffer__(self, flags):
            raise error
    output = np.array([-3.0])
    with pytest.raises(RuntimeError) as seen:
        core.reduce_rows(RefusingExporter(), output)
    assert seen.value is error
    assert output.tobytes() == np.array([-3.0]).tobytes()





@pytest.mark.parametrize('mode', ['down', 'flush'])
def test_real_arithmetic_mode_refuses_atomically_and_does_not_change_caller(mode):
    packed = np.ones((2, 3))
    output = np.array([-0.0, -912.25])
    before = output.tobytes()
    with changed_arithmetic(mode) as unchanged:
        status = core.reduce_rows(packed, output)
        preserved = unchanged()
    assert status == MODE and preserved
    assert output.tobytes() == before
    assert core.reduce_rows(packed, output) == OK


def test_declared_native_contract_constants():
    assert (core.OK, core.ARGUMENT, core.ARITHMETIC_MODE, core.NONFINITE,
            core.MEMORY, core.PARTIAL_CAPACITY, core.SUM_OVERFLOW) == tuple(range(7))
    assert core.API_VERSION == 1
    assert (core.MAX_ROWS, core.MAX_TERMS) == (64, 12288)
