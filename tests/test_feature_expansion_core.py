"""Selected split-interface arithmetic, range, layout and atomic refusal checks."""
from contextlib import contextmanager
import ctypes
from fractions import Fraction
import math
import platform
import sys
import numpy as np
import pytest
from _feature_expansion_test_support import (
    arithmetic, facade, feature, genuine, literal, layout, prepare, reference,
    native_bundle, checked, LITERALS, LAYOUTS,
)

INTERFACES = ('splits',)

@pytest.mark.parametrize('interface', INTERFACES)
@pytest.mark.parametrize('arrangement', LAYOUTS)
@pytest.mark.parametrize('name', LITERALS)
def test_literal_expansion_and_complete_input_bytes(genuine, interface, arrangement, name):
    columns, z, alpha, penalty = literal(name)
    prepared, splits = prepare(z, alpha, penalty)
    columns = layout(columns, arrangement)
    status, result = checked(genuine, columns, prepared, splits, interface)
    if name == 'subnormal-final':
        assert status == 0 and result[0] == math.ulp(0.)
    if name in ('zeros', 'signed-zero'):
        assert result.tobytes() == np.zeros(len(columns)).tobytes()


@pytest.mark.parametrize('interface', INTERFACES)
@pytest.mark.parametrize('penalty', (.25, -.25, 2.**-100), ids=('positive-quarter', 'negative-quarter', 'tiny-positive'))
@pytest.mark.parametrize('exponent', (-1022, -900, -500, -1, 0, 400, 900, 1023))
def test_fixed_scale_corners(genuine, interface, penalty, exponent):
    columns = np.ldexp(np.array([[.5, -.5, .75, -.75], [1., -1., .5, -.5]]), exponent)
    z = np.array([.5, .5, -.25, -.25])
    alpha = np.array([.25, -.25, .125, -.125])
    prepared, splits = prepare(z, alpha, penalty)
    checked(genuine, columns, prepared, splits, interface)


@pytest.mark.parametrize('interface', INTERFACES)
@pytest.mark.parametrize('over', (False, True))
@pytest.mark.parametrize('rows', (1, 2, 16, 32, 33, 63, 64))
def test_exact_tile_caps(genuine, interface, over, rows):
    module, context = genuine
    width = min(1024, 16384//rows) + int(over)
    columns = np.ones((rows, width))
    prepared, splits = prepare(np.full(width, .5), np.zeros(width), .25)
    bundle, arrays, output = native_bundle(context, columns, prepared, splits, interface)
    before = output.tobytes()
    status = getattr(module, 'gradient_from_' + interface)(bundle)
    if over or rows == 1:
        assert status == module.ARGUMENT and output.tobytes() == before
    else:
        assert status == 0 and output.tobytes() == np.full(rows, width*.5).tobytes()


@pytest.mark.parametrize('interface', INTERFACES)
@pytest.mark.parametrize('fault', ('float32', 'big-endian', 'unaligned', 'F', 'negative',
                                  'strided', 'output-readonly', 'output-length', 'alias',
                                  'exponent64', 'bad-exponent', 'penalty-type', 'view-type', 'bundle-type'))
def test_malformed_bridge_is_atomic(genuine, interface, fault):
    module, context = genuine
    columns, z, alpha, penalty = literal('dyadic')
    prepared, splits = prepare(z, alpha, penalty)
    bundle, arrays, output = native_bundle(context, columns, prepared, splits, interface)
    arrays = list(arrays)
    if fault == 'float32': arrays[0] = arrays[0].astype('float32')
    elif fault == 'big-endian': arrays[0] = arrays[0].astype('>f8')
    elif fault == 'unaligned':
        value = np.ndarray(columns.shape, dtype='float64', buffer=bytearray(columns.nbytes+1), offset=1)
        value[:] = columns
        arrays[0] = value
    elif fault == 'F': arrays[0] = np.array(columns, order='F')
    elif fault == 'negative': arrays[0] = arrays[0][:, ::-1]
    elif fault == 'strided': arrays[0] = layout(columns, 'strided')
    elif fault == 'output-readonly': output.flags.writeable = False
    elif fault == 'output-length': arrays[-1] = np.full(len(columns)+1, -912.25)
    elif fault == 'alias': arrays[-1] = arrays[0].ravel()[:len(columns)]
    elif fault == 'exponent64': arrays[-3] = arrays[-3].astype('int64')
    elif fault == 'bad-exponent': arrays[-3] = np.full(len(z), 2000000000, dtype=np.intc)
    buffers = tuple(memoryview(v) for v in arrays)
    if fault == 'view-type': buffers = (arrays[0], *buffers[1:])
    bundle = (context, buffers, float(prepared[-1]) if fault == 'penalty-type' else prepared[-1])
    if fault == 'bundle-type': bundle = list(bundle)
    before = tuple(v.tobytes() for v in arrays)
    status = getattr(module, 'gradient_from_' + interface)(bundle)
    assert status != 0
    assert tuple(v.tobytes() for v in arrays) == before


@pytest.mark.parametrize('interface', INTERFACES)
@pytest.mark.parametrize('case', ('late-nonfinite', 'late-overflow', 'first-scale-erasure'))
def test_staging_and_first_rounding(genuine, interface, case):
    module, context = genuine
    columns = np.ones((2, 4))
    prepared, splits = prepare(np.ones(4), np.zeros(4), .25)
    if case == 'late-nonfinite': columns[1, -1] = math.nan
    elif case == 'late-overflow': columns[1] = sys.float_info.max
    else:
        # Direct private prepared fixture: first ldexp(.5*.5,-1073) ties to0.
        # A later large restoration must not resurrect that erased first round.
        z = np.full(4, math.ulp(0.))
        prepared = (z, np.full(4, 1000, dtype=np.intc), np.zeros(4), np.zeros(4),
                    np.zeros(4, dtype=np.intc), 0)
        with np.errstate(all='ignore'):
            splits = tuple(arithmetic._split_operand(v) for v in (prepared[0], prepared[2], prepared[3]))
        columns[:] = .5
        # The genuine frexp exponent format is C-int on both audited hosts.
        assert memoryview(np.frexp(np.ones(4))[1]).format == 'i'
        assert memoryview(prepared[1]).format == memoryview(prepared[4]).format == 'i'
    status, output = checked(genuine, columns, prepared, splits, interface)
    if case == 'first-scale-erasure':
        assert status == 0 and output.tobytes() == np.zeros(2).tobytes()
    else:
        assert status != 0


@pytest.mark.parametrize('interface', INTERFACES)
@pytest.mark.parametrize('shift', (-8, 0, 8))
def test_independent_exact_dyadic_fraction(genuine, interface, shift):
    columns, z, alpha, _ = literal('dyadic')
    columns = np.ldexp(columns, shift)
    penalty = .25
    prepared, splits = prepare(z, alpha, penalty)
    status, result = checked(genuine, columns, prepared, splits, interface)
    expected = np.array([float(sum((Fraction(float(x))*
                         (Fraction(float(a))+Fraction(penalty)*Fraction(float(b)))
                         for x, a, b in zip(row, z, alpha)), Fraction())) for row in columns])
    assert status == 0 and result.tobytes() == expected.tobytes()


@contextmanager
def changed_arithmetic(mode):
    # Exact existing platform controls from tests/test_ordered_reduction_core.py.
    if sys.platform == 'win32':
        control = ctypes.CDLL('ucrtbase')._controlfp_s
        control.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        control.restype = ctypes.c_int
        state = ctypes.c_uint()
        assert control(ctypes.byref(state), 0, 0) == 0
        saved = state.value
        value, mask = ((0x100, 0x300) if mode == 'down' else (0x01000000, 0x03000000))
        enter = lambda: control(ctypes.byref(state), value, mask)
        leave = lambda: control(ctypes.byref(state), saved, mask)
        def unchanged():
            assert control(ctypes.byref(state), 0, 0) == 0
            return state.value & mask == value
    elif sys.platform == 'linux' and platform.machine().lower() in ('x86_64', 'amd64') and mode == 'down':
        library = ctypes.CDLL(None)
        get, set_mode = library.fegetround, library.fesetround
        get.argtypes, get.restype = [], ctypes.c_int
        set_mode.argtypes, set_mode.restype = [ctypes.c_int], ctypes.c_int
        saved = get()
        assert saved >= 0
        enter, leave = lambda: set_mode(0x400), lambda: set_mode(saved)
        unchanged = lambda: get() == 0x400
    else:
        pytest.skip('No existing audited control for this platform/mode')
    try:
        assert enter() == 0
        yield unchanged
    finally:
        assert leave() == 0


@pytest.mark.parametrize('interface', INTERFACES)
@pytest.mark.parametrize('mode', ('down', 'flush'))
def test_native_mode_declines_without_writes(genuine, interface, mode):
    module, context = genuine
    columns, z, alpha, penalty = literal('dyadic')
    prepared, splits = prepare(z, alpha, penalty)
    bundle, arrays, output = native_bundle(context, columns, prepared, splits, interface)
    before = output.tobytes()
    with changed_arithmetic(mode) as unchanged:
        status = getattr(module, 'gradient_from_' + interface)(bundle)
        preserved = unchanged()
    assert status == module.ARITHMETIC_MODE and preserved
    assert output.tobytes() == before


def test_real_extension_contract(genuine):
    module, _ = genuine
    assert module.API_VERSION == 2 and module.COMPILED_PY_HEX == sys.hexversion
    assert (module.MAX_ROWS, module.MAX_ENTRIES) == (64, 16384)
    assert module.MAX_WIDTH == 1024
    assert module.__name__ == 'dwd._feature_expansion_accel'
    assert module.gradient_from_splits.__self__ is module
    assert module.BETA_API_VERSION == 1
    assert module.beta_from_split.__self__ is module

def erasure_fixture(dtype):
    columns = np.full((2, 4), .5)
    z = np.full(4, math.ulp(0.))
    prepared = (z, np.full(4, 1000, dtype=dtype), np.zeros(4), np.zeros(4),
                np.zeros(4, dtype=dtype), 0)
    with np.errstate(all='ignore'):
        splits = tuple(arithmetic._split_operand(v)
                       for v in (prepared[0], prepared[2], prepared[3]))
    return columns, prepared, splits



@pytest.mark.parametrize('interface', INTERFACES)
def test_native_long_exponents_refuse_atomically_and_preserve_literal_fallback(genuine, interface):
    module, context = genuine
    columns, prepared, splits = erasure_fixture('long')
    assert memoryview(prepared[1]).format == memoryview(prepared[4]).format == 'l'
    assert prepared[1].itemsize in (4, 8)
    bundle, arrays, output = native_bundle(context, columns, prepared, splits, interface)
    before = tuple(value.tobytes() for value in arrays)
    status = getattr(module, 'gradient_from_' + interface)(bundle)
    assert status == module.ARGUMENT
    assert tuple(value.tobytes() for value in arrays) == before
    # This intentionally foreign prepared representation is not produced by
    # _feature_gradient. Win64 long/4 passes the facade's integer-size screen
    # and declines in C; Linux long/8 declines in the facade. Both retain the
    # exact original complete tile. Do not claim native arithmetic ran here.
    assert facade.try_gradient(feature, columns, prepared, splits) is None
    expected = reference._gradient_tile(columns, prepared)
    actual = feature._gradient_tile(columns, prepared, _splits=splits)
    assert expected.tobytes() == actual.tobytes() == np.zeros(2).tobytes()
    assert tuple(value.tobytes() for value in arrays) == before
