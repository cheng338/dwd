"""Selected split-interface arithmetic, range, layout and atomic refusal checks."""
from fractions import Fraction
import math
import sys
import numpy as np
import pytest
from _arithmetic_mode_test_support import changed_arithmetic
from _feature_expansion_test_support import (
    arithmetic, facade, feature, expansion_bridge, genuine, literal, layout, prepare, reference,
    native_bundle, checked, LITERALS, LAYOUTS,
)

@pytest.mark.parametrize('arrangement', LAYOUTS)
@pytest.mark.parametrize('name', LITERALS)
def test_literal_expansion_and_complete_input_bytes(expansion_bridge, arrangement, name):
    columns, z, alpha, penalty = literal(name)
    prepared, splits = prepare(z, alpha, penalty)
    columns = layout(columns, arrangement)
    status, result = checked(expansion_bridge, columns, prepared, splits)
    if name == 'subnormal-final':
        module, context = expansion_bridge
        assert status == (module.GUARD if context is None else module.OK)
        assert result[0] == math.ulp(0.)
    if name in ('zeros', 'signed-zero'):
        assert result.tobytes() == np.zeros(len(columns)).tobytes()


@pytest.mark.parametrize('penalty', (.25, -.25, 2.**-100), ids=('positive-quarter', 'negative-quarter', 'tiny-positive'))
@pytest.mark.parametrize('exponent', (-1022, -900, -500, -1, 0, 400, 900, 1023))
def test_fixed_scale_corners(expansion_bridge, penalty, exponent):
    columns = np.ldexp(np.array([[.5, -.5, .75, -.75], [1., -1., .5, -.5]]), exponent)
    z = np.array([.5, .5, -.25, -.25])
    alpha = np.array([.25, -.25, .125, -.125])
    prepared, splits = prepare(z, alpha, penalty)
    checked(expansion_bridge, columns, prepared, splits)


@pytest.mark.parametrize('over', (False, True))
@pytest.mark.parametrize('rows', (1, 2, 16, 32, 33, 63, 64))
def test_exact_tile_caps(expansion_bridge, over, rows):
    module, context = expansion_bridge
    width = min(1024, 16384//rows) + int(over)
    columns = np.ones((rows, width))
    prepared, splits = prepare(np.full(width, .5), np.zeros(width), .25)
    bundle, arrays, output = native_bundle(context, columns, prepared, splits)
    before = output.tobytes()
    status = module.gradient_from_splits(bundle)
    if over or rows == 1:
        assert status == module.ARGUMENT and output.tobytes() == before
    elif context is None:
        assert status == module.GUARD and output.tobytes() == before
    else:
        assert status == 0 and output.tobytes() == np.full(rows, width*.5).tobytes()


@pytest.mark.parametrize('fault', ('float32', 'big-endian', 'unaligned', 'F', 'negative',
                                  'strided', 'output-readonly', 'output-length', 'alias',
                                  'exponent64', 'bad-exponent', 'penalty-type', 'view-type', 'bundle-type'))
def test_malformed_bridge_is_atomic(expansion_bridge, fault):
    module, context = expansion_bridge
    columns, z, alpha, penalty = literal('dyadic')
    prepared, splits = prepare(z, alpha, penalty)
    bundle, arrays, output = native_bundle(context, columns, prepared, splits)
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
    status = module.gradient_from_splits(bundle)
    if fault == 'bad-exponent':
        assert status != 0
    else:
        assert status == module.ARGUMENT
    assert tuple(v.tobytes() for v in arrays) == before


@pytest.mark.parametrize('case', ('late-nonfinite', 'late-overflow', 'first-scale-erasure'))
def test_staging_and_first_rounding(genuine, case):
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
    status, output = checked(genuine, columns, prepared, splits)
    if case == 'first-scale-erasure':
        assert status == 0 and output.tobytes() == np.zeros(2).tobytes()
    else:
        assert status != 0


@pytest.mark.parametrize('shift', (-8, 0, 8))
def test_independent_exact_dyadic_fraction(expansion_bridge, shift):
    columns, z, alpha, _ = literal('dyadic')
    columns = np.ldexp(columns, shift)
    penalty = .25
    prepared, splits = prepare(z, alpha, penalty)
    status, result = checked(expansion_bridge, columns, prepared, splits)
    expected = np.array([float(sum((Fraction(float(x))*
                         (Fraction(float(a))+Fraction(penalty)*Fraction(float(b)))
                         for x, a, b in zip(row, z, alpha)), Fraction())) for row in columns])
    module, context = expansion_bridge
    assert status == (module.GUARD if context is None else module.OK)
    assert result.tobytes() == expected.tobytes()




@pytest.mark.parametrize('mode', ('down', 'flush'))
def test_native_mode_declines_without_writes(genuine, mode):
    module, context = genuine
    columns, z, alpha, penalty = literal('dyadic')
    prepared, splits = prepare(z, alpha, penalty)
    bundle, arrays, output = native_bundle(context, columns, prepared, splits)
    before = output.tobytes()
    with changed_arithmetic(mode) as unchanged:
        status = module.gradient_from_splits(bundle)
        preserved = unchanged()
    assert status == module.ARITHMETIC_MODE and preserved
    assert output.tobytes() == before


def test_real_extension_contract(expansion_bridge):
    module, _ = expansion_bridge
    assert module.API_VERSION == 2 and module.COMPILED_PY_HEX == sys.hexversion
    assert (module.MAX_ROWS, module.MAX_ENTRIES) == (64, 16384)
    assert module.MAX_WIDTH == 1024
    assert module.__name__ == 'dwd._feature_expansion_accel'
    assert module.gradient_from_splits.__self__ is module
    assert not hasattr(module, 'gradient_from_prepared')
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



def test_native_long_exponents_refuse_atomically_and_preserve_literal_fallback(expansion_bridge):
    module, context = expansion_bridge
    columns, prepared, splits = erasure_fixture('long')
    assert memoryview(prepared[1]).format == memoryview(prepared[4]).format == 'l'
    assert prepared[1].itemsize in (4, 8)
    bundle, arrays, output = native_bundle(context, columns, prepared, splits)
    before = tuple(value.tobytes() for value in arrays)
    status = module.gradient_from_splits(bundle)
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
