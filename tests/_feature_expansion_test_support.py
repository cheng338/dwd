"""Shared fixtures for package-native feature expansion correctness tests.

No package path is inserted: dwd resolves from the selected source or installed
wheel. A missing optional artifact skips only native-required fixtures; the
installed-wheel CI route preflight independently requires native acceptance.
"""
import importlib.util
import math
import sys
import numpy as np
import pytest
from dwd import _linear_feature_diagnostics as feature
from dwd import _feature_expansion as facade
from dwd import _compensated_residual as arithmetic
import _feature_expansion_reference as reference

LITERALS = ('zeros', 'signed-zero', 'dyadic', 'three-family-cancel',
            'erased-coefficient', 'regularizer-low', 'half-even', 'subnormal-final',
            'restored-underflow', 'restored-erasure', 'overflow', 'nonfinite')
LAYOUTS = ('C', 'F', 'strided', 'reversed', 'readonly')


def prepare(z, alpha, penalty=0.2):
    pm, pe = math.frexp(penalty)
    am, ae = np.frexp(alpha)
    zm, ze = np.frexp(z)
    high, low = arithmetic._products(np.full(len(alpha), pm), am)
    prepared = (zm, ze, high, low, ae, pe)
    with np.errstate(all='ignore'):
        splits = tuple(arithmetic._split_operand(v) for v in (zm, high, low))
    return prepared, splits


def literal(name):
    columns = np.array([[1., 2., -1., .5, -2., 4., .25, -4.],
                        [2., -1., 4., -2., .5, 1., -4., .25]])
    z = np.array([.5, -.25, .75, -.5, .125, -.125, .25, -.75])
    alpha = np.array([.25, -.125, .5, -.25, .0625, -.5, .125, .75])
    penalty = .2
    if name in ('zeros', 'signed-zero'):
        columns[:] = 0.0
        z[:] = 0.0
        alpha[:] = 0.0
        if name == 'signed-zero':
            columns[0, ::2] = -0.0
            z[::2] = -0.0
            alpha[1::2] = -0.0
    elif name == 'three-family-cancel':
        z = -alpha * .25
        penalty = .25
    elif name == 'erased-coefficient':
        alpha[:] = [2.**52, -2.**52, 2.**52, -2.**52, 0, 0, 0, 0]
        z[:] = [1., 0., -1., 1., 0., 0., 0., 0.]
        columns[:] = 1.
        penalty = 2.
    elif name == 'regularizer-low':
        alpha[:] = math.nextafter(.75, 1.)
        penalty = math.nextafter(.5, 1.)
    elif name == 'half-even':
        columns[:] = 1.
        alpha[:] = 0.
        z[:] = [1., 2.**-53, 2.**-54, -2.**-54, 0, 0, 0, 0]
    elif name == 'subnormal-final':
        columns[:] = 1.
        alpha[:] = 0.
        z[:] = [math.nextafter(sys.float_info.min, math.inf), -sys.float_info.min, 0, 0, 0, 0, 0, 0]
    elif name == 'restored-underflow':
        columns[:] = .5
        alpha[:] = 0.
        z[:] = sys.float_info.min
    elif name == 'restored-erasure':
        columns[:] = 2.**-100
        alpha[:] = 0.
        z[:] = sys.float_info.min
    elif name == 'overflow':
        columns[:] = sys.float_info.max
        alpha[:] = 0.
        z[:] = 2.
    elif name == 'nonfinite':
        columns[1, -1] = math.inf
    return columns, z, alpha, penalty


def layout(columns, name):
    if name == 'F':
        return np.array(columns, order='F')
    if name == 'strided':
        storage = np.zeros((len(columns), columns.shape[1]*2))
        storage[:, ::2] = columns
        return storage[:, ::2]
    if name == 'reversed':
        return columns[:, ::-1]
    out = columns.copy(order='C')
    if name == 'readonly':
        out.flags.writeable = False
    return out


@pytest.fixture(scope='module')
def genuine():
    if importlib.util.find_spec('dwd._feature_expansion_accel') is None:
        pytest.skip('This build has no feature-expansion extension.')
    assert sys.getprofile() is None and sys.gettrace() is None
    columns, z, alpha, penalty = literal('dyadic')
    prepared, splits = prepare(z, alpha, penalty)
    # Collection may already have imported the old facade with a cold cache.
    # Qualify its real ordered route before constructing the fused context.
    # This is explicit unit-fixture preparation, not a timing/prewarm claim.
    from dwd import _ordered_gradient_reduction as ordered
    planes = tuple(np.zeros(columns.shape) for _ in range(6))
    reduced = ordered.try_reduce(feature, planes)
    if importlib.util.find_spec('dwd._ordered_reduction_accel') is not None:
        assert type(reduced) is np.ndarray and reduced.tobytes() == np.zeros(len(columns)).tobytes()
    else:
        assert reduced is None
    assert ordered._LOAD_ATTEMPTED is True
    feature._gradient_tile(columns, prepared, _splits=splits)
    output = facade.try_gradient(feature, columns, prepared, splits)
    assert type(output) is np.ndarray, 'The actual fused path must accept ordinary input'
    assert output.tobytes() == reference._gradient_tile(columns, prepared).tobytes()
    assert facade._ACCEL is sys.modules['dwd._feature_expansion_accel']
    assert facade._ENTRY.__self__ is facade._ACCEL
    return facade._ACCEL, facade._CONTEXT


def native_bundle(context, columns, prepared, splits, interface, *, copy_tile=True):
    z, ze, high, low, ae, pe = prepared
    tile = np.array(columns, order='C', copy=True) if copy_tile else columns
    rhs = tuple(v for split in splits for v in split) if interface == 'splits' else (z, high, low)
    output = np.full(len(columns), -912.25)
    arrays = (tile, *rhs, ze, ae, output)
    return (context, tuple(memoryview(v) for v in arrays), pe), arrays, output


def checked(genuine, columns, prepared, splits, interface):
    module, context = genuine
    bundle, arrays, output = native_bundle(context, columns, prepared, splits, interface)
    before = [v.tobytes() for v in arrays]
    entry = getattr(module, 'gradient_from_' + interface)
    expected_error = None
    try:
        expected = reference._gradient_tile(columns, prepared)
    except (FloatingPointError, OverflowError) as exc:
        expected_error = type(exc)
    status = entry(bundle)
    assert all(v.tobytes() == b for v, b in zip(arrays[:-1], before[:-1]))
    if expected_error:
        assert status != 0
        assert output.tobytes() == before[-1]
    else:
        assert status == 0
        assert output.tobytes() == expected.tobytes()
    return status, output


