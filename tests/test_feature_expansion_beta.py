"""Warm-only beta expansion, exact arithmetic, refusal and callback contracts."""
import gc
import importlib
import math
import sys
import types
from fractions import Fraction

import numpy as np
import pytest

from _feature_expansion_test_support import (
    arithmetic, facade, feature, genuine, layout, LAYOUTS, prepare, reference,
)

assert facade.INTERFACE == 'G-S'
CASES = ('zeros', 'signed-zero', 'dyadic', 'cancel', 'low-product', 'half-even',
         'subnormal-final', 'restored-underflow', 'restored-erasure', 'overflow',
         'nonfinite', 'zero-alpha')


def data(name='dyadic'):
    columns = np.array([[1., 2., -1., .5, -2., 4., .25, -4.],
                        [2., -1., 4., -2., .5, 1., -4., .25]])
    alpha = np.array([.5, -.25, .75, -.5, .125, -.125, .25, -.75])
    if name in ('zeros', 'signed-zero'):
        columns[:] = 0.; alpha[:] = 0.
        if name == 'signed-zero':
            columns[0, ::2] = -0.; alpha[1::2] = -0.
    elif name == 'cancel':
        columns[:] = 1.; alpha[:] = [2.**52, 1., -2.**52, -1., 0, 0, 0, 0]
    elif name == 'low-product':
        columns[:] = math.nextafter(.75, 1.)
        alpha[:] = math.nextafter(.5, 1.)
    elif name == 'half-even':
        columns[:] = 1.; alpha[:] = [1., 2.**-53, 2.**-54, -2.**-54, 0, 0, 0, 0]
    elif name == 'subnormal-final':
        columns[:] = 1.
        alpha[:] = [math.nextafter(sys.float_info.min, math.inf), -sys.float_info.min, 0, 0, 0, 0, 0, 0]
    elif name in ('restored-underflow', 'restored-erasure'):
        columns[:] = .5 if name == 'restored-underflow' else 2.**-100
        alpha[:] = sys.float_info.min
    elif name == 'overflow':
        columns[:] = sys.float_info.max; alpha[:] = 2.
    elif name == 'nonfinite':
        columns[1, -1] = math.inf
    elif name == 'zero-alpha':
        alpha[:] = -0.
    return columns, alpha


def prepared(alpha):
    mantissa, exponent = np.frexp(alpha)
    return (mantissa, exponent), arithmetic._split_operand(mantissa)


def bundle(context, columns, prep, split):
    output = np.full(len(columns), -912.25)
    arrays = (np.array(columns, order='C', copy=True), *split, prep[1], output)
    return (context, tuple(memoryview(value) for value in arrays)), arrays, output


def unprofiled_beta(genuine, columns, alpha):
    assert sys.getprofile() is None and sys.gettrace() is None
    module, context = genuine
    prep, split = prepared(alpha)
    native, arrays, output = bundle(context, columns, prep, split)
    before = [value.tobytes() for value in arrays]
    error = None
    try:
        expected = reference._beta_tile(columns, *prep)
    except (FloatingPointError, OverflowError) as exc:
        error = exc
    status = module.beta_from_split(native)
    assert all(value.tobytes() == old for value, old in zip(arrays[:-1], before[:-1]))
    actual = facade.try_beta(feature, columns, prep, split)
    if error is None:
        assert status == module.OK
        assert output.tobytes() == expected.tobytes() == actual.tobytes()
        assert actual.flags.owndata and not np.shares_memory(actual, columns)
        rational = np.array([float(sum((Fraction(float(x))*Fraction(float(y))
                                       for x, y in zip(row, alpha)), Fraction()))
                             for row in columns])
        assert actual.tobytes() == rational.tobytes()
        assert feature._beta_tile(columns, *prep, _split=split).tobytes() == expected.tobytes()
    else:
        assert status != module.OK and output.tobytes() == before[-1] and actual is None
        with pytest.raises(type(error)) as caught:
            feature._beta_tile(columns, *prep, _split=split)
        assert str(caught.value) == str(error)


@pytest.mark.parametrize('arrangement', LAYOUTS)
@pytest.mark.parametrize('name', CASES)
def test_literal_beta_bits_and_full_tile_fallback(genuine, arrangement, name):
    columns, alpha = data(name)
    columns = layout(columns, arrangement)
    before = columns.tobytes(), alpha.tobytes()
    unprofiled_beta(genuine, columns, alpha)
    assert before == (columns.tobytes(), alpha.tobytes())


CAPS = ((2, 1), (64, 32), (2, 1024), (65, 1), (2, 1025), (1, 1), (2, 0))


@pytest.mark.parametrize('shape', CAPS, ids=('min', 'max-rows', 'max-width', 'rows-over', 'width-over', 'scalar', 'empty'))
def test_original_tile_caps_and_atomic_refusal(genuine, shape):
    module, context = genuine
    columns, alpha = np.ones(shape), np.full(shape[1], .5)
    prep, split = prepared(alpha)
    native, arrays, output = bundle(context, columns, prep, split)
    before = output.tobytes()
    status = module.beta_from_split(native)
    accepted = 2 <= shape[0] <= 64 and 0 < shape[1] <= 1024 and shape[0]*shape[1] <= 16384
    if accepted:
        assert status == module.OK
        assert output.tobytes() == reference._beta_tile(columns, *prep).tobytes()
    else:
        assert status == module.ARGUMENT and output.tobytes() == before


MALFORMED = ('float32', 'nonnative', 'unaligned', 'F', 'negative', 'strided',
             'input-rank', 'native-long', 'exponent-shape', 'readonly-output',
             'output-shape', 'overlap')


@pytest.mark.parametrize('change', MALFORMED)
def test_beta_bridge_layout_refusal_is_atomic(genuine, change):
    module, context = genuine
    columns, alpha = data()
    prep, split = prepared(alpha)
    _, arrays, output = bundle(context, columns, prep, split)
    arrays = list(arrays)
    if change == 'float32': arrays[0] = arrays[0].astype('float32')
    elif change == 'nonnative': arrays[0] = arrays[0].astype('>f8' if sys.byteorder == 'little' else '<f8')
    elif change == 'unaligned':
        raw = bytearray(arrays[0].nbytes+1)
        arrays[0] = np.ndarray(columns.shape, dtype='float64', buffer=raw, offset=1)
        arrays[0][:] = columns
    elif change == 'F': arrays[0] = np.array(arrays[0], order='F')
    elif change == 'negative': arrays[0] = arrays[0][:, ::-1]
    elif change == 'strided': arrays[0] = np.tile(arrays[0], (1, 2))[:, ::2]
    elif change == 'input-rank': arrays[0] = arrays[0].reshape(-1)
    elif change == 'native-long': arrays[5] = arrays[5].astype('long')
    elif change == 'exponent-shape': arrays[5] = np.zeros(len(alpha)+1, dtype=np.intc)
    elif change == 'readonly-output': output.flags.writeable = False
    elif change == 'output-shape': arrays[-1] = output.reshape(1, -1)
    elif change == 'overlap': arrays[-1] = arrays[0].reshape(-1)[:len(columns)]
    before = [value.tobytes() for value in arrays]
    status = module.beta_from_split((context, tuple(memoryview(value) for value in arrays)))
    assert status == module.ARGUMENT
    assert before == [value.tobytes() for value in arrays]


GUARDS = ('beta-code', 'beta-default', 'helper-code', 'facade-code',
          'native-entry', 'product-alias', 'restore-code', 'numpy-frexp')


def forbidden(*args, **kwargs):
    raise AssertionError('Unknown replacement must not execute')


@pytest.mark.parametrize('change', GUARDS)
def test_final_barrier_checks_beta_family(genuine, monkeypatch, change):
    module, context = genuine
    columns, alpha = data(); prep, split = prepared(alpha)
    native, arrays, output = bundle(context, columns, prep, split)
    before = output.tobytes()
    if change == 'beta-code': monkeypatch.setattr(feature._beta_tile, '__code__', forbidden.__code__)
    elif change == 'beta-default': monkeypatch.setitem(feature._beta_tile.__kwdefaults__, '_split', object())
    elif change == 'helper-code': monkeypatch.setattr(feature._try_fused_beta, '__code__', forbidden.__code__)
    elif change == 'facade-code': monkeypatch.setattr(facade.try_beta, '__code__', forbidden.__code__)
    elif change == 'native-entry': monkeypatch.setattr(module, 'beta_from_split', forbidden)
    elif change == 'product-alias': monkeypatch.setattr(feature, '_products', forbidden)
    elif change == 'restore-code': monkeypatch.setattr(feature._restore_gradient_terms, '__code__', forbidden.__code__)
    elif change == 'numpy-frexp': monkeypatch.setattr(np, 'frexp', forbidden)
    # Retain the genuine C function even when its public binding is replaced.
    assert facade._BETA_ENTRY(native) == module.GUARD
    assert output.tobytes() == before


def test_natural_gradient_then_beta_then_gradient(genuine):
    module, context = genuine
    columns, alpha = data(); prep, split = prepared(alpha)
    assert facade._BETA_ENTRY is module.beta_from_split
    assert facade._BETA_ENTRY.__self__ is module
    out = facade.try_beta(feature, columns, prep, split)
    assert out.tobytes() == reference._beta_tile(columns, *prep).tobytes()
    gp, gs = prepare(alpha, alpha, .25)
    assert facade.try_gradient(feature, columns, gp, gs).tobytes() == reference._gradient_tile(columns, gp).tobytes()
    # Original128-row tile budget gives16 columns and a literal scalar tail.
    x, tiled_alpha = np.tile(columns.T, (16, 9))[:, :17], np.tile(alpha, 16)
    before = x.tobytes(), tiled_alpha.tobytes()
    assert feature._feature_coefficients(x, tiled_alpha).tobytes() == reference._feature_coefficients(x, tiled_alpha).tobytes()
    assert before == (x.tobytes(), tiled_alpha.tobytes())
    assert facade._CONTEXT is context


@pytest.mark.parametrize('state', ('facade-absent', 'context-none', 'entry-none', 'binary-absent'))
def test_beta_never_imports_or_constructs_context(genuine, monkeypatch, state):
    columns, alpha = data(); prep, split = prepared(alpha)
    expected = reference._beta_tile(columns, *prep)
    if state == 'facade-absent': monkeypatch.delitem(sys.modules, 'dwd._feature_expansion')
    elif state == 'context-none': monkeypatch.setattr(facade, '_CONTEXT', None)
    elif state == 'entry-none': monkeypatch.setattr(facade, '_BETA_ENTRY', None)
    else: monkeypatch.delitem(sys.modules, facade._NAME)
    monkeypatch.setattr(importlib, 'import_module', forbidden)
    monkeypatch.setattr(facade, '_native_entry', forbidden)
    monkeypatch.setattr(facade, '_make_context', forbidden)
    assert feature._beta_tile(columns, *prep, _split=split).tobytes() == expected.tobytes()


def test_absent_beta_capability_keeps_gradient_context_but_beta_declines(genuine, monkeypatch):
    module, context = genuine
    columns, alpha = data(); gp, gs = prepare(alpha, alpha, .25)
    with monkeypatch.context() as m:
        # Exact real artifact minus only the optional advertised beta entry;
        # explicit capability fixture, not a new binary provenance claim.
        m.delattr(module, 'BETA_API_VERSION')
        m.delattr(module, 'beta_from_split')
        m.setattr(facade, '_CONTEXT', None)
        m.setattr(facade, '_BETA_ENTRY', None)
        assert facade.try_gradient(feature, columns, gp, gs).tobytes() == reference._gradient_tile(columns, gp).tobytes()
        assert facade._BETA_ENTRY is None
        prep, split = prepared(alpha)
        assert facade.try_beta(feature, columns, prep, split) is None


@pytest.mark.parametrize('hook', ('profile', 'trace'))
def test_hooks_decline_and_literal_beta_remains(genuine, hook):
    module, context = genuine
    columns, alpha = data(); prep, split = prepared(alpha)
    expected = reference._beta_tile(columns, *prep)
    native, arrays, output = bundle(context, columns, prep, split); before = output.tobytes()
    setter = sys.setprofile if hook == 'profile' else sys.settrace
    getter = sys.getprofile if hook == 'profile' else sys.gettrace
    saved = getter()
    def observe(frame, event, arg): return observe
    try:
        setter(observe)
        assert facade.try_beta(feature, columns, prep, split) is None
        assert module.beta_from_split(native) == module.GUARD
        actual = feature._beta_tile(columns, *prep, _split=split)
    finally:
        setter(saved)
    assert output.tobytes() == before and actual.tobytes() == expected.tobytes()


def test_finalizer_after_precheck_declines(genuine, monkeypatch):
    columns, alpha = data(); prep, split = prepared(alpha)
    original, observed = np.frexp, []
    class Victim:
        def __del__(self):
            observed.append(True); np.frexp = object()
    victim = Victim(); victim.cycle = victim
    pending = [victim]; del victim
    view = memoryview
    def acquire(value):
        if pending:
            pending.clear(); gc.collect()
        return view(value)
    monkeypatch.setattr(facade, 'memoryview', acquire, raising=False)
    try:
        assert facade.try_beta(feature, columns, prep, split) is None
    finally:
        np.frexp = original
    assert observed == [True]


@pytest.mark.parametrize('fault', ('refusal', 'noninteger', 'unexpected', 'memory'))
def test_fault_only_entry_preserves_public_fallback(genuine, monkeypatch, fault):
    columns, alpha = data(); prep, split = prepared(alpha)
    marker = RuntimeError('Q unexpected marker')
    def injected(bundle):
        if fault == 'refusal': return 9
        if fault == 'noninteger': return False
        if fault == 'unexpected': raise marker
        raise MemoryError('Q injected native allocation')
    # Explicit fault-only qualification bypass, never native-use evidence.
    monkeypatch.setattr(facade, '_beta_native_entry', lambda owner: injected)
    monkeypatch.setattr(facade, '_BETA_ENTRY', injected)
    if fault == 'refusal':
        assert feature._beta_tile(columns, *prep, _split=split).tobytes() == reference._beta_tile(columns, *prep).tobytes()
    elif fault == 'noninteger':
        with pytest.raises(TypeError, match='exact integer'): feature._beta_tile(columns, *prep, _split=split)
    elif fault == 'unexpected':
        with pytest.raises(RuntimeError) as caught: feature._beta_tile(columns, *prep, _split=split)
        assert caught.value is marker
    else:
        x = np.tile(columns.T, (1, 9))[:, :17]
        assert feature._feature_coefficients(x, alpha).tobytes() == reference._feature_coefficients(x, alpha).tobytes()


def test_scalar_no_new_modules(genuine, monkeypatch):
    names = ('dwd._feature_expansion', 'dwd._feature_expansion_accel',
             'dwd._ordered_gradient_reduction', 'dwd._ordered_reduction_accel')
    for name in names: monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(importlib, 'import_module', forbidden)
    x, alpha = np.ones((2049, 1)), np.full(2049, .5)
    assert feature._feature_coefficients(x, alpha).tobytes() == np.array([1024.5]).tobytes()
    assert all(name not in sys.modules for name in names)


def test_genuine_refusal_preserves_first_scalar_error(genuine):
    module, context = genuine
    columns = np.array([[sys.float_info.max, sys.float_info.max], [sys.float_info.min*.5, 0.]])
    alpha = np.ones(2); prep, split = prepared(alpha)
    native, arrays, output = bundle(context, columns, prep, split); before = output.tobytes()
    assert module.beta_from_split(native) == module.SUM_OVERFLOW
    assert output.tobytes() == before and facade.try_beta(feature, columns, prep, split) is None
    with pytest.raises(FloatingPointError) as vector: reference._beta_tile(columns, *prep)
    with pytest.raises(FloatingPointError) as original: reference._feature_coefficients(columns.T, alpha)
    with pytest.raises(FloatingPointError) as actual: feature._feature_coefficients(columns.T, alpha)
    assert str(vector.value) != str(original.value) and str(actual.value) == str(original.value)


@pytest.mark.parametrize('point', ('tile-copy', 'private-output'))
def test_allocation_failure_replays_all_scalar_columns(genuine, monkeypatch, point):
    columns, alpha = data(); x = np.tile(columns.T, (1, 9))[:, :17]
    original_column, visited = feature._beta_column, []
    def scalar(column, mantissa, exponent):
        visited.append(True); return original_column(column, mantissa, exponent)
    monkeypatch.setattr(feature, '_beta_column', scalar)
    if point == 'private-output':
        old = sys.modules[facade._OLD_NAME]
        def allocation(*args, **kwargs): raise MemoryError('Q private-output fault')
        # Fault-only initial qualification bypass lets the allocator raise at
        # its real callsite; it is never used as native acceptance evidence.
        monkeypatch.setattr(facade, '_initial', lambda owner, previous: True)
        monkeypatch.setattr(old, '_EMPTY', (old._EMPTY[0], allocation, *old._EMPTY[2:]))
    else:
        # Explicit callsite allocation fault: covers unchanged outer MemoryError
        # scalar replay without pretending ndarray.copy can be monkeypatched.
        def allocation(*args, **kwargs): raise MemoryError('Q tile-copy fault')
        monkeypatch.setattr(feature, '_try_fused_beta', allocation)
    assert feature._feature_coefficients(x, alpha).tobytes() == reference._feature_coefficients(x, alpha).tobytes()
    assert len(visited) == 17


@pytest.mark.parametrize('cached', ('unknown', 'subclass'))
def test_unknown_cached_facade_has_no_attribute_hooks(genuine, monkeypatch, cached):
    class NoHooks:
        def __getattribute__(self, name): raise AssertionError('Unknown metadata access')
    class SubModule(types.ModuleType):
        def __getattribute__(self, name): raise AssertionError('Subclass metadata access')
    value = NoHooks() if cached == 'unknown' else SubModule('dwd._feature_expansion')
    monkeypatch.setitem(sys.modules, 'dwd._feature_expansion', value)
    columns, alpha = data(); prep, split = prepared(alpha)
    assert feature._beta_tile(columns, *prep, _split=split).tobytes() == reference._beta_tile(columns, *prep).tobytes()
