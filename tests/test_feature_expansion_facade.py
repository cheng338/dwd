"""Genuine G-S acceptance, callback guards and original fallback contracts."""
import gc
import importlib
import sys
import types

import numpy as np
import pytest

from _feature_expansion_test_support import (
    arithmetic, facade, feature, expansion_bridge, genuine, literal, native_bundle, prepare, reference,
    layout, LAYOUTS,
)



def test_facade_has_no_interface_selector():
    assert not hasattr(facade, 'INTERFACE')


def data():
    columns, z, alpha, penalty = literal('dyadic')
    prepared, splits = prepare(z, alpha, penalty)
    return columns, prepared, splits


def test_actual_unprofiled_acceptance_matches_original(genuine):
    source, prepared, splits = data()
    for arrangement in LAYOUTS:
        columns = layout(source, arrangement)
        before = columns.tobytes()
        output = facade.try_gradient(feature, columns, prepared, splits)
        assert type(output) is np.ndarray
        assert output.tobytes() == reference._gradient_tile(columns, prepared).tobytes()
        assert output.flags.owndata and not np.shares_memory(output, columns)
        assert columns.tobytes() == before
        assert facade._ENTRY.__self__ is genuine[0]


GUARD_CHANGES = (
    'product-alias', 'product-code', 'product-kwdefault', 'fsum-code',
    'restore-code', 'restore-default', 'frexp-alias', 'ldexp-alias',
    'asarray-alias', 'empty-alias', 'broadcast-alias', 'broadcast-body',
    'geterr-code', 'errstate-enter-code', 'finfo-tiny-code', 'finfo-tiny-member',
    'ordered-entry-code', 'ordered-callback', 'ordered-loader', 'ordered-owner',
    'ordered-native-entry', 'numpy-module', 'nonstring-dict-key', 'facade-code',
)


def forbidden_body(*args, **kwargs):
    # No closure cells, so it can replace a plain function's code in place.
    raise AssertionError('Mutated body must not execute')


@pytest.mark.parametrize('change', GUARD_CHANGES)
def test_final_c_barrier_refuses_changed_callbacks_without_calling_them(genuine, monkeypatch, change):
    module, context = genuine
    columns, prepared, splits = data()
    bundle, arrays, output = native_bundle(context, columns, prepared, splits)
    old = sys.modules['dwd._ordered_gradient_reduction']
    calls = []
    def trap(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Unknown replacement must not execute')
    if change == 'product-alias': monkeypatch.setattr(feature, '_products', trap)
    elif change == 'product-code': monkeypatch.setattr(arithmetic._products, '__code__', forbidden_body.__code__)
    elif change == 'product-kwdefault': monkeypatch.setitem(arithmetic._products.__kwdefaults__, '_split_b', object())
    elif change == 'fsum-code': monkeypatch.setattr(arithmetic._fsum, '__code__', forbidden_body.__code__)
    elif change == 'restore-code': monkeypatch.setattr(feature._restore_gradient_terms, '__code__', forbidden_body.__code__)
    elif change == 'restore-default': monkeypatch.setattr(feature._restore_gradient_terms, '__defaults__', (None,))
    elif change.endswith('-alias'):
        monkeypatch.setattr(np, {'frexp-alias': 'frexp', 'ldexp-alias': 'ldexp',
                                'asarray-alias': 'asarray', 'empty-alias': 'empty',
                                'broadcast-alias': 'broadcast_to'}[change], trap)
    elif change == 'broadcast-body': monkeypatch.setitem(np.broadcast_to.__dict__, '__wrapped__', trap)
    elif change == 'geterr-code': monkeypatch.setattr(np.geterr, '__code__', forbidden_body.__code__)
    elif change == 'errstate-enter-code': monkeypatch.setattr(np.errstate.__enter__, '__code__', forbidden_body.__code__)
    elif change == 'finfo-tiny-code':
        member = np.finfo.__dict__['tiny']
        function = member.fget if type(member) is property else member.__dict__['func']
        monkeypatch.setattr(function, '__code__', forbidden_body.__code__)
    elif change == 'finfo-tiny-member': monkeypatch.setattr(np.finfo, 'tiny', property(trap))
    elif change == 'ordered-entry-code': monkeypatch.setattr(old.try_reduce, '__code__', forbidden_body.__code__)
    elif change == 'ordered-callback': monkeypatch.setattr(old, '_callbacks', trap)
    elif change == 'ordered-loader': monkeypatch.setattr(old, '_native_entry', trap)
    elif change == 'ordered-owner': monkeypatch.setattr(old, '_native_module', trap)
    elif change == 'ordered-native-entry': monkeypatch.setattr(old._ACCEL, 'reduce_rows', trap)
    elif change == 'numpy-module': monkeypatch.setitem(sys.modules, 'numpy', object())
    elif change == 'nonstring-dict-key':
        class NoEquality:
            def __hash__(self): return 0
            def __eq__(self, other): raise AssertionError('Unknown equality hook')
        monkeypatch.setitem(np.__dict__, NoEquality(), object())
    elif change == 'facade-code': monkeypatch.setattr(facade.try_gradient, '__code__', forbidden_body.__code__)
    before = output.tobytes()
    status = module.gradient_from_splits(bundle)
    assert status == module.GUARD and output.tobytes() == before
    assert calls == []


@pytest.mark.parametrize('hook', ('profile', 'trace'))
def test_active_hooks_keep_literal_tile_and_direct_native_refuses(genuine, hook):
    module, context = genuine
    columns, prepared, splits = data()
    expected = reference._gradient_tile(columns, prepared)
    bundle, arrays, output = native_bundle(context, columns, prepared, splits)
    before = output.tobytes()
    setter = sys.setprofile if hook == 'profile' else sys.settrace
    getter = sys.getprofile if hook == 'profile' else sys.gettrace
    saved = getter()
    def observe(frame, event, arg): return observe
    try:
        setter(observe)
        declined = facade.try_gradient(feature, columns, prepared, splits)
        status = module.gradient_from_splits(bundle)
        actual = feature._gradient_tile(columns, prepared, _splits=splits)
    finally:
        setter(saved)
    assert declined is None and status == module.GUARD
    assert output.tobytes() == before and actual.tobytes() == expected.tobytes()


def test_missing_binary_attempt_once_then_safe_negative_cache(genuine, monkeypatch):
    columns, prepared, splits = data()
    monkeypatch.setattr(facade, '_LOAD_ATTEMPTED', False)
    monkeypatch.setattr(facade, '_ACCEL', None)
    monkeypatch.setattr(facade, '_ENTRY', None)
    monkeypatch.delitem(sys.modules, facade._NAME)
    calls = []
    def missing(name):
        assert name == facade._NAME
        calls.append(name)
        raise ImportError('Explicit missing-new-artifact fixture')
    monkeypatch.setattr(importlib, 'import_module', missing)
    assert facade.try_gradient(feature, columns, prepared, splits) is None
    assert facade.try_gradient(object(), object(), object(), object()) is None
    assert len(calls) == 1


@pytest.mark.parametrize('cached', ('unknown', 'none', 'old-abi'))
def test_cached_unknown_native_declines_before_import_hooks(genuine, monkeypatch, cached):
    columns, prepared, splits = data()
    monkeypatch.setattr(facade, '_LOAD_ATTEMPTED', False)
    monkeypatch.setattr(facade, '_ACCEL', None)
    monkeypatch.setattr(facade, '_ENTRY', None)
    class NoHooks:
        def __getattribute__(self, name): raise AssertionError('Unknown attribute hook')
    value = NoHooks() if cached == 'unknown' else None
    if cached == 'old-abi':
        value = types.ModuleType(facade._NAME)
        value.API_VERSION = 1
        value.COMPILED_PY_HEX = 0
    monkeypatch.setitem(sys.modules, facade._NAME, value)
    def forbidden(*args, **kwargs): raise AssertionError('Cached module must not be reimported')
    monkeypatch.setattr(importlib, 'import_module', forbidden)
    assert facade.try_gradient(feature, columns, prepared, splits) is None


def test_finalizer_during_preparation_is_seen_by_final_c_barrier(genuine, monkeypatch):
    columns, prepared, splits = data()
    old_frexp = np.frexp
    observed = []
    class Victim:
        def __del__(self):
            observed.append('finalized')
            np.frexp = object()
    victim = Victim()
    victim.cycle = victim
    pending = [victim]
    del victim
    original_view = memoryview
    def prepare_view(value):
        # Explicit test-only preparation hook. It changes no numerical/guard
        # function and invokes the real GC finalizer before the genuine C call.
        if pending:
            pending.clear()
            gc.collect()
        return original_view(value)
    monkeypatch.setattr(facade, 'memoryview', prepare_view, raising=False)
    try:
        actual = facade.try_gradient(feature, columns, prepared, splits)
    finally:
        np.frexp = old_frexp
    assert observed == ['finalized'] and actual is None


@pytest.mark.parametrize('fault', ('refusal', 'noninteger', 'unexpected', 'memory'))
def test_explicit_fault_injection_preserves_fallback_and_exceptions(genuine, monkeypatch, fault):
    columns, prepared, splits = data()
    expected = reference._gradient_tile(columns, prepared)
    marker = RuntimeError('exact unexpected marker')
    def injected(bundle):
        if fault == 'refusal': return 9
        if fault == 'noninteger': return False
        if fault == 'unexpected': raise marker
        raise MemoryError('explicit new native allocation fixture')
    # Fault-only loader bypass, never a genuine native activation witness.
    monkeypatch.setattr(facade, '_native_entry', lambda owner: injected)
    if fault == 'refusal':
        assert feature._gradient_tile(columns, prepared, _splits=splits).tobytes() == expected.tobytes()
    elif fault == 'noninteger':
        with pytest.raises(TypeError, match='exact integer'):
            feature._gradient_tile(columns, prepared, _splits=splits)
    elif fault == 'unexpected':
        with pytest.raises(RuntimeError) as caught:
            feature._gradient_tile(columns, prepared, _splits=splits)
        assert caught.value is marker
    else:
        x = np.tile(columns.T, (1, 9))[:, :17]
        z = np.full(len(x), .5)
        alpha = np.zeros(len(x))
        original_column, visited = feature._gradient_column, []
        def scalar(column, preparation):
            visited.append(len(visited))
            return original_column(column, preparation)
        monkeypatch.setattr(feature, '_gradient_column', scalar)
        actual = feature._feature_gradient(x, z, alpha, .25)
        expected_full = reference._feature_gradient(x, z, alpha, .25)
        assert actual.tobytes() == expected_full.tobytes()
        assert len(visited) == 17


def test_scalar_branch_imports_neither_fused_nor_ordered(monkeypatch):
    names = ('dwd._feature_expansion', 'dwd._feature_expansion_accel',
             'dwd._ordered_gradient_reduction', 'dwd._ordered_reduction_accel')
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    x = np.ones((2049, 1))
    z, alpha = np.full(2049, .5), np.zeros(2049)
    actual = feature._feature_gradient(x, z, alpha, .25)
    assert actual.tobytes() == np.array([1024.5]).tobytes()
    assert all(name not in sys.modules for name in names)


def test_first_eligible_tile_does_not_import_new_native_before_old_facade(monkeypatch):
    columns, prepared, splits = data()
    monkeypatch.delitem(sys.modules, 'dwd._ordered_gradient_reduction', raising=False)
    def forbidden(*args, **kwargs): raise AssertionError('No new native load before old facade')
    monkeypatch.setattr(facade, '_native_entry', forbidden)
    assert facade.try_gradient(feature, columns, prepared, splits) is None


def test_genuine_native_refusal_restores_first_scalar_error(genuine):
    module, context = genuine
    # First coordinate's complete sum overflows; the later coordinate contains
    # a subnormal restored term. Vector restoration sees the latter first,
    # whereas original scalar replay must expose the earlier sum overflow.
    columns = np.array([[sys.float_info.max, sys.float_info.max],
                        [sys.float_info.min*.5, 0.]])
    z, alpha = np.ones(2), np.zeros(2)
    prepared, splits = prepare(z, alpha, .25)
    bundle, arrays, output = native_bundle(context, columns, prepared, splits)
    before = output.tobytes()
    assert module.gradient_from_splits(bundle) == module.SUM_OVERFLOW
    assert output.tobytes() == before
    assert facade.try_gradient(feature, columns, prepared, splits) is None
    with pytest.raises(FloatingPointError) as tile_error:
        reference._gradient_tile(columns, prepared)
    with pytest.raises(FloatingPointError) as original_error:
        reference._feature_gradient(columns.T, z, alpha, .25)
    with pytest.raises(FloatingPointError) as candidate_error:
        feature._feature_gradient(columns.T, z, alpha, .25)
    assert str(tile_error.value) != str(original_error.value)
    assert str(candidate_error.value) == str(original_error.value)


def test_cold_ordered_cache_settles_before_context_then_refuses_mutation(genuine, monkeypatch):
    """Defer initial capture, but never rebase a later changed private cache."""
    from dwd import _ordered_gradient_reduction as ordered
    if importlib.util.find_spec('dwd._ordered_reduction_accel') is None:
        pytest.skip('This transition requires the genuine old ordered artifact.')
    module, original_context = genuine
    columns, prepared, splits = data()
    expected = reference._gradient_tile(columns, prepared)
    for name, value in (('_LOAD_ATTEMPTED', False), ('_LOAD_COMPLETE', False),
                        ('_ACCEL', None), ('_ENTRY', None)):
        monkeypatch.setattr(ordered, name, value)
    monkeypatch.setattr(facade, '_CONTEXT', None)
    monkeypatch.setattr(facade, '_BETA_ENTRY', None)
    # Model an already imported, cold private facade without changing a guard,
    # native entry or arithmetic callback. This is not a public import-order API.
    first = facade.try_gradient(feature, columns, prepared, splits)
    assert first is None and facade._CONTEXT is None
    assert ordered._LOAD_ATTEMPTED is False and ordered._ACCEL is None
    reduced = ordered.try_reduce(feature, tuple(np.zeros(columns.shape) for _ in range(6)))
    assert type(reduced) is np.ndarray and reduced.tobytes() == np.zeros(len(columns)).tobytes()
    assert ordered._LOAD_ATTEMPTED is True and ordered._ACCEL is not None
    assert ordered._LOAD_COMPLETE is True
    settled = facade.try_gradient(feature, columns, prepared, splits)
    assert type(settled) is np.ndarray and settled.tobytes() == expected.tobytes()
    settled_context = facade._CONTEXT
    assert settled_context is not None and settled_context is not original_context
    monkeypatch.setattr(ordered, '_ENTRY', None)
    bundle, arrays, output = native_bundle(settled_context, columns, prepared, splits)
    before = tuple(value.tobytes() for value in arrays)
    assert module.gradient_from_splits(bundle) == module.GUARD
    assert before == tuple(value.tobytes() for value in arrays)
    assert facade.try_gradient(feature, columns, prepared, splits) is None
    assert feature._gradient_tile(columns, prepared, _splits=splits).tobytes() == expected.tobytes()
    alpha = np.full(columns.shape[1], .5)
    assert feature._feature_coefficients(columns.T, alpha).tobytes() == reference._feature_coefficients(columns.T, alpha).tobytes()
    assert facade._CONTEXT is settled_context
