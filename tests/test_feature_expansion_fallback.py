"""Fallback checks run even when every optional native artifact is absent."""
import importlib
import sys
import types
import numpy as np
import pytest
from _feature_expansion_test_support import feature, facade, prepare, literal, reference
from dwd import _ordered_gradient_reduction as ordered


def reset_cache(monkeypatch):
    for name, value in (('_LOAD_ATTEMPTED', False), ('_ACCEL', None),
                        ('_ENTRY', None), ('_CONTEXT', None), ('_BETA_ENTRY', None)):
        monkeypatch.setattr(facade, name, value)


def test_missing_expansion_binary_is_negatively_cached_and_exact(monkeypatch):
    reset_cache(monkeypatch)
    monkeypatch.delitem(sys.modules, facade._NAME, raising=False)
    calls = []
    original = importlib.import_module
    def missing(name, *args, **kwargs):
        if name == facade._NAME:
            calls.append(name)
            raise ImportError('Absent optional expansion artifact')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(importlib, 'import_module', missing)
    assert facade._native_entry(feature) is None
    assert facade._LOAD_ATTEMPTED is True
    assert facade.try_gradient(object(), object(), object(), object()) is None
    assert facade._native_entry(feature) is None
    eligible_import = (sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 12)
                       and sys.platform in ('win32', 'linux') and sys.maxsize > 2**32
                       and facade._GIL_DISABLED in (None, 0))
    assert len(calls) == int(eligible_import)
    columns, z, alpha, penalty = literal('dyadic')
    x = columns.T
    assert feature._feature_gradient(x, z, alpha, penalty).tobytes() == reference._feature_gradient(x, z, alpha, penalty).tobytes()
    assert feature._feature_coefficients(x, alpha).tobytes() == reference._feature_coefficients(x, alpha).tobytes()


def test_absent_ordered_artifact_preserves_complete_gradient_and_beta(monkeypatch):
    reset_cache(monkeypatch)
    monkeypatch.setattr(ordered, '_LOAD_ATTEMPTED', True)
    monkeypatch.setattr(ordered, '_LOAD_COMPLETE', True)
    monkeypatch.setattr(ordered, '_ACCEL', None)
    monkeypatch.setattr(ordered, '_ENTRY', None)
    # A stable negative old-native cache can still support a genuine fused
    # context. Require complete results, not an undocumented import order.
    columns, z, alpha, penalty = literal('dyadic')
    assert feature._feature_gradient(columns.T, z, alpha, penalty).tobytes() == reference._feature_gradient(columns.T, z, alpha, penalty).tobytes()
    assert feature._feature_coefficients(columns.T, alpha).tobytes() == reference._feature_coefficients(columns.T, alpha).tobytes()
    assert ordered._LOAD_ATTEMPTED is True
    assert ordered._ACCEL is None and ordered._ENTRY is None


@pytest.mark.parametrize('state', ('unknown', 'none', 'module-subclass'))
def test_unknown_cached_artifact_never_probes_metadata_or_importer(monkeypatch, state):
    reset_cache(monkeypatch)
    class NoHooks:
        def __getattribute__(self, name): raise AssertionError('Unknown attribute hook')
    class SubModule(types.ModuleType):
        def __getattribute__(self, name): raise AssertionError('Module subclass attribute hook')
    value = NoHooks() if state == 'unknown' else SubModule(facade._NAME) if state == 'module-subclass' else None
    monkeypatch.setitem(sys.modules, facade._NAME, value)
    def forbidden(*args, **kwargs): raise AssertionError('Cached object was reimported')
    monkeypatch.setattr(importlib, 'import_module', forbidden)
    assert facade._native_entry(feature) is None


@pytest.mark.parametrize('key,value', [('API_VERSION', 0), ('API_VERSION', True),
    ('COMPILED_PY_HEX', sys.hexversion ^ 1), ('COMPILED_PY_HEX', True),
    ('MAX_ROWS', 65), ('MAX_ENTRIES', 2049)])
def test_native_metadata_refuses_wrong_abi_or_caps_without_loading(monkeypatch, key, value):
    module = types.ModuleType(facade._NAME)
    module.API_VERSION, module.COMPILED_PY_HEX = 2, sys.hexversion
    module.MAX_ROWS, module.MAX_ENTRIES = 64, 16384
    module.MAX_WIDTH = 1024
    setattr(module, key, value)
    monkeypatch.setitem(sys.modules, facade._NAME, module)
    assert facade._native_module(module, feature) is None


def test_cold_beta_never_imports_or_creates_context(monkeypatch):
    reset_cache(monkeypatch)
    def forbidden(*args, **kwargs): raise AssertionError('Cold beta attempted initialization')
    monkeypatch.setattr(facade, '_native_entry', forbidden)
    monkeypatch.setattr(facade, '_make_context', forbidden)
    columns, _, alpha, _ = literal('dyadic')
    assert feature._feature_coefficients(columns.T, alpha).tobytes() == reference._feature_coefficients(columns.T, alpha).tobytes()
    assert facade._LOAD_ATTEMPTED is False and facade._CONTEXT is None
