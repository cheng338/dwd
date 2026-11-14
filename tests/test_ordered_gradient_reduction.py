"""Package ordered-gradient guards, actual dispatch, and portable fallback.

Native status injections deliberately bypass entry qualification only in
fault tests. Real-entry checks skip only an intentionally absent binary or
a runtime outside the declared native platform scope. Native wheel CI also
requires an actual entry witness; it must not use portable skips as success.
"""
import importlib
import importlib.util
import math
import platform
import sysconfig
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import psutil
import pytest

import dwd._linear_feature_diagnostics as feature
import dwd._ordered_gradient_reduction as facade
import dwd._compensated_residual as arithmetic
import dwd._feature_report_optimization as support


@pytest.fixture(autouse=True)
def fresh_native_cache(monkeypatch):
    monkeypatch.setattr(facade, '_LOAD_ATTEMPTED', False)
    monkeypatch.setattr(facade, '_ACCEL', None)
    monkeypatch.setattr(facade, '_ENTRY', None)


def planes():
    rows, cols = np.arange(4)[:, None], np.arange(31)[None, :]
    return tuple(np.array(((rows*13+cols*7+i*11) % 47-23)/1024.0, order='F') for i in range(6))


def reference(values):
    return np.array([feature._fsum([value for plane in values for value in plane[i].tolist()])
                     for i in range(len(values[0]))])


def contraction(n=128, p=17):
    r, c = np.arange(n)[:, None], np.arange(p)[None, :]
    X = ((r*17+c*31+r*c) % 257-128).astype(float)/256.0
    z = ((np.arange(n)*13) % 31-15).astype(float)/1024.0
    alpha = ((np.arange(n)*7) % 19-9).astype(float)/512.0
    return X, z, alpha, .25


def observe_calls(operation, entry):
    calls = []
    old = sys.getprofile()
    def profile(frame, event, arg):
        if event == 'c_call' and arg is entry:
            calls.append(arg)
    try:
        sys.setprofile(profile)
        result = operation()
    finally:
        sys.setprofile(old)
    return result, calls


def require_entry():
    # Test-side eligibility is explicit, not inferred from production refusal.
    # A present binary on a qualified runtime must import and dispatch.
    if (sys.implementation.name != 'cpython'
            or sys.platform not in ('win32', 'linux')
            or platform.machine().lower() not in ('amd64', 'x86_64')
            or sys.maxsize <= 2**32
            or sysconfig.get_config_var('Py_GIL_DISABLED') not in (None, 0)):
        pytest.skip('Native ordered-gradient qualification requires a GIL-enabled '
                    'Windows/Linux x86-64 CPython runtime.')
    if importlib.util.find_spec('dwd._ordered_reduction_accel') is None:
        pytest.skip('Portable build has no ordered-gradient binary.')
    importlib.import_module('dwd._ordered_reduction_accel')
    assert facade._callbacks(feature), 'Ordinary callback proof must activate'
    entry = facade._native_entry(feature)
    assert entry is not None, 'Present binary must qualify on this ordinary input'
    return entry


def test_real_direct_native_entry_is_observed_and_exact():
    values = planes()
    expected = reference(values)
    entry = require_entry()
    actual, calls = observe_calls(lambda: facade.try_reduce(feature, values), entry)
    assert len(calls) == 1 and actual is not None
    assert actual.tobytes() == expected.tobytes()
    assert facade._ACCEL is sys.modules['dwd._ordered_reduction_accel']
    assert entry.__self__ is facade._ACCEL


def test_real_feature_gradient_dispatch_then_scalar_tail_matches_original(monkeypatch):
    args = contraction()
    entry = require_entry()
    with monkeypatch.context() as patched:
        patched.setattr(facade, '_native_entry', lambda ignored: None)
        expected = feature._feature_gradient(*args)
    actual, calls = observe_calls(lambda: feature._feature_gradient(*args), entry)
    assert len(calls) == 1  # one16-column tile, then one untouched scalar column
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('layout', ['C', 'F', 'strided', 'readonly'])
def test_plain_layouts_are_exact(layout):
    values = planes()
    if layout == 'C':
        values = tuple(np.array(a, order='C') for a in values)
    elif layout == 'strided':
        values = tuple(a[:, ::-1] for a in values)
    elif layout == 'readonly':
        for value in values:
            value.setflags(write=False)
    before = [a.tobytes() for a in values]
    require_entry()
    actual = facade.try_reduce(feature, values)
    assert actual is not None and actual.tobytes() == reference(values).tobytes()
    assert before == [a.tobytes() for a in values]


@pytest.mark.parametrize('error_type', [ImportError, OSError])
def test_missing_optional_binary_is_negatively_cached_before_allocation(monkeypatch, error_type):
    values = planes()
    monkeypatch.delitem(sys.modules, 'dwd._ordered_reduction_accel', raising=False)
    calls = []
    def missing(name):
        calls.append(name)
        raise error_type('test-optional-binary-missing')
    monkeypatch.setattr(facade.importlib, 'import_module', missing)
    results, allocations = observe_calls(lambda: [facade.try_reduce(feature, values) for _ in range(3)], np.empty)
    assert results == [None, None, None]
    assert calls == ['dwd._ordered_reduction_accel'] and not allocations
    assert facade._LOAD_ATTEMPTED and facade._ACCEL is facade._ENTRY is None


def test_unexpected_import_error_propagates(monkeypatch):
    monkeypatch.delitem(sys.modules, 'dwd._ordered_reduction_accel', raising=False)
    error = RuntimeError('unexpected-import-failure')
    def failed(name):
        raise error
    monkeypatch.setattr(facade.importlib, 'import_module', failed)
    with pytest.raises(RuntimeError) as seen:
        facade.try_reduce(feature, planes())
    assert seen.value is error


class NoHooks:
    def __eq__(self, other):
        raise AssertionError('Unexpected comparison hook')
    def __hash__(self):
        raise AssertionError('Unexpected hash hook')
    def __call__(self, *args, **kwargs):
        raise AssertionError('Unexpected call hook')
    def __getattr__(self, name):
        raise AssertionError('Unexpected attribute hook '+name)


@pytest.mark.parametrize('kind', ['unknown_object', 'module_subclass', 'malformed_spec', 'cached_none'])
def test_cold_facade_declines_unknown_cached_modules_without_import(monkeypatch, kind):
    values = planes()
    native_name = 'dwd._ordered_reduction_accel'
    if kind == 'unknown_object':
        class NoCachedInspection(NoHooks):
            def __getattribute__(self, name):
                raise AssertionError('Unexpected cached object inspection '+name)
        cached = NoCachedInspection()
    elif kind == 'module_subclass':
        class NoModuleInspection(types.ModuleType):
            def __getattribute__(self, name):
                raise AssertionError('Unexpected cached module subclass inspection '+name)
        cached = NoModuleInspection(native_name)
    elif kind == 'malformed_spec':
        # Genuine metadata ensures this reaches the spec type guard. Do not
        # create a fresh nominal module which might decline earlier for API.
        require_entry()
        cached = facade._ACCEL
        monkeypatch.setitem(cached.__dict__, '__spec__', NoHooks())
    else:
        cached = None
    monkeypatch.setitem(sys.modules, native_name, cached)
    monkeypatch.setattr(facade, '_LOAD_ATTEMPTED', False)
    monkeypatch.setattr(facade, '_ACCEL', None)
    monkeypatch.setattr(facade, '_ENTRY', None)
    imports = []
    def unexpected_import(name):
        imports.append(name)
        raise AssertionError('Cached entry must be qualified without importlib')
    monkeypatch.setattr(facade.importlib, 'import_module', unexpected_import)
    results, allocations = observe_calls(
        lambda: [facade.try_reduce(feature, values) for _ in range(2)], np.empty)
    assert results == [None, None] and not imports and not allocations
    assert facade._LOAD_ATTEMPTED and facade._ACCEL is facade._ENTRY is None


def test_cold_facade_accepts_genuine_preloaded_binary_without_import(monkeypatch):
    values = planes()
    expected = reference(values)
    entry = require_entry()
    module = facade._ACCEL
    monkeypatch.setattr(facade, '_LOAD_ATTEMPTED', False)
    monkeypatch.setattr(facade, '_ACCEL', None)
    monkeypatch.setattr(facade, '_ENTRY', None)
    imports = []
    def unexpected_import(name):
        imports.append(name)
        raise AssertionError('Genuine cached entry must not be reimported')
    monkeypatch.setattr(facade.importlib, 'import_module', unexpected_import)
    actual, calls = observe_calls(lambda: facade.try_reduce(feature, values), entry)
    assert not imports and len(calls) == 1
    assert actual is not None and actual.tobytes() == expected.tobytes()
    assert facade._LOAD_ATTEMPTED and facade._ACCEL is module and facade._ENTRY is entry


@pytest.mark.parametrize('field', ['API_VERSION', 'MAX_ROWS', 'MAX_TERMS', '__name__', '__file__', '__spec__', '__loader__', 'reduce_rows'])
def test_unknown_native_metadata_declines_without_hooks(monkeypatch, field):
    require_entry()
    module = facade._ACCEL
    monkeypatch.setitem(module.__dict__, field, NoHooks())
    assert facade._native_module(module, feature) is None
    assert facade.try_reduce(feature, planes()) is None


@pytest.mark.parametrize('value', [None, 0, True, 2])
def test_old_or_mismatched_native_api_is_declined(monkeypatch, value):
    require_entry()
    monkeypatch.setattr(facade._ACCEL, 'API_VERSION', value)
    assert facade.try_reduce(feature, planes()) is None


def test_native_registration_replacement_is_declined_without_hooks(monkeypatch):
    require_entry()
    monkeypatch.setitem(sys.modules, 'dwd._ordered_reduction_accel', NoHooks())
    assert facade.try_reduce(feature, planes()) is None


def test_entry_from_wrong_module_owner_is_declined(monkeypatch):
    require_entry()
    monkeypatch.setattr(facade._ACCEL, 'reduce_rows', math.fsum)
    assert facade.try_reduce(feature, planes()) is None


@pytest.mark.parametrize('field', ['origin', 'name', 'loader'])
def test_unknown_spec_metadata_declines(monkeypatch, field):
    require_entry()
    monkeypatch.setitem(facade._ACCEL.__spec__.__dict__, field, NoHooks())
    assert facade.try_reduce(feature, planes()) is None


@pytest.mark.parametrize('name', ['empty', 'isfinite', 'geterr'])
def test_unknown_numpy_entries_decline_without_calling_them(monkeypatch, name):
    values = planes()
    monkeypatch.setattr(np, name, NoHooks())
    assert facade.try_reduce(feature, values) is None


def test_unknown_float64_alias_declines_before_dtype_equality(monkeypatch):
    values = planes()
    monkeypatch.setattr(np, 'float64', NoHooks())
    assert facade.try_reduce(feature, values) is None


@pytest.mark.parametrize('name', ['ndarray', 'ufunc'])
def test_unknown_numpy_type_alias_declines_without_hooks(monkeypatch, name):
    values = planes()
    monkeypatch.setattr(np, name, NoHooks())
    assert facade.try_reduce(feature, values) is None


@pytest.mark.parametrize('name', ['np', '_arithmetic', '_support'])
def test_replaced_facade_module_declines_without_hooks(monkeypatch, name):
    values = planes()
    class NoModuleDictionary(NoHooks):
        def __getattribute__(self, attribute):
            if attribute == '__dict__':
                raise AssertionError('Unexpected module dictionary access')
            return object.__getattribute__(self, attribute)
    monkeypatch.setattr(facade, name, NoModuleDictionary())
    assert facade.try_reduce(feature, values) is None


@pytest.mark.parametrize('kind', ['list', 'count', 'subclass', 'float32', 'shape', 'scalar', 'oversize'])
def test_unsupported_planes_decline_before_native_entry(monkeypatch, kind):
    values = planes()
    if kind == 'list':
        values = list(values)
    elif kind == 'count':
        values = values[:5]
    elif kind == 'subclass':
        class Subclass(np.ndarray):
            pass
        values = (values[0].view(Subclass),) + values[1:]
    elif kind == 'float32':
        values = (values[0].astype(np.float32),) + values[1:]
    elif kind == 'shape':
        values = (values[0][:, :-1],) + values[1:]
    elif kind == 'scalar':
        values = tuple(value[:1] for value in values)
    else:
        values = tuple(np.zeros((64, 33)) for _ in range(6))
    monkeypatch.setattr(facade, '_native_entry', NoHooks())
    assert facade.try_reduce(feature, values) is None


def test_unknown_dispatcher_type_cannot_authorize_wrapped_attribute_hook(monkeypatch):
    values = planes()
    monkeypatch.setattr(support, '_NUMPY_DISPATCHER', NoHooks)
    monkeypatch.setattr(np, 'geterr', NoHooks())
    assert facade.try_reduce(feature, values) is None


@pytest.mark.parametrize('name', ['_products', '_fsum', '_restore_gradient_terms'])
def test_changed_feature_callback_declines(monkeypatch, name):
    values = planes()
    monkeypatch.setattr(feature, name, NoHooks())
    assert facade.try_reduce(feature, values) is None


@pytest.mark.parametrize('field,value', [('__defaults__', (None,)), ('__kwdefaults__', {'test_key': None})])
def test_changed_fsum_call_contract_declines(monkeypatch, field, value):
    values = planes()
    monkeypatch.setattr(arithmetic._fsum, field, value)
    assert facade.try_reduce(feature, values) is None


def test_owned_code_change_before_lazy_facade_import_is_not_adopted(monkeypatch):
    # Emulate a late change after the earlier split-token capture. Keep function
    # identity, globals, name and filename; only its actual behavior changes.
    scope = {}
    code = compile('def _fsum(terms):\n    return math.fsum(terms)+1.0\n', arithmetic._fsum.__code__.co_filename, 'exec')
    exec(code, scope)
    monkeypatch.setattr(arithmetic._fsum, '__code__', scope['_fsum'].__code__)
    path = Path(facade.__file__)
    spec = importlib.util.spec_from_file_location('dwd._ordered_gradient_reduction_preimport_test', path)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh.try_reduce(feature, planes()) is None


def test_geterr_callback_mode_declines_without_invoking_hook():
    values = planes()
    calls = []
    old = np.geterrcall()
    try:
        np.seterrcall(lambda *args: calls.append(args))
        with np.errstate(over='call'):
            assert facade.try_reduce(feature, values) is None
    finally:
        np.seterrcall(old)
    assert not calls


@pytest.mark.parametrize('status', [1, 2, 3, 4, 5, 6])
def test_injected_native_refusal_returns_none_and_full_gradient_replays(monkeypatch, status):
    args = contraction()
    with monkeypatch.context() as patch:
        patch.setattr(facade, '_native_entry', lambda ignored: None)
        expected = feature._feature_gradient(*args)
    calls = []
    def refused(packed, output):
        calls.append(packed.shape)
        output[:] = 999.0  # deliberate fault: never publish private values
        return status
    monkeypatch.setattr(facade, '_native_entry', lambda ignored: refused)
    actual = feature._feature_gradient(*args)
    assert calls and actual.tobytes() == expected.tobytes()


def test_injected_invalid_success_is_discarded(monkeypatch):
    def invalid(packed, output):
        output[:] = math.nan
        return 0
    monkeypatch.setattr(facade, '_native_entry', lambda ignored: invalid)
    assert facade.try_reduce(feature, planes()) is None


@pytest.mark.parametrize('status', [False, 0.0, None, np.int64(0)])
def test_injected_wrong_status_type_raises(monkeypatch, status):
    monkeypatch.setattr(facade, '_native_entry', lambda ignored: lambda packed, output: status)
    with pytest.raises(TypeError, match='exact integer'):
        facade.try_reduce(feature, planes())


@pytest.mark.parametrize('error', [RuntimeError('unexpected-native'), KeyboardInterrupt('test-interrupt')])
def test_injected_unexpected_native_error_propagates_same_object(monkeypatch, error):
    def failed(packed, output):
        raise error
    monkeypatch.setattr(facade, '_native_entry', lambda ignored: failed)
    with pytest.raises(type(error)) as seen:
        facade.try_reduce(feature, planes())
    assert seen.value is error


def test_packing_memoryerror_unwinds_to_existing_scalar_replay(monkeypatch):
    args = contraction()
    with monkeypatch.context() as patch:
        patch.setattr(facade, '_native_entry', lambda ignored: None)
        expected = feature._feature_gradient(*args)
    real_empty = np.empty
    calls = []
    def injected_empty(shape, *args, **kwargs):
        if type(shape) is tuple and len(shape) == 2 and shape == (16, 768):
            calls.append(shape)
            raise MemoryError('test-only native packed allocation')
        return real_empty(shape, *args, **kwargs)
    require_entry()
    # Fault-only eligibility override allows injection at the new allocation;
    # ordinary replacement guards are tested separately without bypasses.
    monkeypatch.setattr(facade, '_callbacks', lambda ignored: True)
    monkeypatch.setattr(np, 'empty', injected_empty)
    monkeypatch.setattr(support, '_standard_split_bindings', lambda: True)
    actual = feature._feature_gradient(*args)
    assert calls == [(16, 768)] and actual.tobytes() == expected.tobytes()


def test_fresh_scalar_gradient_does_not_import_new_facade_or_binary():
    root = str(Path(feature.__file__).resolve().parents[1])
    code = '''import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import dwd._linear_feature_diagnostics as feature
names=('dwd._ordered_gradient_reduction','dwd._ordered_reduction_accel')
assert all(name not in sys.modules for name in names)
X=np.arange(4098,dtype=float).reshape(2049,2)/8192
z=np.ones(2049)/4096
alpha=np.zeros(2049)
actual=feature._feature_gradient(X,z,alpha,.25)
expected=np.array([feature._fsum((X[:,i]*z).tolist()) for i in range(2)])
assert actual.tobytes()==expected.tobytes()
assert all(name not in sys.modules for name in names)
print('scalar-exact-no-ordered-import')
'''
    kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if sys.platform == 'win32':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    child = psutil.Popen([sys.executable, '-I', '-B', '-c', code, root], **kwargs)
    identity = (child.pid, child.create_time())
    stdout, stderr = child.communicate(timeout=60)
    assert child.returncode == 0, (identity, stderr)
    assert stdout.strip() == 'scalar-exact-no-ordered-import'


def test_cached_miss_bypasses_callback_entry_and_plane_inspection(monkeypatch):
    values = planes()
    monkeypatch.delitem(sys.modules, 'dwd._ordered_reduction_accel', raising=False)
    imports = []
    def missing(name):
        imports.append(name)
        raise ImportError('deliberately absent optional binary')
    monkeypatch.setattr(facade.importlib, 'import_module', missing)
    assert facade.try_reduce(feature, values) is None
    assert facade._LOAD_ATTEMPTED is True and facade._ACCEL is None
    monkeypatch.setattr(facade, '_callbacks', NoHooks())
    monkeypatch.setattr(facade, '_native_entry', NoHooks())
    results, allocations = observe_calls(
        lambda: [facade.try_reduce(NoHooks(), NoHooks()) for _ in range(3)], np.empty)
    assert results == [None, None, None] and not allocations
    assert imports == ['dwd._ordered_reduction_accel']


def test_first_absent_attempt_still_qualifies_before_one_import(monkeypatch):
    values = planes()
    monkeypatch.delitem(sys.modules, 'dwd._ordered_reduction_accel', raising=False)
    imports = []
    def missing(name):
        imports.append(name)
        raise ImportError('deliberately absent optional binary')
    monkeypatch.setattr(facade.importlib, 'import_module', missing)
    with monkeypatch.context() as changed:
        changed.setattr(feature, '_fsum', NoHooks())
        assert facade.try_reduce(feature, values) is None
        assert not imports and facade._LOAD_ATTEMPTED is False
    assert facade.try_reduce(feature, values) is None
    assert imports == ['dwd._ordered_reduction_accel']
    monkeypatch.setattr(facade, '_callbacks', NoHooks())
    assert facade.try_reduce(feature, values) is None
    assert imports == ['dwd._ordered_reduction_accel']


def test_positive_native_cache_keeps_callback_validation_and_actual_dispatch(monkeypatch):
    values = planes()
    expected = reference(values)
    entry = require_entry()
    module = facade._ACCEL
    with monkeypatch.context() as changed:
        changed.setattr(feature, '_fsum', NoHooks())
        result, calls = observe_calls(lambda: facade.try_reduce(feature, values), entry)
        assert result is None and not calls
        assert facade._ACCEL is module and facade._ENTRY is entry
    result, calls = observe_calls(lambda: facade.try_reduce(feature, values), entry)
    assert len(calls) == 1 and result.tobytes() == expected.tobytes()


def test_cached_miss_replays_original_checked_summation_after_late_callback_change(monkeypatch):
    X, z, alpha, twice_penalty = contraction()
    penalty_mantissa, penalty_exponent = math.frexp(twice_penalty)
    alpha_mantissa, alpha_exponent = np.frexp(alpha)
    z_mantissa, z_exponent = np.frexp(z)
    reg_high, reg_low = feature._products(np.full(len(alpha), penalty_mantissa), alpha_mantissa)
    prepared = (z_mantissa, z_exponent, reg_high, reg_low, alpha_exponent, penalty_exponent)
    splits = support.repeated_operand_splits((z_mantissa, reg_high, reg_low), feature._products)
    assert splits is not None
    columns = X[:, :16].T
    expected = np.array([feature._gradient_column(column, prepared) for column in columns])
    monkeypatch.delitem(sys.modules, 'dwd._ordered_reduction_accel', raising=False)
    def missing(name):
        raise ImportError('deliberately absent optional binary')
    monkeypatch.setattr(facade.importlib, 'import_module', missing)
    assert facade.try_reduce(feature, planes()) is None
    checked = feature._fsum
    lengths = []
    def recorded_checked(terms):
        assert type(terms) is list
        lengths.append(len(terms))
        return checked(terms)
    # Deliberately mutate only after valid split preparation. A cached miss
    # changes no callback semantics: the literal original reduction still runs.
    monkeypatch.setattr(feature, '_fsum', recorded_checked)
    monkeypatch.setattr(facade, '_callbacks', NoHooks())
    actual = feature._gradient_tile(columns, prepared, _splits=splits)
    assert lengths == [6*len(z)]*len(columns)
    assert actual.tobytes() == expected.tobytes()
