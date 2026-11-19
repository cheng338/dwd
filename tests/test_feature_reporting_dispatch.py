"""Exact report-specific native reuse and original callback/refusal contracts."""
from contextlib import contextmanager
from fractions import Fraction
import importlib
import importlib.util
from pathlib import Path
import struct
import platform
import sysconfig
import sys
import types

import numpy as np
import pytest

import dwd
from dwd import _compiled_residual as native
from dwd import _compiled_scores as compiled
from dwd import _feature_report_optimization as support
from dwd import _linear_feature_diagnostics as feature

NATIVE = ('full' if importlib.util.find_spec('dwd._residual_accel') is not None
          else 'nores')
POLICY = '_native_feature_reporting_available'
REPORT_NATIVE_DOMAIN = (sys.implementation.name == 'cpython'
    and sys.platform in ('win32', 'linux') and sys.version_info[:2] >= (3, 11)
    and sys.maxsize > 2**32 and platform.machine().lower() in ('amd64', 'x86_64')
    and sysconfig.get_config_var('Py_GIL_DISABLED') in (None, 0)
    and (sys.float_info.radix, sys.float_info.mant_dig, sys.float_info.max_exp) == (2, 53, 1024))


@pytest.fixture(autouse=True)
def genuine_native_graph():
    if NATIVE == 'full':
        assert type(native._ACCEL) is types.ModuleType
        for name in ('score_rows', 'score_affine_rows'):
            entry = native._ACCEL.__dict__.get(name)
            assert type(entry) is types.BuiltinFunctionType and entry.__self__ is native._ACCEL
    else:
        assert native._ACCEL is None


def same(actual, expected):
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes(order='C') == expected.tobytes(order='C')
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            same(left, right)
    else:
        assert struct.pack('!d', float(actual)) == struct.pack('!d', float(expected))


def case(kind='ordinary', layout='C'):
    matrix = ((np.arange(9*17).reshape(9, 17) % 19)-9).astype(float)/16
    beta = ((np.arange(17) % 5)-2).astype(float)/8
    offset = .125
    if kind == 'cancellation':
        matrix[:, :3] = (2.**52, 1., -2.**52)
        matrix[:, 3:] = 0.
        beta[:] = 1.
        offset = -.5
    elif kind == 'signed_zero':
        matrix[:] = -0.
        beta[:] = 1.
        offset = -0.
    elif kind == 'subnormal':
        matrix[:] = 0.
        matrix[:, 0] = np.nextafter(0., 1.)
        beta[:] = 1.
        offset = 0.
    elif kind == 'overflow':
        matrix[:] = 0.
        matrix[:, :3] = (np.finfo(float).max, np.finfo(float).max, -np.finfo(float).max)
        beta[:] = 1.
    if layout == 'F':
        matrix = np.asfortranarray(matrix)
    elif layout == 'strided':
        wider = np.empty((len(matrix), 2*matrix.shape[1]))
        wider[:, ::2] = matrix
        matrix = wider[:, ::2]
    return matrix, beta, offset


def separate(matrix, beta, offset):
    return (feature.compensated_kernel_matvec(matrix, beta),
            feature.affine_scores(matrix, beta, offset, policy='compensated'))


def outcome(call):
    try:
        return ('returned', call())
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return ('refused', type(exc), str(exc))


def same_outcome(actual, expected):
    assert actual[0] == expected[0]
    if actual[0] == 'returned':
        same(actual[1], expected[1])
    else:
        assert actual[1:] == expected[1:]


@contextmanager
def witness():
    rows = []
    previous = sys.getprofile()
    accel = native._ACCEL
    def observe(frame, event, arg):
        owner = frame.f_globals.get('__name__')
        name = frame.f_code.co_name
        if event in ('call', 'return') and owner in (
                'dwd._feature_report_optimization', 'dwd._compiled_scores',
                'dwd._compiled_affine_scores') and name in (
                    '_fused_pair', 'try_expanded_dense_rows'):
            rows.append((event, owner, name, arg is not None if event == 'return' else None))
        elif (event == 'c_call' and type(arg) is types.BuiltinFunctionType
              and arg.__self__ is accel and arg.__name__ in ('score_rows', 'score_affine_rows')):
            rows.append((event, 'native', arg.__name__, None))
    try:
        sys.setprofile(observe)
        yield rows
    finally:
        sys.setprofile(previous)
    assert sys.getprofile() is previous


def called(rows, owner, name, event='call'):
    return any(row[:3] == (event, owner, name) for row in rows)


@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
@pytest.mark.parametrize('kind', ['ordinary', 'cancellation', 'signed_zero', 'subnormal', 'overflow'])
def test_exact_original_output_and_first_refusal(kind, layout):
    matrix, beta, offset = case(kind, layout)
    before = matrix.tobytes(), beta.tobytes(), matrix.strides
    expected = outcome(lambda: separate(matrix, beta, offset))
    actual = outcome(lambda: feature._feature_scores_and_margins(matrix, beta, offset))
    same_outcome(actual, expected)
    assert before == (matrix.tobytes(), beta.tobytes(), matrix.strides)
    if kind in ('ordinary', 'cancellation'):
        raw = np.array([float(sum((Fraction.from_float(float(x))*Fraction.from_float(float(a))
                                  for x, a in zip(row, beta)), Fraction())) for row in matrix])
        joint = np.array([float(sum((Fraction.from_float(float(x))*Fraction.from_float(float(a))
                                    for x, a in zip(row, beta)), Fraction())
                                + Fraction.from_float(offset)) for row in matrix])
        same(actual[1], (raw, joint))


@pytest.mark.skipif(NATIVE == 'full' and not REPORT_NATIVE_DOMAIN,
                    reason='Native route expectation requires the qualified platform')
def test_real_route_and_native_acceptance():
    matrix, beta, offset = case()
    expected = separate(matrix, beta, offset)
    with witness() as rows:
        actual = feature._feature_scores_and_margins(matrix, beta, offset)
    same(actual, expected)
    fused = called(rows, 'dwd._feature_report_optimization', '_fused_pair')
    if NATIVE == 'nores':
        assert fused and ('return', 'dwd._feature_report_optimization', '_fused_pair', True) in rows
        assert not any(row[1] == 'native' for row in rows)
    else:
        assert not fused
        for module, entry in (('dwd._compiled_scores', 'score_rows'),
                              ('dwd._compiled_affine_scores', 'score_affine_rows')):
            assert called(rows, 'native', entry, 'c_call')
            assert ('return', module, 'try_expanded_dense_rows', True) in rows


def test_missing_binary_preserves_portable_sharing(monkeypatch):
    matrix, beta, offset = case()
    monkeypatch.setattr(native, '_ACCEL', None)
    expected = separate(matrix, beta, offset)
    with witness() as rows:
        actual = feature._feature_scores_and_margins(matrix, beta, offset)
    same(actual, expected)
    assert called(rows, 'dwd._feature_report_optimization', '_fused_pair')
    assert ('return', 'dwd._feature_report_optimization', '_fused_pair', True) in rows
    assert not any(row[1] == 'native' for row in rows)


@pytest.mark.parametrize('which', ['raw', 'affine'])
def test_changed_callback_keeps_whole_operation_order(monkeypatch, which):
    matrix, beta, offset = case()
    events = []
    raw, affine = feature.compensated_kernel_matvec, feature.affine_scores
    marker = RuntimeError('callback exception object')
    def raw_hook(*args):
        events.append('raw')
        if which == 'raw':
            raise marker
        return raw(*args)
    def affine_hook(*args, **kwargs):
        events.append('affine')
        assert kwargs == {'policy': 'compensated'}
        if which == 'affine':
            raise marker
        return affine(*args, **kwargs)
    monkeypatch.setattr(feature, 'compensated_kernel_matvec', raw_hook)
    monkeypatch.setattr(feature, 'affine_scores', affine_hook)
    with pytest.raises(RuntimeError) as caught:
        feature._feature_scores_and_margins(matrix, beta, offset)
    assert caught.value is marker
    assert events == (['raw'] if which == 'raw' else ['raw', 'affine'])


class UnknownPolicy:
    def __getattribute__(self, name):
        if name in ('__code__', '__defaults__', '__kwdefaults__'):
            raise AssertionError('unknown policy metadata was inspected')
        return object.__getattribute__(self, name)
    def __call__(self, *args, **kwargs):
        raise AssertionError('unknown policy was called')


@pytest.mark.parametrize('replacement', ['object', 'function'])
def test_pre_feature_import_policy_replacement_is_not_blessed(monkeypatch, replacement):
    matrix, beta, offset = case()
    expected = separate(matrix, beta, offset)
    def forbidden(*args, **kwargs):
        raise AssertionError('preimport replacement policy was called')
    monkeypatch.setattr(support, POLICY, UnknownPolicy() if replacement == 'object' else forbidden)
    name = 'dwd._feature_cost_fresh_test'
    spec = importlib.util.spec_from_file_location(name, Path(feature.__file__))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    same(module._feature_scores_and_margins(matrix, beta, offset), expected)


@pytest.mark.parametrize('mode', ['call', 'log'])
def test_numpy_callback_policy_retains_original_path(mode):
    matrix, beta, offset = case()
    class Sink:
        def __call__(self, *args):
            raise AssertionError('ordinary inputs should not emit a floating callback')
        def write(self, message):
            raise AssertionError('ordinary inputs should not emit a floating log')
    old = np.geterrcall()
    try:
        np.seterrcall(Sink())
        with np.errstate(all=mode):
            expected = separate(matrix, beta, offset)
            with witness() as rows:
                actual = feature._feature_scores_and_margins(matrix, beta, offset)
            same(actual, expected)
            assert not called(rows, 'dwd._feature_report_optimization', '_fused_pair')
            assert not any(row[1] == 'native' for row in rows)
    finally:
        np.seterrcall(old)


@pytest.mark.skipif(NATIVE != 'full' or not REPORT_NATIVE_DOMAIN,
                    reason='Native reporting dispatch contract')
def test_gradient_contraction_is_fresh_and_exact(monkeypatch):
    matrix, beta, _ = case()
    expected = feature.compensated_kernel_matvec(matrix, beta)
    with witness() as rows:
        actual = feature._feature_reporting_matvec(matrix, beta)
    same(actual, expected)
    assert called(rows, 'native', 'score_rows', 'c_call')
    marker = RuntimeError('late gradient callback')
    def changed(*args):
        raise marker
    monkeypatch.setattr(feature, 'compensated_kernel_matvec', changed)
    with pytest.raises(RuntimeError) as caught:
        feature._feature_reporting_matvec(matrix, beta)
    assert caught.value is marker


@pytest.mark.parametrize('change', ['body', 'defaults', 'unknown_module', 'custom_module', 'none_module'])
@pytest.mark.skipif(NATIVE != 'full' or not REPORT_NATIVE_DOMAIN,
                    reason='Native reporting dispatch contract')
def test_cached_affine_mutation_declines_new_route(monkeypatch, change):
    matrix, beta, offset = case()
    expected = separate(matrix, beta, offset)
    module = importlib.import_module('dwd._compiled_affine_scores')
    function = module.try_expanded_dense_rows
    if change == 'body':
        def forbidden(*args, **kwargs):
            raise AssertionError('mutated cached native wrapper was executed')
        code = forbidden.__code__.replace(co_name=function.__code__.co_name,
                                          co_qualname=function.__code__.co_qualname,
                                          co_filename=function.__code__.co_filename)
        monkeypatch.setattr(function, '__code__', code)
    elif change == 'defaults':
        monkeypatch.setattr(function, '__kwdefaults__', {'reporting': True})
    else:
        class CacheTrap:
            def __getattribute__(self, name):
                if name in ('__spec__', '__dict__'):
                    raise AssertionError('unknown cached module was inspected')
                return object.__getattribute__(self, name)
        class CustomModule(types.ModuleType):
            def __getattribute__(self, name):
                if name in ('__spec__', '__dict__'):
                    raise AssertionError('custom cached module was inspected')
                return super().__getattribute__(name)
        value = (CacheTrap() if change == 'unknown_module' else
                 CustomModule('custom') if change == 'custom_module' else None)
        monkeypatch.setitem(sys.modules, 'dwd._compiled_affine_scores', value)
    with witness() as rows:
        actual = feature._feature_scores_and_margins(matrix, beta, offset)
    same(actual, expected)
    assert not called(rows, 'native', 'score_affine_rows', 'c_call')


@pytest.mark.skipif(NATIVE != 'full' or not REPORT_NATIVE_DOMAIN,
                    reason='Native reporting dispatch contract')
def test_raw_exception_precedes_lazy_affine_import(monkeypatch):
    module_name = 'dwd._compiled_affine_scores'
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    # Controlled raw exception, after selecting the existing reporting adapter.
    marker = RuntimeError('raw fails before any affine import')
    def fail(*args):
        raise marker
    monkeypatch.setattr(feature, '_feature_reporting_available', lambda kind: True)
    monkeypatch.setattr(support._scores, '_reporting_kernel_matvec', fail)
    with pytest.raises(RuntimeError) as caught:
        feature._feature_scores_and_margins(*case())
    assert caught.value is marker and module_name not in sys.modules


@pytest.mark.parametrize('failure', ['refusal', 'unexpected', 'allocation'])
@pytest.mark.skipif(NATIVE != 'full' or not REPORT_NATIVE_DOMAIN,
                    reason='Native reporting dispatch contract')
def test_existing_native_failure_contract_with_explicit_guard_bypass(monkeypatch, failure):
    matrix, beta, _ = case()
    expected = feature.compensated_kernel_matvec(matrix, beta)
    marker = RuntimeError('native unexpected object')
    calls = []
    def fault(*args):
        calls.append('native')
        if failure == 'unexpected':
            raise marker
        return 1
    # This fault test deliberately bypasses admission, not native arithmetic.
    monkeypatch.setattr(feature, '_feature_reporting_available', lambda kind: True)
    if failure != 'allocation':
        monkeypatch.setattr(native._ACCEL, 'score_rows', fault)
    else:
        original = np.empty
        fired = []
        def empty(shape, *args, **kwargs):
            if isinstance(shape, tuple) and len(shape) == 2 and not fired:
                fired.append(True)
                raise MemoryError('native scratch allocation')
            return original(shape, *args, **kwargs)
        monkeypatch.setattr(np, 'empty', empty)
    if failure == 'unexpected':
        with pytest.raises(RuntimeError) as caught:
            feature._feature_reporting_matvec(matrix, beta)
        assert caught.value is marker and calls
    else:
        same(feature._feature_reporting_matvec(matrix, beta), expected)
        if failure == 'refusal':
            assert calls
        else:
            assert fired == [True]


@pytest.mark.skipif(NATIVE != 'full' or not REPORT_NATIVE_DOMAIN,
                    reason='Native reporting dispatch contract')
def test_late_affine_callback_is_seen_after_raw_return(monkeypatch):
    matrix, beta, offset = case()
    marker = RuntimeError('late affine callback object')
    events = []
    def changed(*args, **kwargs):
        events.append(kwargs)
        assert kwargs == {'policy': 'compensated'}
        raise marker
    previous = sys.getprofile()
    def observe(frame, event, arg):
        if (event == 'return' and frame.f_globals.get('__name__') == 'dwd._kernel_scores'
                and frame.f_code.co_name == '_reporting_kernel_matvec'):
            monkeypatch.setattr(feature, 'affine_scores', changed)
    try:
        sys.setprofile(observe)
        with pytest.raises(RuntimeError) as caught:
            feature._feature_scores_and_margins(matrix, beta, offset)
        assert caught.value is marker
    finally:
        sys.setprofile(previous)
    assert events == [{'policy': 'compensated'}]


@pytest.mark.skipif(NATIVE != 'full' or not REPORT_NATIVE_DOMAIN, reason='Native-presence policy refusal boundary')
@pytest.mark.parametrize('change', ['body', 'defaults', 'module'])
def test_supported_policy_mutation_keeps_original_pair(monkeypatch, change):
    matrix, beta, offset = case()
    events = []
    function = compiled._supported
    if change == 'body':
        original = types.FunctionType(function.__code__, function.__globals__,
                                      function.__name__, function.__defaults__)
        original.__kwdefaults__ = dict(function.__kwdefaults__)
        monkeypatch.setitem(function.__globals__, '_cost_policy_original', original)
        monkeypatch.setitem(function.__globals__, '_cost_policy_events', events)
        def changed(*args, **kwargs):
            _cost_policy_events.append('policy')
            return _cost_policy_original(*args, **kwargs)
        code = changed.__code__.replace(co_name=function.__code__.co_name,
                                       co_qualname=function.__code__.co_qualname,
                                       co_filename=function.__code__.co_filename)
        monkeypatch.setattr(function, '__code__', code)
    elif change == 'defaults':
        monkeypatch.setattr(function, '__kwdefaults__', {'reporting': True})
    else:
        replacement = types.SimpleNamespace(implementation=sys.implementation,
                                            platform='linux', version_info=sys.version_info,
                                            maxsize=sys.maxsize, float_info=sys.float_info)
        monkeypatch.setattr(compiled, 'sys', replacement)
    expected = separate(matrix, beta, offset)
    expected_events = list(events)
    events.clear()
    with witness() as rows:
        actual = feature._feature_scores_and_margins(matrix, beta, offset)
    same(actual, expected)
    assert events == expected_events
    assert not called(rows, 'dwd._feature_report_optimization', '_fused_pair')
    assert not called(rows, 'native', 'score_affine_rows', 'c_call')
