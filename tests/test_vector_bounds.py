"""Finite zero-fit checks for vectorized outward bounds and ordered fallback.

The unchanged batching suite supplies its pinned scalar oracle. Direct bound
stages are compared bitwise, and exact Fraction actions independently check the
certificate. Synthetic stage injections distinguish private refusal points;
they are not claims that inconsistent dot observations arise from a real fit.
"""
import ctypes
import functools
import gc
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import weakref

import numpy as np
import pytest

from dwd import _linear_feature_diagnostics as feature
from dwd import _quadratic_bounds as bounds
from test_feature_batching import data, exact_dot, frac, outcome, same, scalar_upper, scalar


ETA = float.fromhex('0x1p-1074')
MAX = float.fromhex('0x1.fffffffffffffp+1023')


@pytest.fixture(autouse=True)
def unchanged_policy():
    policy, callback = np.geterr(), np.geterrcall()
    yield
    assert np.geterr() == policy and np.geterrcall() is callback


def scalar_stages(value, absolute, constants):
    eps, allowance, denominator, tail = constants
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        numerator = bounds._upward_nonnegative(absolute+allowance)
        absolute_upper = bounds._upward_nonnegative(numerator/denominator)
        inner = bounds._upward_nonnegative((4*eps)*absolute_upper)
        error = bounds._upward_nonnegative(inner+tail)
    return numerator, absolute_upper, inner, error, math.nextafter(abs(value)+error, math.inf)


@pytest.mark.parametrize('layout', ['plain', 'strided', 'readonly'])
def test_array_upward_matches_scalar_at_binary64_boundaries(layout):
    values = np.array([0., -0., ETA, 2*ETA, np.finfo(float).tiny,
                       math.nextafter(np.finfo(float).tiny, 0.),
                       2.**-500, math.nextafter(1., 0.), 1., math.nextafter(1., math.inf),
                       2.**500, math.nextafter(MAX, 0.)])
    if layout == 'strided':
        storage = np.zeros(2*len(values))
        storage[::2] = values
        values = storage[::2]
    if layout == 'readonly':
        values.setflags(write=False)
    before = values.tobytes(), values.flags.writeable
    expected = np.array([bounds._upward_nonnegative(float(v)) for v in values])
    same(bounds._upward_nonnegative_array(values), expected)
    assert (values.tobytes(), values.flags.writeable) == before
    for original, upper in zip(values, expected):
        assert frac(upper) >= frac(original)


@pytest.mark.parametrize('value', [-1., -ETA, math.nan, math.inf, -math.inf, MAX])
def test_array_upward_preserves_each_scalar_refusal(value):
    expected = outcome(lambda: bounds._upward_nonnegative(value))
    assert expected[0] == 'refused'
    same(outcome(lambda: bounds._upward_nonnegative_array(np.array([0., value, 1.]))), expected)


@pytest.mark.parametrize('n,width', [(n, width) for n in (1, 2, 31, 128) for width in (1, 2, 7, 64)])
def test_all_five_bound_stages_match_original_and_exact_action(monkeypatch, n, width):
    F, _, _, signed = data(n, width, 'strided')
    with np.errstate(all='ignore'):
        products = F.T*signed
    values = np.array([math.fsum(row.tolist()) for row in products])
    absolute = np.array([math.fsum(np.abs(row).tolist()) for row in products])
    constants = feature._dual_dot_constants(n)
    expected = [scalar_stages(float(v), float(a), constants) for v, a in zip(values, absolute)]
    stages = []
    original = bounds._upward_nonnegative_array
    def observe(array):
        result = original(array)
        stages.append(result.copy())
        return result
    monkeypatch.setattr(bounds, '_upward_nonnegative_array', observe)
    error, upper = bounds._dual_coordinate_bounds(values, absolute, constants)
    assert len(stages) == 4
    for index, actual in enumerate(stages):
        same(actual, np.array([row[index] for row in expected]))
    same(error, np.array([row[3] for row in expected]))
    same(upper, np.array([row[4] for row in expected]))
    for column, value, allowance, bound in zip(F.T, values, error, upper):
        exact = exact_dot(column, signed)
        assert abs(exact-frac(value)) <= frac(allowance)
        assert abs(exact) <= frac(bound)


@pytest.mark.parametrize('kind', ['zero', 'subnormal', 'normal_boundary', 'large', 'near_max'])
def test_bound_stage_range_fixtures_are_bitwise_scalar(kind):
    magnitude = {'zero': 0., 'subnormal': ETA, 'normal_boundary': np.finfo(float).tiny,
                 'large': 2.**900, 'near_max': MAX/4}[kind]
    values = np.array([magnitude, -magnitude, 0.])
    absolute = np.array([magnitude, 2*magnitude, magnitude])
    constants = feature._dual_dot_constants(31)
    expected = [scalar_stages(float(v), float(a), constants) for v, a in zip(values, absolute)]
    error, upper = bounds._dual_coordinate_bounds(values, absolute, constants)
    same(error, np.array([row[3] for row in expected]))
    same(upper, np.array([row[4] for row in expected]))
    assert np.all(error > 0.)


@pytest.mark.parametrize('n', [1, 31, 512, 2048])
def test_deterministic_multi_exponent_finite_bound_parity(monkeypatch, n):
    rng = np.random.RandomState(7103+n)
    exponents = rng.randint(-1073, 901, size=257)
    exponents[:8] = [-1073, -1072, -1022, -1000, -500, 0, 500, 900]
    with np.errstate(under='ignore'):
        absolute = np.ldexp(rng.randint(8, 16, size=257)/8., exponents)
        values = absolute*(rng.randint(-8, 9, size=257)/8.)
    assert np.isfinite(absolute).all() and np.all(absolute >= 0.)
    constants = feature._dual_dot_constants(n)
    expected = [scalar_stages(float(v), float(a), constants) for v, a in zip(values, absolute)]
    stages = []
    original = bounds._upward_nonnegative_array
    def observe(array):
        result = original(array)
        stages.append(result.copy())
        return result
    monkeypatch.setattr(bounds, '_upward_nonnegative_array', observe)
    error, upper = bounds._dual_coordinate_bounds(values, absolute, constants)
    assert len(stages) == 4
    for index, actual in enumerate(stages):
        same(actual, np.array([row[index] for row in expected]))
    same(error, np.array([row[3] for row in expected]))
    same(upper, np.array([row[4] for row in expected]))
    assert all(frac(bound) >= abs(frac(value)) for bound, value in zip(upper, values))


@pytest.mark.parametrize('stage', range(4))
def test_each_outward_stage_refuses_finite_nextafter_overflow(monkeypatch, stage):
    constants = feature._dual_dot_constants(4)
    scalar_calls, vector_calls = [], []
    scalar_original = bounds._upward_nonnegative
    vector_original = bounds._upward_nonnegative_array
    def scalar_injection(value):
        index = len(scalar_calls)
        scalar_calls.append(value)
        return scalar_original(MAX if index == stage else value)
    def vector_injection(values):
        index = len(vector_calls)
        vector_calls.append(values.copy())
        return vector_original(np.full_like(values, MAX) if index == stage else values)
    with monkeypatch.context() as patch:
        patch.setattr(bounds, '_upward_nonnegative', scalar_injection)
        expected = outcome(lambda: scalar_stages(1., 1., constants))
    with monkeypatch.context() as patch:
        patch.setattr(bounds, '_upward_nonnegative_array', vector_injection)
        actual = outcome(lambda: bounds._dual_coordinate_bounds(np.ones(1), np.ones(1), constants))
    assert expected[0] == 'refused' and 'exceeds finite arithmetic' in expected[2]
    same(actual, expected)
    assert len(scalar_calls) == len(vector_calls) == stage+1


def test_final_coordinate_infinity_is_returned_not_outward_stage_refusal():
    # Synthetic finite observations isolate the final nextafter contract. This
    # deliberately inconsistent absolute sum is not a physical dot fixture.
    constants = feature._dual_dot_constants(2)
    expected = scalar_stages(MAX, 0., constants)
    error, upper = bounds._dual_coordinate_bounds(np.array([MAX]), np.zeros(1), constants)
    same(error, np.array([expected[3]]))
    assert upper[0] == expected[4] == math.inf


def test_stock_dispatch_really_uses_vector_bounds(monkeypatch):
    F, _, _, signed = data(2, 131)
    assert bounds._standard_feature_bound_bindings()
    calls = []
    original = bounds._dual_coordinate_bounds
    def observe(values, absolute, constants):
        calls.append(len(values))
        return original(values, absolute, constants)
    monkeypatch.setattr(bounds, '_dual_coordinate_bounds', observe)
    same(feature._dual_coordinate_upper(F, signed), scalar_upper(F, signed))
    assert calls == [64, 64, 3]


@pytest.mark.parametrize('reverse', [False, True])
def test_earlier_bound_overflow_and_later_fsum_overflow_keep_scalar_order(reverse):
    F = np.array([[MAX/2, 2.**1023], [MAX/2, 2.**1023]])
    if reverse:
        F = F[:, ::-1]
    signed = np.ones(2)
    expected = outcome(lambda: scalar_upper(F, signed))
    assert expected[0] == 'refused'
    same(outcome(lambda: feature._dual_coordinate_upper(F, signed)), expected)
    assert ('accumulation' if reverse else 'bound exceeds') in expected[2]


def test_hooked_later_fsum_valueerror_does_not_mask_earlier_bound_error(monkeypatch):
    F = np.array([[MAX/2, 1.], [MAX/2, 1.]])
    signed = np.ones(2)
    original = bounds.fsum
    def selective(values):
        if values == [1., 1.]:
            raise ValueError('later hook refusal')
        return original(values)
    monkeypatch.setattr(bounds, 'fsum', selective)
    assert not bounds._standard_feature_bound_bindings()
    expected = outcome(lambda: scalar_upper(F, signed))
    same(outcome(lambda: feature._dual_coordinate_upper(F, signed)), expected)
    assert expected[0] == 'refused' and 'bound exceeds' in expected[2]


def test_bound_allocation_failure_releases_arrays_and_disables_later_tiles(monkeypatch):
    F, _, _, signed = data(2, 131)
    calls, references = [], []
    def unavailable(values, absolute, constants):
        calls.append(len(values))
        references.extend((weakref.ref(values), weakref.ref(absolute)))
        raise MemoryError('new bound allocation unavailable')
    monkeypatch.setattr(bounds, '_dual_coordinate_bounds', unavailable)
    same(feature._dual_coordinate_upper(F, signed), scalar_upper(F, signed))
    assert calls == [64]
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize('exception', [FloatingPointError, OverflowError])
def test_numeric_bound_failure_replays_old_tile_then_resumes_vector_work(monkeypatch, exception):
    F, _, _, signed = data(2, 131)
    original = bounds._dual_coordinate_bounds
    calls = []
    def first_refuses(values, absolute, constants):
        calls.append(len(values))
        if len(calls) == 1:
            raise exception('new vector bound refusal')
        return original(values, absolute, constants)
    monkeypatch.setattr(bounds, '_dual_coordinate_bounds', first_refuses)
    same(feature._dual_coordinate_upper(F, signed), scalar_upper(F, signed))
    assert calls == [64, 64, 3]


def test_unexpected_bound_exception_escapes_unchanged(monkeypatch):
    F, _, _, signed = data(2, 7)
    sentinel = RuntimeError('unexpected bound implementation error')
    def broken(*args):
        raise sentinel
    monkeypatch.setattr(bounds, '_dual_coordinate_bounds', broken)
    with pytest.raises(RuntimeError) as caught:
        feature._dual_coordinate_upper(F, signed)
    assert caught.value is sentinel


HOOKS = ['upward', 'fsum', 'math_nextafter', 'numpy_add', 'numpy_divide',
         'numpy_multiply', 'numpy_absolute', 'numpy_abs', 'numpy_nextafter',
         'numpy_isfinite', 'numpy_less', 'errstate_class', 'errstate_enter', 'errstate_exit']


def replace_hook(patch, hook):
    if hook in ('upward', 'fsum'):
        owner, name = bounds, '_upward_nonnegative' if hook == 'upward' else 'fsum'
    elif hook == 'math_nextafter':
        owner, name = math, 'nextafter'
    elif hook.startswith('numpy_'):
        owner, name = np, hook.removeprefix('numpy_')
    elif hook == 'errstate_class':
        original = np.errstate
        class Wrapped(original):
            pass
        patch.setattr(np, 'errstate', Wrapped)
        return
    else:
        owner = np.errstate
        name = '__enter__' if hook == 'errstate_enter' else '__exit__'
    original = getattr(owner, name)
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return original(*args, **kwargs)
    patch.setattr(owner, name, wrapped)


@pytest.mark.parametrize('hook', HOOKS)
def test_postimport_dependency_hooks_keep_original_scalar_tile(monkeypatch, hook):
    F, _, _, signed = data(4, 7)
    assert bounds._standard_feature_bound_bindings()
    def forbidden(*args):
        raise AssertionError('custom dependency must not enter vector bounds')
    with monkeypatch.context() as patch:
        replace_hook(patch, hook)
        patch.setattr(feature, '_dual_dot_tile_vector', forbidden)
        assert not bounds._standard_feature_bound_bindings()
        actual = feature._dual_coordinate_upper(F, signed)
        expected = scalar_upper(F, signed)
    same(actual, expected)
    assert bounds._standard_feature_bound_bindings()


PREIMPORT = r'''
import functools, json, math, pathlib, sys
import numpy as np
import scipy, sklearn
source, hook = pathlib.Path(sys.argv[1]).resolve(), sys.argv[2]
sys.path.insert(0, str(source))
before_policy, before_callback = np.geterr(), np.geterrcall()
if hook in ('upward', 'fsum'):
    from dwd import _quadratic_bounds as owner
    name = '_upward_nonnegative' if hook == 'upward' else 'fsum'
elif hook in ('math_fsum', 'math_nextafter'):
    owner, name = math, hook.removeprefix('math_')
elif hook == 'numpy_nextafter':
    owner, name = np, 'nextafter'
elif hook == 'errstate_class':
    owner, name = np, 'errstate'
else:
    owner, name = np.errstate, '__enter__' if hook == 'errstate_enter' else '__exit__'
original = getattr(owner, name)
if hook == 'errstate_class':
    class replacement(original):
        pass
else:
    @functools.wraps(original)
    def replacement(*args, **kwargs):
        return original(*args, **kwargs)
setattr(owner, name, replacement)
try:
    from dwd import _quadratic_bounds as bounds
    from dwd import _linear_feature_diagnostics as feature
    assert not bounds._standard_feature_bound_bindings()
    def forbidden(*args):
        raise AssertionError('preimport wrapper was blessed')
    feature._dual_dot_tile_vector = forbidden
    F = np.arange(28, dtype=float).reshape(4, 7)/32
    signed = np.array([.5, -.25, .75, -.5])
    expected = []
    for column in F.T:
        value, error = bounds._compensated_dot(column, signed)
        expected.append(math.nextafter(abs(value)+error, math.inf))
    actual = feature._dual_coordinate_upper(F, signed)
    assert actual.tobytes() == np.array(expected).tobytes()
    assert pathlib.Path(bounds.__file__).resolve() == source/'dwd'/'_quadratic_bounds.py'
    assert pathlib.Path(feature.__file__).resolve() == source/'dwd'/'_linear_feature_diagnostics.py'
finally:
    setattr(owner, name, original)
assert np.geterr() == before_policy and np.geterrcall() is before_callback
print(json.dumps(dict(hook=hook, qualified_while_hooked=False, scalar_parity=True,
                     state_restored=True, fits=0, source=str(source))))
'''


@pytest.mark.parametrize('hook', ['upward', 'fsum', 'math_fsum', 'math_nextafter',
                                 'numpy_nextafter', 'errstate_class', 'errstate_enter', 'errstate_exit'])
def test_preimport_wrappers_are_not_blessed_by_captured_identity(hook):
    source = Path(bounds.__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-I', '-B', '-c', PREIMPORT, str(source), hook],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt == dict(hook=hook, qualified_while_hooked=False, scalar_parity=True,
                           state_restored=True, fits=0, source=str(source))


@pytest.mark.parametrize('mode', ['ignore', 'warn', 'raise', 'call', 'log', 'print'])
def test_vector_bounds_preserve_numpy_policy_and_callback(monkeypatch, mode, capfd):
    F, _, _, signed = data(4, 7)
    observed, vector_calls = [], []
    original = bounds._dual_coordinate_bounds
    def observe(*args):
        vector_calls.append(True)
        return original(*args)
    class Logger:
        def write(self, message):
            observed.append(message)
    callback = Logger() if mode == 'log' else lambda *args: observed.append(args)
    previous = np.geterrcall()
    monkeypatch.setattr(bounds, '_dual_coordinate_bounds', observe)
    try:
        np.seterrcall(callback)
        with np.errstate(all=mode):
            actual = feature._dual_coordinate_upper(F, signed)
            expected = scalar_upper(F, signed)
            assert np.geterr() == dict.fromkeys(('divide', 'over', 'under', 'invalid'), mode)
            assert np.geterrcall() is callback
    finally:
        np.seterrcall(previous)
    same(actual, expected)
    assert vector_calls == [True] and observed == []
    captured = capfd.readouterr()
    assert captured.out == captured.err == ''


def floating_control(mode):
    if sys.platform == 'win32':
        control = ctypes.CDLL('ucrtbase')._controlfp_s
        control.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        control.restype = ctypes.c_int
        current = ctypes.c_uint()
        assert control(ctypes.byref(current), 0, 0) == 0
        saved = current.value
        value, mask = {'down': (0x100, 0x300), 'up': (0x200, 0x300),
                       'zero': (0x300, 0x300), 'flush': (0x01000000, 0x03000000)}[mode]
        return (lambda: control(ctypes.byref(current), value, mask),
                lambda: control(ctypes.byref(current), saved, mask))
    if (sys.platform == 'linux' and mode != 'flush'
            and platform.machine().lower() in ('x86_64', 'amd64')):
        library = ctypes.CDLL(None)
        get_mode, set_mode = library.fegetround, library.fesetround
        get_mode.argtypes, get_mode.restype = [], ctypes.c_int
        set_mode.argtypes, set_mode.restype = [ctypes.c_int], ctypes.c_int
        saved = get_mode()
        assert saved >= 0
        value = {'down': 0x400, 'up': 0x800, 'zero': 0xc00}[mode]
        return lambda: set_mode(value), lambda: set_mode(saved)
    pytest.skip('Requires the audited Windows control API or Linux x86-64 rounding API')


@pytest.mark.parametrize('mode', ['down', 'up', 'zero', 'flush'])
def test_nonstandard_ieee_state_keeps_original_refusal_and_restores(mode):
    F, _, _, _ = data(4, 7)
    y, slopes = np.array([-1., 1., -1., 1.]), np.ones(4)
    assert feature._round_to_nearest()
    enter, leave = floating_control(mode)
    try:
        entered = enter()
        actual = feature._dual_endpoint(F, y, slopes, .75, 1.)
        # The frozen scalar helper was imported before changing the IEEE state.
        expected = scalar._dual_endpoint(F, y, slopes, .75, 1.)
    finally:
        restored = leave()
    assert entered == restored == 0
    same(actual, expected)
    assert actual[0] == 0. and actual[1]['fallback_reason'] == 'nonzero_feature_dual_bound_unavailable'
    assert feature._round_to_nearest()
    feature._dual_dot_constants(4)  # The original gradual-underflow check succeeds again.
