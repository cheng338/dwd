"""Finite native-cap, exactness, cold-start and fault-only scheduling checks.

The scheduling injection tests do NOT claim genuine native acceptance. Real
bridge/facade tests use the built artifact and unchanged definition tokens.
Whole-estimator public-state and performance comparisons belong to the root's
separate uninstrumented fixed-work panel, not these unit checks.
"""
from hashlib import sha256
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np
import psutil
import pytest

from _feature_expansion_test_support import (
    arithmetic, facade, feature, genuine, literal, layout, prepare, reference,
    native_bundle,
)

# The shipped native ABI has a fixed, bounded entry budget.
CAP = 16384


@pytest.fixture(scope='module')
def native_required():
    # Optional source-only builds skip native-required checks, matching the
    # shared genuine fixture. Native wheel qualification separately requires
    # the artifact and successful route acceptance.
    import importlib.util
    if importlib.util.find_spec('dwd._feature_expansion_accel') is None:
        pytest.skip('This build has no feature-expansion extension.')


def beta_prepare(alpha):
    mantissa, exponent = np.frexp(alpha)
    return (mantissa, exponent), arithmetic._split_operand(mantissa)


def beta_bundle(context, columns, prep, split):
    output = np.full(len(columns), -912.25)
    arrays = (np.array(columns, order='C', copy=True), *split, prep[1], output)
    return (context, tuple(memoryview(value) for value in arrays)), arrays, output


def run_direct(genuine, route, columns, z, alpha, penalty):
    module, context = genuine
    if route == 'gradient':
        prep, splits = prepare(z, alpha, penalty)
        bundle, arrays, output = native_bundle(context, columns, prep, splits)
        entry = module.gradient_from_splits
        calculate = lambda: reference._gradient_tile(columns, prep)
        public = lambda: facade.try_gradient(feature, columns, prep, splits)
    else:
        prep, split = beta_prepare(alpha)
        bundle, arrays, output = beta_bundle(context, columns, prep, split)
        entry = module.beta_from_split
        calculate = lambda: reference._beta_tile(columns, *prep)
        public = lambda: facade.try_beta(feature, columns, prep, split)
    before = [value.tobytes() for value in arrays]
    error = None
    try:
        expected = calculate()
    except (FloatingPointError, OverflowError) as exc:
        error = exc
    status = entry(bundle)
    actual = public()
    assert before[:-1] == [value.tobytes() for value in arrays[:-1]]
    if error is None:
        assert status == module.OK and type(actual) is np.ndarray
        assert output.tobytes() == actual.tobytes() == expected.tobytes()
        assert actual.flags.owndata and not np.shares_memory(actual, columns)
    else:
        assert status != module.OK and actual is None and output.tobytes() == before[-1]
    return status, output


@pytest.mark.parametrize('route', ('gradient', 'beta'))
@pytest.mark.parametrize('shape', ((2, 1), (64, 32), (64, CAP//64),
    (CAP//1024, 1024), (65, 1), (2, 1025), (64, CAP//64+1), (1, 1), (2, 0)))
def test_native_caps_and_atomic_refusal(native_required, genuine, route, shape):
    module, context = genuine
    assert (module.API_VERSION, module.MAX_ROWS, module.MAX_WIDTH, module.MAX_ENTRIES) == (2, 64, 1024, CAP)
    columns = np.ones(shape)
    if route == 'gradient':
        prep, split = prepare(np.full(shape[1], .5), np.zeros(shape[1]), .25)
        bundle, arrays, output = native_bundle(context, columns, prep, split)
        entry = module.gradient_from_splits
    else:
        prep, split = beta_prepare(np.full(shape[1], .5))
        bundle, arrays, output = beta_bundle(context, columns, prep, split)
        entry = module.beta_from_split
    before = [value.tobytes() for value in arrays]
    status = entry(bundle)
    accepted = 2 <= shape[0] <= 64 and 0 < shape[1] <= 1024 and shape[0]*shape[1] <= CAP
    if accepted:
        assert status == module.OK and output.tobytes() == np.full(shape[0], shape[1]*.5).tobytes()
    else:
        assert status == module.ARGUMENT and output.tobytes() == before[-1]
    assert before[:-1] == [value.tobytes() for value in arrays[:-1]]


@pytest.mark.parametrize('beta', (False, True))
@pytest.mark.parametrize('n', (32, 128, 256, 512, 1024, 1025, 2049))
def test_real_warm_hint_respects_all_caps(native_required, genuine, beta, n):
    width = feature._tile_columns(n, 137)
    actual = feature._warm_native_batch_columns(n, 137, width, beta=beta)
    expected = width if width <= 1 or n > 1024 else width*(min(64, CAP//n, 137)//width)
    assert actual == expected
    if actual > width:
        assert actual % width == 0 and actual <= 64 and n <= 1024 and actual*n <= CAP
    assert feature._warm_native_batch_columns(n, width, width, beta=beta) == width


@pytest.mark.parametrize('route', ('gradient', 'beta'))
@pytest.mark.parametrize('name', ('dyadic', 'half-even', 'subnormal-final',
    'restored-underflow', 'restored-erasure', 'overflow', 'nonfinite'))
@pytest.mark.parametrize('arrangement', ('C', 'reversed', 'readonly'))
def test_larger_literal_bits_and_error_atomicity(native_required, genuine, route, name, arrangement):
    columns, z, alpha, penalty = literal(name)
    # Repetition preserves each operation's literal operand values and order.
    # Use z as beta coefficients so its edge literals remain exercised.
    alpha = z.copy() if route == 'beta' else alpha
    columns = np.tile(columns, (CAP//512, 32))
    z, alpha = np.tile(z, 32), np.tile(alpha, 32)
    columns = layout(columns, arrangement)
    before = columns.tobytes(), z.tobytes(), alpha.tobytes()
    run_direct(genuine, route, columns, z, alpha, penalty)
    assert before == (columns.tobytes(), z.tobytes(), alpha.tobytes())


@pytest.mark.parametrize('route', ('gradient', 'beta'))
@pytest.mark.parametrize('failure', ('nonfinite', 'overflow'))
def test_failed_late_row_publishes_no_prefix(native_required, genuine, route, failure):
    rows, width = CAP//256, 256
    columns = np.ones((rows, width))
    columns[-1, -1] = math.inf if failure == 'nonfinite' else sys.float_info.max
    if failure == 'overflow':
        columns[-1, -2] = sys.float_info.max
    z, alpha = np.ones(width), np.ones(width)
    status, _ = run_direct(genuine, route, columns, z, alpha, .25)
    assert status != genuine[0].OK


@pytest.mark.parametrize('route', ('gradient', 'beta'))
@pytest.mark.parametrize('mutation', ('tile-code', 'tile-defaults', 'numpy-policy', 'numpy-alias'))
def test_large_facade_refuses_late_mutation(native_required, genuine, monkeypatch, route, mutation):
    columns = np.ones((CAP//256, 256))
    z, alpha = np.ones(256), np.ones(256)
    prep, split = prepare(z, alpha, .25) if route == 'gradient' else beta_prepare(alpha)
    tile = feature._gradient_tile if route == 'gradient' else feature._beta_tile
    context = facade._CONTEXT
    def forbidden(*args, **kwargs):
        raise AssertionError('Replaced callback must not execute')
    with monkeypatch.context() as patch:
        if mutation == 'tile-code': patch.setattr(tile, '__code__', forbidden.__code__)
        elif mutation == 'tile-defaults': patch.setattr(tile, '__defaults__', (None,))
        elif mutation == 'numpy-alias': patch.setattr(np, 'frexp', forbidden)
        else:
            old = np.seterr(over='call')
        try:
            result = (facade.try_gradient(feature, columns, prep, split) if route == 'gradient'
                      else facade.try_beta(feature, columns, prep, split))
            assert result is None and facade._CONTEXT is context
        finally:
            if mutation == 'numpy-policy': np.seterr(**old)


@pytest.mark.parametrize('route', ('gradient', 'beta'))
def test_cap_sized_finalizer_after_precheck_hits_genuine_guard(native_required, genuine, monkeypatch, route):
    columns = np.ones((CAP//256, 256))
    z, alpha = np.full(256, .5), np.full(256, .25)
    prep, split = prepare(z, alpha, .25) if route == 'gradient' else beta_prepare(alpha)
    expected = (reference._gradient_tile(columns, prep) if route == 'gradient'
                else reference._beta_tile(columns, *prep))
    module, context = genuine
    original_frexp, original_view = np.frexp, memoryview
    observed, acquired = [], []
    class Victim:
        def __del__(self):
            observed.append('finalized-after-precheck')
            np.frexp = object()
    victim = Victim()
    victim.cycle = victim
    pending = [victim]
    del victim
    def acquire(value):
        # Same explicit preparation hook as the existing small-tile regression.
        # It replaces no numerical callback or guard, and the genuine C entry
        # still receives the original immutable context after all exports.
        if pending:
            pending.clear()
            gc.collect()
        acquired.append((value, value.tobytes()))
        return original_view(value)
    monkeypatch.setattr(facade, 'memoryview', acquire, raising=False)
    assert facade._ENTRY is module.gradient_from_splits
    assert facade._BETA_ENTRY is module.beta_from_split
    try:
        result = (facade.try_gradient(feature, columns, prep, split) if route == 'gradient'
                  else facade.try_beta(feature, columns, prep, split))
    finally:
        np.frexp = original_frexp
    assert observed == ['finalized-after-precheck'] and result is None
    assert len(acquired) == (16 if route == 'gradient' else 7)
    assert all(value.tobytes() == before for value, before in acquired)
    assert facade._CONTEXT is context
    # Restoring the real callback makes the same context usable, without a
    # new context, new loader, skipped guard, or replacement native entry.
    actual = (facade.try_gradient(feature, columns, prep, split) if route == 'gradient'
              else facade.try_beta(feature, columns, prep, split))
    assert actual.tobytes() == expected.tobytes() and facade._CONTEXT is context


def schedule_fixture():
    n, p = 256, 2*(CAP//256)+19
    features = np.ones((n, p))
    features[0] = np.arange(1., p+1.)
    return features, np.full(n, .5), np.full(n, .25), .25


@pytest.mark.parametrize('route', ('gradient', 'beta'))
@pytest.mark.parametrize('failure', ('none', 'memory', 'floating', 'overflow', 'unexpected', 'success'))
def test_fault_only_group_replay_and_disable_scope(monkeypatch, route, failure):
    features, z, alpha, penalty = schedule_fixture()
    width = feature._tile_columns(*features.shape)
    assert width == 8
    events = []
    attempts = []
    class Marker(RuntimeError):
        pass
    def forced_hint(n, remaining, oldwidth, *, beta=False):
        # Explicit fault injection: not a witness of genuine native admission.
        return oldwidth*(min(64, CAP//n, remaining)//oldwidth)
    def injected(columns, prepared, split):
        start = int(columns[0, 0])-1
        large = len(columns) > width
        events.append((start, len(columns), large))
        if large:
            attempts.append(start)
            if failure == 'none': return None
            if failure == 'memory': raise MemoryError('Injected optional larger allocation')
            if failure == 'floating': raise FloatingPointError('Injected optional range refusal')
            if failure == 'overflow': raise OverflowError('Injected optional range refusal')
            if failure == 'unexpected': raise Marker('Do not mask an unexpected error')
        return (reference._gradient_tile(columns, prepared) if route == 'gradient'
                else reference._beta_tile(columns, *prepared))
    monkeypatch.setattr(feature, '_warm_native_batch_columns', forced_hint)
    monkeypatch.setattr(feature, '_try_fused_gradient' if route == 'gradient' else '_try_fused_beta', injected)
    call = (lambda: feature._feature_gradient(features, z, alpha, penalty)) if route == 'gradient' else (
            lambda: feature._feature_coefficients(features, alpha))
    if failure == 'unexpected':
        with pytest.raises(Marker, match='Do not mask'):
            call()
        assert events == [(0, width, False), (width, min(64, CAP//256), True)]
        return
    expected = (reference._feature_gradient(features, z, alpha, penalty) if route == 'gradient'
                else reference._feature_coefficients(features, alpha))
    assert call().tobytes() == expected.tobytes()
    assert events[0] == (0, width, False)
    assert attempts and attempts[0] == width
    if failure != 'success':
        assert attempts == [width]
        # Every old tile, including the refused group's first tile, is replayed
        # once in order. Optional MemoryError does not disable original tiles.
        original = [(start, min(width, features.shape[1]-start), False)
                    for start in range(0, features.shape[1], width)
                    if min(width, features.shape[1]-start) > 1]
        assert [event for event in events if not event[2]] == original
    else:
        positions = [index for start, length, _ in events for index in range(start, start+length)]
        assert positions == list(range(features.shape[1]))


@pytest.mark.parametrize('route', ('gradient', 'beta'))
@pytest.mark.parametrize('shape', ((1025, 19), (2049, 1), (256, 1)))
def test_oversize_width_and_scalar_preserve_original(route, shape):
    features = np.ones(shape)
    alpha, z = np.full(shape[0], .25), np.full(shape[0], .5)
    oldwidth = feature._tile_columns(*shape)
    assert feature._warm_native_batch_columns(shape[0], shape[1], oldwidth, beta=(route == 'beta')) == oldwidth
    # The helper may be consulted for n>1024 when oldwidth>1; it must decline.
    # No mutation of callbacks is needed to compare the actual public result.
    actual = (feature._feature_gradient(features, z, alpha, .25) if route == 'gradient'
              else feature._feature_coefficients(features, alpha))
    expected = (reference._feature_gradient(features, z, alpha, .25) if route == 'gradient'
                else reference._feature_coefficients(features, alpha))
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('route', ('gradient', 'beta'))
@pytest.mark.parametrize('first_bad', (1, 9))
def test_fault_only_refused_group_preserves_first_error(monkeypatch, route, first_bad):
    features, z, alpha, penalty = schedule_fixture()
    z[:] = 1.; alpha[:] = 1.
    # The earlier column has a summation overflow; a later column has a
    # nonfinite product. A large vector attempt must not change precedence.
    features[:, first_bad] = sys.float_info.max
    features[:, first_bad+8] = math.inf
    if route == 'gradient': alpha[:] = 0.
    original_call = (lambda: reference._feature_gradient(features, z, alpha, penalty)) if route == 'gradient' else (
                    lambda: reference._feature_coefficients(features, alpha))
    with pytest.raises((FloatingPointError, OverflowError)) as expected:
        original_call()
    def forced_hint(n, remaining, oldwidth, *, beta=False):
        return oldwidth*(min(64, CAP//n, remaining)//oldwidth)
    def injected(columns, prepared, split):
        if len(columns) > 8:
            raise FloatingPointError('Later group failure must not replace original earlier error')
        return (reference._gradient_tile(columns, prepared) if route == 'gradient'
                else reference._beta_tile(columns, *prepared))
    monkeypatch.setattr(feature, '_warm_native_batch_columns', forced_hint)
    monkeypatch.setattr(feature, '_try_fused_gradient' if route == 'gradient' else '_try_fused_beta', injected)
    call = (lambda: feature._feature_gradient(features, z, alpha, penalty)) if route == 'gradient' else (
           lambda: feature._feature_coefficients(features, alpha))
    with pytest.raises(type(expected.value)) as actual:
        call()
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize('state', ('absent', 'unknown-module', 'unknown-context', 'cold', 'no-beta-entry'))
def test_size_hint_is_hook_free_and_never_initializes(monkeypatch, state):
    class Trap:
        def __getattribute__(self, name):
            raise AssertionError('Unknown state hook executed')
    name = 'dwd._feature_expansion'
    module = types.ModuleType(name)
    module._CONTEXT = (None,)*6
    module._BETA_ENTRY = None
    if state == 'absent': monkeypatch.delitem(sys.modules, name, raising=False)
    elif state == 'unknown-module': monkeypatch.setitem(sys.modules, name, Trap())
    else:
        module._CONTEXT = Trap() if state == 'unknown-context' else None if state == 'cold' else (None,)*6
        monkeypatch.setitem(sys.modules, name, module)
    assert feature._warm_native_batch_columns(256, 64, 8, beta=True) == 8


CHILD = r'''
import importlib.util, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(1, sys.argv[2])
import numpy as np
from dwd import _linear_feature_diagnostics as feature
import _feature_expansion_reference as reference
mode = sys.argv[3]
x = np.ones((256, 145)); alpha = np.full(256, .25); z = np.full(256, .5)
name = 'dwd._feature_expansion'
if mode == 'cold-beta':
    before = dict(sys.modules)
    assert name not in before or sys.modules[name]._CONTEXT is None
    result = feature._feature_coefficients(x, alpha)
    assert result.tobytes() == reference._feature_coefficients(x, alpha).tobytes()
    module = sys.modules.get(name)
    assert module is None or (module._CONTEXT is None and module._LOAD_ATTEMPTED is False)
else:
    from dwd import _feature_expansion as facade
    from dwd import _ordered_gradient_reduction as ordered
    # Complete the existing ordered loader without changing its state manually.
    ordered.try_reduce(feature, tuple(np.zeros((2, 8)) for _ in range(6)))
    if mode == 'old-artifact':
        from dwd import _feature_expansion_accel as old
        assert old.API_VERSION == 1 and old.MAX_ENTRIES == 2048
        from hashlib import sha256
        assert sha256(pathlib.Path(old.__file__).read_bytes()).hexdigest() == sys.argv[4]
    else:
        assert importlib.util.find_spec('dwd._feature_expansion_accel') is None
    result = feature._feature_gradient(x, z, alpha, .25)
    assert result.tobytes() == reference._feature_gradient(x, z, alpha, .25).tobytes()
    assert facade._ENTRY is None and facade._CONTEXT is None
    assert feature._feature_coefficients(x, alpha).tobytes() == reference._feature_coefficients(x, alpha).tobytes()
print(json.dumps({'mode': mode, 'passed': True, 'source': feature.__file__}), flush=True)
'''


@pytest.mark.parametrize('mode', ('cold-beta', 'missing-artifact', 'old-artifact'))
def test_fresh_process_cold_and_old_artifact(tmp_path, mode):
    old_path = os.environ.get('WARM_BATCH_OLD_ARTIFACT')
    old_sha = os.environ.get('WARM_BATCH_OLD_ARTIFACT_SHA256')
    if mode == 'old-artifact':
        if old_path is None and old_sha is None:
            pytest.skip('No pinned historical native artifact supplied for this optional compatibility check.')
        assert old_path and old_sha, 'Historical artifact path and SHA256 must be supplied together'
    source = Path(feature.__file__).resolve().parent
    child_root = tmp_path/'source'
    def ignored(directory, names):
        return [name for name in names if name == '__pycache__' or name.endswith('.pyc')
                or (mode != 'cold-beta' and name.startswith('_feature_expansion_accel'))]
    shutil.copytree(source, child_root/'dwd', ignore=ignored)
    old_sha = old_sha if mode == 'old-artifact' else ''
    if mode == 'old-artifact':
        old = Path(old_path).resolve(strict=True)
        assert sha256(old.read_bytes()).hexdigest() == old_sha
        assert old.name.startswith('_feature_expansion_accel')
        shutil.copy2(old, child_root/'dwd'/old.name)
    script = tmp_path/'child.py'
    script.write_text(CHILD, encoding='utf-8')
    support = Path(reference.__file__).resolve().parent
    command = [sys.executable, '-I', str(script), str(child_root), str(support), mode, old_sha]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    owner = psutil.Process(process.pid)
    creation = owner.create_time()
    try:
        stdout, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        if owner.is_running() and owner.create_time() == creation:
            process.terminate()
        stdout, stderr = process.communicate(timeout=30)
        pytest.fail('Fresh-process test exceeded 180 seconds: '+stderr)
    receipt = dict(pid=process.pid, creation_time=creation, argv=command,
                   direct_exit=process.returncode, stdout=stdout, stderr=stderr)
    (tmp_path/'CHILD-RESULT.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    assert process.returncode == 0, stdout+'\n'+stderr
    assert json.loads(stdout.splitlines()[-1])['passed'] is True
