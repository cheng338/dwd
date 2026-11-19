"""Fresh-process regressions for ordered-loader/fused-context sequencing.

Use the selected DWD package and the helpers shipped with this test module.
Each case owns a fresh interpreter so earlier fixtures cannot prewarm its state.
The process boundary and monitoring are test-only.
"""
from contextlib import suppress
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil
import pytest


_CHILD = r'''
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import types

package_root = Path(sys.argv[1])
tests_dir = Path(sys.argv[2])
sys.path[:0] = [str(tests_dir), str(package_root)]
import numpy as np
from _feature_expansion_test_support import (
    arithmetic, facade, feature, literal, native_bundle, prepare, reference,
)
from dwd import _ordered_gradient_reduction as ordered

assert Path(feature.__file__).resolve().parent.parent == package_root
assert Path(sys.modules['_feature_expansion_test_support'].__file__).resolve().parent == tests_dir
assert Path(reference.__file__).resolve() == tests_dir / '_feature_expansion_reference.py'
assert sys.getprofile() is None and sys.gettrace() is None
assert facade._CONTEXT is None
assert ordered._LOAD_ATTEMPTED is False
assert getattr(ordered, '_LOAD_COMPLETE', False) is False
assert ordered._ACCEL is None and ordered._ENTRY is None
columns, z, alpha, penalty = literal('dyadic')
prepared, splits = prepare(z, alpha, penalty)
expected = reference._gradient_tile(columns, prepared)
planes = tuple(np.zeros(columns.shape) for _ in range(6))
scenario = sys.argv[3]


def warm_ordered():
    result = ordered.try_reduce(feature, planes)
    assert type(result) is np.ndarray
    assert result.tobytes() == np.zeros(len(columns)).tobytes()
    assert ordered._LOAD_ATTEMPTED is True
    assert ordered._LOAD_COMPLETE is True
    assert ordered._ACCEL is not None and ordered._ENTRY is not None


def accepted_gradient_and_beta():
    result = facade.try_gradient(feature, columns, prepared, splits)
    assert type(result) is np.ndarray, 'Settled gradient must use genuine fusion'
    assert result.tobytes() == expected.tobytes()
    context = facade._CONTEXT
    assert context is not None
    mantissa, exponent = np.frexp(alpha)
    split = arithmetic._split_operand(mantissa)
    beta = facade.try_beta(feature, columns, (mantissa, exponent), split)
    assert type(beta) is np.ndarray, 'Settled beta must use genuine fusion'
    assert beta.tobytes() == reference._beta_tile(columns, mantissa, exponent).tobytes()
    assert feature._feature_gradient(columns.T, z, alpha, penalty).tobytes() == reference._feature_gradient(columns.T, z, alpha, penalty).tobytes()
    assert feature._feature_coefficients(columns.T, alpha).tobytes() == reference._feature_coefficients(columns.T, alpha).tobytes()
    assert facade._CONTEXT is context
    return context


if scenario == 'cold-preimport':
    # The old module exists, but its optional native loader has not run.
    # The baseline creates a permanently stale context at this first call.
    assert facade.try_gradient(feature, columns, prepared, splits) is None
    assert facade._CONTEXT is None and facade._LOAD_ATTEMPTED is False
    first = feature._gradient_tile(columns, prepared, _splits=splits)
    assert first.tobytes() == expected.tobytes()
    assert ordered._LOAD_COMPLETE is True
    assert facade._CONTEXT is None
    accepted_gradient_and_beta()

elif scenario == 'warm-first':
    warm_ordered()
    accepted_gradient_and_beta()

elif scenario in ('missing-import', 'missing-oserror'):
    original = importlib.import_module
    calls = []
    error = ImportError if scenario == 'missing-import' else OSError
    def missing(name, *args, **kwargs):
        if name == ordered._MODULE_NAME:
            calls.append(name)
            raise error('Explicit absent ordered artifact')
        return original(name, *args, **kwargs)
    importlib.import_module = missing
    try:
        assert ordered.try_reduce(feature, planes) is None
        assert ordered._LOAD_ATTEMPTED is True and ordered._LOAD_COMPLETE is True
        assert ordered._ACCEL is None and ordered._ENTRY is None
        assert ordered.try_reduce(feature, planes) is None
        assert len(calls) == 1
    finally:
        importlib.import_module = original
    # A completed negative cache is terminal; it is not an in-flight load.
    accepted_gradient_and_beta()
    assert ordered._ACCEL is None and ordered._ENTRY is None

elif scenario in ('negative-cached-none', 'negative-cached-module', 'negative-unsupported'):
    original = importlib.import_module
    def forbidden_old_import(name, *args, **kwargs):
        if name == ordered._MODULE_NAME:
            raise AssertionError('Cached/rejected old state must not invoke importer')
        return original(name, *args, **kwargs)
    saved_gil_disabled = ordered._GIL_DISABLED
    if scenario == 'negative-cached-none':
        sys.modules[ordered._MODULE_NAME] = None
    elif scenario == 'negative-cached-module':
        # Exact module type but missing the required ABI/entry metadata.
        sys.modules[ordered._MODULE_NAME] = types.ModuleType(ordered._MODULE_NAME)
    else:
        ordered._GIL_DISABLED = 1
    importlib.import_module = forbidden_old_import
    try:
        assert ordered._native_entry(feature) is None
        assert ordered._LOAD_ATTEMPTED is True and ordered._LOAD_COMPLETE is True
        assert ordered._ACCEL is None and ordered._ENTRY is None
        assert ordered._native_entry(feature) is None
    finally:
        importlib.import_module = original
        ordered._GIL_DISABLED = saved_gil_disabled
    accepted_gradient_and_beta()
    assert ordered._ACCEL is None and ordered._ENTRY is None

elif scenario == 'reentrant-import':
    original = importlib.import_module
    calls = []
    def reentrant(name, *args, **kwargs):
        if name == ordered._MODULE_NAME:
            calls.append(name)
            assert ordered._LOAD_ATTEMPTED is True
            assert ordered._LOAD_COMPLETE is False
            assert facade.try_gradient(feature, columns, prepared, splits) is None
            assert facade._CONTEXT is None and facade._LOAD_ATTEMPTED is False
            assert ordered._native_entry(feature) is None
            assert ordered._LOAD_COMPLETE is False
            nested = feature._gradient_tile(columns, prepared, _splits=splits)
            assert nested.tobytes() == expected.tobytes()
            assert facade._CONTEXT is None
        return original(name, *args, **kwargs)
    importlib.import_module = reentrant
    try:
        warm_ordered()
        assert len(calls) == 1
    finally:
        importlib.import_module = original
    accepted_gradient_and_beta()

elif scenario == 'concurrent-inflight':
    original = importlib.import_module
    entered, release = threading.Event(), threading.Event()
    result, errors = [], []
    def blocked_import(name, *args, **kwargs):
        if name == ordered._MODULE_NAME:
            entered.set()
            assert release.wait(30), 'Test did not release the blocked importer'
        return original(name, *args, **kwargs)
    def load_in_thread():
        try:
            result.append(ordered.try_reduce(feature, planes))
        except BaseException as error:
            errors.append(error)
    importlib.import_module = blocked_import
    worker = threading.Thread(target=load_in_thread)
    worker.start()
    try:
        assert entered.wait(30), 'Loader never entered the controlled import'
        assert ordered._LOAD_ATTEMPTED is True and ordered._LOAD_COMPLETE is False
        assert ordered._ACCEL is None and ordered._ENTRY is None
        assert facade.try_gradient(feature, columns, prepared, splits) is None
        assert facade._CONTEXT is None and facade._LOAD_ATTEMPTED is False
        assert ordered._native_entry(feature) is None
        assert ordered._LOAD_COMPLETE is False
        assert feature._gradient_tile(columns, prepared, _splits=splits).tobytes() == expected.tobytes()
        assert facade._CONTEXT is None
    finally:
        release.set()
        worker.join(30)
        importlib.import_module = original
    assert not worker.is_alive(), 'Owned loader thread did not terminate'
    assert not errors, repr(errors)
    assert len(result) == 1 and type(result[0]) is np.ndarray
    assert result[0].tobytes() == np.zeros(len(columns)).tobytes()
    assert ordered._LOAD_ATTEMPTED is True and ordered._LOAD_COMPLETE is True
    accepted_gradient_and_beta()

elif scenario in ('interrupted-runtime', 'interrupted-keyboard', 'interrupted-systemexit'):
    original = importlib.import_module
    calls = []
    marker = (RuntimeError('Unexpected import interruption')
              if scenario == 'interrupted-runtime'
              else KeyboardInterrupt('Interrupted ordered import')
              if scenario == 'interrupted-keyboard'
              else SystemExit('SystemExit during ordered import'))
    def interrupted(name, *args, **kwargs):
        if name == ordered._MODULE_NAME:
            calls.append(name)
            assert ordered._LOAD_ATTEMPTED is True and ordered._LOAD_COMPLETE is False
            assert facade.try_gradient(feature, columns, prepared, splits) is None
            assert facade._CONTEXT is None
            raise marker
        return original(name, *args, **kwargs)
    importlib.import_module = interrupted
    try:
        try:
            ordered._native_entry(feature)
        except BaseException as caught:
            assert caught is marker
        else:
            raise AssertionError('Unexpected import exception was suppressed')
    finally:
        importlib.import_module = original
    assert len(calls) == 1
    assert ordered._LOAD_ATTEMPTED is True and ordered._LOAD_COMPLETE is False
    assert ordered._ACCEL is None and ordered._ENTRY is None
    assert ordered._native_entry(feature) is None
    assert ordered._LOAD_COMPLETE is False
    assert facade.try_gradient(feature, columns, prepared, splits) is None
    assert facade._CONTEXT is None and facade._LOAD_ATTEMPTED is False
    assert feature._gradient_tile(columns, prepared, _splits=splits).tobytes() == expected.tobytes()
    assert feature._feature_coefficients(columns.T, alpha).tobytes() == reference._feature_coefficients(columns.T, alpha).tobytes()
    assert facade._CONTEXT is None

elif scenario.startswith('invalid-completion-'):
    warm_ordered()
    state = scenario[len('invalid-completion-'):]
    hook_calls = []
    class ForeignTruthy:
        def __bool__(self):
            hook_calls.append('bool')
            raise AssertionError('Completion state must not invoke truthiness hooks')
        def __eq__(self, other):
            hook_calls.append('eq')
            raise AssertionError('Completion state must not invoke equality hooks')
    if state == 'missing':
        del ordered._LOAD_COMPLETE
    elif state == 'false':
        ordered._LOAD_COMPLETE = False
    elif state == 'one':
        ordered._LOAD_COMPLETE = 1
    elif state == 'foreign':
        ordered._LOAD_COMPLETE = ForeignTruthy()
    else:
        raise AssertionError('Unknown completion state: ' + state)
    assert facade.try_gradient(feature, columns, prepared, splits) is None
    assert facade._CONTEXT is None and facade._LOAD_ATTEMPTED is False
    assert feature._gradient_tile(columns, prepared, _splits=splits).tobytes() == expected.tobytes()
    assert facade._CONTEXT is None
    mantissa, exponent = np.frexp(alpha)
    split = arithmetic._split_operand(mantissa)
    assert facade.try_beta(feature, columns, (mantissa, exponent), split) is None
    assert hook_calls == []

elif scenario.startswith('mutate-'):
    warm_ordered()
    context = accepted_gradient_and_beta()
    key = scenario[len('mutate-'):]
    replacement = False if key in ('_LOAD_ATTEMPTED', '_LOAD_COMPLETE') else None
    setattr(ordered, key, replacement)
    bundle, arrays, output = native_bundle(context, columns, prepared, splits)
    before = tuple(value.tobytes() for value in arrays)
    assert facade._ACCEL.gradient_from_splits(bundle) == facade._ACCEL.GUARD
    assert tuple(value.tobytes() for value in arrays) == before
    assert facade.try_gradient(feature, columns, prepared, splits) is None
    mantissa, exponent = np.frexp(alpha)
    split = arithmetic._split_operand(mantissa)
    assert facade.try_beta(feature, columns, (mantissa, exponent), split) is None
    assert facade._CONTEXT is context, 'Existing immutable context must never be rebased'

else:
    raise AssertionError('Unknown test scenario: ' + scenario)

print(json.dumps({'scenario': scenario, 'pid': os.getpid(), 'ppid': os.getppid(), 'passed': True}))
'''


_SCENARIOS = (
    'cold-preimport', 'warm-first', 'missing-import', 'missing-oserror',
    'negative-cached-none', 'negative-cached-module', 'negative-unsupported',
    'reentrant-import', 'concurrent-inflight', 'interrupted-runtime',
    'interrupted-keyboard', 'interrupted-systemexit',
    'invalid-completion-missing', 'invalid-completion-false',
    'invalid-completion-one', 'invalid-completion-foreign',
    'mutate-_LOAD_ATTEMPTED', 'mutate-_LOAD_COMPLETE', 'mutate-_ACCEL',
    'mutate-_ENTRY',
)


@pytest.mark.parametrize('scenario', _SCENARIOS)
def test_ordered_loader_completion_in_fresh_process(scenario):
    package = importlib.util.find_spec('dwd')
    assert package is not None and package.origin is not None
    package_root = Path(package.origin).resolve().parent.parent
    tests_dir = Path(__file__).resolve().parent
    for module in ('dwd._feature_expansion_accel', 'dwd._ordered_reduction_accel'):
        if importlib.util.find_spec(module) is None:
            pytest.skip('This fresh-process native regression requires both optional artifacts')
    environment = os.environ.copy()
    # These tiny correctness cases run serially; avoid nested native pools.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '1'
    windows_venv = os.name == 'nt' and sys.prefix != sys.base_prefix
    with psutil.Popen(
        [sys.executable, '-I', '-B', '-c', _CHILD, str(package_root), str(tests_dir), scenario],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding='utf-8', errors='replace', env=environment,
    ) as child:
        identity = (child.pid, child.create_time())
        try:
            stdout, stderr = child.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            # A Windows venv launcher can leave its interpreter holding both pipes.
            with suppress(psutil.NoSuchProcess):
                if windows_venv:
                    for interpreter in child.children():
                        with suppress(psutil.NoSuchProcess):
                            interpreter.kill()
                child.kill()
            stdout, stderr = child.communicate(timeout=10)
            pytest.fail(f'{scenario} timed out for {identity}:\n{stdout}\n{stderr}')
        assert child.returncode == 0, (
            f'{scenario} failed for {identity} with exit {child.returncode}:\n'
            f'{stdout}\n{stderr}'
        )
        result = json.loads(stdout.strip().splitlines()[-1])
        assert result['scenario'] == scenario and result['passed'] is True
        # Windows venvs can launch Python through a waiting redirector.
        assert result['pid'] == child.pid or (windows_venv and result['ppid'] == child.pid), result
