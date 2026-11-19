"""Residual worker limits use the caller's available CPUs without changing math."""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import psutil
import pytest
import dwd


def _run_child(tmp_path, mask, *, dynamic=False):
    parent = psutil.Process()
    before = parent.cpu_affinity() if hasattr(parent, 'cpu_affinity') else None
    budget = int(os.environ.get('IR51_NATIVE_BUDGET', '16' if os.name == 'nt' else '20'))
    assert 1 <= budget <= (16 if os.name == 'nt' else 20)
    output = tmp_path / ('affinity-' + mask + ('-dynamic' if dynamic else '') + '.json')
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join(str(value) for value in sys.path if value)
    command = [sys.executable, '-B', str(Path(__file__).with_name('affinity_child_probe.py')),
               '--output', str(output), '--mask', mask, '--budget', str(budget),
               '--package-root', str(Path(dwd.__file__).resolve().parent.parent)]
    if dynamic:
        command.append('--dynamic')
    finished = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert finished.returncode == 0, finished.stdout + finished.stderr
    if before is not None:
        assert parent.cpu_affinity() == before, 'The parent affinity must remain unchanged'
    result = json.loads(output.read_text(encoding='utf-8'))
    if result['status'] == 'unsupported':
        pytest.skip(result['reason'])
    assert result['status'] == 'complete'
    assert set(result['restored_affinity']) == set(result['original_affinity'])
    assert result['all_residual_pools_drained']
    return result


@pytest.mark.parametrize('mask', ['1', '2', 'available'])
def test_native_rows_respect_owned_child_affinity(tmp_path, record_property, mask):
    result = _run_child(tmp_path, mask)
    record_property('child_affinity_result', json.dumps(result, sort_keys=True))
    for record in result['records']:
        expected = max(1, min(result['observed_blas_budget'], result['system_cpu_count'] or 1,
                              len(record['affinity']), record['n'] // 1024))
        assert record['workers'] == expected
        assert record['configured_workers'] == (None if expected == 1 else expected)
        assert len(record['threads']) <= expected
        assert record['output_sha256'] == record['direct_sha256']
    initial_mask = result['records'][0]['affinity']
    for n, count in result['count_only'].items():
        assert count == max(1, min(result['observed_blas_budget'], result['system_cpu_count'] or 1,
                                   len(initial_mask), int(n) // 1024))


def test_affinity_count_changes_retire_previous_scope_pool(tmp_path, record_property):
    result = _run_child(tmp_path, 'available', dynamic=True)
    record_property('child_affinity_result', json.dumps(result, sort_keys=True))
    phases = {record['phase']: record for record in result['records'] if record['phase'] != 'initial'}
    assert phases['one']['workers'] == 1 and phases['one']['configured_workers'] is None
    if len(result['original_affinity']) >= 2:
        assert phases['two']['workers'] == min(2, result['observed_blas_budget'])
    assert phases['reexpanded']['workers'] == min(4, len(result['original_affinity']),
                                                  result['observed_blas_budget'],
                                                  result['system_cpu_count'] or 1)
    assert all(record['previous_pool_drained'] for record in result['records'])
    assert all(record['output_sha256'] == record['direct_sha256'] for record in result['records'])


@pytest.fixture
def cpu_helper():
    from dwd import _residual_cpu
    return _residual_cpu


def _cpu_os(system=16, *, affinity=None, present=True, name='posix'):
    namespace = types.SimpleNamespace(cpu_count=lambda: system, name=name)
    if present:
        namespace.sched_getaffinity = affinity if affinity is not None else lambda pid: {0, 1}
    return namespace


@pytest.mark.parametrize('system,mask,expected', [(16, {0}, 1), (16, {3, 7}, 2),
                                                (16, frozenset({2, 9}), 2),
                                                (2, set(range(8)), 2), (20, set(range(40)), 20)])
def test_available_count_clamps_live_affinity(cpu_helper, monkeypatch, system, mask, expected):
    seen = []

    def affinity(pid):
        seen.append(pid)
        return mask

    monkeypatch.setattr(cpu_helper, 'os', _cpu_os(system, affinity=affinity))
    assert cpu_helper.available_cpu_count() == expected
    assert seen == [0]


@pytest.mark.parametrize('system', [None, 0, -3, False, True, 2.5, '16'])
def test_unknown_system_count_remains_conservative_without_affinity_query(cpu_helper, monkeypatch, system):
    def forbidden(pid):
        raise AssertionError('Serial system cap needs no affinity query')

    monkeypatch.setattr(cpu_helper, 'os', _cpu_os(system, affinity=forbidden))
    assert cpu_helper.available_cpu_count() == 1


@pytest.mark.parametrize('error_type', [OSError, NotImplementedError])
def test_unavailable_system_count_selects_serial(cpu_helper, monkeypatch, error_type):
    namespace = _cpu_os()

    def failed():
        raise error_type('system count unavailable')

    namespace.cpu_count = failed
    monkeypatch.setattr(cpu_helper, 'os', namespace)
    assert cpu_helper.available_cpu_count() == 1


@pytest.mark.parametrize('error_type', [OSError, NotImplementedError, AttributeError,
                                       TypeError, ValueError, ImportError])
def test_failed_supported_affinity_query_is_serial(cpu_helper, monkeypatch, error_type):
    def failed(pid):
        raise error_type('affinity unavailable')

    monkeypatch.setattr(cpu_helper, 'os', _cpu_os(16, affinity=failed))
    assert cpu_helper.available_cpu_count() == 1


@pytest.mark.parametrize('mask', [set(), None, 3, '03', [0, 1], {-1}, {True}, {'0'}, {1.0, 2}])
def test_empty_or_malformed_affinity_result_is_serial(cpu_helper, monkeypatch, mask):
    monkeypatch.setattr(cpu_helper, 'os', _cpu_os(16, affinity=lambda pid: mask))
    assert cpu_helper.available_cpu_count() == 1


def test_platform_without_affinity_api_keeps_system_fallback(cpu_helper, monkeypatch):
    monkeypatch.setattr(cpu_helper, 'os', _cpu_os(12, present=False, name='other'))
    assert cpu_helper.available_cpu_count() == 12


@pytest.mark.parametrize('stage', ['system', 'affinity', 'windows'])
@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit, RuntimeError])
def test_interrupts_and_unexpected_errors_are_not_hidden(cpu_helper, monkeypatch, stage, error_type):
    error = error_type('do not hide this failure')

    def failed(*args):
        raise error

    namespace = _cpu_os()
    if stage == 'system':
        namespace.cpu_count = failed
    elif stage == 'affinity':
        namespace.sched_getaffinity = failed
    else:
        namespace = _cpu_os(present=False, name='nt')
        monkeypatch.setattr(cpu_helper, '_windows_affinity_count', failed)
    monkeypatch.setattr(cpu_helper, 'os', namespace)
    with pytest.raises(error_type) as caught:
        cpu_helper.available_cpu_count()
    assert caught.value is error


def test_affinity_capacity_is_requeried_after_shrink_and_expand(cpu_helper, monkeypatch):
    masks = iter([{0, 1, 2, 3}, {0}, {0, 1}, set(range(16))])
    monkeypatch.setattr(cpu_helper, 'os', _cpu_os(16, affinity=lambda pid: next(masks)))
    assert [cpu_helper.available_cpu_count() for _ in range(4)] == [4, 1, 2, 16]


@pytest.mark.parametrize('count,expected', [(2, 2), (64, 16), (0, 1), (-1, 1),
                                          (None, 1), (True, 1), (3.5, 1)])
def test_windows_query_count_is_validated_and_system_clamped(cpu_helper, monkeypatch, count, expected):
    monkeypatch.setattr(cpu_helper, 'os', _cpu_os(16, present=False, name='nt'))
    monkeypatch.setattr(cpu_helper, '_windows_affinity_count', lambda: count)
    assert cpu_helper.available_cpu_count() == expected


@pytest.mark.parametrize('error_type', [OSError, ImportError])
def test_windows_binding_failure_selects_conservative_serial(cpu_helper, monkeypatch, error_type):
    def failed():
        raise error_type('Windows affinity binding unavailable')

    monkeypatch.setattr(cpu_helper, 'os', _cpu_os(16, present=False, name='nt'))
    monkeypatch.setattr(cpu_helper, '_windows_affinity_functions', failed)
    assert cpu_helper.available_cpu_count() == 1


def test_windows_primary_group_is_only_a_conservative_cap(cpu_helper, monkeypatch):
    import ctypes

    def query(handle, process_pointer, system_pointer):
        mask = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1
        ctypes.cast(process_pointer, ctypes.POINTER(ctypes.c_size_t))[0] = mask
        ctypes.cast(system_pointer, ctypes.POINTER(ctypes.c_size_t))[0] = mask
        return 1

    monkeypatch.setattr(cpu_helper, 'os', _cpu_os(128, present=False, name='nt'))
    monkeypatch.setattr(cpu_helper, '_windows_affinity_functions', lambda: (object(), lambda: 17, query))
    assert cpu_helper.available_cpu_count() == 8 * ctypes.sizeof(ctypes.c_size_t)


@pytest.mark.parametrize('case', ['one', 'high_bits', 'empty_process', 'empty_system', 'outside_system', 'api_false'])
def test_windows_pointer_width_mask_and_api_result(cpu_helper, monkeypatch, case):
    import ctypes
    width = 8 * ctypes.sizeof(ctypes.c_size_t)
    high = 1 << (width - 1)
    masks = {'one': (1, 1, 1, 1), 'high_bits': (1 | high, 1 | high, 1, 2),
             'empty_process': (0, 3, 1, 1), 'empty_system': (1, 0, 1, 1),
             'outside_system': (2, 1, 1, 1), 'api_false': (3, 3, 0, 1)}
    process, system, success, expected = masks[case]
    calls = []

    def query(handle, process_pointer, system_pointer):
        calls.append(handle)
        ctypes.cast(process_pointer, ctypes.POINTER(ctypes.c_size_t))[0] = process
        ctypes.cast(system_pointer, ctypes.POINTER(ctypes.c_size_t))[0] = system
        return success

    monkeypatch.setattr(cpu_helper, '_windows_affinity_functions', lambda: (object(), lambda: 17, query))
    assert cpu_helper._windows_affinity_count() == expected
    assert calls == [17]


def test_windows_caches_bindings_but_reads_fresh_masks(cpu_helper, monkeypatch):
    import ctypes
    loaded, queries = [], []
    masks = iter([1, 3])

    def current_process():
        return 19

    def query(handle, process_pointer, system_pointer):
        queries.append(handle)
        ctypes.cast(process_pointer, ctypes.POINTER(ctypes.c_size_t))[0] = next(masks)
        ctypes.cast(system_pointer, ctypes.POINTER(ctypes.c_size_t))[0] = 15
        return 1

    library = types.SimpleNamespace(GetCurrentProcess=current_process, GetProcessAffinityMask=query)

    def load(name, **options):
        loaded.append((name, options))
        return library

    cpu_helper._windows_affinity_functions.cache_clear()
    monkeypatch.setattr(ctypes, 'WinDLL', load, raising=False)
    try:
        assert cpu_helper._windows_affinity_count() == 1
        assert cpu_helper._windows_affinity_count() == 2
        assert loaded == [('kernel32.dll', {'use_last_error': True, 'winmode': 0x00000800})]
        assert queries == [19, 19]
        assert current_process.restype is ctypes.c_void_p
        assert query.argtypes == (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                  ctypes.POINTER(ctypes.c_size_t))
    finally:
        cpu_helper._windows_affinity_functions.cache_clear()


@pytest.mark.parametrize('n', [0, 1, 255, 256, 1023, 1024, 2047])
def test_small_rows_do_not_discover_pools_or_cpu(n, monkeypatch):
    from dwd import _compiled_residual as compiled
    import threadpoolctl

    def forbidden(*args):
        raise AssertionError('Small residual must retain its discovery-free serial path')

    monkeypatch.setattr(threadpoolctl, 'threadpool_info', forbidden)
    monkeypatch.setattr(compiled, '_available_cpu_count', forbidden)
    assert compiled._worker_count(n) == 1


@pytest.mark.parametrize('pools', [[], [{'user_api': 'blas', 'num_threads': 1}],
                                  [{'user_api': 'openmp', 'num_threads': 16}],
                                  [{'user_api': 'blas', 'num_threads': 0}]])
def test_serial_budget_does_not_discover_affinity(pools, monkeypatch):
    from dwd import _compiled_residual as compiled
    import threadpoolctl

    def forbidden():
        raise AssertionError('Serial BLAS budget needs no CPU query')

    monkeypatch.setattr(threadpoolctl, 'threadpool_info', lambda: pools)
    monkeypatch.setattr(compiled, '_available_cpu_count', forbidden)
    assert compiled._worker_count(16384) == 1


@pytest.mark.parametrize('n,capacity,pool_budgets,expected', [
    (2048, 16, [16], 2), (3071, 16, [16], 2), (3072, 16, [16], 3),
    (8192, 2, [16], 2), (8192, 16, [8, 3], 3), (8192, 1, [16], 1),
    (16384, 40, [16], 16), (20480, 40, [20], 20),
])
def test_worker_count_intersects_rows_blas_and_available_cpus(n, capacity, pool_budgets, expected, monkeypatch):
    from dwd import _compiled_residual as compiled
    import threadpoolctl
    calls = []

    def available():
        calls.append('cpu')
        return capacity

    pools = [{'user_api': 'blas', 'num_threads': count} for count in pool_budgets]
    pools.append({'user_api': 'openmp', 'num_threads': 1})
    monkeypatch.setattr(threadpoolctl, 'threadpool_info', lambda: pools)
    monkeypatch.setattr(compiled, '_available_cpu_count', available)
    assert compiled._worker_count(n) == expected
    assert calls == ['cpu']


def test_public_kernel_fit_observes_the_restricted_child_cap(tmp_path, record_property):
    parent = psutil.Process()
    before = parent.cpu_affinity() if hasattr(parent, 'cpu_affinity') else None
    output = tmp_path / 'public-fit-affinity.json'
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join(str(value) for value in sys.path if value)
    budget = 16 if os.name == 'nt' else 20
    command = [sys.executable, '-B', str(Path(__file__).with_name('public_fit_affinity_probe.py')),
               '--output', str(output), '--budget', str(budget),
               '--package-root', str(Path(dwd.__file__).resolve().parent.parent)]
    finished = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=120,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert finished.returncode == 0, finished.stdout + finished.stderr
    if before is not None:
        assert parent.cpu_affinity() == before
    result = json.loads(output.read_text(encoding='utf-8'))
    record_property('public_fit_affinity_result', json.dumps(result, sort_keys=True))
    if result['status'] == 'unsupported':
        pytest.skip(result['reason'])
    assert result['status'] == 'complete'
    assert result['restricted_affinity'] == result['original_affinity'][:1]
    assert result['n_iter'] == 1 and result['backend'] == 'cholesky'
    assert result['feature_shape'] == [2048, 1]
    assert result['psd_validation'] == 'internally constructed PSD kernel (known family)'
    assert result['eigenvalue_check_seconds'] == 0.
    assert result['setup_blas_budget'] == 1 and result['residual_blas_budget'] == budget
    assert result['control'] == 'injected ordinary-bound unavailability; real compensated recovery'
    assert result['ordinary_bound_constructor_calls'] > 0 and result['test_hooks_restored']
    diagnostics = result['linear_system_diagnostics']
    assert diagnostics['ordinary_residual_declines'] > 0
    assert diagnostics['native_residual_attempts'] > 0
    assert diagnostics['native_residual_attempts'] == (
        diagnostics.get('native_residual_checks', 0) + diagnostics.get('native_residual_declines', 0))
    assert diagnostics['compensated_residual_checks'] > 0
    if sys.platform != 'win32' and sys.version_info[:2] == (3, 11):
        # CPython 3.11 lacks sumprod; the separate compiled qualification is Windows-only.
        assert diagnostics.get('native_residual_checks', 0) == 0
        assert result['portable_residual_calls'] == diagnostics['compensated_residual_checks']
        assert result['worker_counts'] == [] and result['dispatch_records'] == []
    else:
        assert result['worker_counts'] and all(row == [2048, 1] for row in result['worker_counts'])
        assert result['dispatch_records'] and all(
            row['affinity'] == result['restricted_affinity'] for row in result['dispatch_records'])
    assert result['input_unchanged'] and result['finite_predictions'] and result['finite_parameters']
    assert set(result['original_affinity']) == set(result['restored_affinity'])
