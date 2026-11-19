"""Bounded owned-process cleanup controls against the selected package tree."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

import psutil
import pytest

from _residual_probe_python import probe_python


def _require_native():
    from dwd import _compiled_residual as compiled
    if compiled._ACCEL is None:
        pytest.skip('Optional native residual extension unavailable; actual-native control cannot run')


def _require_sentinel():
    from dwd import _residual_startup_cleanup as cleanup
    capability = cleanup._qualified_startup()
    if capability is None or capability[1] != 'sentinel':
        pytest.skip('This raw-sentinel interruption control requires a supported sentinel runtime')


def _run_probe(case):
    import dwd
    source_root = Path(dwd.__file__).resolve().parent.parent
    executable, environment = probe_python()
    command = [executable, '-B', str(Path(__file__).with_name('fixtures')/'residual_cleanup_probe.py'),
               '--source-root', str(source_root), '--case', case]
    environment['PYTHONPATH'] = os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p and Path(p).is_dir())
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=environment, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    process = psutil.Process(child.pid)
    creation = process.create_time()
    handshake = queue.Queue()
    reader = threading.Thread(target=lambda: handshake.put(child.stdout.readline()), daemon=True)
    reader.start()
    try:
        first = handshake.get(timeout=15)
        reader.join(2)
        assert not reader.is_alive()
        started = json.loads(first)
        assert started == dict(pid=child.pid, creation_time=creation)
        stdout, stderr = child.communicate('GO\n', timeout=20)
        assert child.returncode == 0, (stdout, stderr)
        rows = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        result_rows = [row['result'] for row in rows if 'result' in row]
        assert len(result_rows) == 1, (stdout, stderr)
        assert not process.is_running()
        return dict(command=command, child_pid=child.pid, child_creation_time=creation,
                    actual_exit=child.returncode,
                    owned_child_killed=False, owned_child_closed=True, stderr=stderr,
                    result=result_rows[0])
    finally:
        if child.poll() is None:
            assert process.create_time() == creation
            child.kill()
            child.communicate(timeout=10)
        reader.join(2)


@pytest.mark.skipif(sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 12),
                    reason='Exact trace-control windows qualified for CPython3.12')
@pytest.mark.parametrize('case', ['declined_observe', 'declined_exit'])
def test_wakes_declined_worker_without_claiming_drain(case, record_property):
    evidence = _run_probe(case)
    result = evidence['result']
    assert result['primary_preserved'] and result['cancellation_published']
    assert result['scope_context_reset'] and result['shutdown_flag']
    assert result['bootstrap_unready_at_propagation']
    assert result['owned_native_at_propagation']
    assert result['test_bootstrap_release_after_propagation']
    assert not result['omitted_registered'] and result['owned_native_remaining'] == []
    receipt = result['cleanup_receipts'][0]
    assert receipt['shutdown_requested'] is True
    assert receipt['qualified'] is False and receipt['drained'] is False
    assert receipt['repeated_interrupts'] == 1
    assert receipt['startup_readiness_wait']['outcome'] == 'expired'
    assert not result['rescue_performed']
    record_property('ir27_evidence', json.dumps(evidence, sort_keys=True))


@pytest.mark.skipif(sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 12),
                    reason='Native trace/completion controls qualified for CPython3.12')
@pytest.mark.parametrize('mode', ['native_startup_ready', 'native_per_call', 'native_reused'])
def test_native_interrupt_preserves_primary_and_proves_completion(mode, record_property):
    _require_native()
    evidence = _run_probe(mode)
    result = evidence['result']
    assert result['primary_preserved'] and not result['partial_output_published']
    assert result['cancellation_published'] and result['native_calls'] >= 1
    assert result['foreign_thread_survived'] and result['owned_native_remaining'] == []
    assert result['scope_handle_retired'] and result['scope_context_reset']
    assert result['completion_interrupted_while_native_alive']
    receipt = result['cleanup_receipts'][0]
    assert receipt['drained'] and receipt['qualified'] and receipt['shutdown_requested']
    assert receipt['repeated_interrupts'] == 2 and not receipt['registry_mutation']
    record_property('ir27_evidence', json.dumps(evidence, sort_keys=True))


def test_scope_exit_defers_cleanup_interrupt_until_completion(record_property):
    _require_sentinel()
    evidence = _run_probe('scope_exit_interrupt')
    result = evidence['result']
    assert result['original_cleanup_interrupt_preserved']
    assert result['completion_interrupted_while_native_alive']
    assert result['scope_handle_retired'] and result['scope_context_reset']
    assert result['owned_native_remaining'] == []
    receipt = result['cleanup_receipts'][0]
    assert receipt['drained'] and receipt['repeated_interrupts'] == 1
    record_property('ir27_evidence', json.dumps(evidence, sort_keys=True))


@pytest.mark.parametrize('mode', ['normal_reused', 'serial_retire', 'budget_retire'])
def test_normal_native_scope_and_retirement(mode, record_property):
    _require_native()
    evidence = _run_probe(mode)
    result = evidence['result']
    assert result['exact_output_equality']
    assert result['scope_handle_retired'] and result['scope_context_reset']
    assert result['owned_native_remaining'] == []
    assert all(r['drained'] and r['repeated_interrupts'] == 0 for r in result['cleanup_receipts'])
    record_property('ir27_evidence', json.dumps(evidence, sort_keys=True))


def test_unqualified_completion_retains_exact_scope_handle(record_property):
    evidence = _run_probe('unqualified_scope')
    result = evidence['result']
    assert result['unproved_drain_raised'] and result['exact_handle_retained']
    assert result['registry_unchanged'] and result['scope_context_reset']
    assert result['native_exit_independently_observed']
    receipt = result['cleanup_receipts'][0]
    assert receipt['shutdown_requested'] and not receipt['drained']
    record_property('ir27_evidence', json.dumps(evidence, sort_keys=True))
