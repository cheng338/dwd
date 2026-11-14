"""Bounded readiness and completion controls against the selected package."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

import psutil
import pytest


pytestmark = pytest.mark.skipif(
    sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 12),
    reason='Exact bootstrap/trace seams are primary CPython3.12 controls')


def run_probe(case, budget=10., delay=2.5):
    import dwd
    root = Path(dwd.__file__).resolve().parent.parent
    assert (root/'dwd'/'__init__.py').is_file(), str(root)
    executable = getattr(sys, '_base_executable', None) or sys.executable
    command = [executable, '-B', str(Path(__file__).with_name('fixtures')/'residual_readiness_probe.py'),
               '--source-root', str(root), '--case', case,
               '--budget-ms', str(budget), '--delay-ms', str(delay)]
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PYTHONPATH'] = os.pathsep.join(
        str(Path(p).resolve()) for p in sys.path if p and Path(p).is_dir())
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    process = psutil.Process(child.pid)
    creation = process.create_time()
    handshake = queue.Queue()
    reader = threading.Thread(target=lambda: handshake.put(child.stdout.readline()), daemon=True)
    reader.start()
    try:
        first = handshake.get(timeout=20)
        reader.join(2)
        assert not reader.is_alive()
        assert json.loads(first) == dict(pid=child.pid, creation_time=creation)
        stdout, stderr = child.communicate('GO\n', timeout=30)
        rows = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        results = [row['result'] for row in rows if 'result' in row]
        assert not process.is_running()
        return dict(command=command, child_pid=child.pid, child_creation_time=creation,
                    actual_exit=child.returncode, owned_child_closed=True,
                    owned_child_killed=False, stdout=stdout, stderr=stderr,
                    result=results[0] if len(results) == 1 else None)
    finally:
        if child.poll() is None:
            assert process.create_time() == creation
            child.kill()
            child.communicate(timeout=10)
        reader.join(2)


def checked(record_property, case, budget=10., delay=2.5):
    evidence = run_probe(case, budget, delay)
    # Preserve raw completed child output even if a later semantic check fails.
    record_property('residual_readiness_evidence', json.dumps(evidence, sort_keys=True))
    assert evidence['actual_exit'] == 0, evidence
    row = evidence['result']
    assert row is not None and row['case'] == case and row['implementation'] == 'selected_package'
    assert row['native_remaining'] == [] and row['scope_context_reset']
    if case == 'ordinary':
        return row
    assert row['primary_preserved'] and row['retained_traceback']
    assert row['cancellation_published'] and not row['partial_output_published']
    assert row['foreign_thread_survived'] and not row['omitted_registered']
    assert not row['test_rescue_wake']
    assert row['containment_release_at'] >= row['propagation_at']
    assert row['whole_call_seconds'] >= 0
    assert row['cleanup_receipt_count'] == len(row['cleanup_receipts']) >= 1
    assert len(row['cleanup_receipts']) == len(row['cleanup_calls'])
    if row['deadline_snapshots']:
        assert row['events'].index('shutdown-requested') < row['events'].index('startup-readiness-wait')
    for receipt, call in zip(row['cleanup_receipts'], row['cleanup_calls']):
        assert receipt['shutdown_requested'] and not receipt['registry_mutation']
        assert receipt['qualified'] is receipt['drained']
        assert call['end'] >= call['start']
        if receipt['drained']:
            assert call['native_remaining'] == []
        wait = receipt['startup_readiness_wait']
        if wait['attempted']:
            assert wait['eligible'] and wait['budget_seconds'] == budget / 1000
            assert wait['elapsed_seconds'] >= 0
            if wait['outcome'] == 'ready':
                assert wait['elapsed_seconds'] < wait['budget_seconds']
                assert receipt['drained']
            elif wait['outcome'] == 'expired':
                assert wait['elapsed_seconds'] >= wait['budget_seconds']
                assert not receipt['drained']
            else:
                assert wait['outcome'] == 'unproved' and not receipt['drained']
        else:
            assert wait == dict(eligible=False, attempted=False, budget_seconds=None,
                                outcome='not_eligible', elapsed_seconds=None)
    if not row['cleanup_receipts'][-1]['drained']:
        assert not row['at_propagation']['whole_completion_cached']
    return row


@pytest.mark.parametrize('budget', [1., 10., 50.])
@pytest.mark.parametrize('position,multiplier', [('below', .25), ('boundary', 1.), ('above', 2.)])
def test_fixed_readiness_grid(budget, position, multiplier, record_property):
    row = checked(record_property, 'delayed', budget, budget * multiplier)
    wait = row['cleanup_receipts'][0]['startup_readiness_wait']
    assert wait['attempted'] and wait['outcome'] in ('ready', 'expired')
    assert len(row['deadline_snapshots']) == 1
    # Requested delay is not an acceptance oracle: observed scheduling and the
    # frozen absolute deadline determine the result, including boundary cases.


@pytest.mark.parametrize('budget', [1., 10., 50.])
def test_held_readiness_expires(budget, record_property):
    row = checked(record_property, 'never', budget)
    assert row['cleanup_receipts'][0]['startup_readiness_wait']['outcome'] == 'expired'
    assert row['at_propagation']['omitted_native_alive']


STRUCTURAL = ['ordinary', 'ready_omitted', 'prelaunch', 'ambiguous_call', 'wrong_owner',
              'capture_declined', 'capture_ready', 'source_changed', 'event_changed', 'membership_changed',
              'sentinel_unacquired', 'mixed', 'rapid_completion', 'repeated_interrupts',
              'bootstrap_failure']


@pytest.mark.parametrize('case', STRUCTURAL)
def test_ownership_capture_and_completion_edges(case, record_property):
    row = checked(record_property, case, 50. if case == 'rapid_completion' else 10.)
    receipts = row['cleanup_receipts']
    if case == 'ordinary':
        assert row['exact_outputs'] == [4, 4, 1]
        assert row['reused_same_pool'] and row['budget_retired_pool'] and row['serial_retired_pool']
        return
    wait = receipts[0]['startup_readiness_wait']
    if case in ('prelaunch', 'ambiguous_call', 'wrong_owner', 'capture_declined'):
        assert not wait['attempted'] and not receipts[0]['drained']
    elif case in ('ready_omitted', 'capture_ready'):
        assert not wait['attempted'] and receipts[0]['drained']
        assert receipts[0]['owned_workers'] == 1
    elif case in ('source_changed', 'event_changed', 'membership_changed'):
        assert wait['outcome'] == 'unproved' and not receipts[0]['drained']
    elif case in ('sentinel_unacquired', 'mixed', 'repeated_interrupts', 'bootstrap_failure'):
        assert wait['outcome'] == 'expired' and not receipts[0]['drained']
    elif case == 'rapid_completion':
        assert wait['outcome'] in ('ready', 'expired')
    if case.startswith('capture_'):
        assert row['capture_fault_fired'] and row['cleanup_calls'][0]['capture_interrupt_preserved']
        assert receipts[0]['repeated_interrupts'] == 1
    if case == 'repeated_interrupts':
        assert receipts[0]['repeated_interrupts'] == 2
        assert row['cleanup_calls'][0]['first_repeated_interrupt_preserved']
        assert len(row['deadline_snapshots']) == 3
        assert all(value == row['deadline_snapshots'][0] for value in row['deadline_snapshots'])
    if case == 'sentinel_unacquired':
        assert row['native_sentinel_snapshot'] == dict(published=True, locked=False, started=False)
        assert row['at_propagation']['omitted_native_alive']
    if case == 'bootstrap_failure':
        assert not row['at_propagation']['omitted_native_alive']
        assert 'test_bootstrap_returned_without_readiness' in row['events']


@pytest.mark.parametrize('case', ['signal_ready', 'signal_expiry'])
def test_actual_pending_interrupt_during_readiness(case, record_property):
    row = checked(record_property, case, 50., 2.5)
    assert 'interrupt_main_requested' in row['events']
    assert row['cleanup_receipts'][0]['repeated_interrupts'] >= 1
    assert row['cleanup_calls'][0]['deferred_type'] == 'KeyboardInterrupt'
    wait = row['cleanup_receipts'][0]['startup_readiness_wait']
    if case == 'signal_expiry':
        assert wait['outcome'] == 'expired' and row['at_propagation']['omitted_native_alive']
    else:
        assert wait['outcome'] in ('ready', 'expired')


@pytest.mark.parametrize('case', ['reused_ready', 'reused_expiry'])
def test_reused_scope_preserves_primary_and_reports_every_attempt(case, record_property):
    row = checked(record_property, case, 50., 2.5)
    if case == 'reused_expiry':
        assert row['cleanup_receipt_count'] == 2
        assert row['scope_handle_retained']
        assert all(r['startup_readiness_wait']['outcome'] == 'expired' for r in row['cleanup_receipts'])
        assert len(row['deadline_snapshots']) == 2
        assert row['deadline_snapshots'][1][1] >= row['deadline_snapshots'][0][2]
    else:
        assert row['cleanup_receipt_count'] in (1, 2)
    # Each private readiness wait is bounded separately. Whole propagation may
    # include both waits and actual completion; it has no claimed 50 ms bound.


@pytest.mark.parametrize('case', ['mixed', 'ordinary', 'never', 'partial_interrupt'])
def test_partial_registered_drain_keeps_aggregate_unproved(case, record_property):
    row = checked(record_property, case, 10.)
    if case == 'ordinary':
        assert all(r['drained'] for r in row['cleanup_receipts'])
        return
    receipt = row['cleanup_receipts'][0]
    partial = receipt['partial_registered_drain']
    assert not receipt['drained'] and row['at_propagation']['omitted_native_alive']
    if case == 'never':
        assert not partial['attempted'] and partial['outcome'] == 'not_eligible'
    else:
        assert partial == dict(attempted=True, workers=1, completed=True, outcome='completed')
        assert partial['completed'] is True
        assert row['at_propagation']['registered_native_alive'] == []
        assert 'partial-drained' in row['events']
    if case == 'partial_interrupt':
        assert row['completion_fault_fired']
        assert 'partial_wait_interrupted_with_native_alive' in row['events']
        assert receipt['repeated_interrupts'] == 1
