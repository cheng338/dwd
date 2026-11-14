"""Cleanup bookkeeping faults preserve wake, completion and primary error."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from dwd import _residual_startup_cleanup as cleanup


def _idle_owned_pool():
    capability = cleanup._qualified_startup()
    if capability is None or capability[1] != 'sentinel':
        pytest.skip('These bookkeeping controls require the supported raw-sentinel adapter')
    pool = cleanup.own_default_executor(ThreadPoolExecutor(max_workers=1))
    assert pool.submit(lambda: 42).result(timeout=5) == 42
    tokens = tuple(worker._tstate_lock for worker in pool._threads)
    assert tokens and all(token is not None for token in tokens)
    return pool, tokens


def test_binding_interrupt_still_wakes_and_drains(monkeypatch):
    pool, tokens = _idle_owned_pool()
    primary = KeyboardInterrupt('binding lookup interrupted before wake')
    original = cleanup._check_shutdown_binding
    observed = []

    def interrupted_once(executor):
        original(executor)
        if not observed:
            observed.append(executor)
            assert executor._shutdown is False
            raise primary

    try:
        with monkeypatch.context() as patch:
            patch.setattr(cleanup, '_check_shutdown_binding', interrupted_once)
            receipt, interruption = cleanup.drain_owned_executor(pool)
        assert interruption is primary
        assert observed == [pool]
        assert receipt['shutdown_requested'] and receipt['qualified'] and receipt['drained']
        assert receipt['repeated_interrupts'] == 1
        assert all(not token.locked() for token in tokens)
        assert cleanup._owned_shutdown(pool).completed(pool)
    finally:
        pool.shutdown(wait=True)


def test_completion_record_interrupt_preserves_first_and_completed_authority(monkeypatch):
    pool, tokens = _idle_owned_pool()
    adapter = cleanup._owned_shutdown(pool)
    primary = KeyboardInterrupt('first cleanup interruption')
    secondary = KeyboardInterrupt('completion record interruption')
    original_wait = cleanup._wait_completions
    original_record = adapter.record_completion
    wait_faults, record_faults = [], []

    def interrupted_wait(mechanism, captured):
        if not wait_faults:
            wait_faults.append(primary)
            raise primary
        return original_wait(mechanism, captured)

    def interrupted_record(executor):
        # Recording is reached only after actual raw-token completion.
        assert all(not token.locked() for token in tokens)
        if not record_faults:
            assert adapter.completed_workers is None
            record_faults.append(secondary)
            raise secondary
        return original_record(executor)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(cleanup, '_wait_completions', interrupted_wait)
            patch.setattr(adapter, 'record_completion', interrupted_record)
            receipt, interruption = cleanup.drain_owned_executor(pool)
        assert interruption is primary
        assert wait_faults == [primary] and record_faults == [secondary]
        assert receipt['drained'] and receipt['shutdown_requested']
        assert receipt['repeated_interrupts'] == 2
        assert adapter.completed(pool)
        # A legitimate post-completion join may erase the sentinel attribute;
        # the previously proved completion state still permits idempotent drain.
        for worker in pool._threads:
            worker.join(timeout=0)
            assert worker._tstate_lock is None
        repeated, pending = cleanup.drain_owned_executor(pool)
        assert pending is None and repeated['drained'] and repeated['previously_completed']
    finally:
        pool.shutdown(wait=True)
