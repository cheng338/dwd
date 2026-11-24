"""Owned-subprocess native cleanup and scope-retirement controls."""
from contextlib import nullcontext
import sys
import threading
import types

from residual_cleanup_probe import _line
from _residual_native_threads import (
    live_native_ids, require_native_exit, wait_native_gone, worker_completion_probe,
)


def interrupted_native(mode):
    import concurrent.futures.thread as impl
    import numpy as np
    from dwd import _compiled_residual as compiled
    from dwd import _residual_startup_cleanup as cleanup
    from dwd import _residual_worker_scope as scopes

    assert compiled._ACCEL is not None, 'Required matched native residual extension absent'
    native = compiled._ACCEL.evaluate
    cancelled, release = threading.Event(), threading.Event()
    entered = [threading.Event(), threading.Event()]
    primary = KeyboardInterrupt('primary owned-call fault')
    prewake = KeyboardInterrupt('secondary pre-wake fault')
    during_completion = KeyboardInterrupt('third raw-completion fault')
    native_ids, completed, calls, captured, receipts = [], [], [], [], []
    faults, pools = [], []
    owner = scopes.ResidualWorkerScope() if mode == 'native_reused' else None
    foreign_release = threading.Event()
    foreign = threading.Thread(target=foreign_release.wait, name='ir27-unrelated-native-control')
    foreign.start()
    foreign_id = foreign.native_id
    wake_line = _line(impl.ThreadPoolExecutor.shutdown, 'self._work_queue.put(None)')
    start_line = _line(threading.Thread.start, 'self._started.wait()')
    real_owned, real_wait = scopes.drain_owned_executor, cleanup._wait_completions
    original_trace = sys.gettrace()

    def evaluate(*args):
        status = native(*args)
        assert status == 0
        part = 0 if args[5] == 0 else 1
        native_ids.append(threading.get_native_id())
        calls.append(part)
        entered[part].set()
        assert release.wait(10), 'Probe did not release its owned evaluator'
        completed.append(part)
        return status

    trusted = types.SimpleNamespace(evaluate=evaluate)
    compiled._ACCEL = compiled._TRUSTED_ACCEL = trusted
    compiled._TRUSTED_EVALUATE = evaluate
    compiled._worker_count = lambda n: 2
    compiled.Event = lambda: cancelled

    def poll(*args, **kwargs):
        assert all(event.wait(5) for event in entered)
        raise primary

    if mode != 'native_startup_ready':
        compiled.wait = poll

    def startup_trace(frame, event, arg):
        if event == 'line' and frame.f_code is threading.Thread.start.__code__ and frame.f_lineno == start_line:
            worker = frame.f_locals['self']
            if getattr(worker, '_target', None) is impl._worker:
                assert entered[0].wait(5)
                assert worker._started.is_set()
                pool = worker._args[0]()
                assert worker not in pool._threads and worker not in impl._threads_queues
                faults.append('ready_unregistered_start')
                raise primary
        return startup_trace

    def wake_trace(frame, event, arg):
        if event == 'line' and frame.f_code is impl.ThreadPoolExecutor.shutdown.__code__ and frame.f_lineno == wake_line:
            if pools and frame.f_locals['self'] is pools[0] and 'pre_wake' not in faults:
                assert cancelled.is_set() and not release.is_set()
                faults.append('pre_wake')
                raise prewake
        return wake_trace

    def observe(stage, workers):
        if stage == 'captured':
            captured.extend(workers)

    def owned(pool, error=None, **kwargs):
        assert error is primary
        pools.append(pool)
        sys.settrace(wake_trace)
        result, interruption = real_owned(pool, error=error, observe=observe, **kwargs)
        assert interruption is prewake
        receipts.append(result)
        return result, interruption

    def wait_completion(mechanism, tokens):
        if 'completion_wait' not in faults:
            assert mechanism == 'sentinel'
            assert tokens and all(token.locked() for token in tokens)
            assert native_ids and live_native_ids(native_ids) == set(native_ids) and not release.is_set()
            assert all(worker._tstate_lock is not None for worker in captured)
            if owner is not None:
                assert owner.executor is pools[0], 'Ownership cleared before drain'
            faults.append('completion_wait')
            raise during_completion
        release.set()
        return real_wait(mechanism, tokens)

    scopes.drain_owned_executor, cleanup._wait_completions = owned, wait_completion
    caught, returned = None, False
    try:
        with owner if owner is not None else nullcontext():
            if mode == 'native_startup_ready':
                sys.settrace(startup_trace)
            try:
                compiled.compiled_values(np.eye(65), np.ones(65), np.zeros(65), .125, .25)
                returned = True
            except BaseException as error:
                caught = error
            finally:
                sys.settrace(original_trace)
        assert caught is primary and not returned
        wait_native_gone(native_ids)
        assert len(receipts) == 1 and receipts[0]['drained']
        assert sorted(completed) == sorted(calls)
        assert foreign.is_alive() and live_native_ids([foreign_id]) == {foreign_id}
        assert scopes._SCOPE.get() is None
        return dict(case=mode, variant='selected_source', primary_preserved=True, partial_output_published=False,
                    cancellation_published=cancelled.is_set(), native_calls=len(calls),
                    owned_native_remaining=[], foreign_thread_survived=True,
                    scope_handle_retired=owner is None or owner.executor is None,
                    scope_context_reset=True, cleanup_receipts=receipts, faults=faults,
                    completion_interrupted_while_native_alive=True,
                    workload='actual native arithmetic; forced2workers on65rows; no throughput claim')
    finally:
        sys.settrace(original_trace)
        scopes.drain_owned_executor, cleanup._wait_completions = real_owned, real_wait
        release.set()
        cancelled.set()
        for pool in pools:
            pool.shutdown(wait=False)
        if native_ids:
            wait_native_gone(native_ids)
        foreign_release.set()
        foreign.join(5)
        assert not foreign.is_alive()


def scope_exit_interrupt():
    """No body error: first cleanup interrupt is deferred until native exit."""
    from concurrent.futures import ThreadPoolExecutor
    from dwd import _residual_startup_cleanup as cleanup
    from dwd import _residual_worker_scope as scopes

    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    ids, observations, receipts = [], [], []
    interruption = KeyboardInterrupt('ordinary scope-exit cleanup fault')
    real_wait, real_owned = cleanup._wait_completions, scopes.drain_owned_executor
    scope = scopes.ResidualWorkerScope()
    pool = ThreadPoolExecutor(max_workers=1)

    def task():
        ids.append(threading.get_native_id())
        started.set()
        assert release.wait(10)
        finished.set()
        return 42

    def wait(mechanism, tokens):
        assert scope.executor is pool
        if not observations:
            assert tokens and all(token.locked() for token in tokens)
            assert live_native_ids(ids) == {ids[0]} and not finished.is_set()
            observations.append('interrupted_while_alive')
            raise interruption
        release.set()
        return real_wait(mechanism, tokens)

    def owned(executor, **kwargs):
        result = real_owned(executor, **kwargs)
        receipts.append(result[0])
        return result

    cleanup._wait_completions, scopes.drain_owned_executor = wait, owned
    caught, return_state = None, None
    try:
        with scope:
            scope.executor, scope.workers = pool, 1
            future = pool.submit(task)
            assert started.wait(5)
            worker, = pool._threads
            assert ids == [worker.native_id]
            worker_completed = worker_completion_probe(worker)
    except KeyboardInterrupt as error:
        caught = error
        assert caught is interruption
        # Capture the worker's raw completion token and application state at
        # propagation. A kernel thread ID can outlast its CPython thread state.
        return_state = dict(worker_completed=worker_completed(), task_finished=finished.is_set(),
                            future_done=future.done(), handle_retired=scope.executor is None,
                            scope_context_reset=scopes._SCOPE.get() is None,
                            native_remaining=sorted(live_native_ids(ids)))
        assert return_state['worker_completed'] and return_state['task_finished'], return_state
        assert return_state['future_done'] and future.result(0) == 42, return_state
        assert return_state['handle_retired'] and return_state['scope_context_reset'], return_state
        require_native_exit(return_state['native_remaining'], [worker.native_id])
        native_after_poll = sorted(live_native_ids(ids))
        assert not native_after_poll
    finally:
        cleanup._wait_completions, scopes.drain_owned_executor = real_wait, real_owned
        release.set()
        pool.shutdown(wait=False)
        wait_native_gone(ids)
    assert caught is interruption and observations == ['interrupted_while_alive']
    return dict(case='scope_exit_interrupt', variant='selected_source', cleanup_receipts=receipts,
                original_cleanup_interrupt_preserved=True, owned_native_remaining=native_after_poll,
                scope_handle_retired=True, scope_context_reset=True,
                completion_interrupted_while_native_alive=True, observed_before_rescue=return_state)


def ordinary_retirement(mode):
    import numpy as np
    from dwd import _compiled_residual as compiled
    from dwd import _residual_worker_scope as scopes

    assert compiled._ACCEL is not None, 'Required matched native residual extension absent'
    args = (np.eye(65), np.ones(65), np.zeros(65), .125, .25)
    budget = [2]
    compiled._worker_count = lambda n: budget[0]
    real_owned = scopes.drain_owned_executor
    receipts, owned_ids = [], []

    def owned(executor, **kwargs):
        owned_ids.extend(t.native_id for t in executor._threads)
        result = real_owned(executor, **kwargs)
        receipts.append(result[0])
        return result

    scopes.drain_owned_executor = owned
    try:
        with scopes.ResidualWorkerScope() as scope:
            first = compiled.compiled_values(*args)
            old = scope.executor
            assert old is not None
            if mode == 'serial_retire':
                budget[0] = 1
            elif mode == 'budget_retire':
                budget[0] = 3
            second = compiled.compiled_values(*args)
            assert all(a.tobytes() == b.tobytes() for a, b in zip(first, second))
            if mode == 'serial_retire':
                assert scope.executor is None and old._shutdown
            elif mode == 'budget_retire':
                assert scope.executor is not old and old._shutdown and scope.workers == 3
            else:
                assert mode == 'normal_reused' and scope.executor is old
        wait_native_gone(owned_ids)
        assert scope.executor is None and scopes._SCOPE.get() is None
        assert all(r['drained'] and r['repeated_interrupts'] == 0 for r in receipts)
        return dict(case=mode, variant='selected_source', exact_output_equality=True, cleanup_receipts=receipts,
                    owned_native_remaining=[], scope_handle_retired=True, scope_context_reset=True,
                    workload='actual native arithmetic; forced small worker budget; no throughput claim')
    finally:
        scopes.drain_owned_executor = real_owned


def unqualified_scope():
    """A wake without supported completion proof must retain exact ownership."""
    from concurrent.futures import ThreadPoolExecutor
    import concurrent.futures.thread as impl
    from dwd import _residual_startup_cleanup as cleanup
    from dwd import _residual_worker_scope as scopes

    scope = scopes.ResidualWorkerScope()
    pool = ThreadPoolExecutor(max_workers=1)
    identity = pool.submit(threading.get_native_id).result(5)
    workers = tuple(pool._threads)
    before = [(worker, impl._threads_queues[worker]) for worker in workers]
    real_capability, real_owned = cleanup._qualified_startup, scopes.drain_owned_executor
    receipts, caught = [], None

    def owned(executor, **kwargs):
        result = real_owned(executor, **kwargs)
        receipts.append(result[0])
        return result

    cleanup._qualified_startup = lambda: None  # Declared unsupported-adapter control.
    scopes.drain_owned_executor = owned
    try:
        with scope:
            scope.executor, scope.workers = pool, 1
    except BaseException as error:
        caught = error
    finally:
        cleanup._qualified_startup, scopes.drain_owned_executor = real_capability, real_owned
    retained = scope.executor is pool
    assert isinstance(caught, RuntimeError) and 'completion is unproved' in str(caught)
    assert retained and scope.closed and scopes._SCOPE.get() is None
    assert len(receipts) == 1 and receipts[0]['shutdown_requested']
    assert not receipts[0]['qualified'] and not receipts[0]['drained']
    assert all(impl._threads_queues.get(worker) is queue for worker, queue in before)
    wait_native_gone([identity])  # Independent observation, not a claimed helper drain.
    return dict(case='unqualified_scope', variant='selected_source', cleanup_receipts=receipts,
                unproved_drain_raised=True, exact_handle_retained=retained,
                registry_unchanged=True, scope_context_reset=True,
                native_exit_independently_observed=True)
