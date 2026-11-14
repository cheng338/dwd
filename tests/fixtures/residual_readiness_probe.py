"""Contained readiness controls; injected schedules are not Ctrl+C rates."""
import argparse
import concurrent.futures.thread as impl
from contextlib import nullcontext
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
import _thread
import weakref

import psutil


RETAINED = []


def native_ids():
    return {row.id for row in psutil.Process().threads()}


def wait_gone(ids, timeout=5):
    deadline = time.perf_counter() + timeout
    while set(ids) & native_ids():
        if time.perf_counter() >= deadline:
            raise AssertionError('Owned native worker remained after explicit test containment')
        time.sleep(.001)


def source_line(function, text):
    lines, first = inspect.getsourcelines(function)
    found = [first+i for i, line in enumerate(lines) if line.strip() == text]
    if len(found) != 1:
        raise AssertionError('Unsupported exact source seam: ' + text)
    return found[0]


def origins(root):
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'dwd' or name.startswith('dwd.'):
            filename = getattr(module, '__file__', None)
            if filename:
                path = Path(filename).resolve()
                assert path.is_relative_to((root/'dwd').resolve()), (name, str(path))
                result[name] = str(path)
    return result


def clock_metadata():
    result = {}
    for name in ('monotonic', 'perf_counter'):
        info = time.get_clock_info(name)
        result[name] = dict(implementation=info.implementation,
                           resolution=info.resolution, monotonic=info.monotonic,
                           adjustable=info.adjustable)
    return result


def ordinary(cleanup, scope):
    receipts, pools, ids, outputs = [], [], set(), []
    real = scope.drain_owned_executor

    def observed(pool, *args, **kwargs):
        value = real(pool, *args, **kwargs)
        receipts.append(value[0])
        return value

    def task(value):
        ids.add(threading.get_native_id())
        return value * value

    scope.drain_owned_executor = observed
    try:
        with scope.ResidualWorkerScope() as lifetime:
            for workers in (2, 2, 1):
                with scope.executor_for_call(workers, impl.ThreadPoolExecutor,
                        impl.ThreadPoolExecutor, True) as pool:
                    pools.append(pool)
                    outputs.append(pool.submit(task, workers).result(timeout=5))
            scope.retire_for_serial_call()
            assert lifetime.executor is None
        assert pools[0] is pools[1] and pools[1] is not pools[2]
        wait_gone(ids)
        assert outputs == [4, 4, 1]
        assert all(row['drained'] for row in receipts)
        assert all(not row.get('startup_readiness_wait', {}).get('attempted', False)
                   for row in receipts)
        return dict(case='ordinary', cleanup_receipts=receipts, exact_outputs=outputs,
                    native_remaining=[], scope_context_reset=scope._SCOPE.get() is None,
                    reused_same_pool=True, budget_retired_pool=True, serial_retired_pool=True,
                    test_containment_release=False)
    finally:
        scope.drain_owned_executor = real


def exceptional(cleanup, scope, case, budget_ms, delay_ms):
    """Keep genuine native bootstraps gated; never use Thread.is_alive as proof."""
    assert sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 12)
    start_code = threading.Thread.start.__code__
    wait_line = source_line(threading.Thread.start, 'self._started.wait()')
    acquire_line = source_line(threading.Thread._set_tstate_lock,
                               'self._tstate_lock.acquire()')
    real_launch, real_start = threading._start_new_thread, threading.Thread.start
    real_cleanup, old_trace = scope.drain_owned_executor, sys.gettrace()
    primary = KeyboardInterrupt('IR39 deterministic post-launch startup interruption')
    capture_interrupt = KeyboardInterrupt('IR39 interrupted pending capture')
    repeat_interrupts = [KeyboardInterrupt('IR39 readiness retry 1'),
                         KeyboardInterrupt('IR39 readiness retry 2')]
    repeated_first = repeat_interrupts[0]
    bootstrap_gate, bootstrap_entered = threading.Event(), threading.Event()
    sentinel_gate, sentinel_published = threading.Event(), threading.Event()
    known_release, known_entered = threading.Event(), threading.Event()
    foreign_release, foreign_entered = threading.Event(), threading.Event()
    schedule, controller_done = threading.Event(), threading.Event()
    signal_armed, signal_issued = threading.Event(), threading.Event()
    partial_wait_entered = threading.Event()
    cancellation = threading.Event()
    workers, pools, owned_ids, known_ids, foreign_ids = [], [], [], [], []
    receipts, cleanup_calls, events, timing_snapshots = [], [], [], []
    mutations, scheduled_at, release_times, readiness_observed = [], [], [], []
    task_calls, task_results = [], []
    caught = None
    lifetime = None
    capture_fault_fired = False
    completion_fault_fired = False
    native_sentinel_snapshot = None
    rapid_observation = {}
    mixed = case in ('mixed', 'partial_interrupt')
    reused = case in ('reused_ready', 'reused_expiry')
    held = case in ('never', 'reused_expiry', 'event_changed', 'membership_changed',
                   'wrong_owner', 'sentinel_unacquired', 'mixed', 'partial_interrupt',
                   'capture_declined', 'source_changed', 'signal_expiry', 'repeated_interrupts')
    signal_case = case in ('signal_ready', 'signal_expiry')

    def foreign_body():
        foreign_ids.append(threading.get_native_id())
        foreign_entered.set()
        assert foreign_release.wait(15)

    foreign = threading.Thread(target=foreign_body, name='ir39-unowned-control')
    foreign.start()
    assert foreign_entered.wait(5)

    def controller_body():
        try:
            if not schedule.wait(10):
                return
            if signal_case and signal_armed.is_set():
                events.append('interrupt_main_requested')
                _thread.interrupt_main()
                signal_issued.set()
                if case == 'signal_expiry':
                    return
            if mixed:
                # A known registered task outlives the readiness budget.
                if case == 'partial_interrupt':
                    assert partial_wait_entered.wait(5)
                time.sleep(max(.030, 3 * budget_ms / 1000))
                release_times.append(dict(kind='registered', at=time.perf_counter()))
                known_release.set()
            elif not held:
                time.sleep(max(0, delay_ms) / 1000)
                release_times.append(dict(kind='bootstrap', at=time.perf_counter()))
                bootstrap_gate.set()
        finally:
            controller_done.set()

    controller = threading.Thread(target=controller_body, name='ir39-release-controller')
    controller.start()

    def bootstrap_trace(frame, event, arg):
        nonlocal native_sentinel_snapshot
        if (case == 'sentinel_unacquired' and event == 'line'
                and frame.f_code is threading.Thread._set_tstate_lock.__code__
                and frame.f_lineno == acquire_line):
            token = frame.f_locals['self']._tstate_lock
            native_sentinel_snapshot = dict(published=True, locked=token.locked(),
                                            started=frame.f_locals['self']._started.is_set())
            sentinel_published.set()
            if not sentinel_gate.wait(15):
                os._exit(91)
        if (event == 'line' and frame.f_code is threading.Thread._bootstrap_inner.__code__
                and frame.f_locals['self']._started.is_set() and not readiness_observed):
            readiness_observed.append(time.perf_counter())
            sys.settrace(None)
        return bootstrap_trace

    def held_launch(function, args):
        worker = function.__self__
        if worker not in workers:
            workers.append(worker)
        if case == 'prelaunch':
            threading._start_new_thread = real_launch
            events.append('prelaunch_interruption')
            raise primary

        def held_bootstrap():
            owned_ids.append(threading.get_native_id())
            bootstrap_entered.set()
            if not bootstrap_gate.wait(15):
                os._exit(92)
            if case == 'bootstrap_failure':
                events.append('test_bootstrap_returned_without_readiness')
                return  # Actual native exit, no _started or CPython completion token.
            sys.settrace(bootstrap_trace)
            function(*args)

        value = real_launch(held_bootstrap, ())
        assert bootstrap_entered.wait(5)
        if case == 'ambiguous_call':
            threading._start_new_thread = real_launch
            events.append('ambiguous_lowlevel_call_interruption')
            raise primary
        return value

    def startup_trace(frame, event, arg):
        if event == 'line' and frame.f_code is start_code and frame.f_lineno == wait_line:
            worker = frame.f_locals['self']
            if worker in workers:
                assert worker not in pools[0]._threads and worker not in impl._threads_queues
                threading._start_new_thread = real_launch
                if case == 'ready_omitted':
                    bootstrap_gate.set()
                    assert worker._started.wait(5)
                elif case == 'sentinel_unacquired':
                    bootstrap_gate.set()
                    assert sentinel_published.wait(5)
                    assert native_sentinel_snapshot == dict(published=True, locked=False, started=False)
                elif case == 'bootstrap_failure':
                    bootstrap_gate.set()
                    wait_gone(owned_ids)
                if case == 'wrong_owner':
                    old_args = worker._args
                    worker._args = (weakref.ref(foreign_pool),) + old_args[1:]
                    mutations.append(lambda: setattr(worker, '_args', old_args))
                events.append('startup_wait_interruption')
                raise primary
        return startup_trace

    def timing_from_stack():
        frame = inspect.currentframe().f_back
        while frame:
            if frame.f_code is real_cleanup.__code__:
                timing = frame.f_locals.get('readiness_timing')
                if timing is not None:
                    timing_snapshots.append(list(timing))
                return
            frame = frame.f_back

    def observe(event, captured):
        events.append(event)
        if event == 'startup-readiness-wait':
            assert pools[0]._shutdown and cancellation.is_set()
            assert 'shutdown-requested' in events
            timing_from_stack()
            if signal_case:
                signal_armed.set()
            if not schedule.is_set():
                scheduled_at.append(time.perf_counter())
                schedule.set()
            if case == 'event_changed' and not mutations:
                original = workers[0]._started
                workers[0]._started = threading.Event()
                mutations.append(lambda: setattr(workers[0], '_started', original))
            elif case == 'membership_changed' and not mutations:
                pools[0]._threads.add(foreign)
                mutations.append(lambda: pools[0]._threads.remove(foreign))
            elif case == 'source_changed' and not mutations:
                threading.Thread.start = lambda self: (_ for _ in ()).throw(AssertionError('not called'))
                mutations.append(lambda: setattr(threading.Thread, 'start', real_start))
            elif case == 'rapid_completion':
                bootstrap_gate.set()
                # Ownership was captured before wake, but this gated native
                # bootstrap has no sentinel until it becomes ready. Retain the
                # genuine token once published; never call Thread.join or fake
                # the clock. OS-native absence is checked independently after
                # the real helper returns, outside this readiness observer.
                token = None
                observer_end = time.perf_counter() + 5
                while True:
                    if workers[0]._started.is_set():
                        if token is None:
                            token = workers[0]._tstate_lock
                        assert type(token) is _thread.LockType
                        assert workers[0]._tstate_lock is token
                        if (not token.locked() and not hasattr(workers[0], '_target')
                                and not hasattr(workers[0], '_args')):
                            rapid_observation.update(
                                observed_at=time.perf_counter(), token_unlocked=True,
                                token_identity_preserved=True, target_args_deleted=True,
                                token_capture='after genuine standard bootstrap readiness')
                            break
                    if time.perf_counter() >= observer_end:
                        raise AssertionError('Rapid worker did not complete within test containment bound')
                    time.sleep(0)
            elif case == 'repeated_interrupts' and repeat_interrupts:
                time.sleep(.001)
                raise repeat_interrupts.pop(0)
            elif signal_case:
                # Pending interrupt is requested by a separate actual native
                # thread while this helper observer is inside its retry region.
                end = time.perf_counter() + .020
                while not signal_issued.is_set() and time.perf_counter() < end:
                    time.sleep(.0002)
                assert signal_issued.is_set(), 'Pending signal controller did not acknowledge'


    def retry_fault_trace(frame, event, arg):
        nonlocal capture_fault_fired, completion_fault_fired
        if event != 'line' or frame.f_code.co_filename != real_cleanup.__code__.co_filename:
            return retry_fault_trace
        if (case.startswith('capture_') and not capture_fault_fired
                and frame.f_code.co_name == 'capture'
                and frame.f_locals.get('pending') is not None):
            # First next source line after the local pending tuple commit.
            capture_fault_fired = True
            assert frame.f_locals['pending'][0] is workers[0]
            if case == 'capture_declined':
                threading.Thread.start = lambda self: (_ for _ in ()).throw(AssertionError('not called'))
            else:
                bootstrap_gate.set()
                assert workers[0]._started.wait(5)
            events.append('pending_capture_interrupted')
            raise capture_interrupt
        if (case == 'partial_interrupt' and not completion_fault_fired
                and frame.f_code.co_name == '_wait_completions'):
            completion_fault_fired = True
            assert set(known_ids) & native_ids()
            partial_wait_entered.set()
            events.append('partial_wait_interrupted_with_native_alive')
            raise capture_interrupt
        return retry_fault_trace

    def observed_cleanup(pool, error=None, **kwargs):
        assert error is primary
        started_at = time.perf_counter()
        if case.startswith('capture_') or case == 'partial_interrupt':
            sys.settrace(retry_fault_trace)
        try:
            receipt, deferred = real_cleanup(pool, error=error, observe=observe, **kwargs)
            receipts.append(receipt)
            cleanup_calls.append(dict(start=started_at, end=time.perf_counter(),
                                      deferred_type=None if deferred is None else type(deferred).__name__,
                                      capture_interrupt_preserved=deferred is capture_interrupt,
                                      first_repeated_interrupt_preserved=deferred is repeated_first,
                                      native_remaining=sorted(set(owned_ids + known_ids) & native_ids())))
            return receipt, deferred
        finally:
            sys.settrace(old_trace)

    def known_task():
        known_ids.append(threading.get_native_id())
        known_entered.set()
        assert known_release.wait(15)
        return 17

    def cancelled_task():
        task_calls.append('entered')
        # A ready omitted worker may enter its queued callback before the main
        # thread processes the injected interruption. Hold that callback until
        # the actual owner publishes cancellation; never race publication.
        assert cancellation.wait(5)
        return None

    # Used only as a deliberately wrong weakref owner; it never starts work.
    foreign_pool = impl.ThreadPoolExecutor(max_workers=1)
    scope.drain_owned_executor = observed_cleanup
    began = time.perf_counter()
    propagated = None
    at_propagation = None
    try:
        with (scope.ResidualWorkerScope() if reused else nullcontext()) as lifetime:
            with scope.executor_for_call(2, impl.ThreadPoolExecutor,
                                        impl.ThreadPoolExecutor, True) as pool:
                pools.append(pool)
                if mixed:
                    pool.submit(known_task)
                    assert known_entered.wait(5)
                threading._start_new_thread = held_launch
                sys.settrace(startup_trace)
                try:
                    task_results.append(pool.submit(cancelled_task))
                except BaseException:
                    cancellation.set()
                    raise
                finally:
                    sys.settrace(old_trace)
                    threading._start_new_thread = real_launch
    except BaseException as error:
        caught = error
        RETAINED.append(error)
        propagated = time.perf_counter()
        at_propagation = dict(omitted_native_alive=sorted(set(owned_ids) & native_ids()),
                              registered_native_alive=sorted(set(known_ids) & native_ids()),
                              foreign_native_alive=sorted(set(foreign_ids) & native_ids()))
        at_propagation['whole_completion_cached'] = bool(
            pools and cleanup._owned_shutdown(pools[0]).completed(pools[0]))
    finally:
        sys.settrace(old_trace)
        threading._start_new_thread = real_launch
        threading.Thread.start = real_start
        for restore in reversed(mutations):
            restore()
        containment_at = time.perf_counter()
        bootstrap_gate.set()
        sentinel_gate.set()
        known_release.set()
        partial_wait_entered.set()
        schedule.set()
        controller.join(5)
        assert not controller.is_alive()
        # The package must have issued its wake. A rescue wake is recorded only
        # if no shutdown was reached; it never establishes package completion.
        rescue_wake = bool(pools and not pools[0]._shutdown)
        if rescue_wake:
            impl.ThreadPoolExecutor.shutdown(pools[0], wait=False)
        wait_gone(owned_ids + known_ids)
        foreign_survived = bool(set(foreign_ids) & native_ids())
        foreign_release.set()
        foreign.join(5)
        assert not foreign.is_alive()
        foreign_pool.shutdown(wait=True)
        scope.drain_owned_executor = real_cleanup
    assert caught is primary, (type(caught).__name__, str(caught))
    assert cancellation.is_set() and not task_results
    assert receipts and all(row['shutdown_requested'] for row in receipts)
    assert at_propagation is not None and at_propagation['foreign_native_alive'] == foreign_ids
    assert foreign_survived and scope._SCOPE.get() is None
    if all(row['drained'] for row in receipts):
        assert not at_propagation['omitted_native_alive'] and not at_propagation['registered_native_alive']
    if case.startswith('capture_'):
        assert capture_fault_fired
    if case == 'partial_interrupt':
        assert completion_fault_fired
    return dict(case=case, injection='controlled bootstrap/trace; pending interrupt_main only when named',
                evaluator='real ThreadPoolExecutor and native OS workers; cancellation-only Python task',
                primary_preserved=True, first_error_type=type(caught).__name__,
                primary_notes=list(getattr(caught, '__notes__', ())),
                retained_traceback=True, cancellation_published=True, partial_output_published=False,
                queued_callback_entries=len(task_calls), submit_returned_future=bool(task_results),
                cleanup_receipts=receipts, cleanup_calls=cleanup_calls, cleanup_receipt_count=len(receipts),
                whole_call_seconds=propagated-began, propagation_at=propagated,
                at_propagation=at_propagation, containment_release_at=containment_at,
                test_containment_release=True, test_rescue_wake=rescue_wake,
                native_remaining=sorted(set(owned_ids+known_ids) & native_ids()),
                omitted_registered=bool(workers and workers[0] in impl._threads_queues),
                owned_native_ids=owned_ids, registered_native_ids=known_ids,
                foreign_thread_survived=foreign_survived, scope_context_reset=scope._SCOPE.get() is None,
                scope_handle_retained=bool(reused and lifetime.executor is pools[0]),
                schedule_at=scheduled_at, actual_release_times=release_times,
                bootstrap_readiness_observed_at=readiness_observed,
                readiness_observation='native trace after actual standard _started.set returns',
                deadline_snapshots=timing_snapshots, events=events,
                native_sentinel_snapshot=native_sentinel_snapshot,
                rapid_completion_observation=rapid_observation,
                capture_fault_fired=capture_fault_fired, completion_fault_fired=completion_fault_fired)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--budget-ms', type=float, default=10.)
    parser.add_argument('--delay-ms', type=float, default=2.5)
    args = parser.parse_args()
    root = Path(args.source_root).resolve()
    assert (root/'dwd'/'__init__.py').is_file()
    sys.path.insert(0, str(root))
    from dwd import _residual_startup_cleanup as cleanup
    from dwd import _residual_worker_scope as scope
    origins(root)
    assert args.budget_ms in (1., 10., 50.)
    assert hasattr(cleanup, '_STARTUP_READY_WAIT_SECONDS')
    cleanup._STARTUP_READY_WAIT_SECONDS = args.budget_ms / 1000
    print(json.dumps(dict(pid=os.getpid(), creation_time=psutil.Process().create_time())), flush=True)
    assert sys.stdin.readline() == 'GO\n'
    result = ordinary(cleanup, scope) if args.case == 'ordinary' else exceptional(
        cleanup, scope, args.case, args.budget_ms, args.delay_ms)
    result.update(implementation='selected_package', budget_ms=args.budget_ms, requested_delay_ms=args.delay_ms,
                  origins=origins(root), timer_clock='perf_counter', clocks=clock_metadata())
    print(json.dumps(dict(result=result), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
