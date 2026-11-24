"""Contained cleanup faults against the selected installed/source package."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import threading

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _residual_native_threads import live_native_ids, wait_native_gone

_RETAINED = []


def _line(function, text):
    lines, start = inspect.getsourcelines(function)
    found = [start+i for i, value in enumerate(lines) if value.strip() == text]
    if len(found) != 1:
        raise RuntimeError('Unqualified interpreter source line: '+text)
    return found[0]


def _origins(root):
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'dwd' or name.startswith('dwd.'):
            filename = getattr(module, '__file__', None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to((root/'dwd').resolve()):
                    raise AssertionError('Wrong package origin: '+name+' '+str(path))
                result[name] = str(path)
    return result


def _declined_startup(exit_probe):
    """Gate the actual native bootstrap; fault at exact post-launch wait line.

    The launcher is restored before the first exception. The real bootstrap
    gate remains held through exception propagation, forcing readiness expiry.
    A second trace exception is injected after _shutdown=True, before queue.put.
    """
    import concurrent.futures.thread as impl
    import numpy as np
    from dwd import _compiled_residual as compiled
    from dwd import _residual_startup_cleanup as cleanup
    from dwd import _residual_worker_scope as scope

    if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('This declined-startup control is qualified for CPython3.12 only.')
    start_line = _line(threading.Thread.start, 'self._started.wait()')
    wake_line = _line(impl.ThreadPoolExecutor.shutdown, 'self._work_queue.put(None)')
    real_launch = threading._start_new_thread
    real_owned_cleanup = getattr(scope, 'drain_owned_executor', None)
    release_bootstrap = threading.Event()
    launched = threading.Event()
    owned_ids, workers, pools, receipts = [], [], [], []
    primary, secondary = KeyboardInterrupt('primary post-launch'), KeyboardInterrupt('secondary pre-wake')
    cancellation = threading.Event()
    events = []
    caught = None

    def held_launch(function, args):
        def held():
            owned_ids.append(threading.get_native_id())
            launched.set()
            if not release_bootstrap.wait(15):
                os._exit(91)  # Last resort is confined to this owned child.
            function(*args)
        identity = real_launch(held, ())
        if not launched.wait(5):
            raise RuntimeError('Owned native bootstrap did not start.')
        return identity

    def startup_trace(frame, event, arg):
        if event == 'line' and frame.f_code is threading.Thread.start.__code__ and frame.f_lineno == start_line:
            worker = frame.f_locals['self']
            if getattr(worker, '_target', None) is impl._worker:
                worker_args = worker._args
                pool = worker_args[0]()
                assert worker not in pool._threads and worker not in impl._threads_queues
                assert not worker._started.is_set()
                workers.append(worker)
                pools.append(pool)
                threading._start_new_thread = real_launch
                events.append('first_interrupt_before_readiness')
                raise primary
        return startup_trace

    def wake_trace(frame, event, arg):
        if event == 'line' and frame.f_code is impl.ThreadPoolExecutor.shutdown.__code__ and frame.f_lineno == wake_line:
            pool = frame.f_locals['self']
            if pools and pool is pools[0] and 'second_interrupt_before_queue_put' not in events:
                assert pool._shutdown and cancellation.is_set()
                events.append('second_interrupt_before_queue_put')
                raise secondary
        return wake_trace

    def observed_owned_cleanup(pool, error=None, **kwargs):
        assert error is primary
        sys.settrace(wake_trace)
        result, interruption = real_owned_cleanup(pool, error=error, **kwargs)
        receipts.append(result)
        assert result['qualified'] is False and result['drained'] is False
        assert result['shutdown_requested'] is True and interruption is secondary
        return result, interruption

    # This control needs a real executor/native thread but no arithmetic result.
    # Cancellation occurs before the queued evaluator starts. Label the stand-in.
    class UnusedNative:
        @staticmethod
        def evaluate(*args):
            raise AssertionError('Cancelled startup must not publish native arithmetic.')
    compiled._ACCEL = UnusedNative
    compiled._worker_count = lambda n: 2
    compiled.Event = lambda: cancellation
    assert real_owned_cleanup is not None
    scope.drain_owned_executor = observed_owned_cleanup
    original_trace = sys.gettrace()
    try:
        threading._start_new_thread = held_launch
        sys.settrace(startup_trace)
        try:
            compiled.compiled_values(np.eye(8), np.ones(8), np.zeros(8), 0.125, 0.25)
        except BaseException as error:
            caught = error
            _RETAINED.append(error)  # Keep traceback/executor reachable through exit.
        finally:
            sys.settrace(original_trace)
            threading._start_new_thread = real_launch
        assert caught is primary
        assert events == ['first_interrupt_before_readiness', 'second_interrupt_before_queue_put']
        assert len(workers) == len(pools) == 1 and len(owned_ids) == 1
        unready_at_propagation = not workers[0]._started.is_set()
        native_at_propagation = sorted(live_native_ids(owned_ids))
        assert unready_at_propagation and native_at_propagation == owned_ids
        # Release only the controlled bootstrap gate after recording the
        # unproved package return. The package has already supplied the wake.
        release_bootstrap.set()
        assert workers[0]._started.wait(5)
        wait_native_gone(owned_ids)  # Natural exit after test gate release; no rescue wake.
        result = dict(case='declined_exit' if exit_probe else 'declined_observe', variant='selected_source',
            injection='deterministic trace at two exact Python source lines; not ordinary Ctrl+C',
            evaluator='unused stand-in; real owned ThreadPoolExecutor/native worker',
            events=events, primary_preserved=caught is primary,
            secondary_is_cause=caught.__cause__ is secondary,
            cancellation_published=cancellation.is_set(), cleanup_receipts=receipts,
            omitted_registered=workers[0] in impl._threads_queues,
            owned_native_remaining=sorted(live_native_ids(owned_ids)),
            shutdown_flag=pools[0]._shutdown, retained_traceback=True,
            bootstrap_unready_at_propagation=unready_at_propagation,
            owned_native_at_propagation=native_at_propagation,
            test_bootstrap_release_after_propagation=True,
            scope_context_reset=scope._SCOPE.get() is None)
        if exit_probe:
            result['rescue_performed'] = False
            return result  # Natural exit after wake, with retained traceback.
        result.update(rescue_performed=False, owned_native_after_wake=[])
        return result
    finally:
        sys.settrace(original_trace)
        threading._start_new_thread = real_launch
        if real_owned_cleanup is not None:
            scope.drain_owned_executor = real_owned_cleanup
        release_bootstrap.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--case', choices=('declined_observe', 'declined_exit',
        'native_startup_ready', 'native_per_call', 'native_reused', 'scope_exit_interrupt',
        'normal_reused', 'serial_retire', 'budget_retire', 'unqualified_scope'), required=True)
    args = parser.parse_args()
    root = Path(args.source_root).resolve()
    if not (root/'dwd'/'__init__.py').is_file():
        raise RuntimeError('Absent selected package root: '+str(root))
    sys.path.insert(0, str(root))
    # Imports and their origins are checked before the parent fault handshake.
    from dwd import _compiled_residual, _residual_worker_scope
    _origins(root)
    print(json.dumps(dict(pid=os.getpid(), creation_time=psutil.Process().create_time())), flush=True)
    assert sys.stdin.readline() == 'GO\n'
    if args.case.startswith('declined_'):
        result = _declined_startup(args.case == 'declined_exit')
    else:
        import residual_cleanup_candidate_probe as candidate
        if args.case.startswith('native_'):
            result = candidate.interrupted_native(args.case)
        elif args.case == 'scope_exit_interrupt':
            result = candidate.scope_exit_interrupt()
        elif args.case == 'unqualified_scope':
            result = candidate.unqualified_scope()
        else:
            result = candidate.ordinary_retirement(args.case)
    result['origins'] = _origins(root)
    print(json.dumps(dict(result=result), allow_nan=False), flush=True)
    if args.case == 'declined_exit':
        print(json.dumps(dict(interpreter_exit_entered=True)), flush=True)


if __name__ == '__main__':
    main()
