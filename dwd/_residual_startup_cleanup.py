"""Protected retirement for exclusively owned default CPython executors.

The original startup-window adapter remains capability/ownership guarded. The
owned-pool path publishes an idempotent public wake even when completion proof
declines, and waits only on admitted raw completion primitives. It does not
edit thread registries or claim general asynchronous-exception safety.
Callers must publish their call-local cancellation event before cleanup (or
pass it explicitly), own the executor exclusively and preserve the body error.
"""
import concurrent.futures.thread as thread_impl
import _thread
import inspect
import sys
import threading
import time
from types import BuiltinFunctionType, MethodDescriptorType
from weakref import ReferenceType, ref
from ._cleanup_diagnostics import (capture_exception_state, note_cleanup_failure,
                                   restore_exception_state, reraise_preserving)


_START = threading.Thread.start
_ADJUST = thread_impl.ThreadPoolExecutor._adjust_thread_count
_WORKER = thread_impl._worker
_SHUTDOWN = thread_impl.ThreadPoolExecutor.shutdown
_LOWLEVEL = _thread.start_new_thread
_SENTINEL = getattr(_thread, '_set_sentinel', None)
_JOINABLE = getattr(_thread, 'start_joinable_thread', None)
_HANDLE = getattr(_thread, '_ThreadHandle', None)
_HANDLE_JOIN = getattr(_HANDLE, 'join', None)
_HANDLE_DONE = getattr(_HANDLE, 'is_done', None)
_WORKER_CONTEXT = getattr(thread_impl, 'WorkerContext', None)
_CONTEXT_METHODS = {name: getattr(_WORKER_CONTEXT, name, None)
                    for name in ('__init__', 'initialize', 'run', 'finalize')}
_BOOTSTRAP_METHODS = {name: getattr(threading.Thread, name, None) for name in (
    '_bootstrap', '_bootstrap_inner', '_set_ident', '_set_tstate_lock', '_set_native_id')}


# Only source retrieval is cached. Every admission still checks the live runtime
# and callable bindings below. A changed code object or inspect hook invalidates
# this two-entry cache; unavailable source is never cached as a successful proof.
_SOURCE_CACHE = {}

# Private evaluation policy: only exact owned post-launch, not-yet-ready workers
# receive this additional readiness interval. It is not a total cleanup timeout.
_STARTUP_READY_WAIT_SECONDS = 0.010


def _cached_source(function, kind):
    provider = inspect.getsourcelines if kind == 'lines' else inspect.getsource
    code = getattr(function, '__code__', None)
    provider_code = getattr(provider, '__code__', None)
    cached = _SOURCE_CACHE.get(kind)
    if (cached is not None and cached[0] is function
            and cached[1] is code and cached[2] is provider
            and cached[3] is provider_code):
        return cached[4]
    result = provider(function)
    if kind == 'lines':
        result = (tuple(result[0]), result[1])
    _SOURCE_CACHE[kind] = (function, code, provider, provider_code, result)
    return result


def _qualified_startup():
    """Return the verified wait line and completion mechanism, or decline."""
    if sys.implementation.name != 'cpython':
        return None
    if (threading.Thread.start is not _START
            or _START.__module__ != 'threading'
            or _START.__qualname__ != 'Thread.start'):
        return None
    try:
        lines, start = _cached_source(_START, 'lines')
    except (OSError, TypeError, AttributeError):
        return None
    # Source layout (including comments on the wait line) varies by release.
    statements = [line.split('#', 1)[0].strip() for line in lines]
    text = ''.join(''.join(statements).split())
    matches = [start + i for i, line in enumerate(lines)
               if statements[i] == 'self._started.wait()']
    if len(matches) != 1:
        return None
    legacy = '_start_new_thread(self._bootstrap,())'
    handles = [name for name in ('_handle', '_os_thread_handle')
               if ('_start_joinable_thread(self._bootstrap,handle=self.' + name
                   + ',daemon=self.daemon)') in text]
    if legacy in text and not handles:
        if (not isinstance(_LOWLEVEL, BuiltinFunctionType)
                or _LOWLEVEL.__module__ != '_thread'
                or _LOWLEVEL.__name__ != 'start_new_thread'
                or _thread.start_new_thread is not _LOWLEVEL
                or getattr(threading, '_start_new_thread', None) is not _LOWLEVEL
                or getattr(threading, '_set_sentinel', None) is not _SENTINEL
                or getattr(_thread, '_set_sentinel', None) is not _SENTINEL
                or not isinstance(_SENTINEL, BuiltinFunctionType)
                or _SENTINEL.__module__ != '_thread'
                or _SENTINEL.__name__ != '_set_sentinel'):
            return None
        try:
            sentinel_source = _cached_source(_BOOTSTRAP_METHODS['_set_tstate_lock'], 'text')
        except (OSError, TypeError, AttributeError):
            return None
        if '_tstate_lock' not in sentinel_source:
            return None
        required, mechanism = legacy, 'sentinel'
    elif len(handles) == 1 and legacy not in text:
        if (not isinstance(_JOINABLE, BuiltinFunctionType)
                or _JOINABLE.__module__ != '_thread'
                or _JOINABLE.__name__ != 'start_joinable_thread'
                or getattr(_thread, 'start_joinable_thread', None) is not _JOINABLE
                or getattr(threading, '_start_joinable_thread', None) is not _JOINABLE
                or _HANDLE is None or _HANDLE.__module__ != '_thread'
                or _HANDLE.__name__ != '_ThreadHandle'
                or getattr(_thread, '_ThreadHandle', None) is not _HANDLE
                or getattr(threading, '_ThreadHandle', None) is not _HANDLE
                or not isinstance(_HANDLE_JOIN, MethodDescriptorType)
                or not isinstance(_HANDLE_DONE, MethodDescriptorType)
                or _HANDLE.join is not _HANDLE_JOIN
                or _HANDLE.is_done is not _HANDLE_DONE):
            return None
        mechanism = handles[0]
        required = '_start_joinable_thread(self._bootstrap,handle=self.' + mechanism + ',daemon=self.daemon)'
    else:
        return None
    # A real completion primitive is meaningful only with the standard
    # bootstrap. Replaced methods are outside this recovery scope.
    for name, method in _BOOTSTRAP_METHODS.items():
        if name == '_set_tstate_lock' and mechanism != 'sentinel':
            continue
        if (method is None or getattr(threading.Thread, name, None) is not method
                or getattr(method, '__module__', None) != 'threading'
                or getattr(method, '__qualname__', None) != 'Thread.' + name
                or getattr(getattr(method, '__code__', None), 'co_filename', None) != _START.__code__.co_filename):
            return None
    if text.count(required) != 1 or text.index(required) >= text.index('self._started.wait()'):
        return None
    return matches[0], mechanism


def _qualified_wait_line():
    startup = _qualified_startup()
    return None if startup is None else startup[0]


def _owned_worker_arguments(worker, executor):
    args = getattr(worker, '_args', ())
    if (getattr(worker, '_target', None) is not _WORKER
            or thread_impl._worker is not _WORKER or not args
            or type(args[0]) is not ReferenceType or args[0]() is not executor):
        return False
    if len(args) == 4:
        return (args[1] is executor._work_queue and args[2] is None and args[3] == ())
    # CPython 3.14's default WorkerContext moves initializer state into args[1].
    # Custom executor contexts or replaced methods are deliberately declined.
    if (len(args) != 3 or _WORKER_CONTEXT is None
            or getattr(thread_impl, 'WorkerContext', None) is not _WORKER_CONTEXT
            or type(args[1]) is not _WORKER_CONTEXT
            or args[2] is not executor._work_queue
            or args[1].initializer is not None or args[1].initargs != ()):
        return False
    return all(method is not None
               and getattr(_WORKER_CONTEXT, name, None) is method
               and getattr(method, '__module__', None) == thread_impl.__name__
               and getattr(method, '__qualname__', None) == 'WorkerContext.' + name
               for name, method in _CONTEXT_METHODS.items())


def _capture_postlaunch_worker(executor, error):
    """Capture exact omitted-worker ownership before readiness or wake.

    In particular, an exception at the low-level start CALL itself is ambiguous
    and is not admitted. No global thread list, other executor or guessed name
    supplies ownership evidence.
    """
    if type(executor) is not thread_impl.ThreadPoolExecutor:
        return None
    if threading.Thread.start is not _START or thread_impl.ThreadPoolExecutor._adjust_thread_count is not _ADJUST:
        return None
    startup = _qualified_startup()
    if startup is None:
        return None
    wait_line, mechanism = startup
    start_frames = []
    adjust_frames = []
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        if frame.f_code is _START.__code__ and trace.tb_lineno == wait_line:
            start_frames.append(frame.f_locals.get('self'))
        if frame.f_code is _ADJUST.__code__ and frame.f_locals.get('self') is executor:
            adjust_frames.append(frame.f_locals.get('t'))
        trace = trace.tb_next
    if len(start_frames) != 1 or len(adjust_frames) != 1:
        return None
    worker = start_frames[0]
    if worker is not adjust_frames[0] or type(worker) is not threading.Thread:
        return None
    if worker in executor._threads or worker in thread_impl._threads_queues:
        return None
    if not _owned_worker_arguments(worker, executor):
        return None
    return worker, mechanism


def capture_started_worker(executor, error):
    """Preserve the legacy ready-only capture contract.

    A successful low-level launch does not prove bootstrap completion. This
    entry point still declines an unset readiness flag without waiting.
    """
    captured = _capture_postlaunch_worker(executor, error)
    if captured is None:
        return None
    worker, mechanism = captured
    return worker if _completion_token(worker, mechanism) is not None else None


def _completion_token(worker, mechanism):
    if (type(worker) is not threading.Thread
            or type(getattr(worker, '_started', None)) is not threading.Event
            or not worker._started.is_set()):
        return None
    if mechanism == 'sentinel':
        token = getattr(worker, '_tstate_lock', None)
        return token if type(token) is _thread.LockType else None
    token = getattr(worker, mechanism, None)
    if (type(token) is _HANDLE and type(worker.ident) is int and worker.ident > 0
            and token.ident == worker.ident):
        return token
    return None


def _wait_sentinels(sentinels):
    # Unlike interrupted Thread.join, this never releases another thread's
    # thread-state lock. CPython releases it when that thread state is cleared.
    polls = 0
    while any(sentinel.locked() for sentinel in sentinels):
        # Idle workers normally finish during these bounded scheduler yields.
        # Longer active cleanup backs off without assuming a completion time.
        time.sleep(0 if polls < 64 else .001)
        polls += 1


def _wait_completions(mechanism, tokens):
    if mechanism == 'sentinel':
        _wait_sentinels(tokens)
        return
    polls = 0
    while not all(token.is_done() for token in tokens):
        time.sleep(0 if polls < 64 else .001)
        polls += 1
    # Early 3.13 is_done only observes thread-state teardown; it does not join
    # the OS thread. The builtin idempotent join completes that final step.
    # Retrying it after interruption never releases a foreign sentinel lock.
    for token in tokens:
        token.join(0)


def publish_cancellation(cancelled):
    """Publish the owned call event despite further KeyboardInterrupts.

    The caller already owns the primary error. This does not catch unrelated
    failures or promise protection against asynchronous exceptions at any byte.
    """
    while True:
        try:
            cancelled.set()
            return
        except KeyboardInterrupt:
            pass


def _unregistered_adjust_worker(executor, error):
    """Whether this exact failed submit may have omitted a worker."""
    trace = None if error is None else error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        if frame.f_code is _ADJUST.__code__ and frame.f_locals.get('self') is executor:
            worker = frame.f_locals.get('t')
            if worker is not None and worker not in executor._threads:
                return True
        trace = trace.tb_next
    return False


class _OwnedShutdown:
    """Weakly bound public-shutdown adapter for one exclusively owned pool."""

    def __init__(self, executor):
        self.executor_ref = ref(executor)
        self.completed_workers = None

    def completed(self, executor):
        return (self.executor_ref() is executor and executor._shutdown is True
                and self.completed_workers is not None
                and self.completed_workers == frozenset(executor._threads))

    def record_completion(self, executor):
        self.completed_workers = frozenset(executor._threads)

    def __call__(self, wait=True, *, cancel_futures=False):
        executor = self.executor_ref()
        if executor is None:
            return
        if not wait:
            # Preserve the nonblocking public API; wake alone records no drain.
            interruption = None
            interruption_state = None
            while True:
                try:
                    _check_shutdown_binding(executor)
                    _SHUTDOWN(executor, wait=False, cancel_futures=cancel_futures)
                    break
                except KeyboardInterrupt as error:
                    if interruption is None:
                        interruption = error
                        interruption_state = capture_exception_state(error)
                except BaseException as failure:
                    if interruption is not None:
                        note_cleanup_failure(interruption, failure,
                            'Owned nonblocking shutdown failed; wake is not established.')
                        reraise_preserving(interruption, interruption_state)
                    raise
            if interruption is not None:
                reraise_preserving(interruption, interruption_state)
            return
        receipt, interruption = drain_owned_executor(executor, cancel_futures=cancel_futures)
        if not receipt['drained']:
            failure = RuntimeError('Owned executor was woken but completion is unproved; drain not established')
            if interruption is not None:
                interruption_state = capture_exception_state(interruption)
                note_cleanup_failure(interruption, failure,
                    'Owned executor cleanup failed; final drain is not established.')
                reraise_preserving(interruption, interruption_state)
            raise failure
        if interruption is not None:
            raise interruption


_OWNED_SHUTDOWN_CALL = _OwnedShutdown.__call__
_OWNED_SHUTDOWN_CODE = _OwnedShutdown.__call__.__code__
_SHUTDOWN_CODE = _SHUTDOWN.__code__
_CODE_BINDINGS = tuple((function, getattr(function, '__code__', None))
                       for function in (_START, _ADJUST, _WORKER,
                           *_BOOTSTRAP_METHODS.values(), *_CONTEXT_METHODS.values())
                       if function is not None)


def _owned_shutdown(executor):
    shutdown = executor.shutdown
    if (type(shutdown) is _OwnedShutdown
            and _OwnedShutdown.__call__ is _OWNED_SHUTDOWN_CALL
            and _OwnedShutdown.__call__.__code__ is _OWNED_SHUTDOWN_CODE
            and shutdown.executor_ref() is executor):
        return shutdown
    return None


def _check_shutdown_binding(executor):
    if (type(executor) is not thread_impl.ThreadPoolExecutor
            or thread_impl.ThreadPoolExecutor.shutdown is not _SHUTDOWN
            or _SHUTDOWN.__code__ is not _SHUTDOWN_CODE
            or _SHUTDOWN.__module__ != thread_impl.__name__
            or _SHUTDOWN.__qualname__ != 'ThreadPoolExecutor.shutdown'
            or (_owned_shutdown(executor) is None
                and getattr(executor.shutdown, '__func__', None) is not _SHUTDOWN)):
        raise RuntimeError('Owned default executor shutdown binding is unavailable; drain not established')


def own_default_executor(executor):
    """Install protected public shutdown without a strong executor cycle."""
    _check_shutdown_binding(executor)
    if _owned_shutdown(executor) is None:
        executor.shutdown = _OwnedShutdown(executor)
    return executor


def drain_owned_executor(executor, error=None, cancelled=None, observe=None, *, cancel_futures=False):
    """Wake an exclusively owned default pool, then prove admitted completion.

    Requires the unchanged standard shutdown binding on an exactly owned pool.
    Replaced shutdown methods are refused before claiming wake or completion.
    Returns (receipt, first_cleanup_interrupt). The owner re-raises that first
    interrupt after successful cleanup if no earlier body error exists. Wake
    publication occurs even when completion qualification declines or raises.
    An unproved receipt must not be interpreted as a drain, nor followed by a
    Thread.join that can corrupt the available sentinel evidence.
    """
    first_interrupt = None
    first_interrupt_state = None
    repeated = 0

    def retry(function, *args):
        nonlocal first_interrupt, first_interrupt_state, repeated
        while True:
            try:
                return function(*args)
            except KeyboardInterrupt as interruption:
                if first_interrupt is None:
                    first_interrupt = interruption
                    first_interrupt_state = capture_exception_state(interruption)
                repeated += 1

    owned_shutdown = None
    pending_startup = None
    readiness_timing = None
    readiness_outcome = None
    readiness_report = None
    partial_report = {'attempted': False, 'workers': 0,
                      'completed': False, 'outcome': 'not_eligible'}

    workers = ()
    tokens = ()
    mechanism = None
    qualified = False
    reason = 'Completion capability or owned-worker readiness is unavailable'

    def capture():
        nonlocal owned_shutdown
        pending = None
        _check_shutdown_binding(executor)
        owned_shutdown = _owned_shutdown(executor)
        if owned_shutdown is not None and owned_shutdown.completed(executor):
            return tuple(executor._threads), (), None, True, None, None
        startup = _qualified_startup()
        if (startup is None
                or threading.Thread.start is not _START
                or thread_impl.ThreadPoolExecutor._adjust_thread_count is not _ADJUST
                or thread_impl._worker is not _WORKER
                or any(getattr(function, '__code__', None) is not code
                       for function, code in _CODE_BINDINGS)):
            return (), (), None, False, reason, None
        mechanism = startup[1]
        workers = tuple(executor._threads)
        captured = _capture_postlaunch_worker(executor, error) if error is not None else None
        possible_omitted = _unregistered_adjust_worker(executor, error)
        omitted = None
        tokens = tuple(_completion_token(worker, mechanism) for worker in workers)
        if captured is not None:
            worker, captured_mechanism = captured
            started = getattr(worker, '_started', None)
            if captured_mechanism == mechanism:
                if type(started) is threading.Event and not started.is_set():
                    if possible_omitted and all(token is not None for token in tokens):
                        # Retain proved ownership before wake can finish the
                        # worker and delete its target/args. Pending ownership
                        # belongs to this same capture tuple across retry.
                        pending = (worker, started, mechanism, frozenset(workers))
                else:
                    omitted_token = _completion_token(worker, mechanism)
                    if omitted_token is not None:
                        omitted = worker
                        workers += (omitted,)
                        tokens += (omitted_token,)
        qualified = (not possible_omitted or omitted is not None) and all(
            token is not None for token in tokens)
        why = ('Possible omitted worker completion is unproved'
               if possible_omitted and omitted is None else None if qualified else reason)
        return workers, tokens, mechanism, qualified, why, pending

    def request_wake():
        # Unknown replacements are explicitly outside the owned-public-shutdown
        # contract. KeyboardInterrupt during this check is still deferred.
        _check_shutdown_binding(executor)
        _SHUTDOWN(executor, wait=False, cancel_futures=cancel_futures)

    def begin_readiness_wait():
        nonlocal readiness_timing
        if readiness_timing is not None:
            return
        budget = _STARTUP_READY_WAIT_SECONDS
        if type(budget) is not float or not 0.0 <= budget <= 1.0:
            raise ValueError('Private startup readiness budget must be a finite float in [0, 1] seconds')
        now = time.perf_counter()
        # This tuple is committed once. Retried actions never renew its deadline.
        readiness_timing = (budget, now, now + budget)

    def pending_binding_is_valid():
        worker, started, saved_mechanism, registered = pending_startup
        _check_shutdown_binding(executor)
        startup = _qualified_startup()
        return (startup is not None and startup[1] == saved_mechanism
                and threading.Thread.start is _START
                and thread_impl.ThreadPoolExecutor._adjust_thread_count is _ADJUST
                and thread_impl._worker is _WORKER
                and not any(getattr(function, '__code__', None) is not code
                            for function, code in _CODE_BINDINGS)
                and _owned_shutdown(executor) is owned_shutdown
                and type(worker) is threading.Thread
                and getattr(worker, '_started', None) is started
                and type(started) is threading.Event
                and frozenset(executor._threads) == registered
                and worker not in executor._threads
                and worker not in thread_impl._threads_queues)

    def promote_pending_startup():
        nonlocal readiness_outcome
        worker, started, saved_mechanism, _ = pending_startup
        deadline = readiness_timing[2]
        while True:
            if not pending_binding_is_valid():
                readiness_outcome = ('unproved', None, time.perf_counter())
                return None
            if readiness_outcome is not None:
                return readiness_outcome[1]
            now = time.perf_counter()
            if now >= deadline:
                readiness_outcome = ('expired', None, now)
                return None
            if started.is_set():
                token = _completion_token(worker, saved_mechanism)
                observed = time.perf_counter()
                if observed >= deadline:
                    readiness_outcome = ('expired', None, observed)
                    return None
                if token is None:
                    readiness_outcome = ('unproved', None, observed)
                    return None
                # One outcome commit preserves an in-budget observation even
                # if later bookkeeping is interrupted after the deadline.
                readiness_outcome = ('ready', token, observed)
                return token
            time.sleep(min(.001, deadline - now))

    def readiness_metadata():
        status, _, observed = readiness_outcome
        budget, started_at, _ = readiness_timing
        return {'eligible': True, 'attempted': True,
                'budget_seconds': budget, 'outcome': status,
                'elapsed_seconds': max(0.0, observed - started_at)}

    # Capture tokens before wake can finish a worker and erase its target/args.
    # The finally is also required for unsupported or failed capture: no token
    # admission may bypass the idempotent public queue wake.
    try:
        try:
            if cancelled is not None:
                retry(cancelled.set)
            workers, tokens, mechanism, qualified, reason, pending_startup = retry(capture)
            if observe is not None:
                retry(observe, 'captured', workers)
        finally:
            retry(request_wake)
            if observe is not None:
                retry(observe, 'shutdown-requested', workers)
        if pending_startup is not None:
            # Wake precedes any readiness delay. This budget bounds only the
            # readiness policy, not scheduling, wake retries or the later drain.
            retry(begin_readiness_wait)
            if observe is not None:
                retry(observe, 'startup-readiness-wait', (pending_startup[0],))
            promoted = retry(promote_pending_startup)
            readiness_report = retry(readiness_metadata)
            if promoted is not None:
                workers += (pending_startup[0],)
                tokens += (promoted,)
                qualified = True
                reason = None
            else:
                reason = 'Owned omitted-worker readiness wait ended without completion proof: ' + readiness_report['outcome']
        if not qualified and pending_startup is not None and tokens:
            # Separate partial cleanup: the omitted worker remains unproved.
            # Never record complete executor retirement from this branch.
            def valid_registered_tokens():
                return (pending_binding_is_valid()
                        and len(workers) == len(tokens)
                        and all(_completion_token(worker, mechanism) is token
                                for worker, token in zip(workers, tokens)))

            def drain_registered_only():
                nonlocal partial_report
                if not valid_registered_tokens():
                    partial_report = {**partial_report, 'completed': False,
                                      'outcome': 'unproved'}
                    return
                partial_report = {'attempted': True, 'workers': len(workers),
                                  'completed': False, 'outcome': 'unproved'}
                _wait_completions(mechanism, tokens)
                if valid_registered_tokens():
                    partial_report = {'attempted': True, 'workers': len(workers),
                                      'completed': True, 'outcome': 'completed'}

            retry(drain_registered_only)
            if partial_report['completed'] and observe is not None:
                retry(observe, 'partial-drained', workers)
        if qualified:
            if mechanism is not None:
                retry(_wait_completions, mechanism, tokens)
            if owned_shutdown is not None:
                retry(owned_shutdown.record_completion, executor)
            if observe is not None:
                retry(observe, 'drained', workers)
    except BaseException as cleanup_error:
        # With no body error, the first cleanup interrupt remains primary even
        # if a later non-interrupt failure prevents proof of completion.
        if first_interrupt is not None and error is None and cleanup_error is not first_interrupt:
            note_cleanup_failure(first_interrupt, cleanup_error,
                'Owned cleanup failed; final drain is not established.')
            reraise_preserving(first_interrupt, first_interrupt_state)
        raise
    receipt = {'qualified': qualified, 'drained': qualified,
               'shutdown_requested': True, 'reason': reason,
               'owned_workers': len(workers), 'repeated_interrupts': repeated,
               'completion_mechanism': mechanism,
               'all_captured_workers_completed': qualified,
               'registry_mutation': False,
               'scope': 'Exclusively owned default executor; admitted CPython completion tokens',
               'previously_completed': qualified and mechanism is None,
               'partial_registered_drain': partial_report,
               'startup_readiness_wait': readiness_report if readiness_report is not None else {
                   'eligible': False, 'attempted': False, 'budget_seconds': None,
                   'outcome': 'not_eligible', 'elapsed_seconds': None}}
    if first_interrupt is not None:
        restore_exception_state(first_interrupt, first_interrupt_state)
    return receipt, first_interrupt


def drain_started_before_registration(executor, error, observe=None):
    """Drain the proven omitted worker plus registered workers, or decline.

    Repeated KeyboardInterrupt is deferred during admitted cleanup; the caller
    re-raises its original exception after this returns. Other failures remain
    visible. The observer is isolated-test instrumentation and is not a package
    API proposal. No executor/thread registry is edited.
    """
    repeated = 0

    def retry(function, *args):
        nonlocal repeated
        while True:
            try:
                return function(*args)
            except KeyboardInterrupt:
                repeated += 1

    worker = retry(capture_started_worker, executor, error)
    if worker is None:
        return {'qualified': False, 'drained': False,
                'reason': 'No exact post-launch owned-worker proof',
                'repeated_interrupts': repeated}
    while True:
        try:
            workers = tuple(executor._threads) + (worker,)
            startup = _qualified_startup()
            if startup is None:
                raise RuntimeError('Bootstrap changed during cleanup; drain not established')
            mechanism = startup[1]
            tokens = tuple(_completion_token(owned, mechanism) for owned in workers)
            if any(token is None for token in tokens):
                raise RuntimeError('Owned worker readiness/completion is unavailable; drain not established')
            if observe is not None:
                observe('captured', workers)
            # Cancellation was published before entry. Pending callbacks may
            # still run their cancellation check; sentinel completion covers
            # those callbacks too. Shutdown(wait=False) is idempotent.
            retry(lambda: executor.shutdown(wait=False))
            if observe is not None:
                observe('shutdown-requested', workers)
            retry(_wait_completions, mechanism, tokens)
            if observe is not None:
                observe('drained', workers)
            return {'qualified': True, 'drained': True,
                    'owned_workers': len(workers),
                    'repeated_interrupts': repeated,
                    'completion_mechanism': mechanism,
                    'all_captured_workers_completed': True,
                    'registry_mutation': False,
                    'scope': 'Exact CPython successful-start wait window only'}
        except KeyboardInterrupt:
            repeated += 1
