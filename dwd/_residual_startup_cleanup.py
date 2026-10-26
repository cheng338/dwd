"""Guarded cleanup for one proven CPython Thread.start interruption window.

Derived from the qualified C04 direct-v5 control. CPython's sentinel and
joinable-handle bootstraps are admitted by capability and ownership checks,
not by minor version. Unknown implementations decline this recovery; this
is not a general asynchronous-exception safety claim.
The caller must set its call-local cancellation event before invoking this
helper, must exclusively own the executor, and must preserve the original error.
"""
import concurrent.futures.thread as thread_impl
import _thread
import inspect
import sys
import threading
import time
from types import BuiltinFunctionType, MethodDescriptorType
from weakref import ReferenceType


_START = threading.Thread.start
_ADJUST = thread_impl.ThreadPoolExecutor._adjust_thread_count
_WORKER = thread_impl._worker
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


def _qualified_startup():
    """Return the verified wait line and completion mechanism, or decline."""
    if sys.implementation.name != 'cpython':
        return None
    if (threading.Thread.start is not _START
            or _START.__module__ != 'threading'
            or _START.__qualname__ != 'Thread.start'):
        return None
    try:
        lines, start = inspect.getsourcelines(_START)
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
            sentinel_source = inspect.getsource(_BOOTSTRAP_METHODS['_set_tstate_lock'])
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


def capture_started_worker(executor, error):
    """Return the positively identified omitted worker, else decline unchanged.

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
    # A successful low-level launch does not prove that bootstrap completed.
    # A bootstrap error can leave _started unset forever. Admit only readiness
    # already observed here; do not wait speculatively to establish it.
    if _completion_token(worker, mechanism) is None:
        return None
    return worker


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
    while any(sentinel.locked() for sentinel in sentinels):
        time.sleep(.01)


def _wait_completions(mechanism, tokens):
    if mechanism == 'sentinel':
        _wait_sentinels(tokens)
        return
    while not all(token.is_done() for token in tokens):
        time.sleep(.01)
    # Early 3.13 is_done only observes thread-state teardown; it does not join
    # the OS thread. The builtin idempotent join completes that final step.
    # Retrying it after interruption never releases a foreign sentinel lock.
    for token in tokens:
        token.join(0)


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
