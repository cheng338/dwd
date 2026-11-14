"""Lazily reuse an executor within one kernel-solver invocation."""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
from threading import get_ident
import os
from ._residual_startup_cleanup import drain_owned_executor, own_default_executor


_SCOPE = ContextVar('dwd_residual_worker_scope', default=None)


def _drain_default_executor(executor, error=None):
    """Return a deferred interrupt only after positively established drain."""
    try:
        receipt, interruption = drain_owned_executor(executor, error=error)
        if error is not None:
            error.add_note('Interrupted-start recovery: ' + repr(receipt))
        if not receipt['drained']:
            failure = RuntimeError('Owned executor was woken but completion is unproved; drain not established')
            if interruption is not None and error is None:
                raise interruption from failure
            raise failure
        return interruption
    except BaseException as cleanup_error:
        if error is not None and cleanup_error is not error:
            error.add_note('Owned executor cleanup failed; final drain is not established. '
                           'See the chained cleanup error.')
            raise error from cleanup_error
        raise


class ResidualWorkerScope:
    """Own a lazy executor in one process/thread, without retaining fit arrays."""

    def __init__(self):
        self.owner = (os.getpid(), get_ident())
        self.executor = None
        self.workers = None
        self.busy = False
        self.closed = False

    def __enter__(self):
        self.token = _SCOPE.set(self)
        return self

    def drain(self, error=None):
        executor = self.executor
        if executor is not None:
            interruption = _drain_default_executor(executor, error)
            # Keep ownership while wake or completion may still need recovery.
            self.executor = None
            self.workers = None
            if interruption is not None and error is None:
                raise interruption
        else:
            self.workers = None

    def __exit__(self, *exc):
        try:
            self.closed = True
            self.drain(exc[1])
        finally:
            _SCOPE.reset(self.token)
        return False

    def eligible(self):
        return (not self.closed and not self.busy
                and self.owner == (os.getpid(), get_ident()))


def scoped_residual_workers(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        # Recursive solvers get independent lifetimes; no implicit cross-fit pool.
        with ResidualWorkerScope():
            return function(*args, **kwargs)
    return wrapped


@contextmanager
def executor_for_call(workers, factory, default_factory, allow_reuse):
    scope = _SCOPE.get()
    if (scope is None or not scope.eligible() or factory is not default_factory
            or not allow_reuse):
        # Hooks and foreign scalar conversions keep the former per-call lifetime.
        # Retire a previous qualified pool before entering that fallback.
        if scope is not None and scope.eligible():
            scope.drain()
        if factory is not default_factory:
            # Preserve caller-provided executor/subclass context semantics.
            with factory(max_workers=workers) as executor:
                yield executor
        else:
            executor = own_default_executor(factory(max_workers=workers))
            try:
                yield executor
            except BaseException as error:
                _drain_default_executor(executor, error)
                raise
            else:
                interruption = _drain_default_executor(executor)
                if interruption is not None:
                    raise interruption
        return
    scope.busy = True
    try:
        if scope.workers != workers:
            scope.drain()
        if scope.executor is None:
            scope.executor = own_default_executor(factory(max_workers=workers))
            scope.workers = workers
        try:
            yield scope.executor
        except BaseException as error:
            # The dispatcher has already set its call-local cancellation event.
            scope.drain(error)
            raise
    finally:
        scope.busy = False


def retire_for_serial_call():
    scope = _SCOPE.get()
    if scope is not None and scope.eligible():
        scope.drain()
