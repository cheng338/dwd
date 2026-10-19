"""Lazily reuse an executor within one kernel-solver invocation."""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
from threading import get_ident
import os


_SCOPE = ContextVar('dwd_residual_worker_scope', default=None)


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

    def drain(self):
        executor, self.executor = self.executor, None
        self.workers = None
        if executor is not None:
            executor.shutdown(wait=True)

    def __exit__(self, *exc):
        try:
            self.closed = True
            self.drain()
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
        with factory(max_workers=workers) as executor:
            yield executor
        return
    scope.busy = True
    try:
        if scope.workers != workers:
            scope.drain()
        if scope.executor is None:
            scope.executor = factory(max_workers=workers)
            scope.workers = workers
        try:
            yield scope.executor
        except BaseException:
            # The dispatcher has already set its call-local cancellation event.
            scope.drain()
            raise
    finally:
        scope.busy = False


def retire_for_serial_call():
    scope = _SCOPE.get()
    if scope is not None and scope.eligible():
        scope.drain()
