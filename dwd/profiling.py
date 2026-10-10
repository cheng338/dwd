"""Optional observations of the guarded native residual screen.

This module records dispatch and elapsed time, without changing residual
arithmetic, precision bounds, or the solver's subsequent acceptance checks.
"""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import asyncio
import threading
import time


_CURRENT = ContextVar('dwd_residual_profile', default=None)
_ROUTES = ('unsupported', 'pre_dispatch', 'compiled_dispatch',
           'compiled_values', 'scalar_fallback')


def _owner():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), None if task is None else id(task)


def _active_profile():
    profile = _CURRENT.get()
    if profile is None or not profile._active or profile._owner != _owner():
        return None
    return profile


class _ResidualCall:
    """One invocation, including partial work before a declined result."""
    def __init__(self):
        self.route = 'pre_dispatch'
        self.decline_reason = 'unspecified'
        self.compiled_outcome = None
        self.compiled_seconds = 0.0
        self.scalar_started = 0
        self.scalar_completed = 0


class ResidualProfile:
    """Read accumulated observations with :meth:`as_dict`.

    Instances are yielded by :func:`residual_profile`. The innermost active
    profile receives observations. Child threads and asyncio tasks must open
    their own scope: inherited context does not write into a parent's profile.
    Counts concern the native screen only, not all residual checks in a fit.
    """
    def __init__(self):
        self._owner = _owner()
        self._active = False
        self._counts = Counter()
        self._declines = Counter()
        self._routes = {route: Counter() for route in _ROUTES}
        self._compiled = Counter()
        self._wall_seconds = 0.0
        self._compiled_seconds = 0.0

    def _run(self, function, *args):
        sample = _ResidualCall()
        started = time.perf_counter()
        outcome = 'raised'
        try:
            result = function(*args, _sample=sample)
            outcome = 'declined' if result is None else 'bounded_return'
            return result
        finally:
            elapsed = time.perf_counter() - started
            self._counts['calls'] += 1
            self._counts[outcome] += 1
            self._counts['scalar_sumprod_calls_started'] += sample.scalar_started
            self._counts['scalar_sumprod_calls_completed'] += sample.scalar_completed
            self._wall_seconds += elapsed
            route = self._routes[sample.route]
            route['calls'] += 1
            route[outcome] += 1
            route['wall_seconds'] += elapsed
            if outcome == 'declined':
                self._declines[sample.decline_reason] += 1
            if sample.compiled_outcome is not None:
                self._compiled['calls'] += 1
                self._compiled[sample.compiled_outcome] += 1
                self._compiled_seconds += sample.compiled_seconds

    def as_dict(self):
        """Return an independent, JSON-compatible snapshot.

        A ``bounded_return`` means the native routine returned residuals and
        error allowances. It does not mean the solver accepted an original
        system solution. ``compiled_helper`` measures the existing dispatcher;
        its decline can mean an unavailable extension, a small problem, or a
        rejected input. A returned tuple does not by itself certify accuracy:
        the following precision screens can still decline it.

        Route times include the whole native invocation classified by its
        furthest route. They partition ``wall_seconds``; compiled-helper time
        is nested inside them and must not be added again. Portable residual
        work after a native decline, and time elsewhere in a fit, are excluded.
        Profiling has overhead and is intended for diagnosis, not default fit
        timing. Counts of scalar calls are actual attempted/completed calls,
        including partial work before a decline or exception.
        """
        outcome_keys = ('calls', 'bounded_return', 'declined', 'raised')
        return {
            'schema_version': 1,
            'scope': 'guarded_native_residual_screen',
            'counts': {key: int(self._counts[key]) for key in outcome_keys + (
                'scalar_sumprod_calls_started', 'scalar_sumprod_calls_completed')},
            'wall_seconds': self._wall_seconds,
            'routes': {
                name: {**{key: int(values[key]) for key in outcome_keys},
                       'wall_seconds': float(values['wall_seconds'])}
                for name, values in self._routes.items()},
            'decline_reasons': dict(self._declines),
            'compiled_helper': {
                **{key: int(self._compiled[key])
                   for key in ('calls', 'values_returned', 'declined', 'raised')},
                'wall_seconds': self._compiled_seconds},
        }


@contextmanager
def residual_profile():
    """Observe native residual dispatch in this thread and asyncio task.

    Example::

        from dwd.profiling import residual_profile
        with residual_profile() as profile:
            model.fit(X, y)
        observations = profile.as_dict()

    Each call creates a fresh profile. Nested scopes collect independently:
    the outer profile excludes inner calls and resumes after inner exit,
    including exceptional exit. Exceptions are never suppressed. No numerical
    functions or process-wide settings are replaced. Outside an active scope,
    no timer is read and no observation is allocated for a residual call.
    Parallel workers require their own scopes; results are not auto-merged.
    """
    profile = ResidualProfile()
    token = _CURRENT.set(profile)
    profile._active = True
    try:
        yield profile
    finally:
        # A copied Context can retain this object after the enclosing scope
        # exits. Such a stale binding must not restart observation.
        profile._active = False
        _CURRENT.reset(token)
