"""Optional compiled row arithmetic; numerical acceptance remains in Python."""
from concurrent.futures import ThreadPoolExecutor, wait
import os
from threading import Event
from types import BuiltinFunctionType
from ._residual_worker_scope import executor_for_call, retire_for_serial_call
from ._residual_startup_cleanup import drain_started_before_registration

_DEFAULT_EXECUTOR = ThreadPoolExecutor

import numpy as np

try:
    from . import _residual_accel as _ACCEL
except (ImportError, OSError):
    _ACCEL = None


_COMPILED_DOMAIN_REFUSAL = object()
_TRUSTED_ACCEL = _ACCEL
_TRUSTED_EVALUATE = getattr(_ACCEL, "evaluate", None)


# Bound time spent in a native call without changing row arithmetic. One row is
# indivisible, so the work bound is best effort when its width alone exceeds it.
_CHUNK_MAX_ROWS = 64
_CHUNK_MAX_TERMS = 1 << 20
_SMALL_SINGLE_CALL_ROWS = 256
_CANCEL_POLL_SECONDS = .01


def _chunk_rows(n):
    if n <= _SMALL_SINGLE_CALL_ROWS:
        return n
    return max(1, min(_CHUNK_MAX_ROWS, _CHUNK_MAX_TERMS // n))


def _worker_count(n):
    """Respect the caller's active native-library thread budget."""
    # Fewer than 2048 rows cannot use two workers under the row budget.
    # Avoid enumerating loaded BLAS libraries for every small residual check.
    if n < 2048:
        return 1
    from threadpoolctl import threadpool_info
    budgets = [int(pool['num_threads']) for pool in threadpool_info()
               if pool.get('user_api') == 'blas' and pool.get('num_threads', 0) > 0]
    budget = min(budgets) if budgets else 1
    return max(1, min(budget, os.cpu_count() or 1, n // 1024))


def compiled_values(K, x, rhs, shift, intercept):
    """Preserve the existing tuple-or-None dispatcher interface."""
    outcome = _compiled_values_or_refusal(K, x, rhs, shift, intercept)
    return None if outcome is _COMPILED_DOMAIN_REFUSAL else outcome


_DEFAULT_COMPILED_VALUES = compiled_values


def _compiled_values_or_refusal(K, x, rhs, shift, intercept):
    """Return scores, expanded residuals and row maxima, or decline.

    The optional extension releases the GIL. Threads read the same kernel and
    write disjoint output ranges; no extra Gram matrix or process is created.
    Its arithmetic guards are additional to the unchanged caller's checks.
    """
    # The scalar route has less setup overhead for tiny vectors.
    if _ACCEL is None or len(x) < 8:
        return None
    n = len(x)
    scores, residual, maxima = np.empty(n), np.empty(n), np.empty(n)
    workers = _worker_count(n)
    chunk_rows = _chunk_rows(n)
    cancelled = Event() if workers > 1 else None

    def evaluate(part):
        start, stop = n * part // workers, n * (part + 1) // workers
        try:
            native_shift, native_intercept = float(shift), float(intercept)
            for first in range(start, stop, chunk_rows):
                if cancelled is not None and cancelled.is_set():
                    return 1
                last = min(stop, first + chunk_rows)
                owner = _ACCEL
                evaluator = owner.evaluate
                status = evaluator(
                    K, x, rhs, native_shift, native_intercept, first, last,
                    scores[first:last], residual[first:last], maxima[first:last])
                if status != 0:
                    if cancelled is not None:
                        cancelled.set()
                    if (status == 2 and owner is _TRUSTED_ACCEL
                            and evaluator is _TRUSTED_EVALUATE
                            and isinstance(evaluator, BuiltinFunctionType)):
                        return _COMPILED_DOMAIN_REFUSAL
                    return status
            return 0
        except BaseException:
            if cancelled is not None:
                cancelled.set()
            raise

    if workers == 1:
        retire_for_serial_call()
        statuses = [evaluate(0)]
    else:
        primary_error = None
        try:
            with executor_for_call(workers, ThreadPoolExecutor, _DEFAULT_EXECUTOR,
                                   type(shift) is float and type(intercept) is float and
                                   _ACCEL is _TRUSTED_ACCEL and
                                   getattr(_ACCEL, "evaluate", None) is _TRUSTED_EVALUATE) as executor:
                try:
                    futures = [executor.submit(evaluate, part) for part in range(workers)]
                    pending = set(futures)
                    while pending:
                        # A pending Python signal need not wake an indefinite
                        # Future.result wait on Windows. Return to Python regularly.
                        _, pending = wait(pending, timeout=_CANCEL_POLL_SECONDS)
                    statuses = [future.result() for future in futures]
                except BaseException as error:
                    # Stop new native chunks before shutdown joins the workers.
                    # Preserve the original exception and discard all outputs.
                    cancelled.set()
                    primary_error = error
                    try:
                        cleanup = drain_started_before_registration(executor, error)
                        error.add_note('Interrupted-start recovery: ' + repr(cleanup))
                    except BaseException as cleanup_error:
                        error.add_note('Startup cleanup failed; worker drain is not established. '
                                       'See the chained cleanup error.')
                        raise error from cleanup_error
                    raise
        except BaseException as context_error:
            if primary_error is not None and context_error is not primary_error:
                primary_error.add_note('Executor context shutdown also failed; final drain is not '
                                       'established. See the chained context error.')
                raise primary_error from context_error
            raise
    # Join every worker and propagate exceptions before interpreting refusal.
    if any(status is _COMPILED_DOMAIN_REFUSAL for status in statuses):
        return _COMPILED_DOMAIN_REFUSAL
    if any(status != 0 for status in statuses):
        return None
    return scores, residual, maxima
