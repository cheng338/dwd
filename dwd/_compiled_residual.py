"""Optional compiled row arithmetic; numerical acceptance remains in Python."""
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np

try:
    from . import _residual_accel as _ACCEL
except (ImportError, OSError):
    _ACCEL = None


def _worker_count(n):
    """Respect the caller's active native-library thread budget."""
    from threadpoolctl import threadpool_info
    budgets = [int(pool['num_threads']) for pool in threadpool_info()
               if pool.get('user_api') == 'blas' and pool.get('num_threads', 0) > 0]
    budget = min(budgets) if budgets else 1
    return max(1, min(budget, os.cpu_count() or 1, n // 1024))


def compiled_values(K, x, rhs, shift, intercept):
    """Return scores, expanded residuals and row maxima, or decline.

    The optional extension releases the GIL. Threads read the same kernel and
    write disjoint output ranges; no extra Gram matrix or process is created.
    Its arithmetic guards are additional to the unchanged caller's checks.
    """
    if _ACCEL is None or len(x) < 256:
        return None
    n = len(x)
    scores, residual, maxima = np.empty(n), np.empty(n), np.empty(n)
    workers = _worker_count(n)

    def evaluate(part):
        start, stop = n * part // workers, n * (part + 1) // workers
        return _ACCEL.evaluate(K, x, rhs, float(shift), float(intercept), start, stop,
                               scores[start:stop], residual[start:stop], maxima[start:stop])

    if workers == 1:
        statuses = [evaluate(0)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            statuses = list(executor.map(evaluate, range(workers)))
    if any(status != 0 for status in statuses):
        return None
    return scores, residual, maxima
