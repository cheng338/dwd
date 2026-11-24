"""Dependency-floor CI probe for an environment with adjustable BLAS pools."""
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import dwd._compiled_residual as compiled


def test_worker_count_observes_live_blas_limits():
    matrix = np.ones((2, 2))
    np.dot(matrix, matrix)  # Load any lazy BLAS runtime before setting limits.
    for limit in (1, 2):
        with threadpool_limits(limits=limit, user_api='blas'):
            budgets = {pool['num_threads'] for pool in threadpool_info()
                       if pool['user_api'] == 'blas'}
            assert budgets == {limit}
            expected = min(limit, compiled._available_cpu_count())
            assert compiled._worker_count(4096) == expected
