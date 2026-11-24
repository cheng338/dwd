"""Parent resource scopes for sklearn searches that time their final refit."""
from contextlib import contextmanager

from joblib import parallel_config
from threadpoolctl import threadpool_limits


@contextmanager
def single_worker_search_parent():
    """Use one native thread and one default model job in the search parent.

    These numerical fixtures use estimators without explicit parallel worker
    settings. An explicit search.n_jobs still controls candidate/fold workers.
    """
    with parallel_config(n_jobs=1), threadpool_limits(limits=1):
        yield
