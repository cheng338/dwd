"""Keep failed fits from exposing stale or partially updated model state."""

from functools import wraps


def _clear_fitted_state(estimator):
    # Public learned attributes follow sklearn's trailing-underscore convention.
    # Private precomputation caches have their own data/parameter validity checks
    # and remain reusable across tuning-path fits, including after a failure.
    for name in tuple(vars(estimator)):
        if ((name.endswith('_') and not name.startswith('_'))
                or name == '_Xfit'):
            delattr(estimator, name)


def fit_with_cleanup(fit):
    """Leave an estimator unfitted if any part of a new fit fails.

    Clear the old result before validation, then clear partial results on an
    exception or interruption. Preserve constructor parameters, checked private
    caches, the original exception, and the public fit signature.
    """
    @wraps(fit)
    def wrapped(self, *args, **kwargs):
        _clear_fitted_state(self)
        try:
            return fit(self, *args, **kwargs)
        except BaseException:
            _clear_fitted_state(self)
            raise

    return wrapped
