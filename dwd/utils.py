import numpy as np
from collections.abc import Mapping


def parameters_equal(left, right):
    """Compare cached kernel parameters without ambiguous NumPy truth values.

    Container kinds and array dtypes are part of the cache key. Unknown objects
    must supply a scalar equality result; otherwise the cache is invalidated.
    Stateful callable kernels must be treated as immutable while cached.
    """
    if left is right:
        return True
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            return False
        if left.dtype != right.dtype or left.shape != right.shape:
            return False
        if left.dtype.hasobject:
            return all(parameters_equal(a, b) for a, b in zip(left.flat, right.flat))
        return bool(np.array_equal(left, right))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping)):
            return False
        return (left.keys() == right.keys() and
                all(parameters_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right) and
                all(parameters_equal(a, b) for a, b in zip(left, right)))
    try:
        result = left == right
        return bool(result) if np.ndim(result) == 0 else False
    except (TypeError, ValueError):
        return False


def pm1(y):
    """
    Converts binary label vector y into +/- 1s

    Parameters
    ----------
    y: array-like, (n_samples, )
        The original labels. Must have binary labels.

    Output
    ------
    y: array-like, (n_samples, )
        y with +/- 1 values. Note the positive label is determined
        by np.unique(y)[1].
    """
    y = np.array(y).reshape(-1)
    labels = np.unique(y)

    # check binary
    if len(labels) != 2:
        raise ValueError('y must have binary labels;'
                         ' found {} labels'.format(len(labels)))

    y_pm1 = np.ones(len(y))
    y_pm1[y == labels[0]] = -1
    return y_pm1
