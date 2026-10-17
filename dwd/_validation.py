"""Validation shared by real-valued estimator inputs."""

import numpy as np
from sklearn.utils.validation import check_X_y


def contains_complex(value):
    """Recognize complex scalars hidden in an object-dtype array too.

    Ordinary numeric arrays need only a dtype check. Object arrays require
    checking their entries before a float cast can discard imaginary parts.
    """
    if np.iscomplexobj(value):
        return True
    values = np.asarray(value)
    if values.dtype.kind != 'O':
        return False
    pending, visited = [values], set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        for item in current.flat:
            if np.iscomplexobj(item):
                return True
            if isinstance(item, np.ndarray) and item.dtype.kind == 'O':
                pending.append(item)
    return False


def require_real(value, name):
    """Reject unsupported complex values without converting their real parts."""
    if contains_complex(value):
        raise ValueError(f'{name} must be real.')


def check_real_X_y(X, y, **kwargs):
    """Apply sklearn data validation and reject object-wrapped complex labels."""
    X, y = check_X_y(X, y, **kwargs)
    require_real(y, 'y')
    return X, y
