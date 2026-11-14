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


def validate_unregularized_design(X):
    """Refuse design rank loss when the represented penalty shift is zero."""
    n, p = X.shape
    if n < p + 1:
        raise ValueError('The zero-shift augmented design has fewer rows than columns; '
                         'the MM system is singular.')
    # Own the augmented design. Scaling columns changes neither its exact rank
    # nor the model, and avoids making a rank decision from the squared system.
    features = X.toarray() if hasattr(X, 'toarray') else X
    design = np.empty((n, p + 1), dtype=np.float64)
    design[:, 0] = 1.
    design[:, 1:] = features
    del features
    if not np.isfinite(design).all():
        raise FloatingPointError('Nonfinite zero-shift augmented design.')
    scales = np.max(np.abs(design), axis=0)
    if np.any(scales == 0):
        raise ValueError('The zero-shift augmented design has a zero column; '
                         'the MM system is singular.')
    design /= scales
    singular_values = np.linalg.svd(design, compute_uv=False)
    if not np.isfinite(singular_values).all():
        raise FloatingPointError('Nonfinite zero-shift design singular values.')
    threshold = np.finfo(np.float64).eps * max(design.shape) * singular_values[0]
    if singular_values[-1] <= threshold:
        raise ValueError('The zero-shift augmented design is rank deficient or '
                         'numerically unresolved at float64 precision.')
