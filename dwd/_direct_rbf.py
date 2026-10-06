"""Direct-distance evaluation for the optional named RBF policy.

Each represented pair is evaluated by the same coordinate-order calculation,
irrespective of row identity or surrounding batch.  Dense ordinary distances
use SciPy cdist; CSR rows use an index-ordered merge without dense conversion.
Exceptional range cases use exact represented-input squared distances before
multiplying by gamma.  This is not a universally correctly rounded kernel or a
PSD repair.  It deliberately changes the historical norm/dot arithmetic policy.
"""

from fractions import Fraction
import math
from numbers import Integral, Real

import numpy as np
from scipy.sparse import issparse
from scipy.spatial.distance import cdist


_CHECK_ELEMENTS = 65536
_MAX_DENSE_TILE_ELEMENTS = 262144
_TINY = np.finfo(np.float64).tiny


def _finite_dense(array):
    iterator = np.nditer(
        array, flags=['external_loop', 'buffered', 'zerosize_ok'],
        op_flags=['readonly'], buffersize=_CHECK_ELEMENTS)
    return all(np.isfinite(chunk).all() for chunk in iterator)


def _validated_features(value, name):
    if issparse(value):
        if value.format != 'csr':
            raise TypeError(f'{name} must be a float64 dense array or CSR matrix.')
        if value.dtype != np.dtype(np.float64) or value.ndim != 2:
            raise TypeError(f'{name} must have float64 dtype and two dimensions.')
        if not _finite_dense(value.data):
            raise ValueError(f'{name} must contain only finite values.')
        # Canonicalize a private copy: duplicate/unsorted CSR coordinates and
        # explicit zeros must never alter or become aliases of caller storage.
        result = value.copy()
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            result.sum_duplicates()
        result.sort_indices()
        result.eliminate_zeros()
        if not _finite_dense(result.data):
            raise ValueError(f'{name} has nonfinite values after CSR canonicalization.')
    else:
        result = np.asarray(value)
        if result.dtype != np.dtype(np.float64) or result.ndim != 2:
            raise TypeError(f'{name} must have float64 dtype and two dimensions.')
        if not _finite_dense(result):
            raise ValueError(f'{name} must contain only finite values.')
    if result.shape[1] < 1:
        raise ValueError(f'{name} must have at least one feature.')
    return result


def _row_items(array, index):
    if issparse(array):
        start, stop = array.indptr[index:index + 2]
        return ((int(j), float(v)) for j, v in
                zip(array.indices[start:stop], array.data[start:stop]))
    return ((j, float(v)) for j, v in enumerate(array[index]))


def _pair_coordinates(X, i, Y, j):
    """Yield value pairs in increasing feature order, with implicit CSR zeros."""
    x_iter, y_iter = _row_items(X, i), _row_items(Y, j)
    x, y = next(x_iter, None), next(y_iter, None)
    while x is not None or y is not None:
        if y is None or (x is not None and x[0] < y[0]):
            yield x[1], 0.0
            x = next(x_iter, None)
        elif x is None or y[0] < x[0]:
            yield 0.0, y[1]
            y = next(y_iter, None)
        else:
            yield x[1], y[1]
            x, y = next(x_iter, None), next(y_iter, None)


def _merged_squared_distance(X, i, Y, j):
    total = 0.0
    for x, y in _pair_coordinates(X, i, Y, j):
        delta = x - y
        total += delta * delta
    return total


def _exceptional_kernel(X, i, Y, j, gamma_fraction):
    squared = Fraction(0)
    for x, y in _pair_coordinates(X, i, Y, j):
        if x != y:
            difference = Fraction.from_float(x) - Fraction.from_float(y)
            squared += difference * difference
    if not squared:
        return 1.0
    exponent = gamma_fraction * squared
    try:
        represented_exponent = float(exponent)
    except OverflowError:
        return 0.0
    # One scalar libm operation per pair prevents batch-length-dependent SIMD
    # exp dispatch. Exact distance/exponent arithmetic does not imply exact exp.
    return math.exp(-represented_exponent)


def accurate_rbf(X, Y=None, gamma=None, *, block_size=256):
    """Return the direct-distance RBF matrix for finite float64 dense/CSR rows.

    ``gamma=None`` means 1/n_features; gamma=0 returns ones. ``block_size``
    bounds both dimensions of temporary distance tiles. Input validation uses
    bounded buffers, and CSR is copied/canonicalized without dense conversion.
    The required output still occupies n_X*n_Y float64 values. Dense tile row
    counts additionally account for the feature count; a single row can exceed
    the scratch target. Caller arrays are read-only from this function's view.

    Ordinary cdist and index-ordered sparse reductions are separate reference
    implementations. Equality across dense/CSR representations is tested, not
    guaranteed for every compiler. Within one representation/environment,
    pair arithmetic does not depend on self identity, row order or block shape.
    """
    if (isinstance(block_size, (bool, np.bool_)) or
            not isinstance(block_size, Integral) or block_size <= 0):
        raise ValueError('block_size must be a positive integer.')
    X = _validated_features(X, 'X')
    Y = X if Y is None else _validated_features(Y, 'Y')
    if X.shape[1] != Y.shape[1]:
        raise ValueError('X and Y must have the same feature count.')
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    elif (isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, Real)):
        raise ValueError('gamma must be a finite nonnegative real number.')
    try:
        gamma = float(gamma)
    except (OverflowError, ValueError) as error:
        raise ValueError('gamma must be a finite nonnegative real number.') from error
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError('gamma must be a finite nonnegative real number.')
    output = np.empty((X.shape[0], Y.shape[0]), dtype=np.float64)
    if gamma == 0:
        output.fill(1.0)
        return output
    gamma_fraction = Fraction.from_float(gamma)
    # This conservative threshold catches squared-distance underflow even when
    # cdist rounds a nonzero distance to zero. Scaling by gamma happens only
    # after the exact distance in those cases, so large gamma can recover it.
    small_distance = _TINY * X.shape[1]
    tile_rows = min(int(block_size), math.isqrt(_MAX_DENSE_TILE_ELEMENTS),
                    max(1, _MAX_DENSE_TILE_ELEMENTS // X.shape[1]))
    dense = not issparse(X) and not issparse(Y)
    for x_start in range(0, X.shape[0], tile_rows):
        x_stop = min(x_start + tile_rows, X.shape[0])
        for y_start in range(0, Y.shape[0], tile_rows):
            y_stop = min(y_start + tile_rows, Y.shape[0])
            distances = (cdist(X[x_start:x_stop], Y[y_start:y_stop],
                               metric='sqeuclidean') if dense else None)
            for i in range(x_start, x_stop):
                for j in range(y_start, y_stop):
                    distance = (float(distances[i-x_start, j-y_start])
                                if dense else _merged_squared_distance(X, i, Y, j))
                    if not math.isfinite(distance) or distance <= small_distance:
                        value = _exceptional_kernel(X, i, Y, j, gamma_fraction)
                    else:
                        value = math.exp(-(gamma * distance))
                    output[i, j] = value
    return output


# Alias for callers that identify the numerical policy explicitly.
direct_rbf = accurate_rbf
