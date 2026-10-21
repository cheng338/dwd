"""Selectively repair cancellation-sensitive RBF distances.

This is a construction policy, not an eigenvalue/PSD repair. Its forward-error
envelope is a conservative trigger under ordinary floating-point assumptions;
it does not certify every unflagged distance or every downstream decision.
"""
import math
import numpy as np
from scipy.sparse import issparse
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from ._direct_rbf import _rbf_from_validated_features, _pair_coordinates

_EPS = np.finfo(float).eps
_TINY = np.finfo(float).tiny
_TILE = 256
_FEATURE_SCRATCH = 262144


def _factor(p):
    operations = (2 * p + 3) * _EPS
    g = operations / (1.0 - operations) if operations < .5 else math.inf
    return 2.0 * g / (1.0 - g) if g < 1 else math.inf


def _all_clear(distances, x_norm, y_norm, p):
    # The row-norm envelope is monotone in each nonnegative norm. Use the
    # identical operation ordering and underflow allowance as the tile path.
    # A finite maximum rejects NaN/overflow distances without a matrix-sized
    # boolean temporary. Reductions do not modify represented distances.
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        bound = (_factor(p) / math.sqrt(_EPS)) * (np.max(x_norm) + np.max(y_norm))
        bound += (2 * p + 3) * _TINY / math.sqrt(_EPS)
        return math.isfinite(float(np.max(distances))) and np.min(distances) > bound


def _packed_flags(distances, x_norm, y_norm, p):
    if _all_clear(distances, x_norm, y_norm, p):
        return None
    n, m = distances.shape
    packed = np.zeros((n, (m+7)//8), dtype=np.uint8)
    factor = _factor(p) / math.sqrt(_EPS)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        allowance = (2*p+3)*_TINY / math.sqrt(_EPS)
        bound = factor*(np.max(x_norm)+np.max(y_norm))+allowance
        # Finite nonnegative norms and a finite upper envelope establish every
        # local envelope is finite. A finite distance maximum excludes NaN/+Inf;
        # any -Inf distance still satisfies the original <= predicate.
        finite = (type(x_norm) is np.ndarray and type(y_norm) is np.ndarray
                  and x_norm.dtype == y_norm.dtype == np.dtype('float64')
                  and math.isfinite(float(bound)) and math.isfinite(factor) and factor >= 0
                  and math.isfinite(allowance) and allowance >= 0
                  and np.min(x_norm) >= 0 and np.min(y_norm) >= 0
                  and math.isfinite(float(np.max(distances))))
        for a in range(0,n,_TILE):
            b=min(a+_TILE,n)
            for c in range(0,m,_TILE):
                d=min(c+_TILE,m)
                block=distances[a:b,c:d]
                # Identical norm objects only select a useful scanning strategy.
                # The same conservative bound proves each skipped tile is clear.
                if finite and (x_norm is y_norm) and np.min(block) > bound:
                    continue
                envelope=factor*(x_norm[a:b,None]+y_norm[None,c:d])
                envelope+=allowance
                flags=(block<=envelope) if finite else (~np.isfinite(block)|~np.isfinite(envelope)|(block<=envelope))
                packed[a:b,c//8:(d+7)//8]=np.packbits(flags,axis=1)
    return packed


def _equal(X, i, Y, j):
    if not issparse(X) and not issparse(Y):
        return np.array_equal(X[i], Y[j])
    # Canonical sparse storage may retain explicit zeros, so index-array
    # equality alone is insufficient. No sparse feature array is densified.
    return all(a == b for a, b in _pair_coordinates(X, i, Y, j))


def _apply_pairs(K, packed, X, Y, gamma):
    # After public pairwise validation, identity proves each same-index row
    # denotes exactly the same represented feature vector. This is narrower
    # than self_kernel metadata, equal shapes, or copied equal-valued arrays.
    # Off-diagonal duplicates and every cross-query pair keep their guards.
    if X is Y:
        np.fill_diagonal(K, 1.0)
        if packed is not None:
            rows = np.arange(K.shape[0])
            masks = np.right_shift(np.uint8(128), (rows % 8).astype(np.uint8))
            packed[rows, rows // 8] &= np.bitwise_not(masks)
    if packed is None or not packed.any():
        return K
    limit = max(1, min(256, _FEATURE_SCRATCH // X.shape[1]))
    dense = not issparse(X) and not issparse(Y)
    # The reduction scans packed bytes once; nonzero rows remain in
    # ascending order, as in the original per-row loop.
    for i in np.flatnonzero(packed.any(axis=1)):
        columns = np.flatnonzero(np.unpackbits(packed[i], count=K.shape[1]))
        for start in range(0, len(columns), limit):
            selected = columns[start:start + limit]
            if dense:
                rows = Y[selected]
                identical = np.all(rows == X[i], axis=1)
            else:
                identical = np.fromiter((_equal(X, i, Y, int(j)) for j in selected),
                                        dtype=bool, count=len(selected))
            K[i, selected[identical]] = 1.0
            distinct = selected[~identical]
            if len(distinct):
                # Inputs already passed the public finite/dtype checks and CSR
                # normalization. Validate no full basis here. Only the selected
                # rows are copied, within the explicit feature scratch target.
                K[i, distinct] = _rbf_from_validated_features(
                    X[i:i+1], rows[~identical] if dense else Y[distinct], gamma, 256)[0]
    return K


def guarded_rbf(X, Y, *, gamma=None, x_norm=None, mode='sklearn', self_kernel=False):
    """Evaluate qualified, validated/canonical named RBF inputs only."""
    X, Y = check_pairwise_arrays(X, Y)
    gamma = 1.0 / X.shape[1] if gamma is None else float(gamma)
    if gamma == 0.0:
        return np.ones((X.shape[0], Y.shape[0]), dtype=float)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        x_norm = row_norms(X, squared=True) if x_norm is None else x_norm
        y_norm = x_norm if Y is X else row_norms(Y, squared=True)
        if mode == 'norm_sum':
            K = safe_sparse_dot(X, Y.T, dense_output=True)
            K *= -2.0
            for start in range(0, len(x_norm), 128):
                stop = min(start + 128, len(x_norm))
                K[start:stop] += x_norm[start:stop, None] + y_norm[None, :]
            np.maximum(K, 0.0, out=K)
        else:
            # The public distance validator rejects supplied overflowed norms,
            # although finite features are valid. Let its original internal
            # norm calculation run in that range, then repair the flagged pairs.
            K = euclidean_distances(
                X, Y, X_norm_squared=x_norm if np.isfinite(x_norm).all() else None,
                Y_norm_squared=y_norm if np.isfinite(y_norm).all() else None,
                squared=True)
        packed = _packed_flags(K, x_norm, y_norm, X.shape[1])
        K *= -gamma
        np.exp(K, out=K)
    _apply_pairs(K, packed, X, Y, gamma)
    if mode == 'norm_sum' and self_kernel:
        for start in range(0, len(K), 128):
            stop = min(start + 128, len(K))
            block = K[start:stop, start:stop]
            lower = np.tril_indices(stop-start, -1)
            block[lower] = block.T[lower]
            K[stop:, start:stop] = K[start:stop, stop:].T
        np.fill_diagonal(K, 1.0)
    return K
