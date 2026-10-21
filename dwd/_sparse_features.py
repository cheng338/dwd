"""Private sparse-feature normalization for squared-distance calculations."""

import numpy as np
from scipy.sparse import csr_array, issparse


def promote_sparse_features(X, *, force_float64=False):
    """Promote before a format conversion can sum duplicate coordinates.

    sklearn's pairwise routines retain float32 inputs and otherwise use
    float64. Corrected kernel fitting already requires float64. Leave complex
    inputs to the public validator instead of discarding imaginary parts.
    """
    if not issparse(X) or X.dtype.kind == 'c':
        return X
    dtype = (np.float64 if force_float64 or X.dtype != np.dtype(np.float32)
             else np.float32)
    if X.dtype == np.dtype(dtype):
        return X
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        return X.astype(dtype)


def csr_is_canonical(X):
    """Inspect CSR storage without trusting mutable cached ordering flags.

    A fresh public CSR view shares buffers but has its own metadata. SciPy's
    check scans indices in compiled code, without a dense or nnz-sized scratch
    array and without changing the caller's flags or buffers.
    """
    if not issparse(X) or X.format != 'csr':
        return True
    return csr_array(X, copy=False).has_canonical_format


def checked_prediction_basis(X):
    """Reject old norm/dot fitted bases with duplicate training coordinates.

    A model trained before sparse normalization may have coefficients for an
    incorrect Gram matrix. Query-time normalization cannot repair that fit.
    Unique unsorted coordinates are safe to sort privately; direct-distance
    fitted models use their separate, already normalized construction path.
    """
    if not issparse(X) or X.format != 'csr' or csr_is_canonical(X):
        return X
    result = X.copy()
    # Do not sum coordinates: refusal must also cover duplicate overflow, and
    # cached SciPy metadata may have become stale after index-array mutation.
    result.has_sorted_indices = False
    result.has_canonical_format = False
    result.sort_indices()
    if not csr_is_canonical(result):
        raise ValueError(
            'Stored RBF training features contain duplicate sparse coordinates; '
            'the fitted coefficients may use an incorrect historical kernel. '
            'Refit the model with the original training data, labels and settings '
            'before prediction.')
    return result


def canonical_sparse_features(X, *, copy=True):
    """Sum CSR duplicates before norms, privately unless storage is owned.

    Squaring individual stored entries does not compute a represented row's
    norm when coordinates repeat. Explicit zeros are retained. This helper is
    for feature distances, not caller-supplied/precomputed kernel matrices.
    """
    if not issparse(X) or X.format != 'csr' or csr_is_canonical(X):
        return X
    if X.dtype.kind == 'c':
        raise ValueError('Sparse features must be real.')
    result = promote_sparse_features(X)
    if result is X and copy:
        result = X.copy()
    # Mutable indices can invalidate SciPy's cached flags. Force both checks
    # on the owned result rather than trusting flags copied with old storage.
    result.has_sorted_indices = False
    result.has_canonical_format = False
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        result.sum_duplicates()
    for start in range(0, len(result.data), 65536):
        if not np.isfinite(result.data[start:start + 65536]).all():
            raise ValueError('Sparse features have nonfinite values after summing duplicate coordinates.')
    return result
