"""Private sparse input promotion and feature-coordinate normalization."""

import numpy as np
from scipy.sparse import csr_array, issparse
from ._integer_sparse import _integer_sparse_to_float64


def promote_sparse_features(X, *, force_float64=False):
    """Normalize integer coordinates and select the sparse working dtype.

    Signed/unsigned integer duplicates are summed exactly, then each coordinate
    is converted once to float64. This includes precomputed inputs routed here;
    callable returned kernels and direct score helpers keep separate contracts.
    The final integer may still round, and earlier input loss is not repaired.
    This helper promotes integer storage; linear prediction separately retains
    verified-unique input for its ordinary dot-product dtype conversion.

    Floating and Boolean inputs retain cast-first behavior. Pairwise routines
    retain float32 when appropriate; corrected fitting requires float64. Leave
    complex inputs to public validation instead of discarding imaginary parts.
    """
    if not issparse(X) or X.dtype.kind == 'c':
        return X
    dtype = (np.float64 if force_float64 or X.dtype != np.dtype(np.float32)
             else np.float32)
    if X.dtype == np.dtype(dtype):
        return X
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        if X.dtype.kind in "iu":
            return _integer_sparse_to_float64(X)
        return X.astype(dtype)


def promote_kernel_query(X, basis):
    """Select named pairwise working dtype before sparse format conversion.

    Named pairwise kernels retain float32 only when both operands are float32.
    Promote mixed-precision sparse queries before conversion or duplicate
    aggregation can round their represented coordinates in float32. Integer
    inputs use promote_sparse_features' exact-coordinate rule at shared input
    promotion boundaries. Callable return values and direct score helpers
    retain their separate contracts.
    This does not convert fitted bases or repair historical duplicate bases.
    """
    return promote_sparse_features(
        X, force_float64=getattr(basis, 'dtype', None) != np.dtype(np.float32))


def promote_cosine_query(X, basis):
    """Compatibility name for cosine fit/validation preparation callers."""
    return promote_kernel_query(X, basis)


def csr_is_canonical(X):
    """Inspect CSR storage without trusting mutable cached ordering flags.

    A fresh public CSR view shares buffers but has its own metadata. SciPy's
    check scans indices in compiled code, without a dense or nnz-sized scratch
    array and without changing the caller's flags or buffers.
    """
    if not issparse(X) or X.format != 'csr':
        return True
    return csr_array(X, copy=False).has_canonical_format


def checked_prediction_basis(X, *, kernel_name='RBF'):
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
            f'Stored {kernel_name} training features contain duplicate sparse coordinates; '
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


def prepare_laplacian_features(X):
    """Protect CSR storage from the pairwise Manhattan in-place conversion.

    Leave summation, dtype selection and arithmetic errors to sklearn's
    existing calculation. Dense and non-CSR inputs retain their normal path.
    """
    if not issparse(X) or X.format != 'csr':
        return X
    if csr_is_canonical(X):
        return X
    return X.copy()
