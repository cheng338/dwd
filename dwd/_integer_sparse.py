"""Exact integer coordinates before conversion to floating sparse inputs.

Only duplicate-bearing integer inputs without a proven safe floating bound
use arbitrary-precision summation. Floating, boolean and complex inputs are
handled by the caller under their existing working-representation policy.
"""
import numpy as np
from scipy.sparse import bsr_array, csc_array, csr_array


_EXACT_FLOAT64_INTEGER_LIMIT = 1 << 53
_COMPRESSED_VIEW = {"csr": csr_array, "csc": csc_array, "bsr": bsr_array}


def _ordinary_integer_cast(X):
    """Exactly the old cast-first operation, including output index policy."""
    return X.astype(np.float64)


def _fresh_compressed_is_canonical(X):
    # The three-array constructor preserves duplicates. Cache properties are
    # read on the fresh wrapper only, never on caller X. No value operation is
    # performed through the shared buffers.
    if X.format in ("csr", "csc"):
        # Same public fresh-view idiom as csr_is_canonical; metadata is private.
        return bool(_COMPRESSED_VIEW[X.format](X, copy=False).has_canonical_format)
    kwargs = {"blocksize": X.blocksize} if X.format == "bsr" else {}
    view = _COMPRESSED_VIEW[X.format](
        (X.data, X.indices, X.indptr), shape=X.shape, copy=False, **kwargs
    )
    return bool(view.has_canonical_format)


def _coo_is_sorted_unique(X):
    # Never read X.has_canonical_format, including for COO with stale flags.
    rows, cols = X.coords
    if len(rows) < 2:
        return True
    return bool(np.all(
        (rows[1:] > rows[:-1])
        | ((rows[1:] == rows[:-1]) & (cols[1:] > cols[:-1]))
    ))


def _dtype_absolute_bound(dtype):
    bits = 8 * dtype.itemsize
    # Include the signed minimum's magnitude; no np.abs on fixed-width data.
    return (1 << (bits - 1)) if dtype.kind == "i" else (1 << bits) - 1


def _groups(first, second):
    """Return sorting order, unique starts and sorted first coordinates."""
    order = np.lexsort((second, first))
    first_sorted = first[order]
    second_sorted = second[order]
    changed = np.empty(len(order), dtype=bool)
    changed[0] = True
    changed[1:] = ((first_sorted[1:] != first_sorted[:-1])
                   | (second_sorted[1:] != second_sorted[:-1]))
    return order, np.flatnonzero(changed), first_sorted


def _exact_group_values(values, order, starts):
    # Only the exceptional path pays for Python-integer objects. For BSR,
    # trailing block dimensions are retained and reduceat adds elementwise.
    objects = values[order].astype(object)
    totals = np.add.reduceat(objects, starts, axis=0)
    return np.asarray(totals, dtype=np.float64)


def _exact_compressed(X, count):
    # BSR's major/minor coordinates are block coordinates; no scalar expansion.
    n_major = len(X.indptr) - 1
    major = np.repeat(np.arange(n_major, dtype=np.intp), np.diff(X.indptr))
    minor = X.indices[:count]
    order, starts, major_sorted = _groups(major, minor)
    if len(starts) == count:
        # Unsorted-but-unique data do not need exact accumulation.
        return _ordinary_integer_cast(X)
    values = _exact_group_values(X.data[:count], order, starts)
    indices = minor[order][starts]
    indptr = np.empty_like(X.indptr)
    indptr[0] = 0
    np.cumsum(
        np.bincount(major_sorted[starts], minlength=n_major),
        dtype=indptr.dtype,
        out=indptr[1:],
    )
    kwargs = {"blocksize": X.blocksize} if X.format == "bsr" else {}
    result = type(X)((values, indices, indptr), shape=X.shape, **kwargs)
    # Only the newly constructed output receives flags. The ordering above is
    # sorted and unique; zero-valued coordinates/blocks are deliberately kept.
    result.has_sorted_indices = True
    result.has_canonical_format = True
    return result


def _exact_coo(X):
    rows, cols = X.coords
    order, starts, _ = _groups(rows, cols)
    if len(starts) == len(X.data):
        return _ordinary_integer_cast(X)
    values = _exact_group_values(X.data, order, starts)
    coords = (rows[order][starts], cols[order][starts])
    result = type(X)((values, coords), shape=X.shape)
    result.has_canonical_format = True
    return result


def _integer_sparse_to_float64(X):
    if X.ndim != 2 or X.format in ("dok", "dia"):
        return _ordinary_integer_cast(X)
    if X.format not in ("csr", "csc", "coo", "bsr", "lil"):
        # No new behavior/domain is claimed for future formats or subclasses.
        return _ordinary_integer_cast(X)

    count = (int(X.indptr[-1]) if X.format in _COMPRESSED_VIEW else int(X.nnz))
    if count == 0:
        return _ordinary_integer_cast(X)
    if count <= _EXACT_FLOAT64_INTEGER_LIMIT // _dtype_absolute_bound(X.dtype):
        # E.g. int8 permits 2**46 contributions; int32 permits 2**22.
        # This proves all intermediate group sums are exact without min/max.
        return _ordinary_integer_cast(X)

    if X.format == "lil":
        # In the inspected SciPy source this is flattening, not summation.
        # The resulting CSR owns its buffers; tolil may canonicalize that
        # private result but cannot touch caller rows/data lists.
        private = X.tocsr()
        return _integer_sparse_to_float64(private).tolil()

    if X.format in _COMPRESSED_VIEW and _fresh_compressed_is_canonical(X):
        return _ordinary_integer_cast(X)
    if X.format == "coo" and _coo_is_sorted_unique(X):
        return _ordinary_integer_cast(X)

    values = X.data[:count] if X.format in _COMPRESSED_VIEW else X.data
    bound = max(abs(int(values.min())), abs(int(values.max())))
    if bound <= _EXACT_FLOAT64_INTEGER_LIMIT // count:
        return _ordinary_integer_cast(X)

    if X.format in _COMPRESSED_VIEW:
        return _exact_compressed(X, count)
    return _exact_coo(X)


def prepare_integer_linear_query(X):
    """Normalize duplicate integers, retaining verified-unique query storage.

    The caller has already restricted this helper to sparse integer inputs.
    A unique coordinate only needs the dot product's usual scalar conversion;
    avoid allocating a float64 copy when exact coalescing has no work to do.
    """
    if X.ndim == 2:
        if X.format in ("dok", "dia"):
            return X
        if X.format in _COMPRESSED_VIEW and _fresh_compressed_is_canonical(X):
            return X
        if X.format == "coo" and _coo_is_sorted_unique(X):
            return X
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return _integer_sparse_to_float64(X)
