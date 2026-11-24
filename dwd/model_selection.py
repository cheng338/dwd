"""Explicit search helpers for DWD estimators.

fit_shared_fold_search shares private, immutable fold snapshots between
eligible serial KernMD candidates. KernMD.fit and sklearn search algorithms
remain unchanged.
"""

from collections import deque, namedtuple
from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from copyreg import __newobj_ex__, dispatch_table as registered_reducers
from datetime import date, timedelta
from fractions import Fraction
from functools import lru_cache, partial
from inspect import CO_VARARGS, CO_VARKEYWORDS, getattr_static, isdatadescriptor
from io import BytesIO
from itertools import chain, combinations, zip_longest
from mmap import mmap
from numbers import Integral
from os import urandom
from pickle import HIGHEST_PROTOCOL, Pickler, PicklingError, Unpickler, UnpicklingError, dumps, loads
from sys import modules
from threading import Lock
from types import CodeType, FunctionType, MethodType
from weakref import WeakValueDictionary, ref

import joblib
import numpy as np
from joblib import Parallel
from joblib._parallel_backends import LokyBackend, MultiprocessingBackend
from joblib.parallel import BatchedCalls, get_active_backend
from sklearn import get_config
from sklearn.base import _fit_context
from sklearn.model_selection import GridSearchCV, ParameterGrid, _search, check_cv
from sklearn.metrics import check_scoring

from ._fit_state import fit_with_cleanup
from ._validation import _boxed_sparse_scalars, _pandas_target_array, _temporal_label_key, check_real_X_y
from .kern_md import KernMD, kern_md


__all__ = ['fit_shared_fold_search']
_MISSING = object()
# Only a recognized native temporary reconstruction uses the copy-on-write optimization.
_JOBLIB_TEMP_LOADER = getattr(getattr(joblib, 'numpy_pickle', None), 'load_temporary_memmap', None)
_JOBLIB_MAP_OPENER = getattr(getattr(joblib, '_memmapping_reducer', None), '_strided_from_memmap', None)
_ITERABLE_CV = type(check_cv(()))
_FIT_WRAPPER_CODE = _fit_context(prefer_skip_nested_validation=False)(lambda: None).__code__


def _kernel_supported(kernel, kwargs):
    if type(kernel) is not str or kernel not in ('linear', 'rbf'):
        return False
    if kwargs is None:
        return True
    if type(kwargs) is not dict:
        return False
    if not kwargs:
        return True
    if kernel != 'rbf' or set(kwargs) != {'gamma'} or type(kwargs['gamma']) not in (int, float):
        return False
    try:
        return bool(np.isfinite(float(kwargs['gamma'])) and kwargs['gamma'] > 0)
    except (OverflowError, TypeError, ValueError):
        return False


def _eligibility(search, X, y, fit_params):
    if type(search) is not GridSearchCV or type(search.estimator) is not KernMD:
        return 'search_or_estimator_type'
    if (type(search.n_jobs) is not int or search.n_jobs != 1
            or not (search.scoring is None or type(search.scoring) is str
                    and search.scoring == 'accuracy')
            or type(search.refit) is not bool):
        return 'search_options'
    if (fit_params or get_config().get('enable_metadata_routing', False)
            or get_config().get('array_api_dispatch', False)
            or getattr(search, '_skl_callbacks', None)
            or hasattr(search, '_parent_callback_ctx')
            or getattr(search.estimator, '_skl_callbacks', None)):
        return 'callbacks_or_metadata'
    # Keep unsupported array-like validation on the ordinary sklearn path.
    if (type(X) is not np.ndarray or X.ndim != 2 or X.dtype != np.dtype('float64')
            or not all(X.shape) or not np.isfinite(X).all()
            or type(y) is not np.ndarray or y.ndim != 1 or len(y) != len(X)
            or y.dtype.kind not in 'biuf' or not np.isfinite(y).all()):
        return 'input_scope'
    if type(search.cv) not in (list, tuple) or not search.cv:
        return 'explicit_folds_required'
    for pair in search.cv:
        if type(pair) not in (list, tuple) or len(pair) != 2:
            return 'fold_scope'
        for rows in pair:
            if (type(rows) is not np.ndarray or rows.ndim != 1
                    or rows.dtype.kind not in 'iu' or not rows.size
                    or np.any(rows >= len(X)) or np.any(rows < 0)):
                return 'fold_scope'
    grids = search.param_grid
    if type(grids) is dict:
        grids = [grids]
    if (type(grids) is not list or not grids
            or any(type(grid) is not dict for grid in grids)
            or any(type(key) is not str or type(values) not in (list, tuple)
                   for grid in grids for key, values in grid.items())):
        return 'grid_scope'
    try:
        candidates = list(ParameterGrid(search.param_grid))
    except (TypeError, ValueError):
        return 'grid_scope'
    base = search.estimator.get_params(deep=False)
    if (base['naive_bayes'] is not False
            or not _kernel_supported(base['kernel'], base['kernel_kws'])):
        return 'kernel_or_parameter_scope'
    for candidate in candidates:
        if not set(candidate).issubset({'kernel', 'kernel_kws', 'naive_bayes'}):
            return 'parameter_scope'
        params = dict(base, **candidate)
        if (params['naive_bayes'] is not False
                or not _kernel_supported(params['kernel'], params['kernel_kws'])):
            return 'kernel_or_parameter_scope'
    return None


class _FrozenMatrix:
    """Bytes own the data; every access gets its own ndarray shape metadata."""

    def __init__(self, X):
        self.shape = tuple(X.shape)
        self.dtype = X.dtype
        # tobytes makes an immutable snapshot even for readonly caller aliases.
        self._bytes = X.tobytes(order='C')

    @classmethod
    def _try_contiguous_rows(cls, root, indices):
        """Return one immutable root span, or decline to ordinary row gathering."""
        if (type(root._bytes) is not bytes or len(root.shape) != 2
                or root.dtype != np.dtype('float64')
                or type(indices) is not np.ndarray or indices.ndim != 1
                or indices.dtype != np.dtype(np.intp) or not indices.size):
            return None
        first, last = int(indices[0]), int(indices[-1])
        if first < 0 or last >= root.shape[0] or last - first != len(indices) - 1:
            return None
        # Endpoints alone do not prove order, uniqueness or interior bounds.
        if not np.array_equal(indices, np.arange(first, last + 1, dtype=np.intp)):
            return None
        row_bytes = root.shape[1] * root.dtype.itemsize
        frozen = cls.__new__(cls)
        frozen.shape = (len(indices), root.shape[1])
        frozen.dtype = root.dtype
        frozen._bytes = memoryview(root._bytes)[first * row_bytes:(last + 1) * row_bytes]
        return frozen

    def view(self):
        return np.frombuffer(self._bytes, dtype=self.dtype).reshape(self.shape)


class _FoldToken:
    """Unexported, registered capability for one private cached row subset."""

    def __init__(self, store, key, frozen):
        self._store = store
        self._key = key
        self._frozen = frozen
        self.shape = frozen.shape
        self.dtype = frozen.dtype
        self.ndim = 2

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        array = self._frozen.view()
        if dtype is not None and np.dtype(dtype) != array.dtype:
            if copy is False:
                raise ValueError('A dtype conversion requires a copy.')
            return array.astype(dtype)
        return array.copy() if copy else array

    def _borrow(self):
        if self._store._tokens.get(self._key) is not self:
            raise ValueError('Unregistered private fold token.')
        return self._frozen.view()


class _FoldDataset:
    """Only row indexing is supported; it is the inspected sklearn boundary."""

    def __init__(self, X):
        self._root = _FrozenMatrix(X)
        self.shape = self._root.shape
        self.dtype = self._root.dtype
        self.ndim = 2
        self._tokens = {}
        self._index_calls = 0
        self._cache_hits = 0
        self._borrows = 0

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        array = self._root.view()
        if dtype is not None and np.dtype(dtype) != array.dtype:
            if copy is False:
                raise ValueError('A dtype conversion requires a copy.')
            return array.astype(dtype)
        return array.copy() if copy else array

    def __getitem__(self, key):
        if type(key) is not tuple or len(key) != 2 or key[1] is not Ellipsis:
            raise TypeError('Private fold store supports row indexing only.')
        rows = key[0]
        if type(rows) is not np.ndarray or rows.ndim != 1 or rows.dtype.kind not in 'iu':
            raise TypeError('Private fold store requires explicit integer indices.')
        # Canonical intp keys retain order and duplicates. No value-based matching.
        indices = rows.astype(np.intp, copy=False)
        cache_key = (len(indices), indices.tobytes())
        self._index_calls += 1
        token = self._tokens.get(cache_key)
        if token is not None:
            self._cache_hits += 1
            return token
        frozen = _FrozenMatrix._try_contiguous_rows(self._root, indices)
        if frozen is None:
            subset = self._root.view()[indices, :]
            frozen = _FrozenMatrix(subset)
        token = _FoldToken(self, cache_key, frozen)
        self._tokens[cache_key] = token
        return token

    def info(self):
        return {
                'index_calls': self._index_calls, 'cache_hits': self._cache_hits,
                'cached_subsets': len(self._tokens), 'borrowed_fits': self._borrows,
                'root_bytes': len(self._root._bytes),
                'cached_bytes': sum(len(t._frozen._bytes) for t in self._tokens.values()),
                'storage_scope': ('Logical root and cached-subset payload bytes during this call; '
                                  'cached subsets may share immutable root storage; '
                                  'not distinct allocations, retained memory or process peak')}


class _SharedFoldKernMD(KernMD):
    """Only trusted fold tokens bypass a copy; public fit still owns storage."""

    @fit_with_cleanup
    def fit(self, X, y):
        if type(X) is not _FoldToken or not _kernel_supported(self.kernel, self.kernel_kws):
            return super().fit(X, y)
        basis = X._borrow()
        checked, y = check_real_X_y(basis, y, dtype='numeric')
        if checked is not basis:
            # Unexpected validator conversion has no private adoption contract.
            return super().fit(checked, y)
        X._store._borrows += 1
        self.classes_ = np.unique(y)
        self.n_features_in_ = checked.shape[1]
        self.kernel_ = self.kernel
        self.kernel_kws_ = deepcopy(self.kernel_kws)
        self._Xfit = checked
        K = self._compute_kernel(self._Xfit)
        self.dual_coef_, self.intercept_ = kern_md(K, y, naive_bayes=self.naive_bayes)
        self.intercept_ = np.atleast_1d(self.intercept_)
        self.dual_coef_ = self.dual_coef_.reshape(1, -1)
        return self


def _publicize_best(search):
    fitted = vars(search).get('best_estimator_')
    if type(fitted) is _SharedFoldKernMD:
        public = KernMD(**fitted.get_params(deep=False))
        # Final refit already used canonical fit, including its ownership copy
        # and failed-fit cleanup. Preserve that state without another fit.
        public.__dict__.update(vars(fitted))
        search.best_estimator_ = public


def _restore_scorer(search, previous):
    if previous is _MISSING:
        vars(search).pop('scorer_', None)
    else:
        search.scorer_ = previous


def _note_cleanup_error(error, cleanup_error):
    # A secondary cleanup problem must not replace the original fit error.
    try:
        error.add_note('Shared-fold cleanup also raised '
                       + type(cleanup_error).__name__ + '; the original exception is retained.')
    except BaseException:
        pass


def _uses_worker_processes(parallelism):
    if isinstance(parallelism, Parallel):
        # sklearn has already entered this executor. Parallel.__call__ uses
        # its configured backend and effective count even inside a new context.
        return (not getattr(parallelism._backend, 'supports_sharedmem', False)
                and parallelism._effective_n_jobs() > 1)
    backend, configured_jobs = get_active_backend()
    if getattr(backend, 'supports_sharedmem', False):
        return False
    requested_jobs = configured_jobs if parallelism is None else parallelism
    if requested_jobs is None:
        requested_jobs = backend.default_n_jobs
    # Parallel converts a context count to int before configuring its backend.
    try:
        jobs = int(requested_jobs)
    except (TypeError, ValueError, OverflowError):
        return False  # Let the stock fit report invalid settings in its usual order.
    return jobs not in (0, 1) and backend.effective_n_jobs(jobs) > 1


def _backing_memmap(value):
    """Follow the same array base links joblib uses to reopen a NumPy map."""
    root = value
    seen = set()
    while not isinstance(base := getattr(root, 'base', None), mmap):
        if base is None or id(base) in seen:
            return None
        seen.add(id(base))
        root = base
    return root if isinstance(root, np.memmap) else None


def _reject_mapped_objects(value, n_jobs):
    # Read storage metadata only. Object slots in a file cannot safely be
    # interpreted as Python references by a process reopening that file.
    if (isinstance(value, np.ndarray) and value.dtype.hasobject and value.base is not None
            and _backing_memmap(value) is not None and _uses_worker_processes(n_jobs)):
        raise ValueError('Process search cannot use file-backed object arrays. '
                         'Use an in-memory object array, or supply folds as a list '
                         'or tuple of numeric train/test index arrays.')


def _map_mode_text(mode):
    """Read a plain mode string without invoking custom string comparisons."""
    kind = type(mode)
    if kind is str:
        return mode
    if kind is np.str_:
        return str.__str__(mode)
    if issubclass(kind, str):
        base = np.str_ if issubclass(kind, np.str_) else str
        equal = getattr_static(kind, '__eq__')
        unequal = getattr_static(kind, '__ne__')
        if ((equal is str.__eq__ or equal is base.__eq__)
                and (unequal is str.__ne__ or unequal is base.__ne__)):
            return str.__str__(mode)
    return None


def _unsafe_mapped_layout(value, root, *, mode=_MISSING, fortran=None):
    # Reopening a copy-on-write mapping loses parent-private edits in any layout.
    # For other mappings joblib omits strides for contiguous views and uses the
    # backing mapping's order.
    if (root.mode if mode is _MISSING else mode) != 'c':
        if value.flags.c_contiguous or value.flags.f_contiguous:
            mapped_fortran = root.ndim > 1 and root.flags.f_contiguous if fortran is None else fortran
            if (value.flags.c_contiguous == value.flags.f_contiguous
                    or value.flags.f_contiguous == mapped_fortran):
                return False
        elif min(value.strides) >= 0 or not any(
                stride < 0 and size > 1 for stride, size in zip(value.strides, value.shape)):
            return False

    return True


def _needs_process_snapshot(value, n_jobs, mapped_modes=None):
    """Check the mapping against the executor that would transport it."""
    if type(value) not in (np.ndarray, np.memmap) or value.base is None:
        return False
    root = _backing_memmap(value)
    if root is None:
        return False
    mode = root.mode
    if not _unsafe_mapped_layout(value, root, mode=mode) or not _uses_worker_processes(n_jobs):
        return False
    if mapped_modes is not None:
        mapped_modes[id(root)] = _map_mode_text(mode)
    return True


def _snapshot_search_input(value):
    detached = np.array(value, order='K', copy=True)
    detached.flags.writeable = value.flags.writeable
    return detached


def _parallel_search_input(value, n_jobs, mapped_modes=None):
    """Detach mapped arrays that reopening the file in a worker cannot reproduce."""
    if isinstance(value, np.ndarray) and value.dtype.hasobject:
        _reject_mapped_objects(value, n_jobs)
        return value
    return _snapshot_search_input(value) if _needs_process_snapshot(value, n_jobs, mapped_modes) else value


def _parallel_pandas_values(values, pd, n_jobs, check_only, bind=None, parents=()):
    if isinstance(values, np.ndarray):
        if bind is not None:
            bind(values, parents)
            return values
        if check_only:
            _reject_mapped_objects(values, n_jobs)
            return values
        return _parallel_search_input(values, n_jobs)
    if bind is not None:
        parents += (values,)
    kind = type(values)
    if kind is pd.arrays.IntervalArray:
        left = _parallel_pandas_values(values._left, pd, n_jobs, check_only, bind, parents)
        right = _parallel_pandas_values(values._right, pd, n_jobs, check_only, bind, parents)
        if left is values._left and right is values._right:
            return values
        return kind._simple_new(left, right, values.dtype)
    if kind is pd.arrays.SparseArray:
        from pandas._libs.sparse import BlockIndex, IntIndex

        data = _parallel_pandas_values(values.sp_values, pd, n_jobs, check_only, bind, parents)
        index = values.sp_index
        if bind is not None:
            parents += (index,)
        # Inspect the compressed buffers themselves. BlockIndex.indices would
        # materialize every non-fill position and hide the original mapping.
        if type(index) is IntIndex:
            buffers = (index.indices,)
        elif type(index) is BlockIndex:
            buffers = (index.blocs, index.blengths)
        else:
            buffers = ()
        positions = tuple(_parallel_pandas_values(buf, pd, n_jobs, check_only, bind, parents) for buf in buffers)
        if any(new is not old for new, old in zip(positions, buffers)):
            index = type(index)(index.length, *positions)
        if data is values.sp_values and index is values.sp_index:
            return values
        return kind._simple_new(data, index, values.dtype)
    if kind is pd.Categorical:
        codes = _parallel_pandas_values(values._ndarray, pd, n_jobs, check_only, bind, parents)
        if bind is not None:
            parents += (values.dtype,)
        categories = _parallel_pandas_index(values.categories, pd, n_jobs, check_only, bind, parents)
        if codes is values._ndarray and categories is values.categories:
            return values
        if codes is values._ndarray:
            codes = _snapshot_search_input(codes)
        return pd.Categorical.from_codes(codes, categories=categories, ordered=values.ordered)

    if kind in (pd.arrays.IntegerArray, pd.arrays.FloatingArray, pd.arrays.BooleanArray):
        fields = ('_data', '_mask')
    elif kind in (pd.arrays.DatetimeArray, pd.arrays.TimedeltaArray,
                  pd.arrays.PeriodArray, pd.arrays.StringArray):
        fields = ('_ndarray',)
    else:
        return values
    buffers = tuple(getattr(values, field) for field in fields)
    if bind is not None:
        for buffer in buffers:
            bind(buffer, parents)
        return values
    for buffer in buffers:
        _reject_mapped_objects(buffer, n_jobs)
    if check_only or not any(_needs_process_snapshot(buffer, n_jobs) for buffer in buffers):
        return values
    # Copy a masked array as one unit so that pandas ownership is independent
    # for both its data and mask, even when only one was a private mapping.
    detached = values.copy()
    for field, original in zip(fields, buffers):
        buffer = getattr(detached, field)
        buffer.flags.writeable = original.flags.writeable
    return detached


def _parallel_pandas_index(index, pd, n_jobs, check_only, bind=None, parents=()):
    kind = type(index)
    if bind is not None:
        parents += (index,)
        if kind is pd.IntervalIndex:
            # IntervalIndex pickles these two Index objects directly.
            for endpoint in (index.left, index.right):
                _parallel_pandas_index(endpoint, pd, n_jobs, check_only, bind, parents)
            return index
    if kind is pd.MultiIndex:
        levels = [_parallel_pandas_index(level, pd, n_jobs, check_only, bind, parents) for level in index.levels]
        codes = [_parallel_pandas_values(code, pd, n_jobs, check_only, bind, parents) for code in index.codes]
        if (all(left is right for left, right in zip(levels, index.levels))
                and all(left is right for left, right in zip(codes, index.codes))):
            return index
        # Match pandas' pickle reconstruction of an existing MultiIndex.
        # Revalidation recodes real NA levels produced by groupby(dropna=False).
        return pd.MultiIndex(levels=levels, codes=codes, sortorder=index.sortorder,
                             names=index.names, verify_integrity=False)
    if kind in (pd.Index, pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex,
                pd.CategoricalIndex, pd.IntervalIndex):
        values = _parallel_pandas_values(index._data, pd, n_jobs, check_only, bind, parents)
        if values is not index._data:
            result = kind._simple_new(values, name=index.name)
            # Match pandas' copy of newer temporal indexes. Earlier versions
            # already carried the frequency in the copied array.
            if kind in (pd.DatetimeIndex, pd.TimedeltaIndex) and hasattr(index, '_freq'):
                result._freq = index._freq
            return result
    # In particular, never materialize a RangeIndex to inspect its storage.
    return index


@lru_cache(maxsize=16)
def _sparse_uint64_conversion_dtype(pd, dtype, fill_type, *, take):
    # Equal SparseDtypes can have differently typed fills and promote
    # differently. Include their type in the key. Probe two rows irrespective
    # of the size or density of the input.
    positions = pd.arrays.SparseArray([1, 0], fill_value=0).sp_index
    sample = pd.arrays.SparseArray._simple_new(np.array([2**53 + 1], dtype=dtype.subtype),
                                              positions, dtype)
    if take:
        selected = sample.take([1, 0], allow_fill=True)
        if selected.dtype.subtype.kind == 'O':
            # An object result can contain text produced by an intermediate
            # string cast. Inspect the position that started as an integer.
            return np.asarray(selected[1]).dtype
        return selected.dtype.subtype
    return np.asarray(sample).dtype


def _integer_values_lose_precision(stored, dtype):
    precision = np.finfo(dtype)
    bits = precision.nmant + 1
    signed = stored.dtype.kind == 'i'
    width = stored.dtype.itemsize * 8 - signed
    if bits >= width or not stored.size:
        return False
    if int(stored.max()) <= 2**bits and (not signed or int(stored.min()) >= -2**bits):
        return False
    with np.errstate(over='ignore', invalid='ignore'):
        floats = stored.astype(precision.dtype)
        # The upper endpoint of either integer range is outside a valid reverse cast.
        return bool(np.any(floats >= float(2**width)) or
                    not np.array_equal(stored, floats.astype(stored.dtype)))


def _conversion_loses_sparse_uint64(values, dtype):
    if dtype.kind in 'SUMm':
        return bool(values.npoints)
    if dtype.kind not in 'fc':
        return False
    precision = np.finfo(dtype)
    real_dtype = precision.dtype
    fill = values.fill_value
    if isinstance(fill, Integral):
        try:
            with np.errstate(over='ignore'):
                converted = real_dtype.type(fill)
            if not np.isfinite(converted) or int(converted) != int(fill):
                return True
        except OverflowError:
            return True
    return _integer_values_lose_precision(values.sp_values, dtype)


_TEMPORAL_LABEL_TYPES = (date, timedelta, np.datetime64, np.timedelta64)


def _sparse_label_scalar(value):
    if not isinstance(value, np.generic) or value.dtype.kind not in 'buifc':
        return value
    value = value.item()
    if isinstance(value, np.inexact) and not value.imag:
        # item() retains extended precision as a NumPy scalar. Its equality
        # can reject an exactly equal Decimal or Fraction. The binary ratio
        # preserves all of its significand, including on wider platforms.
        real = value.real
        return Fraction(*real.as_integer_ratio()) if np.isfinite(real) else float(real)
    return value


def _sparse_fill_changed(fill, observed, pd):
    if fill is observed:
        return False
    if isinstance(fill, np.void) or isinstance(observed, np.void):
        # NumPy cannot compare a byte record with an implicitly boxed bytes
        # or tuple label. Records that kept their dtype preserve raw payloads.
        if not (isinstance(fill, np.void) and isinstance(observed, np.void) and fill.dtype == observed.dtype):
            return True
        if fill.tobytes() == observed.tobytes():
            return False
        return bool(fill != observed)
    if isinstance(fill, pd.Interval) and isinstance(observed, pd.Interval):
        # Interval equality compares endpoints with NumPy's mixed arithmetic.
        # Compare each endpoint with the same precision as a standalone label.
        return (fill.closed != observed.closed or _sparse_fill_changed(fill.left, observed.left, pd)
                or _sparse_fill_changed(fill.right, observed.right, pd))
    actual = _sparse_label_scalar(observed)
    expected = _sparse_label_scalar(fill)
    if pd.isna(expected) or pd.isna(actual):
        return bool(pd.isna(expected)) != bool(pd.isna(actual))
    if isinstance(expected, _TEMPORAL_LABEL_TYPES) or isinstance(actual, _TEMPORAL_LABEL_TYPES):
        return (not isinstance(expected, _TEMPORAL_LABEL_TYPES) or not isinstance(actual, _TEMPORAL_LABEL_TYPES)
                or _temporal_label_key(expected, pd) != _temporal_label_key(actual, pd))
    if isinstance(actual, Integral) and isinstance(expected, (float, complex, np.inexact)):
        expected, actual = actual, expected
    if isinstance(expected, Integral) and isinstance(actual, (float, complex, np.inexact)):
        real = actual.real
        return (bool(actual.imag) or not np.isfinite(real) or real != np.trunc(real)
                or int(real) != int(expected))
    return actual != expected


def _sparse_uint64_fold_loses(values, pd):
    fill = values.fill_value
    if isinstance(fill, np.unsignedinteger) or not len(values):
        return False
    if not values.npoints:
        # With no stored entries pandas uses a separate take branch. Older
        # implementations cast the fill to the subtype, including -1 or NaN.
        try:
            with np.errstate(invalid='ignore', over='ignore'):
                actual = values.take([0], allow_fill=True)[0]
        except (OverflowError, TypeError, ValueError):
            return True
        return _sparse_fill_changed(fill, actual, pd)

    # Recompression drops rows comparing equal to the fill. NumPy's mixed
    # comparison can equate adjacent integers even when take kept uint64.
    real = fill.real if isinstance(fill, (complex, np.complexfloating)) and not fill.imag else fill
    if isinstance(real, (int, np.integer, float, np.floating)) and float(2**53) <= real <= float(2**64):
        stored = values.sp_values
        matches = stored[stored == fill]
        if matches.size and (int(real) == 2**64 or np.any(matches != np.uint64(int(real)))):
            return True
    if not values.sp_index.ngaps:
        return False
    taken = _sparse_uint64_conversion_dtype(pd, values.dtype, type(fill), take=True)
    return _conversion_loses_sparse_uint64(values, taken)


def _sparse_target_values_lose(values, pd, block_rows):
    if isinstance(values.fill_value, np.unsignedinteger):
        return False
    # Index, bare SparseArray and ArrayManager rows use positional takes that
    # preserve compressed entries. All carriers can undergo array conversion.
    loses = block_rows and _sparse_uint64_fold_loses(values, pd)
    if not loses:
        fill = values.fill_value
        # Recompression of a stored value equal to the fill can introduce
        # gaps. The fold's default conversion then promotes its other entries.
        gaps = values.sp_index.ngaps or (block_rows and isinstance(fill, (int, float, complex, np.number))
                                         and np.any(values.sp_values == fill))
        if gaps and not values.npoints:
            # For example, NumPy can truncate a long text fill to the print
            # width of uint64. One implicit element exercises the actual cast.
            loses = _sparse_fill_changed(fill, np.asarray(values[:1])[0], pd)
        elif gaps:
            converted = _sparse_uint64_conversion_dtype(pd, values.dtype, type(fill), take=False)
            loses = _conversion_loses_sparse_uint64(values, converted)
    return loses


def _sparse_frame_take_dtype(values, pd):
    if (not isinstance(values, pd.arrays.SparseArray) or values.dtype.subtype.kind not in 'buifcMm'
            or not values.sp_index.ngaps):
        return values.dtype
    if values.dtype.subtype.kind in 'Mm' and _native_temporal_target_exact(values, None, pd):
        return values.dtype
    # A nonempty positional fold can select both a stored value and a gap.
    # The all-gap case follows a separate take path on older pandas.
    if not values.npoints:
        return values.take([0], allow_fill=True).dtype
    positions = pd.arrays.SparseArray([1, 0], fill_value=0).sp_index
    sample = pd.arrays.SparseArray._simple_new(values.sp_values[:1], positions, values.dtype)
    return sample.take([1, 0], allow_fill=True).dtype


def _integer_probe_indices(stored):
    """Represent the occupied range and any integer that cannot survive binary64."""
    if not stored.size:
        return []
    low, high = np.argmin(stored), np.argmax(stored)
    if low == high:
        return [low]
    indices = [low, high]
    if int(stored[low]) < -(2**53) or int(stored[high]) > 2**53:
        with np.errstate(invalid='ignore', over='ignore'):
            rounded = stored.astype(np.float64)
            width = stored.dtype.itemsize * 8 - (stored.dtype.kind == 'i')
            changed = (rounded >= float(2**width)) | (rounded.astype(stored.dtype) != stored)
        if changed.any():
            indices.append(np.argmax(changed))
    return indices


def _has_sparse_levels(values, pd):
    return isinstance(values, pd.Categorical) and isinstance(values.categories.dtype, pd.SparseDtype)


def _native_temporal_target_exact(values, dtype, pd):
    # Native pandas units already box into Timestamp/Timedelta without
    # rescaling. A scalar's Python type alone does not identify its tick size.
    categorical = type(values) is pd.Categorical
    if categorical:
        if type(values.categories) is not pd.Index:
            return False
        backing = values.categories.array
    else:
        backing = values
    if type(backing) is not pd.arrays.SparseArray:
        return False
    native = backing.dtype.subtype
    fill = backing.fill_value
    return (native.kind in 'Mm' and native.isnative and type(fill) is native.type and fill.dtype == native
            and np.datetime_data(native) in (('s', 1), ('ms', 1), ('us', 1), ('ns', 1))
            and (dtype is None or dtype.kind == 'O' or dtype == native)
            and (not categorical or not len(values) or values.codes.min() >= 0))


def _distinct_boxed_labels(stored, pd):
    # Distinct immutable temporal boxes may carry equal ticks in different
    # units, zones or folds. Keep those representations separate. For other
    # scalar classes only collapse repeated references to the same object.
    unique = {}
    for identity, value in {id(v): v for v in stored}.items():
        kind = type(value)
        if kind is pd.Timestamp or kind is pd.Timedelta:
            unit = value.unit
            tick = value.value if unit == 'ns' else int(value.asm8.view('i8'))
            key = (kind, unit, tick, id(getattr(value, 'tz', None)), getattr(value, 'fold', None))
        elif kind is np.datetime64 or kind is np.timedelta64:
            key = (kind, value.dtype, int(value.view('i8')))
        else:
            key = identity
        unique.setdefault(key, value)
    return np.fromiter(unique.values(), dtype=object, count=len(unique))


def _exact_interval_endpoint_key(value):
    kind = type(value)
    if kind in (int, np.int64, np.uint64):
        integer = int(value)
        if -(2**63) <= integer < 2**63 and int(float(integer)) == integer:
            return kind, integer < 0, True
    elif kind in (float, np.float64) and -(2**53) <= value <= 2**53:
        return kind, value < 0, value.is_integer()
    return None


def _exact_numeric_interval_sample(sample, pd):
    # Pandas infers each side separately. Group endpoints only after checking
    # each actual integer for exact binary64 promotion. Keep closure, sign,
    # type and fractional sides distinct when testing a real pandas take.
    groups = {}
    for label in sample:
        if type(label) is str:
            key = str
        elif type(label) is pd.Interval:
            left = _exact_interval_endpoint_key(label.left)
            right = _exact_interval_endpoint_key(label.right)
            if left is None or right is None:
                return sample
            key = label.closed, left, right
        else:
            return sample
        groups.setdefault(key, label)
    return np.fromiter(groups.values(), dtype=object, count=len(groups))


def _frame_numeric_probe(values, pd, *, retain_rows=False):
    """Small native extension array with observed values, sparse fills and missing labels."""
    if isinstance(values, pd.arrays.SparseArray):
        stored = values.sp_values
        fill = values.fill_value
        if stored.dtype.kind in 'iu':
            sample = stored[_integer_probe_indices(stored)]
        elif stored.dtype.kind in 'Mm':
            # Unit conversion can round an interior tick even if the extrema
            # are exact. Retain each distinct stored label and one actual gap.
            sample = np.unique(stored)
        elif stored.dtype.kind == 'O':
            # A mixed fold can keep its labels boxed while a homogeneous
            # selection coerces temporal values or interval endpoints.
            inferred_boxes = _TEMPORAL_LABEL_TYPES + (pd.Interval,)
            if not isinstance(fill, inferred_boxes) and not any(isinstance(v, inferred_boxes) for v in stored):
                return None
            sample = stored if retain_rows else _distinct_boxed_labels(stored, pd)
            if not retain_rows and type(fill) in (float, np.float64) and np.isnan(fill):
                sample = _exact_numeric_interval_sample(sample, pd)
        elif stored.dtype.kind in 'bfc':
            # An integral floating fill can narrow to the stored dtype.
            # A text fill can turn a numerical payload into text on take.
            sample = stored[:1]
            if stored.size:
                indices = [np.argmax(~np.isnan(stored))]
                if stored.dtype.kind == 'c':
                    indices.append(np.argmax(stored.imag != 0))
                if stored.dtype.kind in 'fc' and np.finfo(stored.dtype).nmant > 52:
                    narrow = np.float64 if stored.dtype.kind == 'f' else np.complex128
                    with np.errstate(invalid='ignore', over='ignore'):
                        changed = (stored != stored.astype(narrow)) & ~np.isnan(stored)
                    if changed.any():
                        indices.append(np.argmax(changed))
                sample = stored[indices]
        elif isinstance(fill, _TEMPORAL_LABEL_TYPES):
            # Legacy pandas also allows a temporal fill on byte-record and
            # other storage. Exercise its actual take and scalar boxing.
            sample = stored[:1]
        else:
            return None
        if (stored.dtype.kind in 'buifc' and not isinstance(fill, np.timedelta64)
                and isinstance(fill, (int, float, complex, np.number)) and stored.size):
            # Recompression can absorb an adjacent integer comparing equal
            # to a floating fill or introduce a gap at an explicit fill.
            matches = stored[stored == fill]
            if matches.size:
                sample = np.append(sample, [matches.min(), matches.max()]).astype(stored.dtype, copy=False)
        mask = np.ones(len(sample) + bool(values.sp_index.ngaps), dtype=bool)
        mask[len(sample):] = False
        positions = pd.arrays.SparseArray(mask, fill_value=False).sp_index
        return pd.arrays.SparseArray._simple_new(sample, positions, values.dtype)
    if isinstance(values, pd.Categorical):
        if isinstance(values.categories.dtype, pd.SparseDtype):
            # Keep the actual dictionary and each observed code, including -1.
            # Taking the sparse levels into a dense array can round the very
            # values we need as our reference. Scalar lookup stays exact.
            codes = np.unique(values.codes)
            levels = values.categories.array
            # A boxed level can itself carry numeric data, e.g. the endpoints
            # of an interval. Other backings can contain a numeric fill.
            # Even a point-only take can consult a temporal fill when pandas
            # recompresses its result. Only observed codes enter the probe.
            if (levels.dtype.subtype.kind not in 'buifcOMm' and not isinstance(levels.fill_value, _TEMPORAL_LABEL_TYPES)
                    and not any(pd.api.types.is_number(levels[int(code)]) for code in codes if code >= 0)):
                return None
            return pd.Categorical.from_codes(codes, dtype=values.dtype)
        if values.categories.dtype.kind not in 'iu':
            return None
        codes = np.unique(values.codes)
        used = codes[codes >= 0]
        native = getattr(values.categories.dtype, 'numpy_dtype', values.categories.dtype)
        labels = values.categories.take(used).to_numpy(dtype=native)
        sample = used[_integer_probe_indices(labels)]
        if len(used) < len(codes):
            sample = np.append(sample, -1)
        return pd.Categorical.from_codes(sample, dtype=values.dtype)
    if isinstance(values, pd.arrays.IntegerArray):
        placed = values.dropna()
        native = placed.to_numpy(dtype=values.dtype.numpy_dtype)
        sample = _integer_probe_indices(native)
        if len(placed) < len(values):
            sample.append(-1)
        return placed.take(np.asarray(sample, dtype=np.intp), allow_fill=True)
    return None


def _object_point_subsets(points, scalars, pd):
    # Inference on a mixed object array can leave it boxed, but convert a
    # homogeneous fold to a temporal, period or interval array. Infer the
    # native dtype of each observed scalar. This retains metadata such as
    # time zones, units, interval closure and endpoint dtype.
    if len(points) < 2 or pd.api.types.infer_dtype(scalars) in (
            'integer', 'floating', 'mixed-integer-float', 'complex', 'boolean', 'decimal', 'string', 'bytes'):
        return
    by_dtype = {}
    by_step = {}
    ordinary_units = {}
    multiplied_units = {}
    for i, row in enumerate(points):
        try:
            dtype = pd.array(scalars[i:i + 1]).dtype
        except (OverflowError, ValueError, TypeError):
            # Some boxed labels do not have a native extension dtype.
            dtype = type(scalars[i])
        scalar = scalars[i]
        # Pandas may infer the same native dtype from different NumPy tick
        # units. Keep those units distinct when checking dtype promotion.
        unit = None
        if isinstance(scalar, (np.datetime64, np.timedelta64)):
            base, step = np.datetime_data(scalar.dtype)
            unit = (scalar.dtype.kind, base)
            by_step.setdefault((dtype, unit), {}).setdefault(step, []).append(row)
            (ordinary_units if step == 1 else multiplied_units).setdefault(unit, []).append(row)
        elif isinstance(scalar, _TEMPORAL_LABEL_TYPES):
            kind = 'm' if isinstance(scalar, timedelta) else 'M'
            ordinary_units.setdefault((kind, dtype), []).append(row)
        raw = scalar.asm8 if isinstance(scalar, (pd.Timestamp, pd.Timedelta)) else scalar
        by_dtype.setdefault((dtype, unit, _time_range_band(raw)), []).append(row)
    # A multiplied unit can stay boxed when it leads a selection, yet
    # normalize incorrectly after an ordinary tick starts native inference.
    # Pair unit families in either order, irrespective of multiplier count.
    for (kind, _), ordinary in ordinary_units.items():
        for (other_kind, _), multiplied in multiplied_units.items():
            if other_kind == kind:
                yield np.asarray(ordinary + multiplied)
                yield np.asarray(multiplied + ordinary)
    # Within one unit and inferred dtype, the step multiplies each scalar's
    # tick count. Check the pooled values and each observed step individually;
    # pairing every multiplier would repeat the same dtype promotion.
    for steps in by_step.values():
        if len(steps) > 1:
            for rows in steps.values():
                yield np.asarray(rows)
    if len(by_dtype) < 2:
        return
    for rows in by_dtype.values():
        yield np.asarray(rows)
    # Two compatible dtypes can also promote one another. For example,
    # integer and floating endpoints with the same interval closure can form
    # a floating IntervalArray, even when a third label prevents inference.
    families = {}
    for (dtype, _, _), rows in by_dtype.items():
        family = (type(dtype), getattr(dtype, 'closed', None), str(getattr(dtype, 'tz', '')),
                  getattr(dtype, 'freq', None))
        families.setdefault(family, []).append(rows)
    for groups in families.values():
        if len(groups) > 2 and sum(map(len, groups)) < len(points):
            yield np.concatenate(groups)
        for a, b in combinations(groups, 2):
            if len(a) + len(b) < len(points):
                yield np.asarray(a + b)


def _interval_endpoint_group(value, pd):
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        zone = getattr(value, 'tz', None)
        if zone is not None:
            from pandas._libs.tslibs.timezones import get_timezone
            zone = get_timezone(zone)
            try:
                hash(zone)
            except TypeError:
                zone = id(value.tz)
        return type(value), value.unit, _time_range_band(value.asm8), zone
    if isinstance(value, Integral):
        # Negative, signed-range, unsigned-range and oversized Python integers
        # follow distinct pandas inference paths. Keep the original scalar type.
        n = int(value)
        if n < -(2**63) or n >= 2**64:
            # A group can stay boxed even if gathering its largest endpoint
            # alone would overflow. Probe that conversion class separately.
            try:
                float(n)
            except OverflowError:
                return type(value), n < 0, 'float-overflow'
        return type(value), n < 0, n < -(2**63), n >= 2**63, n >= 2**64
    return type(value)


def _interval_selections(points, scalars, optional, pd):
    # Pandas infers one interval closure at a time. A float endpoint, or a
    # missing row on its own, can promote an entire integer endpoint array.
    # Group by representations, not by distinct label values; exercise all
    # values in a group together so that interior inexact values are checked.
    by_closure = {}
    missing = {}
    for origin, (rows, labels) in enumerate([(points, scalars), *optional]):
        for row, label in zip(rows, labels):
            if isinstance(label, pd.Interval):
                group = (origin, _interval_endpoint_group(label.left, pd), _interval_endpoint_group(label.right, pd))
                by_closure.setdefault(label.closed, {}).setdefault(group, []).append(row)
            elif pd.api.types.is_scalar(label) and pd.isna(label):
                missing.setdefault((origin, type(label)), []).append(row)
    for groups in by_closure.values():
        for rows in groups.values():
            yield np.asarray(rows)
            for absent in missing.values():
                yield np.asarray(rows + absent)
                yield np.asarray(absent + rows)
        for first, second in combinations(groups.values(), 2):
            yield np.asarray(first + second)
            yield np.asarray(second + first)


def _object_point_selections(points, scalars, optional, pd):
    has_intervals = (any(isinstance(v, pd.Interval) for v in scalars)
                     or any(isinstance(labels[0], pd.Interval) for _, labels in optional))
    if has_intervals:
        yield from _interval_selections(points, scalars, optional, pd)
        intervals = np.fromiter((isinstance(v, pd.Interval) for v in scalars), dtype=bool, count=len(points))
        points, scalars = points[~intervals], scalars[~intervals]
    other_rows = [rows for rows, _ in optional]
    subsets = _object_point_subsets(points, scalars, pd)
    if has_intervals and len(points):
        subsets = chain((points,), subsets)
    for rows in subsets:
        yield rows
        for n in range(1, len(other_rows) + 1):
            for other in combinations(other_rows, n):
                yield np.concatenate((rows, *other))


def _sparse_category_subsets(sample, pd):
    # Converting categorical codes takes stored levels, dictionary gaps and
    # missing codes (-1) down different pandas promotion paths. One of each
    # observed code covers repetitions. Exercise folds that omit a group.
    codes = sample.codes
    valid = np.flatnonzero(codes >= 0)
    levels = sample.categories.array
    locations = levels.sp_index.lookup_array(codes[valid].astype(np.int32))
    stored = locations >= 0
    points = valid[stored]
    gaps = valid[~stored]
    missing = np.flatnonzero(codes < 0)
    groups = [rows for rows in (points, gaps, missing) if len(rows)]
    for size in range(1, len(groups)):
        for chosen in combinations(groups, size):
            yield np.concatenate(chosen)
    if levels.dtype.subtype.kind == 'O' and len(points) > 1:
        # Vary the dictionary gap and the missing code independently for
        # each distinct inference path. There is at most one of each.
        optional = [(rows, [sample[rows[0]]]) for rows in (gaps, missing) if len(rows)]
        yield from _object_point_selections(points, levels.sp_values[locations[stored]], optional, pd)


def _temporal_frame_values(converted, dtype):
    # DataFrame._values wraps the interleaved array, normalizing its tick unit.
    from pandas.core.construction import ensure_wrapped_if_datetimelike
    return np.asarray(ensure_wrapped_if_datetimelike(np.asarray(converted, dtype=dtype)))


def _object_temporal_fold_loses(frame, values, positions, pd, risk):
    # A homogeneous selection from boxed sparse points may infer native time
    # storage. If all columns can do this, the frame's output depends on the
    # same selection in its companions, not just their parent dtypes.
    stored = values.sp_index.to_int_index().indices
    rows = stored[positions[positions < len(stored)]]
    if np.any(positions == len(stored)):
        rows = np.concatenate((rows, np.setdiff1d(np.arange(len(values)), stored, assume_unique=True)))
    if risk is not None and not risk[rows].any():
        return False
    return _temporal_rows_lose(frame, rows, pd)


def _temporal_rows_lose(frame, rows, pd):
    if len(rows) and len(rows) == len(frame) and np.array_equal(rows, np.arange(len(frame))):
        # pandas may shallow-copy an identity take without selecting any
        # sparse values. Repeat the last row to force the gather while
        # preserving every observed value and their order of inference.
        rows = np.append(rows, rows[-1])
    folded = frame.take(rows)
    converted = np.asarray(folded)
    for i, row in enumerate(rows):
        for j in range(frame.shape[1]):
            expected = frame.iat[row, j]
            if (_sparse_fill_changed(expected, folded.iat[i, j], pd)
                    or _sparse_fill_changed(expected, converted[i, j], pd)):
                return True
    return False


_SEARCH_TIME_UNITS = {'s': 10**9, 'ms': 10**6, 'us': 1000, 'ns': 1}


def _time_range_band(value):
    # Values of one unit can land on different sides of a native cast
    # boundary, causing pandas to keep only some selections boxed.
    if not isinstance(value, (np.datetime64, np.timedelta64)):
        return None
    unit, step = np.datetime_data(value.dtype)
    if step != 1 or unit not in _SEARCH_TIME_UNITS or np.isnat(value):
        return None
    ticks = abs(int(value.view('i8'))) * _SEARCH_TIME_UNITS[unit]
    for scale in (1, 1000, 10**6, 10**9):
        if ticks <= (2**63 - 1) * scale:
            return scale


def _standard_time_class(value, pd):
    if type(value) in (str, np.str_):
        return 'text', None, 0
    if (value is None or value is pd.NaT or value is pd.NA
            or (type(value) in (float, np.float64) and np.isnan(value))):
        return 'missing', None, 0
    if type(value) in (pd.Timestamp, pd.Timedelta):
        if getattr(value, 'tz', None) is not None:
            return None
        raw = value.asm8
    elif type(value) in (np.datetime64, np.timedelta64):
        raw = value
    else:
        return None
    if np.isnat(raw):
        return 'missing', None, 0
    unit, step = np.datetime_data(raw.dtype)
    if step != 1 or unit not in _SEARCH_TIME_UNITS:
        return None
    return raw.dtype.kind, unit, _time_range_band(raw)


def _standard_time_states(sample, classes, pd):
    # Within standard units the dtype depends on the resolutions present.
    # Test local choices of units, a gap, a string and a missing scalar.
    # Their number is independent of the number of other target columns.
    groups = {}
    for i, (kind, unit, _) in enumerate(classes):
        groups.setdefault((kind, unit, type(sample[i]), i == sample.npoints), []).append(i)
    # A native resolution is governed by the finest participating unit and
    # pandas' normalization of the leading scalar. Check each such pair
    # and the complete time group. Text, a stored missing value and the
    # one actual gap are separate conversion branches.
    times = tuple(g for g in groups if g[0] in ('M', 'm') and not g[-1])
    other = tuple(g for g in groups if g not in times)
    if len(other) > 3:
        return None
    time_groups = [(), *((g,) for g in times), *combinations(times, 2)]
    if len(times) > 2:
        time_groups.append(times)
    states = {}
    for extra in chain.from_iterable(combinations(other, n) for n in range(len(other) + 1)):
        for points in time_groups:
            group = points + extra
            if not group:
                continue
            state = frozenset(key[0] for key in group)
            dtypes, bad_casts = states.setdefault(state, (set(), set()))
            # Exercise each leading representation, including NumPy and
            # pandas boxes of the same unit and a possible leading gap.
            for start in range(len(group)):
                rows = [i for key in group[start:] + group[:start] for i in groups[key]]
                expected = [sample[i] for i in rows]
                selected = sample.take(rows, allow_fill=True)
                if any(_sparse_fill_changed(x, selected[i], pd) for i, x in enumerate(expected)):
                    return None
                dtypes.add(selected.dtype)
                if 'object' not in bad_casts:
                    converted = np.asarray(selected.astype(object))
                    if any(_sparse_fill_changed(x, y, pd) for x, y in zip(expected, converted)):
                        bad_casts.add('object')
                native = selected.dtype.subtype
                if native.kind in 'Mm' and selected.sp_index.ngaps:
                    base, step = np.datetime_data(native)
                    if step != 1 or base not in _SEARCH_TIME_UNITS:
                        return None
                    for unit, scale in _SEARCH_TIME_UNITS.items():
                        if scale <= _SEARCH_TIME_UNITS[base] and unit not in bad_casts:
                            try:
                                converted = selected.to_numpy(dtype=np.dtype(f'{native.kind}8[{unit}]'))
                            except (ValueError, TypeError, OverflowError):
                                bad_casts.add(unit)
                            else:
                                if any(_sparse_fill_changed(x, y, pd) for x, y in zip(expected, converted)):
                                    bad_casts.add(unit)
    return states


def _standard_time_frame_risks(arrays, probes, pd):
    # Native interleaving either boxes these columns or uses the finest of
    # their selected temporal resolutions. In-range ticks cannot round or
    # overflow on widening. A text marker prevents a native temporal frame
    # whenever its row is selected.
    from pandas.core.internals.managers import ensure_np_dtype, interleaved_dtype
    columns = []
    fixed = []
    common_kind = set()
    risky = blocked = missing = layout = None
    aligned = True
    gaps = False
    bands = []
    boxed_bands = []
    fixed_units = set()
    for column, values in enumerate(arrays):
        if type(values) is pd.arrays.SparseArray and values.dtype.subtype.kind == 'O':
            sample = probes.get(column)
            fill = _standard_time_class(values.fill_value, pd)
            if sample is None or (fill is None and (values.sp_index.ngaps or type(values.fill_value) not in (int, bool, float))):
                return None
            classes = [_standard_time_class(sample[i], pd) for i in range(len(sample))]
            if any(c is None for c in classes):
                return None
            common_kind.update(c[0] for c in classes if c[0] in ('M', 'm'))
            if any(c[2] > 1 for c in classes):
                # Check positions only when an observed value could overflow.
                placed = values.sp_index.to_int_index().indices
                limits = np.full(len(values), fill[2] if values.sp_index.ngaps else 0, dtype=np.int32)
                points = values.sp_values
                by_id = {id(x): x for x in points}
                by_id = {i: _standard_time_class(x, pd)[2] for i, x in by_id.items()}
                limits[placed] = np.fromiter((by_id[id(x)] for x in points), dtype=np.int32, count=len(placed))
                bands.append(limits)
                boxed_bands.append(limits)
            columns.append((values, sample, classes))
            gaps |= (bool(values.sp_index.ngaps) or fill is None or fill[0] != 'missing'
                     or any(c[0] == 'missing' for c in classes))
        else:
            if type(values) in (pd.arrays.DatetimeArray, pd.arrays.TimedeltaArray):
                raw = np.asarray(values)
            elif (type(values) is pd.Categorical and type(values.categories) in (pd.DatetimeIndex, pd.TimedeltaIndex)
                  and not (values.codes < 0).any()):
                raw = values.categories.to_numpy()[values.codes]
            else:
                return None
            if raw.dtype.kind not in 'Mm':
                return None
            unit, step = np.datetime_data(raw.dtype)
            if step != 1 or unit not in _SEARCH_TIME_UNITS or np.isnat(raw).any():
                return None
            common_kind.add(raw.dtype.kind)
            fixed_units.add(unit)
            fixed.append(values.dtype)
            ticks = raw.view('i8')
            limit = (2**63 - 1) // _SEARCH_TIME_UNITS[unit]
            far = ((ticks < -limit) | (ticks > limit)).reshape(-1, raw.shape[-1]).any(axis=0)
            if far.any():
                limits = np.where(far, 1000, 1)
                for scale in (1000, 10**6):
                    limit = min(2**63 - 1, (2**63 - 1) * scale // _SEARCH_TIME_UNITS[unit])
                    above = ((ticks < -limit) | (ticks > limit)).reshape(-1, raw.shape[-1]).any(axis=0)
                    limits[above] = scale * 1000
                bands.append(limits)
    if len(common_kind) != 1:
        return None
    if bands:
        largest = np.maximum.reduce(bands)
        risky = largest > 1
        if not gaps and len(boxed_bands) == len(columns):
            # To interleave at a given unit, at least one selected column
            # must supply that native unit. It includes this row, so its
            # own scalar must fit. An all-distant row cannot supply ns
            # just because some other row has an in-range scalar.
            smallest = np.minimum.reduce(boxed_bands)
            risky.fill(False)
            for unit, scale in _SEARCH_TIME_UNITS.items():
                if unit in fixed_units:
                    risky |= largest > scale
                else:
                    risky |= (largest > scale) & (smallest <= scale)
    if risky is not None or gaps:
        for values, _, _ in columns:
            # 0=time, 1=text, 2=missing. Gaps occupy their actual positions.
            def code(x):
                return 1 if type(x) in (str, np.str_) else 2 if pd.isna(x) else 0
            labels = np.full(len(values), code(values.fill_value), dtype=np.uint8)
            labels[values.sp_index.to_int_index().indices] = np.fromiter(
                (code(x) for x in values.sp_values), dtype=np.uint8, count=values.npoints)
            blocked = labels == 1 if blocked is None else blocked | (labels == 1)
            missing = labels == 2 if missing is None else missing | (labels == 2)
            if layout is None:
                layout = labels
            else:
                aligned &= np.array_equal(layout, labels)
        if risky is not None:
            risky &= ~blocked
    if risky is None:
        risky = np.zeros(len(columns[0][0]), dtype=bool)
    if not gaps and not risky.any():
        # Row alignment and native promotion have already been checked.
        # Only the synthetic sparse columns need a take and an object cast.
        # Reuse small, gap-free probes with identical metadata and ordered
        # object references. The probes own those references throughout.
        seen = set()
        for values, sample, _ in columns:
            if len(columns) > 1 and len(sample) <= 16 and type(sample.dtype) is pd.SparseDtype:
                key = (id(sample.dtype.subtype), id(sample.fill_value), tuple(map(id, sample.sp_values)))
                if key in seen:
                    continue
                seen.add(key)
            if _frame_conversion_loses([values], {0: sample}, np.dtype(object), pd, True, take=True):
                return None
        return risky

    # With aligned markers every column sees the same kinds of rows. For
    # example an older pandas can mishandle boxing a duration and a gap,
    # while the actual frame is native for every selection of that kind.
    summaries = []
    try:
        for _, sample, classes in columns:
            # Local inference can differ for two scalars of the same
            # representation on opposite sides of a resolution limit.
            # Pooling them would hide singleton or missing-plus-point takes.
            ranges = {}
            for i, (kind, unit, band) in enumerate(classes):
                if kind in ('M', 'm'):
                    if ranges.setdefault((type(sample[i]), unit), band) != band:
                        return None
            states = _standard_time_states(sample, classes, pd)
            if states is None:
                return None
            summaries.append(states)
        units = fixed_units.copy()
        for states in summaries:
            for choices, _ in states.values():
                for dtype in choices:
                    if dtype.subtype.kind in 'Mm':
                        base, step = np.datetime_data(dtype.subtype)
                        if step != 1 or base not in _SEARCH_TIME_UNITS:
                            return None
                        units.add(base)
        if risky.any() and 'ns' not in units:
            scale = min((_SEARCH_TIME_UNITS[u] for u in units), default=None)
            if scale is None:
                risky.fill(False)
            else:
                risky &= largest > scale
        for state in set().union(*(s.keys() for s in summaries)):
            options = [s[state] for s in summaries if state in s]
            if any(bad & units for _, bad in options):
                return None
            if any('object' in bad for _, bad in options):
                if aligned and len(options) == len(summaries):
                    dtypes = fixed + [d for choices, _ in options for d in choices]
                    if ensure_np_dtype(interleaved_dtype(dtypes)).kind in common_kind:
                        continue
                if 'missing' not in state:
                    return None
                risky |= missing
    except (ValueError, TypeError, OverflowError, NotImplementedError):
        return None
    return risky


def _joint_temporal_selections(boxed, kinds, pd, risk=None):
    # Only jointly compatible rows can yield a native temporal frame. A
    # string in either of two boxed columns can hide their intersection.
    # Group observed combinations of scalar metadata, then pool compatible
    # rows to exercise promotion across output resolutions.
    groups = {}
    compatible = {kind: {} for kind in kinds}
    anchor = int(np.argmax(risk)) if risk is not None and np.count_nonzero(risk) == 1 else None
    for row in range(len(boxed[0])):
        signature = []
        possible = set(kinds)
        for values in boxed:
            scalar = values[row]
            if isinstance(scalar, _TEMPORAL_LABEL_TYPES) and not pd.isna(scalar):
                kind = 'm' if isinstance(scalar, (timedelta, np.timedelta64)) else 'M'
                possible.intersection_update((kind,))
                native = scalar.asm8 if isinstance(scalar, (pd.Timestamp, pd.Timedelta)) else scalar
                unit = None
                if isinstance(native, np.generic):
                    base, step = np.datetime_data(native.dtype)
                    # Different multipliers are tested within their source
                    # column. They share this branch of frame inference.
                    unit = base, step == 1
                signature.append((type(scalar), unit, _time_range_band(native), id(getattr(scalar, 'tzinfo', None)),
                                  values.sp_index.lookup(row) < 0))
            elif pd.api.types.is_scalar(scalar) and pd.isna(scalar):
                signature.append((type(scalar), getattr(scalar, 'dtype', None), values.sp_index.lookup(row) < 0))
            else:
                possible.clear()
            if not possible:
                break
        if possible:
            key = tuple(signature)
            if anchor is not None and row != anchor and key not in groups:
                # A single risky row and the first member of each compatible
                # class can show a loss before the full groups are collected.
                yield [anchor, row]
                yield [row, anchor]
            rows = groups.setdefault(key, [])
            rows.append(row)
            for kind in possible:
                compatible[kind][key] = rows
    priorities = {key for key, rows in groups.items() if risk is None or risk[rows].any()}
    for key, rows in groups.items():
        if key in priorities:
            yield rows
    for kind, signatures in compatible.items():
        if len(signatures) < 2:
            continue
        # One row class can supply a fine resolution and another an
        # out-of-range value. A third class (a Python date, for example)
        # can keep their pooled frame boxed. Exercise each observed pair.
        primary = [rows for key, rows in signatures.items() if key in priorities]
        others = [rows for key, rows in signatures.items() if key not in priorities]
        for first, second in chain(((a, b) for a in primary for b in others), combinations(primary, 2)):
            yield first + second
            yield second + first
        if primary and len(signatures) > 2:
            yield [row for rows in signatures.values() for row in rows]
            # Missing-only columns can require more than two row classes to
            # infer a temporal frame. Pool all classes that select the native
            # family on this pandas version or supply a missing scalar.
            native_rows = [rows for rows in signatures.values()
                           if all(pd.isna(v[rows[0]]) or v.take(rows, allow_fill=True).dtype.subtype.kind == kind
                                  for v in boxed)]
            if 2 < len(native_rows) < len(signatures):
                yield [row for rows in native_rows for row in rows]


def _frame_conversion_loses(arrays, probes, dtype, pd, block_rows, *, take=False, temporal_frame=None, risk=None):
    boxed = dtype is not None and dtype.kind == 'O'
    for column, values in enumerate(arrays):
        if (boxed and type(values) is pd.Categorical
                and type(values.categories) is pd.Index and type(values.categories.array) is pd.arrays.SparseArray
                and values.categories.dtype.subtype.kind == 'O' and not values.categories.array.sp_index.ngaps):
            # astype(object) casts the dictionary before indexing its codes.
            # A gap-free native object dictionary is already that exact array.
            continue
        if _native_temporal_target_exact(values, dtype, pd):
            continue
        if (temporal_frame is not None and isinstance(values, pd.arrays.SparseArray)
                and values.dtype.subtype.kind == 'O'):
            sample = _frame_numeric_probe(values, pd, retain_rows=True)
        else:
            if column not in probes:
                probes[column] = _frame_numeric_probe(values, pd)
            sample = probes[column]
        if (boxed and type(values) is pd.arrays.SparseArray and values.dtype.subtype == np.int64
                and type(sample.fill_value) in (float, np.float64) and sample.fill_value == 0
                and not _integer_values_lose_precision(sample.sp_values, np.dtype('float64'))):
            # This take/boxing path can widen to binary64. A zero fill is
            # exact and the integer probe includes any nonrepresentable ID.
            continue
        if sample is not None and len(sample):
            original_scalars = _boxed_sparse_scalars(sample, pd)
            rows = np.arange(len(sample))
            selections = [rows]
            if take and isinstance(sample, pd.arrays.SparseArray):
                # The probe stores its points first and its one gap last.
                # When a fold selects only points, pandas can recompress at
                # the original subtype. Including a gap can promote them
                # before recompression. Take each selection from the same
                # parent so the all-gap take follows its actual pandas path.
                points, gap = rows[:sample.npoints], rows[sample.npoints:]
                if len(points) and len(gap):
                    selections.extend((points, gap))
                if block_rows and sample.dtype.subtype.kind == 'O':
                    selections = chain(selections, _object_point_selections(
                        points, sample.sp_values, [(gap, [sample.fill_value])] if len(gap) else [], pd))
            elif take and isinstance(sample, pd.Categorical) and isinstance(sample.categories.dtype, pd.SparseDtype):
                selections = chain(selections, _sparse_category_subsets(sample, pd))
            for positions in selections:
                try:
                    if (temporal_frame is not None and isinstance(sample, pd.arrays.SparseArray)
                            and sample.dtype.subtype.kind == 'O'):
                        if _object_temporal_fold_loses(temporal_frame, values, positions, pd, risk):
                            return True
                        continue
                    selected = sample.take(positions, allow_fill=block_rows) if take else sample
                    # These are the two interleaving paths pandas uses for an
                    # extension block. astype(object) and bare np.asarray can
                    # have different intermediate precision on the same version.
                    with np.errstate(invalid='ignore', over='ignore'):
                        if dtype is None:  # The vector's default array conversion.
                            converted = np.asarray(selected)
                        else:
                            converted = (np.asarray(selected.astype(dtype)) if dtype.kind == 'O' or not block_rows
                                         else selected.to_numpy(dtype=dtype))
                            if dtype.kind in 'Mm':
                                converted = _temporal_frame_values(converted, dtype)
                except (ValueError, TypeError, OverflowError, NotImplementedError):
                    return True
                selected_scalars = _boxed_sparse_scalars(selected, pd) if take else original_scalars
                for i, position in enumerate(positions):
                    expected = original_scalars[position]
                    if ((take and _sparse_fill_changed(expected, selected_scalars[i], pd))
                            or _sparse_fill_changed(expected, converted[i], pd)):
                        return True
        elif (isinstance(values, np.ndarray) and dtype.kind in 'fc' and values.dtype.kind in 'iu'
              and _integer_values_lose_precision(values, dtype)):
            return True
        elif (dtype is not None and dtype.kind in 'Mm' and values.dtype != dtype
              and (isinstance(values, (pd.arrays.DatetimeArray, pd.arrays.TimedeltaArray))
                   or (isinstance(values, pd.Categorical) and values.categories.dtype.kind in 'Mm'))):
            # A finer common dtype can overflow a dense companion's ticks.
            original = np.asarray(values)
            try:
                converted = _temporal_frame_values(values.to_numpy(dtype=dtype), dtype)
                if not np.array_equal(original.view('i8'), converted.astype(original.dtype).view('i8')):
                    return True
            except (ValueError, TypeError, OverflowError):
                return True
    return False


def _masks_float_promotion(dtype):
    return dtype.kind not in 'uibfc' or (dtype.kind in 'fc' and np.finfo(dtype).nmant >= 63)


def _boxed_frame_can_infer_time(boxed, pd):
    has_time = False
    for values in boxed:
        fill = values.fill_value
        if ((values.sp_index.ngaps and isinstance(fill, _TEMPORAL_LABEL_TYPES))
                or any(isinstance(v, _TEMPORAL_LABEL_TYPES) for v in values.sp_values)):
            has_time = True
        elif not ((values.sp_index.ngaps and pd.api.types.is_scalar(fill) and pd.isna(fill))
                  or pd.isna(values.sp_values).any()):
            # Every selection contains a non-temporal, non-missing label.
            return False
    return has_time


def _multioutput_sparse_target_loses(frame, pd):
    manager = frame._mgr
    block_rows = getattr(manager, 'blocks', None) is not None
    if block_rows and not manager.any_extension_types:
        return False
    arrays = manager.arrays
    if not any((isinstance(v, pd.arrays.SparseArray)
                and (v.dtype.subtype.kind in 'buifcMmO' or isinstance(v.fill_value, _TEMPORAL_LABEL_TYPES)))
               or _has_sparse_levels(v, pd)
               for v in arrays):
        return False
    # Matching native dtypes for all values and fills need no promotion on
    # take or frame conversion. Check this without reading any observation.
    native = getattr(arrays[0].dtype, 'subtype', arrays[0].dtype)
    if isinstance(native, np.dtype) and native.kind in 'buifc' and native.isnative and all(
            (type(v) is np.ndarray and v.dtype == native)
            or (type(v) is pd.arrays.SparseArray and v.dtype.subtype == native
                and type(v.fill_value) is native.type) for v in arrays):
        return False
    # Use the dtype operations pandas uses for a multi-block frame. This
    # avoids converting unrelated target data or copying DataFrame metadata.
    from pandas.core.internals.managers import ensure_np_dtype, interleaved_dtype
    original = [v.dtype for v in arrays]
    probes = {}
    try:
        common = ensure_np_dtype(interleaved_dtype(original))
    except (ValueError, TypeError, OverflowError, NotImplementedError):
        # The frame itself uses this operation before converting any column.
        return True
    if _frame_conversion_loses(arrays, probes, common, pd, block_rows):
        return True
    if not block_rows:
        # ArrayManager takes preserve storage and dtype, but a fold's
        # categorical conversion can depend on which codes it contains.
        return _frame_conversion_loses(arrays, probes, common, pd, False, take=True)
    selected = [_sparse_frame_take_dtype(v, pd) for v in arrays]
    folded = common if selected == original else ensure_np_dtype(interleaved_dtype(selected))
    temporal_frame = None
    risk = None
    boxed = {i: v for i, v in enumerate(arrays)
             if isinstance(v, pd.arrays.SparseArray) and v.dtype.subtype.kind == 'O'}
    if any(probes.get(i) is not None for i in boxed):
        kinds = []
        for kind in 'Mm':
            native = pd.SparseDtype(f'{kind}8[ns]')
            dtypes = [native if i in boxed else d for i, d in enumerate(selected)]
            if ensure_np_dtype(interleaved_dtype(dtypes)).kind == kind:
                kinds.append(kind)
        if kinds and _boxed_frame_can_infer_time(boxed.values(), pd):
            risk = _standard_time_frame_risks(arrays, probes, pd)
            if risk is not None and not risk.any():
                return False
            temporal_frame = pd.DataFrame(frame, copy=False)
            if len(boxed) > 1:
                try:
                    for rows in _joint_temporal_selections(list(boxed.values()), kinds, pd, risk):
                        if _temporal_rows_lose(temporal_frame, rows, pd):
                            return True
                except (ValueError, TypeError, OverflowError, NotImplementedError):
                    return True
    if _frame_conversion_loses(arrays, probes, folded, pd, block_rows, take=True, temporal_frame=temporal_frame, risk=risk):
        return True
    if not _masks_float_promotion(folded):
        return False
    # A fold can omit every gap that introduces object or extended precision
    # while including a gap that introduces float64. Intersect the stored
    # positions of the masking columns to check that actual subset once.
    rows = None
    for values, dtype in zip(arrays, selected):
        if (isinstance(values, pd.arrays.SparseArray) and dtype != values.dtype
                and _masks_float_promotion(dtype.subtype)):
            stored = values.sp_index.to_int_index().indices
            rows = stored if rows is None else np.intersect1d(rows, stored, assume_unique=True)
    if rows is None or not len(rows):
        return False
    subset = []
    dtypes = []
    for values, taken_dtype in zip(arrays, selected):
        dtype = values.dtype
        if isinstance(values, pd.arrays.SparseArray) and values.dtype.subtype.kind in 'buifcMm':
            values = values.take(rows)
            if values.sp_index.ngaps:
                dtype = taken_dtype
        elif isinstance(values, (pd.Categorical, pd.arrays.IntegerArray)):
            values = values.take(rows)
        elif isinstance(values, np.ndarray) and values.dtype.kind in 'iu':
            values = values[..., rows]
        subset.append(values)
        dtypes.append(dtype)
    folded = ensure_np_dtype(interleaved_dtype(dtypes))
    return _frame_conversion_loses(subset, {}, folded, pd, block_rows, take=True)


def _check_sparse_search_target(y):
    pd = modules.get('pandas')
    if pd is None or isinstance(y, np.ndarray):
        return
    if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
        if _multioutput_sparse_target_loses(y, pd):
            raise ValueError(
                'This pandas sparse target loses class IDs during search conversion or row selection. '
                'Prepare owning target columns that retain their labels when the whole frame is converted. '
                'For mixed integer and floating labels, use exact object columns if the estimator accepts them.')
        return
    values = _pandas_target_array(y)
    if not (_has_sparse_levels(values, pd) or isinstance(values, pd.arrays.SparseArray)):
        return
    block_rows = isinstance(y, (pd.Series, pd.DataFrame)) and getattr(y._mgr, 'blocks', None) is not None
    array_frame = isinstance(y, pd.DataFrame) and not block_rows
    if isinstance(values, pd.Categorical):
        # The legacy ArrayManager boxes a one-column categorical frame.
        dtype = np.dtype(object) if array_frame else None
        if _frame_conversion_loses([values], {}, dtype, pd, block_rows, take=True):
            raise ValueError(
                'This pandas sparse categorical target loses class IDs during search conversion or row selection. '
                'Prepare an owning target from the original categorical scalar at each row.')
        return
    temporal_fill = isinstance(values.fill_value, _TEMPORAL_LABEL_TYPES)
    if values.dtype.subtype.name == 'uint64' and not array_frame and not temporal_fill:
        if _sparse_target_values_lose(values, pd, block_rows):
            raise ValueError(
                'This pandas sparse uint64 target loses class IDs during search conversion or row selection. '
                'Prepare y with owning values that preserve both the stored labels and the fill. '
                'For a Series with uint64-representable labels, use '
                'pd.Series(y.to_numpy(dtype=np.uint64, copy=True), index=y.index, name=y.name).')
        return
    kind = values.dtype.subtype.kind
    if ((kind not in 'buifcMmO' and not temporal_fill)
            or (kind in 'buifc' and type(values.fill_value) is values.dtype.subtype.type)):
        return
    # ArrayManager casts even a single output to its sparse subtype. Other
    # carriers use the sparse array's default conversion for a single output.
    dtype = values.dtype.subtype if array_frame else None
    probes = {}
    if (_frame_conversion_loses([values], probes, dtype, pd, block_rows) or
            _frame_conversion_loses([values], probes, dtype, pd, block_rows, take=True)):
        raise ValueError(
            'This pandas sparse target loses class IDs during search conversion or row selection. '
            'Prepare an owning target using the original scalar at each row.')


def _exact_fold_sparse_values(values, pd):
    if type(values) is not pd.arrays.SparseArray or values.dtype.subtype.name != 'uint64':
        return values
    fill = values.fill_value
    if not _sparse_uint64_fold_loses(values, pd):
        return values
    integral = (isinstance(fill, Integral) or
                isinstance(fill, (float, np.floating)) and np.isfinite(fill) and fill == np.floor(fill))
    if not integral or not 0 <= int(fill) < 2**64:
        raise ValueError(
            'Pandas cannot select exact feature values with this sparse uint64 fill. '
            'Supply exact owning feature values with a dtype that also represents the fill.')
    # Reuse the compressed payload, including private data just detached for
    # process transport. No dense float representation is created.
    dtype = pd.SparseDtype(values.dtype.subtype, np.uint64(fill))
    return type(values)._simple_new(values.sp_values, values.sp_index, dtype)


def _parallel_search_features(value, n_jobs, *, check_only=False, process=True):
    """Prepare an exact DataFrame for transport and lossless fold selection."""
    pd = modules.get('pandas')
    if type(value) is not getattr(pd, 'DataFrame', None):
        return _parallel_search_input(value, n_jobs)

    manager = value._mgr
    blocks = getattr(manager, 'blocks', None)
    if not process and (blocks is None or not manager.any_extension_types):
        # ArrayManager takes native row positions without fill promotion.
        return value
    if blocks is None:
        # Older pandas also has a manager with one stored array per column.
        arrays = getattr(manager, 'arrays', None)
        if arrays is None or len(arrays) != value.shape[1]:
            raise ValueError('Cannot inspect this pandas DataFrame storage manager for process search.')
    else:
        arrays = (block.values for block in blocks)

    replacements = {}
    for position, values in enumerate(arrays):
        prepared = _parallel_pandas_values(values, pd, n_jobs, check_only) if process else values
        if blocks is not None and not check_only:
            prepared = _exact_fold_sparse_values(prepared, pd)
        if prepared is not values:
            replacements[position] = prepared
    index = _parallel_pandas_index(value.index, pd, n_jobs, check_only) if process else value.index
    columns = _parallel_pandas_index(value.columns, pd, n_jobs, check_only) if process else value.columns
    if not replacements and index is value.index and columns is value.columns:
        return value

    # Preserve pandas' pickle metadata. DataFrame.copy(deep=False) also
    # deep-copies attrs, which can reject objects that loky can serialize.
    state = pd.DataFrame.__getstate__(value)
    state['_mgr'] = manager.copy(deep=False)
    result = object.__new__(pd.DataFrame)
    pd.DataFrame.__setstate__(result, state)
    if index is not value.index:
        result.index = index
    if columns is not value.columns:
        result.columns = columns
    if replacements:
        if blocks is None:
            for position, prepared in replacements.items():
                result.isetitem(position, prepared)
        else:
            result_blocks = list(result._mgr.blocks)
            for position, prepared in replacements.items():
                result_blocks[position] = blocks[position].make_block_same_class(prepared)
            result._mgr.blocks = tuple(result_blocks)
    return result


def _explicit_folds(cv):
    return (isinstance(cv, Iterable) and not isinstance(cv, (Integral, str, bytes))
            and not hasattr(cv, 'split') and not hasattr(cv, 'get_metadata_routing'))


def _inspect_pandas_selector(value, pd, n_jobs, bind=None):
    """Visit an exact Index or Series without replacing its public identity."""
    parents = ()
    if type(value) is pd.Series:
        if bind is not None:
            parents = (value, value._mgr)
        blocks = getattr(value._mgr, 'blocks', ()) if bind is not None else ()
        _parallel_pandas_values(value._values, pd, n_jobs, True, bind, parents + tuple(blocks))
        value = value.index
    _parallel_pandas_index(value, pd, n_jobs, True, bind, parents)


def _stored_fold_items(value):
    # Inspect only stored references, without calling an iterator supplied by
    # a custom container. tuple.__iter__ also covers ordinary named tuples.
    if isinstance(value, tuple):
        return tuple.__iter__(value)
    if isinstance(value, list):
        return list.__iter__(value)
    if isinstance(value, deque):
        return deque.__iter__(value)
    if type(value) in (np.ndarray, np.memmap) and value.dtype == object and value.ndim:
        return iter(value)
    return ()


def _reject_mapped_fold_pair(pair, n_jobs):
    _reject_mapped_objects(pair, n_jobs)
    pd = modules.get('pandas')
    for row in _stored_fold_items(pair):
        if pd is not None and (type(row) is pd.Index or type(row) is pd.Series):
            _inspect_pandas_selector(row, pd, n_jobs)
        else:
            _reject_mapped_objects(row, n_jobs)


def _reject_mapped_fold_contents(cv, n_jobs, checked_pairs=()):
    # Check each container before reading the references it holds.
    _reject_mapped_objects(cv, n_jobs)
    for index, pair in enumerate(_stored_fold_items(cv)):
        if index >= len(checked_pairs) or pair is not checked_pairs[index]:
            _reject_mapped_fold_pair(pair, n_jobs)


def _unchanged_checked_pair(pair, train, test, parallel):
    if type(pair) in (tuple, list, deque):
        return len(pair) == 2 and pair[0] is train and pair[1] is test
    if (isinstance(pair, tuple) and type(pair).__iter__ is tuple.__iter__
            and isinstance(getattr(type(pair), '_fields', None), tuple)):
        # Named tuples store the same two references as plain tuples.
        return (tuple.__len__(pair) == 2 and tuple.__getitem__(pair, 0) is train
                and tuple.__getitem__(pair, 1) is test)
    if (type(pair) not in (np.ndarray, np.memmap) or pair.ndim != 2
            or pair.shape[0] != 2 or pair.dtype.hasobject or _needs_process_snapshot(pair, parallel)):
        return False
    # Indexing a numeric matrix creates new row views. Compare their storage
    # descriptions, including offsets and strides, without reading the values.
    return all(type(row) in (np.ndarray, np.memmap)
               and row.__array_interface__ == view.__array_interface__
               for row, view in zip((train, test), pair))


class _SelectedSplits:
    """Delegate split after the selected override has already been truth-tested."""

    def __init__(self, selected, frame_executor=None):
        self.selected = selected
        self.frame_executor = frame_executor

    def split(self, *args, **kwargs):
        if self.frame_executor is not None:
            _parallel_search_features(args[0], self.frame_executor, check_only=True)
        return self.selected.split(*args, **kwargs)


@lru_cache(maxsize=8)
def _joblib_methods(cls):
    # Only joblib's known classes are cached, never a user subclass.
    return frozenset(name for base in cls.__mro__ for name, value in vars(base).items()
                     if callable(value) or isinstance(value, (staticmethod, classmethod, property)))


def _joblib_queue(backend):
    """Return the expected backend, pool and live queue for joblib's process transports."""
    loky = isinstance(backend, LokyBackend)
    if loky:
        from joblib.executor import MemmappingExecutor
        from joblib.externals.loky.process_executor import _SafeQueue

        backend_type, pool_name, pool_type = LokyBackend, '_workers', MemmappingExecutor
        queue_name, queue_type = '_call_queue', _SafeQueue
    elif isinstance(backend, MultiprocessingBackend):
        from joblib.pool import CustomizablePicklingQueue, MemmappingPool

        backend_type, pool_name, pool_type = MultiprocessingBackend, '_pool', MemmappingPool
        queue_name, queue_type = '_inqueue', CustomizablePicklingQueue
    else:
        return None
    pool = vars(backend).get(pool_name)
    if pool is None:
        return (backend_type, None, None) if type(backend) is backend_type else None
    if getattr_static(backend, pool_name) is not pool:
        return None
    if type(pool) is _SubmissionPool:
        pool = pool.pool
    if type(pool) is not pool_type:
        return None
    queue = vars(pool).get(queue_name)
    if type(queue) is not queue_type:
        return None
    return backend_type, pool, queue


def _custom_search_reducer(backend, cls):
    if cls in registered_reducers:
        return True
    transport = _joblib_queue(backend)
    if transport is None:
        return False
    backend_type, _, queue = transport
    # These hooks run after split collection and can replace the reducer
    # mapping before the batch is sent.
    for name in ('start_call', 'submit'):
        if getattr_static(backend, name) is not getattr_static(backend_type, name):
            return True
    if backend_type is LokyBackend:
        from joblib.externals.loky.backend import reduction

        if reduction.get_loky_pickler_name() not in ('cloudpickle', 'pickle') or cls in reduction._dispatch_table:
            return True
        table = getattr_static(reduction.get_loky_pickler(), 'dispatch_table', None)
        if isinstance(table, Mapping) and cls in table:
            return True
    reducers = vars(queue).get('_reducers') if queue is not None else None
    return reducers is not None and cls in reducers


def _native_fold_serializer(backend):
    transport = _joblib_queue(backend)
    if transport is None:
        return False
    backend_type, pool, queue = transport
    loky = backend_type is LokyBackend
    if loky:
        from joblib.externals.loky.backend.reduction import get_loky_pickler_name

        if get_loky_pickler_name() not in ('cloudpickle', 'pickle'):
            return False
    elif type(backend) is not MultiprocessingBackend:
        return False

    native_methods = _joblib_methods(backend_type)
    for cls in type(backend).__mro__:
        if cls is backend_type:
            break
        # A customized start_call can install a temporary submission observer.
        if not native_methods.isdisjoint(vars(cls)):
            return False
    state = vars(backend)
    if not native_methods.isdisjoint(state):
        return False
    if pool is None:
        # A task-collecting Parallel can postpone native pool creation.
        # Recheck the real executor when dispatch actually begins.
        return type(backend) is backend_type
    if not _joblib_methods(type(pool)).isdisjoint(vars(pool)) or not _joblib_methods(type(queue)).isdisjoint(vars(queue)):
        return False
    reducers = vars(queue).get('_reducers')
    if type(reducers) is not dict:
        return False
    if not loky and (vars(pool).get('_forward_reducers') is not reducers
                     or pool._quick_put is not queue._send):
        return False
    from joblib._memmapping_reducer import ArrayMemmapForwardReducer

    return all(type(reducers.get(kind)) is ArrayMemmapForwardReducer for kind in (np.ndarray, np.memmap))


def _standard_search_reduction(search, *, allow_getstate=False):
    names = ('__getattribute__', '__reduce__', '__reduce_ex__', '__getstate__', '__setstate__',
             '__getnewargs__', '__getnewargs_ex__')
    return all(getattr_static(search, name, _MISSING) is getattr_static(GridSearchCV, name, _MISSING)
               for name in names if not (allow_getstate and name == '__getstate__'))


def _restore_fold_ndarray(value, writeable):
    # A process that writes must neither mutate a shared joblib temporary nor
    # fail because the queue selected a read-only temporary for this payload.
    if writeable and (not value.flags.writeable or getattr(value, 'mode', None) in ('r+', 'w+')):
        return np.array(value, copy=True, order='K', subok=False)
    array = np.asarray(value)
    array.flags.writeable = writeable
    return array


class _FoldNdarrayPayload:
    def __init__(self, source, detached):
        # Keep both identities alive through the queue feeder. This distinct
        # ndarray view goes through joblib's original size and mode policy.
        self.source = source
        self.detached = detached
        self.transport = None if detached is None else detached.view()
        self.lock = Lock() if detached is None else None

    def snapshot(self):
        # A separately captured source can be serialized ahead of search.cv.
        # Either request must use the same snapshot, including concurrent
        # state requests. Unused whole stacks and numeric pairs stay lazy.
        if self.transport is None:
            with self.lock:
                if self.transport is None:
                    self.detached = _snapshot_search_input(self.source)
                    self.transport = self.detached.view()
        return self.detached

    def __reduce__(self):
        detached = self.snapshot()
        return _restore_fold_ndarray, (self.transport, detached.flags.writeable)


class _FoldArrays(dict):
    """Identify detached folds and numeric aliases from the same open mapping."""

    def __init__(self):
        super().__init__()
        self.sources = {}
        self.modes = {}
        self.views = {}
        self.selector_buffers = set()
        self.selector_carriers = {}
        self.custom_selector_state = False

    def bind(self, source, detached):
        payload = self.get(id(source))
        if payload is None:
            owner = _backing_memmap(source)
            payload = self[id(source)] = _FoldNdarrayPayload(source, detached)
            self.sources.setdefault(id(owner), []).append(source)
        if detached is not None:
            self[id(detached)] = payload
        return payload

    def overlapping_map(self, view):
        # Indexing a 3-D stack twice makes two row objects. Recognize the
        # scorer's view by its memory, keeping the identity of each object.
        if view.base is None or view.dtype.hasobject:
            return None
        owner = _backing_memmap(view)
        sources = self.sources.get(id(owner))
        if not sources:
            return None
        # Bound the stride-overlap calculation. NumPy conservatively returns
        # True if complex strides exceed the budget within the same mapping.
        return owner if any(np.may_share_memory(view, source, max_work=256) for source in sources) else None

    def selector_capture(self, value):
        """Does an unverified reducer control a selector or an overlapping capture?"""
        if not self.selector_buffers:
            return False
        if type(value) is _FoldNdarrayPayload:
            # A detached view can be a scorer's independent alias. Its source
            # is intentionally held weakly, so the payload alone cannot prove
            # that a custom reduction of it is independent of the selector.
            return value.source is None or id(value.source) in self.selector_buffers
        if id(value) in self.selector_buffers:
            return True
        if value.base is None or value.dtype.hasobject:
            return False
        sources = self.sources.get(id(_backing_memmap(value)), ())
        return any(id(source) in self.selector_buffers
                   and np.may_share_memory(value, source, max_work=256) for source in sources)

    def detached_view(self, view):
        key = id(view)
        cached = self.views.get(key)
        if cached is not None and cached[0]() is view:
            _, strides, payload = cached
            saved = payload.detached
            if (view.shape == saved.shape and view.strides == strides and view.dtype is saved.dtype
                    and view.flags.writeable == saved.flags.writeable and view.dtype.kind in 'biufc'
                    and view.dtype.metadata is None):
                # Reuse joblib's temporary only when the logical element bytes
                # still match, including NaN payloads and the sign of zero.
                element = (np.dtype(f'u{view.itemsize}') if view.itemsize in (1, 2, 4, 8)
                           else np.dtype((np.uint8, (view.itemsize,))))
                if np.array_equal(view.view(element), saved.view(element)):
                    return payload

        def discard(dead, registry=ref(self)):
            bound = registry()
            if bound is not None and bound.views.get(key, (None,))[0] is dead:
                del bound.views[key]

        payload = _FoldNdarrayPayload(None, _snapshot_search_input(view))
        self.views[key] = ref(view, discard), view.strides, payload
        return payload

    @staticmethod
    def reopen_needs_snapshot(view, reduced):
        # The queue already read the custom map's mode and storage order.
        # Inspect only a recognized native recipe, leaving opaque results alone.
        if (type(reduced) is tuple and len(reduced) == 2 and reduced[0] is _JOBLIB_MAP_OPENER
                and type(reduced[1]) is tuple and len(reduced[1]) == 9):
            args = reduced[1]
            mode = _map_mode_text(args[2])
            if mode is not None and type(args[4]) is str:
                return _unsafe_mapped_layout(view, None, mode=mode, fortran=args[4] == 'F')
        return None

    def raw_overlap(self, view, reduced):
        if view.base is None or view.dtype.hasobject or self.reopen_needs_snapshot(view, reduced) is False:
            return False
        return any(np.may_share_memory(view, source, max_work=256)
                   for sources in self.sources.values() for source in sources)


class _ProcessSplits:
    """Prepare sklearn's actual splits at dispatch without consuming public cv again."""

    def __init__(self, checked, parallel, cv=_MISSING, search=None, install_worker_state=None,
                 native_checked=False):
        self.checked = checked
        self.native_checked = native_checked
        self.cv = cv
        self.parallel = parallel
        self.search = search
        self.install_worker_state = install_worker_state
        self.make_checked = None
        self.rows = None
        self.fold_arrays = _FoldArrays()

    def _bind_fold_ndarray(self, source, detached=None):
        if type(source) not in (np.ndarray, np.memmap) or not (self.fold_arrays or _native_fold_serializer(self.parallel._backend)):
            return None
        # Both kinds already become owning ndarrays when DWD detaches them.
        # A direct capture of the unsafe original must use the same values.
        return self.fold_arrays.bind(source, detached)

    def _deferred_fold_array(self, source):
        payload = self._bind_fold_ndarray(source)
        if payload is None:
            return lambda: _snapshot_search_input(source)
        bindings = self.fold_arrays

        def make():
            detached = payload.snapshot()
            # Publish the identity before the native state hook or a custom
            # deepcopy hook can expose the detached array to this queue.
            bindings[id(detached)] = payload
            return detached

        return make

    def _defer_worker_cv(self, make_cv):
        self._worker_cv_lock = Lock()
        self._make_worker_cv = make_cv

    def _prepare_container_cv(self, memo, pending):
        # Only defer a full pair when its references can all be replaced in the
        # native outer shell. Custom records can also refer to that pair in
        # metadata. Give their deepcopy hooks the complete memo on this thread.
        simple_outer = (type(self.cv) in (list, tuple, deque)
                        or self.cv.ndim == 1 and self.cv.dtype == object)
        pending_snapshots = {id(pair): self._deferred_fold_array(pair) for pair in pending}
        if not (pending and simple_outer and all(
                id(pair) in memo or (type(pair) in (list, tuple, deque)
                                     and all(id(row) in memo and id(row) not in pending_snapshots
                                             for row in pair))
                for pair in self.cv)):
            for pair in pending:
                memo[id(pair)] = pending_snapshots[id(pair)]()
            self._worker_cv = deepcopy(self.cv, memo)
            return

        # The memo already pins every direct row and numeric pair. This copy
        # freezes the owned shells, preserving repeated records and row aliases.
        template = deepcopy(self.cv, memo)

        def make_cv():
            # Every top-level item is memoized. Only the native outer shell and
            # the pending numeric payloads are copied by the serialization thread.
            replacements = {id(pair): pair for pair in template}
            for pair in pending:
                replacements[id(pair)] = pending_snapshots[id(pair)]()
            return deepcopy(template, replacements)

        self._defer_worker_cv(make_cv)

    @property
    def worker_cv(self):
        # A scorer may carry the search even when sklearn omits caller=search.
        # Materialize at its first state request, including on a joblib feeder.
        if self._worker_cv_error is not None:
            raise BaseException.with_traceback(self._worker_cv_error, None)
        if self._make_worker_cv is not None:
            with self._worker_cv_lock:
                if self._make_worker_cv is not None:
                    self._worker_cv = self._make_worker_cv()
                    self._make_worker_cv = None
        return self._worker_cv

    def needs_worker_cv(self, cv):
        return (self._worker_cv_error is not None or self._make_worker_cv is not None
                or self._worker_cv is not cv)

    def _prepare_wrappers(self, memo, prepare, checked_changed, pairs):
        # Only the selected splitter was unpacked. Its yielded rows are in
        # the memo already, including references stored by custom records.
        # For an independent public wrapper inspect native storage only.
        pd = modules.get('pandas')

        def needs_copy(value):
            if isinstance(value, np.ndarray):
                if not value.dtype.hasobject:
                    return prepare(value) is not value
                _reject_mapped_objects(value, self.parallel)
            elif pd is not None and (type(value) is pd.Index or type(value) is pd.Series):
                return prepare(value) is not value
            return False

        wrappers = {self.checked: checked_changed} if self.native_checked else {}
        if type(self.cv) is _ITERABLE_CV:
            wrappers.setdefault(self.cv, False)
        changed_pairs = set()
        if checked_changed and self.cv is not self.checked:
            changed_pairs = {id(pair) for pair, (train, test) in zip(self.checked.cv, pairs)
                             if memo[id(train)] is not train or memo[id(test)] is not test}
        for wrapper in wrappers:
            changed = needs_copy(wrapper.cv)
            for pair in _stored_fold_items(wrapper.cv):
                changed |= needs_copy(pair) or id(pair) in changed_pairs
                for row in _stored_fold_items(pair):
                    changed |= needs_copy(row)
            wrappers[wrapper] |= changed
        if not any(wrappers.values()):
            return {}

        # Preserve the wrappers, their shared records and metadata as one graph.
        # Prepare all numeric leaves before invoking either wrapper's hooks. A
        # custom copy hook runs on the fitting thread. Only a consumer of
        # the prepared wrapper should observe a failure from that hook.
        graph = {**memo, id(self.search): self.search}
        factories = {}
        for wrapper, changed in wrappers.items():
            if not changed:
                graph[id(wrapper)] = wrapper
                graph[id(wrapper.cv)] = wrapper.cv
                for pair in _stored_fold_items(wrapper.cv):
                    graph[id(pair)] = pair
        for wrapper, changed in wrappers.items():
            if changed:
                copied = graph.copy()
                try:
                    prepared = deepcopy(wrapper, copied)
                except Exception as error:
                    def failed(failure=BaseException.with_traceback(error, None)):
                        raise BaseException.with_traceback(failure, None)

                    factories[wrapper] = failed
                else:
                    graph = copied
                    factories[wrapper] = lambda value=prepared: value
        return factories

    def split(self, *args, **kwargs):
        # The exact sklearn wrapper has already collected even a one-pass cv.
        # It unpacks each pair inside split(), before yielding the two indices.
        _reject_mapped_objects(self.checked.cv, self.parallel)
        for pair in self.checked.cv:
            _reject_mapped_objects(pair, self.parallel)
        # Collect references before taking snapshots: records may reuse index
        # buffers. Check the yielded rows without iterating custom pairs again.
        pairs = list(self.checked.split(*args, **kwargs))
        self._make_worker_cv = None
        self._worker_cv_error = None
        self.rows = []
        memo = {}
        changed = False
        pd = modules.get('pandas')

        def bind_selector_buffer(source, parents):
            # Pandas reduces its own wrapper and metadata. Only its unsafe
            # stored arrays need a binding in the existing worker pickler.
            payload = self.fold_arrays.get(id(source))
            if payload is None:
                _reject_mapped_objects(source, self.parallel)
                if not _needs_process_snapshot(source, self.parallel, self.fold_arrays.modes):
                    return
            if not self.fold_arrays.selector_buffers:
                backend = self.parallel._backend
                if (self.search is None or not _native_fold_serializer(backend)
                        or not _standard_search_reduction(self.search, allow_getstate=True)
                        or _custom_search_reducer(backend, type(self.search))):
                    raise PicklingError(_PANDAS_SELECTOR_TRANSPORT)
                self.fold_arrays.custom_selector_state = (getattr_static(self.search, '__getstate__')
                                                          is not getattr_static(GridSearchCV, '__getstate__'))
            if payload is None:
                self.fold_arrays.bind(source, _snapshot_search_input(source))
            self.fold_arrays.selector_buffers.add(id(source))
            # A custom reducer on an intermediate pandas object can omit
            # the array entirely. Record only the path to an affected leaf.
            for parent in parents:
                self.fold_arrays.selector_carriers[id(parent)] = parent

        def prepare(value):
            nonlocal changed
            key = id(value)
            if key not in memo:
                if pd is not None and (type(value) is pd.Index or type(value) is pd.Series):
                    _inspect_pandas_selector(value, pd, self.parallel, bind_selector_buffer)
                    safe = value
                else:
                    payload = self.fold_arrays.get(key) if self.fold_arrays.selector_buffers else None
                    safe = (payload.snapshot() if payload is not None else
                            _parallel_search_input(value, self.parallel, self.fold_arrays.modes))
                memo[key] = safe
                if safe is not value:
                    changed = True
                    self._bind_fold_ndarray(value, safe)
            return memo[key]

        for train, test in pairs:
            self.rows.append((prepare(train), prepare(test)))
        checked_changed = changed
        if self.cv is _MISSING:
            # These indices came from a separate splitter selected for this
            # evaluation. The public and original checked cv keep their state.
            if self.fold_arrays:
                self.install_worker_state(self.parallel, prepare_state=False)
            return iter(self.rows)
        # A serialized caller can be inspected by a scorer in the worker.
        # Keep ordinary public containers and named fields in that copy too.
        containers = (list, tuple, deque, np.ndarray, np.memmap)
        if self.native_checked or type(self.cv) is _ITERABLE_CV:
            factories = self._prepare_wrappers(memo, prepare, checked_changed, pairs)
            self.make_checked = factories.get(self.checked)
        if type(self.cv) is _ITERABLE_CV:
            make_cv = factories.get(self.cv)
            self._worker_cv = self.cv
            if make_cv is not None:
                self._defer_worker_cv(make_cv)
        elif not _explicit_folds(self.cv):
            self._worker_cv = self.cv
        elif isinstance(self.cv, Iterator):
            self._worker_cv = ()
        elif type(self.cv) in (np.ndarray, np.memmap) and not self.cv.dtype.hasobject:
            # A reversed fold axis likewise disappears from the split rows.
            if _needs_process_snapshot(self.cv, self.parallel, self.fold_arrays.modes):
                self._defer_worker_cv(self._deferred_fold_array(self.cv))
            else:
                self._worker_cv = self.cv
        elif type(self.cv) in containers:
            numeric_pairs = []
            different_records = False
            custom_records = False
            for i, (pair, public_pair) in enumerate(zip_longest(
                    self.checked.cv, _stored_fold_items(self.cv), fillvalue=_MISSING)):
                if public_pair is not pair:
                    different_records = True
                if public_pair is _MISSING:
                    continue
                if public_pair is not pair:
                    # sklearn already collected its fitting records. Prepare the
                    # currently stored public references without unpacking a
                    # custom record again or altering the separate fitting rows.
                    _reject_mapped_objects(public_pair, self.parallel)
                    for row in _stored_fold_items(public_pair):
                        prepare(row)
                if type(public_pair) in (np.ndarray, np.memmap) and not public_pair.dtype.hasobject:
                    numeric_pairs.append(public_pair)
                elif not isinstance(public_pair, containers) and public_pair is pair:
                    memo[id(pair)] = self.rows[i]
                    custom_records = True
            if custom_records and not different_records:
                self._worker_cv = self.rows
            else:
                pending = []
                for pair in numeric_pairs:
                    if id(pair) not in memo:
                        memo[id(pair)] = pair
                        if _needs_process_snapshot(pair, self.parallel, self.fold_arrays.modes):
                            pending.append(pair)
                if pending or changed or custom_records:
                    try:
                        # A fold may point back to this search. Let pickle bind
                        # that reference to the worker's search when it writes
                        # the graph. A deepcopy here would re-enter worker_cv.
                        if self.search is not None:
                            memo[id(self.search)] = self.search
                        self._prepare_container_cv(memo, pending)
                    except Exception as error:
                        # The checked rows are ready. An opaque record's copy
                        # hook must run on this thread, but some sklearn jobs
                        # never serialize the search. Deliver its failure only
                        # to a consumer of that state, without rerunning hooks.
                        self._worker_cv_error = BaseException.with_traceback(error, None)
                else:
                    self._worker_cv = self.cv
        else:
            self._worker_cv = self.rows
        if not self.native_checked and getattr(self.search, '_checked_cv_orig', None) is self.checked:
            # Preserve sklearn's own checked folds when they transport the
            # same rows, including native two-element lists and matrix views.
            original = self.checked.cv
            if (checked_changed or type(original) is not list or len(original) != len(self.rows)
                    or any(not _unchanged_checked_pair(pair, train, test, self.parallel)
                           for pair, (train, test) in zip(original, self.rows))):
                self.make_checked = lambda: check_cv(self.rows)
        if self.make_checked is not None or self.needs_worker_cv(self.search.cv):
            self.install_worker_state(self.parallel)
        elif self.fold_arrays:
            # The selected rows can still reach a scorer after a setup hook replaces public cv.
            self.install_worker_state(self.parallel, prepare_state=False)
        return iter(self.rows)


def _worker_pickle_state(state, make_replacements, saved_getstate, hook_ids):
    hook_error = ('Custom __getstate__ exposes a temporary search hook outside '
                  'an attribute dictionary or a sequence of (name, value) pairs. '
                  'Omit that hook from the state or fit this search with n_jobs=1.')
    if id(state) in hook_ids:
        if saved_getstate is _MISSING:
            raise TypeError(hook_error)
        return saved_getstate
    if id(state) in make_replacements:
        return make_replacements[id(state)]()
    shells = (dict, list, tuple)
    if type(state) not in shells:
        if (make_replacements and state is not None
                and type(state) not in (str, bytes, bool, int, float, complex)):
            raise TypeError('Preparing process folds requires __getstate__ to expose '
                            'fold references in a plain dict, list or tuple. Use a '
                            'standard pickle state or fit this search with n_jobs=1.')
        return state

    # Only copy paths that contain a replacement. The native hook may have
    # omitted the folds altogether. Keep unrelated containers, dictionary
    # keys and arbitrary objects (such as the scorer) with their own pickler.
    copies = {}
    containers = {}
    parents = {}
    changed = set()
    hook_owners = set()
    pending = [state]
    while pending:
        value = pending.pop()
        key = id(value)
        if key in containers:
            continue
        containers[key] = value
        for item in value.values() if type(value) is dict else value:
            child = id(item)
            if child in hook_ids:
                if saved_getstate is _MISSING:
                    hook_owners.add(key)
                else:
                    copies[child] = saved_getstate
                    changed.add(key)
            elif child in make_replacements:
                if child not in copies:
                    copies[child] = make_replacements[child]()
                changed.add(key)
            elif type(item) in shells:
                parents.setdefault(child, []).append(key)
                pending.append(item)

    # With no pre-existing instance hook there is no value to substitute.
    # Omit the synthetic attribute from a dict or a list/tuple of its items.
    # Any other visible alias would leave the worker with the live fit hook.
    omitted_records = set()
    for key in hook_owners:
        value = containers[key]
        if type(value) is dict:
            if any(type(name) is not str or name != '__getstate__'
                   for name, item in value.items() if id(item) in hook_ids):
                raise TypeError(hook_error)
            changed.add(key)
        else:
            if (value is state or len(value) != 2 or type(value[0]) is not str
                    or value[0] != '__getstate__' or id(value[1]) not in hook_ids):
                raise TypeError(hook_error)
            for parent in parents[key]:
                sequence = containers[parent]
                if (type(sequence) not in (tuple, list)
                        or not all(type(pair) in (tuple, list) and len(pair) == 2
                                   and type(pair[0]) is str for pair in sequence)):
                    raise TypeError(hook_error)
                changed.add(parent)
            omitted_records.add(key)
    if not changed:
        return state
    pending = list(changed)
    while pending:
        for parent in parents.get(pending.pop(), ()):
            if parent not in changed:
                changed.add(parent)
                pending.append(parent)

    # Allocate all affected mutable shells before filling any of them. Every
    # cycle among these native containers passes through a list or dictionary.
    for key in changed:
        value = containers[key]
        if type(value) is not tuple:
            copies[key] = value.copy()

    def worker_value(value):
        key = id(value)
        if key in changed and key not in copies:
            copies[key] = tuple(worker_value(item) for item in value if id(item) not in omitted_records)
        return copies.get(key, value)

    for key in changed:
        value = containers[key]
        if type(value) is list:
            copies[key][:] = [worker_value(item) for item in value if id(item) not in omitted_records]
        elif type(value) is dict:
            for name, item in value.items():
                if key in hook_owners and id(item) in hook_ids:
                    del copies[key][name]
                else:
                    replacement = worker_value(item)
                    if replacement is not item:
                        copies[key][name] = replacement
    return worker_value(state)


def _stored_search_method(search, name):
    # A data descriptor (in particular a named slot) takes precedence over
    # the instance dictionary, even if the dictionary contains the same key.
    if isdatadescriptor(getattr_static(type(search), name, None)):
        return getattr(search, name, _MISSING)
    return vars(search).get(name, _MISSING)


def _restore_search_method(search, name, saved):
    if saved is _MISSING:
        delattr(search, name)
    else:
        setattr(search, name, saved)


def _as_is(value):
    return value


_ATOMS = frozenset(map(id, (type(None), bool, int, float, str, bytes)))
_CONTAINERS = frozenset(map(id, (list, tuple, dict)))


def _scope_value(value, end):
    return value


class _ScopeReference:
    __slots__ = ('value',)

    def __init__(self, value, end):
        self.value = value


class _NativeMemoRef:
    def __init__(self, graph, value):
        self.graph = graph
        self.value = value

    def __reduce__(self):
        self.graph.pickler()._dwd_state.start = (self.value, None)
        return _as_is, (self.value,)


def _allocate_view(plan, key, original):
    return type(original)() if key in plan[0] else original


def _choose_tuple_view(plan, key, original, values):
    return values if key in plan[0] else original


def _populate_view(target, body):
    plan, key, ledger, entries, point = body
    changed, writes = plan
    if key in changed:
        saved, position = ledger
        saved.extend(entries)
        stop = len(saved) if point is None else min(len(saved), writes[point])
        if type(target) is list:
            target.extend(saved[position:stop])
        else:
            for name, value in saved[position:stop]:
                target[name] = value
        ledger[1] = stop
    return target


class _ReplayView:
    def __init__(self, graph, scope):
        self.graph = graph
        self.scope = scope
        self.ledger = [[], 0]
        self.sent = self.scheduled = 0
        self.points = []

    def body(self, complete):
        scope = self.scope
        active = not scope.active
        stop = len(scope.entries) if complete else scope.complete_entries
        entries = scope.entries[self.sent:stop]
        self.sent = stop
        if scope.mapping:
            entries = [(self.graph.replay(name, False), self.graph.replay(value, active))
                       for name, value in entries]
        else:
            entries = [self.graph.replay(value, active) for value in entries]
        point = None
        if not complete:
            point = self.graph.next_point.to_bytes(8, 'little')
            self.graph.next_point += 1
            self.points.append(point)
        return self.graph.plan, id(scope.source), self.ledger, entries, point

    def __reduce__(self):
        scope = self.scope
        args = self.graph.plan, id(scope.source), _NativeMemoRef(self.graph, scope.source)
        if type(scope.source) is tuple:
            # A tuple cannot change its immediate children.
            values = tuple(self.graph.replay(value, not scope.active) for value in scope.source)
            return _choose_tuple_view, (*args, values)
        return _allocate_view, args, self.body(not scope.open), None, None, _populate_view


class _SyncView:
    def __init__(self, view, complete):
        self.view = view
        self.complete = complete

    def __reduce__(self):
        return _populate_view, (self.view, self.view.body(self.complete))


def _after_sync(syncs, value):
    return value


class _BeforeScope:
    def __init__(self, value, updates):
        self.value = value
        self.updates = updates

    def __reduce__(self):
        self.value.graph.pickler()._dwd_state.start = (self.updates, None)
        return _after_sync, (self.updates, self.value)


class _FinishReplay:
    def __init__(self, graph, view, parent):
        self.graph = graph
        self.view = view
        self.parent = parent

    def __reduce__(self):
        state = self.graph.pickler()._dwd_state
        update = _SyncView(self.view, True)
        state.start = (update, None)
        state.ends.append((self.parent, None))
        return _ScopeReference, (update, state.end_marker)


class _WorkerScope:
    """Let the queue write the original object, including its native memo."""

    def __init__(self, graph, value, *, parent=None, reference=False, active=True, state=False):
        self.graph = graph
        self.value = value
        self.parent = parent
        self.reference = reference
        self.active = active
        self.state = state

    def __reduce__(self):
        graph = self.graph
        state = graph.pickler()._dwd_state
        value = self.value
        scope = None
        make = graph.replacements.get(id(value)) if self.active else None
        if make is not None:
            value = make()
        elif id(type(value)) in _CONTAINERS:
            value, scope = graph.enter(value, self.active)
        elif self.state:
            value = _worker_pickle_state(value, graph.replacements, _MISSING, ())
        # Suspend observation of this helper's reduction args up to the
        # actual value. Native pickle visits the private final argument
        # depth-first, even when that argument is already in its memo.
        state.start = (value, scope)
        state.ends.append((self.parent, scope))
        restore = _ScopeReference if self.reference else _scope_value
        return restore, (value, state.end_marker)


class _ContainerScope:
    def __init__(self, source, active):
        self.source = source
        self.active = active
        self.mapping = self.key = type(source) is dict
        # Tuple edges cannot be inserted, removed or reordered by pickle
        # callbacks. The native tuple itself preserves the complete record.
        self.entries = source if type(source) is tuple else []
        self.complete_entries = 0
        self.open = 0


class _WorkerReductionGraph:
    def __init__(self, pickler):
        self.replacements = {}
        self.pickler = ref(pickler)
        self.scopes = {}
        self.views = {}
        self.open_views = {}
        self.plan = object()
        self.next_point = 0

    def affected(self, source, include_open):
        pending = [source]
        seen = set()
        while pending:
            value = pending.pop()
            key = id(value)
            if key in self.replacements:
                return True
            if key not in seen:
                seen.add(key)
                scope = self.scopes.get(key)
                if scope is not None:
                    if include_open and scope.open:
                        return True
                    if scope.mapping:
                        pending.extend(value for _, value in scope.entries)
                    else:
                        pending.extend(scope.entries)
        return False

    def enter(self, source, active):
        key = id(source)
        scope = self.scopes.get(key)
        if scope is None:
            scope = self.scopes[key] = _ContainerScope(source, active)
        elif scope.active != active:
            view = self.views.get(key)
            if view is None:
                if not self.affected(source, include_open=True):
                    # Raw tuples use their immutable edges directly. A
                    # nonempty tuple can be revisited before its final memo
                    # entry, while a later mutable child is still unseen.
                    if type(source) is not tuple or not source or key in self.pickler().memo.copy():
                        return source, None
                view = self.views[key] = _ReplayView(self, scope)
                if scope.open and type(source) is not tuple:
                    self.open_views[key] = view
            return view, None
        if type(source) is tuple and not active:
            return source, None
        scope.open += 1
        return source, scope

    def replay(self, value, active):
        if id(type(value)) in _CONTAINERS or id(value) in self.replacements:
            return _WorkerScope(self, value, active=active)
        return value

    def defer(self, value, state=False):
        if state:
            return _WorkerScope(self, value, state=True)
        return self.replay(value, True)


class _WorkerCVFactory:
    def __init__(self, splits):
        self.splits = splits

    def __call__(self):
        return self.splits.worker_cv

    def source_records(self):
        records = self.splits.cv if type(self.splits.cv) is tuple else self.splits.checked.cv
        return records if type(records) in (tuple, list) else ()

    def constructor_cv(self):
        splits = self.splits
        prepared = self()
        records = splits.checked.cv
        if prepared is not splits.rows or type(records) not in (tuple, list):
            return prepared
        # The normalized splitter creates a fresh pair at each position. A
        # constructor can also read the caller's repeated pair by identity.
        # Keep differing snapshots distinct, even for one source record.
        seen = {}
        result = prepared
        for i, (source, pair) in enumerate(zip(records, prepared)):
            key = id(source), id(pair[0]), id(pair[1])
            first = seen.setdefault(key, pair)
            if first is not pair:
                if result is prepared:
                    result = prepared.copy()
                result[i] = first
        return result

    def records(self, prepared):
        # sklearn already collected these records. Never call the public
        # CV's iterator again, even when it doubles as a keyword mapping.
        splits = self.splits
        yield splits.cv, prepared
        # A public tuple can contain separately copied pairs. A constructor
        # reading those pairs must see the same objects as the worker's cv.
        records = self.source_records()
        if type(prepared) in (tuple, list):
            pairs = {}
            for source, pair in zip(records, prepared):
                key = id(source)
                if pairs.setdefault(key, pair) is not pair:
                    # A repeatedly yielded record with different snapshots
                    # does not identify one prepared pair.
                    pairs[key] = _MISSING
            for source in records:
                pair = pairs.pop(id(source), _MISSING)
                if pair is not _MISSING:
                    yield source, pair
        for payload in splits.fold_arrays.values():
            if payload.source is not None and payload.detached is not None:
                yield payload.source, payload.detached


def _constructor_members(value, bindings):
    # NEWOBJ_EX expands the reconstructed container's stored members.
    # Its own reducer can reorder or omit fields, or create new ones.
    if isinstance(value, tuple):
        members = tuple(tuple.__iter__(value))
    elif isinstance(value, dict):
        members = {key: item for key, item in dict.items(value)}
    else:
        return value
    if not members:
        return value
    replacements = {id(source): partial(_as_is, prepared) for source, prepared in bindings
                    if prepared is not source}
    projected = _worker_pickle_state(members, replacements, _MISSING, ())
    return value if projected is members else projected


class _NativeConstructorArgs:
    def __new__(cls, /, *args, **kwargs):
        return args, kwargs

    @classmethod
    def capture(cls, positional, keywords):
        # Keep one tuple holding the original arguments. In older protocols
        # functools.partial can retain a fresh mapping and copy a shared one.
        capture = object.__new__(cls)
        capture.fields = cls, positional, keywords
        return capture

    def __reduce__(self):
        return __newobj_ex__, self.fields


class _ConstructorProjection:
    def __init__(self, native, bindings):
        self.native = native
        self.bindings = bindings

    def __reduce__(self):
        return _constructor_members, (self.native, self.bindings)

    def slot(self, index):
        bindings = self.bindings
        if bindings.selector.extended:
            return _constructor_arg, (self, index)
        return _as_is, (bindings.graph.defer(self.native.fields[index + 1]),)


def _constructor_arg(value, index):
    return value[index]


class _ConstructorBindings:
    def __init__(self, value, factory, graph, selector, positional, keywords, state):
        self.factory = factory
        self.graph = graph
        self.selector = selector
        self.prepare = lru_cache(maxsize=1)(factory.constructor_cv)
        self.roots = (state, *tuple.__iter__(positional), *dict.values(keywords))
        graph.replacements = _ConstructorReplacements(graph.replacements, value, factory, self.prepare, selector)

    def records(self):
        return self.factory.records(self.prepare())

    def __reduce__(self):
        # The original constructor container has just been saved by its
        # native reducer. Only send references it actually serialized.
        memo = self.graph.pickler().memo.copy()
        bindings = [(source, prepared) for source, prepared in self.records() if id(source) in memo]
        # A constructor reducer can reuse a list or tuple also exposed in
        # its companion arguments or plain search state. Reuse the graph's
        # active view of such a shell, including when it was saved first.
        pending = list(self.roots)
        seen = set()
        while pending:
            source = pending.pop()
            key = id(source)
            if key in seen or id(type(source)) not in _CONTAINERS or key in self.graph.replacements:
                continue
            seen.add(key)
            if key in memo and self.graph.affected(source, include_open=False):
                bindings.append((self.graph.replay(source, False), self.graph.defer(source)))
            pending.extend(source.values() if type(source) is dict else source)
        return _as_is, (bindings,)


class _ConstructorReplacements(dict):
    """Publish the whole CV's row bindings at the first use of a member."""

    def __init__(self, previous, value, factory, prepare, selector):
        super().__init__(previous)
        self[id(value)] = prepare
        self.factory = factory
        self.prepare = prepare
        self.selector = selector
        self.pending = {id(record) for record in factory.source_records()}
        self.pending.update(id(p.source) for p in factory.splits.fold_arrays.values() if p.source is not None)
        self.pending.difference_update(self)

    def resolve(self, key):
        if key in self.pending and self.selector.extended:
            for source, prepared in self.factory.records(self.prepare()):
                if source is not prepared:
                    self.setdefault(id(source), partial(_as_is, prepared))
            self.pending.clear()

    def __contains__(self, key):
        self.resolve(key)
        return super().__contains__(key)

    def get(self, key, default=None):
        self.resolve(key)
        return super().get(key, default)

    def clear(self):
        super().clear()
        self.pending.clear()
        self.factory = self.prepare = self.selector = None


def _call_constructor(new, cls, args, kwargs):
    return new(cls, *args, **kwargs)


class _ConstructorCallable:
    """Observe the opcode choice only when a CV is a whole argument container."""

    def __init__(self, function, cls, protocol):
        self.function = function
        self.cls = cls
        self.protocol = protocol
        self.extended = False

    @property
    def __name__(self):
        name = self.function.__name__
        self.extended = issubclass(type(name), str) and str.__eq__(name, '__newobj_ex__')
        if self.extended and self.protocol < 4:
            # The inner native pickle performs protocol 3's partial argument
            # expansion. Bind the real constructor once, as native pickle does.
            self.function = partial(_call_constructor, self.cls.__new__)
            return '_call_constructor'
        return name

    def __call__(self, *args):
        return self.function(*args)

    def __reduce__(self):
        return _as_is, (self.function,)


class _WorkerPositionalArgs(tuple):
    def __new__(cls, value, graph):
        result = super().__new__(cls, (graph.defer(item) for item in tuple.__iter__(value)))
        result.scoped = graph.defer(value)
        return result

    def __reduce__(self):
        return _as_is, (self.scoped,)


class _WorkerKeywordArgs(dict):
    def __init__(self, value, graph):
        super().__init__(value)
        self.scoped = graph.defer(value)

    def __reduce__(self):
        return _as_is, (self.scoped,)


class _FoldConstructorTuple(tuple):
    def __new__(cls, projected):
        result = super().__new__(cls)
        result.projected = projected
        return result

    def __reduce__(self):
        return self.projected.slot(0)


class _FoldConstructorDict(dict):
    def __init__(self, projected):
        self.projected = projected

    def __reduce__(self):
        return self.projected.slot(1)


def _constructor_slot(value, kind, graph):
    if type(value) is kind:
        wrapper = _WorkerPositionalArgs if kind is tuple else _WorkerKeywordArgs
        return wrapper(value, graph)
    return graph.defer(value)


def _worker_constructor_args(function, args, graph, state):
    if tuple.__len__(args) == 3:
        cls, positional, keywords = tuple.__getitem__(args, slice(None))
        # Pickle recognizes NEWOBJ_EX by the callable's name. Adapt valid
        # argument triples without a second, possibly observable name
        # lookup. Subclasses keep their own reduction and native memo.
        if (issubclass(type(cls), type) and issubclass(type(positional), tuple)
                and issubclass(type(keywords), dict)):
            factories = [getattr(graph.replacements.get(id(v)), '__wrapped__', None) for v in (positional, keywords)]
            for value, factory in zip((positional, keywords), factories):
                if isinstance(factory, _WorkerCVFactory) and callable(function):
                    selector = _ConstructorCallable(function, cls, graph.pickler()._dwd_state.protocol)
                    bindings = _ConstructorBindings(value, factory, graph, selector, positional, keywords, state)
                    projected = _ConstructorProjection(_NativeConstructorArgs.capture(positional, keywords), bindings)
                    return selector, (cls, _FoldConstructorTuple(projected), _FoldConstructorDict(projected))
            positional = _constructor_slot(positional, tuple, graph)
            keywords = _constructor_slot(keywords, dict, graph)
            return function, (cls, positional, keywords)
    if function is __newobj_ex__:
        return function, args  # Let the native pickler diagnose the malformed triple.
    return function, tuple(graph.defer(value) for value in tuple.__iter__(args))


def _worker_search_reduction(reduced, replacements, graph):
    if not replacements or not isinstance(reduced, tuple) or not 2 <= tuple.__len__(reduced) <= 6:
        return reduced
    # Pickle reads reduction fields directly, even on a tuple subclass.
    fields = tuple.__getitem__(reduced, slice(None))
    streams = [i for i in range(3, min(5, len(fields)))
               if getattr_static(type(fields[i]), '__next__', _MISSING) is not _MISSING]
    graph.replacements = {key: lru_cache(maxsize=1)(make) for key, make in replacements.items()}

    def worker_items(iterator, dictitems):
        # Pickle can fetch the next item before writing the previous one.
        # Retain the original reference until pickle actually writes it.
        while True:
            try:
                item = next(iterator)
            except StopIteration:
                return
            if not dictitems:
                yield graph.defer(item)
            elif isinstance(item, tuple) and tuple.__len__(item) == 2:
                key = tuple.__getitem__(item, 0)
                value = tuple.__getitem__(item, 1)
                yield key, graph.defer(value)
            else:
                yield item  # Let the native pickler diagnose a malformed pair.

    prepared = list(fields)
    args = fields[1]
    if isinstance(args, tuple):
        prepared[0], prepared[1] = _worker_constructor_args(fields[0], args, graph, fields[2] if len(fields) > 2 else None)
    if len(fields) > 2 and fields[2] is not None:
        # Native pickle writes state after consuming both item iterators.
        # They can legitimately update that state as their final action.
        prepared[2] = graph.defer(fields[2], state=True)
    for i in streams:
        prepared[i] = worker_items(fields[i], i == 4)
    return tuple(prepared)


class _WorkerPickleState:
    """Fold bindings belonging to one dump on a backend-owned pickler."""

    def __init__(self, batch, pickler, buffer, protocol):
        self.batch = batch.batch
        self.search = batch.search
        self.replacements = batch.replacements
        self.protocol = protocol
        self.buffer = buffer
        self.primary = None
        self.dump_started = False
        self.id_prefix = urandom(16)
        self.bindings = []
        self.graph = None
        self.array_policy = None
        self.raw_captures = None
        self.applied_raw = None
        self.conflicting_captures = False
        self.deferred_replacements = batch.deferred_replacements
        if self.replacements is not None:
            self.track_search(pickler, self.replacements)

    def track_search(self, pickler, replacements):
        self.replacements = replacements
        self.graph = _WorkerReductionGraph(pickler)
        self.start = self.scope = None
        self.end_marker = bytes(2)
        self.ends = []
        self.scoped_ids = False

    def external_id(self, value):
        return None

    def capture_policy(self, pickler):
        if self.raw_captures is not None:
            return self.raw_captures
        if self.applied_raw is not None:
            return self.applied_raw
        table = _pickler_dispatch_table(pickler)
        # Inspect concrete keys by identity. A speculative mapping lookup
        # could run __missing__, or a user-defined metaclass hash, twice.
        return issubclass(type(table), dict) and any(key is type(self.search) for key in tuple(dict.keys(table)))

    def select_capture_policy(self, raw):
        # Fast picklers can revisit a search. Its choices only constrain a
        # mapped capture after the stream actually used that capture policy.
        if self.applied_raw is not None and self.applied_raw != raw:
            self.conflicting_captures = True
        self.raw_captures = raw

    def fold_id(self, value):
        # Delimit the native serialization of each immediate value.
        # Keys and opaque objects keep their own independent reductions.
        graph = self.graph
        if value is graph.plan:
            return self.id_prefix
        if value is self.end_marker:
            self.scope, finished = self.ends.pop()
            if finished is not None:
                finished.open -= 1
                if not finished.open:
                    view = graph.open_views.pop(id(finished.source), None)
                    if view is not None:
                        parent, self.scope = self.scope, None
                        return _FinishReplay(graph, view, parent)
            return None
        if self.start is not None:
            if value is self.start[0]:
                _, self.scope = self.start
                self.start = None
                return self.external_id(value)
            return None
        scope = self.scope
        if scope is None:
            if id(type(value)) not in _CONTAINERS:
                return self.external_id(value)
            if type(value) is tuple:
                # No wrapper or mutable traversal cursor is needed for
                # an ordinary tuple. Its children keep their own pickler.
                known = graph.scopes.get(id(value))
                if known is None:
                    graph.scopes[id(value)] = _ContainerScope(value, False)
                    return self.external_id(value)
                if not known.active:
                    return self.external_id(value)
        active = False
        if scope is not None:
            scope.complete_entries = len(scope.entries)
            if scope.key:
                scope.name = value
            else:
                active = scope.active
                if type(scope.source) is not tuple:
                    scope.entries.append((scope.name, value) if scope.mapping else value)
            if scope.mapping:
                scope.key = not scope.key
        if id(type(value)) in _ATOMS:
            return self.external_id(value)
        if not active or id(value) not in graph.replacements:
            external = self.external_id(value)
            if external is not None:
                return external
        if not graph.pickler().fast and not graph.open_views and (not active or id(value) not in graph.replacements):
            if type(value) is list or type(value) is dict:
                previous = graph.scopes.get(id(value))
                if previous is not None and not previous.open and previous.active == active:
                    # Native pickle has already memoized this exact
                    # mutable shell. Its GET emits no further children.
                    return None
        self.scope = None
        self.scoped_ids = True
        token = _WorkerScope(graph, value, parent=scope, reference=True, active=active)
        if graph.open_views:
            updates = []
            for view in graph.open_views.values():
                stop = view.scope.complete_entries
                if stop > max(view.sent, view.scheduled):
                    view.scheduled = stop
                    updates.append(_SyncView(view, False))
            if updates:
                return _BeforeScope(token, tuple(updates))
        return token


class _PayloadPickleState(_WorkerPickleState):
    def external_id(self, value):
        key = id(value)
        if key not in self.external and not (
                self.separate_bytes and type(value) is bytes and len(value) >= 64 * 1024):
            return None
        # Hold the actual objects for the outer pickler. Equal byte strings
        # can be distinct, and a direct backend substitution need not
        # apply to a separately captured reference.
        if key not in self.positions:
            self.positions[key] = self.id_prefix + len(self.bindings).to_bytes(8, 'big')
            self.bindings.append(value)
        return self.positions[key]


_PERSISTENT_ID = Pickler.persistent_id
_LIVE_PERSISTENT_ID = isdatadescriptor(_PERSISTENT_ID)
_WorkerArrayPolicy = namedtuple('_WorkerArrayPolicy', 'native wrappers bindings')


def _empty_persistent_id(value):
    return None


def _pickler_dispatch_table(pickler, default=registered_reducers):
    # A Python attribute can shadow the C slot that actually selects reducers.
    try:
        return Pickler.dispatch_table.__get__(pickler, type(pickler))
    except AttributeError:
        return default


def _registered_reducer(pickler, kind, missing):
    table = _pickler_dispatch_table(pickler, missing)
    if table is missing:
        # The C pickler's default dict lookup propagates hash/equality
        # failures, even a KeyError. Only an absent entry is a miss.
        return registered_reducers.get(kind, missing)
    try:
        return table[kind]
    except KeyError:
        return missing


def _instance_reduction(value, protocol):
    method = getattr(value, '__reduce_ex__', _MISSING)
    return value.__reduce__() if method is _MISSING else method(protocol)


def _call_pickler_method(method, owner, value):
    return method(owner(), value)


class _WorkerPickleStream(Pickler):
    """Capture the native pickle of the submitted batch with its fold bindings."""

    _dwd_primary_active = False

    if _LIVE_PERSISTENT_ID:
        # Older C picklers consult a writable callback slot on every object.
        # A backend's own Python method or descriptor precedes this shim in the MRO.
        @property
        def persistent_id(self):
            native = _PERSISTENT_ID.__get__(self, type(self))
            if self._dwd_primary_active and native is self._dwd_installed_id:
                native = self._dwd_native_id
            if native is _MISSING or native is _empty_persistent_id:
                raise AttributeError('persistent_id')
            return native

        @persistent_id.setter
        def persistent_id(self, callback):
            wrapped = self._dwd_primary_active and _PERSISTENT_ID.__get__(self, type(self)) is self._dwd_installed_id
            _PERSISTENT_ID.__set__(self, callback)  # Preserve the native callable check.
            if wrapped:
                self._dwd_install_id(callback)

        @persistent_id.deleter
        def persistent_id(self):
            _PERSISTENT_ID.__delete__(self)

        def _dwd_install_id(self, native):
            self._dwd_native_id = native
            callback = self._dwd_local_id if native is _MISSING or native is _empty_persistent_id else self._dwd_live_id
            self._dwd_installed_id = callback
            _PERSISTENT_ID.__set__(self, callback)

        def _dwd_live_id(self, value):
            # The backend may read, decorate or replace the C slot directly
            # during its callback. Resume with its final binding before the
            # C pickler visits a child, including after a backend veto.
            native = self._dwd_native_id
            _PERSISTENT_ID.__set__(self, native)
            try:
                token = native(value)
            finally:
                self._dwd_native_id = _PERSISTENT_ID.__get__(self, type(self))
                _PERSISTENT_ID.__set__(self, self._dwd_installed_id)
            if token is not None:
                raise PicklingError('Custom persistent IDs require a matching backend unpickler.')
            return self._dwd_local_id(value)

    def _dwd_checked_id(self, value):
        if self._dwd_native_id(value) is not None:
            raise PicklingError('Custom persistent IDs require a matching backend unpickler.')
        return self._dwd_local_id(value)

    def _dwd_select_id(self, native):
        self._dwd_native_id = native
        callback = self._dwd_checked_id
        # The native persistent-ID lookup can itself inspect reducer_override.
        # Only the C pickler's next lookup selects the worker reducer.
        self._dwd_lookup_reducer = self._dwd_state.replacements is not None
        return callback

    def dump(self, value):
        state = vars(self).get('_dwd_state')
        if state is None or value is not state.batch:
            return super().dump(value)
        if state.dump_started or state.buffer.tell() or state.buffer.getvalue():
            raise PicklingError(_OWNING_FOLD_MESSAGE)
        state.dump_started = True
        if state.deferred_replacements is not None:
            if (state.deferred_replacements is dict and state.array_policy is not None
                    and state.array_policy.bindings.selector_buffers):
                # These batches bind selector buffers but have no CV container
                # to replace. Keep the live reducer checks; the native memo
                # already preserves the identity of every untouched container.
                state.replacements = dict
            else:
                state.track_search(self, state.deferred_replacements)
                self._dwd_local_id = state.fold_id
        if _LIVE_PERSISTENT_ID:
            try:
                native = _PERSISTENT_ID.__get__(self, type(self))
            except AttributeError:
                native = _MISSING
            # CPython can keep a class method as an unbound function and
            # manufacture a fresh bound method on each descriptor read.
            if (type(native) is MethodType and native.__self__ is self
                    and _PERSISTENT_ID.__get__(self, type(self)) is not native):
                self._dwd_unbound_id = native
            self._dwd_install_id(native)
        else:
            # Newer C picklers select this hook at dump entry, before reducer_override.
            self._dwd_lookup_id = True
        self._dwd_primary_active = True
        self._dwd_lookup_reducer = _LIVE_PERSISTENT_ID and state.replacements is not None
        try:
            result = super().dump(value)
            if _LIVE_PERSISTENT_ID and _PERSISTENT_ID.__get__(self, type(self)) is not self._dwd_installed_id:
                raise PicklingError(_OWNING_FOLD_MESSAGE)
            if not self._dwd_lookup_reducer and (_LIVE_PERSISTENT_ID or not self._dwd_lookup_id):
                state.primary = self._dwd_capture_primary(state)
            return result
        finally:
            self._dwd_lookup_reducer = False
            self._dwd_primary_active = False
            if _LIVE_PERSISTENT_ID:
                if _PERSISTENT_ID.__get__(self, type(self)) is self._dwd_installed_id:
                    native = self._dwd_native_id
                    _PERSISTENT_ID.__set__(self, _empty_persistent_id if native is _MISSING else native)
                self._dwd_installed_id = None
            else:
                self._dwd_lookup_id = False
            self._dwd_native_id = None

    def _dwd_capture_primary(self, state):
        # Capture the memo and external references before a backend can clear
        # them or write a trailer. The ordinary worker loader reads one object.
        serialized = state.buffer.getvalue()
        if not serialized:
            raise PicklingError(_OWNING_FOLD_MESSAGE)
        bindings = tuple(state.bindings)
        if state.graph is not None and (state.start is not None or state.ends):
            raise PicklingError(_OWNING_FOLD_MESSAGE)
        if state.replacements is not None and (state.conflicting_captures
                                               or (state.applied_raw and state.raw_captures is None)):
            raise PicklingError("The mapped fold capture policy does not match the completed batch. "
                                "Build the folds from owning indices before creating the search and scorer.")
        if state.graph is None:
            return (_load_worker_items, (serialized, bindings, state.id_prefix)) if bindings else (loads, (serialized,))
        if not bindings and not state.scoped_ids:
            return loads, (serialized,)
        views = state.graph.views
        plan = frozenset(key for key, view in views.items()
                         if state.graph.affected(view.scope.source, include_open=False))
        observations = {}
        partial_views = [views[key] for key in plan if views[key].points]
        if partial_views:
            memo = self.memo.copy()
            for view in partial_views:
                receiver = memo[id(view.scope.source)][0]
                for point in view.points:
                    observations[memo[id(point)][0]] = (point, receiver)
        return _load_worker_items, (serialized, bindings, state.id_prefix, plan, observations)


_OWNING_FOLD_MESSAGE = ('Could not verify the original batch and its fold indices in the custom pickler\'s primary stream. '
                        'Build the folds from owning indices before creating the search and scorer.')
_UNVERIFIED_MAP_LAYOUT = ('Could not verify the native layout of an overlapping mapped view. '
                          'Build the folds and scorer from owning arrays before fitting.')
_PANDAS_SELECTOR_TRANSPORT = ('Cannot preserve mapped pandas fold indices with this worker serializer. '
                              'Build the folds and scorer with owning indices before fitting.')


def _stock_worker_pickler(base):
    from multiprocessing.reduction import ForkingPickler
    from joblib.externals.loky.backend import reduction
    from joblib.pool import CustomizablePickler

    if base is CustomizablePickler or base is ForkingPickler:
        return True
    if base is not reduction.get_loky_pickler():
        return False
    name = reduction.get_loky_pickler_name()
    if name == 'pickle':
        return base.__bases__ == (Pickler,)
    cloudpickle = getattr(reduction, 'cloudpickle', None)
    return name == 'cloudpickle' and base.__bases__ == (getattr(cloudpickle, 'CloudPickler', None),)


def _cache_pickler_type(make):
    # A backend class can close over training data. Its adapter inherits
    # from it, so neither cache keys nor values may keep the classes alive.
    adapters = WeakValueDictionary()

    def reuse(base):
        key = id(base)
        adapter = adapters.get(key)
        if adapter is None:
            adapter = adapters[key] = make(base)
        return adapter

    return reuse


@_cache_pickler_type
def _worker_pickler_type(base):
    native_lookup = base.__getattribute__

    class BatchPickler(base, _WorkerPickleStream):
        def _dwd_reduce_worker_object(self, native, value):
            state = self._dwd_state
            search = value is state.search
            arrays = state.array_policy
            bindings = arrays.bindings if arrays is not None else None
            selector = bindings and id(value) in bindings.selector_carriers
            if not search:
                if arrays is None or (not selector and type(value) is not np.ndarray and type(value) is not np.memmap
                                      and type(value) is not _FoldNdarrayPayload):
                    return NotImplemented if native is _MISSING else native(value)
            kind = type(value)
            replacements = state.replacements() if search else None
            reduced = NotImplemented if native is _MISSING else native(value)
            used_array_wrapper = False
            if reduced is NotImplemented:
                missing = object()
                registered = _registered_reducer(self, kind, missing)
                if search:
                    state.select_capture_policy(registered is not missing)
                elif selector:
                    if registered is not missing:
                        # Run the selected hook once so that its own failure
                        # wins. A replacement wrapper can omit its buffers.
                        registered(value)
                        raise PicklingError(_PANDAS_SELECTOR_TRANSPORT)
                elif ((kind is _FoldNdarrayPayload and registered is missing)
                      or (kind is not _FoldNdarrayPayload and registered is arrays.native[kind])):
                    # A dynamic table still gets its one native lookup. Only
                    # adapt the queue's own reducer or DWD's payload default.
                    registered = arrays.wrappers[kind]
                    used_array_wrapper = True
                reduced = (_instance_reduction(value, state.protocol) if registered is missing
                           else registered(value))
            elif search:
                state.select_capture_policy(True)
            elif selector:
                raise PicklingError(_PANDAS_SELECTOR_TRANSPORT)
            if (not search and not selector and not used_array_wrapper and arrays.bindings
                    and arrays.bindings.selector_capture(value)):
                # The backend's override or active dispatch table handled this
                # value. Its hook ran once and any exception already propagated.
                raise PicklingError(_PANDAS_SELECTOR_TRANSPORT)
            if reduced is NotImplemented:
                # A native reducer returned an invalid reduction. Let pickle
                # reject it without invoking that reducer again.
                return None
            if search and bindings and bindings.custom_selector_state:
                # Verify the state that the hook actually returned. A hook
                # can derive new captures that must retain its own meaning.
                public = vars(value)
                saved = reduced[2] if type(reduced) is tuple and len(reduced) >= 3 else None
                if (type(saved) is not dict or len(saved) != len(public)
                        or any(type(key) is not str for key in public)
                        or any(type(key) is not str or key not in public or public[key] is not item
                               for key, item in saved.items())):
                    raise PicklingError(_PANDAS_SELECTOR_TRANSPORT)
            return _worker_search_reduction(reduced, replacements, state.graph) if search else reduced

    id_descriptor = getattr_static(base, 'persistent_id')
    if _LIVE_PERSISTENT_ID and id_descriptor is not _PERSISTENT_ID and isdatadescriptor(id_descriptor):
        # A backend descriptor may write the C slot from a reducer as well
        # as from persistent_id. Let its own setter run against the native
        # binding and then pick up its resulting slot.
        native_setattr = base.__setattr__

        def set_backend_attribute(pickler, name, value):
            if (name != 'persistent_id' or not pickler._dwd_primary_active
                    or _PERSISTENT_ID.__get__(pickler, type(pickler)) is not pickler._dwd_installed_id):
                return native_setattr(pickler, name, value)
            native = pickler._dwd_native_id
            _PERSISTENT_ID.__set__(pickler, _empty_persistent_id if native is _MISSING else native)
            try:
                return native_setattr(pickler, name, value)
            finally:
                pickler._dwd_install_id(_PERSISTENT_ID.__get__(pickler, type(pickler)))

        BatchPickler.__setattr__ = set_backend_attribute

    # Backend setup and subsequent callbacks see the backend's own hook. Only
    # the native stream receives the adapter, with the final binding.
    stock_lookup = (native_lookup is Pickler.__getattribute__
                    and getattr_static(base, '__getattr__', _MISSING) is _MISSING
                    and not isdatadescriptor(getattr_static(base, 'reducer_override', None)))
    if _LIVE_PERSISTENT_ID and stock_lookup and _stock_worker_pickler(base):
        def read_stock_reducer(pickler):
            state = vars(pickler)
            selected = state.get('_dwd_lookup_reducer', False)
            if selected:
                pickler._dwd_lookup_reducer = False
            if 'reducer_override' in state:
                native = state['reducer_override']
            else:
                parent = super(BatchPickler, pickler)
                if not selected:
                    return parent.reducer_override
                native = getattr(parent, 'reducer_override', _MISSING)
            return partial(pickler._dwd_reduce_worker_object, native) if selected else native

        def set_stock_reducer(pickler, native):
            vars(pickler)['reducer_override'] = native

        def delete_stock_reducer(pickler):
            try:
                del vars(pickler)['reducer_override']
            except KeyError:
                raise AttributeError('reducer_override') from None

        # Stock serializers use ordinary attribute lookup. Keep that fast path
        # for the many other attributes read while tracking fold references.
        BatchPickler.reducer_override = property(read_stock_reducer, set_stock_reducer, delete_stock_reducer)
    else:
        has_fallback = getattr_static(base, '__getattr__', _MISSING) is not _MISSING

        def read_custom_attribute(pickler, name):
            if name == 'persistent_id' and vars(pickler).get('_dwd_lookup_id'):
                pickler._dwd_lookup_id = False
                try:
                    native = native_lookup(pickler, name)
                except AttributeError:
                    # Let Python dispatch the backend's __getattr__ once.
                    # A lookup may have side effects even when it fails.
                    if has_fallback:
                        pickler._dwd_fallback_id = True
                    raise
                return pickler._dwd_select_id(native)
            if name != 'reducer_override' or not vars(pickler).get('_dwd_lookup_reducer'):
                return native_lookup(pickler, name)
            pickler._dwd_lookup_reducer = False
            try:
                native = native_lookup(pickler, name)
            except AttributeError:
                native = _MISSING
                fallback = getattr(super(BatchPickler, pickler), '__getattr__', None)
                if fallback is not None:
                    try:
                        native = fallback(name)
                    except AttributeError:
                        pass
            return partial(pickler._dwd_reduce_worker_object, native)

        BatchPickler.__getattribute__ = read_custom_attribute
        if not _LIVE_PERSISTENT_ID and has_fallback:
            def read_custom_fallback(pickler, name):
                selected = name == 'persistent_id' and vars(pickler).pop('_dwd_fallback_id', False)
                native = super(BatchPickler, pickler).__getattr__(name)
                if selected:
                    return pickler._dwd_select_id(native)
                return native

            BatchPickler.__getattr__ = read_custom_fallback
    return BatchPickler


_SUBMISSION_LOCK = Lock()


class _SubmissionPool:
    """Forward to joblib's pool, adapting only a registered native batch."""

    def __init__(self, pool):
        self.pool = pool
        self.batches = {}

    def __getattr__(self, name):
        return getattr(self.pool, name)

    def adapt(self, batch, outgoing):
        with _SUBMISSION_LOCK:
            worker = self.batches.get(id(batch))
            if worker is not None and worker.batch is batch:
                worker.forwarded = True
                if outgoing is batch:
                    return worker
                return _WorkerSearchBatch(outgoing, worker.backend, worker.search, worker.replacements)
        return outgoing

    def submit(self, fn, *args, **kwargs):
        return self.pool.submit(self.adapt(fn, fn), *args, **kwargs)

    def apply_async(self, func, *args, **kwargs):
        from joblib._utils import _TracebackCapturingWrapper

        batch = func.func if type(func) is _TracebackCapturingWrapper else func
        return self.pool.apply_async(self.adapt(batch, func), *args, **kwargs)


def _dispatch_native_batch(parallel, dispatch, worker):
    # A custom submitter sees the real BatchedCalls. Its configured pickler
    # then reduces that same object inside the worker-state adapter, even if
    # the reducer copies the list of jobs or overrides the batch's methods.
    attr = '_workers' if isinstance(worker.backend, LokyBackend) else '_pool'
    state = vars(worker.backend)
    key = id(worker.batch)
    worker.forwarded = False
    with _SUBMISSION_LOCK:
        pool = state[attr]
        if type(pool) is not _SubmissionPool:
            pool = state[attr] = _SubmissionPool(pool)
        pool.batches[key] = worker
    try:
        result = dispatch(worker.batch)
        if not worker.forwarded and not parallel._aborting:
            raise RuntimeError('The custom backend did not forward the batch to its joblib pool. '
                               'Build the folds and scorer with owning index arrays.')
        return result
    finally:
        with _SUBMISSION_LOCK:
            del pool.batches[key]
            if not pool.batches and state.get(attr) is pool:
                state[attr] = pool.pool


class _WorkerSearchBatch:
    """Adapt one search's outgoing reduction without modifying the search."""

    def __init__(self, batch, backend, search, replacements, data=None):
        self.batch = batch
        self.backend = backend
        self.search = search
        self.replacements = replacements
        self.data = data
        self.fold_arrays = None
        self.bind_originals = False
        self.deferred_replacements = None

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, index):
        return self.batch[index]

    def __call__(self):
        return self.batch()

    def __reduce_ex__(self, protocol):
        if self.data is not None:
            # Let the outer queue choose the serializer for external task
            # items. Its private marker also carries the actual protocol.
            if self.fold_arrays and self.fold_arrays.selector_buffers:
                raise PicklingError(_PANDAS_SELECTOR_TRANSPORT)
            return _as_is, (_ExternalSerialization(self, protocol),)
        if self.replacements is None and self.deferred_replacements is None and not _native_fold_serializer(self.backend):
            # The active loky pickler can change after split preparation.
            if self.fold_arrays and self.fold_arrays.selector_buffers:
                raise PicklingError(_PANDAS_SELECTOR_TRANSPORT)
            return _as_is, (self.batch,)
        return self.encode(protocol)

    def _array_reducers(self, native, pickler):
        fold_arrays = self.fold_arrays
        bind_originals = self.bind_originals
        owner = ref(pickler)

        # The outer caller's arrays retain their requested mapping mode.
        # Only a writable, detached fold temporary can use a private
        # mapping instead of eagerly copying that temporary in every task.
        mapping_disabled = getattr(native[np.ndarray], '_mmap_mode', _MISSING) is None
        private_transports = {}

        def reduce_fold(payload):
            restore, args = payload.__reduce__()
            transport, writeable = args
            if fold_arrays is not None and (writeable or mapping_disabled):
                private_transports[id(transport)] = transport
            return restore, args

        def reduce_array(value):
            if fold_arrays is None:
                return native[type(value)](value)
            if private_transports.get(id(value)) is value:
                if mapping_disabled:
                    # A finite size threshold can otherwise select joblib's
                    # temporary memmap loader even with mmap_mode=None.
                    return loads, (dumps(value, protocol=HIGHEST_PROTOCOL),)
                reduced = native[type(value)](value)
                if _JOBLIB_TEMP_LOADER is not None and len(reduced) == 2 and reduced[0] is _JOBLIB_TEMP_LOADER:
                    args = reduced[1]
                    if len(args) == 3 and args[1] in ('r', 'r+', 'w+'):
                        # Keep joblib's own file and resource-tracker reference.
                        return reduced[0], (args[0], 'c', args[2])
                return reduced
            payload = fold_arrays.get(id(value))
            if payload is None or value is payload.source:
                # A backend can install a search reducer in dump, including
                # through a mapping with a dynamic lookup. Preserve raw
                # captures in opaque state without probing that mapping.
                # A subclass can supply its own pickle state while sending
                # selected pandas buffers through joblib's native reducers.
                # Continue to honor its ordinary captures and its state hook.
                if (not bind_originals and not fold_arrays.selector_capture(value)
                        or payload is None and (value.base is None or value.dtype.hasobject)):
                    return native[type(value)](value)
                pickler = owner()
                state = vars(pickler).get('_dwd_state')
                if state.capture_policy(pickler):
                    if fold_arrays.selector_capture(value):
                        raise PicklingError(_PANDAS_SELECTOR_TRANSPORT)
                    # Joblib's actual recipe contains the mode and order it
                    # just read. Use it for both early and selected raw views.
                    reduced = native[type(value)](value)
                    if state.applied_raw is None and (payload is not None or fold_arrays.raw_overlap(value, reduced)):
                        state.applied_raw = True
                    return reduced
                if payload is None:
                    backing = fold_arrays.overlapping_map(value)
                    if backing is None:
                        return native[type(value)](value)
                    # A subclass can store an alias and expose its canonical
                    # mode through a getter. Prefer the mode the folds used.
                    mode = fold_arrays.modes.get(id(backing), _MISSING)
                    if mode is _MISSING:
                        mode = _map_mode_text(getattr_static(backing, 'mode', None))
                    if mode in ('r', 'r+', 'w+'):
                        fortran = (np.ndarray.ndim.__get__(backing) > 1
                                   and np.ndarray.flags.__get__(backing).f_contiguous)
                        if not _unsafe_mapped_layout(value, None, mode=mode, fortran=fortran):
                            reduced = native[type(value)](value)
                            if fold_arrays.reopen_needs_snapshot(value, reduced) is not False:
                                raise PicklingError(_UNVERIFIED_MAP_LAYOUT)
                            return reduced
                    if not _unsafe_mapped_layout(value, backing):
                        raise PicklingError(_UNVERIFIED_MAP_LAYOUT)
                    payload = fold_arrays.detached_view(value)
                state.applied_raw = False
            return _as_is, (payload,)

        def release():
            # A backend can save or wrap these reducers in a replacement table.
            nonlocal fold_arrays
            fold_arrays = None
            private_transports.clear()

        return {np.ndarray: reduce_array, np.memmap: reduce_array, _FoldNdarrayPayload: reduce_fold}, release

    def encode(self, protocol, forking=False):
        # The feeder supplies its protocol. Reuse the queue's exact reducers,
        # including joblib's memmapping policy and its temporary-file manager.
        if forking:
            from multiprocessing.reduction import ForkingPickler

            base = ForkingPickler
            reducers = None
        elif isinstance(self.backend, MultiprocessingBackend):
            from joblib.pool import CustomizablePickler

            base = CustomizablePickler
            pool = getattr(self.backend, '_pool', None)
            reducers = getattr(getattr(pool, '_inqueue', None), '_reducers', None)
        else:
            from joblib.externals.loky.backend.reduction import get_loky_pickler

            base = get_loky_pickler()
            workers = getattr(self.backend, '_workers', None) if isinstance(self.backend, LokyBackend) else None
            reducers = getattr(getattr(workers, '_call_queue', None), '_reducers', None)
        stream = BytesIO()
        # Joblib can reduce an owning array to a large immutable pickle blob.
        # Hand those bytes to the original queue once instead of copying them
        # into this batch's metadata and then into the queue a second time.
        separate_bytes = self.data is None and (type(self.backend) is LokyBackend
                          or self.fold_arrays and isinstance(self.backend, LokyBackend))
        separate_values = separate_bytes or self.data is not None
        native_state = self.replacements is None
        adapted = separate_values or not native_state or self.deferred_replacements is not None
        cls = _worker_pickler_type(base) if adapted else base
        native_arrays = None
        if self.fold_arrays and reducers is not None and _native_fold_serializer(self.backend):
            # Joblib copies the queue template into each new pickler. Its
            # constructor or a later saving hook can change the template.
            native_arrays = {kind: reducers[kind] for kind in (np.ndarray, np.memmap)}
        elif self.fold_arrays and self.fold_arrays.selector_buffers:
            raise PicklingError(_PANDAS_SELECTOR_TRANSPORT)
        pickler = cls(stream, protocol) if forking else cls(stream, reducers=reducers, protocol=protocol)
        release = None
        array_policy = None
        state = None
        try:
            if native_arrays is not None:
                array_reducers, release = self._array_reducers(native_arrays, pickler)
                array_policy = _WorkerArrayPolicy(native_arrays, array_reducers, self.fold_arrays)
            if not adapted:
                if self.fold_arrays and self.fold_arrays.selector_buffers:
                    raise PicklingError(_PANDAS_SELECTOR_TRANSPORT)
                pickler.dump(self.batch)
                return loads, (stream.getvalue(),)

            state_type = _PayloadPickleState if separate_values else _WorkerPickleState
            state = state_type(self, pickler, stream, protocol)
            state.array_policy = array_policy
            if separate_values:
                state.external = self.data or {}
                state.separate_bytes = separate_bytes
                state.positions = {}
            vars(pickler)['_dwd_state'] = state
            pickler._dwd_local_id = state.external_id if native_state else state.fold_id
            pickler.dump(self.batch)
            # An ordinary unpickler consumes the first stream. A later dump or
            # a backend retry cannot certify bytes it wrote by another route.
            primary = state.primary
            if primary is None:
                raise PicklingError(_OWNING_FOLD_MESSAGE)
            serialized = primary[1][0]
            delivered = stream.getvalue()
            if delivered is not serialized and not delivered.startswith(serialized):
                raise PicklingError(_OWNING_FOLD_MESSAGE)
            return primary
        finally:
            if release is not None:
                release()
            if state is not None:
                fields = vars(pickler)
                fields.pop('_dwd_state', None)
                fields.pop('_dwd_local_id', None)
                unbound = fields.pop('_dwd_unbound_id', None)
                if unbound is not None and _PERSISTENT_ID.__get__(pickler, type(pickler)) is unbound:
                    # Restore CPython's non-owning class-method binding after
                    # the outer backend has completed its own post-dump work.
                    _PERSISTENT_ID.__set__(pickler, partial(_call_pickler_method, unbound.__func__, ref(pickler)))
                if state.graph is not None:
                    # The native memo can hold a private scope or replay view.
                    # Detach the graph's own references even after a veto.
                    state.graph.replacements.clear()
                    state.graph.scopes.clear()
                    state.graph.views.clear()
                    state.graph.open_views.clear()


class _ExternalSerialization:
    def __init__(self, batch, protocol):
        self.batch = batch
        self.protocol = protocol

    def __reduce__(self):
        return self.batch.encode(self.protocol)


def _encode_forking_items(marker):
    return marker.batch.encode(marker.protocol, forking=True)


@lru_cache(maxsize=1)
def _register_external_marker():
    # A custom backend may hand its items to multiprocessing, which maintains
    # a separate native dispatch table. Register only our transport marker.
    from multiprocessing.reduction import ForkingPickler

    ForkingPickler.register(_ExternalSerialization, _encode_forking_items)


def _count_container_writes(serialized, observations):
    # A restore hook can change a partially populated list or dictionary.
    # Count the entries committed by pickle at each observation point from
    # the original stream, preserving the native order without rerunning hooks.
    from pickletools import genops, markobject, stackslice

    counts = dict.fromkeys((receiver for _, receiver in observations.values()), 0)
    points = {}
    stack = []
    marks = []
    memoized = set()
    for opcode, arg, _ in genops(serialized):
        name = opcode.name
        before, after = opcode.stack_before, opcode.stack_after
        if name == 'MARK':
            marks.append(len(stack))
            stack.append(None)
        elif name in ('MEMOIZE', 'PUT', 'BINPUT', 'LONG_BINPUT'):
            index = len(memoized) if name == 'MEMOIZE' else arg
            memoized.add(index)
            if index in counts:
                stack[-1] = index
            if index in observations:
                point, receiver = observations[index]
                points[point] = counts[receiver]
        elif name in ('GET', 'BINGET', 'LONG_BINGET'):
            stack.append(arg if arg in counts else None)
        elif name == 'DUP':
            stack.append(stack[-1])
        elif stackslice in before:
            mark = marks.pop()
            if name in ('SETITEMS', 'APPENDS', 'ADDITEMS'):
                receiver = stack[mark - 1]
                if receiver is not None and name != 'ADDITEMS':
                    items = len(stack) - mark - 1
                    counts[receiver] += items // 2 if name == 'SETITEMS' else items
                del stack[mark:]
            else:
                del stack[mark - before.index(markobject):]
                stack.extend([None] * len(after))
        elif name in ('SETITEM', 'APPEND', 'BUILD'):
            if name != 'BUILD' and stack[-len(before)] is not None:
                counts[stack[-len(before)]] += 1
            del stack[1 - len(before):]
        else:
            if name == 'POP' and marks and len(stack) == marks[-1] + 1:
                marks.pop()
            del stack[len(stack) - len(before):]
            stack.extend([None] * len(after))
    return points


def _load_worker_items(serialized, bindings, id_prefix, changed=(), observations=None):
    writes = _count_container_writes(serialized, observations) if observations else {}
    reader = Unpickler(BytesIO(serialized))

    def persistent_load(token):
        if type(token) is _ScopeReference:
            return token.value
        if type(token) is bytes and token.startswith(id_prefix):
            if token == id_prefix:
                return changed, writes
            if len(token) == len(id_prefix) + 8:
                index = int.from_bytes(token[-8:], 'big')
                if index < len(bindings):
                    return bindings[index]
        raise UnpicklingError('Custom persistent IDs require a matching backend unpickler.')

    reader.persistent_load = persistent_load
    return reader.load()


class _WorkerBatchItem:
    def __init__(self, items, index):
        self.items = items
        self.index = index

    def __call__(self, *visible_args, **visible_kwargs):
        function, args, kwargs, positions = self.items[self.index]
        args = list(args)
        for position in positions:
            args[position] = visible_args[position]
        return function(*args, **kwargs, **visible_kwargs)


def _prepare_external_batch(batch, backend, search, replacements):
    _register_external_marker()
    data = {}
    packed = []
    items = _WorkerSearchBatch(packed, backend, search, replacements, data)
    visible = []
    for index, (function, args, kwargs) in enumerate(batch.items):
        # External backends can substitute X/y and any direct array argument
        # (including train/test) by position or keyword before serialization.
        positions = [i for i, value in enumerate(args) if i in (1, 2) or isinstance(value, np.ndarray)]
        visible_args = [None] * len(args)
        packed_args = list(args)
        for position in positions:
            visible_args[position] = args[position]
            packed_args[position] = None
        visible_kwargs = {key: value for key, value in kwargs.items() if isinstance(value, np.ndarray)}
        packed_kwargs = {key: value for key, value in kwargs.items() if key not in visible_kwargs}
        for value in (*visible_args, *visible_kwargs.values()):
            if value is not None and type(value) not in (str, bytes, bool, int, float, complex):
                data[id(value)] = value
        packed.append((function, packed_args, packed_kwargs, positions))
        visible.append((_WorkerBatchItem(items, index), tuple(visible_args), visible_kwargs))
    batch.items = visible


@lru_cache(maxsize=4)
def _fit_with_local_evaluator(fit_code):
    # A descriptor can execute user code on every lookup or assignment. Leave
    # sklearn's method dispatch intact and adapt its local evaluator instead.
    # Only the nested function constant changes in this private copy of fit.
    evaluators = [item for item in fit_code.co_consts
                  if type(item) is CodeType and item.co_name == 'evaluate_candidates']
    if len(evaluators) != 1:
        raise RuntimeError('Unrecognized sklearn candidate evaluator for process-fold preparation.')
    native, = evaluators
    params = native.co_varnames[:native.co_argcount]
    if (params not in (('candidate_params', 'cv', 'more_results'),
                       ('candidate_params', 'cv', 'more_results', 'callback_ctx'))
            or native.co_kwonlyargcount or native.co_posonlyargcount
            or native.co_flags & (CO_VARARGS | CO_VARKEYWORDS)):
        raise RuntimeError('Unrecognized sklearn candidate arguments for process-fold preparation.')
    # Let Python bind the original defaults and capture the original cells in
    # exactly their native order. The generated names all come from sklearn.
    arguments = ', '.join(params)
    captures = ', '.join(native.co_freevars)
    adapter = compile(
        f'def bind({captures}):\n'
        f'    def evaluate_candidates({arguments}):\n'
        f'        return _dwd_evaluate((lambda: ({captures})).__closure__, {arguments})\n',
        '<dwd candidate evaluator>', 'exec')
    bind, = (item for item in adapter.co_consts if type(item) is CodeType)
    wrapper, = (item for item in bind.co_consts if type(item) is CodeType)
    if wrapper.co_freevars != native.co_freevars:
        raise RuntimeError('Incompatible sklearn candidate closure for process-fold preparation.')
    wrapper = wrapper.replace(co_qualname=native.co_qualname)
    constants = tuple(wrapper if item is native else item for item in fit_code.co_consts)
    return fit_code.replace(co_consts=constants), native


def _fit_process_search(search, X, y, fit_params, stock_fit, *, defer_frame=False):
    # BaseSearchCV calls _run_search with an evaluator accepting a cv override.
    # Keep both constructor parameters on the real search at every callback.
    input_cv = _MISSING
    fit_code, native_code = _fit_with_local_evaluator(stock_fit.__code__)
    namespace = stock_fit.__globals__
    saved_getstate = _MISSING
    native_getstate = _MISSING
    active_splits = None
    saved_dispatch = None
    bind_originals = False

    def fold_replacements(splits):
        cv = search.cv
        make_replacements = {}
        if splits is not None and splits.cv is not _MISSING and splits.rows is not None:
            if splits.make_checked is not None:
                make_replacements[id(splits.checked)] = splits.make_checked
            if splits.needs_worker_cv(cv):
                make_replacements[id(cv)] = _WorkerCVFactory(splits)
        elif isinstance(cv, Iterator):
            make_replacements[id(cv)] = tuple
        return make_replacements

    def worker_state():
        # Invoke the native bound hook on the actual object. It may use slots,
        # external identity checks, or return a self reference for the pickler.
        make_replacements = fold_replacements(active_splits)
        # A backend can wrap this hook while a batch is being serialized.
        # Restore the original attribute in the outgoing state in either case.
        active_getstate = search.__getstate__
        hook_ids = {id(worker_state)}
        if active_getstate is not saved_getstate and callable(active_getstate):
            hook_ids.add(id(active_getstate))
        return _worker_pickle_state(native_getstate(), make_replacements, saved_getstate, hook_ids)

    def install_worker_state(parallel, prepare_state=True):
        nonlocal saved_getstate, native_getstate, saved_dispatch, bind_originals
        if native_getstate is _MISSING and saved_dispatch is None:
            bind_originals = _standard_search_reduction(search)
        # A custom saving hook can derive metadata from the parent's fields.
        # Let it run on the untouched search and adapt its reduction in the batch.
        batch_state = (prepare_state and native_getstate is _MISSING
                       and (not bind_originals or _custom_search_reducer(parallel._backend, type(search))))
        deferred = (native_getstate is _MISSING and not batch_state and active_splits is not None
                    and (active_splits.fold_arrays or isinstance(search.cv, Iterator))
                    and _native_fold_serializer(parallel._backend))
        if saved_dispatch is None and (batch_state or deferred or (active_splits is not None and active_splits.fold_arrays)):
            # Only this evaluation's Parallel instance sees the envelope.
            # The search and its pickle descriptors remain native at callbacks.
            native_dispatch = parallel._dispatch
            saved = vars(parallel).get('_dispatch', _MISSING)

            def dispatch(batch):
                splits = active_splits
                backend = parallel._backend

                def replacements():
                    return fold_replacements(splits)

                native_backend = _native_fold_serializer(backend)
                if splits is not None and splits.fold_arrays.selector_buffers and not native_backend:
                    raise PicklingError(_PANDAS_SELECTOR_TRANSPORT)
                arrays = splits.fold_arrays if splits is not None and native_backend else None
                if arrays or ((batch_state or deferred) and native_backend):
                    batch = _WorkerSearchBatch(batch, backend, search, replacements if batch_state else None)
                    batch.fold_arrays = arrays
                    batch.bind_originals = bind_originals
                    if deferred:
                        batch.deferred_replacements = replacements if prepare_state else dict
                elif batch_state or deferred:
                    queue = _joblib_queue(backend)
                    if type(batch) is BatchedCalls and queue is not None and queue[1] is not None:
                        worker = _WorkerSearchBatch(batch, backend, search, replacements)
                        return _dispatch_native_batch(parallel, native_dispatch, worker)
                    _prepare_external_batch(batch, backend, search, replacements)
                return native_dispatch(batch)

            parallel._dispatch = dispatch
            saved_dispatch = (parallel, saved)
        if prepare_state and not (batch_state or deferred) and native_getstate is _MISSING:
            saved_getstate = _stored_search_method(search, '__getstate__')
            native_getstate = search.__getstate__
            search.__getstate__ = worker_state

    def evaluate(cells, candidate_params, cv, *options):
        nonlocal active_splits, native_getstate, saved_dispatch
        evaluate_candidates = FunctionType(native_code, namespace, closure=cells)
        # Read the actual executor and checked splitter from sklearn's closure.
        fit_cells = dict(zip(native_code.co_freevars, cells, strict=True))
        parallel = fit_cells['parallel'].cell_contents
        process = _uses_worker_processes(parallel)

        if defer_frame:
            # Snapshot at the first task of this evaluation, after the splitter
            # sees the caller's frame. The parent also refits on that frame.
            prepared = _MISSING

            def delayed(function):
                task = namespace['delayed'](function)
                # A frame that needs no copy can use the native task directly
                # after the first binding. Keep the evaluator's globals out of
                # this closure so a prepared frame is released with the call.
                if function is not namespace['_fit_and_score'] or prepared is X:
                    return task

                def bind(estimator, X, *args, **kwargs):
                    nonlocal prepared
                    if prepared is _MISSING:
                        prepared = _parallel_search_features(X, parallel, process=process)
                    return task(estimator, prepared, *args, **kwargs)

                return bind

            evaluate_candidates = FunctionType(native_code, dict(namespace, delayed=delayed), closure=cells)

        if not process:
            return evaluate_candidates(candidate_params, cv, *options)

        # sklearn selects "cv or checked" before collecting candidates.
        # A subclass's truth test can mutate the checked splitter or raise.
        override = cv if cv else None
        # sklearn 1.6/1.7 keep the already-checked splitter local to fit.
        # Newer versions make it available directly on the search.
        if 'cv_orig' in fit_cells:
            checked = fit_cells['cv_orig'].cell_contents
        else:
            checked = search._checked_cv_orig
        # A setup callback may have changed the public cv since fit entry.
        public_cv = search.cv
        _reject_mapped_objects(public_cv, parallel)
        native_checked = checked is input_cv or checked is public_cv
        if (override is not None and override is not checked) or type(checked) is not _ITERABLE_CV:
            # An overriding splitter owns its generated indices. Separate
            # explicit folds can still travel with a serialized search.
            checked_pairs = checked.cv if type(checked) is _ITERABLE_CV else ()
            _reject_mapped_fold_contents(checked_pairs, parallel)
            public = public_cv.cv if type(public_cv) is _ITERABLE_CV else public_cv
            if _explicit_folds(public):
                _reject_mapped_fold_contents(public, parallel, checked_pairs)
            if override is None or override is checked:
                cv = None
            elif override is public_cv and type(override) is _ITERABLE_CV:
                active_splits = cv = _ProcessSplits(override, parallel, public_cv, search,
                                                    install_worker_state, native_checked=True)
            elif type(override) is _ITERABLE_CV:
                active_splits = cv = _ProcessSplits(override, parallel, search=search,
                                                    install_worker_state=install_worker_state)
            else:
                cv = override if defer_frame else _SelectedSplits(override)
        else:
            active_splits = cv = _ProcessSplits(checked, parallel, public_cv, search,
                                                install_worker_state, native_checked)
        if defer_frame:
            # Split counting can select this process executor, and collecting
            # candidates can change the frame. Check at the call to split.
            cv = _SelectedSplits(checked if cv is None else cv, parallel)
        # Reusable unused folds travel with the original search. Its
        # pickle hook may depend on the instance's identity or slots.
        if active_splits is None and not isinstance(public_cv, Iterator):
            return evaluate_candidates(candidate_params, cv, *options)
        try:
            if isinstance(public_cv, Iterator):
                install_worker_state(parallel)
            return evaluate_candidates(candidate_params, cv, *options)
        finally:
            if saved_dispatch is not None:
                executor, saved = saved_dispatch
                _restore_search_method(executor, '_dispatch', saved)
                saved_dispatch = None
            if native_getstate is not _MISSING:
                _restore_search_method(search, '__getstate__', saved_getstate)
                native_getstate = _MISSING
            if active_splits is not None:
                active_splits._worker_cv_error = None
                active_splits = None

    def checked_cv(cv, *args, **kwargs):
        nonlocal input_cv
        input_cv = cv
        jobs = search.n_jobs
        _reject_mapped_objects(cv, jobs)
        if defer_frame and _uses_worker_processes(jobs):
            # After setup, reject mapped object pointers before a splitter
            # can inspect X. Numeric snapshots wait for the worker tasks.
            _parallel_search_features(X, jobs, check_only=True)
        return namespace['check_cv'](cv, *args, **kwargs)

    bindings = dict(namespace, check_cv=checked_cv, _dwd_evaluate=evaluate)
    fit = FunctionType(fit_code, bindings, stock_fit.__name__, stock_fit.__defaults__, stock_fit.__closure__)
    fit.__kwdefaults__ = stock_fit.__kwdefaults__
    return _fit_context(prefer_skip_nested_validation=False)(fit)(search, X, y, **fit_params)


def fit_shared_fold_search(search, X, y, **fit_params):
    """Fit a GridSearchCV with private reusable KernMD fold snapshots when safe.

    This is an explicit opt-in helper. Calling search.fit directly, or using
    dwd.cv.run_cv, does not activate it. The same search object is returned;
    sklearn performs all splitting, scoring, ranking and final refitting.

    Parameters
    ----------
    search : sklearn.model_selection.GridSearchCV
        Search to fit. Sharing requires an exact GridSearchCV containing an
        exact KernMD, n_jobs=1, Boolean refit and scoring=None or 'accuracy'.
        Search subclasses use their ordinary fit method.
    X : array-like
        Training data. Sharing requires a nonempty, finite, native float64
        ndarray with two dimensions. C, Fortran, strided and read-only arrays
        are accepted; their flags never prove private ownership.
    y : array-like
        Labels. Sharing requires a finite, one-dimensional numeric ndarray.
        A sparse pandas uint64 target that rounds class IDs during search
        conversion or row selection requires an exact owning target before fit.
        For a multi-output DataFrame this includes the common dtype used when
        pandas converts all target columns together.
        Keep its data and fill representation stable from entry through refit.
    **fit_params : dict
        Arguments forwarded to ordinary search.fit. Supplying any disables
        sharing, as do metadata routing and callbacks.

    Returns
    -------
    search : GridSearchCV
        The supplied object, fitted in place. A successful refit on the shared
        route produces a normal KernMD with independent, writable fitted
        feature storage and restores the original constructor estimator.
        On the fallback route, GridSearchCV refits the selected estimator
        when refitting is enabled.

    Notes
    -----
    Sharing also requires explicit train/test index pairs in a list or tuple,
    ordinary dict/list parameter grids with list/tuple values, and only named
    'linear' or 'rbf' kernels. The base estimator and every candidate must use
    naive_bayes=False and either no kernel keywords or a positive, finite scalar
    RBF gamma. Callable/precomputed kernels, arbitrary kernel configurations,
    custom scorers/refit callables, other input types and unsupported options
    take ordinary sklearn fit. With the stock GridSearchCV.fit implementation,
    the helper prepares NumPy X, y and directly supplied array-valued fit
    parameters when multiple process workers are configured on entry. For
    arrays of exact type numpy.ndarray or numpy.memmap whose dtype contains
    no Python objects, it copies copy-on-write inputs to preserve private
    edits. It also copies mapped views that reverse a non-singleton axis or whose
    contiguous memory order differs from the backing map. If a setup callback
    will switch a serial search to process workers, make owning copies of
    those affected inputs before calling the helper.
    For an exact pandas DataFrame used as X, the helper checks the executor
    selected for candidate evaluation. With stock fit, it preserves integer
    values when selecting sparse uint64 feature folds. For an overridden fit
    or a DataFrame subclass, prepare exact features before calling the helper.
    The mapping checks apply to NumPy blocks,
    native datetime, timedelta and period buffers, nullable numeric and Boolean
    data and masks, categorical codes and categories, interval endpoints, and
    pandas sparse values and positions. Ordinary, categorical, interval,
    datetime-like and multilevel row and column indexes are checked. File-backed
    object buffers in these representations are rejected. Other extension
    buffers, custom index types and attrs must be suitable for native transport.
    The parent splitter and final refit receive the supplied frame. Each
    process evaluation prepares the worker frame when its first task is built.
    Keep training data stable during dispatch of that evaluation's tasks.
    Explicit folds can be an iterable of train/test pairs or a stacked numeric
    ndarray. The helper prepares supported mapped fold indices using the
    executor sklearn selects for candidate evaluation, including after a
    setup callback changes the worker count.
    Selected exact pandas Index and Series objects retain their pandas type,
    labels and metadata. On the standard joblib process path, their supported
    NumPy-backed values and Series index buffers receive the same mapping
    checks as DataFrame storage. Repeated references to a selector and
    overlapping NumPy views of buffers requiring a snapshot use the parent values.
    If a custom serializer prevents this protection, build the selectors
    and their scorer references from owning arrays before fitting.
    This includes explicit folds passed through sklearn's check_cv, either
    as the constructor cv or as a splitter selected for an evaluation. A
    scorer reading a separately retained alternate splitter should use
    owning indices in that splitter. When a different splitter supplies all
    the fitting rows, use owning indices for unrelated public folds read by
    the scorer as well. If a callback or subclass replaces a
    public fold record after sklearn checks the folds, workers use the
    checked rows for fitting and see the current public native records in cv.
    Keep object-valued inputs in ordinary in-memory arrays. When process
    workers are configured on entry, the helper rejects file-backed object
    arrays supplied directly as X, y or fit parameters. It also checks
    supported explicit folds for file-backed object arrays when evaluating
    candidates in processes. If a subclass selects a different splitter, workers
    can also receive the original folds. The helper leaves unused custom fold
    iterators alone. Keep all contents of those records, including train and
    test indices, free of file-backed object arrays. Plain (train, test) tuples
    of numeric index arrays are supported.
    Splitter objects and objects with metadata routing keep their own
    behavior. The parent search retains its public cv and refit parameters
    throughout fitting.

    A custom __getstate__ runs on the parent search with its original fields.
    For NumPy folds it may omit the folds. For affected pandas selectors,
    use owning inputs if the hook changes, adds or omits fields, or if the
    subclass supplies a custom restore method. The worker batch adapts its
    outgoing reduction.
    For worker serialization, references to cv and _checked_cv_orig can be
    replaced in plain dictionary values, lists and tuples, including nested
    or shared containers. The hook is responsible for references inside
    custom objects or dictionary keys. If a fold replacement is needed and
    the hook returns an opaque object as its entire state, use plain state
    or fit the search serially.

    Preserving custom fold metadata can invoke copy hooks on the fitting
    thread even when no task serializes the search. Hook side effects occur
    when the hook runs. If this worker-only preparation raises an Exception,
    it surfaces when the serialized search state needs the prepared public
    folds. Interrupts propagate immediately.

    X is snapshotted into immutable bytes and ordered row subsets are cached.
    Each temporary fold estimator gets independent shape metadata. Final refit
    still uses canonical KernMD.fit, including its ownership copy. No private
    store, token or estimator type is retained by this helper on a successfully
    returned search; serialization uses public estimator types. Kernel keyword
    copying and numerical formulas are unchanged.

    search.shared_fold_info contains data-only diagnostics for the latest
    attempt: route ('shared' or 'stock'), reason for a stock fallback, and
    completed (true only on successful return). Shared attempts also report
    cache counts and logical payload byte counts when available. cached_bytes
    sums subset lengths, including views sharing the immutable root; it does not
    count distinct allocations. A surviving borrowed subset view can retain the
    whole private root. These are not retained-memory or process-peak measures.
    This metadata has no trailing underscore so a failed
    attempt cannot itself mark an unfitted search as fitted. Each invocation
    replaces the diagnostics.

    The immutable copies can increase peak memory. Ordinary complete k-fold
    splitting can retain approximately k+1 feature matrices during search.
    Fewer repeated candidates may not amortize preparation. C-order snapshots
    can change memory layout and floating-point reduction order; this helper
    does not provide a general no-slowdown guarantee.

    On a shared-path error or interruption, the original constructor estimator
    and prior scorer identity (or absence) are restored. A private partial best
    estimator is converted to a normal KernMD without another fit. Other partial
    or previous learned attributes follow sklearn behavior; this is not a
    transactional rollback. If that conversion itself fails, the private partial
    estimator is discarded and the original exception is preserved. Unsupported
    stock-path failures keep ordinary sklearn behavior. Concurrent use of the
    same search or mutation of private helper state is unsupported.

    Raises
    ------
    TypeError
        If search is not a GridSearchCV instance.
    Exception
        Validation and fit exceptions propagate without changing their identity.

    Examples
    --------
    >>> from joblib import parallel_config
    >>> from sklearn.model_selection import GridSearchCV, StratifiedKFold
    >>> from threadpoolctl import threadpool_limits
    >>> from dwd.kern_md import KernMD
    >>> folds = list(StratifiedKFold(3).split(X, y))  # doctest: +SKIP
    >>> search = GridSearchCV(KernMD(kernel='rbf'),
    ...     {'kernel_kws': [{'gamma': 0.01}, {'gamma': 0.1}]},
    ...     cv=folds, n_jobs=1, scoring='accuracy')  # doctest: +SKIP
    >>> with parallel_config(backend='sequential', n_jobs=1), threadpool_limits(limits=1):  # doctest: +SKIP
    ...     fitted = fit_shared_fold_search(search, X, y)
    """
    if not isinstance(search, GridSearchCV):
        raise TypeError('search must be a GridSearchCV instance')
    info = {'route': 'stock', 'reason': None, 'completed': False}
    search.shared_fold_info = info
    _check_sparse_search_target(y)
    reason = _eligibility(search, X, y, fit_params)
    if reason is not None:
        info['reason'] = reason
        has_callbacks = (hasattr(GridSearchCV, 'set_callbacks')
                         and (getattr(search, '_skl_callbacks', None)
                              or hasattr(search, '_parent_callback_ctx')))
        # Instrumentation can wrap either class and can be removed between fits.
        # Only specialize sklearn's own inherited fit body on this call.
        fit = search.fit
        stock_body = None
        if (type(fit) is MethodType and fit.__self__ is search and type(fit.__func__) is FunctionType
                and fit.__func__ is getattr_static(_search.BaseSearchCV, 'fit')
                and fit.__func__.__code__ is _FIT_WRAPPER_CODE
                and _FIT_WRAPPER_CODE.co_freevars == ('fit_method', 'prefer_skip_nested_validation')
                and fit.__func__.__globals__ is _fit_context.__globals__):
            body = getattr(fit.__func__, '__wrapped__', None)
            fit_method, validation_policy = fit.__func__.__closure__
            if (type(body) is FunctionType and body.__globals__ is vars(_search)
                    and body.__code__.co_qualname == 'BaseSearchCV.fit'
                    and fit_method.cell_contents is body and validation_policy.cell_contents is False):
                stock_body = body
        jobs = search.n_jobs if stock_body is not None else _MISSING
        defer_frame = type(X) is getattr(modules.get('pandas'), 'DataFrame', None)
        if (stock_body is not None and (jobs is None or isinstance(jobs, Integral))
                and (jobs != 1 or has_callbacks or defer_frame)):
            # Read stored data without binding a possibly stateful cv descriptor.
            # The fit-local check_cv sees the actual value after callback setup.
            entry_cv = getattr_static(search, 'cv', _MISSING)
            process = _uses_worker_processes(jobs)
            if process:
                _reject_mapped_objects(entry_cv, jobs)
                if not defer_frame:
                    X = _parallel_search_input(X, jobs)
                elif has_callbacks:
                    _parallel_search_features(X, jobs, check_only=True)
                y = _parallel_search_input(y, jobs)
                fit_params = {key: _parallel_search_input(value, jobs) for key, value in fit_params.items()}
            # Subclass hooks and callbacks can select another method or splitter
            # during fit. Keep sklearn's native lookups on the real search.
            run_method = getattr_static(search, '_run_search', None)
            if (has_callbacks or defer_frame or type(search) is not GridSearchCV
                    or type(entry_cv) is _ITERABLE_CV
                    or getattr_static(GridSearchCV, '__getattribute__') is not object.__getattribute__
                    or getattr_static(type(entry_cv), '__get__', None) is not None
                    or _explicit_folds(entry_cv) or type(run_method) is not FunctionType
                    or run_method.__globals__ is not vars(_search)
                    or run_method.__code__.co_qualname != 'GridSearchCV._run_search'):
                result = _fit_process_search(search, X, y, fit_params, stock_body, defer_frame=defer_frame)
                info['completed'] = True
                return result
        result = fit(X, y, **fit_params)
        info['completed'] = True
        return result

    info['route'] = 'shared'
    store = _FoldDataset(X)
    original = search.estimator
    previous_scorer = vars(search).get('scorer_', _MISSING)
    try:
        search.estimator = _SharedFoldKernMD(**original.get_params(deep=False))
        result = search.fit(store, y)
        _publicize_best(search)
        if search.scoring is None:
            # Passthrough scorers retain their constructor estimator.
            search.scorer_ = check_scoring(original, scoring=None)
        info.update(store.info())
        info['completed'] = True
        return result
    except BaseException as error:
        info['completed'] = False
        # Supported searches are exact GridSearchCV instances with no callbacks.
        vars(search)['estimator'] = original
        _restore_scorer(search, previous_scorer)
        try:
            _publicize_best(search)
            info.update(store.info())
        except BaseException as cleanup_error:
            if type(vars(search).get('best_estimator_')) is _SharedFoldKernMD:
                vars(search).pop('best_estimator_', None)
            _note_cleanup_error(error, cleanup_error)
        raise
    finally:
        vars(search)['estimator'] = original
        # Break store -> token -> store cycles after recording active statistics.
        # Returned estimators do not retain tokens; their ndarray bases remain valid.
        store._tokens.clear()
