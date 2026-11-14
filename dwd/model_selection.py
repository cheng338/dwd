"""Explicit search helpers for DWD estimators.

fit_shared_fold_search shares private, immutable fold snapshots between
eligible serial KernMD candidates. KernMD.fit and sklearn search algorithms
remain unchanged.
"""

from copy import deepcopy

import numpy as np
from sklearn import get_config
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import check_scoring

from ._fit_state import fit_with_cleanup
from ._validation import check_real_X_y
from .kern_md import KernMD, kern_md


__all__ = ['fit_shared_fold_search']
_MISSING = object()


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
    **fit_params : dict
        Arguments forwarded unchanged to ordinary search.fit. Supplying any
        disables sharing, as do metadata routing and callbacks.

    Returns
    -------
    search : GridSearchCV
        The supplied object, fitted in place. With refitting enabled, its
        best_estimator_ is a normal KernMD with independent, writable fitted
        feature storage. Its original constructor estimator is restored.

    Notes
    -----
    Sharing also requires explicit train/test index pairs in a list or tuple,
    ordinary dict/list parameter grids with list/tuple values, and only named
    'linear' or 'rbf' kernels. The base estimator and every candidate must use
    naive_bayes=False and either no kernel keywords or a positive, finite scalar
    RBF gamma. Callable/precomputed kernels, arbitrary kernel configurations,
    custom scorers/refit callables, other input types and unsupported options
    take ordinary sklearn fit with the original inputs.

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
    >>> from sklearn.model_selection import GridSearchCV, StratifiedKFold
    >>> from dwd.kern_md import KernMD
    >>> folds = list(StratifiedKFold(3).split(X, y))  # doctest: +SKIP
    >>> search = GridSearchCV(KernMD(kernel='rbf'),
    ...     {'kernel_kws': [{'gamma': 0.01}, {'gamma': 0.1}]},
    ...     cv=folds, n_jobs=1, scoring='accuracy')  # doctest: +SKIP
    >>> fitted = fit_shared_fold_search(search, X, y)  # doctest: +SKIP
    """
    if not isinstance(search, GridSearchCV):
        raise TypeError('search must be a GridSearchCV instance')
    info = {'route': 'stock', 'reason': None, 'completed': False}
    search.shared_fold_info = info
    reason = _eligibility(search, X, y, fit_params)
    if reason is not None:
        info['reason'] = reason
        result = search.fit(X, y, **fit_params)
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
