"""Reuse training norms within an eligible batched RBF prediction call."""
import inspect
import math
from pathlib import Path
import numpy as np
from scipy.sparse import csr_array, csr_matrix
import sklearn.metrics.pairwise as pairwise
import sklearn.utils.extmath as extmath
from sklearn.utils import _param_validation
from . import _rbf
from . import kernel_utils as ku
from . import _sparse_features as sparse_features
from . import _guarded_rbf as guarded

_ROW_NORMS=extmath.row_norms
_PAIRWISE=ku.pairwise_kernels
_RBF=pairwise.rbf_kernel
_EUCLIDEAN=pairwise.euclidean_distances
_CHECK=pairwise.check_pairwise_arrays
_NORM_SUM=_rbf.norm_sum_rbf
_SUM_WITH_NORMS=_rbf._norm_sum_rbf_with_norms
_CSR_CANONICAL=sparse_features.csr_is_canonical
_SPARSE_FEATURES=sparse_features.canonical_sparse_features
_CHECKED_BASIS=sparse_features.checked_prediction_basis
_GUARDED_RBF=guarded.guarded_rbf
_SPARSE_DOT=extmath.safe_sparse_dot

def _source_function(function,module,name,*,_resolved=None):
    """Decline ordinary pre-import wrappers, not just later replacements."""
    def resolve(value):
        path=Path(value)
        if _resolved is None:
            return path.resolve()
        if path not in _resolved:
            _resolved[path]=path.resolve()
        return _resolved[path]

    try:
        # sklearn's own parameter decorators are supported. A caller wrapper
        # remains observable even when functools.wraps supplies __wrapped__.
        allowed={resolve(module.__file__),resolve(_param_validation.__file__)}
        f=function
        while True:
            if resolve(f.__code__.co_filename) not in allowed:
                return False
            if not hasattr(f,'__wrapped__'):break
            f=f.__wrapped__
        f=inspect.unwrap(function)
        return (f.__module__==module.__name__ and f.__code__.co_name==name and
                resolve(f.__code__.co_filename)==resolve(module.__file__))
    except (AttributeError,TypeError,ValueError):
        return False

def _source_functions_ok():
    # A single initialization checks several functions from the same files.
    # Resolve each path once in this pass; do not retain filesystem state for
    # later direct checks or a module reload.
    resolved={}
    functions=((_ROW_NORMS,extmath,'row_norms'),
               (_RBF,pairwise,'rbf_kernel'),
               (_PAIRWISE,pairwise,'pairwise_kernels'),
               (_EUCLIDEAN,pairwise,'euclidean_distances'),
               (_CHECK,pairwise,'check_pairwise_arrays'),
               (_NORM_SUM,_rbf,'norm_sum_rbf'),
               (_SUM_WITH_NORMS,_rbf,'_norm_sum_rbf_with_norms'))
    # Preserve eager checking of every helper, including exception behavior.
    return all(tuple(_source_function(*args,_resolved=resolved) for args in functions))

_SOURCE_OK=_source_functions_ok()


def _sparse_source_functions_ok():
    resolved={}
    functions=((_CSR_CANONICAL,sparse_features,'csr_is_canonical'),
               (_SPARSE_FEATURES,sparse_features,'canonical_sparse_features'),
               (_CHECKED_BASIS,sparse_features,'checked_prediction_basis'),
               (_GUARDED_RBF,guarded,'guarded_rbf'),
               (_SPARSE_DOT,extmath,'safe_sparse_dot'))
    return all(tuple(_source_function(*args,_resolved=resolved) for args in functions))


_SPARSE_SOURCE_OK=_sparse_source_functions_ok()


_DOT_SOURCE_OK=_source_function(_SPARSE_DOT,extmath,'safe_sparse_dot')
_GUARDED_SOURCE_OK=_source_function(_GUARDED_RBF,guarded,'guarded_rbf')


def _common_evaluator_helpers_ok(mode):
    """Norm reuse requires unchanged evaluators for both dense and CSR inputs."""
    if (not _DOT_SOURCE_OK or
            extmath.safe_sparse_dot is not _SPARSE_DOT or
            pairwise.safe_sparse_dot is not _SPARSE_DOT or
            _rbf.safe_sparse_dot is not _SPARSE_DOT):
        return False
    if mode in ('guarded_sklearn_rel_v1','guarded_norm_sum_rel_v1',
                'guarded_sklearn_rel_v2','guarded_norm_sum_rel_v2') and (
            not _GUARDED_SOURCE_OK or
            guarded.guarded_rbf is not _GUARDED_RBF or
            guarded.row_norms is not _ROW_NORMS or
            guarded.euclidean_distances is not _EUCLIDEAN or
            guarded.check_pairwise_arrays is not _CHECK or
            guarded.safe_sparse_dot is not _SPARSE_DOT):
        return False
    return True


def _canonical_csr_inputs(train,query,original_query,mode):
    """Qualify unchanged canonical CSR storage for this prediction call only."""
    if not _SPARSE_SOURCE_OK:
        return False
    if any(type(x) not in (csr_matrix,csr_array) or x.ndim!=2 or
           x.dtype!=np.dtype('float64') or not x.dtype.isnative
           for x in (train,query,original_query)):
        return False
    # Existing normalization and legacy-basis checks remain observable when
    # replaced. Noncanonical inputs use that original per-batch route, so this
    # optimization cannot suppress duplicate repair or unsafe-model refusal.
    if (sparse_features.csr_is_canonical is not _CSR_CANONICAL or
            sparse_features.canonical_sparse_features is not _SPARSE_FEATURES or
            ku.canonical_sparse_features is not _SPARSE_FEATURES or
            sparse_features.checked_prediction_basis is not _CHECKED_BASIS or
            ku.checked_prediction_basis is not _CHECKED_BASIS):
        return False
    # Inspect the represented indices afresh. Never trust SciPy's mutable
    # canonicality flags, a prior prediction, or read-only status on buffers.
    return _CSR_CANONICAL(train) and _CSR_CANONICAL(query)

def standard_kernel_sources_ok():
    return (_SOURCE_OK and ku.pairwise_kernels is _PAIRWISE and pairwise.pairwise_kernels is _PAIRWISE
            and extmath.row_norms is _ROW_NORMS and pairwise.row_norms is _ROW_NORMS
            and _rbf.row_norms is _ROW_NORMS and _rbf.norm_sum_rbf is _NORM_SUM
            and _rbf._norm_sum_rbf_with_norms is _SUM_WITH_NORMS
            and pairwise.rbf_kernel is _RBF and pairwise.euclidean_distances is _EUCLIDEAN
            and pairwise.check_pairwise_arrays is _CHECK
            and type(pairwise.PAIRWISE_KERNEL_FUNCTIONS) is dict
            and pairwise.PAIRWISE_KERNEL_FUNCTIONS.get('rbf') is _RBF)



# Fresh, call-local validation reuse. No array values or validation state survive
# a public prediction call. The original numerical core is unchanged.
from sklearn.utils import validation as _validation_module
from sklearn import _config as _config_module

_ARRAY_CHECK = pairwise.check_array
_FINITE_CHECK = _validation_module._assert_all_finite
_CONFIG_GET = _config_module.get_config
_CORE = guarded._guarded_rbf_validated
_VALIDATION_SOURCE_OK = all((
    _source_function(_ARRAY_CHECK, _validation_module, 'check_array'),
    _source_function(_FINITE_CHECK, _validation_module, '_assert_all_finite'),
    _source_function(_CONFIG_GET, _config_module, 'get_config'),
    _source_function(_CORE, guarded, '_guarded_rbf_validated')))
_VALIDATION_FUNCTIONS = (_CHECK, _ARRAY_CHECK, _FINITE_CHECK, _CONFIG_GET, _GUARDED_RBF, _CORE)
_VALIDATION_CODES = tuple(getattr(f, '__code__', None) for f in _VALIDATION_FUNCTIONS)
_VALIDATION_DEFAULTS = tuple(getattr(f, '__defaults__', None) for f in _VALIDATION_FUNCTIONS)
_VALIDATION_KWDEFAULTS = tuple(dict(getattr(f, '__kwdefaults__', None) or {}) for f in _VALIDATION_FUNCTIONS)


def _validation_sources_unchanged():
    functions = (guarded.check_pairwise_arrays, pairwise.check_array,
                 _validation_module._assert_all_finite, _config_module.get_config,
                 guarded.guarded_rbf, guarded._guarded_rbf_validated)
    return (_VALIDATION_SOURCE_OK and functions == _VALIDATION_FUNCTIONS and
            pairwise.check_pairwise_arrays is _CHECK and ku.check_array is _ARRAY_CHECK and
            _validation_module.check_array is _ARRAY_CHECK and
            _validation_module._get_config is _CONFIG_GET and
            all(getattr(f, '__code__', None) is code and getattr(f, '__defaults__', None) == defaults and
                (getattr(f, '__kwdefaults__', None) or {}) == kwdefaults
                for f, code, defaults, kwdefaults in zip(functions, _VALIDATION_CODES,
                    _VALIDATION_DEFAULTS, _VALIDATION_KWDEFAULTS)) and
            not _CONFIG_GET()['assume_finite'])


def _same_dense_storage(left, right):
    # Equal start address/shape/strides and exact native float64 identify the
    # same represented elements, even when NumPy normalizes a dtype view.
    return (type(left) is type(right) is np.ndarray and
            left.dtype == right.dtype == np.dtype('float64') and
            left.dtype.isnative and right.dtype.isnative and
            left.dtype.metadata is right.dtype.metadata is None and
            left.shape == right.shape and left.strides == right.strides and
            left.flags.writeable == right.flags.writeable and
            left.__array_interface__['data'][0] == right.__array_interface__['data'][0])


def _dense_validation_candidate(train, query, original_query, mode):
    return (mode == 'guarded_norm_sum_rel_v2' and
            all(type(x) is np.ndarray and x.ndim == 2 and x.dtype == np.dtype('float64')
                and x.dtype.isnative and x.dtype.metadata is None for x in (train, query, original_query)) and
            _same_dense_storage(query, original_query) and query.flags.c_contiguous and
            train.shape[0] > 0 and query.shape[0] > 0 and train.shape[1] == query.shape[1] > 0 and
            _validation_sources_unchanged())


def _contained_dense_batch(batch, query):
    if (type(batch) is not np.ndarray or batch.ndim != 2 or batch.dtype != query.dtype or
            batch.dtype.metadata is not None or batch.shape[1] != query.shape[1] or
            batch.shape[0] <= 0 or batch.strides != query.strides or
            batch.flags.writeable != query.flags.writeable):
        return False
    offset = batch.__array_interface__['data'][0] - query.__array_interface__['data'][0]
    return (offset >= 0 and offset % query.strides[0] == 0 and
            offset // query.strides[0] + batch.shape[0] <= query.shape[0])


def prepare_norm_kernel(estimator,query,original_query,*,_validated_query=None):
    from .gen_kern_dwd import (KernGDWD,_ORIGINAL_TRAINING_KERNEL,_ORIGINAL_KNOWN_PSD_KERNEL)
    if not _SOURCE_OK or type(estimator) is not KernGDWD:
        return None
    if (getattr(estimator._compute_kernel,'__func__',None) is not ku._ORIGINAL_COMPUTE_KERNEL or
        getattr(estimator._kernel_decision_product,'__func__',None) is not ku._ORIGINAL_KERNEL_DECISION_PRODUCT or
        getattr(estimator.decision_function,'__func__',None) is not ku._ORIGINAL_DECISION_FUNCTION or
        getattr(estimator._compute_training_kernel,'__func__',None) is not _ORIGINAL_TRAINING_KERNEL or
        getattr(estimator._known_psd_kernel,'__func__',None) is not _ORIGINAL_KNOWN_PSD_KERNEL):
        return None
    if getattr(estimator,'kernel_',estimator.kernel)!='rbf': return None
    mode=getattr(estimator,'kernel_computation_','sklearn')
    if mode not in ('sklearn','norm_sum','guarded_sklearn_rel_v1','guarded_norm_sum_rel_v1',
                    'guarded_sklearn_rel_v2','guarded_norm_sum_rel_v2'): return None
    if not _common_evaluator_helpers_ok(mode): return None
    train=estimator._Xfit
    dense=all(type(x) is np.ndarray and x.ndim==2 and x.dtype==np.dtype('float64') and x.dtype.isnative
              for x in (train,query,original_query))
    if not dense and not _canonical_csr_inputs(train,query,original_query,mode):
        return None
    if (ku.pairwise_kernels is not _PAIRWISE or pairwise.pairwise_kernels is not _PAIRWISE or
        extmath.row_norms is not _ROW_NORMS or pairwise.row_norms is not _ROW_NORMS or
        _rbf.row_norms is not _ROW_NORMS or _rbf.norm_sum_rbf is not _NORM_SUM or
        _rbf._norm_sum_rbf_with_norms is not _SUM_WITH_NORMS or
        pairwise.rbf_kernel is not _RBF or pairwise.euclidean_distances is not _EUCLIDEAN or
        pairwise.check_pairwise_arrays is not _CHECK or
        type(pairwise.PAIRWISE_KERNEL_FUNCTIONS) is not dict or
        pairwise.PAIRWISE_KERNEL_FUNCTIONS.get('rbf') is not _RBF): return None
    kws=getattr(estimator,'kernel_kws_',estimator.kernel_kws)
    if kws is not None and (type(kws) is not dict or set(kws)-{'gamma'}): return None
    gamma=(kws or {}).get('gamma')
    if gamma is not None:
        if type(gamma) not in (int,float,np.float64,np.int64): return None
        try:
            if not math.isfinite(gamma) or gamma<0: return None
        except (OverflowError,ValueError,TypeError): return None
    errors=np.geterr()
    if errors['under']!='ignore' or any(v not in ('warn','ignore') for v in errors.values()): return None
    try:
        with np.errstate(over='ignore',invalid='ignore'):
            norms=_ROW_NORMS(train,squared=True)
        if not np.isfinite(norms).all(): return None
    except MemoryError:
        return None
    gamma=1./train.shape[1] if gamma is None else gamma
    validated_pair = None
    if (_validated_query is query and
            _dense_validation_candidate(train, query, original_query, mode)):
        # The original public check_array has just validated this exact query.
        # Fresh finite training norms established train finiteness this call.
        validated_pair = (train, query)
    def compute(batch):
        if (validated_pair is not None and _validation_sources_unchanged() and
                _contained_dense_batch(batch, query)):
            return _CORE(validated_pair[0], batch, gamma=gamma, x_norm=norms,
                         mode='norm_sum', preserve_errors=True)
        if mode in ('guarded_sklearn_rel_v1','guarded_norm_sum_rel_v1',
                    'guarded_sklearn_rel_v2','guarded_norm_sum_rel_v2'):
            from ._guarded_rbf import guarded_rbf
            if mode in ('guarded_sklearn_rel_v2','guarded_norm_sum_rel_v2'):
                return guarded_rbf(train,batch,gamma=gamma,x_norm=norms,
                                   mode='norm_sum' if mode=='guarded_norm_sum_rel_v2' else 'sklearn',
                                   preserve_errors=True)
            return guarded_rbf(train,batch,gamma=gamma,x_norm=norms,
                               mode='norm_sum' if mode=='guarded_norm_sum_rel_v1' else 'sklearn')
        if mode=='norm_sum':
            return _SUM_WITH_NORMS(train,batch,gamma=gamma,x_norm=norms)
        checked_train,checked_query=_CHECK(train,batch)
        K=_EUCLIDEAN(checked_train,checked_query,X_norm_squared=norms,squared=True)
        K*=-gamma
        np.exp(K,out=K)
        return K
    return compute



_ORIGINAL_PREPARE_NORM_KERNEL = prepare_norm_kernel
_ORIGINAL_PREPARE_CODE = prepare_norm_kernel.__code__


def prepare_public_norm_kernel(estimator, query, original_query):
    """Private handoff only from the original post-check_array public call site.

    Replaced prepare hooks still receive their original three arguments. Direct
    prepare_norm_kernel calls do not receive a validation handoff.
    """
    live = prepare_norm_kernel
    if live is _ORIGINAL_PREPARE_NORM_KERNEL and live.__code__ is _ORIGINAL_PREPARE_CODE:
        return live(estimator, query, original_query, _validated_query=query)
    return live(estimator, query, original_query)
