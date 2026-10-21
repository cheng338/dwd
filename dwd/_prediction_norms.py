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
    if mode in ('guarded_sklearn_rel_v1','guarded_norm_sum_rel_v1') and (
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


def prepare_norm_kernel(estimator,query,original_query):
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
    if mode not in ('sklearn','norm_sum','guarded_sklearn_rel_v1','guarded_norm_sum_rel_v1'): return None
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
    def compute(batch):
        if mode in ('guarded_sklearn_rel_v1','guarded_norm_sum_rel_v1'):
            from ._guarded_rbf import guarded_rbf
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

