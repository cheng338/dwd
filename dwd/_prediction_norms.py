"""Reuse training norms within an eligible batched RBF prediction call."""
import inspect
import math
from pathlib import Path
import numpy as np
import sklearn.metrics.pairwise as pairwise
import sklearn.utils.extmath as extmath
from sklearn.utils import _param_validation
from . import _rbf
from . import kernel_utils as ku

_ROW_NORMS=extmath.row_norms
_PAIRWISE=ku.pairwise_kernels
_RBF=pairwise.rbf_kernel
_EUCLIDEAN=pairwise.euclidean_distances
_CHECK=pairwise.check_pairwise_arrays
_NORM_SUM=_rbf.norm_sum_rbf
_SUM_WITH_NORMS=_rbf._norm_sum_rbf_with_norms

def _source_function(function,module,name):
    """Decline ordinary pre-import wrappers, not just later replacements."""
    try:
        # sklearn's own parameter decorators are supported. A caller wrapper
        # remains observable even when functools.wraps supplies __wrapped__.
        allowed={Path(module.__file__).resolve(),Path(_param_validation.__file__).resolve()}
        f=function
        while True:
            if Path(f.__code__.co_filename).resolve() not in allowed:
                return False
            if not hasattr(f,'__wrapped__'):break
            f=f.__wrapped__
        f=inspect.unwrap(function)
        return (f.__module__==module.__name__ and f.__code__.co_name==name and
                Path(f.__code__.co_filename).resolve()==Path(module.__file__).resolve())
    except (AttributeError,TypeError,ValueError):
        return False

_SOURCE_OK=all((_source_function(_ROW_NORMS,extmath,'row_norms'),
                _source_function(_RBF,pairwise,'rbf_kernel'),
                _source_function(_PAIRWISE,pairwise,'pairwise_kernels'),
                _source_function(_EUCLIDEAN,pairwise,'euclidean_distances'),
                _source_function(_CHECK,pairwise,'check_pairwise_arrays'),
                _source_function(_NORM_SUM,_rbf,'norm_sum_rbf'),
                _source_function(_SUM_WITH_NORMS,_rbf,'_norm_sum_rbf_with_norms')))

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
    if mode not in ('sklearn','norm_sum'): return None
    train=estimator._Xfit
    if any(type(x) is not np.ndarray or x.ndim!=2 or x.dtype!=np.dtype('float64') or not x.dtype.isnative
           for x in (train,query,original_query)): return None
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
        if mode=='norm_sum':
            return _SUM_WITH_NORMS(train,batch,gamma=gamma,x_norm=norms)
        checked_train,checked_query=_CHECK(train,batch)
        K=_EUCLIDEAN(checked_train,checked_query,X_norm_squared=norms,squared=True)
        K*=-gamma
        np.exp(K,out=K)
        return K
    return compute

