"""Versioned provenance for standard named RBF self construction.

Qualified self kernels use guarded norm-sum construction upfront; the estimator
records the matching cross-query policy. The generation marker rejects stale
training caches. Exact symmetry does not prove the rounded entries are PSD.
"""
import numpy as np
from scipy.sparse import csr_matrix, csr_array

POLICY = "norm_first_v1"


def qualified(estimator, features=None):
    from . import gen_kern_dwd as model
    if (type(estimator) is not model._ORIGINAL_KERNGDWD_TYPE
            or estimator.rbf_computation != 'standard'
            or type(estimator.kernel) is not str or estimator.kernel != 'rbf'):
        return False
    from .kernel_utils import _ORIGINAL_COMPUTE_KERNEL
    methods = (('fit', model._ORIGINAL_FIT),
               ('_compute_kernel', _ORIGINAL_COMPUTE_KERNEL),
               ('_compute_training_kernel', model._ORIGINAL_TRAINING_KERNEL),
               ('_capture_kernel_basis', model._ORIGINAL_CAPTURE_KERNEL_BASIS),
               ('_known_psd_kernel', model._ORIGINAL_KNOWN_PSD_KERNEL),
               ('_guarded_rbf_eligible', model._ORIGINAL_GUARDED_RBF_ELIGIBLE),
               ('cv_init', model._ORIGINAL_CV_INIT),
               ('_cv_cache_matches', model._ORIGINAL_CV_CACHE_MATCHES))
    if any(getattr(getattr(estimator, name), '__func__', None) is not original
           for name, original in methods):
        return False
    if features is not None and (features is not getattr(estimator, '_Xfit', None)
            or type(features) not in (np.ndarray, csr_matrix, csr_array)
            or features.dtype != np.dtype(float)):
        return False
    from ._prediction_norms import (_common_evaluator_helpers_ok,
                                    standard_kernel_sources_ok)
    return (estimator._guarded_rbf_eligible() and standard_kernel_sources_ok()
            and _common_evaluator_helpers_ok('guarded_sklearn_rel_v2'))
