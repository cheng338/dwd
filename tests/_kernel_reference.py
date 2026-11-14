"""Test-only estimator-shaped access to the private reference solver.

This is not a compatibility implementation of the removed public selector.
Public constructor, clone/grid, cache-provenance and exact-estimator dispatch
tests must use the real KernGDWD. Numerical reference tests can retain their
original prediction/callback assertions through this small data/readout scaffold.
The numerical call is directly to dwd._kernel_solver.solve_kernel(reference).
"""
import numpy as np
from sklearn.utils import check_array, check_random_state

from dwd._acceleration_policy import resolve_acceleration
from dwd._fit_state import fit_with_cleanup
from dwd._sparse_features import promote_sparse_features
from dwd._validation import check_real_X_y
from dwd.gen_kern_dwd import KernGDWD, _resolved_affine_computation
from dwd.utils import pm1


class PrivateReferenceKernGDWD:
    """Private-solver proxy around an exact public preparation/readout object.

    Keeping the contained object exact is essential: subclassing KernGDWD would
    intentionally lose qualified named-RBF construction provenance. No method
    identity is spoofed and the production class is never modified.
    """

    implementation = 'reference'

    def __init__(self, *args, **kwargs):
        # Preserve the numerical tests' deliberate historical reference start
        # through a supported explicit request, not a public default or alias.
        kwargs.setdefault('initialization', 'random')
        object.__setattr__(self, '_model', KernGDWD(*args, **kwargs))

    def __getattr__(self, name):
        if name == '_model':
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, '_model'), name)

    def __setattr__(self, name, value):
        if name == '_model':
            object.__setattr__(self, name, value)
        else:
            setattr(self._model, name, value)

    def __delattr__(self, name):
        delattr(self._model, name)

    def get_params(self, deep=True):
        return self._model.get_params(deep=deep)

    def set_params(self, **params):
        self._model.set_params(**params)
        return self

    def __sklearn_clone__(self):
        from sklearn.base import clone
        return type(self)(**clone(self._model).get_params(deep=False))

    def __sklearn_tags__(self):
        return self._model.__sklearn_tags__()

    def cv_init(self, X):
        raise NotImplementedError('Private reference scaffold has no public CV/cache contract.')

    def fit(self, X, y, sample_weight=None, *, K=None, K_eig=None,
            alpha_init=None, offset_init=None, validation_data=None):
        _fit_reference(self._model, X, y, sample_weight, K=K, K_eig=K_eig,
                       alpha_init=alpha_init, offset_init=offset_init,
                       validation_data=validation_data)
        return self


@fit_with_cleanup
def _fit_reference(self, X, y, sample_weight=None, *, K=None, K_eig=None,
                   alpha_init=None, offset_init=None, validation_data=None):
    from dwd._kernel_solver import solve_kernel

    self._validate_options(validation_data)
    effective_acceleration = resolve_acceleration(
        self.acceleration, implementation='reference', backend=self.backend)
    if sample_weight is not None:
        raise NotImplementedError('Sample weights are not implemented for KernGDWD.')
    if self.rbf_computation == 'direct' and K is not None:
        raise ValueError("rbf_computation='direct' constructs K from features; supplied K is unsupported.")
    X = promote_sparse_features(X, force_float64=True)
    X, y = check_real_X_y(X, y, accept_sparse='csr', dtype=np.float64)
    self.classes_ = np.unique(y)
    if len(self.classes_) != 2:
        raise ValueError('KernGDWD requires exactly two classes.')
    self.n_features_in_ = X.shape[1]
    X = self._capture_kernel_basis(X)
    internally_constructed = K is None
    self.kernel_computation_ = 'sklearn'
    self.kernel_symmetry_correction_ = 0.0
    self.kernel_self_construction_ = None
    if K is None:
        K = self._compute_training_kernel(X)
    if np.shape(K) != (X.shape[0], X.shape[0]):
        raise ValueError('K must be the square training kernel for X.')
    K = check_array(K, accept_sparse=False, dtype=np.float64)

    # Private solve_kernel(None) means ZERO. The test scaffold requests its
    # intended initialization explicitly and retains public option validation.
    initial = self.initialization
    if alpha_init is None:
        if initial == 'zero':
            alpha_init = np.zeros(len(y))
        else:
            alpha_init = check_random_state(self.random_state).normal(size=len(y))
            alpha_init /= np.linalg.norm(alpha_init)
    result = solve_kernel(
        K, pm1(y), self.lambd, q=self.q, K_eig=K_eig,
        alpha_init=alpha_init, offset_init=offset_init,
        max_iter=self.max_iter, obj_tol=self.obj_tol,
        stopping=self.stopping, tol=self.tol, backend=self.backend,
        psd_known=internally_constructed and self._known_psd_kernel(),
        callback=self.callback, validation=self._prepare_validation(validation_data),
        patience=self.patience, min_delta=self.min_delta,
        check_interval=self.check_interval, implementation='reference',
        acceleration=effective_acceleration,
        residual_check_order=self.residual_check_order,
        affine_computation=_resolved_affine_computation(self),
        exact_recovery=self.exact_recovery, linear_action=self.linear_action)
    result['diagnostics']['requested_acceleration'] = self.acceleration
    result['diagnostics']['acceleration'] = effective_acceleration
    self._set_fit_result(result)
    return self


def kernel_route(*args, implementation='optimized', **kwargs):
    """Select a real public estimator or an explicitly private test scaffold."""
    if implementation == 'optimized':
        return KernGDWD(*args, **kwargs)
    if implementation == 'reference':
        return PrivateReferenceKernGDWD(*args, **kwargs)
    raise ValueError('Unknown private test route.')
