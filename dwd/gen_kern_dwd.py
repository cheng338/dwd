import numpy as np
from dwd._objective_mean import objective_mean as _objective_mean
from scipy.linalg import eigh
from copy import deepcopy
from numbers import Integral, Real

from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state
from dwd._validation import check_real_X_y, contains_complex, require_real

from dwd.utils import pm1, parameters_equal
from dwd.gen_dwd import V, V_grad
from dwd.kernel_utils import KernelClfMixin, _PREDICTION_BATCH_DEFAULT
from dwd.cv import run_cv
from dwd._eigen import validated_eigh
from dwd._fit_state import fit_with_cleanup
from dwd._linear_action import validate_linear_action
from dwd._residual_order import resolve_residual_check_order
from dwd._acceleration_policy import resolve_acceleration
from dwd._sparse_features import (canonical_sparse_features, csr_is_canonical,
                                  promote_sparse_features)


def _validate_rbf_computation(estimator):
    policy = getattr(estimator, 'rbf_computation', 'standard')
    if not isinstance(policy, str) or policy not in ('standard', 'direct'):
        raise ValueError("rbf_computation must be 'standard' or 'direct'.")
    if policy == 'direct' and (estimator.solver_mode != 'schur'
            or not isinstance(estimator.kernel, str) or estimator.kernel != 'rbf'):
        raise ValueError("rbf_computation='direct' requires named RBF and solver_mode='schur'.")


def _validate_affine_computation(estimator):
    policy = getattr(estimator, 'affine_computation', 'standard')
    if not isinstance(policy, str) or policy not in ('auto', 'standard', 'joint'):
        raise ValueError("affine_computation must be 'auto', 'standard' or 'joint'.")
    if policy == 'joint' and estimator.solver_mode != 'schur':
        raise ValueError("affine_computation='joint' requires solver_mode='schur'.")


def _resolved_affine_computation(estimator):
    """Resolve a new fit without mutating its requested constructor policy."""
    policy = getattr(estimator, 'affine_computation', 'standard')
    if policy == 'auto':
        return 'joint' if estimator.solver_mode == 'schur' else 'standard'
    return policy


def _validate_exact_recovery(estimator):
    policy = getattr(estimator, 'exact_recovery', 'standard')
    if not isinstance(policy, str) or policy not in ('standard', 'extended'):
        raise ValueError("exact_recovery must be 'standard' or 'extended'.")
    if policy == 'extended' and estimator.solver_mode != 'schur':
        raise ValueError("exact_recovery='extended' requires solver_mode='schur'.")


def _validate_residual_check_order(estimator):
    # Full eligibility is resolved after fit has selected an actual eigensystem.
    resolve_residual_check_order(estimator.residual_check_order,
        implementation=estimator.implementation, solver_mode=estimator.solver_mode,
        backend=estimator.backend)


class KernGDWD(KernelClfMixin, BaseEstimator):
    r"""
    Kernel Generalized Distance Weighted Discrimination

    Fits kernel generalized DWD with the loss-plus-penalty formulation of
    Wang and Zou (2018), Another look at distance-weighted discrimination,
    JRSS B 80(1), 177-198, https://doi.org/10.1111/rssb.12244.
    The default solver uses majorization-minimization (MM); an explicit
    L-BFGS backend is also available.

    Parameters
    ----------
    lambd : float, default=1.0
        Coefficient of alpha.T @ K @ alpha in the objective
        mean(V_q(y * f)) + lambd * alpha.T @ K @ alpha. The intercept is
        unregularized. Must be positive for corrected solver_mode='schur';
        legacy mode retains its historical nonnegative validation.
        This parameter is not the SOCP slack penalty C.

    q : float, default=1.0
        Positive generalized DWD loss exponent. q=1 gives standard DWD.

    kernel : str or callable, default='linear'
        Named pairwise kernel, 'precomputed', or a matrix-level callable
        ``kernel(X_train, X_query, **kernel_kws)`` returning training-by-query values.

    kernel_kws : dict or None, default=None
        Kernel keyword arguments, such as {'gamma': 0.1} for the RBF kernel.

    rbf_computation : {'standard', 'direct'}, default='standard'
        Standard selectively repairs cancellation-sensitive distances for
        qualified named RBF kernels in Schur mode, retaining the ordinary
        construction for other pairs. The fitted construction policy is saved
        so later predictions and CV caches use consistent kernel arithmetic.
        Direct evaluates coordinate differences for named RBF kernels in Schur
        mode, with range-safe exceptional handling. It can change floating-point
        kernel entries, fitted coefficients and selected observations. It can
        also cost substantially more. Only gamma is accepted in kernel_kws.
        The fitted policy is retained for scoring;
        supplied training K and replaced construction hooks are unsupported.

    affine_computation : {'auto', 'standard', 'joint'}, default='auto'
        Evaluation policy for the full decision expression K @ alpha + b in
        Schur mode, including explicit validation-stopping scores. Joint avoids
        rounding the dot product before cancellation with the intercept. It
        refines uncertain rows and can use expensive exact rational recovery
        for exceptional range or unresolved signs. Ordinary accepted scores are
        not guaranteed correctly rounded. Kernel construction, solver updates,
        objective checks and the score > 0 tie rule are unchanged. Corrected
        scores may change predictions, CV selection or validation stopping.
        Auto selects joint scoring for corrected Schur models and standard
        scoring for the historical legacy solver. Explicit standard preserves
        the former scoring path. Older objects without this parameter retain
        the class-level standard fallback, including when cloned or refitted.
        The fitted ``affine_computation_`` policy survives constructor changes;
        refit to change it. Callback training scores retain solver arithmetic.

    exact_recovery : {'standard', 'extended'}, default='standard'
        Resource profile for lazy certified MM recovery after ordinary numerical
        failure. Extended requires Schur mode and raises the exact rank and
        rational bit caps from 16/4096 to 32/8192. Both profiles retain the
        65,536-entry and 1,000,000-operation caps for factor preparation and
        each MM preparation/step, with unchanged complete-PSD and accuracy
        checks. This can recover additional small problems at substantial cost;
        it does not guarantee convergence or bound wall time and total memory.
        Healthy fits and L-BFGS do not use this recovery. ``exact_recovery_`` records
        the fitted profile; changing the constructor parameter requires refitting.

    linear_action : {'factor', 'inverse'}, default='factor'
        Computation of the shifted Cholesky system's inverse action in optimized
        Schur MM. Factor uses triangular solves; inverse constructs a dense B
        once per solve, then uses B @ rhs. Both retain the same original-kernel
        acceptance, refinement and bounded recovery. Inverse is explicit and
        may cost more for short or early-stopped fits; there is no size selector.
        Auto with a supplied eigensystem, reference, legacy, explicit spectral
        and L-BFGS routes do not support inverse. Explicit Cholesky still checks
        any supplied eigenpairs. Numerical build refusal uses the factor;
        allocation failure propagates. Centered, anchor and spectral recovery
        release B and use their existing actions. ``requested_linear_action_``
        preserves intent; ``linear_action_`` records the final active action.
        B is private to one solve and is neither fitted state nor a CV cache.
        Float64 B adds 8*n*n bytes, with inverse construction potentially holding
        three such arrays simultaneously, plus vectors and native workspace.
        Thus B alone is 512 MiB at n=8192 or 2 GiB at n=16384; total fit memory
        also includes the kernel, factor, features and other existing state.

    implicit_P: bool
        Use the implicit inverse-product solver. False remains unsupported for
        kernel DWD; use the independent small-system tests as a reference.

    solver_mode: {'legacy', 'schur'}
        New legacy fits emit a FutureWarning; prediction does not.
        'legacy' preserves the released update for reproducibility, including
        its known algebra errors. 'schur' is the corrected published MM update.
        Neither mode regularizes the intercept. Scientific comparisons must
        identify the mode explicitly; old accuracy is not guaranteed to persist.

    max_iter, obj_tol, random_state:
        Maximum accepted MM updates per attempt, absolute successive-objective
        stopping tolerance,
        and initialization seed. Objective tolerance is not a stationarity proof.

    backend : {'auto', 'spectral', 'cholesky', 'lbfgs'}, default='auto'
        Optimized auto uses a supplied eigenbasis, or a Cholesky factor of the
        shifted kernel. Cholesky solves are checked against the original MM equations;
        bounded refinement and a centered factor can recover inaccurate solves.
        Recovery does not change the objective. Reference accepts only auto or
        spectral and always uses a checked eigenbasis, with spectral refinement.
        Invalid supplied eigenpairs are rejected. For trusted internally
        generated RBF kernels, optimized auto fits without a
        callback, including explicit restart acceleration, may restart once
        through the existing spectral backend after
        constrained MM recovery is exhausted. The original initialization,
        kernel and stopping settings are reused. The successful attempt keeps
        max_iter; discarded work and total time are reported separately, so
        total work can exceed one attempt. Explicit backends retain their
        behavior. Unresolved failures raise rather than return an unchecked
        model. Spectral retry keeps its original-kernel score/objective checks;
        it does not waive or reuse a rejected Cholesky coefficient state.

    implementation : {'optimized', 'reference'}, default='optimized'
        Reference retains upstream's coefficient-subtraction MM and Gaussian
        unit-vector initialization, with correctness and numerical repairs.
        Optimized solves directly for the next state, avoiding cancellation and
        normally avoiding eigendecomposition. Both use the same DWD objective
        and an unregularized intercept.

    stopping : {'objective', 'fixed', 'optimality', 'validation'}, default='objective'
        The default retains the absolute successive-objective rule, obj_tol=1e-5
        and max_iter=100 per attempt. Validation stopping requires explicit
        validation_data. An eligible automatic spectral restart can add discarded
        work, recorded separately from the successful attempt.

    initialization : {'auto', 'zero', 'random'}, default='auto'
        Auto starts optimized fits at zero and reference or explicit legacy fits
        at a Gaussian unit vector, controlled by random_state. Explicit fit-time
        alpha_init overrides this setting.

    tol, patience, min_delta, check_interval:
        Numerical optimality tolerance and explicit validation-stopping controls.

    callback : callable or None
        Receives an iteration snapshot. Returning True or raising StopIteration
        requests a stop; snapshot arrays cannot modify the solver's state.
        Spectral-coordinate and L-BFGS checkpoints are checked against the
        original kernel before callbacks or validation observe them. These
        numerical checks add work only when such observations are requested.

    prediction_batch_size : positive int or None
        Bound the number of query rows used in each prediction kernel allocation.

    acceleration : {'auto', None, 'restart'}, default='auto'
        Auto uses score-space momentum with objective-based restart on optimized
        Schur MM (including spectral), and no acceleration on reference, legacy
        or L-BFGS routes. Explicit None preserves ordinary MM; explicit restart
        remains unsupported on those other routes. Resolution never depends on
        stopping policy, data, size, tolerance or observed convergence.
        It preserves the objective and free intercept but changes the
        finite-iteration estimator. Objective-change stopping need not give the
        same solution quality as ordinary MM. Reference, legacy and L-BFGS modes
        do not support explicit restart. Spectral backend requests remain spectral.
        The constructor request is not mutated. requested_acceleration_ and
        acceleration_ record the fitted request and effective policy. Older
        pickles without this parameter retain None, including after clone/refit.

    residual_check_order : {'auto', 'refinement_first', 'adaptive'}, default='auto'
        Auto selects adaptive scheduling for optimized Schur Cholesky MM,
        including acceleration='restart', when no supplied or cached
        eigenbasis is used. Other supported routes retain refinement_first.
        Explicit values retain their previous supported-route validation.
        Adaptive first observes two discarded refinement trials whose original
        states pass the accurate residual checks, then probes those same checks
        before trying a correction. A declined or failed probe falls back to
        refinement. Accuracy gates and recovery budgets are unchanged, but a
        different accepted coefficient state can change finite-iteration results.
        Reference, explicit spectral, L-BFGS and legacy routes do
        not support adaptive ordering. Automatic spectral recovery, if needed,
        uses refinement_first; anchor recovery retains its original checks.
        ``requested_residual_check_order_`` preserves the fit request and
        ``residual_check_order_`` records the successful attempt's policy.
        Diagnostics also record the initial attempt's order after resolution.
        Older serialized objects missing the constructor field retain
        refinement_first when cloned or refitted.
    """

    # Old serialized estimators predate this constructor parameter.
    rbf_computation = 'standard'
    residual_check_order = 'refinement_first'
    affine_computation = 'standard'
    exact_recovery = 'standard'
    linear_action = 'factor'
    acceleration = None

    def __init__(self, lambd=1.0, q=1.0, kernel='linear',
                 kernel_kws=None, implicit_P=True, max_iter=100,
                 obj_tol=1e-5, random_state=None, solver_mode='schur',
                 backend='auto', stopping='objective', initialization='auto',
                 tol=1e-6, patience=3, min_delta=0., check_interval=1,
                 callback=None, prediction_batch_size=None, implementation='optimized',
                 acceleration='auto', residual_check_order='auto',
                 rbf_computation='standard', affine_computation='auto',
                 exact_recovery='standard', linear_action='factor'):
        self.implementation = implementation
        self.acceleration = acceleration
        self.residual_check_order = residual_check_order
        self.rbf_computation = rbf_computation
        self.affine_computation = affine_computation
        self.exact_recovery = exact_recovery
        self.linear_action = linear_action
        self.lambd = lambd
        self.q = q

        self.kernel = kernel
        self.kernel_kws = kernel_kws

        self.implicit_P = implicit_P
        self.max_iter = max_iter
        self.obj_tol = obj_tol
        self.random_state = random_state
        self.solver_mode = solver_mode
        self.backend = backend
        self.stopping = stopping
        self.initialization = initialization
        self.tol = tol
        self.patience = patience
        self.min_delta = min_delta
        self.check_interval = check_interval
        self.callback = callback
        self.prediction_batch_size = prediction_batch_size

    @fit_with_cleanup
    def fit(self, X, y, sample_weight=None, *, K=None, K_eig=None,
            alpha_init=None, offset_init=None, validation_data=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target vector relative to X

        sample_weight : None
            Sample weights are not implemented. Every non-None value raises.

        alpha_init, offset_init : optional
            Finite real initial coefficients and intercept. Complex values,
            including complex scalars inside object arrays, are unsupported.

        validation_data : (X_validation, y_validation) or None
            Explicit monitoring data, used only with stopping='validation'.
            For a precomputed kernel, X_validation is query-by-training.
            The labels must belong to the fitted training classes; a validation
            set containing only one of those classes is allowed.

        Returns
        -------
        self : object
        """
        self._validate_options(validation_data)
        effective_acceleration = resolve_acceleration(self.acceleration,
            implementation=self.implementation, solver_mode=self.solver_mode,
            backend=self.backend)
        if self.rbf_computation == 'direct' and K is not None:
            raise ValueError("rbf_computation='direct' constructs K from features; supplied K is unsupported.")
        if sample_weight is not None:
            raise NotImplementedError('Sample weights are not implemented for KernGDWD.')
        # Callers own K's correspondence to X. Supplied eigenpairs are also
        # checked numerically against K before any corrected MM iterations.
        if self.kernel == 'rbf':
            X = promote_sparse_features(X, force_float64=self.solver_mode == 'schur')
        X, y = check_real_X_y(X, y, accept_sparse='csr',
                         dtype=np.float64 if self.solver_mode == 'schur' else 'numeric')
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError('KernGDWD requires exactly two classes.')
        self.n_features_in_ = X.shape[1]
        X = self._capture_kernel_basis(X)
        internally_constructed = K is None
        used_cv_cache = False
        self.kernel_computation_ = 'sklearn'
        self.kernel_symmetry_correction_ = 0.0

        if K is None:
            if self._cv_cache_matches(X):
                used_cv_cache = True
                K = self._cv_K
                self.kernel_computation_ = getattr(
                    self, '_cv_kernel_computation', 'sklearn')
                self.kernel_symmetry_correction_ = getattr(
                    self, '_cv_kernel_symmetry_correction', 0.0)
                if K_eig is None:
                    K_eig = self._K_eig
            else:
                K = self._compute_training_kernel(X)
        if np.shape(K) != (X.shape[0], X.shape[0]):
            raise ValueError('K must be the square training kernel for X.')
        # Normalize the public array-like argument for both solve and diagnostics.
        K = check_array(K, accept_sparse=False,
                        dtype=np.float64 if self.solver_mode == 'schur' else 'numeric')

        initial = self.initialization
        if initial == 'auto':
            initial = ('random' if self.implementation == 'reference' or self.solver_mode == 'legacy' else 'zero')
        if alpha_init is None and initial == 'zero':
            alpha_init = np.zeros(len(y))
        elif alpha_init is None and self.solver_mode != 'legacy':
            alpha_init = check_random_state(self.random_state).normal(size=len(y))
            alpha_init /= np.linalg.norm(alpha_init)
        # Leave the legacy random initialization inside the preserved function.
        if self.solver_mode == 'legacy':
            alpha, offset, obj_vals, c = solve_gen_kern_dwd(
                K, y, self.lambd, q=self.q, alpha_init=alpha_init,
                offset_init=offset_init, implicit_P=self.implicit_P,
                obj_tol=self.obj_tol if self.stopping == 'objective' else 0.,
                max_iter=self.max_iter, K_eig=K_eig,
                random_state=self.random_state, solver_mode='legacy')
            reached = bool(self.stopping == 'objective' and len(obj_vals) > 1 and
                           abs(obj_vals[-1] - obj_vals[-2]) < self.obj_tol)
            signed_y = pm1(y)
            K_alpha = K.dot(alpha)
            z = signed_y * V_grad(signed_y * (K_alpha + offset), q=self.q) / len(y)
            gradient_alpha = K.dot(z) + 2 * self.lambd * K_alpha
            result = dict(alpha=alpha, offset=offset, objective_history=obj_vals,
                          n_iter=len(obj_vals)-1, returned_iteration=len(obj_vals)-1,
                          termination_reason='objective_tolerance' if reached else 'max_iter',
                          converged=False, criterion_reached=reached,
                          gradient_inf_norm=float(max(abs(z.sum()), np.max(np.abs(gradient_alpha)))),
                          rkhs_gradient_norm=None, dual_gap=None, C=c,
                          backend='legacy', diagnostics={'legacy_algebra': True})
        else:
            from dwd._kernel_solver import solve_kernel
            from dwd._kernel_recovery import solve_with_spectral_restart
            validation = self._prepare_validation(validation_data)
            # A known family name alone does not establish construction
            # provenance when a subclass or an instance replaces either method.
            can_restart = (
                internally_constructed and K_eig is None
                and self.solver_mode == 'schur'
                and isinstance(self.kernel, str) and self.kernel == 'rbf'
                and self.implementation == 'optimized' and self.backend == 'auto'
                and effective_acceleration in (None, 'restart') and self.callback is None
                and getattr(self._compute_kernel, '__func__', None)
                    is KernelClfMixin._compute_kernel
                and getattr(self._compute_training_kernel, '__func__', None)
                    is KernGDWD._compute_training_kernel
                and getattr(self._known_psd_kernel, '__func__', None)
                    is KernGDWD._known_psd_kernel
                and (not used_cv_cache or (
                    getattr(self, '_cv_trusted_rbf_construction', False)
                    and getattr(self.cv_init, '__func__', None) is KernGDWD.cv_init
                    and getattr(self._cv_cache_matches, '__func__', None)
                        is KernGDWD._cv_cache_matches))
                and self._known_psd_kernel())
            result = solve_with_spectral_restart(
                solve_kernel,
                K, pm1(y), self.lambd, eligible=can_restart,
                q=self.q, K_eig=K_eig,
                alpha_init=alpha_init, offset_init=offset_init,
                max_iter=self.max_iter, obj_tol=self.obj_tol,
                stopping=self.stopping, tol=self.tol, backend=self.backend,
                psd_known=internally_constructed and self._known_psd_kernel(),
                callback=self.callback, validation=validation,
                patience=self.patience, min_delta=self.min_delta,
                check_interval=self.check_interval, implementation=self.implementation,
                acceleration=effective_acceleration, residual_check_order=self.residual_check_order,
                affine_computation=_resolved_affine_computation(self),
                linear_action=self.linear_action,
                **({'exact_recovery': self.exact_recovery}
                   if self.exact_recovery != 'standard' else {}))
        result['diagnostics']['requested_acceleration'] = self.acceleration
        result['diagnostics']['acceleration'] = effective_acceleration
        self._set_fit_result(result)
        return self

    def _validate_options(self, validation_data=None):
        self._validate_direct_rbf_configuration()
        _validate_affine_computation(self)
        _validate_exact_recovery(self)
        validate_linear_action(self.linear_action, implementation=self.implementation,
                               solver_mode=self.solver_mode, backend=self.backend)
        _validate_residual_check_order(self)
        if self.implementation not in ('optimized', 'reference'):
            raise ValueError("implementation must be 'optimized' or 'reference'.")
        if self.implementation == 'reference' and self.solver_mode == 'legacy':
            raise ValueError('The repaired reference requires solver_mode=schur.')
        resolve_acceleration(self.acceleration, implementation=self.implementation,
            solver_mode=self.solver_mode, backend=self.backend)
        if self.solver_mode not in ('legacy', 'schur'):
            raise ValueError("solver_mode must be 'legacy' or 'schur'.")
        if not self.implicit_P:
            raise NotImplementedError('Kernel DWD supports only implicit_P=True.')
        if self.backend not in ('auto', 'spectral', 'cholesky', 'lbfgs'):
            raise ValueError("backend must be 'auto', 'spectral', 'cholesky', or 'lbfgs'.")
        if self.stopping not in ('objective', 'fixed', 'optimality', 'validation'):
            raise ValueError('Unknown stopping policy.')
        if self.initialization not in ('auto', 'zero', 'random'):
            raise ValueError("initialization must be 'auto', 'zero', or 'random'.")
        if self.callback is not None and not callable(self.callback):
            raise TypeError('callback must be callable or None.')
        batch = self.prediction_batch_size
        if batch is not None and (isinstance(batch, (bool, np.bool_)) or
                                 not isinstance(batch, Integral) or batch <= 0):
            raise ValueError('prediction_batch_size must be a positive integer or None.')
        if self.stopping == 'validation' and validation_data is None:
            raise ValueError("stopping='validation' requires explicit validation_data.")
        if self.stopping != 'validation' and validation_data is not None:
            raise ValueError("validation_data is used only with stopping='validation'.")
        if self.solver_mode == 'legacy' and (self.backend not in ('auto', 'spectral') or
                self.stopping not in ('objective', 'fixed') or self.callback is not None):
            raise ValueError('Legacy mode supports only the spectral backend, objective/fixed stopping, and no callback.')

    def _validate_direct_rbf_configuration(self):
        _validate_rbf_computation(self)
        if self.rbf_computation == 'direct':
            if self.kernel_kws is not None and not isinstance(self.kernel_kws, dict):
                raise TypeError('Direct RBF kernel_kws must be a dictionary or None.')
            if not self._direct_rbf_eligible():
                raise ValueError('Direct RBF requires finite nonnegative gamma, no other '
                                 'kernel keywords, and the standard kernel construction hooks.')

    def _direct_rbf_eligible(self):
        """Use the opt-in policy only with supported construction provenance."""
        return (self.rbf_computation == 'direct'
                and self.solver_mode == 'schur'
                and isinstance(self.kernel, str) and self.kernel == 'rbf'
                and not (set(self.kernel_kws or {}) - {'gamma'})
                and getattr(self._compute_kernel, '__func__', None)
                    is KernelClfMixin._compute_kernel
                and getattr(self._compute_training_kernel, '__func__', None)
                    is KernGDWD._compute_training_kernel
                and getattr(self._known_psd_kernel, '__func__', None)
                    is KernGDWD._known_psd_kernel
                and self._known_psd_kernel())

    def _guarded_rbf_eligible(self):
        from .kernel_utils import _ORIGINAL_COMPUTE_KERNEL
        if not (self.rbf_computation == 'standard' and self.solver_mode == 'schur'
                and isinstance(self.kernel, str) and self.kernel == 'rbf'
                and getattr(self._compute_kernel, '__func__', None) is _ORIGINAL_COMPUTE_KERNEL
                and getattr(self._compute_training_kernel, '__func__', None) is _ORIGINAL_TRAINING_KERNEL
                and getattr(self._known_psd_kernel, '__func__', None) is _ORIGINAL_KNOWN_PSD_KERNEL):
            return False
        kws = self.kernel_kws
        if kws is not None and type(kws) is not dict:
            return False
        gamma = (kws or {}).get('gamma')
        if gamma is not None and type(gamma) not in (bool, int, float, np.float64, np.int64):
            return False
        errors = np.geterr()
        if errors['under'] != 'ignore' or any(v not in ('warn', 'ignore') for v in errors.values()):
            return False
        if set(kws or {}) - {'gamma'} or not self._known_psd_kernel():
            return False
        from ._prediction_norms import standard_kernel_sources_ok
        return standard_kernel_sources_ok()

    def _compute_training_kernel(self, X):
        """Construct the requested RBF policy without changing solver checks."""
        if self.rbf_computation == 'direct':
            self._validate_direct_rbf_configuration()
        if self._direct_rbf_eligible():
            self.kernel_computation_ = 'direct_v1'
            self.kernel_symmetry_correction_ = 0.0
            return self._compute_kernel(X)
        guarded = self._guarded_rbf_eligible()
        self.kernel_computation_ = 'guarded_sklearn_rel_v1' if guarded else 'sklearn'
        self.kernel_symmetry_correction_ = 0.0
        K = self._compute_kernel(X)
        if (self.solver_mode == 'schur' and isinstance(self.kernel, str)
                and self.kernel == 'rbf'
                and getattr(self._compute_kernel, '__func__', None)
                    is KernelClfMixin._compute_kernel
                and self._known_psd_kernel()):
            from ._rbf import self_kernel_asymmetry
            asymmetry, tolerance = self_kernel_asymmetry(K)
            if asymmetry > tolerance:
                self.kernel_computation_ = 'guarded_norm_sum_rel_v1' if guarded else 'norm_sum'
                self.kernel_symmetry_correction_ = asymmetry
                # Discard the old matrix; reconstruct from trusted features.
                # Existing PSD/solver checks still apply to the result.
                del K
                K = self._compute_kernel(X)
        return K

    def _known_psd_kernel(self):
        """Only internally constructed kernels with known PSD parameters qualify."""
        if not isinstance(self.kernel, str):
            return False
        if self.kernel == 'linear':
            return True
        kws = self.kernel_kws or {}
        gamma = kws.get('gamma', None)
        gamma_ok = gamma is None or (isinstance(gamma, Real) and np.isfinite(gamma) and gamma >= 0)
        if self.kernel == 'rbf':
            return gamma_ok
        if self.kernel in ('poly', 'polynomial'):
            degree, coef0 = kws.get('degree', 3), kws.get('coef0', 1)
            return bool(gamma_ok and isinstance(degree, Integral) and degree >= 0 and
                        isinstance(coef0, Real) and np.isfinite(coef0) and coef0 >= 0)
        return False

    def _prepare_validation(self, validation_data):
        if validation_data is None:
            return None
        if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
            raise ValueError('validation_data must be an (X_validation, y_validation) pair.')
        X_val, y_val = validation_data
        if self.kernel == 'rbf':
            X_val = promote_sparse_features(X_val, force_float64=True)
        X_val, y_val = check_real_X_y(X_val, y_val, accept_sparse='csr', dtype=np.float64)
        if not np.isin(y_val, self.classes_).all():
            raise ValueError('Validation labels must belong to the training classes.')
        if self.kernel != 'precomputed' and X_val.shape[1] != self.n_features_in_:
            raise ValueError('Validation feature count differs from training data.')
        K_val = self._compute_kernel(X_val).T
        if hasattr(K_val, 'toarray'):
            K_val = K_val.toarray()
        return (np.asarray(K_val), np.where(y_val == self.classes_[1], 1., -1.))

    def _set_fit_result(self, result):
        precision = result.get('prediction_precision', 'ordinary')
        if precision not in ('ordinary', 'compensated', 'adaptive'):
            raise ValueError('Invalid fitted kernel prediction precision.')
        self.prediction_precision_ = precision
        self.affine_computation_ = result.get('affine_computation', 'standard')
        self.exact_recovery_ = result.get('exact_recovery', 'standard')
        self.intercept_ = np.asarray(result['offset']).reshape(-1)
        self.dual_coef_ = np.asarray(result['alpha']).reshape(1, -1)
        self.objective_history_ = np.asarray(result['objective_history'])
        self.obj_vals_ = self.objective_history_.tolist()
        self.n_iter_ = int(result['n_iter'])
        self.returned_iteration_ = int(result['returned_iteration'])
        self.final_objective_ = float(result.get('final_objective',
                                     self.objective_history_[self.returned_iteration_]))
        self.converged_ = bool(result['converged'])
        self.termination_reason_ = result['termination_reason']
        self.criterion_reached_ = bool(result.get('criterion_reached', self.converged_ or
                                      self.termination_reason_ == 'objective_tolerance'))
        self.C_ = result['C']
        self.C_conversion_finite_ = bool(np.isfinite(self.C_))
        self.gradient_inf_norm_ = result['gradient_inf_norm']
        self.rkhs_gradient_norm_ = result['rkhs_gradient_norm']
        self.dual_gap_ = result['dual_gap']
        self.dual_equality_residual_ = result.get('dual_equality_residual')
        self.stationarity_residual_ = self.rkhs_gradient_norm_
        self.stationarity_checked_ = self.rkhs_gradient_norm_ is not None
        self.optimality_met_ = self.converged_
        self.objective_tolerance_met_ = bool(result.get('objective_tolerance_met',
                    self.n_iter_ > 0 and abs(self.objective_history_[-1] -
                                            self.objective_history_[-2]) < self.obj_tol))
        self.backend_ = result['backend']
        self.diagnostics_ = result['diagnostics']
        self.requested_linear_action_ = self.diagnostics_.get('requested_linear_action', self.linear_action)
        self.linear_action_ = self.diagnostics_.get('linear_action', 'factor')
        self.requested_acceleration_ = self.diagnostics_.get(
            'requested_acceleration', self.acceleration)
        self.acceleration_ = self.diagnostics_.get('acceleration', None)
        # Preserve constructor intent and the successful fit's actual route.
        self.diagnostics_.setdefault('requested_residual_check_order', self.residual_check_order)
        self.diagnostics_.setdefault('residual_check_order', 'refinement_first')
        self.diagnostics_.setdefault('initial_residual_check_order', self.diagnostics_['residual_check_order'])
        self.requested_residual_check_order_ = self.diagnostics_['requested_residual_check_order']
        self.residual_check_order_ = self.diagnostics_['residual_check_order']
        self.validation_history_ = result.get('validation_history',
                                               self.diagnostics_.get('validation_history', []))

    def _capture_kernel_basis(self, X):
        """Own prediction features and snapshot the parameters used to fit."""
        self.kernel_ = self.kernel
        self.kernel_kws_ = deepcopy(self.kernel_kws)
        if self.kernel == 'precomputed' and isinstance(X, np.ndarray):
            # Prediction uses only the training row count in this mode. A view
            # keeps its shape stable without duplicating the square Gram matrix.
            self._Xfit = X.view()
        elif isinstance(X, np.ndarray):
            self._Xfit = X.copy(order='K')
        else:
            self._Xfit = X.copy()
        if self.kernel == 'rbf' and self.rbf_computation != 'direct':
            self._Xfit = canonical_sparse_features(self._Xfit, copy=False)
        return self._Xfit

    @fit_with_cleanup
    def cv_init(self, X):
        """
        Prepare a tuning path and invalidate any previous fitted predictor.

        New training rows cannot be combined with old coefficients. A failed
        preparation also leaves the estimator unfitted; checked private caches
        remain subject to their existing data and parameter validity checks.
        """
        self._validate_direct_rbf_configuration()
        _validate_affine_computation(self)
        _validate_exact_recovery(self)
        validate_linear_action(self.linear_action, implementation=self.implementation,
                               solver_mode=self.solver_mode, backend=self.backend)
        # Promote before constructing the Gram matrix, not after rounding it.
        if self.kernel == 'rbf':
            X = promote_sparse_features(X, force_float64=self.solver_mode == 'schur')
        X = check_array(X, accept_sparse='csr',
                        dtype=np.float64 if self.solver_mode == 'schur' else 'numeric')
        if self.kernel == 'precomputed':
            # Unlike ordinary prediction, the CV cache reuses the matrix's
            # values. Own them so caller mutation cannot corrupt a later fit.
            X = X.copy()
        X = self._capture_kernel_basis(X)

        # Record construction-time provenance, not merely the methods that
        # happen to be bound later when fit reuses this cache.
        trusted_rbf = (
            self.solver_mode == 'schur'
            and isinstance(self.kernel, str) and self.kernel == 'rbf'
            and getattr(self._compute_kernel, '__func__', None)
                is KernelClfMixin._compute_kernel
            and getattr(self._compute_training_kernel, '__func__', None)
                is KernGDWD._compute_training_kernel
            and getattr(self._known_psd_kernel, '__func__', None)
                is KernGDWD._known_psd_kernel
            and getattr(self.cv_init, '__func__', None) is KernGDWD.cv_init
            and getattr(self._cv_cache_matches, '__func__', None)
                is KernGDWD._cv_cache_matches
            and self._known_psd_kernel())
        K = self._compute_training_kernel(X)
        if (self.backend in ('auto', 'cholesky') and self.solver_mode == 'schur'
                and self.implementation == 'optimized'):
            self._K_eig = None
        else:
            # Explicit path initialization opts into eigenvalue reuse across
            # lambd/q candidates. A normal single fit need not do this work.
            self._set_K_eig(K)
        self._cv_K = K
        self._cv_trusted_rbf_construction = bool(trusted_rbf)
        self._cv_kernel_computation = self.kernel_computation_
        self._cv_rbf_generation = ('relative_guard_v1', self.kernel_computation_)
        self._cv_kernel_symmetry_correction = self.kernel_symmetry_correction_
        # X is already owned by this preparation, including precomputed mode.
        self._cv_X = X
        self._cv_kernel = self.kernel
        self._cv_kernel_kws = deepcopy(self.kernel_kws)
        self._cv_solver_mode = self.solver_mode
        self._cv_backend = self.backend
        self._cv_implementation = self.implementation
        self._cv_rbf_computation = self.rbf_computation
        self._cv_affine_computation = self.affine_computation
        self._cv_exact_recovery = self.exact_recovery
        return self

    def _cv_cache_matches(self, X):
        """Invalidate hidden precomputation whenever data or kernel changes."""
        if self.exact_recovery != getattr(self, '_cv_exact_recovery', 'standard'):
            return False
        if self.affine_computation != getattr(self, '_cv_affine_computation', 'standard'):
            return False
        requested = self.rbf_computation
        if requested != getattr(self, '_cv_rbf_computation', 'standard'):
            return False
        cached_policy = getattr(self, '_cv_kernel_computation', None)
        if requested == 'standard' and isinstance(self.kernel, str) and self.kernel == 'rbf':
            guarded_cache = cached_policy in ('guarded_sklearn_rel_v1', 'guarded_norm_sum_rel_v1')
            generation = getattr(self, '_cv_rbf_generation', None)
            if generation is not None and generation != ('relative_guard_v1', cached_policy):
                return False
            if guarded_cache and generation is None:
                return False
            if generation is None and self._guarded_rbf_eligible():
                # Unversioned old caches can contain the legacy geometry.
                # Prediction of old fitted models remains legacy; new fitting
                # rebuilds such a cache under the newly requested policy.
                return False
            # An explicit cv_init owns its represented K. Later instrumentation
            # or an errstate change does not alter that matrix's construction;
            # its recorded policy remains paired with coefficients fitted to K.

        if requested == 'direct':
            if cached_policy != 'direct_v1' or not self._direct_rbf_eligible():
                return False
        elif cached_policy == 'direct_v1':
            return False
        if (not hasattr(self, '_cv_X') or
                not hasattr(self, '_cv_kernel_computation')):
            return False
        if (self.kernel == 'rbf' and cached_policy != 'direct_v1'
                and not csr_is_canonical(self._cv_X)):
            # Earlier norm/dot caches could encode an incorrect Gram matrix
            # for duplicate CSR storage. Equal represented features do not
            # establish that such a cached matrix is reusable.
            return False
        # Corrected fits and cv_init construct K from float64 features. Compare
        # that same representation before deciding whether a CV candidate needs
        # new preparation; float32 inputs otherwise miss every lambda-path cache.
        if self.solver_mode == 'schur' and X.dtype != np.dtype(np.float64):
            X = X.astype(np.float64, copy=False)
        if (not parameters_equal(self.kernel, self._cv_kernel) or
                not parameters_equal(self.kernel_kws, self._cv_kernel_kws) or
                self.solver_mode != self._cv_solver_mode or
                self.backend != self._cv_backend or
                self.implementation != self._cv_implementation or
                X.dtype != self._cv_X.dtype or
                X.shape != self._cv_X.shape):
            return False
        if hasattr(X, 'toarray') or hasattr(self._cv_X, 'toarray'):
            if not (hasattr(X, 'toarray') and hasattr(self._cv_X, 'toarray')):
                return False
            return (X != self._cv_X).nnz == 0
        return np.array_equal(X, self._cv_X)

    def _set_K_eig(self, X):
        """
        Precomputes eigen decomposition of K matrix which makes
        cross-validation much faster.
        """
        self._K_eig = get_K_eig(X, reference=self.implementation == 'reference')

    def _get_K_eig(self):
        if hasattr(self, '_K_eig'):
            return self._K_eig
        else:
            return None


_ORIGINAL_KERNGDWD_TYPE = KernGDWD
_ORIGINAL_TRAINING_KERNEL = KernGDWD._compute_training_kernel
_ORIGINAL_KNOWN_PSD_KERNEL = KernGDWD._known_psd_kernel


def _direct_prediction_kernel(estimator):
    """Use call-local basis validation only for unchanged built-in direct hooks."""
    from .kernel_utils import (_ORIGINAL_COMPUTE_KERNEL,
                               _ORIGINAL_KERNEL_DECISION_PRODUCT,
                               _ORIGINAL_DECISION_FUNCTION)
    if (type(estimator) is not _ORIGINAL_KERNGDWD_TYPE or
            getattr(estimator, 'kernel_', estimator.kernel) != 'rbf' or
            getattr(estimator, 'kernel_computation_', None) != 'direct_v1' or
            getattr(estimator._compute_kernel, '__func__', None)
                is not _ORIGINAL_COMPUTE_KERNEL or
            getattr(estimator._compute_training_kernel, '__func__', None)
                is not _ORIGINAL_TRAINING_KERNEL or
            getattr(estimator._known_psd_kernel, '__func__', None)
                is not _ORIGINAL_KNOWN_PSD_KERNEL or
            getattr(estimator._kernel_decision_product, '__func__', None)
                is not _ORIGINAL_KERNEL_DECISION_PRODUCT or
            getattr(estimator.decision_function, '__func__', None)
                is not _ORIGINAL_DECISION_FUNCTION):
        return None
    from ._direct_rbf import _prediction_basis_kernel
    basis_kernel = _prediction_basis_kernel(estimator._Xfit)
    if basis_kernel is None:
        return None

    def compute(query):
        gamma = (getattr(estimator, 'kernel_kws_', estimator.kernel_kws) or {}).get('gamma')
        if isinstance(gamma, bool):
            gamma = float(gamma)
        return basis_kernel(query, gamma=gamma)

    return compute


class KernGDWDCV(KernelClfMixin, BaseEstimator):
    """
    Fit kernel generalized DWD with cross-validation.
    Reuses fold-specific kernels and, when needed, validated eigenpairs across
    compatible candidates. The search loop is serial; runtime depends on the
    grid and problem size.

    Parameters
    ----------
    lambd_vals: list of floats
        The lambda values to cross-validate over.

    q_vals: list of floats
        The q-values to cross validate over.

    kernel: str, callable
        The kernel to use.

    kernel_kws_vals: list of dicts
        The kernel parameters to validate over.

    rbf_computation : {'standard', 'direct'}, default='standard'
        Forwarded to every candidate and the final refit. Direct requires named
        RBF in Schur mode with gamma-only kernel keywords; see KernGDWD for
        numerical compatibility and cost limits. Standard preserves existing
        construction behavior.

    affine_computation : {'auto', 'standard', 'joint'}, default='auto'
        Forwarded to every candidate and the final refit. Auto selects joint
        for Schur mode and standard for legacy mode. Joint uses the full affine
        decision evaluation described by KernGDWD and requires Schur mode.
        It can change CV scores and the chosen candidate.

    exact_recovery : {'standard', 'extended'}, default='standard'
        Forwarded to every candidate and the final refit. Extended requires
        Schur mode and increases the bounded exact recovery capacity described
        by KernGDWD; it does not weaken numerical acceptance checks.

    linear_action : {'factor', 'inverse'}, default='factor'
        Forwarded to every candidate and final refit; each solve owns its
        workspace. See KernGDWD for route, recovery and setup-memory limits.

    cv:
        Cross-validation splitter or fold count, interpreted by
        sklearn.model_selection.check_cv.

    scoring:
        Scorer name or callable accepted by sklearn.metrics.check_scoring.
        Must return a finite real scalar.

    acceleration : {'auto', None, 'restart'}, default='auto'
        Forwarded unchanged to every candidate and the final KernGDWD refit.
        The option changes finite-iteration behavior. Use None to retain
        ordinary MM. Auto selects restart only for optimized Schur MM.

    residual_check_order : {'auto', 'refinement_first', 'adaptive'}, default='auto'
        Forwarded to every candidate and the final refit. Adaptive ordering is
        restricted to optimized Cholesky MM, including restart acceleration,
        and may change
        finite-iteration results; see KernGDWD.

    """
    # Old serialized CV estimators retain the standard constructor policy.
    rbf_computation = 'standard'
    residual_check_order = 'refinement_first'
    affine_computation = 'standard'
    exact_recovery = 'standard'
    linear_action = 'factor'
    acceleration = None

    def __init__(self,
                 lambd_vals=np.logspace(-2, 2, 10),
                 q_vals=np.logspace(-2, 2, 5),
                 kernel='linear',
                 kernel_kws_vals=None,
                 cv=5, scoring='accuracy', max_iter=100, obj_tol=1e-5,
                 random_state=None, solver_mode='schur', backend='auto',
                 stopping='objective', initialization='auto', tol=1e-6,
                 patience=3, min_delta=0., check_interval=1, callback=None,
                 prediction_batch_size=None, implementation='optimized', acceleration='auto',
                 residual_check_order='auto', rbf_computation='standard',
                 affine_computation='auto', exact_recovery='standard', linear_action='factor'):

        self.lambd_vals = lambd_vals
        self.implementation = implementation
        self.acceleration = acceleration
        self.residual_check_order = residual_check_order
        self.rbf_computation = rbf_computation
        self.affine_computation = affine_computation
        self.exact_recovery = exact_recovery
        self.linear_action = linear_action
        self.q_vals = q_vals
        self.kernel = kernel
        self.kernel_kws_vals = kernel_kws_vals

        self.cv = cv
        self.scoring = scoring
        self.max_iter = max_iter
        self.obj_tol = obj_tol
        self.random_state = random_state
        self.solver_mode = solver_mode
        self.backend = backend
        self.stopping = stopping
        self.initialization = initialization
        self.tol = tol
        self.patience = patience
        self.min_delta = min_delta
        self.check_interval = check_interval
        self.callback = callback
        self.prediction_batch_size = prediction_batch_size

    @fit_with_cleanup
    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target vector relative to X

        sample_weight : None
            Sample weights are not implemented. Every non-None value raises.
        Returns
        -------
        self : object
        """
        if sample_weight is not None:
            raise NotImplementedError('Sample weights are not implemented for KernGDWDCV.')
        _validate_rbf_computation(self)
        _validate_affine_computation(self)
        _validate_exact_recovery(self)
        validate_linear_action(self.linear_action, implementation=self.implementation,
                               solver_mode=self.solver_mode, backend=self.backend)
        _validate_residual_check_order(self)
        if self.stopping == 'validation':
            raise ValueError('KernGDWDCV does not create monitoring splits. Use a plain '
                             'KernGDWD with explicit validation_data inside an explicitly '
                             'constructed training/monitoring/CV procedure.')
        if self.kernel == 'rbf':
            X = promote_sparse_features(X, force_float64=self.solver_mode == 'schur')
        X, y = check_real_X_y(X, y, accept_sparse='csr',
                         dtype='numeric')

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError('KernGDWDCV requires exactly two classes.')
        self.n_features_in_ = X.shape[1]

        # run cross validation
        params = {'q': self.q_vals, 'lambd': self.lambd_vals,
                  'kernel_kws': {} if self.kernel_kws_vals is None else self.kernel_kws_vals}

        best_params, best_score, best_clf, agg_results, all_cv_results = \
            run_cv(clf=KernGDWD(kernel=self.kernel, max_iter=self.max_iter,
                               obj_tol=self.obj_tol, random_state=self.random_state,
                               solver_mode=self.solver_mode, backend=self.backend,
                               stopping=self.stopping, initialization=self.initialization,
                               tol=self.tol, patience=self.patience, min_delta=self.min_delta,
                               check_interval=self.check_interval, callback=self.callback,
                               prediction_batch_size=self.prediction_batch_size,
                               implementation=self.implementation, acceleration=self.acceleration,
                               residual_check_order=self.residual_check_order,
                               rbf_computation=self.rbf_computation,
                               affine_computation=self.affine_computation,
                               exact_recovery=self.exact_recovery, linear_action=self.linear_action),
                   X=X, y=y,
                   params=params,
                   scoring=self.scoring,
                   cv=self.cv,
                   refit_best=True)

        self.best_estimator_ = best_clf

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.agg_cv_results_ = agg_results
        self.all_cv_results_ = all_cv_results

        self._Xfit = self.best_estimator_._Xfit
        self.intercept_ = self.best_estimator_.intercept_
        self.dual_coef_ = self.best_estimator_.dual_coef_
        self.prediction_precision_ = getattr(self.best_estimator_, 'prediction_precision_', 'ordinary')
        self.affine_computation_ = getattr(self.best_estimator_, 'affine_computation_', 'standard')
        self.exact_recovery_ = getattr(self.best_estimator_, 'exact_recovery_', 'standard')
        self.requested_residual_check_order_ = self.best_estimator_.requested_residual_check_order_
        self.residual_check_order_ = self.best_estimator_.residual_check_order_
        self.requested_linear_action_ = self.best_estimator_.requested_linear_action_
        self.linear_action_ = self.best_estimator_.linear_action_
        self.requested_acceleration_ = self.best_estimator_.requested_acceleration_
        self.acceleration_ = self.best_estimator_.acceleration_

        return self

    def decision_function(self, X, *, prediction_batch_size=_PREDICTION_BATCH_DEFAULT):
        """Score with the wrapper's current batch limit, or a per-call override."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, 'best_estimator_')
        batch_size = (getattr(self, 'prediction_batch_size', None)
                      if prediction_batch_size is _PREDICTION_BATCH_DEFAULT
                      else prediction_batch_size)
        return self.best_estimator_.decision_function(
            X, prediction_batch_size=batch_size)


def solve_gen_kern_dwd(K, y, lambd, q=1,
                       alpha_init=None, offset_init=None,
                       sample_weight=None,
                       implicit_P=True,
                       obj_tol=1e-5, max_iter=100,
                       K_eig=None, random_state=None, solver_mode='schur', *,
                       acceleration='auto', linear_action='factor'):

    """
    Solves the kernel gDWD problem using the MM algorithm derived in Wang and Zou (2018).

    Parameters
    ----------
    K: array-like, (n_samples, n_samples)
        The kernel.

    y: array-like, (n_samples, )
        The vector of binary class labels.

    lambd : float
        Coefficient of alpha.T @ K @ alpha in mean(V_q(y * f)) plus the
        squared-norm penalty. Must be positive for solver_mode='schur';
        legacy mode retains its historical nonnegative validation. The
        intercept is unregularized.

    q : float, default=1.0
        Positive generalized DWD loss exponent. q=1 gives standard DWD.

    alpha_init, offset_init:
        Initial values to start the optimization algorithm from.

    sample_weight : None
        Sample weights are unsupported. Any non-None value raises
        NotImplementedError.

    implicit_P : bool, default=True
        Must be True. This historic switch is distinct from linear_action.

    obj_tol: float
        Stopping condition for difference between successive objective
        functions.

    max_iter: int
        Maximum number of iterations to perform.

    K_eig: None or (U, D)
        Optional precomputed eigendecomposition of this exact K. Supplied
        pairs are validated against K for finite values, vector norms,
        orthogonality and eigen-equation consistency. Large matrices use
        bounded numerical probes rather than a full cubic reconstruction;
        these checks are numerical screening, not a formal certificate.
        Invalid supplied pairs are rejected without silently recomputing them.

    acceleration : {'auto', None, 'restart'}, keyword-only, default='auto'
        Auto selects score-space restart for corrected Schur MM and no
        acceleration for legacy mode. Explicit None retains ordinary MM.
        Explicit restart is unsupported in legacy mode. The objective,
        numerical checks and absolute objective-change stopping rule are
        unchanged; the finite sequence and returned state can change.
    """

    # The keyword-only addition preserves every historical positional argument.
    effective_acceleration = resolve_acceleration(acceleration, solver_mode=solver_mode)
    validate_linear_action(linear_action, solver_mode=solver_mode, K_eig=K_eig)
    # argument checking and formatting
    for name, value in (('lambd', lambd), ('q', q), ('obj_tol', obj_tol)):
        require_real(value, name)
    if not np.isfinite(lambd) or lambd < 0:
        raise ValueError("Penalty term must be positive; got (lambd=%r)"
                         % lambd)

    if not np.isfinite(q) or q <= 0:
        raise ValueError("Weight term must be positive; got (q=%r)" % q)
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError('max_iter must be a nonnegative integer.')
    if not np.isfinite(obj_tol) or obj_tol < 0:
        raise ValueError('obj_tol must be finite and nonnegative.')
    if solver_mode not in ('legacy', 'schur'):
        raise ValueError("solver_mode must be 'legacy' or 'schur'.")
    if solver_mode == 'schur' and lambd <= 0:
        raise ValueError("solver_mode='schur' requires lambd > 0.")

    # TODO: add sample weights
    if sample_weight is not None:
        raise NotImplementedError('Sample weights are not implemented for kernel DWD.')

    K, y = check_real_X_y(K, y, accept_sparse=False,
                     dtype=np.float64 if solver_mode == 'schur' else 'numeric')

    if K.shape[0] != K.shape[1]:
        raise ValueError('K must be square.')
    # Block the symmetry check to avoid another large N-by-N temporary.
    for first in range(0, len(K), 128):
        if not np.allclose(K[first:first+128], K[:, first:first+128].T,
                           rtol=1e-10, atol=1e-12):
            raise ValueError('K must be symmetric.')

    # convert y to +/- 1
    y = pm1(y)  # convert y to  y +/- 1

    if solver_mode == 'schur':
        if not implicit_P:
            raise NotImplementedError('Kernel DWD supports only implicit_P=True.')
        from dwd._kernel_solver import solve_kernel
        result = solve_kernel(K, y, lambd, q=q, K_eig=K_eig,
                              alpha_init=alpha_init, offset_init=offset_init,
                              max_iter=max_iter, obj_tol=obj_tol,
                              acceleration=effective_acceleration, linear_action=linear_action)
        return result['alpha'], result['offset'], result['objective_history'].tolist(), result['C']

    n_samples = K.shape[0]
    M = (q + 1) ** 2 / q

    # precompute data
    if implicit_P:

        # precompute data needed to do implicit P^{-1} gamma
        if K_eig is not None:
            Lam, U = validated_eigh(K, supplied=K_eig)
        else:
            U, Lam = get_K_eig(K)

        # see section 4.1 of Wang and Zou, 2017 for details
        pi = Lam ** 2 + (2 * n_samples * lambd / M) * Lam

        Ucs = U.sum(axis=0)
        if solver_mode == 'legacy':
            # Preserve the released solver's trajectory for comparability.
            ULP = np.multiply(U, Lam * (1.0 / pi))
            v = ULP.dot(U.T.dot(np.ones(n_samples)))
            carson = Ucs.dot(np.multiply(ULP, Lam).dot(Ucs))
            g = 1.0 / (n_samples - carson)
        else:
            # Section 4.1 of Wang and Zou: cancel Lam in Lam/pi, then
            # evaluate the Schur denominator without subtracting near equals.
            shift = 2 * n_samples * lambd / M
            negative_tolerance = 100 * np.finfo(float).eps * n_samples * max(1., np.max(np.abs(Lam)))
            if np.min(Lam) < -negative_tolerance:
                raise ValueError('K must be positive semidefinite, not an indefinite kernel. '
                                 'If precomputed in float32, recompute K from float64 features; '
                                 'casting an already rounded Gram matrix cannot restore precision.')
            # Do not silently truncate positive eigenvalues or add a ridge.
            # On nearly singular kernels the canceled inverse can generate
            # huge non-identifiable coefficients whose quadratic norm loses
            # all accuracy. This is a conservative implementation boundary,
            # not a guarantee of accuracy for every matrix below the bound.
            spectral_scale = max(0., float(np.max(Lam)))
            if (np.min(Lam) <= negative_tolerance and
                    (spectral_scale + shift) / shift > 1.0 / np.sqrt(np.finfo(float).eps)):
                raise FloatingPointError('Rank-deficient kernel with regularization too small for stable '
                                         'coefficient arithmetic. Increase lambd or use a separately '
                                         'validated range-space solver; no kernel was modified.')
            shifted_eigenvalues = Lam + shift
            if np.any(shifted_eigenvalues <= 0):
                raise ValueError('K + (2*n*lambd/M)*I must be positive definite.')
            inverse_shifted = 1.0 / shifted_eigenvalues
            ULP = np.multiply(U, inverse_shifted)
            v = ULP.dot(Ucs)
            g = 1.0 / (shift * np.dot(Ucs * Ucs, inverse_shifted))

    else:
        raise NotImplementedError('Kernel DWD supports only implicit_P=True.')

    # initialize variables
    if alpha_init is None:
        if solver_mode == 'schur':
            alpha = np.zeros(n_samples)
        else:
            alpha = check_random_state(random_state).normal(size=n_samples)
            alpha /= np.linalg.norm(alpha)
    else:
        if contains_complex(alpha_init):
            raise ValueError('alpha_init must be real.')
        alpha = np.asarray(alpha_init, dtype=float).copy()
        if alpha.shape != (n_samples,) or not np.isfinite(alpha).all():
            raise ValueError('alpha_init must be a finite vector with one coefficient per sample.')

    if offset_init is None:
        offset = 0.0
    else:
        if (np.ndim(offset_init) != 0 or contains_complex(offset_init)
                or not np.isfinite(offset_init)):
            raise ValueError('offset_init must be a finite real scalar.')
        offset = float(offset_init)

    # store objective values
    obj_vals = []
    K_alpha = K.dot(alpha)
    prev_obj = kern_dwd_obj(K, y, q, lambd, alpha, offset, K_alpha=K_alpha)
    if not np.isfinite(prev_obj):
        raise FloatingPointError('Kernel DWD produced a nonfinite initial objective.')
    obj_vals.append(prev_obj)

    import warnings
    warnings.warn(
        "solver_mode='legacy' uses historical kernel update equations. "
        "Use solver_mode='schur' for new fits; existing saved models remain readable.",
        FutureWarning, stacklevel=2)

    for i in range(max_iter):

        # get step
        alpha_step, offset_step = \
            get_step_implicit_P(K=K, y=y, q=q, lambd=lambd,
                                alpha=alpha, offset=offset,
                                U=U, Lam=Lam, pi=pi, v=v, g=g,
                                ULP=ULP, K_alpha=K_alpha,
                                solver_mode=solver_mode)

        # update parameters
        alpha = alpha - alpha_step
        offset = offset - offset_step
        K_alpha = K.dot(alpha)

        # stopping condition
        current_obj = kern_dwd_obj(K, y, q, lambd, alpha, offset,
                                   K_alpha=K_alpha)
        if not np.isfinite(current_obj):
            raise FloatingPointError('Kernel DWD produced a nonfinite objective.')
        obj_vals.append(current_obj)

        if np.abs(current_obj - prev_obj) < obj_tol:
            break
        else:
            prev_obj = current_obj

    # tuning paramter for SOCP formulation
    c = c_from_lambd(K, lambd, q, alpha, K_alpha=K_alpha)
    if np.isnan(c):
        raise FloatingPointError('Kernel DWD produced a nonfinite SOCP-parameter conversion.')

    return alpha, offset, obj_vals, c


def get_K_eig(K, *, reference=False):
    """
    Computes the eigendecomposition of K.

    Parameters
    ----------
    K: (n_samples, n_samples)
        Kernel matrix.

    Output
    ------
    U, D

    U: (n_features, n_features)
        Eigenvectors of K.

    D: (n_features, )
        Eigenvalues of K in decending order.
    """

    Lam, U = validated_eigh(K, drivers=('evd', 'evr', 'evx'))
    Lam = Lam[::-1]  # sort evals in decending order
    # A reversed-column view has negative strides. BLAS would copy that entire
    # O(n^2) array at every matrix-vector product; materialize it once here.
    U = np.asfortranarray(U[:, ::-1])

    return U, Lam


def get_step_implicit_P(K, y, q, lambd, alpha, offset, U, Lam, pi, v, g,
                        ULP=None, K_alpha=None, solver_mode='schur'):
    """
    Computes the step size for one step of the MM algorithm.
    See Wang and Zou, 2017 for details.
    """
    n_samples = K.shape[0]

    if K_alpha is None:
        K_alpha = K.dot(alpha)
    z = y * V_grad(y * (K_alpha + offset), q=q) / n_samples

    alice = z + 2 * lambd * alpha

    if solver_mode == 'schur':
        # K v = 1 - shift*v. Cancel shift analytically in the intercept
        # numerator/denominator to avoid subtracting near-equals at tiny lambda.
        factor = n_samples * q / ((q + 1) ** 2)
        shift = 2 * lambd * factor
        h = (v.dot(z) - (1.0 / factor) * (1.0 - shift * v).dot(alpha)) / v.sum()
        if ULP is None:
            ULP = np.multiply(U, 1.0 / (Lam + shift))
        offset_step = factor * h
        alpha_step = factor * ULP.dot(U.T.dot(alice)) - v * offset_step
        return alpha_step, offset_step

    bob = np.insert(-v, 0, values=1)
    projection = z if solver_mode == 'legacy' else v
    bob = (sum(z) - projection.T.dot(K.dot(alice))) * bob

    if ULP is None:
        if solver_mode == 'legacy':
            ULP = np.multiply(U, Lam * (1.0 / pi))
        else:
            ULP = np.multiply(U, 1.0 / (Lam + 2 * n_samples * lambd * q /
                                        ((q + 1) ** 2)))
    cathy = ULP.dot(U.T.dot(alice))
    cathy = np.insert(cathy, 0, 0)

    P_inv_gamma = g * bob + cathy

    step = (n_samples * q / ((q + 1) ** 2)) * P_inv_gamma

    offset_step = step[0]
    alpha_step = step[1:]

    return alpha_step, offset_step


def get_step_explicit_P(K, y, q, lambd, alpha, offset, P_inv):
    """
    Computes the step size for one step of the MM algorithm.
    See Wang and Zou, 2017 for details.
    """
    n_samples = K.shape[0]

    z = y * V_grad(y * (K.dot(alpha) + offset), q=q) / n_samples

    gamma = K.dot(z) + 2 * lambd * K.dot(alpha)
    gamma = np.insert(gamma, 0, z.sum())

    step = (n_samples * q / (q + 1) ** 2) * P_inv @ gamma

    offset_step = step[0]
    alpha_step = step[1:]

    return alpha_step, offset_step


def kern_dwd_obj(K, y, q, lambd, alpha, offset, K_alpha=None):
    """
    Objective function for kernel DWD.
    """
    if K_alpha is None:
        K_alpha = K.dot(alpha)
    return _objective_mean(V(y * (K_alpha + offset), q=q),
                           lambd * alpha.T.dot(K_alpha), inside=True)


def c_from_lambd(K, lambd, q, alpha, K_alpha=None):
    """
    Return the descriptive SOCP C conversion, evaluated in log space.

    It can be zero for a zero fitted norm or infinity when the conversion is
    outside floating-point range. Neither changes the supplied lambd or fit.
    from lambda.
    """
    if K_alpha is None:
        K_alpha = K.dot(alpha)
    if not np.isfinite(q) or q <= 0:
        raise ValueError('q must be finite and positive.')
    norm_squared = float(alpha.T.dot(K_alpha))
    if not np.isfinite(norm_squared) or norm_squared < 0:
        raise FloatingPointError('Kernel DWD has a nonfinite or negative fitted RKHS norm squared.')
    if norm_squared == 0:
        return 0.
    with np.errstate(over='ignore', under='ignore'):
        log_c = (np.log1p(q) + q * np.log1p(1. / q)
                 + .5 * (q + 1.) * np.log(norm_squared))
        return float(np.exp(log_c))
