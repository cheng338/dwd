"""Exact full-kernel generalized DWD solvers with a free intercept.

The objective is mean(V_q(y * (K @ alpha + b))) + lambd * alpha.T @ K @ alpha.
No positive eigenvalues are discarded and no intercept penalty is introduced.
This private implementation is shared by the public sklearn estimator.
"""
from time import perf_counter
from types import MappingProxyType
from math import fsum

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, LinAlgError
from scipy.linalg.lapack import dpocon
from scipy.optimize import minimize

from .gen_dwd import V, V_grad
from ._eigen import validated_eigh
from ._kernel_linear_system import KernelLinearSystem
from ._reference_mm import reference_update
from ._spectral_linear_system import SpectralLinearSystem
from ._quadratic_bounds import _upward_nonnegative, _compensated_dot, _compensated_quadratic
from ._accelerated_mm import RestartedMM
from ._kernel_scores import compensated_kernel_matvec, adaptive_kernel_matvec
from ._kernel_mm_recovery import KernelMMRecovery


def _scalar(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0
            or not np.issubdtype(np.asarray(value).dtype, np.number)
            or np.iscomplexobj(value)):
        raise ValueError(f'{name} must be a finite real scalar.')
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        qualifier = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{name} must be finite and {qualifier}.')
    return value


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(value)


def _matrix(K):
    if np.iscomplexobj(K):
        raise ValueError('K must be real.')
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or len(K) < 2:
        raise ValueError('K must be square with at least two samples.')
    scale, symmetry, norm_bound = 0., 0., 0.
    # Bounded temporary arrays, including for large dense kernels.
    for start in range(0, len(K), 128):
        block = K[start:start + 128]
        if not np.isfinite(block).all():
            raise ValueError('K must contain only finite values.')
        scale = max(scale, float(np.max(np.abs(block))))
        norm_bound = max(norm_bound, float(np.max(np.sum(np.abs(block), axis=1))))
        symmetry = max(symmetry, float(np.max(np.abs(block - K[:, start:start + 128].T))))
    if symmetry > 100 * np.finfo(float).eps * max(1., scale):
        raise ValueError('K must be symmetric.')
    if not np.isfinite(norm_bound):
        raise FloatingPointError('Kernel scale exceeds finite arithmetic.')
    return K, symmetry, norm_bound


def _check_spectrum(values, n, shift):
    """Apply the same PSD/rank safeguards to full or eigenvalues-only results."""
    if np.shape(values) != (n,) or not np.isfinite(values).all():
        raise FloatingPointError('Invalid kernel eigenvalues from spectral preparation.')
    limit = 100 * np.finfo(float).eps * n * max(1., float(np.max(np.abs(values))))
    if values.min() < -limit:
        raise ValueError('K must be positive semidefinite. Recompute a rounded '
                         'float32 Gram matrix from float64 features if necessary.')
    return float(limit)


def _eigensystem(K, supplied, shift, *, reference=False):
    n = len(K)
    (values, vectors), eigen_info = validated_eigh(
        K, supplied=supplied, return_info=True,
        drivers=('evd', 'evr', 'evx'))
    limit = _check_spectrum(values, n, shift)
    # Only negative floating-point roundoff is made inactive. Every strictly
    # positive eigendirection, however small, remains part of the objective.
    return np.ascontiguousarray(vectors), np.maximum(values, 0.), {
        'min_input_eigenvalue': float(values.min()),
        'positive_eigenvalues_retained': int(np.count_nonzero(values > 0)),
        'negative_roundoff_eigenvalues_clipped': int(np.count_nonzero(values < 0)),
        'psd_tolerance': float(limit),
        'eigendecomposition_supplied': supplied is not None,
        'eigendecomposition_driver': eigen_info['accepted_driver'],
        'eigendecomposition_validation': eigen_info,
    }


def _nonnegative_quadratic(value, scale, name):
    if not np.isfinite(value):
        raise FloatingPointError(f'Nonfinite {name}.')
    if value < -100 * np.finfo(float).eps * max(1., scale):
        raise FloatingPointError(f'Negative {name}; the kernel or coefficient '
                                 'arithmetic is not numerically reliable.')
    return max(0., float(value))


def _check_initial_quadratic(K, alpha):
    """Check and return the actual initial scores, quadratic and error budget.

    Keep the ordinary length-n reduction bound when it passes. Otherwise a
    compensated calculation can show that this bound was overconservative.
    Acceptance then bounds the ACTUAL observed quadratic by its discrepancy
    from the compensated result plus that result's forward-error bound.
    The caller reuses the observed scores and quadratic: no initial coefficient
    is changed, and a bound for a different calculation is never substituted.
    Huge cancellation-dominated states remain outside this precision boundary.
    """
    started = perf_counter()
    details = {'initial_quadratic_method': 'zero',
               'initial_quadratic_fast_error_bound': 0.,
               'initial_quadratic_acceptance_threshold': float(np.sqrt(np.finfo(float).eps)),
               'initial_quadratic_compensated_error_bound': None,
               'initial_quadratic_compensated_discrepancy': None,
               'initial_compensated_score_max_difference': None}
    if not np.any(alpha):
        details['initial_state_check_seconds'] = perf_counter() - started
        return np.zeros(len(alpha)), 0., 0., details
    magnitude = np.abs(alpha)
    absolute_product = 0.
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        for start in range(0, len(K), 128):
            absolute_product += float(magnitude[start:start + 128]
                                      @ (np.abs(K[start:start + 128]) @ magnitude))
        scores = K @ alpha
        computed = float(alpha @ scores)
    eps = np.finfo(float).eps
    neps = len(K) * eps
    gamma = neps / (1. - neps)
    bound = (2 * gamma + gamma * gamma) * absolute_product
    threshold = np.sqrt(eps) * max(1., abs(computed))
    if not np.isfinite(computed) or not np.isfinite(bound) or not np.isfinite(scores).all():
        raise FloatingPointError('Nonfinite initial coefficients, scores or quadratic error bound.')
    details['initial_quadratic_method'] = 'ordinary_forward_bound'
    details['initial_quadratic_fast_error_bound'] = float(bound)
    details['initial_quadratic_acceptance_threshold'] = float(threshold)
    if bound > threshold:
        compensated_scores, compensated, compensated_bound = _compensated_quadratic(K, alpha)
        discrepancy = _upward_nonnegative(abs(computed - compensated))
        bound = _upward_nonnegative(discrepancy + compensated_bound)
        with np.errstate(over='ignore', invalid='ignore'):
            score_difference = float(np.max(np.abs(scores - compensated_scores)))
        if not np.isfinite(score_difference):
            raise FloatingPointError('Nonfinite initial score discrepancy in compensated checking.')
        details.update(initial_quadratic_method='compensated_observed_check',
                       initial_quadratic_compensated_error_bound=compensated_bound,
                       initial_quadratic_compensated_discrepancy=discrepancy,
                       initial_compensated_score_max_difference=score_difference)
        if bound > threshold:
            raise FloatingPointError('Initial coefficients are too cancellation-prone '
                                     'for a reliable RKHS norm after compensated checking; '
                                     'use a numerically stable explicit initial state.')
    details['initial_state_check_seconds'] = perf_counter() - started
    return scores, computed, float(bound), details


def optimality_diagnostics(K, y, alpha, offset, lambd, q, *, scores=None):
    """Return RKHS/intercept residual and a feasible primal-dual bound.

    The dual loss conjugate is -rho**(q/(q+1)) on 0 <= rho <= 1, with
    y.T @ rho = 0 from the unregularized intercept. Dual slopes are projected
    onto that equality before computing the bound. K must already be validated.
    """
    n = len(y)
    scores = K @ alpha if scores is None else scores
    margins = y * (scores + offset)
    norm2 = _nonnegative_quadratic(float(alpha @ scores),
                                  float(np.sum(np.abs(alpha * scores))), 'RKHS norm squared')
    primal = float(np.mean(V(margins, q=q)) + lambd * norm2)
    z = y * V_grad(margins, q=q) / n
    coefficient_residual = z + 2 * lambd * alpha
    gradient_alpha = K @ z + 2 * lambd * scores
    residual2 = _nonnegative_quadratic(float(coefficient_residual @ gradient_alpha),
                                      float(np.sum(np.abs(coefficient_residual * gradient_alpha))),
                                      'RKHS gradient norm squared')
    gradient = float(max(abs(z.sum()), np.sqrt(residual2)))
    rho = -V_grad(margins, q=q)
    lo, hi = -1., 1.
    for _ in range(64):
        mid = (lo + hi) * .5
        projected = np.clip(rho - mid * y, 0., 1.)
        if y @ projected > 0:
            lo = mid
        else:
            hi = mid
    rho = np.clip(rho - ((lo + hi) * .5) * y, 0., 1.)
    signed = rho * y
    dual = float(np.mean(rho ** (q / (q + 1.)))
                 - signed @ (K @ signed) / (4. * lambd * n * n))
    gap = primal - dual
    if not np.isfinite(primal) or not np.isfinite(dual) or not np.isfinite(gradient):
        raise FloatingPointError('Nonfinite objective or optimality diagnostic.')
    # Tiny negative gaps may arise from floating-point summation. Keep the raw
    # value visible; never hide a materially negative gap by clipping it.
    if gap < -1e-8 * max(1., abs(primal), abs(dual)):
        raise FloatingPointError('Inconsistent primal-dual bound; check the kernel '
                                 'and any explicitly supplied eigensystem.')
    if norm2 == 0:
        C = 0.
    else:
        # Equivalent to ((q+1)**(q+1)/q**q)*norm**(q+1), without huge
        # intermediate powers. C is descriptive, not the penalty used here.
        log_C = np.log(q + 1.) + q * np.log1p(1. / q) + .5 * (q + 1.) * np.log(norm2)
        C = float(np.exp(log_C)) if log_C <= np.log(np.finfo(float).max) else float('inf')
    return {'final_objective': primal, 'rkhs_norm_squared': norm2,
            'gradient_inf_norm': float(max(abs(z.sum()), np.max(np.abs(gradient_alpha)))),
            'rkhs_gradient_norm': gradient, 'intercept_gradient': float(z.sum()),
            'dual_objective': dual, 'dual_gap': float(gap),
            'dual_equality_residual': float(abs(y @ rho)), 'C': C}


class _RequestedStop(Exception):
    pass


def solve_kernel(K, y, lambd, q=1, *, K_eig=None, alpha_init=None,
                 offset_init=None, max_iter=100, obj_tol=1e-5,
                 stopping='objective', tol=1e-6, backend='auto',
                 psd_known=False, callback=None, validation=None,
                 patience=3, min_delta=0., check_interval=1,
                 implementation='optimized', acceleration=None):
    """Solve one specified kernel DWD problem; no internal parameter search.

    y contains both -1/+1. ``validation`` is an explicitly supplied pair
    (query-by-training kernel, signed labels). ``psd_known`` is private and
    only for internally constructed kernels of a known PSD family.
    Numerical stopping checks max(RKHS gradient 2-norm, |intercept gradient|)
    against ``tol``. The default remains the upstream absolute objective rule.
    """
    started = perf_counter()
    if implementation not in ('optimized', 'reference'):
        raise ValueError("implementation must be 'optimized' or 'reference'.")
    if implementation == 'reference' and backend not in ('auto', 'spectral'):
        raise ValueError('The coefficient/eigenbasis reference uses backend=auto or spectral.')
    if acceleration is not None and (not isinstance(acceleration, str) or acceleration != 'restart'):
        raise ValueError("acceleration must be None or 'restart'.")
    if acceleration is not None and (implementation != 'optimized' or backend == 'lbfgs'):
        raise ValueError('Acceleration is supported only for optimized MM, not reference or L-BFGS.')
    lambd, q = _scalar(lambd, 'lambd', positive=True), _scalar(q, 'q', positive=True)
    obj_tol, tol = _scalar(obj_tol, 'obj_tol'), _scalar(tol, 'tol', positive=True)
    max_iter = _integer(max_iter, 'max_iter')
    patience, check_interval = _integer(patience, 'patience', 1), _integer(check_interval, 'check_interval', 1)
    min_delta = _scalar(min_delta, 'min_delta')
    if backend not in ('auto', 'spectral', 'cholesky', 'lbfgs'):
        raise ValueError('backend must be auto, spectral, cholesky or lbfgs.')
    if stopping not in ('objective', 'fixed', 'optimality', 'validation'):
        raise ValueError('stopping must be objective, fixed, optimality or validation.')
    if callback is not None and not callable(callback):
        raise ValueError('callback must be callable or None.')
    if stopping == 'validation' and validation is None:
        raise ValueError('Validation stopping requires explicit validation_data.')
    if validation is not None and stopping != 'validation':
        raise ValueError('Provide validation_data only with stopping="validation".')
    K, symmetry, norm_bound = _matrix(K)
    n = len(K)
    y = np.asarray(y, dtype=float)
    if y.shape != (n,) or not np.isfinite(y).all() or set(np.unique(y)) != {-1., 1.}:
        raise ValueError('y must match K and contain both -1 and +1.')
    if alpha_init is not None and np.iscomplexobj(alpha_init):
        raise ValueError('alpha_init must be real.')
    alpha = np.zeros(n) if alpha_init is None else np.asarray(alpha_init, dtype=float).copy()
    if alpha.shape != (n,) or not np.isfinite(alpha).all():
        raise ValueError('alpha_init must be a finite vector of length n.')
    initial_scores, initial_quadratic, initial_quadratic_bound, initial_details = (
        _check_initial_quadratic(K, alpha))
    if offset_init is None:
        offset = 0.
    elif np.ndim(offset_init) != 0 or np.iscomplexobj(offset_init) or not np.isfinite(offset_init):
        raise ValueError('offset_init must be a finite real scalar.')
    else:
        offset = float(offset_init)
    if validation is not None:
        val_K, val_y = np.asarray(validation[0], float), np.asarray(validation[1], float)
        if (val_K.ndim != 2 or val_K.shape[1] != n or not len(val_K)
                or val_y.shape != (len(val_K),) or not np.isfinite(val_K).all()
                or not np.isfinite(val_y).all() or not np.isin(val_y, [-1., 1.]).all()):
            raise ValueError('Validation kernel/labels must be finite query-by-training data.')
    # n/M = n*q/(q+1)^2, evaluated without squaring a huge q.
    step_scale = n * (q / (q + 1.)) / (q + 1.)
    shift = 2 * lambd * step_scale
    if not np.isfinite(shift) or shift <= 0 or not np.isfinite(step_scale) or step_scale <= 0:
        raise ValueError('q and lambd produce an unrepresentable MM shift.')
    requested_backend = backend
    if backend == 'auto':
        backend = 'spectral' if K_eig is not None or implementation == 'reference' else 'cholesky'
    diagnostics = {'symmetry_error': symmetry, 'shift': float(shift),
                   'initial_quadratic_error_bound': initial_quadratic_bound,
                   'positive_eigenvalues_discarded': 0, 'kernel_approximation_used': False,
                   'requested_backend': requested_backend, 'initial_backend': backend,
                   'attempted_backends': [backend], 'auto_fallback': False,
                   'fallback_reason': None, 'cholesky_setup_seconds': 0.,
                   'spectral_setup_seconds': 0., 'eigenvalue_check_seconds': 0.,
                   'implementation': implementation, 'acceleration': acceleration}
    diagnostics.update(initial_details)
    U = values = system = None
    if backend in ('spectral', 'lbfgs') or K_eig is not None:
        spectral_started = perf_counter()
        U, values, details = _eigensystem(K, K_eig, shift, reference=implementation == 'reference')
        diagnostics['spectral_setup_seconds'] = perf_counter() - spectral_started
        diagnostics.update(details)
        diagnostics['psd_validation'] = 'validated eigensystem'
    elif not psd_known:
        check_started = perf_counter()
        check_values, eigen_info = validated_eigh(K, eigvals_only=True, return_info=True)
        _check_spectrum(check_values, n, shift)
        diagnostics['eigenvalue_check_seconds'] = perf_counter() - check_started
        diagnostics['psd_validation'] = 'checked eigenvalues of supplied kernel'
        diagnostics['eigenvalue_validation'] = eigen_info
    else:
        diagnostics['psd_validation'] = 'internally constructed PSD kernel (known family)'
    if implementation == 'reference' or (acceleration is not None and backend == 'spectral'):
        system = SpectralLinearSystem(K, shift, U, values)
        scores = initial_scores
        c = None
    elif backend == 'cholesky':
        cholesky_started = perf_counter()
        system = KernelLinearSystem(K, shift)
        scores = initial_scores
        c = None
        diagnostics['cholesky_setup_seconds'] = perf_counter() - cholesky_started
    else:
        spectral_started = perf_counter()
        c = U.T @ alpha
        # Preserve the exact supplied step-zero state; a checked eigenbasis
        # still has roundoff and does not authorize changing initialization.
        scores = initial_scores
        inverse = 1. / (values + shift)
        h = U.T @ np.ones(n)
        vh = inverse * h
        denominator = float(h @ vh)
        if not np.isfinite(denominator) or denominator <= 0:
            raise FloatingPointError('Invalid free-intercept Schur denominator.')
        diagnostics['spectral_setup_seconds'] += perf_counter() - spectral_started
    diagnostics['effective_backend'] = backend
    setup_seconds = perf_counter() - started
    history, validation_history = [], []
    best_state, best_score, stale = None, -np.inf, 0
    iteration, reason = 0, 'max_iter'
    prediction_precision = 'adaptive'
    state_uses_certified_columns = False

    def reconstructed_scores(coefficients, expected, error_message, *, force_compensated=False):
        """Verify original-K scores, recovering evaluation rather than waiving it.

        Use the model's actual public evaluation policy for reconstruction.
        A fully expanded retry must meet the SAME threshold, and its mode is
        retained for subsequent public queries. Expected optimizer scores are
        never substituted or overwritten by either evaluation.
        """
        nonlocal prediction_precision
        adaptive_drift = None
        if prediction_precision == 'adaptive' and not force_compensated:
            adaptive_scores = adaptive_kernel_matvec(K, coefficients)
            adaptive_drift = float(np.max(np.abs(adaptive_scores - expected)))
            if (np.isfinite(adaptive_drift) and adaptive_drift <= 5e-7 * max(
                    1., float(np.max(np.abs(adaptive_scores))))):
                return adaptive_scores, adaptive_drift
        checked = compensated_kernel_matvec(K, coefficients)
        drift = float(np.max(np.abs(checked - expected)))
        if (not np.isfinite(drift)
                or drift > 5e-7 * max(1., float(np.max(np.abs(checked))))):
            raise FloatingPointError(error_message)
        prediction_precision = 'compensated'
        diagnostics['compensated_score_reconstructions'] = diagnostics.get(
            'compensated_score_reconstructions', 0) + 1
        if adaptive_drift is not None:
            diagnostics['max_adaptive_score_reconstruction_error_before_retry'] = max(
                diagnostics.get('max_adaptive_score_reconstruction_error_before_retry', 0.), adaptive_drift)
        return checked, drift

    def reconstructed_state(coefficients, expected, b, expected_value,
                            score_error, objective_error):
        """Check both original score and objective accuracy before exposure.

        The objective threshold is stricter than the score threshold. A score
        check that passes therefore does not suppress an evaluation retry when
        the original objective calculation still fails its own requirement.
        Neither retry overwrites the expected optimizer state.
        """
        checked, drift = reconstructed_scores(coefficients, expected, score_error)
        for attempt in range(2):
            failure = None
            try:
                norm = _nonnegative_quadratic(float(coefficients @ checked),
                    float(np.sum(np.abs(coefficients * checked))), 'original RKHS norm squared')
                value = float(np.mean(V(y * (checked + b), q=q)) + lambd * norm)
                objective_drift = float(value - expected_value)
                good = (np.isfinite(value) and np.isfinite(objective_drift)
                        and abs(objective_drift) <= 5e-8 * max(1., abs(value)))
            except FloatingPointError as exc:
                failure = str(exc)
                good = False
            if good:
                return checked, drift, value, objective_drift
            if attempt == 1 or prediction_precision == 'compensated':
                raise FloatingPointError(objective_error + (f' {failure}' if failure else ''))
            checked, drift = reconstructed_scores(coefficients, expected, score_error,
                                                    force_compensated=True)
            diagnostics['objective_evaluation_retries'] = diagnostics.get(
                'objective_evaluation_retries', 0) + 1

    def objective():
        norm2 = (initial_quadratic if iteration == 0 else
                 float(c @ (values * c)) if c is not None else float(alpha @ scores))
        value = float(np.mean(V(y * (scores + offset), q=q)) + lambd * norm2)
        if not np.isfinite(value) or not np.isfinite(offset) or not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite kernel DWD objective or state.')
        return value

    def current_alpha():
        return U @ c if c is not None and iteration > 0 else alpha

    def residual():
        z = y * V_grad(y * (scores + offset), q=q) / n
        if c is not None:
            feature_grad = np.sqrt(values) * (U.T @ z + 2 * lambd * c)
            norm = float(np.linalg.norm(feature_grad))
        else:
            a = z + 2 * lambd * alpha
            Ka = K @ z + 2 * lambd * scores
            norm = np.sqrt(_nonnegative_quadratic(float(a @ Ka),
                            float(np.sum(np.abs(a * Ka))), 'RKHS gradient norm squared'))
        return max(abs(float(z.sum())), norm)

    def record():
        nonlocal best_state, best_score, stale, reason
        value = objective()
        validation_due = validation is not None and (iteration > 0 or max_iter == 0) and (
            iteration == 1 or iteration % check_interval == 0 or iteration == max_iter)
        observed_alpha = None
        check_spectral_observation = (c is not None and iteration > 0
                                      and (callback is not None or validation_due))
        if check_spectral_observation or validation_due:
            # Eigen-coordinate scores and norms are inexpensive for unobserved
            # iterations, but they cannot stand in for an unchecked original-K
            # model when a user or validation rule observes a checkpoint.
            # Reject inconsistency; changing scores alone would leave c and the
            # optimized objective in a different numerical representation.
            observed_alpha = current_alpha()
            original_scores, drift, original_value, signed_objective_drift = reconstructed_state(
                observed_alpha, scores, offset, value,
                'Observed kernel checkpoint lost original-kernel score accuracy; '
                'no callback or validation received the inconsistent state.',
                'Observed kernel checkpoint lost original RKHS objective accuracy; '
                'no callback or validation received the inconsistent state.')
            objective_drift = abs(signed_objective_drift)
            diagnostics['observed_checkpoint_checks'] = diagnostics.get('observed_checkpoint_checks', 0) + 1
            diagnostics['observed_checks_are_interval_certificates'] = False
            diagnostics['max_observed_score_discrepancy'] = max(
                diagnostics.get('max_observed_score_discrepancy', 0.), drift)
            diagnostics['max_observed_objective_discrepancy'] = max(
                diagnostics.get('max_observed_objective_discrepancy', 0.), objective_drift)
        history.append(value)
        validation_stop = False
        if validation_due:
            coefficients = current_alpha() if observed_alpha is None else observed_alpha
            validation_scores = (compensated_kernel_matvec(val_K, coefficients)
                                 if prediction_precision == 'compensated'
                                 else adaptive_kernel_matvec(val_K, coefficients))
            score = float(np.mean((validation_scores + offset > 0) == (val_y > 0)))
            validation_history.append({'iteration': iteration, 'score': score,
                                       'prediction_precision': prediction_precision})
            if best_state is None or score > best_score + min_delta:
                best_score, stale = score, 0
                best_state = (coefficients.copy(), float(offset), scores.copy(), value,
                              iteration, prediction_precision, state_uses_certified_columns)
            else:
                stale += 1
            if stale >= patience:
                validation_stop = True
        if callback is not None:
            coefficients = (current_alpha() if observed_alpha is None else observed_alpha).copy()
            callback_scores = scores.copy()
            coefficients.flags.writeable = callback_scores.flags.writeable = False
            decision_values = scores + offset
            decision_values.flags.writeable = False
            state = MappingProxyType({'iteration': iteration, 'alpha': coefficients,
                'offset': float(offset), 'intercept': float(offset), 'objective': value,
                'training_scores': callback_scores, 'decision_values': decision_values,
                'elapsed_seconds': perf_counter() - started})
            try:
                requested = callback(state)
            except StopIteration:
                requested = True
            if requested is not None and not isinstance(requested, (bool, np.bool_)):
                raise ValueError('callback must return True to stop, False or None to continue.')
            if requested:
                reason = 'callback_stop'
                raise _RequestedStop()
        if validation_stop:
            reason = 'validation_patience'
            raise _RequestedStop()
        if stopping == 'optimality' and residual() <= tol:
            reason = 'optimality_tolerance'
            raise _RequestedStop()
        if iteration > 0 and stopping == 'objective' and abs(history[-1] - history[-2]) < obj_tol:
            reason = 'objective_tolerance'
            raise _RequestedStop()

    optimizer_details = {}
    mm_recovery = KernelMMRecovery(K, shift)
    last_proximal_uses_certified_columns = False

    def accelerated_proximal_step(rhs):
        nonlocal last_proximal_uses_certified_columns
        # rhs already includes the extrapolated intercept. A certified action
        # therefore checks and returns its absolute offset with old_offset=0.
        if mm_recovery.active:
            result = mm_recovery.step(rhs, 0.)
            last_proximal_uses_certified_columns = True
            return result
        try:
            coefficients, b = system.solve_constrained(rhs)
            last_proximal_uses_certified_columns = False
            return coefficients, b, system.last_product.copy()
        except FloatingPointError as exc:
            result = mm_recovery.step(rhs, 0., str(exc))
            last_proximal_uses_certified_columns = True
            return result

    def check_mm_descent():
        # A nonfinite or non-descent state cannot be convergence.
        candidate_objective = objective()
        if candidate_objective > history[-1] + 1e-9 * max(1., abs(history[-1])):
            raise FloatingPointError('Kernel MM update increased the original objective; '
                                     'numerical accuracy could not be established.')

    accelerator = (RestartedMM(system, y, lambd, q, step_scale,
                              proximal_step=accelerated_proximal_step)
                   if acceleration is not None else None)
    try:
        record()
        if backend == 'lbfgs' and max_iter > 0:
            sqrt_values = np.sqrt(values)
            inverse_sqrt = np.zeros_like(values)
            np.divide(1., sqrt_values, out=inverse_sqrt, where=values > 0)
            # Unidentifiable zero-eigenvalue directions have no effect on h.
            initial = np.r_[offset, sqrt_values * c]

            def value_gradient(theta):
                b, w = theta[0], theta[1:]
                g = U @ (sqrt_values * w)
                z = y * V_grad(y * (g + b), q=q) / n
                value = float(np.mean(V(y * (g + b), q=q)) + lambd * (w @ w))
                gradient = np.r_[z.sum(), sqrt_values * (U.T @ z) + 2 * lambd * w]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('Nonfinite L-BFGS objective or gradient.')
                return value, gradient

            def accepted(theta):
                nonlocal c, offset, scores, iteration
                offset, c = float(theta[0]), inverse_sqrt * theta[1:]
                scores = U @ (sqrt_values * theta[1:])
                iteration += 1
                record()

            result = minimize(value_gradient, initial, jac=True, method='L-BFGS-B',
                callback=accepted, options={'maxiter': max_iter, 'maxfun': 51 * max_iter + 1,
                                            'gtol': 0., 'ftol': 0., 'maxls': 50, 'maxcor': 20})
            optimizer_details = {'optimizer_success': bool(result.success),
                                 'optimizer_message': str(result.message),
                                 'function_evaluations': int(result.nfev)}
            if iteration < max_iter:
                reason = 'optimizer_stationary' if residual() <= tol else 'optimizer_stagnation'
        else:
            for iteration in range(1, max_iter + 1):
                if accelerator is not None:
                    alpha, offset, scores = accelerator.step(scores, offset, history[-1])
                    state_uses_certified_columns = last_proximal_uses_certified_columns
                    check_mm_descent()
                else:
                    z = y * V_grad(y * (scores + offset), q=q) / n
                    previous_state = alpha, c, scores, offset, state_uses_certified_columns
                    candidate_rhs, previous_offset = scores - step_scale * z, offset
                    was_recovering = mm_recovery.active
                    try:
                        if was_recovering:
                            alpha, offset, scores = mm_recovery.step(candidate_rhs, previous_offset)
                            c = None
                            state_uses_certified_columns = True
                        elif implementation == 'reference':
                            candidate_alpha, candidate_offset, candidate_scores = reference_update(
                                K, alpha, offset, z, lambd, step_scale, system)
                            alpha, offset_change, scores = system.refine_candidate(
                                candidate_rhs, candidate_alpha, candidate_offset - previous_offset,
                                candidate_scores)
                            offset = previous_offset + offset_change
                        elif backend == 'cholesky':
                            alpha, s = system.solve_constrained(candidate_rhs)
                            # The residual check already computed the original scores.
                            scores = system.last_product.copy()
                            offset += s
                        else:
                            a = values * c - step_scale * (U.T @ z)
                            s = float(vh @ a / denominator)
                            c = inverse * (a - h * s)
                            scores = U @ (values * c)
                            offset += s
                        check_mm_descent()
                    except FloatingPointError as exc:
                        # All ordinary proposals allocate new state arrays.
                        # Restore the untouched previous state before retrying
                        # the SAME MM update in a certified representation.
                        alpha, c, scores, offset, state_uses_certified_columns = previous_state
                        if was_recovering:
                            raise
                        alpha, offset, scores = mm_recovery.step(
                            candidate_rhs, previous_offset, str(exc))
                        c = None
                        state_uses_certified_columns = True
                        check_mm_descent()
                # Observation is outside the recovery transaction: a callback,
                # validation error or stop must never be swallowed or replayed.
                record()
    except _RequestedStop:
        pass
    optimization_seconds = perf_counter() - started - setup_seconds
    alpha = current_alpha()
    returned_iteration = iteration
    if validation is not None and best_state is not None:
        (alpha, offset, scores, _, returned_iteration, prediction_precision,
         state_uses_certified_columns) = best_state
    exact_scores, score_drift, _, _ = reconstructed_state(
        alpha, scores, offset, history[returned_iteration],
        'Kernel coefficient conversion lost score accuracy; '
        'check regularization, kernel precision or supplied eigenpairs.',
        'Returned coefficients lost RKHS objective accuracy; '
        'check the kernel, initialization or supplied eigenpairs.')
    if system is not None:
        diagnostics['linear_system_diagnostics'] = dict(system.info)
        diagnostics['auto_fallback'] = bool(system.info['linear_recoveries'])
        diagnostics['fallback_reason'] = ('original_system_residual_or_factorization' 
                                          if diagnostics['auto_fallback'] else None)
    if mm_recovery.info['attempted']:
        diagnostics['mm_function_recovery'] = dict(mm_recovery.info)
        diagnostics['returned_state_uses_certified_columns'] = state_uses_certified_columns
        if mm_recovery.info['accepted_actions']:
            diagnostics['auto_fallback'] = True
            diagnostics['fallback_reason'] = 'certified_original_kernel_mm_function'
        if state_uses_certified_columns:
            diagnostics['coefficient_representation'] = 'certified_sparse_kernel_columns'
    final = optimality_diagnostics(K, y, alpha, offset, lambd, q, scores=exact_scores)
    objective_drift = float(final['final_objective'] - history[returned_iteration])
    if abs(objective_drift) > 5e-8 * max(1., abs(final['final_objective'])):
        raise FloatingPointError('Returned coefficients lost RKHS objective accuracy; '
                                 'check the kernel, initialization or supplied eigenpairs.')
    diagnostics.update(optimizer_details)
    if accelerator is not None:
        diagnostics.update(accelerator.info)
    diagnostics.update({'setup_seconds': setup_seconds, 'optimization_seconds': optimization_seconds,
                        'total_seconds': perf_counter() - started,
                        'score_reconstruction_max_error': score_drift,
                        'prediction_precision': prediction_precision,
                        'objective_reconstruction_error': objective_drift,
                        'optimality_criterion': 'max(RKHS gradient L2, absolute intercept gradient)',
                        'optimality_tolerance': tol})
    result = dict(final, alpha=np.asarray(alpha), offset=float(offset),
                  objective_history=np.asarray(history), n_iter=int(iteration),
                  returned_iteration=int(returned_iteration), termination_reason=reason,
                  converged=bool(final['rkhs_gradient_norm'] <= tol), backend=backend,
                  prediction_precision=prediction_precision,
                  validation_history=validation_history, diagnostics=diagnostics,
                  objective_tolerance_met=bool(iteration > 0 and abs(history[-1] - history[-2]) < obj_tol))
    return result
