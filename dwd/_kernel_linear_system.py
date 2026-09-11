"""Verified solves of the original, free-intercept kernel MM equations.

Factor conditioning is diagnostic information, not an automatic rejection.
Recovery/refinement changes the computation, never K, lambda or the objective.
Residual bounds reported here are floating-point estimates, not interval proofs.
"""
import math
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.linalg.lapack import dpocon
from ._kernel_recovery import ConstrainedSolveFailure
from ._compensated_residual import compensated_residual
from ._native_residual import native_compensated_residual
from ._quadratic_bounds import _compensated_dot, _compensated_quadratic, _upward_nonnegative


def _sum(x):
    try:
        result = math.fsum(np.asarray(x, dtype=float).ravel())
    except (ValueError, OverflowError) as exc:
        raise FloatingPointError('Nonfinite or overflowing kernel-system summation.') from exc
    if not np.isfinite(result):
        raise FloatingPointError('Nonfinite kernel-system summation.')
    return result


class KernelLinearSystem:
    """Solve (K+shift I)x + s 1 = rhs, with sum(x)=target_sum.

    The second factorization acts on the centered kernel and a lifted constant
    direction. It is only a means of solving the SAME original equations.
    If those float64 candidates fail, a lazy anchor-coordinate LU factor solves
    the same constrained system after eliminating one coefficient. No kernel
    direction is removed from the fitted function by that coordinate change.
    Every candidate is checked against those original equations before return.
    """
    def __init__(self, K, shift):
        self.K = K
        self.shift = float(shift)
        self.n = len(K)
        self.factor = None
        self.anchor_action = None
        self.mode = None
        self.v = None
        self.mean = self.K.mean(axis=0)
        self.last_product = None
        self.info = {
            'linear_system': 'original free-intercept constrained MM system',
            'factorization_attempts': [], 'linear_solves': 0,
            'refinement_steps': 0, 'linear_recoveries': 0,
            'max_linear_residual': 0., 'max_constraint_residual': 0.,
            'max_estimated_rkhs_solve_error': 0.,
            'added_objective_regularization': 0.,
            'positive_eigenvalues_discarded': 0,
            'residual_checks_are_interval_certificates': False,
        }
        initial_failed = False
        try:
            self._prepare('original')
        except (LinAlgError, FloatingPointError) as exc:
            self.info['initial_factorization_error'] = str(exc)
            initial_failed = True
        if initial_failed:
            # Leave the except block first: its traceback can retain the old
            # dense array even after references on self are cleared.
            self.factor = self.v = None
            try:
                self._prepare('centered')
            except (LinAlgError, FloatingPointError) as exc:
                self.info['centered_factorization_error'] = str(exc)
            if self.mode != 'centered':
                self.factor = self.v = None
                self._prepare('anchor')
            self.info['linear_recoveries'] += 1

    def _prepare(self, mode):
        started = perf_counter()
        if mode == 'anchor':
            # Eliminate the coefficient-sum constraint exactly in real
            # arithmetic. LU retains stored nonsymmetric rounding rather than
            # silently symmetrizing a nearly constant kernel.
            from ._anchor_linear_system import AnchorLinearSystem
            self.anchor_action = AnchorLinearSystem(self.K, self.shift)
            self.mode = mode
            self.info['factorization_attempts'].append({**self.anchor_action.info,
                'seconds': self.anchor_action.info['factorization_seconds']})
            self.info.update(factor_representation='anchor_lu', anchor_index=0,
                             anchor_accepted_actions=0)
            return
        n = self.n
        a = np.array(self.K, dtype=float, order='F', copy=True)
        if mode == 'centered':
            a -= self.mean[None, :]
            a -= self.mean[:, None]
            a += float(self.mean.mean())
            # The artificial constant eigenvalue is inactive under the
            # constraint. It is NOT jitter applied to the fitted kernel.
            lift = max(self.shift, float(np.max(np.abs(a.diagonal()))), 1.)
            a += (lift - self.shift) / n
        a.flat[::n + 1] += self.shift
        if not np.isfinite(a).all():
            raise FloatingPointError('Nonfinite kernel linear-system preparation.')
        norm = float(np.max(np.sum(np.abs(a), axis=0)))
        factor = cho_factor(a, lower=True, overwrite_a=True, check_finite=False)
        rcond, status = dpocon(factor[0], norm, uplo='L')
        if status != 0 or not np.isfinite(rcond) or rcond < 0:
            raise FloatingPointError('Invalid Cholesky reciprocal-condition estimate '
                                     f'(LAPACK info={status}, rcond={rcond}).')
        self.factor, self.mode = factor, mode
        if mode == 'original':
            self.v = cho_solve(factor, np.ones(n), check_finite=False)
            self.denominator = _sum(self.v)
            if (not np.isfinite(self.v).all() or not np.isfinite(self.denominator)
                    or self.denominator <= 0):
                raise FloatingPointError('Invalid free-intercept Schur denominator.')
        self.info['factorization_attempts'].append({
            'representation': mode, 'rcond': float(rcond),
            'seconds': perf_counter() - started,
        })
        self.info['factor_representation'] = mode

    def _candidate(self, rhs, target_sum):
        if self.mode == 'anchor':
            return self.anchor_action.candidate(rhs, target_sum)
        if self.mode == 'original':
            w = cho_solve(self.factor, rhs, check_finite=False)
            s = (_sum(w) - target_sum) / self.denominator
            x = w - self.v * s
        else:
            projected = rhs - float(np.mean(rhs))
            projected -= target_sum * (self.mean - float(self.mean.mean()))
            x = cho_solve(self.factor, projected, check_finite=False)
            x += (target_sum - _sum(x)) / self.n
            product = self.K @ x
            s = float(np.mean(rhs - product - self.shift * x))
        return x, float(s)

    def _assess(self, rhs, target_sum, x, product, residual, constraint,
                residual_bound=0., constraint_bound=0.):
        maximum = float(np.max(np.abs(residual) + residual_bound))
        scale = max(1., float(np.max(np.abs(rhs))))
        tolerance = 1e-10 * scale
        constraint_tol = 64 * np.finfo(float).eps * max(1., _sum(np.abs(x)), abs(target_sum))
        projected_residual = residual - float(np.mean(residual))
        projected_residual -= constraint * (self.mean - float(self.mean.mean()))
        projected_allowance = np.linalg.norm(residual_bound)
        if constraint_bound:
            projected_allowance += constraint_bound * np.linalg.norm(self.mean - float(self.mean.mean()))
        feature_error = float((np.linalg.norm(projected_residual) + projected_allowance)
                              / (2 * np.sqrt(self.shift))
                              + (abs(constraint) + constraint_bound) * np.sqrt(max(0., float(self.mean.mean()))))
        feature_scale = max(1., np.sqrt(abs(float(x @ product))))
        if not np.isfinite([maximum, constraint_tol, feature_error, feature_scale]).all():
            raise FloatingPointError('Nonfinite constrained linear-solve accuracy estimate.')
        acceptable = (maximum <= tolerance and abs(constraint) + constraint_bound <= constraint_tol
                      and feature_error <= 5e-7 * feature_scale)
        return acceptable, maximum, feature_error

    def _ordinary_measure(self, rhs, target_sum, x, s, product=None):
        if not np.isfinite(x).all() or not np.isfinite(s):
            raise FloatingPointError('Nonfinite constrained linear-solve result.')
        if product is None:
            product = self.K @ x
        residual = rhs - product - self.shift * x - s
        constraint = target_sum - _sum(x)
        if not np.isfinite(residual).all() or not np.isfinite(constraint):
            raise FloatingPointError('Nonfinite constrained linear-solve residual.')
        acceptable, maximum, feature_error = self._assess(rhs, target_sum, x, product, residual, constraint)
        return acceptable, residual, constraint, product, maximum, feature_error

    def _measure(self, rhs, target_sum, x, s, product=None):
        measured = self._ordinary_measure(rhs, target_sum, x, s, product)
        if not measured[0]:
            return self._compensated_measure(rhs, target_sum, x, s, ordinary_maximum=measured[4])
        return measured

    def _measure_with_native_trial(self, rhs, target_sum, x, s, product=None):
        """Try one inexpensive correction before compensated O(n^2) work.

        The trial is accepted only by the unchanged ordinary checks on its
        freshly recomputed scores. Otherwise discard it and accurately measure
        the untouched original state: its ordinary failure can be cancellation,
        rather than an inaccurate solution. This preflight runs once per factor
        representation for each solve or reference candidate check (at most
        two Cholesky or five spectral representations), outside the existing
        correction budget.
        The anchor recovery uses direct accurate assessment without this trial.
        Only a committed trial counts as a refinement step; attempted/discarded
        inverse actions and their elapsed time are reported separately.
        """
        measured = self._ordinary_measure(rhs, target_sum, x, s, product)
        if measured[0]:
            return x, s, measured
        started = perf_counter()
        self.info['native_refinement_trials'] = self.info.get('native_refinement_trials', 0) + 1
        try:
            dx, ds = self._candidate(measured[1], measured[2])
            with np.errstate(over='raise', invalid='raise', under='ignore'):
                trial_x, trial_s = x + dx, float(s + ds)
            # Do not reuse supplied scores after changing the coefficients.
            trial = self._ordinary_measure(rhs, target_sum, trial_x, trial_s)
        except (LinAlgError, FloatingPointError) as exc:
            trial = None
            self.info['last_native_refinement_trial_error'] = str(exc)
        finally:
            self.info['native_refinement_seconds'] = self.info.get('native_refinement_seconds', 0.) + perf_counter() - started
        if trial is not None and trial[0]:
            self.info['native_refinement_acceptances'] = self.info.get('native_refinement_acceptances', 0) + 1
            self.info['refinement_steps'] += 1
            return trial_x, trial_s, trial
        self.info['native_refinement_discarded'] = self.info.get('native_refinement_discarded', 0) + 1
        return x, s, self._measure(rhs, target_sum, x, s, product=measured[3])

    def _equations_acceptable(self, rhs, target_sum, x, residual, constraint,
                              residual_bound=0., constraint_bound=0.):
        maximum = float(np.max(np.abs(residual) + residual_bound))
        tolerance = 1e-10 * max(1., float(np.max(np.abs(rhs))))
        constraint_tol = 64 * np.finfo(float).eps * max(1., _sum(np.abs(x)), abs(target_sum))
        return (np.isfinite(maximum) and maximum <= tolerance
                and abs(constraint) + constraint_bound <= constraint_tol)

    def _aposteriori_rkhs_bound(self, residual, constraint, bounds, constraint_bound):
        """Bound the original state's error using one auxiliary inverse action.

        The correction is NOT added to the state. The triangle estimate is its
        RKHS norm upper bound plus the global bound on its original-equation
        remainder. Both input and remainder residual uncertainties are carried.
        This numerical estimate retains the global bound's float64 assumptions;
        it is not a formal interval certificate and never checks recursively.
        """
        self.info['aposteriori_inverse_actions'] = self.info.get('aposteriori_inverse_actions', 0) + 1
        correction, scalar = self._candidate(residual, constraint)
        remaining, remaining_constraint, product, bounds2, constraint_bound2 = compensated_residual(
            self.K, self.shift, residual, correction, scalar, constraint)
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            combined = np.nextafter(bounds + bounds2, np.inf)
        combined_constraint = _upward_nonnegative(constraint_bound + constraint_bound2)
        if not np.isfinite(combined).all():
            raise FloatingPointError('Nonfinite a-posteriori residual uncertainty.')
        remainder = self._assess(residual, constraint, correction, product,
                                 remaining, remaining_constraint, combined, combined_constraint)[2]
        _, quadratic, quadratic_bound = _compensated_quadratic(self.K, correction)
        # A materially negative upper bound is invalid even though K is meant
        # to be PSD; clamping it could hide faulty arithmetic or an invalid K.
        upper = _upward_nonnegative(quadratic + quadratic_bound)
        norm_upper = _upward_nonnegative(np.sqrt(upper))
        estimate = _upward_nonnegative(norm_upper + remainder)
        return estimate, {'correction_norm_upper': norm_upper,
                          'remainder_global_bound': remainder,
                          'combined_residual_allowance_max': float(np.max(combined)),
                          'combined_constraint_allowance': combined_constraint,
                          'correction_quadratic_error_bound': quadratic_bound}

    def _compensated_measure(self, rhs, target_sum, x, s, ordinary_maximum=None):
        started = perf_counter()
        measured = self._bounded_native_measure(rhs, target_sum, x, s)
        if measured is None:
            residual, constraint, product, bounds, constraint_bound = compensated_residual(
                self.K, self.shift, rhs, x, s, target_sum)
            acceptable, maximum, feature_error = self._assess(
                rhs, target_sum, x, product, residual, constraint, bounds, constraint_bound)
        else:
            (acceptable, residual, constraint, product, maximum, feature_error,
             bounds, constraint_bound) = measured
        self.info['compensated_residual_checks'] = self.info.get('compensated_residual_checks', 0) + 1
        self.info['compensated_residual_seconds'] = self.info.get('compensated_residual_seconds', 0.) + perf_counter() - started
        self.info['max_compensated_residual_allowance'] = max(self.info.get('max_compensated_residual_allowance', 0.), float(np.max(bounds)))
        if ordinary_maximum is not None:
            self.info['max_rechecked_ordinary_residual'] = max(self.info.get('max_rechecked_ordinary_residual', 0.), ordinary_maximum)
        if not acceptable and self._equations_acceptable(
                rhs, target_sum, x, residual, constraint, bounds, constraint_bound):
            # The original equations already pass. Only the conservative RKHS
            # estimate is being reconsidered; never waive those equation gates.
            check_started = perf_counter()
            self.info['aposteriori_rkhs_checks'] = self.info.get('aposteriori_rkhs_checks', 0) + 1
            threshold = 5e-7 * max(1., np.sqrt(abs(float(x @ product))))
            details = {'global_bound': feature_error, 'tolerance': threshold,
                       'accepted': False, 'interval_certificate': False}
            try:
                estimate, terms = self._aposteriori_rkhs_bound(residual, constraint, bounds, constraint_bound)
                details.update(terms, estimate=estimate)
                if estimate <= threshold:
                    acceptable, feature_error = True, estimate
                    details['accepted'] = True
                    self.info['aposteriori_rkhs_acceptances'] = self.info.get('aposteriori_rkhs_acceptances', 0) + 1
            except (FloatingPointError, LinAlgError) as exc:
                # Fail closed while leaving ordinary bounded recovery available.
                details['error'] = str(exc)
            self.info['last_aposteriori_rkhs_check'] = details
            self.info['aposteriori_rkhs_seconds'] = self.info.get('aposteriori_rkhs_seconds', 0.) + perf_counter() - check_started
        return acceptable, residual, constraint, product, maximum, feature_error

    def _bounded_native_measure(self, rhs, target_sum, x, s):
        """Use a bounded native dot only when its decision is unambiguous.

        An accepted state passes the existing equation, constraint and RKHS
        gates, with a conservative lower bound for the relative RKHS scale.
        A residual lower bound can also prove the original equations fail:
        refinement is then necessary regardless of the accumulation method.
        Uncertain decisions use the expanded calculation on the untouched state.
        """
        started = perf_counter()
        native = native_compensated_residual(self.K, self.shift, rhs, x, s, target_sum)
        self.info['native_residual_attempts'] = self.info.get('native_residual_attempts', 0) + 1
        if native is None:
            self.info['native_residual_declines'] = self.info.get('native_residual_declines', 0) + 1
            self.info['native_residual_seconds'] = self.info.get('native_residual_seconds', 0.) + perf_counter() - started
            return None
        self.info['native_residual_checks'] = self.info.get('native_residual_checks', 0) + 1
        residual, constraint, product, bounds, constraint_bound, score_bounds = native
        acceptable, maximum, feature_error = self._assess(
            rhs, target_sum, x, product, residual, constraint, bounds, constraint_bound)
        if acceptable:
            # Score uncertainty must not enlarge the intended relative RKHS
            # tolerance. Bound x.T K x below using the native uncertainties.
            try:
                quadratic, dot_error = _compensated_dot(x, product)
                propagation, propagation_error = _compensated_dot(np.abs(x), score_bounds)
                error = _upward_nonnegative(dot_error + _upward_nonnegative(
                    propagation + propagation_error))
                with np.errstate(under='ignore'):
                    lower = float(np.nextafter(quadratic - error, -np.inf))
                    scale = max(1., float(np.nextafter(np.sqrt(max(0., lower)), 0.)))
                    threshold = float(np.nextafter(5e-7 * scale, 0.))
                acceptable = feature_error <= threshold
            except FloatingPointError:
                acceptable = False
        with np.errstate(under='ignore'):
            lower_residual = float(np.max(np.nextafter(np.abs(residual) - bounds, -np.inf)))
        equation_failure = lower_residual > 1e-10 * max(1., float(np.max(np.abs(rhs))))
        self.info['native_residual_seconds'] = self.info.get('native_residual_seconds', 0.) + perf_counter() - started
        if acceptable:
            self.info['native_residual_acceptances'] = self.info.get('native_residual_acceptances', 0) + 1
        elif equation_failure:
            self.info['native_residual_equation_failures'] = self.info.get('native_residual_equation_failures', 0) + 1
        else:
            self.info['native_residual_fallbacks'] = self.info.get('native_residual_fallbacks', 0) + 1
            return None
        return (acceptable, residual, constraint, product, maximum, feature_error,
                bounds, constraint_bound)

    def _try_intercept_refinement(self, rhs, target_sum, x, s, measurement):
        """Correct a constant residual without sub-ULP coefficient updates.

        This is a scalar coordinate refinement of the SAME equations. A cheap
        prediction avoids an extra accurate matvec unless this alone may pass.
        The rounded scalar is then checked with a fresh compensated residual;
        neither the prediction nor an unrepresentable update authorizes return.
        A successful correction consumes one of the existing refinement steps.
        """
        _, residual, constraint, product, _, _ = measurement
        # Preserve the usual mean correction. If its maximum residual is too
        # large, the midpoint of the residual range minimizes that same norm
        # over scalar corrections. Halving first avoids an overflowing sum.
        corrections = (_sum(residual) / self.n,
                       .5 * float(np.min(residual)) + .5 * float(np.max(residual)))
        previous = None
        for correction in corrections:
            updated = float(s + correction)
            if not np.isfinite(updated) or updated == s or updated == previous:
                continue
            previous = updated
            difference = updated - s
            if not np.isfinite(difference):
                continue
            # Do not let a conservative norm estimate suppress a potentially
            # valid scalar proposal. The fresh check includes the RKHS gate.
            if not self._equations_acceptable(rhs, target_sum, x, residual - difference, constraint):
                continue
            self.info['intercept_refinement_attempts'] = self.info.get('intercept_refinement_attempts', 0) + 1
            checked = self._compensated_measure(rhs, target_sum, x, updated)
            if not checked[0]:
                continue
            self.info['refinement_steps'] += 1
            self.info['intercept_refinement_steps'] = self.info.get('intercept_refinement_steps', 0) + 1
            return updated, checked
        return None

    def solve_constrained(self, rhs, target_sum=0.):
        rhs = np.asarray(rhs, dtype=float)
        target_sum = float(target_sum)
        if rhs.shape != (self.n,) or not np.isfinite(rhs).all() or not np.isfinite(target_sum):
            raise FloatingPointError('Invalid constrained linear-system right-hand side.')
        self.info['linear_solves'] += 1
        last_error = None
        for attempt in range(3):
            try:
                x, s = self._candidate(rhs, target_sum)
                if self.mode == 'anchor':
                    initial = self._measure(rhs, target_sum, x, s)
                else:
                    x, s, initial = self._measure_with_native_trial(rhs, target_sum, x, s)
                for refinement in range(4):
                    measured = initial if refinement == 0 else self._measure(rhs, target_sum, x, s)
                    if not measured[0] and refinement < 3:
                        corrected = self._try_intercept_refinement(rhs, target_sum, x, s, measured)
                        if corrected is not None:
                            s, measured = corrected
                    good, residual, constraint, product, maximum, error = measured
                    if good:
                        self.last_product = product
                        self.info['max_linear_residual'] = max(self.info['max_linear_residual'], maximum)
                        self.info['max_constraint_residual'] = max(self.info['max_constraint_residual'], abs(constraint))
                        self.info['max_estimated_rkhs_solve_error'] = max(self.info['max_estimated_rkhs_solve_error'], error)
                        if self.mode == 'anchor':
                            self.info['anchor_accepted_actions'] += 1
                        return x, s
                    if refinement == 3:
                        raise FloatingPointError('Constrained kernel solve did not meet residual accuracy '
                                                 f'(max residual={maximum:.3g}, estimated RKHS error={error:.3g}).')
                    dx, ds = self._candidate(residual, constraint)
                    x += dx
                    if self.mode == 'anchor':
                        x[0] = target_sum - _sum(x[1:])
                    s += ds
                    self.info['refinement_steps'] += 1
            except (LinAlgError, FloatingPointError) as exc:
                last_error = str(exc)
            if self.mode == 'anchor':
                break
            # Discard the old factor before allocating the alternative.
            self.factor = self.v = None
            next_mode = 'centered' if self.mode == 'original' else 'anchor'
            try:
                self._prepare(next_mode)
            except (LinAlgError, FloatingPointError) as exc:
                last_error = str(exc)
                if next_mode != 'centered':
                    break
            if next_mode == 'centered' and self.mode != 'centered':
                self.factor = self.v = None
                try:
                    self._prepare('anchor')
                except (LinAlgError, FloatingPointError) as exc:
                    last_error = str(exc)
                    break
            self.info['linear_recoveries'] += 1
        raise ConstrainedSolveFailure('Unable to solve the original kernel MM system accurately '
                                 'with bounded Cholesky, centered and anchor-coordinate recovery. '
                                 'The kernel and regularization were not changed.') from FloatingPointError(last_error)
