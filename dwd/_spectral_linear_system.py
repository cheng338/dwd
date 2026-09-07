"""Coefficient-reference inverse actions through a validated eigenbasis.

This is the repaired upstream numerical route. It never uses a Cholesky
factor. Only equation measurement is shared with the optimized implementation.
"""
from time import perf_counter

import numpy as np
from scipy.linalg import LinAlgError
from ._eigen import validated_eigh
from ._kernel_linear_system import KernelLinearSystem, _sum


class SpectralLinearSystem(KernelLinearSystem):
    def __init__(self, K, shift, vectors, values):
        self.K, self.shift, self.n = K, float(shift), len(K)
        self.mean = self.K.mean(axis=0)
        self.vectors = vectors
        self._equilibration = None
        self._recovery_attempted = False
        self._recovery_next = 0
        self._active_recovery_attempt = None
        self.inverse = 1. / (values + shift)
        if not np.isfinite(self.inverse).all() or np.any(self.inverse <= 0):
            raise FloatingPointError('Unrepresentable inverse of the shifted kernel spectrum.')
        self.v = self._inverse_action(np.ones(self.n))
        self.denominator = _sum(self.v)
        if self.denominator <= 0:
            raise FloatingPointError('Invalid spectral free-intercept Schur denominator.')
        self.last_product = None
        self.info = {'linear_system': 'original free-intercept constrained MM system',
                     'factor_representation': 'validated_eigenbasis',
                     'linear_solves': 0, 'refinement_steps': 0,
                     'max_linear_residual': 0., 'max_constraint_residual': 0.,
                     'max_estimated_rkhs_solve_error': 0.,
                     'added_objective_regularization': 0.,
                     'positive_eigenvalues_discarded': 0,
                     'residual_checks_are_interval_certificates': False,
                     'linear_recoveries': 0}

    def _inverse_action(self, rhs):
        if self._equilibration is not None:
            scale = self._equilibration
            return scale * (self.vectors @ (self.inverse * (self.vectors.T @ (scale * rhs))))
        return self.vectors @ (self.inverse * (self.vectors.T @ rhs))

    def _build_equilibrated(self):
        """Prepare D (K + shift I) D; never change the fitted kernel.

        Powers of two avoid intermediate scaling roundoff for normal results.
        A lost nonzero entry, overflow, or nonpositive represented eigenvalue
        fails closed. Original-equation checks still determine solve accuracy.
        """
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            diagonal = np.diag(self.K) + self.shift
            if not np.isfinite(diagonal).all() or np.any(diagonal <= 0.):
                raise FloatingPointError('The shifted diagonal cannot be equilibrated.')
            powers = np.rint(-.5 * np.log2(diagonal)).astype(np.int64)
            scale = np.ldexp(np.ones(self.n), powers)
            if not np.isfinite(scale).all() or np.any(scale <= 0.):
                raise FloatingPointError('Unrepresentable spectral equilibration scale.')
            matrix = np.empty_like(self.K, dtype=float, order='F')
            for start in range(0, self.n, 128):
                block = self.K[start:start + 128]
                scaled = np.ldexp(block, powers[start:start + len(block), None] + powers[None, :])
                if not np.isfinite(scaled).all() or np.any((block != 0.) & (scaled == 0.)):
                    raise FloatingPointError('Spectral equilibration lost a nonzero matrix entry.')
                matrix[start:start + len(block)] = scaled
            scaled_shift = np.ldexp(np.full(self.n, self.shift), 2 * powers)
            if not np.isfinite(scaled_shift).all() or np.any(scaled_shift <= 0.):
                raise FloatingPointError('Unrepresentable equilibrated MM shift.')
            matrix.flat[::self.n + 1] += scaled_shift
        (values, vectors), eigen_info = validated_eigh(matrix, return_info=True)
        if not np.isfinite(values).all() or np.any(values <= 0.):
            raise FloatingPointError('The equilibrated shifted system has no positive represented spectrum.')
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            inverse = 1. / values
            if not np.isfinite(inverse).all() or np.any(inverse <= 0.):
                raise FloatingPointError('Unrepresentable equilibrated spectral inverse.')
            v = scale * (vectors @ (inverse * (vectors.T @ scale)))
        denominator = _sum(v)
        if not np.isfinite(v).all() or denominator <= 0.:
            raise FloatingPointError('Invalid equilibrated free-intercept Schur denominator.')
        details = {'representation': 'power_of_two_equilibrated_eigenbasis',
                   'scale_exponent_min': int(powers.min()), 'scale_exponent_max': int(powers.max()),
                   'minimum_eigenvalue': float(values.min()), 'maximum_eigenvalue': float(values.max()),
                   'eigen_validation': eigen_info}
        return vectors, inverse, scale, v, denominator, details

    def _prepare_equilibrated(self, reason):
        """The first, single-use recovery keeps its historical diagnostic key."""
        if self._recovery_next != 0:
            raise FloatingPointError('Equilibrated spectral preparation was already attempted.')
        self._recovery_next = 1
        self._prepare_representation('equilibrated', reason)

    def _build_original(self, driver):
        """Fresh original-K eigenpairs define an inverse preconditioner only.

        Apply the public spectral PSD tolerance again to this fresh result.
        Only roundoff-negative values receive the original reference's zero
        inverse-weight floor; every positive mode remains. This approximation
        does not replace K or its objective, and original equations remain
        decisive. There is no Cholesky substitution or added regularization.
        """
        (values, vectors), eigen_info = validated_eigh(self.K, drivers=(driver,), return_info=True)
        if values.shape != (self.n,) or not np.isfinite(values).all():
            raise FloatingPointError('Invalid fresh original-kernel eigenvalues.')
        # Keep this threshold identical to _kernel_solver._check_spectrum;
        # importing the solver here would introduce a circular dependency.
        limit = 100 * np.finfo(float).eps * self.n * max(1., float(np.max(np.abs(values))))
        if not np.isfinite(limit) or values.min() < -limit:
            raise FloatingPointError('The fresh original-kernel spectrum fails the public PSD tolerance.')
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            shifted = np.maximum(values, 0.) + self.shift
            if not np.isfinite(shifted).all() or np.any(shifted <= 0.):
                raise FloatingPointError('The fresh original-kernel basis has nonpositive or nonfinite shifted denominators.')
            inverse = 1. / shifted
            if not np.isfinite(inverse).all() or np.any(inverse <= 0.):
                raise FloatingPointError('Unrepresentable fresh original-kernel spectral inverse.')
            v = vectors @ (inverse * (vectors.T @ np.ones(self.n)))
        denominator = _sum(v)
        if not np.isfinite(v).all() or denominator <= 0.:
            raise FloatingPointError('Invalid fresh original-kernel Schur denominator.')
        details = {'representation': f'original_kernel_{driver}_eigenbasis', 'driver': driver,
                   'minimum_eigenvalue': float(values.min()), 'maximum_eigenvalue': float(values.max()),
                   'minimum_shifted_eigenvalue': float(shifted.min()),
                   'psd_tolerance': float(limit),
                   'negative_roundoff_eigenvalues_clipped': int(np.count_nonzero(values < 0.)),
                   'positive_eigenvalues_retained': int(np.count_nonzero(values > 0.)),
                   'inverse_preconditioner_only': True,
                   'inverse_weight_policy': 'maximum(raw_eigenvalues, 0) + shift',
                   'eigen_validation': eigen_info}
        return vectors, inverse, None, v, denominator, details

    def _release_action(self):
        # The caller can still own its original cache. Release this system's
        # failed recovery basis before allocating another full eigenbasis.
        self.vectors = self.inverse = self._equilibration = self.v = None
        self.denominator = None
        self._active_recovery_attempt = None

    def _prepare_representation(self, kind, reason):
        self._release_action()
        self._recovery_attempted = True
        started = perf_counter()
        failure = None
        try:
            prepared = self._build_equilibrated() if kind == 'equilibrated' else self._build_original(kind)
        except (FloatingPointError, LinAlgError, ValueError, OverflowError) as exc:
            failure = f'{type(exc).__name__}: {exc}'
        # Leave the except block before raising: do not retain a failed dense
        # allocation through its original traceback.
        attempt = {'kind': kind, 'driver': None if kind == 'equilibrated' else kind,
                   'trigger': reason, 'seconds': perf_counter() - started,
                   'success': failure is None, 'preparation_succeeded': failure is None,
                   'accepted': False, 'accurate_actions': 0}
        self.info.setdefault('spectral_recovery_attempts', []).append(attempt)
        if kind == 'equilibrated':
            self.info['spectral_recovery_attempt'] = attempt
        if failure is not None:
            attempt['error'] = failure
            raise FloatingPointError(f'{kind} spectral preparation failed: ' + failure)
        self.vectors, self.inverse, self._equilibration, self.v, self.denominator, details = prepared
        attempt.update(details)
        if kind == 'equilibrated':
            attempt['driver'] = details['eigen_validation'].get('accepted_driver')
        self._active_recovery_attempt = attempt
        self.info['factor_representation'] = details['representation']
        self.info['linear_recoveries'] += 1

    def _raise_recovery_exhausted(self):
        causes = [item['error'] for item in self.info.get('spectral_inverse_failures', [])
                  if item['representation'] == 'validated_eigenbasis']
        causes += [f"{item['kind']}: {item.get('solve_error', item.get('error'))}"
                   for item in self.info.get('spectral_recovery_attempts', [])
                   if 'error' in item or 'solve_error' in item]
        raise FloatingPointError('The coefficient/eigenbasis reference failed accuracy after bounded native '
                                 'spectral recovery (at most five representations; no Cholesky substitution). '
                                 + ' | '.join(causes))

    def _prepare_next_recovery(self, reason):
        # Each entry is consumed BEFORE preparation, including failed attempts.
        # Later right-hand sides cannot restart an exhausted plan.
        plan = ('equilibrated', 'evd', 'evr', 'evx')
        while self._recovery_next < len(plan):
            kind = plan[self._recovery_next]
            failure = None
            try:
                if kind == 'equilibrated':
                    self._prepare_equilibrated(reason)
                else:
                    self._recovery_next += 1
                    self._prepare_representation(kind, reason)
            except (FloatingPointError, LinAlgError) as exc:
                failure = str(exc)
            if failure is None:
                return
            reason = failure
        self._raise_recovery_exhausted()

    def _run_with_recovery(self, operation, *args):
        for _ in range(5):
            if self.vectors is None:
                self._prepare_next_recovery('No usable spectral inverse remains.')
            failure = None
            try:
                result = operation(*args)
            except (FloatingPointError, LinAlgError) as exc:
                failure = f'{type(exc).__name__}: {exc}'
            if failure is None:
                if self._active_recovery_attempt is not None:
                    self._active_recovery_attempt['accepted'] = True
                    self._active_recovery_attempt['accurate_actions'] += 1
                return result
            self.info.setdefault('spectral_inverse_failures', []).append({
                'representation': self.info['factor_representation'], 'error': failure})
            if self._active_recovery_attempt is not None:
                self._active_recovery_attempt['solve_error'] = failure
            # No numerical traceback survives this point. Neither failed
            # coefficient refinements nor basis arrays seed the next attempt.
            self._release_action()
            self._prepare_next_recovery(failure)
        self._raise_recovery_exhausted()

    def _candidate(self, rhs, target_sum):
        w = self._inverse_action(rhs)
        s = (_sum(w) - target_sum) / self.denominator
        return w - self.v * s, float(s)

    def _refinement_budget(self, iteration, budget, initial_residual, measured):
        # Three corrections remain the ordinary bound. A demonstrably
        # contracting original-equation residual can justify five more; this
        # changes neither the MM iteration cap nor the required accuracy.
        if (not measured[0] and iteration == 3 and budget == 3
                and initial_residual > 0. and measured[4] <= .25 * initial_residual):
            self.info['refinement_budget_extensions'] = self.info.get('refinement_budget_extensions', 0) + 1
            self.info['max_refinement_budget'] = 8
            return 8
        return budget

    def _refine_step(self, x, s, residual, constraint, extended):
        dx, ds = self._candidate(residual, constraint)
        updated_x, updated_s = x + dx, s + ds
        self.info['refinement_steps'] += 1
        if extended:
            self.info['extended_refinement_steps'] = self.info.get('extended_refinement_steps', 0) + 1
            if updated_s == s and np.array_equal(updated_x, x):
                self.info['extended_refinement_stagnation_count'] = self.info.get('extended_refinement_stagnation_count', 0) + 1
                raise FloatingPointError('The spectral reference original-equation refinement '
                                         'stagnated in float64 after bounded recovery.')
        return updated_x, updated_s

    def solve_constrained(self, rhs, target_sum=0.):
        rhs = np.asarray(rhs, dtype=float)
        target_sum = float(target_sum)
        if rhs.shape != (self.n,) or not np.isfinite(rhs).all() or not np.isfinite(target_sum):
            raise FloatingPointError('Invalid spectral reference right-hand side.')
        self.info['linear_solves'] += 1
        return self._run_with_recovery(self._solve_constrained, rhs, target_sum)

    def _solve_constrained(self, rhs, target_sum):
        x, s = self._candidate(rhs, target_sum)
        x, s, initial = self._measure_with_native_trial(rhs, target_sum, x, s)
        budget, initial_residual = 3, None
        for refinement in range(9):
            measured = initial if refinement == 0 else self._measure(rhs, target_sum, x, s)
            if initial_residual is None:
                initial_residual = measured[4]
            budget = self._refinement_budget(refinement, budget, initial_residual, measured)
            if not measured[0] and refinement < budget:
                corrected = self._try_intercept_refinement(rhs, target_sum, x, s, measured)
                if corrected is not None:
                    s, measured = corrected
                    if refinement >= 3:
                        self.info['extended_refinement_steps'] = self.info.get('extended_refinement_steps', 0) + 1
            good, residual, constraint, product, maximum, error = measured
            if good:
                self.last_product = product
                self.info['max_linear_residual'] = max(self.info['max_linear_residual'], maximum)
                self.info['max_constraint_residual'] = max(self.info['max_constraint_residual'], abs(constraint))
                self.info['max_estimated_rkhs_solve_error'] = max(self.info['max_estimated_rkhs_solve_error'], error)
                return x, s
            if refinement == budget:
                raise FloatingPointError('The coefficient/eigenbasis reference failed residual accuracy '
                                         'after bounded refinement; no Cholesky substitution was made.')
            x, s = self._refine_step(x, s, residual, constraint, refinement >= 3)

    def refine_candidate(self, rhs, x, s, scores):
        """Check the NEW MM state after coefficient subtraction, in model units.

        Direction errors are multiplied by the MM step scale. Checking this
        original next-state equation accounts for that scale and subtraction
        roundoff, using the same model-accuracy policy as the optimized path.
        """
        # Invalid caller state is not an inverse-accuracy failure. In
        # particular, do not allocate a recovery eigensystem for NaN scores
        # or coefficients. Finite inaccurate candidates still get refinement.
        try:
            if (np.iscomplexobj(rhs) or np.iscomplexobj(x) or np.iscomplexobj(s)
                    or np.ndim(s) != 0 or (scores is not None and np.iscomplexobj(scores))):
                raise ValueError('Candidate state must be real.')
            rhs, x, s = np.asarray(rhs, dtype=float), np.asarray(x, dtype=float), float(s)
            if scores is not None:
                scores = np.asarray(scores, dtype=float)
            valid = (rhs.shape == (self.n,) and x.shape == (self.n,)
                     and np.isfinite(rhs).all() and np.isfinite(x).all() and np.isfinite(s)
                     and (scores is None or (scores.shape == (self.n,) and np.isfinite(scores).all())))
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            raise FloatingPointError('Invalid spectral reference candidate: expected finite real '
                                     'length-n vectors and a finite real scalar offset.')
        # The inner routine never mutates its inputs. Each representation starts
        # from the same actual coefficient-MM candidate, not a diverged iterate.
        return self._run_with_recovery(self._refine_candidate, rhs, x, s, scores)

    def _refine_candidate(self, rhs, x, s, scores):
        x, s, initial = self._measure_with_native_trial(rhs, 0., x, s, product=scores)
        budget, initial_residual = 3, None
        for refinement in range(9):
            measured = initial if refinement == 0 else self._measure(
                rhs, 0., x, s, product=scores)
            if initial_residual is None:
                initial_residual = measured[4]
            budget = self._refinement_budget(refinement, budget, initial_residual, measured)
            if not measured[0] and refinement < budget:
                corrected = self._try_intercept_refinement(rhs, 0., x, s, measured)
                if corrected is not None:
                    s, measured = corrected
                    if refinement >= 3:
                        self.info['extended_refinement_steps'] = self.info.get('extended_refinement_steps', 0) + 1
            good, residual, constraint, product, maximum, error = measured
            if good:
                self.info['max_candidate_residual'] = max(self.info.get('max_candidate_residual', 0.), maximum)
                self.info['max_estimated_rkhs_candidate_error'] = max(
                    self.info.get('max_estimated_rkhs_candidate_error', 0.), error)
                return x, s, product
            if refinement == budget:
                raise FloatingPointError('The coefficient-reference MM candidate failed original-system '
                                         'accuracy after bounded spectral refinement.')
            x, s = self._refine_step(x, s, residual, constraint, refinement >= 3)
            scores = None
