"""Model-unit refinement of the coefficient/eigenbasis reference MM step.

Use exact diagonal eigenpairs and an independent bordered solve. Tiny recovery
eigendecompositions are measured explicitly; no downloaded data are used.
"""
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import dwd._kernel_linear_system as linear_system
import dwd._kernel_solver as solver
from dwd._spectral_linear_system import SpectralLinearSystem
import dwd._spectral_linear_system as spectral


def bordered(K, shift, rhs, target=0.):
    n = len(K)
    A = np.block([[K + shift * np.eye(n), np.ones((n, 1))],
                  [np.ones((1, n)), np.zeros((1, 1))]])
    solution = np.linalg.solve(A, np.r_[rhs, target])
    return solution[:-1], float(solution[-1])


def dense_path(K, y, penalty, q, initial, offset, count):
    alpha, b = initial.copy(), float(offset)
    t = len(y) * q / (q + 1.) ** 2
    shift = 2 * penalty * t
    decisions, objectives = [], []
    for iteration in range(count + 1):
        score = K @ alpha + b
        margins = y * score
        losses = [1. - u if u <= q / (q + 1.)
                  else q ** q / ((q + 1.) ** (q + 1.) * u ** q) for u in margins]
        decisions.append(score)
        objectives.append(np.mean(losses) + penalty * alpha @ K @ alpha)
        if iteration == count:
            break
        slopes = np.array([-1. if u <= q / (q + 1.)
                           else -(q / ((q + 1.) * u)) ** (q + 1.) for u in margins])
        alpha, increment = bordered(K, shift, K @ alpha - t * y * slopes / len(y))
        b += increment
    return np.array(decisions), np.array(objectives), alpha, b


class SpectralCandidateRefinementTests(unittest.TestCase):
    def setUp(self):
        self.n = 12
        self.values = np.linspace(.1, 2., self.n)
        self.K = np.diag(self.values)
        self.U = np.eye(self.n)
        self.rhs = np.linspace(-.3, .7, self.n)

    def test_direction_accepted_in_direction_units_is_repaired_after_step_amplification(self):
        n, t, shift = self.n, self.n / 4., .3
        K = np.eye(n)
        system = SpectralLinearSystem(K, shift, np.eye(n), np.ones(n))
        y = np.tile([-1., 1.], n // 2)
        # q=1 at zero initialization: z=-y/n, lambda=shift/(2t).
        z = -y / n
        direction, intercept_direction = bordered(K, shift, z)
        error = np.zeros(n)
        error[:2] = [.8e-10 / (1 + shift), -.8e-10 / (1 + shift)]
        inaccurate_direction = direction + error
        direction_ok = system._measure(z, 0., inaccurate_direction, intercept_direction)[0]
        self.assertTrue(direction_ok)

        candidate_rhs = -t * z
        candidate_alpha = -t * inaccurate_direction
        candidate_s = -t * intercept_direction
        candidate_scores = K @ candidate_alpha
        before = (candidate_alpha.copy(), candidate_scores.copy())
        good, residual, _, _, _, _ = system._measure(
            candidate_rhs, 0., candidate_alpha, candidate_s, product=candidate_scores)
        self.assertFalse(good)
        self.assertGreater(np.max(np.abs(residual)), 2e-10)
        corrected_alpha, corrected_s, scores = system.refine_candidate(
            candidate_rhs, candidate_alpha, candidate_s, candidate_scores)
        expected_alpha, expected_s = bordered(K, shift, candidate_rhs)
        assert_allclose(corrected_alpha, expected_alpha, rtol=2e-14, atol=2e-15)
        assert_allclose(corrected_s, expected_s, rtol=2e-14, atol=2e-15)
        assert_array_equal(scores, K @ corrected_alpha)
        self.assertEqual(system.info['refinement_steps'], 1)
        assert_array_equal(candidate_alpha, before[0])
        assert_array_equal(candidate_scores, before[1])

    def test_final_candidate_refinement_repairs_coefficient_and_intercept_errors(self):
        system = SpectralLinearSystem(self.K, .07, self.U, self.values)
        expected, expected_s = bordered(self.K, .07, self.rhs)
        corrupted = expected + np.linspace(-.02, .04, self.n)
        s = expected_s + .17
        original = corrupted.copy()
        with patch.object(linear_system, 'cho_factor', side_effect=AssertionError('Unexpected Cholesky')):
            actual, actual_s, scores = system.refine_candidate(self.rhs, corrupted, s, self.K @ corrupted)
        assert_allclose(actual, expected, rtol=3e-14, atol=3e-15)
        assert_allclose(actual_s, expected_s, rtol=3e-14, atol=3e-15)
        assert_array_equal(scores, self.K @ actual)
        assert_array_equal(corrupted, original)
        self.assertEqual(system.info['refinement_steps'], 1)
        self.assertLess(abs(np.sum(actual)), 1e-13)

    def test_valid_candidate_does_not_change_the_coefficient_mm_step(self):
        system = SpectralLinearSystem(self.K, .17, self.U, self.values)
        expected, s = bordered(self.K, .17, self.rhs)
        scores = self.K @ expected
        with patch.object(system, '_candidate', side_effect=AssertionError('Unnecessary correction')):
            actual, actual_s, actual_scores = system.refine_candidate(self.rhs, expected, s, scores)
        assert_array_equal(actual, expected)
        self.assertEqual(actual_s, s)
        assert_array_equal(actual_scores, scores)
        self.assertEqual(system.info['refinement_steps'], 0)

    def test_persistent_bad_correction_fails_after_a_bounded_count(self):
        for include_recovery in (False, True):
            with self.subTest(recovery=include_recovery):
                system = SpectralLinearSystem(self.K, .17, self.U, self.values)
                initial, scores = np.zeros(self.n), np.zeros(self.n)
                modes = []
                def wrong(r, target):
                    modes.append(system.info['factor_representation'])
                    return np.zeros(self.n), 0.
                with patch.object(system, '_candidate', side_effect=wrong), \
                        patch.object(linear_system, 'cho_factor', side_effect=AssertionError('Unexpected Cholesky')), \
                        patch.object(spectral, 'validated_eigh', wraps=spectral.validated_eigh) as prepare:
                    with self.assertRaisesRegex(FloatingPointError, 'candidate failed original-system accuracy'):
                        method = system.refine_candidate if include_recovery else system._refine_candidate
                        method(self.rhs, initial, 0., scores)
                # Preserve the old inner bound; the outer interface permits
                # four additional representations with that same finite budget.
                expected = {'validated_eigenbasis': 4}
                if include_recovery:
                    expected.update({mode: 4 for mode in (
                        'power_of_two_equilibrated_eigenbasis', 'original_kernel_evd_eigenbasis',
                        'original_kernel_evr_eigenbasis', 'original_kernel_evx_eigenbasis')})
                self.assertEqual(Counter(modes), expected)
                self.assertEqual(prepare.call_count, 4 * int(include_recovery))
                attempts = 1 + 4 * int(include_recovery)
                self.assertEqual(system.info['native_refinement_discarded'], attempts)
                self.assertEqual(system.info['refinement_steps'], 3 * attempts)
                self.assertEqual(system.info['linear_solves'], 0)
                self.assertNotIn('refinement_budget_extensions', system.info)
                self.assertIsNone(system.last_product)
                assert_array_equal(initial, np.zeros(self.n))
                assert_array_equal(scores, np.zeros(self.n))

    def test_nonfinite_candidate_fails_without_spectral_or_cholesky_retry(self):
        zero, bad = np.zeros(self.n), np.zeros(self.n)
        bad[3] = np.nan
        cases = [(self.rhs, bad, 0., zero), (bad, zero, 0., zero),
                 (self.rhs, zero, np.inf, zero), (self.rhs, zero, 0., bad),
                 (self.rhs[:-1], zero, 0., zero), (self.rhs, zero[:-1], 0., zero),
                 (self.rhs, zero, [0.], zero), (self.rhs, zero, 0., zero[:-1]),
                 (self.rhs.astype(complex), zero, 0., zero),
                 (self.rhs, zero.astype(complex), 0., zero),
                 (self.rhs, zero, 0j, zero), (self.rhs, zero, 0., zero.astype(complex))]
        for index, (rhs, x, s, scores) in enumerate(cases):
            with self.subTest(case=index):
                system = SpectralLinearSystem(self.K, .17, self.U, self.values)
                with patch.object(system, '_candidate', side_effect=AssertionError('Invalid candidate inverse')) as action, \
                        patch.object(spectral, 'validated_eigh', side_effect=AssertionError('Invalid candidate EVD')) as prepare, \
                        patch.object(linear_system, 'cho_factor', side_effect=AssertionError('Invalid candidate Cholesky')) as factor:
                    with self.assertRaisesRegex(FloatingPointError, 'Invalid spectral reference candidate'):
                        system.refine_candidate(rhs, x, s, scores)
                action.assert_not_called()
                prepare.assert_not_called()
                factor.assert_not_called()
                self.assertFalse(system._recovery_attempted)
                self.assertNotIn('spectral_recovery_attempt', system.info)

    def test_singular_kernel_candidate_is_refined_without_losing_positive_directions(self):
        values = np.array([0., 1e-12, .1, .3, .5, .9, 1., 1.2, 1.4, 1.7, 1.9, 2.])
        K, shift = np.diag(values), .03
        system = SpectralLinearSystem(K, shift, self.U, values)
        expected, expected_s = bordered(K, shift, self.rhs)
        corrupted = expected + np.linspace(-.02, .02, self.n)
        actual, actual_s, scores = system.refine_candidate(self.rhs, corrupted, expected_s, K @ corrupted)
        assert_allclose(actual, expected, rtol=2e-13, atol=3e-14)
        assert_allclose(actual_s, expected_s, rtol=2e-13, atol=3e-14)
        assert_array_equal(scores, K @ actual)
        self.assertNotEqual(scores[1], 0.)
        self.assertEqual(system.info['positive_eigenvalues_discarded'], 0)

    def test_reference_path_matches_independent_mm_oracle_with_nonzero_initial_state(self):
        y = np.r_[-np.ones(5), np.ones(7)]
        initial = np.linspace(-.05, .11, self.n)
        for q in (.2, 1., 3.5):
            with self.subTest(q=q):
                expected_scores, expected_objectives, expected_alpha, expected_b = dense_path(
                    self.K, y, .07, q, initial, .31, 6)
                seen = []
                with patch.object(linear_system, 'cho_factor', side_effect=AssertionError('Unexpected Cholesky')):
                    result = solver.solve_kernel(
                        self.K, y, .07, q=q, implementation='reference',
                        K_eig=(self.U, self.values), alpha_init=initial, offset_init=.31,
                        max_iter=6, stopping='fixed', callback=lambda s: seen.append(dict(s)))
                assert_allclose([s['decision_values'] for s in seen], expected_scores, rtol=2e-12, atol=3e-13)
                assert_allclose(result['objective_history'], expected_objectives, rtol=2e-12, atol=3e-13)
                assert_allclose(result['alpha'], expected_alpha, rtol=2e-12, atol=3e-13)
                assert_allclose(result['offset'], expected_b, rtol=2e-12, atol=3e-13)
                self.assertEqual([s['iteration'] for s in seen], list(range(7)))
                assert_array_equal(seen[0]['alpha'], initial)

    def test_corrupted_reference_candidate_is_repaired_before_any_callback_commit(self):
        y = np.r_[-np.ones(5), np.ones(7)]
        expected_scores, expected_objectives, _, _ = dense_path(
            self.K, y, .07, 1., np.zeros(self.n), 0., 4)
        original_update = solver.reference_update
        count, seen = [], []

        def corrupted_once(*args, **kwargs):
            alpha, offset, scores = original_update(*args, **kwargs)
            count.append(1)
            if len(count) == 1:
                alpha = alpha.copy()
                alpha[:2] += [.02, -.02]
                scores = self.K @ alpha
            return alpha, offset, scores

        with patch.object(solver, 'reference_update', new=corrupted_once):
            result = solver.solve_kernel(
                self.K, y, .07, implementation='reference', K_eig=(self.U, self.values),
                max_iter=4, stopping='fixed', callback=lambda s: seen.append(dict(s)))
        self.assertEqual(len(count), 4)
        self.assertEqual([s['iteration'] for s in seen], list(range(5)))
        assert_allclose([s['decision_values'] for s in seen], expected_scores, rtol=2e-12, atol=3e-13)
        assert_allclose(result['objective_history'], expected_objectives, rtol=2e-12, atol=3e-13)
        self.assertGreaterEqual(result['diagnostics']['linear_system_diagnostics']['refinement_steps'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
