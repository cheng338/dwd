"""Independent equations and bounded recovery for the MM linear-system layer.

The oracle is a pivoted dense solve of the bordered system, not another Schur
implementation. These tests never compute an eigendecomposition or load MNIST.
"""
import math
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import dwd._kernel_linear_system as implementation
from dwd._kernel_linear_system import KernelLinearSystem


def bordered_oracle(K, shift, rhs, target_sum):
    n = len(K)
    bordered = np.empty((n + 1, n + 1), dtype=float)
    bordered[:n, :n] = K + shift * np.eye(n)
    bordered[:n, n] = 1.
    bordered[n, :n] = 1.
    bordered[n, n] = 0.
    expected = np.linalg.solve(bordered, np.r_[rhs, target_sum])
    return expected[:-1], expected[-1]


class KernelLinearSystemGeneralTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(703182)
        self.features = rng.normal(size=(7, 3))
        self.K = self.features @ self.features.T + .4 * np.eye(7)
        self.rhs = rng.normal(size=7)

    def assert_solution(self, system, rhs, target_sum=0., *, rtol=2e-9, atol=3e-10):
        x, s = system.solve_constrained(rhs, target_sum)
        expected_x, expected_s = bordered_oracle(system.K, system.shift, rhs, target_sum)
        assert_allclose(x, expected_x, rtol=rtol, atol=atol)
        assert_allclose(s, expected_s, rtol=rtol, atol=atol)
        assert_allclose(system.K @ x + system.shift * x + s, rhs,
                        rtol=2e-10, atol=3e-10 * max(1., np.max(np.abs(rhs))))
        self.assertLessEqual(abs(math.fsum(x) - target_sum),
                             100 * np.finfo(float).eps * max(1., math.fsum(abs(x)), abs(target_sum)))
        assert_array_equal(system.last_product, system.K @ x)
        return x, s

    def test_spd_singular_zero_and_constant_kernels_match_bordered_oracle(self):
        matrices = {
            'positive_definite': self.K,
            'singular_gram': self.features @ self.features.T,
            'zero': np.zeros((7, 7)),
            'constant': 3. * np.ones((7, 7)),
            'duplicate_rows': np.vstack([self.features[:3], self.features[:3], self.features[:1]])
                              @ np.vstack([self.features[:3], self.features[:3], self.features[:1]]).T,
        }
        for name, K in matrices.items():
            for target in (0., .73, -2.1):
                with self.subTest(kernel=name, target_sum=target):
                    self.assert_solution(KernelLinearSystem(K, .17), self.rhs, target)

    def test_scaled_problems_and_general_q_mm_shifts(self):
        for scale in (1e-6, 1., 1e6):
            for q in (.2, 1., 3.5):
                with self.subTest(scale=scale, q=q):
                    t = len(self.K) * q / (q + 1.) ** 2
                    shift = 2 * (.13 * scale) * t
                    self.assert_solution(KernelLinearSystem(scale * self.K, shift),
                                         scale * self.rhs, -.37,
                                         rtol=4e-9, atol=5e-10 * max(1., scale))

    def test_small_positive_directions_are_not_discarded_at_low_rcond(self):
        K = np.diag([0., 1e-14, 1e-10, .03, .7, 2., 5.])
        system = KernelLinearSystem(K, 1e-8)
        rhs = np.array([.2, -.7, .9, -.4, .3, -.2, .8])
        x, _ = self.assert_solution(system, rhs, 0., rtol=5e-8, atol=3e-8)
        self.assertGreater(abs(K[1, 1] * x[1]), 1e-8)
        self.assertEqual(system.info['positive_eigenvalues_discarded'], 0)
        self.assertEqual(system.info['added_objective_regularization'], 0.)
        self.assertLess(system.info['factorization_attempts'][0]['rcond'], np.sqrt(np.finfo(float).eps))

    def test_zero_kernel_tiny_shift_has_analytic_solution(self):
        shift = 1e-10
        K = np.zeros((7, 7))
        rhs = np.array([1., -1., 2., -2., .5, -.5, 0.])
        system = KernelLinearSystem(K, shift)
        x, s = system.solve_constrained(rhs, 0.)
        assert_allclose(x, rhs / shift, rtol=2e-15)
        self.assertEqual(s, 0.)
        assert_array_equal(system.last_product, np.zeros(7))

    def test_nonzero_initial_mm_state_produces_correct_next_free_intercept_state(self):
        initial = np.array([.12, -.2, .03, .15, -.06, .24, .02])
        offset = .31
        y = np.array([-1., 1., -1., 1., 1., -1., 1.])
        self.assertNotEqual(math.fsum(initial), 0.)
        for q in (.2, 1., 3.5):
            with self.subTest(q=q):
                margins = y * (self.K @ initial + offset)
                cutoff = q / (q + 1.)
                derivative = np.array([-1. if u <= cutoff else -(q / ((q + 1.) * u)) ** (q + 1.)
                                       for u in margins])
                t = len(y) * q / (q + 1.) ** 2
                shift = 2 * .07 * t
                rhs = self.K @ initial - t * y * derivative / len(y)
                system = KernelLinearSystem(self.K, shift)
                x, s = self.assert_solution(system, rhs)
                # Independently test both blocks of the free-intercept system
                # and the score recurrence used by the outer MM iteration.
                assert_allclose(self.K @ x + offset + s,
                                rhs - shift * x + offset, rtol=2e-12, atol=2e-12)

    def test_factorization_failure_recovers_centered_with_nonzero_constraint(self):
        original_factor = implementation.cho_factor
        calls = []

        def fail_first_factor(array, **kwargs):
            calls.append(array.shape)
            if len(calls) == 1:
                raise implementation.LinAlgError('Injected failed original factorization')
            return original_factor(array, **kwargs)

        with patch.object(implementation, 'cho_factor', new=fail_first_factor):
            system = KernelLinearSystem(self.K, .11)
        self.assertEqual(system.mode, 'centered')
        self.assertEqual(system.info['linear_recoveries'], 1)
        for target in (0., 2.5, -1.3):
            with self.subTest(target_sum=target):
                self.assert_solution(system, self.rhs, target)
        self.assertEqual(len(calls), 2)

    def test_one_inaccurate_solve_is_corrected_by_refinement(self):
        system = KernelLinearSystem(self.K, .17)
        original_solve = implementation.cho_solve
        calls = []
        error = np.array([.03, -.02, .01, -.04, .02, .01, -.01])

        def bad_once(*args, **kwargs):
            result = original_solve(*args, **kwargs)
            calls.append(1)
            return result + error if len(calls) == 1 else result

        with patch.object(implementation, 'cho_solve', new=bad_once):
            self.assert_solution(system, self.rhs, .4)
        self.assertGreaterEqual(system.info['refinement_steps'], 1)
        self.assertEqual(system.info['linear_recoveries'], 0)
        self.assertLessEqual(len(calls), 4)

    def test_persistently_bad_original_solve_recovers_centered(self):
        system = KernelLinearSystem(self.K, .17)
        original_matrix = system.factor[0]
        original_solve = implementation.cho_solve
        error = np.array([.03, -.02, .01, -.04, .02, .01, -.01])

        def bad_original(factor, rhs, **kwargs):
            result = original_solve(factor, rhs, **kwargs)
            return result + error if factor[0] is original_matrix else result

        with patch.object(implementation, 'cho_solve', new=bad_original):
            self.assert_solution(system, self.rhs, -.6)
        self.assertEqual(system.mode, 'centered')
        self.assertEqual(system.info['linear_recoveries'], 1)
        self.assertGreaterEqual(system.info['refinement_steps'], 3)

    def test_persistent_wrong_candidates_fail_after_bounded_attempts(self):
        system = KernelLinearSystem(self.K, .17)
        with patch.object(system, '_candidate', return_value=(np.zeros(7), 0.)) as bad:
            with self.assertRaisesRegex(FloatingPointError, 'Unable to solve.*accurately'):
                system.solve_constrained(self.rhs)
        self.assertGreaterEqual(bad.call_count, 2)
        # Two factor representations, each with one initial action, one
        # discarded native trial, and the unchanged three-correction budget.
        self.assertLessEqual(bad.call_count, 10)
        self.assertEqual(system.info['native_refinement_discarded'], 2)
        self.assertEqual(system.info['refinement_steps'], 6)
        self.assertIsNone(system.last_product)
        self.assertEqual(system.info['linear_solves'], 1)

    def test_nonfinite_solve_output_recovers_without_escaping_as_value_error(self):
        system = KernelLinearSystem(self.K, .17)
        original_solve = implementation.cho_solve
        calls = []

        def nonfinite_once(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return np.array([np.inf, -np.inf, 0., 0., 0., 0., 0.])
            return original_solve(*args, **kwargs)

        with patch.object(implementation, 'cho_solve', new=nonfinite_once):
            self.assert_solution(system, self.rhs, .5)
        self.assertEqual(system.mode, 'centered')

    def test_constraint_aware_rkhs_estimate_covers_actual_feature_error(self):
        # Solve the same equations with the WRONG constraint exactly. Its
        # equation residual alone is essentially zero, but its RKHS function
        # need not equal the requested-target solution. This independently
        # tests why the constraint error must enter the estimate.
        features = np.hstack([self.features, np.sqrt(.4) * np.eye(7)])
        K = features @ features.T
        system = KernelLinearSystem(K, .07)
        expected, _ = bordered_oracle(K, .07, self.rhs, 0.)
        for wrong_target in (.5, -.3, .01):
            with self.subTest(wrong_target=wrong_target):
                wrong_x, wrong_s = bordered_oracle(K, .07, self.rhs, wrong_target)
                good, residual, _, _, _, error_estimate = system._measure(
                    self.rhs, 0., wrong_x, wrong_s)
                actual_feature_error = np.linalg.norm(features.T @ (wrong_x - expected))
                self.assertFalse(good)
                self.assertLess(np.linalg.norm(residual), 1e-12)
                self.assertGreater(actual_feature_error, 1e-5)
                self.assertGreaterEqual(error_estimate + 1e-12, actual_feature_error)

    def test_read_only_inputs_multiple_rhs_and_zero_rhs(self):
        K = self.K.copy()
        rhs = self.rhs.copy()
        K.flags.writeable = rhs.flags.writeable = False
        saved_K, saved_rhs = K.copy(), rhs.copy()
        system = KernelLinearSystem(K, .23)
        self.assert_solution(system, rhs)
        self.assert_solution(system, -rhs, 1.)
        x, s = self.assert_solution(system, np.zeros(7))
        assert_array_equal(x, np.zeros(7))
        self.assertEqual(s, 0.)
        assert_array_equal(K, saved_K)
        assert_array_equal(rhs, saved_rhs)
        self.assertEqual(system.info['linear_solves'], 3)

    def test_invalid_rhs_is_rejected_without_factorization_retry(self):
        system = KernelLinearSystem(self.K, .23)
        for rhs, target in ((np.zeros(8), 0.), (np.full(7, np.nan), 0.),
                            (self.rhs, np.inf)):
            with self.subTest(shape=rhs.shape, target=target), \
                    patch.object(system, '_prepare', side_effect=AssertionError('Unexpected factorization')):
                with self.assertRaisesRegex(FloatingPointError, 'right-hand side'):
                    system.solve_constrained(rhs, target)


if __name__ == '__main__':
    unittest.main()
