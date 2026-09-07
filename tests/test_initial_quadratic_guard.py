"""Initial-state arithmetic checks, independent of optimizer convergence."""
from decimal import Decimal, localcontext
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from dwd._kernel_solver import (
    _check_initial_quadratic, _compensated_dot, _compensated_quadratic, solve_kernel,
)
from dwd.gen_dwd import V
from dwd.gen_kern_dwd import KernGDWD


def exact_dot(a, b):
    with localcontext() as context:
        context.prec = 2000
        return sum((Decimal.from_float(float(x)) * Decimal.from_float(float(y))
                    for x, y in zip(a, b)), Decimal(0))


class InitialQuadraticGuardTests(unittest.TestCase):
    def test_compensated_product_and_sum_bound_against_decimal(self):
        cases = [
            ([2.**54, 1., -2.**54, 3.], [1., 1., 1., 1.]),
            ([.1, -.3, 1e100, -1e100], [.7, .2, 1e-100, 1e-100]),
            ([1e-200, -2e-200, 3e-200], [1e-200, 1e-200, 1e-200]),
            ([np.finfo(float).tiny, np.nextafter(0., 1.)], [.5, .5]),
        ]
        for a, b in cases:
            with self.subTest(a=a):
                value, error = _compensated_dot(a, b)
                with localcontext() as context:
                    context.prec = 2000
                    difference = abs(Decimal.from_float(value) - exact_dot(a, b))
                    self.assertLessEqual(difference, Decimal.from_float(error))
                self.assertTrue(np.isfinite(error))

    def test_rowwise_quadratic_bound_covers_decimal_spd_and_singular_values(self):
        rng = np.random.RandomState(341)
        for rank in (3, 8):
            features = rng.normal(size=(8, rank))
            K = features @ features.T
            alpha = rng.normal(size=8) * 13
            _, quadratic, error = _compensated_quadratic(K, alpha)
            with localcontext() as context:
                context.prec = 2000
                exact = sum((Decimal.from_float(float(alpha[i])) * exact_dot(K[i], alpha)
                             for i in range(len(alpha))), Decimal(0))
                self.assertLessEqual(abs(Decimal.from_float(quadratic) - exact),
                                     Decimal.from_float(error))

    def test_overconservative_bound_can_pass_without_changing_observed_state(self):
        K = np.ones((64, 64))
        alpha = np.tile([32., -32.], 32)
        original = alpha.tobytes()
        scores, quadratic, error, info = _check_initial_quadratic(K, alpha)
        threshold = np.sqrt(np.finfo(float).eps)
        self.assertGreater(info['initial_quadratic_fast_error_bound'], threshold)
        self.assertLess(error, threshold)
        self.assertEqual(info['initial_quadratic_method'], 'compensated_observed_check')
        self.assertEqual(quadratic, 0.)
        assert_array_equal(scores, K @ alpha)
        self.assertEqual(alpha.tobytes(), original)

    def test_actual_quadratic_discrepancy_is_included_in_acceptance(self):
        K, alpha = np.ones((64, 64)), np.tile([32., -32.], 32)
        # A small error bar for another calculation must not certify the actual
        # ordinary score/quadratic if the two calculations materially disagree.
        with patch('dwd._kernel_solver._compensated_quadratic',
                   return_value=(np.zeros(64), .01, 1e-12)):
            with self.assertRaisesRegex(FloatingPointError, 'cancellation-prone'):
                _check_initial_quadratic(K, alpha)

    def test_huge_nullspace_coefficients_still_fail(self):
        for magnitude in (1e8, 1e100):
            with self.subTest(magnitude=magnitude):
                with self.assertRaisesRegex(FloatingPointError, 'cancellation-prone'):
                    _check_initial_quadratic(np.ones((64, 64)),
                                             np.tile([magnitude, -magnitude], 32))

    def test_nonfinite_products_or_accumulation_fail_closed(self):
        with self.assertRaises(FloatingPointError):
            _compensated_dot([1e308], [1e308])
        with self.assertRaises(FloatingPointError):
            _compensated_dot([1e308, 1e308], [1., 1.])
        with self.assertRaises(FloatingPointError):
            _check_initial_quadratic(np.eye(4) * 1e308, np.full(4, 1e308))

    def test_unsupported_flush_to_zero_arithmetic_fails_closed(self):
        with patch('dwd._kernel_solver.np.multiply', return_value=np.float64(0.)):
            with self.assertRaisesRegex(FloatingPointError, 'gradual float64 underflow'):
                _compensated_dot([1.], [1.])

    def test_compensated_check_respects_callers_underflow_raise_setting(self):
        with np.errstate(under='raise'):
            value, error = _compensated_dot([1e-200, 2e-200], [1e-200, 1e-200])
            self.assertEqual(value, 0.)
            self.assertGreater(error, 0.)
            self.assertEqual(np.geterr()['under'], 'raise')

    def test_zero_and_ordinary_safe_states_keep_fast_paths(self):
        K = np.diag([1., 2., 3., 4.])
        with patch('dwd._kernel_solver._compensated_quadratic',
                   side_effect=AssertionError('Unexpected compensated fallback')):
            zeros, quadratic, bound, info = _check_initial_quadratic(K, np.zeros(4))
            assert_array_equal(zeros, np.zeros(4))
            self.assertEqual((quadratic, bound, info['initial_quadratic_method']), (0., 0., 'zero'))
            alpha = np.array([.1, -.2, .3, -.4])
            scores, quadratic, bound, info = _check_initial_quadratic(K, alpha)
            assert_array_equal(scores, K @ alpha)
            self.assertEqual(quadratic, float(alpha @ (K @ alpha)))
            self.assertEqual(info['initial_quadratic_method'], 'ordinary_forward_bound')

    def test_public_native_gaussian_vector_and_free_intercept_are_unchanged(self):
        seed, n = 314159, 64
        expected_alpha = np.random.RandomState(seed).normal(size=n)
        expected_alpha /= np.linalg.norm(expected_alpha)
        # A manufactured SPD matrix makes the generic n-stage bound loose,
        # while preserving a small positive norm in the native initial direction.
        K = 1e6 * (np.eye(n) - np.outer(expected_alpha, expected_alpha)) + .01 * np.eye(n)
        labels = np.tile([-1, 1], n // 2)
        seen, checked = [], []
        def observe_initial_check(matrix, alpha):
            state = _check_initial_quadratic(matrix, alpha)
            checked.append((state[0].copy(), state[1]))
            return state
        model = KernGDWD(kernel='precomputed', implementation='reference',
                        random_state=seed, max_iter=0, q=1.7, lambd=.03,
                        callback=lambda state: seen.append(state))
        with patch('dwd._kernel_solver._check_initial_quadratic', side_effect=observe_initial_check):
            model.fit(K, labels, offset_init=.17)
        self.assertEqual(model.dual_coef_.tobytes(), expected_alpha.tobytes())
        self.assertEqual(float(model.intercept_[0]), .17)
        self.assertEqual(model.diagnostics_['initial_quadratic_method'], 'compensated_observed_check')
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]['alpha'].tobytes(), expected_alpha.tobytes())
        # Compare to the actual checked calculation, not another BLAS replay
        # whose reduction may differ with memory alignment under cancellation.
        self.assertEqual(len(checked), 1)
        assert_array_equal(seen[0]['training_scores'], checked[0][0])
        expected_objective = np.mean(V(labels * (seen[0]['training_scores'] + .17), q=1.7))
        expected_objective += .03 * checked[0][1]
        self.assertEqual(model.objective_history_[0], expected_objective)

    def test_both_implementations_reuse_checked_initial_state(self):
        n = 64
        K, alpha = np.ones((n, n)), np.tile([32., -32.], n // 2)
        for implementation in ('reference', 'optimized'):
            with self.subTest(implementation=implementation):
                seen = []
                fit = solve_kernel(K, np.tile([-1., 1.], n // 2), .03,
                                   implementation=implementation, alpha_init=alpha,
                                   offset_init=.17, max_iter=0, psd_known=True,
                                   callback=lambda state: seen.append(state))
                self.assertEqual(fit['alpha'].tobytes(), alpha.tobytes())
                self.assertEqual(fit['offset'], .17)
                self.assertEqual(fit['diagnostics']['initial_quadratic_method'],
                                 'compensated_observed_check')
                assert_array_equal(seen[0]['training_scores'], np.zeros(n))
                self.assertEqual(fit['n_iter'], 0)


if __name__ == '__main__':
    unittest.main()
