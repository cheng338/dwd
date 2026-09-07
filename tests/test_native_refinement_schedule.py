"""Cheap scheduling must preserve the original equations and fallback state."""
import math
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from dwd._kernel_linear_system import KernelLinearSystem
from dwd._spectral_linear_system import SpectralLinearSystem
from dwd._compensated_residual import compensated_residual
from test_aposteriori_rkhs_bound import structured_solution, actual_error
from test_toy_residual_regression import DecimalOracle


class NativeRefinementScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limit = threadpool_limits(2)

    @classmethod
    def tearDownClass(cls):
        cls.limit.restore_original_limits()

    def systems(self, K, shift):
        D, U = eigh(K, driver='evd')
        return KernelLinearSystem(K, shift), SpectralLinearSystem(K, shift, U, np.maximum(D, 0.))

    def test_one_native_correction_avoids_compensation_and_matches_decimal_equations(self):
        rng = np.random.RandomState(921)
        X = rng.normal(size=(19, 7))
        K = X @ X.T + .2*np.eye(19)
        rhs, target, shift = rng.normal(size=19), .37, .11
        oracle = DecimalOracle(K, shift)
        for system in self.systems(K, shift):
            native = system._candidate
            calls = []
            def one_bad_solve(r, t):
                x, s = native(r, t)
                calls.append(1)
                if len(calls) == 1:
                    x = x + np.linspace(-.01, .01, len(K))
                    s += .003
                return x, s
            with self.subTest(backend=type(system).__name__), \
                    patch.object(system, '_candidate', side_effect=one_bad_solve), \
                    patch.object(system, '_compensated_measure', side_effect=AssertionError('Expensive check before native correction')):
                x, s = system.solve_constrained(rhs, target)
            r, c = oracle.residual(rhs, x, s, target)
            self.assertLess(np.max(abs(r)), 1e-11)
            self.assertLess(abs(c), 1e-13)
            self.assertEqual(len(calls), 2)
            self.assertEqual(system.info['native_refinement_trials'], 1)
            self.assertEqual(system.info['native_refinement_acceptances'], 1)
            self.assertEqual(system.info['refinement_steps'], 1)
            assert_array_equal(system.last_product, K @ x)

    def test_reference_candidate_recomputes_scores_after_native_correction(self):
        K, shift = np.diag([.01, .1, 1., 3., 10.]), .03
        rhs = np.array([.1, -2., .7, .2, 1.])
        system = self.systems(K, shift)[1]
        expected, expected_s = system._candidate(rhs, 0.)
        initial = expected + np.array([.01, -.02, .02, -.02, .01])
        initial_s = expected_s+.03
        initial_copy, initial_scores = initial.copy(), K @ initial
        with patch.object(system, '_compensated_measure', side_effect=AssertionError('Unnecessary compensation')):
            x, s, scores = system.refine_candidate(rhs, initial, initial_s, initial_scores)
        assert_array_equal(initial, initial_copy)
        assert_array_equal(scores, K @ x)
        assert_allclose(x, expected, rtol=0., atol=2e-14)
        self.assertAlmostEqual(s, expected_s, places=14)
        r, c = DecimalOracle(K, shift).residual(rhs, x, s, 0.)
        self.assertLess(np.max(abs(r)), 1e-13)
        self.assertLess(abs(c), 1e-13)
        self.assertEqual(system.info['native_refinement_trials'], 1)

    def test_failed_or_nonfinite_trial_restores_compensated_acceptable_original(self):
        n, shift = 30, 1e-6
        rng = np.random.RandomState(91300+n)
        rng.randint(-3, 4, size=(n, 5)); rng.normal(size=(n, 4))
        rhs = rng.normal(size=n)+.23
        K = np.diag(np.exp2(np.linspace(-30, 30, n)))+1024*np.ones((n, n))
        ideal, ideal_s = structured_solution(K, shift, rhs)
        x, s = np.array(list(map(float, ideal))), float(ideal_s)
        s += math.fsum(compensated_residual(K, shift, rhs, x, s, 0.)[0])/n
        original = x.copy()
        for kind in ('zero', 'nonfinite', 'exception'):
            for system in self.systems(K, shift):
                self.assertFalse(system._ordinary_measure(rhs, 0., x, s)[0])
                candidate = system._candidate
                calls = []
                def bad_trial(r, t):
                    calls.append(1)
                    if len(calls) > 1:  # The later a-posteriori action remains genuine.
                        return candidate(r, t)
                    if kind == 'exception':
                        raise FloatingPointError('Injected trial failure')
                    return np.full(n, np.nan) if kind == 'nonfinite' else np.zeros(n), 0.
                with self.subTest(backend=type(system).__name__, fault=kind), \
                        patch.object(system, '_candidate', side_effect=bad_trial):
                    got_x, got_s, measured = system._measure_with_native_trial(rhs, 0., x, s)
                self.assertTrue(measured[0])
                assert_array_equal(got_x, original)
                assert_array_equal(x, original)
                self.assertEqual(got_s, s)
                self.assertEqual(system.info['native_refinement_trials'], 1)
                self.assertEqual(system.info['native_refinement_discarded'], 1)
                self.assertEqual(system.info['refinement_steps'], 0)
                self.assertLessEqual(actual_error(K, got_x, ideal), measured[5])

    def test_persistent_failures_have_one_extra_trial_per_representation_only(self):
        K, rhs = np.eye(4), np.array([1., -1., 1., -1.])
        for system in self.systems(K, .1):
            representations = []
            def wrong_action(r, target):
                representations.append(system.info['factor_representation'])
                return np.zeros(4), 0.
            with self.subTest(backend=type(system).__name__), \
                    patch.object(system, '_candidate', side_effect=wrong_action) as action:
                with self.assertRaises(FloatingPointError):
                    system.solve_constrained(rhs)
            # Each representation has its unchanged initial action, one
            # discarded preflight and three committed corrections.
            attempts = 5 if isinstance(system, SpectralLinearSystem) else 2
            expected_modes = ({'validated_eigenbasis', 'power_of_two_equilibrated_eigenbasis',
                               'original_kernel_evd_eigenbasis', 'original_kernel_evr_eigenbasis',
                               'original_kernel_evx_eigenbasis'}
                              if isinstance(system, SpectralLinearSystem) else {'original', 'centered'})
            self.assertEqual(Counter(representations), {mode: 5 for mode in expected_modes})
            self.assertEqual(system.info['native_refinement_trials'], attempts)
            self.assertEqual(system.info['native_refinement_discarded'], attempts)
            self.assertEqual(system.info['refinement_steps'], 3*attempts)
            self.assertEqual(action.call_count, 5*attempts)
            self.assertEqual(system.info['linear_solves'], 1)
            self.assertEqual(system.info['linear_recoveries'], attempts - 1)
            self.assertNotIn('refinement_budget_extensions', system.info)
            if isinstance(system, SpectralLinearSystem):
                self.assertTrue(system.info['spectral_recovery_attempt']['success'])
                self.assertEqual([r['kind'] for r in system.info['spectral_recovery_attempts']],
                                 ['equilibrated', 'evd', 'evr', 'evx'])
                self.assertTrue(all(not r['accepted'] for r in system.info['spectral_recovery_attempts']))
            else:
                self.assertEqual(len(system.info['factorization_attempts']), 2)
            self.assertIsNone(system.last_product)

    def test_ordinary_success_is_bitwise_unchanged_and_needs_no_inverse_trial(self):
        K, rhs = np.diag([1., 2., 3., 4.]), np.array([1., -1., 2., -2.])
        for system in self.systems(K, .17):
            x, s = system._candidate(rhs, .3)
            expected = system._ordinary_measure(rhs, .3, x, s)
            self.assertTrue(expected[0])
            with patch.object(system, '_candidate', side_effect=AssertionError('Unneeded action')), \
                    patch.object(system, '_compensated_measure', side_effect=AssertionError('Unneeded compensation')):
                got_x, got_s, actual = system._measure_with_native_trial(rhs, .3, x, s, expected[3])
            self.assertIs(got_x, x)
            self.assertEqual(got_s, s)
            for got, wanted in zip(actual, expected):
                assert_array_equal(got, wanted)
            self.assertNotIn('native_refinement_trials', system.info)


if __name__ == '__main__':
    unittest.main(verbosity=2)
