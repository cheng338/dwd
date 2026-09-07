"""Scalar coordinate refinement must satisfy unchanged original equations."""
import math
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from dwd._kernel_linear_system import KernelLinearSystem
from dwd._spectral_linear_system import SpectralLinearSystem
from test_toy_residual_regression import DecimalOracle


class InterceptResidualRefinementTests(unittest.TestCase):
    def test_nearly_constant_rbf_previously_rejected_by_optimized_matches_decimal_equations(self):
        with threadpool_limits(4):
            for n in (30, 75, 120):
                with self.subTest(n=n):
                    rng = np.random.RandomState(91300+n)
                    rng.randint(-3, 4, size=(n, 5))  # Same saved stress-stream state.
                    K = rbf_kernel(rng.normal(size=(n, 4)), gamma=1e-5)
                    rhs = rng.normal(size=n)+.23
                    shift = 1e-6
                    oracle = DecimalOracle(K, shift)
                    D, U = eigh(K, driver='evd')
                    for system in (KernelLinearSystem(K, shift), SpectralLinearSystem(K, shift, U, np.maximum(D, 0.))):
                        x, s = system.solve_constrained(rhs)
                        residual, constraint = oracle.residual(rhs, x, s)
                        self.assertLessEqual(np.max(abs(residual)), 1e-10*max(1., np.max(abs(rhs))))
                        self.assertLessEqual(abs(constraint), 64*np.finfo(float).eps*max(1., math.fsum(abs(x))))
                        self.assertLessEqual(system.info['refinement_steps'], 6)
                        # BLAS/LAPACK rounding determines whether a scalar correction
                        # is needed; the injected case below exercises it explicitly.
                        intercept_steps = system.info.get('intercept_refinement_steps', 0)
                        self.assertGreaterEqual(intercept_steps, 0)
                        self.assertLessEqual(intercept_steps, system.info['refinement_steps'])

    def test_constant_residual_is_repaired_without_changing_any_coefficient(self):
        K = np.diag([.2, .7, 1.3, 2.])
        system = KernelLinearSystem(K, .1)
        rhs = np.array([.3, -.1, .7, -.4])
        x, s = system.solve_constrained(rhs)
        original = x.copy()
        bad_s = s+1e-5
        measured = system._measure(rhs, 0., x, bad_s)
        self.assertFalse(measured[0])
        updated, checked = system._try_intercept_refinement(rhs, 0., x, bad_s, measured)
        self.assertTrue(checked[0])
        assert_array_equal(x, original)
        self.assertAlmostEqual(updated, s, delta=2e-15)
        self.assertEqual(system.info['intercept_refinement_steps'], 1)
        self.assertEqual(system.info['refinement_steps'], 1)

    def test_unrepresentable_scalar_update_is_not_counted_as_refinement(self):
        system = KernelLinearSystem(np.zeros((4, 4)), .1)
        x, product = np.zeros(4), np.zeros(4)
        measurement = (False, np.full(4, .25), 0., product, .25, 0.)
        with patch.object(system, '_compensated_measure', side_effect=AssertionError('Unrepresentable update retried')):
            corrected = system._try_intercept_refinement(np.full(4, 1e16), 0., x, 1e16, measurement)
        self.assertIsNone(corrected)
        self.assertEqual(system.info['refinement_steps'], 0)

    def test_prediction_cannot_authorize_a_failed_exact_recheck(self):
        system = KernelLinearSystem(np.eye(4), .1)
        x, product = np.zeros(4), np.zeros(4)
        measurement = (False, np.ones(4), 0., product, 1., 0.)
        # Simulate a failed actual-equation check after a promising prediction.
        with patch.object(system, '_compensated_measure', return_value=measurement) as rechecked:
            corrected = system._try_intercept_refinement(np.ones(4), 0., x, 0., measurement)
        self.assertIsNone(corrected)
        self.assertEqual(rechecked.call_count, 1)
        self.assertEqual(system.info['refinement_steps'], 0)
        self.assertEqual(system.info['intercept_refinement_attempts'], 1)

    def test_scalar_refinement_cannot_hide_wrong_coefficient_sum(self):
        system = KernelLinearSystem(np.eye(4), .1)
        x = np.ones(4)
        product = x.copy()
        measurement = (False, np.ones(4), -4., product, 1., 1.)
        with patch.object(system, '_compensated_measure', side_effect=AssertionError('Failed constraint ignored')):
            self.assertIsNone(system._try_intercept_refinement(np.full(4, 2.1), 0., x, 0., measurement))
        self.assertEqual(system.info['refinement_steps'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
