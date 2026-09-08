"""Native arithmetic may accelerate checks, never weaken solver gates."""
import unittest
from unittest.mock import patch
import numpy as np
import dwd._kernel_linear_system as linear


class NativeResidualIntegrationTests(unittest.TestCase):
    def test_accepted_bounded_native_state_avoids_expanded_pass(self):
        system = linear.KernelLinearSystem(np.eye(4), .1)
        z = np.zeros(4)
        native = (z, 0., z, np.full(4, 1e-25), 1e-30, np.full(4, 1e-30))
        with patch.object(linear, 'native_compensated_residual', return_value=native), patch.object(
                linear, 'compensated_residual', side_effect=AssertionError('Redundant expanded pass')):
            result = system._compensated_measure(z, 0., z, 0.)
        self.assertTrue(result[0])
        self.assertEqual(system.info['native_residual_acceptances'], 1)

    def test_true_equation_failure_can_refine_without_redundant_pass(self):
        system = linear.KernelLinearSystem(np.eye(4), .1)
        z = np.zeros(4)
        rhs = np.array([1e-8, -1e-8, 0., 0.])
        native = (rhs, 0., z, np.full(4, 1e-24), 1e-30, np.full(4, 1e-30))
        with patch.object(linear, 'native_compensated_residual', return_value=native), patch.object(
                linear, 'compensated_residual', side_effect=AssertionError('Failure already established')):
            result = system._compensated_measure(rhs, 0., z, 0.)
        self.assertFalse(result[0])
        self.assertGreater(result[4], 1e-10)
        self.assertEqual(system.info['native_residual_equation_failures'], 1)

    def test_uncertain_boundary_rechecks_unchanged_original_state(self):
        system = linear.KernelLinearSystem(np.eye(4), .1)
        z = np.zeros(4)
        rhs = np.array([1e-10, -1e-10, 0., 0.])
        native = (rhs, 0., z, np.full(4, 2e-20), 1e-30, np.full(4, 1e-30))
        expanded = (rhs, 0., z, z, 0.)
        with patch.object(linear, 'native_compensated_residual', return_value=native), patch.object(
                linear, 'compensated_residual', return_value=expanded) as fallback:
            result = system._compensated_measure(rhs, 0., z, 0.)
        self.assertTrue(result[0])
        self.assertEqual(fallback.call_count, 1)
        self.assertIs(fallback.call_args.args[2], rhs)
        self.assertIs(fallback.call_args.args[3], z)
        self.assertEqual(system.info['native_residual_fallbacks'], 1)

    def test_unsupported_native_path_uses_existing_helper(self):
        system = linear.KernelLinearSystem(np.eye(4), .1)
        z = np.zeros(4)
        with patch.object(linear, 'native_compensated_residual', return_value=None), patch.object(
                linear, 'compensated_residual', wraps=linear.compensated_residual) as fallback:
            result = system._compensated_measure(z, 0., z, 0.)
        self.assertTrue(result[0])
        self.assertEqual(fallback.call_count, 1)
        self.assertNotIn('native_residual_checks', system.info)

    def test_uncertain_score_cannot_enlarge_relative_rkhs_tolerance(self):
        system = linear.KernelLinearSystem(np.eye(4), 1e-12)
        x = np.ones(4)
        r = np.array([1e-11, -1e-11, 1e-11, -1e-11])
        native = (r, 0., np.full(4, 1e8), np.full(4, 1e-30), 1e-30,
                  np.full(4, 1e8))
        with patch.object(linear, 'native_compensated_residual', return_value=native):
            self.assertIsNone(system._bounded_native_measure(np.ones(4), 4., x, 0.))
        self.assertEqual(system.info['native_residual_fallbacks'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
