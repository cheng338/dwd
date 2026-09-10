"""Mean and midpoint scalar proposals retain all original-system gates."""
import unittest
from unittest.mock import patch
import numpy as np
from dwd._kernel_linear_system import KernelLinearSystem


class InterceptFallbackRegression(unittest.TestCase):
    @staticmethod
    def fixture(rhs=None, *, kernel_scale=1., shift=1., x=None, s=0., target=0.):
        system = KernelLinearSystem(np.eye(3)*kernel_scale, shift)
        rhs = np.array([0., 0., 1.8e-10]) if rhs is None else np.asarray(rhs, float)
        x = np.zeros(3) if x is None else np.asarray(x, float)
        measurement = system._compensated_measure(rhs, target, x, s)
        return system, rhs, x, s, target, measurement

    @staticmethod
    def attempt(fixture):
        system, rhs, x, s, target, measurement = fixture
        return system._try_intercept_refinement(rhs, target, x, s, measurement)

    def test_real_midrange_proposal_rescues_mean_failure(self):
        fixture = self.fixture()
        system, rhs, x, s, target, measurement = fixture
        mean = float(np.mean(measurement[1]))
        self.assertFalse(system._equations_acceptable(rhs, target, x, measurement[1]-mean, measurement[2]))
        before = (system.K.copy(), rhs.copy(), x.copy(), system.shift, system.last_product)
        with patch.object(system, '_compensated_measure', wraps=system._compensated_measure) as fresh:
            result = self.attempt(fixture)
        self.assertIsNotNone(result)
        updated, checked = result
        self.assertAlmostEqual(updated, 9e-11, delta=1e-25)
        self.assertTrue(checked[0])
        self.assertLessEqual(checked[4], 1e-10)
        self.assertEqual(fresh.call_count, 1)
        args = fresh.call_args.args
        self.assertIs(args[0], rhs); self.assertIs(args[2], x)
        self.assertEqual(args[1], target); self.assertEqual(args[3], updated)
        np.testing.assert_array_equal(system.K, before[0])
        np.testing.assert_array_equal(rhs, before[1]); np.testing.assert_array_equal(x, before[2])
        self.assertEqual(system.shift, before[3]); self.assertIs(system.last_product, before[4])
        self.assertEqual(system.info['refinement_steps'], 1)

    def test_healthy_mean_proposal_is_kept_and_freshly_checked_once(self):
        fixture = self.fixture([.25, .25, .25])
        system = fixture[0]
        with patch.object(system, '_compensated_measure', wraps=system._compensated_measure) as fresh:
            updated, measured = self.attempt(fixture)
        self.assertEqual(updated, .25); self.assertTrue(measured[0])
        self.assertEqual(fresh.call_count, 1)
        self.assertEqual(system.info['refinement_steps'], 1)

    def test_constraint_failure_cannot_be_repaired_by_changing_intercept(self):
        fixture = self.fixture([2., 2., 2.+1.8e-10], x=[1., 1., 1.], target=0.)
        system = fixture[0]
        self.assertEqual(fixture[-1][2], -3.)
        with patch.object(system, '_compensated_measure', wraps=system._compensated_measure) as fresh:
            self.assertIsNone(self.attempt(fixture))
        self.assertEqual(fresh.call_count, 0)
        self.assertEqual(system.info['refinement_steps'], 0)

    def test_real_rkhs_failure_is_not_waived_when_midpoint_equations_pass(self):
        fixture = self.fixture(kernel_scale=1e-20, shift=1e-20)
        system, rhs, x, s, target, measurement = fixture
        midpoint = .5*(measurement[1].min()+measurement[1].max())
        self.assertTrue(system._equations_acceptable(rhs, target, x, measurement[1]-midpoint, measurement[2]))
        independently_checked = system._compensated_measure(rhs, target, x, midpoint)
        self.assertFalse(independently_checked[0])
        self.assertGreater(independently_checked[5], 5e-7)
        with patch.object(system, '_compensated_measure', wraps=system._compensated_measure) as fresh:
            self.assertIsNone(self.attempt(fixture))
        self.assertGreaterEqual(fresh.call_count, 1)
        self.assertEqual(system.info['refinement_steps'], 0)

    def test_favorable_prediction_never_overrides_fresh_failure(self):
        fixture = self.fixture([0., 0., 1e-10])
        system = fixture[0]
        rejected = (False, np.array([0., 0., 2e-10]), 0., np.zeros(3), 2e-10, 0.)
        with patch.object(system, '_equations_acceptable', return_value=True), \
             patch.object(system, '_compensated_measure', return_value=rejected) as fresh:
            self.assertIsNone(self.attempt(fixture))
        self.assertGreaterEqual(fresh.call_count, 1)
        self.assertLessEqual(fresh.call_count, 2)
        self.assertEqual(system.info['refinement_steps'], 0)

    def test_no_scalar_can_satisfy_equation_range_wider_than_twice_tolerance(self):
        fixture = self.fixture([0., 0., 3e-10])
        system = fixture[0]
        with patch.object(system, '_compensated_measure', wraps=system._compensated_measure) as fresh:
            self.assertIsNone(self.attempt(fixture))
        self.assertEqual(fresh.call_count, 0)
        self.assertEqual(system.info['refinement_steps'], 0)

    def test_unrepresentable_scalar_change_does_not_count_as_refinement(self):
        fixture = self.fixture()
        system, rhs, x, _, target, measurement = fixture
        s = float(2**52)
        # The only two suggested deltas are below half an ULP of s.
        fixture = (system, rhs, x, s, target, measurement)
        with patch.object(system, '_compensated_measure') as fresh:
            self.assertIsNone(self.attempt(fixture))
        self.assertEqual(fresh.call_count, 0)
        self.assertEqual(system.info['refinement_steps'], 0)

    def test_overflow_cannot_authorize_a_candidate(self):
        system = KernelLinearSystem(np.eye(3), 1.)
        rhs = np.full(3, 1e308); x = np.zeros(3)
        measurement = (False, np.full(3, 1e308), 0., np.zeros(3), 1e308, 1e308)
        with patch.object(system, '_compensated_measure') as fresh, np.errstate(over='raise', invalid='raise'):
            try:
                result = system._try_intercept_refinement(rhs, 0., x, 1e308, measurement)
            except FloatingPointError:
                result = None  # Existing bounded recovery may receive this failure.
        self.assertIsNone(result)
        self.assertEqual(fresh.call_count, 0)
        self.assertEqual(system.info['refinement_steps'], 0)


if __name__ == '__main__':
    unittest.main()
