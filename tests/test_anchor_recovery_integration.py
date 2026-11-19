"""Lazy anchor recovery retains the original constrained-system checks."""
import math
import unittest
from unittest.mock import patch

import numpy as np

from dwd._anchor_linear_system import AnchorLinearSystem
from dwd._kernel_linear_system import KernelLinearSystem
from dwd._spectral_linear_system import SpectralLinearSystem


def bordered_oracle(kernel, shift, rhs, target):
    """Independent dense solve, without eliminating the intercept/constraint."""
    n = len(kernel)
    bordered = np.empty((n + 1, n + 1))
    bordered[:n, :n] = kernel + shift * np.eye(n)
    bordered[:n, n] = 1.
    bordered[n, :n] = 1.
    bordered[n, n] = 0.
    return np.linalg.solve(bordered, np.r_[rhs, target])


class AnchorRecoveryIntegration(unittest.TestCase):
    @staticmethod
    def fixture():
        features = np.array([[1., .2], [-.3, .7], [.4, -.1], [.2, .9]])
        return features @ features.T + .3 * np.eye(4), .2, np.array([.4, -.7, .2, .8])

    def assert_original_solution(self, system, rhs, target, x, scalar):
        expected = bordered_oracle(system.K, system.shift, rhs, target)
        np.testing.assert_allclose(x, expected[:-1], rtol=2e-13, atol=2e-13)
        self.assertAlmostEqual(scalar, expected[-1], delta=2e-13)
        np.testing.assert_allclose(system.K @ x + system.shift * x + scalar,
                                   rhs, rtol=0., atol=1e-12)
        self.assertAlmostEqual(math.fsum(x), target, delta=1e-13)
        self.assertTrue(system._measure(rhs, target, x, scalar)[0])
        np.testing.assert_allclose(system.last_product, system.K @ x,
                                   rtol=1e-13, atol=1e-13)

    def test_healthy_original_actions_never_instantiate_anchor(self):
        kernel, shift, rhs = self.fixture()
        before = kernel.copy()
        with patch('dwd._anchor_linear_system.AnchorLinearSystem',
                   side_effect=AssertionError('Healthy solve allocated anchor LU')) as helper:
            system = KernelLinearSystem(kernel, shift)
            for target in (0., .35):
                x, scalar = system.solve_constrained(rhs, target)
                self.assert_original_solution(system, rhs, target, x, scalar)
        helper.assert_not_called()
        self.assertEqual(system.mode, 'original')
        self.assertIsNone(system.anchor_action)
        self.assertEqual(system.info['linear_recoveries'], 0)
        self.assertEqual(len(system.info['factorization_attempts']), 1)
        np.testing.assert_array_equal(kernel, before)

    def test_failed_cholesky_actions_recover_and_reuse_anchor_against_oracle(self):
        kernel, shift, rhs = self.fixture()
        before, rhs_before = kernel.copy(), rhs.copy()
        system = KernelLinearSystem(kernel, shift)
        original_candidate = system._candidate
        visited = []

        def candidate(right, target):
            visited.append(system.mode)
            if system.mode != 'anchor':
                raise FloatingPointError('Forced numerical inverse-action failure')
            return original_candidate(right, target)

        with patch.object(system, '_candidate', side_effect=candidate), \
             patch('dwd._anchor_linear_system.AnchorLinearSystem', wraps=AnchorLinearSystem) as helper:
            x, scalar = system.solve_constrained(rhs, -.35)
            self.assert_original_solution(system, rhs, -.35, x, scalar)
            factor = system.anchor_action.factor
            second_rhs = rhs[::-1].copy()
            x2, scalar2 = system.solve_constrained(second_rhs, .25)
            self.assert_original_solution(system, second_rhs, .25, x2, scalar2)
        helper.assert_called_once()
        self.assertEqual(visited, ['original', 'centered', 'anchor', 'anchor'])
        self.assertIs(system.anchor_action.factor, factor)
        self.assertIs(system.anchor_action.K, kernel)
        self.assertEqual(system.info['factor_representation'], 'anchor_lu')
        self.assertEqual(system.info['anchor_accepted_actions'], 2)
        self.assertEqual(system.info['linear_recoveries'], 2)
        self.assertIsNone(system.factor)
        self.assertIsNone(system.v)
        self.assertEqual(system.info['added_objective_regularization'], 0.)
        self.assertEqual(system.info['positive_eigenvalues_discarded'], 0)
        np.testing.assert_array_equal(kernel, before)
        np.testing.assert_array_equal(rhs, rhs_before)

    def test_initial_factorization_failures_reach_anchor_once(self):
        kernel, shift, rhs = self.fixture()
        with patch('dwd._kernel_linear_system.cho_factor',
                   side_effect=FloatingPointError('Forced factorization failure')) as cholesky, \
             patch('dwd._anchor_linear_system.AnchorLinearSystem', wraps=AnchorLinearSystem) as helper:
            system = KernelLinearSystem(kernel, shift)
            x, scalar = system.solve_constrained(rhs, .125)
        self.assertEqual(cholesky.call_count, 2)
        helper.assert_called_once()
        self.assertEqual(system.mode, 'anchor')
        self.assert_original_solution(system, rhs, .125, x, scalar)

    def test_centered_preparation_failure_after_original_action_uses_anchor(self):
        kernel, shift, rhs = self.fixture()
        system = KernelLinearSystem(kernel, shift)
        original_candidate = system._candidate

        def candidate(right, target):
            if system.mode == 'original':
                raise FloatingPointError('Forced original action failure')
            return original_candidate(right, target)

        with patch.object(system, '_candidate', side_effect=candidate), \
             patch('dwd._kernel_linear_system.cho_factor',
                   side_effect=FloatingPointError('Forced centered preparation failure')) as cholesky:
            x, scalar = system.solve_constrained(rhs)
        self.assertEqual(cholesky.call_count, 1)
        self.assertEqual(system.mode, 'anchor')
        self.assert_original_solution(system, rhs, 0., x, scalar)

    def test_invalid_cholesky_condition_estimates_allow_valid_lu_recovery(self):
        kernel, shift, rhs = self.fixture()
        with patch('dwd._kernel_linear_system.dpocon', return_value=(np.nan, 0)) as estimator:
            system = KernelLinearSystem(kernel, shift)
            x, scalar = system.solve_constrained(rhs, -.1)
        self.assertEqual(estimator.call_count, 2)
        self.assertEqual(system.mode, 'anchor')
        self.assertGreater(system.anchor_action.info['rcond'], 0.)
        self.assert_original_solution(system, rhs, -.1, x, scalar)

    def test_bad_anchor_candidates_cannot_bypass_any_fresh_gate(self):
        fixtures = (
            ('equation', np.eye(3), 1., np.array([.25, -.25, 0.]), np.zeros(3), 0.),
            ('constraint', np.ones((3, 3)), 1., np.full(3, 2.), np.full(3, .5), 0.),
            ('rkhs', np.eye(3) * 1e-20, 1e-20,
             np.array([0., 0., 1.8e-10]), np.zeros(3), 9e-11),
        )
        for gate, kernel, shift, rhs, bad_x, bad_scalar in fixtures:
            with self.subTest(gate=gate):
                system = KernelLinearSystem(kernel, shift)
                candidate_count = 0
                checked = []
                original_measure = system._measure

                def candidate(right, target):
                    nonlocal candidate_count
                    if system.mode != 'anchor':
                        raise FloatingPointError('Forced earlier representation failure')
                    candidate_count += 1
                    if candidate_count != 1:
                        # Also prevents the auxiliary RKHS inverse action from
                        # inventing an accurate correction to the bad state.
                        raise FloatingPointError('No usable anchor correction')
                    return bad_x.copy(), bad_scalar

                def measure(right, target, x, scalar, product=None):
                    self.assertEqual(system.mode, 'anchor')
                    self.assertIsNone(product)  # No stale/precomputed scores.
                    result = original_measure(right, target, x, scalar, product)
                    checked.append(result)
                    return result

                with patch.object(system, '_candidate', side_effect=candidate), \
                     patch.object(system, '_measure', side_effect=measure), \
                     patch.object(system, '_measure_with_native_trial',
                                  side_effect=AssertionError('Anchor used ordinary inverse preflight')):
                    with self.assertRaisesRegex(FloatingPointError, 'Unable to solve the original'):
                        system.solve_constrained(rhs)
                self.assertEqual(len(checked), 1)
                self.assertFalse(checked[0][0])
                if gate == 'equation':
                    self.assertGreater(checked[0][4], 1e-10)
                elif gate == 'constraint':
                    self.assertLessEqual(checked[0][4], 1e-10)
                    self.assertGreater(abs(checked[0][2]), 1e-12)
                else:
                    self.assertLessEqual(checked[0][4], 1e-10)
                    self.assertEqual(checked[0][2], 0.)
                    self.assertGreater(checked[0][5], 5e-7)
                self.assertIsNone(system.last_product)
                self.assertEqual(system.info['anchor_accepted_actions'], 0)
                self.assertLessEqual(candidate_count, 5)

    def test_all_inverse_actions_failing_is_bounded(self):
        kernel, shift, rhs = self.fixture()
        system = KernelLinearSystem(kernel, shift)
        visited = []

        def fail(right, target):
            visited.append(system.mode)
            raise FloatingPointError('Forced inverse-action failure')

        with patch.object(system, '_candidate', side_effect=fail), \
             patch('dwd._anchor_linear_system.AnchorLinearSystem', wraps=AnchorLinearSystem) as helper:
            with self.assertRaisesRegex(FloatingPointError, 'bounded Cholesky, centered and anchor'):
                system.solve_constrained(rhs)
        self.assertEqual(visited, ['original', 'centered', 'anchor'])
        helper.assert_called_once()
        self.assertEqual(len(system.info['factorization_attempts']), 3)
        self.assertEqual(system.info['refinement_steps'], 0)
        self.assertEqual(system.info['anchor_accepted_actions'], 0)
        self.assertIsNone(system.last_product)

    def test_all_factorizations_failing_is_bounded(self):
        kernel, shift, _ = self.fixture()
        with patch('dwd._kernel_linear_system.cho_factor',
                   side_effect=FloatingPointError('Forced Cholesky failure')) as cholesky, \
             patch('dwd._anchor_linear_system.AnchorLinearSystem',
                   side_effect=FloatingPointError('Forced anchor factorization failure')) as helper:
            with self.assertRaisesRegex(FloatingPointError, 'anchor factorization failure'):
                KernelLinearSystem(kernel, shift)
        self.assertEqual(cholesky.call_count, 2)
        helper.assert_called_once()

    def test_reference_healthy_and_failed_actions_never_use_anchor(self):
        kernel, shift, rhs = self.fixture()
        values, vectors = np.linalg.eigh(kernel)
        with patch('dwd._anchor_linear_system.AnchorLinearSystem',
                   side_effect=AssertionError('Reference switched to anchor LU')) as helper:
            system = SpectralLinearSystem(kernel, shift, vectors, values)
            x, scalar = system.solve_constrained(rhs, .2)
            self.assert_original_solution(system, rhs, .2, x, scalar)
            with patch.object(system, '_candidate',
                              side_effect=FloatingPointError('Forced spectral failure')) as candidate:
                with self.assertRaisesRegex(FloatingPointError, 'no Cholesky substitution'):
                    system.solve_constrained(rhs)
            self.assertLessEqual(candidate.call_count, 5)
        helper.assert_not_called()
        self.assertNotEqual(system.info['factor_representation'], 'anchor_lu')
        self.assertNotIn('anchor_accepted_actions', system.info)


if __name__ == '__main__':
    unittest.main()
