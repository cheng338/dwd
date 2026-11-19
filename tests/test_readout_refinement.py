"""Adjacent-float readouts are checked against unchanged original equations."""
import hashlib
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from dwd._compensated_residual import compensated_residual
from dwd._readout_refinement import ReadoutBudget, polish_readout
import dwd._readout_refinement as readout
import dwd._kernel_linear_system as linear
from dwd._spectral_linear_system import SpectralLinearSystem
from test_toy_residual_regression import DecimalOracle


class ReadoutRefinementTests(unittest.TestCase):
    def small_case(self):
        K = np.ones((2, 2))
        shift, rhs, scalar, target = 1e-10, np.array([1., -1.]), 0., 0.
        x = np.array([np.nextafter(1e10, np.inf), -1e10])
        D, U = eigh(K)
        system = SpectralLinearSystem(K, shift, U, D)
        return K, shift, rhs, x, scalar, target, system

    def check_oracle(self, K, shift, rhs, x, scalar, target):
        r, c = DecimalOracle(K, shift).residual(rhs, x, scalar, target)
        self.assertLessEqual(np.max(abs(r)), 1e-10*max(1., np.max(abs(rhs))))
        self.assertLessEqual(abs(c), 64*np.finfo(float).eps*max(1., math.fsum(abs(x)), abs(target)))

    def test_adjacent_move_is_reproducible_and_preserves_inputs(self):
        K, shift, rhs, x, scalar, target, system = self.small_case()
        saved = [a.copy() for a in (K, rhs, x)]
        first, details = polish_readout(K, shift, rhs, x, scalar, target, system._compensated_measure, ReadoutBudget())
        self.assertIsNotNone(first)
        self.assertEqual(details['moves'], 1)
        changed = np.flatnonzero(first[0] != x)
        self.assertEqual(len(changed), 1)
        i = changed[0]
        self.assertIn(first[0][i], [np.nextafter(x[i], -np.inf), np.nextafter(x[i], np.inf)])
        self.check_oracle(K, shift, rhs, first[0], first[1], target)
        again, again_details = polish_readout(K, shift, rhs, x, scalar, target, system._compensated_measure, ReadoutBudget())
        assert_array_equal(first[0], again[0])
        self.assertEqual(first[1], again[1])
        self.assertEqual(details, again_details)
        for original, after in zip(saved, (K, rhs, x)):
            assert_array_equal(original, after)

    def test_cached_case_permutation_and_rhs_perturbation_pass_decimal_equations(self):
        file = Path(__file__).parent/'fixtures/spectral_n120_cached_evd.npz'
        self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),
                         '3cc44a64e2d8117747d38280afc9d5ed922d6a3bb462eda8d519c0139d7d49ad')
        with np.load(file) as saved:
            K, rhs, D, U = [saved[key].copy() for key in ('K', 'rhs', 'values', 'vectors')]
        permutation = np.random.RandomState(734).permutation(len(rhs))
        perturbation = 1e-8*np.random.RandomState(932).normal(size=len(rhs))
        with threadpool_limits(2):
            for case in ('original', 'permuted', 'perturbed'):
                with self.subTest(case=case):
                    matrix = K[np.ix_(permutation, permutation)] if case == 'permuted' else K.copy()
                    right = rhs[permutation] if case == 'permuted' else rhs.copy()
                    vectors = U[permutation] if case == 'permuted' else U.copy()
                    if case == 'perturbed':
                        right += perturbation
                    before = matrix.copy(), right.copy(), vectors.copy()
                    system = SpectralLinearSystem(matrix, 1e-10, vectors, np.maximum(D, 0.))
                    with patch.object(linear, 'cho_factor', side_effect=AssertionError('Reference changed backend')):
                        x, s = system.solve_constrained(right, .37)
                    self.check_oracle(matrix, 1e-10, right, x, s, .37)
                    self.assertLessEqual(system.info.get('readout_action_moves', 0), 8)
                    self.assertLessEqual(system.info.get('readout_action_matrix_entry_work', 0), 1_048_576)
                    self.assertEqual(system.info['linear_solves'], 1)
                    self.assertEqual(system.info['added_objective_regularization'], 0.)
                    self.assertEqual(system.info['positive_eigenvalues_discarded'], 0)
                    for old, new in zip(before, (matrix, right, vectors)):
                        assert_array_equal(old, new)

    def test_healthy_solve_never_invokes_readout_helper(self):
        K = np.diag([.1, .3, .8, 1.2])
        system = SpectralLinearSystem(K, .2, np.eye(4), K.diagonal())
        with patch.object(system, '_try_readout_refinement', side_effect=AssertionError('Healthy path changed')):
            x, scalar = system.solve_constrained(np.array([1., -1., .2, .5]), .37)
        self.check_oracle(K, .2, np.array([1., -1., .2, .5]), x, scalar, .37)
        self.assertNotIn('readout_refinement_attempts', system.info)

    def test_budget_is_consumed_across_calls_and_gates_before_measurement(self):
        K, shift, rhs, x, scalar, target, system = self.small_case()
        budget = ReadoutBudget(max_moves=1)
        first, _ = polish_readout(K, shift, rhs, x, scalar, target, system._compensated_measure, budget)
        self.assertIsNotNone(first)
        work = budget.matrix_entry_work
        with patch.object(readout, 'compensated_residual', side_effect=AssertionError('Exhausted budget did work')):
            result, details = polish_readout(K, shift, rhs, x, scalar, target, system._compensated_measure, budget)
        self.assertIsNone(result)
        self.assertEqual(details['status'], 'budget_exhausted')
        self.assertEqual(budget.matrix_entry_work, work)
        with patch.object(readout, 'compensated_residual', side_effect=AssertionError('Small budget did work')):
            result, details = polish_readout(K, shift, rhs, x, scalar, target, system._compensated_measure,
                                             ReadoutBudget(max_matrix_entry_work=4*len(x)**2-1))
        self.assertIsNone(result)
        self.assertEqual(details['matrix_entry_work'], 0)

    def test_failed_final_gate_rolls_back_even_after_improved_residual(self):
        K, shift, rhs, x, scalar, target, system = self.small_case()
        saved = x.copy()
        calls = []
        def reject(*args):
            calls.append(args)
            measured = system._compensated_measure(*args)
            self.assertTrue(measured[0])
            return (False, *measured[1:])
        result, details = polish_readout(K, shift, rhs, x, scalar, target, reject, ReadoutBudget())
        self.assertIsNone(result)
        self.assertEqual(details['status'], 'full_accuracy_check_failed')
        self.assertEqual(len(calls), 1)
        assert_array_equal(x, saved)

    def test_prediction_of_improvement_cannot_bypass_fresh_residual(self):
        K, shift, rhs, x, scalar, target, system = self.small_case()
        initial = compensated_residual(K, shift, rhs, x, scalar, target)
        with patch.object(readout, 'compensated_residual', return_value=initial) as measure:
            result, details = polish_readout(K, shift, rhs, x, scalar, target,
                                             lambda *args: self.fail('Unimproved state reached gate'), ReadoutBudget())
        self.assertIsNone(result)
        self.assertEqual(details['status'], 'fresh_residual_did_not_improve')
        self.assertEqual(details['moves'], 0)
        self.assertEqual(measure.call_count, 2)

    def test_wrong_coefficients_cannot_be_repaired_by_relaxing_target_or_gate(self):
        K = np.eye(4)
        system = SpectralLinearSystem(K, .1, K.copy(), np.ones(4))
        x = np.zeros(4)
        with patch.object(system, '_candidate', return_value=(x, 0.)):
            with self.assertRaisesRegex(FloatingPointError, 'at most five representations'):
                system.solve_constrained(np.array([1., -1., 1., -1.]))
        self.assertLessEqual(system.info.get('readout_action_matrix_entry_work', 0), 1_048_576)
        self.assertEqual(system.info.get('readout_refinement_acceptances', 0), 0)
        self.assertIsNone(system.last_product)
        assert_array_equal(x, np.zeros(4))

    def test_readout_cannot_bypass_the_real_coefficient_sum_gate(self):
        K, shift, rhs, x, scalar, _, system = self.small_case()
        result, details = polish_readout(K, shift, rhs, x, scalar, 1.,
                                         system._compensated_measure, ReadoutBudget())
        self.assertIsNone(result)
        self.assertEqual(details['status'], 'full_accuracy_check_failed')
        self.assertGreater(details['moves'], 0)

    def test_all_inverse_attempts_share_one_readout_work_budget(self):
        K = np.eye(4)
        system = SpectralLinearSystem(K, .1, K.copy(), np.ones(4))
        budget = ReadoutBudget(max_matrix_entry_work=100)
        with patch.object(readout, 'ReadoutBudget', return_value=budget) as factory, \
                patch.object(system, '_candidate', return_value=(np.zeros(4), 0.)):
            with self.assertRaises(FloatingPointError):
                system.solve_constrained(np.array([1., -1., 1., -1.]))
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(system.info['readout_refinement_attempts'], 5)
        # Initial measurement + both directions + the mandatory fresh check.
        # The unchanged residual then rejects; later representations cannot
        # afford another attempt from the same remaining 36-unit allowance.
        self.assertEqual(budget.matrix_entry_work, 64)
        self.assertEqual(system.info['readout_action_matrix_entry_work'], 64)

    def test_invalid_state_does_not_attempt_polishing(self):
        K, shift, rhs, x, scalar, target, system = self.small_case()
        for value in (np.nan, np.inf, -np.inf):
            broken = x.copy()
            broken[0] = value
            with self.subTest(value=value), patch.object(readout, 'compensated_residual', side_effect=AssertionError('Invalid state measured')):
                result, details = polish_readout(K, shift, rhs, broken, scalar, target,
                                                 system._compensated_measure, ReadoutBudget())
            self.assertIsNone(result)
            self.assertEqual(details['status'], 'invalid_state')
        with patch.object(readout, 'compensated_residual', side_effect=AssertionError('Complex state measured')):
            result, _ = polish_readout(K.astype(complex), shift, rhs, x, scalar, target,
                                      system._compensated_measure, ReadoutBudget())
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
