"""Original-equation oracles for the lazy, full spectral inverse recovery."""
import math
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import solve
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import dwd._spectral_linear_system as spectral
from dwd._eigen import validated_eigh
from dwd.gen_kern_dwd import KernGDWD
from test_aposteriori_rkhs_bound import structured_solution, actual_error
from test_toy_residual_regression import DecimalOracle, q1_path


def system_for(K, shift):
    values, vectors = validated_eigh(K)
    # The public preparation makes this same PSD-roundoff adjustment before
    # constructing the private system. Equilibrated shifted spectra stay
    # strictly positive; fresh original-K inverses share this roundoff policy.
    return spectral.SpectralLinearSystem(K, shift, vectors, np.maximum(values, 0.))


class EquilibratedSpectralRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(2)

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def check_original_equations(self, K, shift, rhs, target, x, s):
        oracle = DecimalOracle(K, shift)
        residual, constraint = oracle.residual(rhs, x, s, target)
        self.assertLessEqual(max(abs(residual)), 1e-10 * max(1., max(abs(rhs))))
        self.assertLessEqual(abs(constraint), 64 * np.finfo(float).eps *
                             max(1., math.fsum(abs(x)), abs(target)))
        return oracle

    def test_nine_exact_psd_systems_match_actual_decimal100_rkhs_error(self):
        for n in (30, 75, 120):
            rng = np.random.RandomState(91300 + n)
            rng.randint(-3, 4, size=(n, 5))
            rng.normal(size=(n, 4))
            rhs = rng.normal(size=n) + .23
            K = np.diag(np.exp2(np.linspace(-30, 30, n))) + 1024 * np.ones((n, n))
            for index, shift in enumerate((1e-14, 1e-10, 1e-6)):
                target = 0. if index % 2 == 0 else .37
                with self.subTest(n=n, shift=shift):
                    system = system_for(K, shift)
                    with patch.object(spectral, 'validated_eigh', wraps=validated_eigh) as prepare:
                        x, s = system.solve_constrained(rhs, target)
                    # A different healthy provider can solve this natural
                    # input ordinarily. Accuracy is mandatory; recovery is
                    # required only when ordinary arithmetic actually fails.
                    self.assertLessEqual(prepare.call_count, 4)
                    self.assertLessEqual(system.info['linear_recoveries'], prepare.call_count)
                    self.assertEqual(system.info['linear_solves'], 1)
                    self.assertEqual(system.info['positive_eigenvalues_discarded'], 0)
                    self.assertEqual(system.info['added_objective_regularization'], 0.)
                    if prepare.call_count:
                        self.assertTrue(system.info['spectral_recovery_attempts'][-1]['accepted'])
                    oracle = self.check_original_equations(K, shift, rhs, target, x, s)
                    ideal, _ = structured_solution(K, shift, rhs, target)
                    tolerance = 5e-7 * max(1., math.sqrt(abs(float(x @ oracle.product(x)))))
                    self.assertLessEqual(actual_error(K, x, ideal), tolerance)
                    # A second request retains this fixed-shift representation.
                    with patch.object(spectral, 'validated_eigh', side_effect=AssertionError('repeat EVD')):
                        again_x, again_s = system.solve_constrained(rhs, target)
                    assert_array_equal(again_x, x)
                    self.assertEqual(again_s, s)

    def test_injected_ordinary_failure_requires_exactly_one_successful_recovery(self):
        K = np.diag([1., 2., 3., 4.]) + .5 * np.ones((4, 4))
        shift, target = .2, .37
        rhs = np.array([.7, -1., .3, 2.])
        system = system_for(K, shift)
        ordinary = system._solve_constrained

        def fail_only_original(r, c):
            if not system._recovery_attempted:
                raise FloatingPointError('Injected ordinary-only refinement failure')
            return ordinary(r, c)

        with patch.object(system, '_solve_constrained', side_effect=fail_only_original), \
                patch.object(spectral, 'validated_eigh', wraps=validated_eigh) as prepare:
            x, s = system.solve_constrained(rhs, target)
            again_x, again_s = system.solve_constrained(rhs, target)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(system.info['linear_recoveries'], 1)
        self.assertEqual(system.info['linear_solves'], 2)
        self.assertTrue(system.info['spectral_recovery_attempt']['success'])
        assert_array_equal(again_x, x)
        self.assertEqual(again_s, s)
        self.check_original_equations(K, shift, rhs, target, x, s)
        dense = np.block([[K + shift * np.eye(4), np.ones((4, 1))],
                          [np.ones((1, 4)), np.zeros((1, 1))]])
        assert_allclose(np.r_[x, s], solve(dense, np.r_[rhs, target]), rtol=1e-12, atol=1e-12)

    def test_benign_ordinary_path_is_bitwise_unchanged_and_has_no_recovery(self):
        rng = np.random.default_rng(915)
        features = rng.normal(size=(12, 5))
        K, shift, target = features @ features.T, .2, .37
        rhs = rng.normal(size=12)
        system = system_for(K, shift)
        ordinary = system_for(K, shift)
        expected_x, expected_s = ordinary._solve_constrained(rhs, target)
        with patch.object(spectral, 'validated_eigh', side_effect=AssertionError('unnecessary recovery')):
            x, s = system.solve_constrained(rhs, target)
        assert_array_equal(x, expected_x)
        self.assertEqual(s, expected_s)
        dense = np.block([[K + shift * np.eye(12), np.ones((12, 1))],
                          [np.ones((1, 12)), np.zeros((1, 1))]])
        expected = solve(dense, np.r_[rhs, target])
        assert_allclose(np.r_[x, s], expected, rtol=1e-11, atol=1e-11)
        self.assertEqual(system.info['linear_recoveries'], 0)
        self.assertNotIn('spectral_recovery_attempt', system.info)

    def test_original_coefficient_candidate_is_restored_after_failed_refinement(self):
        K = np.diag([1., 3., 7., 11.])
        system = system_for(K, .1)
        rhs = np.array([.7, -1., .3, 2.])
        x, s = np.array([2., -3., 4., -1.]), 7.
        scores = K @ x
        original_x, original_scores = x.copy(), scores.copy()
        original_inner, original_action = system._refine_candidate, system._inverse_action
        starts = []

        def logged_inner(r, a, b, product):
            starts.append((a.copy(), b, product.copy()))
            return original_inner(r, a, b, product)

        def faulty_original_action(r):
            if system._equilibration is None:
                raise FloatingPointError('Manufactured ordinary inverse failure')
            return original_action(r)

        with patch.object(system, '_refine_candidate', side_effect=logged_inner), \
                patch.object(system, '_inverse_action', side_effect=faulty_original_action):
            result_x, result_s, result_scores = system.refine_candidate(rhs, x, s, scores)
        self.assertEqual(len(starts), 2)
        for a, b, product in starts:
            assert_array_equal(a, original_x)
            self.assertEqual(b, s)
            assert_array_equal(product, original_scores)
        assert_array_equal(x, original_x)
        assert_array_equal(scores, original_scores)
        self.check_original_equations(K, .1, rhs, 0., result_x, result_s)
        assert_array_equal(result_scores, K @ result_x)

    def test_failed_preparation_is_recorded_and_never_repeated(self):
        system = system_for(np.eye(4), .1)
        with patch.object(system, '_solve_constrained', side_effect=FloatingPointError('bad original action')), \
                patch.object(spectral, 'validated_eigh', side_effect=FloatingPointError('bad transformed basis')) as prepare:
            for _ in range(2):
                with self.assertRaises(FloatingPointError):
                    system.solve_constrained(np.ones(4))
        self.assertEqual(prepare.call_count, 4)
        self.assertFalse(system.info['spectral_recovery_attempt']['success'])
        self.assertIn('bad transformed basis', system.info['spectral_recovery_attempt']['error'])
        self.assertEqual(system.info['linear_recoveries'], 0)
        self.assertEqual([r['kind'] for r in system.info['spectral_recovery_attempts']],
                         ['equilibrated', 'evd', 'evr', 'evx'])

    def test_computed_corrupt_transformed_basis_is_rejected_by_real_validator(self):
        system = system_for(np.eye(4), .1)
        with patch.object(system, '_solve_constrained', side_effect=FloatingPointError('trigger')), \
                patch('dwd._eigen.eigh', return_value=(np.ones(4), np.zeros((4, 4)))):
            with self.assertRaisesRegex(FloatingPointError, 'failed accuracy after bounded native'):
                system.solve_constrained(np.arange(4.))
        self.assertIsNone(system._equilibration)
        self.assertFalse(system.info['spectral_recovery_attempt']['success'])

    def test_nonpositive_or_unrepresentable_inverse_is_never_clipped(self):
        for values in (np.array([0., 1., 1., 1.]), np.full(4, 1e-320)):
            with self.subTest(values=values):
                system = system_for(np.eye(4), .1)
                with patch.object(system, '_solve_constrained', side_effect=FloatingPointError('trigger')), \
                        patch.object(spectral, 'validated_eigh', return_value=((values, np.eye(4)), {})):
                    with self.assertRaises(FloatingPointError):
                        system.solve_constrained(np.arange(4.))
                self.assertIsNone(system._equilibration)
                self.assertFalse(system.info['spectral_recovery_attempt']['success'])

    def test_overflow_and_lost_nonzero_scaling_fail_without_mutating_kernel(self):
        cases = [(np.eye(2) * 1e308, 1e308),
                 (np.array([[1e308, 1e-320], [1e-320, 1e308]]), 1.)]
        for K, shift in cases:
            with self.subTest(shift=shift):
                system = system_for(np.eye(2), .1)
                system.K, system.shift = K, shift
                original = K.copy()
                with self.assertRaises(FloatingPointError):
                    system._prepare_equilibrated('manufactured unrepresentable scaling')
                assert_array_equal(K, original)
                self.assertIsNone(system._equilibration)

    def test_invalid_rhs_does_not_start_recovery(self):
        system = system_for(np.eye(4), .1)
        with patch.object(spectral, 'validated_eigh', side_effect=AssertionError('invalid-input recovery')):
            for rhs, target in ((np.ones(3), 0.), (np.array([1., np.nan, 0., 0.]), 0.), (np.ones(4), np.inf)):
                with self.assertRaises(FloatingPointError):
                    system.solve_constrained(rhs, target)
        self.assertFalse(system._recovery_attempted)

    def test_public_bad_supplied_basis_remains_an_input_error(self):
        K, y = np.eye(6), np.array([-1., 1., -1., 1., -1., 1.])
        from dwd._kernel_solver import solve_kernel
        with patch.object(spectral, 'validated_eigh', side_effect=AssertionError('input error recovery')):
            with self.assertRaises(ValueError):
                solve_kernel(K, y, .1, implementation='reference',
                             K_eig=(np.zeros_like(K), np.ones(6)), max_iter=1)

    def test_known_public_150_row_reference_trajectory_remains_ordinary(self):
        X, labels = make_classification(n_samples=300, n_features=7, n_informative=4,
                                       class_sep=2., flip_y=0., random_state=712)
        labels = np.where(labels, 1., -1.)
        rows, _ = list(StratifiedKFold(2, shuffle=True, random_state=2718).split(X, labels))[0]
        X, y = StandardScaler().fit_transform(X[rows]), labels[rows]
        from sklearn.metrics.pairwise import rbf_kernel
        K = rbf_kernel(X, gamma=.001)
        oracle = DecimalOracle(K, len(X) * 1e-7 / 2)
        expected = q1_path(oracle, y, 1e-7, np.zeros(len(X)), 0., 5)
        observed = []
        model = KernGDWD(implementation='reference', lambd=1e-7, kernel='rbf',
                        kernel_kws={'gamma': .001}, initialization='zero',
                        stopping='fixed', max_iter=5,
                        callback=lambda state: observed.append((state['decision_values'].copy(), state['objective'])))
        with patch.object(spectral.SpectralLinearSystem, '_prepare_equilibrated',
                          side_effect=AssertionError('benign public path changed')):
            model.fit(X, y)
        self.assertEqual(len(observed), 6)
        for (actual_score, actual_objective), (score, objective) in zip(observed, expected):
            assert_allclose(actual_score, score, rtol=0., atol=2e-8)
            self.assertAlmostEqual(actual_objective, objective, delta=3e-10)


if __name__ == '__main__':
    unittest.main()
