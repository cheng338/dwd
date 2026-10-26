"""Original-equation oracles for the lazy, full spectral inverse recovery."""
from contextlib import ExitStack
from decimal import Decimal, localcontext
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


def benign_case():
    rng = np.random.default_rng(915)
    features = rng.normal(size=(12, 5))
    K, shift, target = features @ features.T, .2, .37
    return K, shift, target, rng.normal(size=12)


def benign_decimal_residuals(K, shift, rhs, target, x, s):
    """Measure the represented original equations independently of BLAS."""
    def decimal(value):
        return Decimal.from_float(float(value))

    with localcontext() as context:
        context.prec = 100
        dx, ds = [decimal(value) for value in x], decimal(s)
        residuals = [sum((decimal(k) * v for k, v in zip(row, dx)), Decimal(0))
                     + decimal(shift) * dx[i] + ds - decimal(rhs[i])
                     for i, row in enumerate(K)]
        constraint = sum(dx, Decimal(0)) - decimal(target)
        scales = [sum((abs(decimal(k) * v) for k, v in zip(row, dx)), Decimal(0))
                  + abs(decimal(shift) * dx[i]) + abs(ds) + abs(decimal(rhs[i]))
                  for i, row in enumerate(K)]
        # Roundoff-scaled backward-error checks for this fixed 12-row fixture;
        # these are neither interval certificates nor package tolerances.
        allowance = decimal(64 * np.finfo(float).eps) * max(Decimal(1), *scales)
        constraint_allowance = decimal(64 * np.finfo(float).eps) * max(
            Decimal(1), sum(map(abs, dx), Decimal(0)), abs(decimal(target)))
        return residuals, constraint, allowance, constraint_allowance


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

    def check_benign_equations(self, K, shift, rhs, target, x, s):
        self.assertTrue(np.isfinite(x).all() and math.isfinite(s))
        residual, constraint, allowance, constraint_allowance = benign_decimal_residuals(
            K, shift, rhs, target, x, s)
        self.assertLessEqual(max(map(abs, residual)), allowance, 'original-equation guard')
        self.assertLessEqual(abs(constraint), constraint_allowance, 'constraint guard')
        dense = np.block([[K + shift * np.eye(len(K)), np.ones((len(K), 1))],
                          [np.ones((1, len(K))), np.zeros((1, 1))]])
        expected = solve(dense, np.r_[rhs, target])
        assert_allclose(np.r_[x, s], expected, rtol=1e-11, atol=1e-11)

    def check_no_recovery(self, system):
        self.assertEqual(system.info['linear_recoveries'], 0)
        self.assertEqual(system.info['factor_representation'], 'validated_eigenbasis')
        self.assertFalse(system._recovery_attempted)
        self.assertEqual(system._recovery_next, 0)
        self.assertIsNone(system._equilibration)
        self.assertIsNone(system._active_recovery_attempt)
        for key in ('spectral_recovery_attempt', 'spectral_recovery_attempts',
                    'spectral_inverse_failures'):
            self.assertNotIn(key, system.info)
        self.assertEqual(system.info['positive_eigenvalues_discarded'], 0)
        self.assertEqual(system.info['added_objective_regularization'], 0.)

    def check_benign_shared_basis(self, mutate_output=None, force_ordinary_failure=False):
        K, shift, target, rhs = benign_case()
        # Control the basis when testing whether the recovery wrapper changes
        # an ordinary solve. Independent valid native bases need not be bitwise
        # identical, especially in the near-null eigenspace of this Gram matrix.
        values, vectors = validated_eigh(K)
        clipped = np.maximum(values, 0.)
        originals = [array.copy(order='K') for array in (K, rhs, values, vectors, clipped)]
        system = spectral.SpectralLinearSystem(K, shift, vectors, clipped)
        ordinary = spectral.SpectralLinearSystem(K, shift, vectors, clipped)
        inverse_before, v_before = system.inverse.copy(), system.v.copy()
        denominator_before = system.denominator
        expected_x, expected_s = ordinary._solve_constrained(rhs, target)
        public = system.solve_constrained
        with ExitStack() as stack:
            stack.enter_context(patch.object(spectral, 'validated_eigh',
                side_effect=AssertionError('unnecessary recovery')))
            if force_ordinary_failure:
                stack.enter_context(patch.object(system, '_solve_constrained',
                    side_effect=FloatingPointError('deliberate ordinary-path failure')))
            if mutate_output is not None:
                def changed(r, c):
                    x, s = public(r, c)
                    assert_array_equal(x, expected_x, err_msg='control baseline differs before mutation')
                    self.assertEqual(s, expected_s, 'control baseline differs before mutation')
                    return mutate_output(x, s)
                stack.enter_context(patch.object(system, 'solve_constrained', side_effect=changed))
            x, s = system.solve_constrained(rhs, target)
        assert_array_equal(x, expected_x, err_msg='exact coefficient guard')
        self.assertEqual(s, expected_s, 'exact intercept guard')
        self.check_benign_equations(K, shift, rhs, target, x, s)
        self.check_no_recovery(system)
        self.assertEqual(system.info['linear_solves'], 1)
        for actual, before in zip((K, rhs, values, vectors, clipped), originals):
            assert_array_equal(actual, before)
        assert_array_equal(system.inverse, inverse_before)
        assert_array_equal(system.v, v_before)
        self.assertEqual(system.denominator, denominator_before)

    def test_benign_ordinary_path_is_bitwise_unchanged_and_has_no_recovery(self):
        self.check_benign_shared_basis()

    def test_independent_native_bases_satisfy_original_equations(self):
        K, shift, target, rhs = benign_case()
        for index in range(8):
            with self.subTest(native_preparation=index):
                values, vectors = validated_eigh(K)
                system = spectral.SpectralLinearSystem(K, shift, vectors, np.maximum(values, 0.))
                with patch.object(spectral, 'validated_eigh', side_effect=AssertionError('unnecessary recovery')):
                    x, s = system.solve_constrained(rhs, target)
                self.check_benign_equations(K, shift, rhs, target, x, s)
                self.check_no_recovery(system)

    def test_wrapper_returns_the_exact_inner_result_without_postprocessing(self):
        K, shift, target, rhs = benign_case()
        system = system_for(K, shift)
        operation = system._solve_constrained
        captured = []

        def capture(r, c):
            value = operation(r, c)
            captured.append(value)
            return value

        with patch.object(system, '_solve_constrained', side_effect=capture) as called, \
                patch.object(spectral, 'validated_eigh', side_effect=AssertionError('unnecessary recovery')):
            result = system.solve_constrained(rhs, target)
        called.assert_called_once()
        self.assertIs(called.call_args.args[0], rhs)
        self.assertEqual(called.call_args.args[1], target)
        self.assertIs(result, captured[0])
        self.check_no_recovery(system)
        self.check_benign_equations(K, shift, rhs, target, *result)

    def test_one_ulp_coefficient_change_fails_exact_guard(self):
        def corrupt(x, s):
            changed = x.copy()
            changed[0] = np.nextafter(changed[0], np.inf)
            return changed, s

        with self.assertRaisesRegex(AssertionError, 'exact coefficient guard'):
            self.check_benign_shared_basis(mutate_output=corrupt)

    def test_one_ulp_intercept_change_fails_exact_guard(self):
        with self.assertRaisesRegex(AssertionError, 'exact intercept guard'):
            self.check_benign_shared_basis(mutate_output=lambda x, s: (x, np.nextafter(s, np.inf)))

    def test_unnecessary_recovery_fails_benign_guard(self):
        with self.assertRaisesRegex(AssertionError, 'unnecessary recovery'):
            self.check_benign_shared_basis(force_ordinary_failure=True)

    def benign_output(self):
        K, shift, target, rhs = benign_case()
        system = system_for(K, shift)
        x, s = system.solve_constrained(rhs, target)
        self.check_benign_equations(K, shift, rhs, target, x, s)
        return K, shift, target, rhs, x, s

    def test_zero_sum_coefficient_corruption_fails_original_equation_guard(self):
        K, shift, target, rhs, x, s = self.benign_output()
        changed = x.copy()
        changed[0] += .01
        changed[1] -= .01
        with self.assertRaisesRegex(AssertionError, 'original-equation guard'):
            self.check_benign_equations(K, shift, rhs, target, changed, s)

    def test_intercept_corruption_fails_original_equation_guard(self):
        K, shift, target, rhs, x, s = self.benign_output()
        with self.assertRaisesRegex(AssertionError, 'original-equation guard'):
            self.check_benign_equations(K, shift, rhs, target, x, s + .01)

    def test_wrong_requested_sum_fails_constraint_guard(self):
        K, shift, target, rhs, x, s = self.benign_output()
        # Keeping the equation solution unchanged isolates the sum constraint.
        with self.assertRaisesRegex(AssertionError, 'constraint guard'):
            self.check_benign_equations(K, shift, rhs, target + .01, x, s)

    def test_altered_kernel_fails_original_equation_guard(self):
        K, shift, target, rhs, x, s = self.benign_output()
        changed = K.copy()
        changed[0, 0] += .01
        with self.assertRaisesRegex(AssertionError, 'original-equation guard'):
            self.check_benign_equations(changed, shift, rhs, target, x, s)

    def test_benign_invalid_supplied_basis_is_not_admitted(self):
        K, _, _, _ = benign_case()
        values, _ = validated_eigh(K)
        with self.assertRaisesRegex(ValueError, 'Supplied eigenpairs failed validation'):
            validated_eigh(K, supplied=(np.zeros_like(K), values))

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
