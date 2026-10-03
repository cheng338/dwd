"""Opt-in scheduling preserves accuracy gates and the default recovery path."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_array_equal
from scipy.linalg import eigh
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from dwd._kernel_linear_system import KernelLinearSystem
from dwd._kernel_solver import solve_kernel
from dwd._kernel_recovery import MMRecoveryExhausted, solve_with_spectral_restart
from dwd._spectral_linear_system import SpectralLinearSystem
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV


class AdaptiveResidualOrderTests(unittest.TestCase):
    def setUp(self):
        self.K = np.diag([1., 2., 3., 4.])
        self.rhs = np.array([1., -1., 2., -2.])
        self.system = KernelLinearSystem(self.K, .17, residual_check_order='adaptive')
        self.x, self.s = self.system._candidate(self.rhs, 0.)
        self.good = (True, np.zeros(4), 0., self.K @ self.x, 0., 0.)
        self.bad = (False, np.ones(4), 0., self.K @ self.x, 1., 1.)

    def train(self, system=None):
        system = self.system if system is None else system
        # Two genuinely attempted/discarded trials, each followed by a passing
        # accurate check on the untouched original state, establish history.
        with patch.object(system, '_ordinary_measure', return_value=self.bad), \
                patch.object(system, '_candidate', return_value=(np.zeros(4), 0.)), \
                patch.object(system, '_compensated_measure', return_value=self.good):
            for _ in range(2):
                x, s, checked = system._measure_with_native_trial(self.rhs, 0., self.x, self.s)
                self.assertIs(x, self.x)
                self.assertEqual(s, self.s)
                self.assertTrue(checked[0])

    def test_default_order_never_probes_even_after_repeated_discarded_trials(self):
        system = KernelLinearSystem(self.K, .17)
        self.train(system)
        calls = []
        def ordinary(*args):
            calls.append('ordinary')
            return self.bad
        def trial(*args):
            calls.append('trial')
            return np.zeros(4), 0.
        def accurate(*args, **kwargs):
            calls.append('accurate')
            return self.good
        with patch.object(system, '_ordinary_measure', side_effect=ordinary), \
                patch.object(system, '_candidate', side_effect=trial), \
                patch.object(system, '_compensated_measure', side_effect=accurate):
            system._measure_with_native_trial(self.rhs, 0., self.x, self.s)
        self.assertEqual(calls, ['ordinary', 'trial', 'ordinary', 'accurate'])
        self.assertNotIn('adaptive_residual_probes', system.info)

    def test_two_observations_then_accurate_probe_can_skip_trial(self):
        self.train()
        self.assertEqual(self.system.info['native_refinement_trials'], 2)
        self.assertNotIn('adaptive_residual_probes', self.system.info)
        with patch.object(self.system, '_ordinary_measure', return_value=self.bad), \
                patch.object(self.system, '_candidate', side_effect=AssertionError('Skipped trial')), \
                patch.object(self.system, '_compensated_measure', return_value=self.good) as measured:
            got_x, got_s, got = self.system._measure_with_native_trial(self.rhs, 0., self.x, self.s)
        self.assertIs(got_x, self.x)
        self.assertEqual(got_s, self.s)
        self.assertIs(got, self.good)
        self.assertEqual(measured.call_count, 1)
        self.assertEqual(self.system.info['adaptive_native_trials_skipped'], 1)
        self.assertEqual(self.system.info['native_refinement_trials'], 2)
        self.assertEqual(self.system.info['refinement_steps'], 0)

    def test_probe_miss_still_accepts_successful_native_trial(self):
        self.train()
        correction = np.array([.01, -.01, .02, -.02])
        with patch.object(self.system, '_ordinary_measure', side_effect=[self.bad, self.good]), \
                patch.object(self.system, '_candidate', return_value=(correction, .01)), \
                patch.object(self.system, '_compensated_measure', return_value=self.bad):
            got_x, got_s, got = self.system._measure_with_native_trial(self.rhs, 0., self.x, self.s)
        assert_array_equal(got_x, self.x + correction)
        self.assertEqual(got_s, self.s + .01)
        self.assertIs(got, self.good)
        self.assertEqual(self.system.info['native_refinement_acceptances'], 1)
        self.assertEqual(self.system._adaptive_unproductive_trials, 0)

    def test_probe_exception_does_not_consume_recovery_or_correction_budget(self):
        self.train()
        # The solve's first candidate and its native trial both remain available.
        with patch.object(self.system, '_ordinary_measure', side_effect=[self.bad, self.good]), \
                patch.object(self.system, '_candidate', return_value=(self.x.copy(), self.s)), \
                patch.object(self.system, '_compensated_measure', side_effect=RuntimeError('Probe failed')):
            self.system.solve_constrained(self.rhs)
        self.assertEqual(self.system.info['linear_recoveries'], 0)
        self.assertEqual(self.system.info['refinement_steps'], 1)
        self.assertEqual(self.system.info['adaptive_residual_probe_errors'], 1)
        self.assertEqual(self.system.info['adaptive_residual_probe_fallbacks'], 1)

    def test_probe_exception_then_discarded_trial_rechecks_original(self):
        self.train()
        with patch.object(self.system, '_ordinary_measure', return_value=self.bad), \
                patch.object(self.system, '_candidate', return_value=(np.zeros(4), 0.)), \
                patch.object(self.system, '_compensated_measure', side_effect=[FloatingPointError('Probe'), self.good]) as measured:
            _, _, got = self.system._measure_with_native_trial(self.rhs, 0., self.x, self.s)
        self.assertIs(got, self.good)
        self.assertEqual(measured.call_count, 2)
        self.assertNotIn('adaptive_residual_probe_reuses', self.system.info)

    def test_process_interrupts_are_not_swallowed_by_optional_probe(self):
        self.train()
        for exception in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception.__name__), \
                    patch.object(self.system, '_ordinary_measure', return_value=self.bad), \
                    patch.object(self.system, '_candidate', side_effect=AssertionError('Interrupt must propagate')), \
                    patch.object(self.system, '_compensated_measure', side_effect=exception):
                with self.assertRaises(exception):
                    self.system._measure_with_native_trial(self.rhs, 0., self.x, self.s)
        self.assertNotIn('adaptive_residual_probe_errors', self.system.info)

    def test_real_accuracy_probe_rejects_bad_state_before_native_correction(self):
        self.train()
        bad_x = self.x + np.array([.01, -.01, .02, -.02])
        bad_s = self.s + .03
        # History is synthetic; all assessments and correction arithmetic here
        # are real. An inaccurate state must still be corrected, not accepted
        # merely because accurate-first ordering is active.
        got_x, got_s, checked = self.system._measure_with_native_trial(self.rhs, 0., bad_x, bad_s)
        self.assertTrue(checked[0])
        self.assertEqual(self.system.info['adaptive_residual_probe_fallbacks'], 1)
        self.assertNotIn('adaptive_residual_probe_acceptances', self.system.info)
        self.assertEqual(self.system.info['native_refinement_acceptances'], 1)
        residual = self.rhs - self.K @ got_x - self.system.shift * got_x - got_s
        self.assertLess(np.max(np.abs(residual)), 1e-12)
        self.assertLess(abs(float(np.sum(got_x))), 1e-13)

    def test_unsuccessful_probe_reused_only_for_unchanged_original_after_trial_discard(self):
        self.train()
        with patch.object(self.system, '_ordinary_measure', return_value=self.bad), \
                patch.object(self.system, '_candidate', return_value=(np.ones(4), 1.)), \
                patch.object(self.system, '_compensated_measure', return_value=self.bad) as measured:
            got_x, got_s, got = self.system._measure_with_native_trial(self.rhs, 0., self.x, self.s)
        self.assertIs(got_x, self.x)
        self.assertEqual(got_s, self.s)
        self.assertIs(got, self.bad)
        self.assertEqual(measured.call_count, 1)
        self.assertEqual(self.system.info['adaptive_residual_probe_reuses'], 1)
        self.assertEqual(self.system._adaptive_unproductive_trials, 0)

    def test_changed_factor_during_trial_forbids_probe_reuse(self):
        self.train()
        def candidate(*args):
            self.system.factor = (self.system.factor[0].copy(), self.system.factor[1])
            return np.zeros(4), 0.
        with patch.object(self.system, '_ordinary_measure', return_value=self.bad), \
                patch.object(self.system, '_candidate', side_effect=candidate), \
                patch.object(self.system, '_compensated_measure', side_effect=[self.bad, self.good]) as measured:
            _, _, got = self.system._measure_with_native_trial(self.rhs, 0., self.x, self.s)
        self.assertIs(got, self.good)
        self.assertEqual(measured.call_count, 2)
        self.assertNotIn('adaptive_residual_probe_reuses', self.system.info)

    def test_probe_identity_includes_coefficients_intercept_rhs_target_and_factor(self):
        identity = self.system._adaptive_measurement_identity(self.rhs, 0., self.x, self.s)
        for rhs, target, x, s in ((self.rhs + 1., 0., self.x, self.s),
                                  (self.rhs, 1., self.x, self.s),
                                  (self.rhs, 0., self.x + 1., self.s),
                                  (self.rhs, 0., self.x, self.s + 1.)):
            self.assertNotEqual(identity, self.system._adaptive_measurement_identity(rhs, target, x, s))
        self.system._prepare('centered')
        self.assertNotEqual(identity, self.system._adaptive_measurement_identity(self.rhs, 0., self.x, self.s))

    def test_preparing_new_factor_resets_history_and_requires_new_observations(self):
        self.train()
        self.system._prepare('centered')
        self.assertEqual(self.system._adaptive_unproductive_trials, 0)
        self.assertEqual(self.system.info['adaptive_residual_history_resets'], 1)
        with patch.object(self.system, '_ordinary_measure', side_effect=[self.bad, self.good]), \
                patch.object(self.system, '_candidate', return_value=(np.zeros(4), 0.)), \
                patch.object(self.system, '_compensated_measure', side_effect=AssertionError('No trained probe')):
            self.system._measure_with_native_trial(self.rhs, 0., self.x, self.s)

    def test_replaced_factor_without_prepare_resets_history(self):
        self.train()
        self.system.factor = (self.system.factor[0].copy(), self.system.factor[1])
        with patch.object(self.system, '_ordinary_measure', return_value=self.good), \
                patch.object(self.system, '_compensated_measure', side_effect=AssertionError('No probe')):
            self.system._measure_with_native_trial(self.rhs, 0., self.x, self.s)
        self.assertEqual(self.system._adaptive_unproductive_trials, 0)
        self.assertEqual(self.system.info['adaptive_residual_history_resets'], 1)

    def test_ordinary_success_avoids_all_adaptive_probe_work(self):
        self.train()
        with patch.object(self.system, '_ordinary_measure', return_value=self.good), \
                patch.object(self.system, '_compensated_measure', side_effect=AssertionError('No probe')), \
                patch.object(self.system, '_candidate', side_effect=AssertionError('No trial')):
            self.system._measure_with_native_trial(self.rhs, 0., self.x, self.s)
        self.assertNotIn('adaptive_residual_probes', self.system.info)

    def test_anchor_recovery_retains_original_direct_checks(self):
        self.train()
        self.system._prepare('anchor')
        with patch.object(self.system, '_measure_with_native_trial', side_effect=AssertionError('No anchor trial')), \
                patch.object(self.system, '_measure', return_value=self.good) as measured:
            self.system.solve_constrained(self.rhs)
        self.assertEqual(measured.call_count, 1)
        self.assertEqual(self.system.info['anchor_accepted_actions'], 1)
        self.assertNotIn('adaptive_residual_probes', self.system.info)

    def test_spectral_inherited_method_retains_default_order(self):
        values, vectors = eigh(self.K)
        system = SpectralLinearSystem(self.K, .17, vectors, values)
        self.train(system)
        self.assertEqual(system.info['native_refinement_trials'], 2)
        self.assertNotIn('adaptive_residual_probes', system.info)


class AdaptiveResidualApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limit = threadpool_limits(1)
        rng = np.random.RandomState(51)
        cls.X = rng.normal(size=(16, 3))
        cls.y = np.tile([-1, 1], 8)

    @classmethod
    def tearDownClass(cls):
        cls.limit.restore_original_limits()

    def test_default_and_explicit_original_order_are_bitwise_identical(self):
        options = dict(lambd=.1, kernel='rbf', kernel_kws={'gamma': .3},
                       stopping='fixed', max_iter=3, backend='cholesky')
        first = KernGDWD(**options).fit(self.X, self.y)
        second = KernGDWD(**options, residual_check_order='refinement_first').fit(self.X, self.y)
        assert_array_equal(first.dual_coef_, second.dual_coef_)
        assert_array_equal(first.intercept_, second.intercept_)
        assert_array_equal(first.objective_history_, second.objective_history_)
        assert_array_equal(first.decision_function(self.X), second.decision_function(self.X))

    def test_clone_parameter_and_real_small_adaptive_fit(self):
        estimator = KernGDWD(lambd=.1, max_iter=3, stopping='fixed', residual_check_order='adaptive')
        copied = clone(estimator)
        self.assertEqual(copied.get_params()['residual_check_order'], 'adaptive')
        copied.fit(self.X, self.y)
        self.assertEqual(copied.diagnostics_['residual_check_order'], 'adaptive')
        self.assertTrue(np.isfinite(copied.decision_function(self.X)).all())

    def test_cv_forwards_option_to_candidates_and_final_refit(self):
        estimator = KernGDWDCV(lambd_vals=[.1, .2], q_vals=[1.], cv=2,
                              max_iter=2, stopping='fixed', residual_check_order='adaptive')
        copied = clone(estimator)
        self.assertEqual(copied.get_params()['residual_check_order'], 'adaptive')
        seen = []
        original = KernGDWD.fit
        def recorded(model, *args, **kwargs):
            seen.append(model.residual_check_order)
            return original(model, *args, **kwargs)
        with patch.object(KernGDWD, 'fit', recorded):
            copied.fit(self.X, self.y)
        self.assertEqual(seen, ['adaptive'] * 5)
        self.assertEqual(copied.best_estimator_.residual_check_order, 'adaptive')

    def test_invalid_api_options_rejected_before_kernel_or_cv_preparation(self):
        invalid = [dict(residual_check_order='wrong'), dict(residual_check_order=None)]
        invalid += [dict(residual_check_order='adaptive', **choice) for choice in (
            dict(implementation='reference'), dict(backend='spectral'), dict(backend='lbfgs'),
            dict(solver_mode='legacy'), dict(acceleration='restart'))]
        for factory in (KernGDWD, KernGDWDCV):
            for options in invalid:
                with self.subTest(factory=factory.__name__, options=options), \
                        patch.object(KernGDWD, '_compute_training_kernel', side_effect=AssertionError('Validation must precede work')):
                    with self.assertRaises(ValueError):
                        factory(**options).fit(self.X, self.y)

    def test_solver_rejects_unsupported_routes_including_auto_with_eigenbasis(self):
        K = self.X @ self.X.T
        for options in (dict(backend='spectral'), dict(backend='lbfgs'),
                        dict(implementation='reference'), dict(acceleration='restart'),
                        dict(K_eig=eigh(K))):
            with self.subTest(options=tuple(options)):
                with self.assertRaisesRegex(ValueError, 'Cholesky'):
                    solve_kernel(K, self.y, .1, residual_check_order='adaptive', **options)
        with self.assertRaisesRegex(ValueError, 'residual_check_order'):
            solve_kernel(K, self.y, .1, residual_check_order='unknown')

    def test_auto_spectral_retry_explicitly_uses_original_order(self):
        failed = MMRecoveryExhausted('test', dict(completed_iterations=0, failed_iteration=1,
                                                setup_seconds=0., optimization_seconds=0.))
        result = dict(n_iter=1, diagnostics={})
        solve = Mock(side_effect=[failed, result])
        got = solve_with_spectral_restart(solve, np.eye(4), self.y[:4], .1,
            eligible=True, backend='auto', residual_check_order='adaptive')
        self.assertIs(got, result)
        self.assertEqual(solve.call_args_list[0].kwargs['residual_check_order'], 'adaptive')
        self.assertEqual(solve.call_args_list[1].kwargs['residual_check_order'], 'refinement_first')
        self.assertEqual(got['diagnostics']['requested_residual_check_order'], 'adaptive')
        self.assertEqual(got['diagnostics']['residual_check_order'], 'refinement_first')


if __name__ == '__main__':
    unittest.main(verbosity=2)
