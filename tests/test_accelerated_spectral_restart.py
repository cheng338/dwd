"""Accelerated typed failure recovery and observer lifecycle controls."""
import unittest
from unittest.mock import patch, Mock
import numpy as np
from dwd._kernel_solver import solve_kernel
from dwd._kernel_linear_system import KernelLinearSystem
from dwd._kernel_mm_recovery import KernelMMRecovery
from dwd._kernel_recovery import solve_with_spectral_restart, ConstrainedSolveFailure, MMRecoveryExhausted
from dwd.gen_kern_dwd import KernGDWD
observations = []

class Check(unittest.TestCase):

    def setUp(self):
        self.K = np.eye(8) / 2 + np.ones((8, 8)) / 8
        self.y = np.tile([-1.0, 1.0], 4)
        self.alpha = np.arange(8) * 1e-05
        self.kw = dict(backend='auto', implementation='optimized', psd_known=True, alpha_init=self.alpha, offset_init=0.001, max_iter=8, stopping='fixed', acceleration='restart', residual_check_order='auto')

    def run_injected(self, kw, fail_at, error=ConstrainedSolveFailure):
        original = KernelLinearSystem.solve_constrained
        calls = [0]
        attempts = []

        def fail(system, rhs):
            calls[0] += 1
            if calls[0] >= fail_at:
                raise error('controlled exhausted constrained solve')
            return original(system, rhs)

        def watched(*args, **options):
            attempts.append(dict(options))
            return solve_kernel(*args, **options)
        with patch.object(KernelLinearSystem, 'solve_constrained', fail), patch.object(KernelMMRecovery, 'step', side_effect=FloatingPointError('controlled exact capacity refusal')):
            result = solve_with_spectral_restart(watched, self.K, self.y, 0.13, eligible=True, **kw)
        return (result, attempts, calls[0])

    def test_recovery_matrix(self):
        for q in (0.5, 1.0, 2.0, 8.0):
            for fail_at in (1, 4):
                kw = dict(self.kw, q=q)
                with self.subTest(q=q, fail_at=fail_at):
                    got, attempts, calls = self.run_injected(kw, fail_at)
                    expected = solve_kernel(self.K, self.y, 0.13, **dict(kw, backend='spectral', residual_check_order='refinement_first'))
                    for key in ('alpha', 'objective_history'):
                        np.testing.assert_array_equal(got[key], expected[key])
                    self.assertEqual(got['offset'], expected['offset'])
                    np.testing.assert_array_equal(self.alpha, np.arange(8) * 1e-05)
                    self.assertEqual(len(attempts), 2)
                    self.assertIs(attempts[0]['alpha_init'], attempts[1]['alpha_init'])
                    detail = got['diagnostics']['spectral_restart']
                    self.assertEqual(detail['discarded_completed_updates'], fail_at - 1)
                    self.assertEqual(detail['max_iter_per_attempt'], 8)
                    self.assertIn('acceleration_counts', detail['failed_attempt'])
                    observations.append({'kind': 'typed_recovery', 'q': q, 'failure_at': fail_at, 'linear_attempts': calls, 'receipt': detail})

    def test_arbitrary_failure_not_retried(self):
        with self.assertRaises(FloatingPointError) as caught:
            self.run_injected(self.kw, 4, FloatingPointError)
        self.assertNotIsInstance(caught.exception, MMRecoveryExhausted)

    def test_callbacks_are_never_replayed(self):
        seen = []

        def callback(state):
            seen.append(state['iteration'])
        with self.assertRaises(FloatingPointError):
            self.run_injected(dict(self.kw, callback=callback), 4)
        self.assertEqual(seen, [0, 1, 2, 3])

    def test_validation_restored_from_only_successful_attempt(self):
        kw = dict(self.kw, validation=(np.zeros_like(self.K), self.y), stopping='validation', patience=7)
        got, attempts, _ = self.run_injected(kw, 4)
        expected = solve_kernel(self.K, self.y, 0.13, **dict(kw, backend='spectral', residual_check_order='refinement_first'))
        for key in ('alpha', 'objective_history', 'validation_history'):
            np.testing.assert_equal(got[key], expected[key])
        self.assertEqual(got['returned_iteration'], expected['returned_iteration'])
        self.assertEqual(got['converged'], expected['converged'])

    def test_retry_is_bounded_and_no_fake_result(self):
        calls = []

        def fails(*a, **kw):
            calls.append(kw['backend'])
            raise MMRecoveryExhausted('injected', dict(completed_iterations=0, failed_iteration=1, setup_seconds=0.0, optimization_seconds=0.0))
        with self.assertRaises(MMRecoveryExhausted):
            solve_with_spectral_restart(fails, self.K, self.y, 0.13, eligible=True, **self.kw)
        self.assertEqual(calls, ['auto', 'spectral'])

    def test_later_active_certified_recovery_is_typed(self):
        actions = []
        attempts = []

        def action(recovery, rhs, old_offset, trigger=None):
            actions.append(trigger)
            if len(actions) >= 4:
                raise FloatingPointError('controlled later certified action failure')
            recovery.action = object()
            recovery.info.update(attempted=True, certificate_status='controlled_exact_action')
            recovery.info['accepted_actions'] += 1
            n = len(rhs)
            border = np.block([[recovery.K + recovery.delta * np.eye(n), np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
            value = np.linalg.solve(border, np.r_[rhs, 0.0])
            return (value[:-1], old_offset + value[-1], recovery.K @ value[:-1])

        def watched(*args, **options):
            attempts.append(options)
            return solve_kernel(*args, **options)
        with patch.object(KernelLinearSystem, 'solve_constrained', side_effect=ConstrainedSolveFailure('controlled initial constrained failure')), patch.object(KernelMMRecovery, 'step', action):
            got = solve_with_spectral_restart(watched, self.K, self.y, 0.13, eligible=True, **self.kw)
        expected = solve_kernel(self.K, self.y, 0.13, **dict(self.kw, backend='spectral', residual_check_order='refinement_first'))
        np.testing.assert_array_equal(got['alpha'], expected['alpha'])
        self.assertEqual(len(attempts), 2)
        detail = got['diagnostics']['spectral_restart']
        self.assertEqual(detail['discarded_completed_updates'], 3)
        self.assertEqual(detail['failed_attempt']['mm_function_recovery_accepted_actions'], 3)
        self.assertGreaterEqual(detail['failed_attempt']['acceleration_counts']['acceleration_numerical_restarts'], 1)
        observations.append({'kind': 'later_active_recovery', 'receipt': detail})

    def test_public_trusted_rbf_dispatch(self):
        X = np.arange(24.0).reshape(8, 3) / 24
        original = KernelLinearSystem.solve_constrained

        def fail(system, rhs):
            raise ConstrainedSolveFailure('controlled trusted-RBF failure')
        model = KernGDWD(kernel='rbf', kernel_kws={'gamma': 0.4}, lambd=0.13, acceleration='restart', stopping='fixed', max_iter=3)
        with patch.object(KernelLinearSystem, 'solve_constrained', fail), patch.object(KernelMMRecovery, 'step', side_effect=FloatingPointError('controlled recovery exhaustion')):
            model.fit(X, self.y)
        self.assertTrue(model.diagnostics_['auto_fallback'])
        self.assertEqual(model.diagnostics_['attempted_backends'], ['cholesky', 'spectral'])
        self.assertTrue(np.isfinite(model.decision_function(X)).all())
