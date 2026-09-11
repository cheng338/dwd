"""Tiny injected-failure controls for the real solver's typed retry boundary.

These are control-flow fixtures, not evidence of natural numerical instability.
The existing 14-case replay covers initial exhaustion on the saved real data.
"""
import unittest
from unittest.mock import Mock, patch

import numpy as np
from dwd._kernel_solver import solve_kernel
from dwd._kernel_linear_system import KernelLinearSystem
from dwd._kernel_mm_recovery import KernelMMRecovery
from dwd._kernel_recovery import (ConstrainedSolveFailure, MMRecoveryExhausted,
                                  solve_with_spectral_restart)


class TestLaterRecoveryBoundary(unittest.TestCase):
    def setUp(self):
        # Small dyadic positive-definite kernel, without cancellation stress.
        self.K = np.eye(6) / 2. + np.ones((6, 6)) / 8.
        self.y = np.array([-1., 1., -1., 1., -1., 1.])
        self.options = dict(backend='auto', implementation='optimized',
                            psd_known=True, max_iter=3, stopping='fixed',
                            alpha_init=np.zeros(6), offset_init=0.)

    def run_guarded(self, watched):
        # Public provenance and eligibility have separate regression tests.
        return solve_with_spectral_restart(watched, self.K, self.y, .13,
                                           eligible=True, **self.options)

    def accepted_action(self, recovery, rhs, old_offset, trigger):
        recovery.action = object()  # Next iteration uses active recovery.
        recovery.info.update(attempted=True, certificate_status='injected_exact_action')
        recovery.info['accepted_actions'] += 1
        n = len(rhs)
        bordered = np.block([[recovery.K + recovery.delta * np.eye(n),
                              np.ones((n, 1))],
                             [np.ones((1, n)), np.zeros((1, 1))]])
        value = np.linalg.solve(bordered, np.r_[rhs, 0.])
        return value[:-1], old_offset + value[-1], recovery.K @ value[:-1]

    def test_later_active_recovery_exhaustion_restarts_and_counts_completed_update(self):
        actions = []

        def action(recovery, rhs, old_offset, trigger=None):
            actions.append((rhs.copy(), old_offset, trigger))
            if len(actions) == 2:
                raise FloatingPointError('injected later active action failure')
            return self.accepted_action(recovery, rhs, old_offset, trigger)

        watched = Mock(wraps=solve_kernel)
        with patch.object(KernelLinearSystem, 'solve_constrained',
                          side_effect=ConstrainedSolveFailure('injected constrained exhaustion')) as constrained, \
             patch.object(KernelMMRecovery, 'step', action):
            result = self.run_guarded(watched)

        self.assertEqual(constrained.call_count, 1)
        self.assertEqual(len(actions), 2)
        self.assertIsNone(actions[1][2])
        self.assertEqual([c.kwargs['backend'] for c in watched.call_args_list],
                         ['auto', 'spectral'])
        retry = result['diagnostics']['spectral_restart']
        self.assertEqual(retry['discarded_completed_updates'], 1)
        self.assertEqual(retry['failed_attempted_iteration'], 2)
        self.assertEqual(retry['failed_attempt']['mm_function_recovery_accepted_actions'], 1)
        self.assertEqual(retry['total_completed_updates'], 4)
        self.assertEqual(retry['total_attempted_updates'], 5)
        self.assertEqual(result['n_iter'], 3)
        self.assertEqual(len(result['objective_history']), 4)
        self.assertEqual(result['termination_reason'], 'max_iter')
        np.testing.assert_array_equal(watched.call_args_list[0].kwargs['alpha_init'], np.zeros(6))

    def test_later_active_descent_rejection_is_not_typed_or_retried(self):
        actions = []

        def action(recovery, rhs, old_offset, trigger=None):
            actions.append((rhs.copy(), old_offset))
            if len(actions) == 1:
                return self.accepted_action(recovery, rhs, old_offset, trigger)
            # Consistent coefficients/scores with deliberately worse objective:
            # the outer descent guard, not action exhaustion, rejects this.
            alpha = np.full(len(rhs), 1000.)
            return alpha, 0., recovery.K @ alpha

        watched = Mock(wraps=solve_kernel)
        with patch.object(KernelLinearSystem, 'solve_constrained',
                          side_effect=ConstrainedSolveFailure('injected constrained exhaustion')), \
             patch.object(KernelMMRecovery, 'step', action):
            with self.assertRaisesRegex(FloatingPointError, 'increased the original objective') as caught:
                self.run_guarded(watched)
        self.assertNotIsInstance(caught.exception, MMRecoveryExhausted)
        self.assertEqual(watched.call_count, 1)
        self.assertEqual(len(actions), 2)

    def test_initial_descent_rejection_then_failed_recovery_remains_untyped(self):
        def bad_proposal(system, rhs):
            alpha = np.full(len(rhs), 1000.)
            system.last_product = self.K @ alpha
            return alpha, 0.

        recovery_error = FloatingPointError('injected recovery failure after descent')
        watched = Mock(wraps=solve_kernel)
        with patch.object(KernelLinearSystem, 'solve_constrained', bad_proposal), \
             patch.object(KernelMMRecovery, 'step', side_effect=recovery_error) as recovery:
            with self.assertRaises(FloatingPointError) as caught:
                self.run_guarded(watched)
        self.assertIs(caught.exception, recovery_error)
        self.assertNotIsInstance(caught.exception, MMRecoveryExhausted)
        self.assertEqual(watched.call_count, 1)
        self.assertEqual(recovery.call_count, 1)
        self.assertIn('increased the original objective', recovery.call_args.args[2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
