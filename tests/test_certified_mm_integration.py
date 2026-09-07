"""Transactional integration controls for bounded original-function MM recovery.

Most numerical failures below are explicitly injected on well-conditioned
dyadic kernels. They test control flow, not a claim of natural instability.
The final public control reproduces the exact stored n=30 rank-five kernel.
"""
import hashlib
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

import dwd._kernel_solver as solver
import dwd._exact_kernel_factor as factors
from dwd._kernel_mm_recovery import KernelMMRecovery
from dwd._kernel_linear_system import KernelLinearSystem
from dwd._spectral_linear_system import SpectralLinearSystem
from dwd._accelerated_mm import RestartedMM
from dwd.gen_dwd import V, V_grad
from dwd.gen_kern_dwd import KernGDWD


def fixture(rank=5):
    if rank == 1:
        F = np.array([[-3.], [1.], [-1.], [3.], [2.], [-2.]])/8.
    else:
        F = np.vstack((np.eye(5), [[1, -1, 2, 0, 1], [-2, 1, 0, 1, -1],
                                   [0, 2, -1, 1, 2]]))/8.
    return F @ F.T, np.where(np.arange(len(F)) % 2, 1., -1.)


def dense_step(K, rhs, delta, oldb):
    """Independent small bordered solve, comparing functions rather than gauge."""
    n = len(K)
    bordered = np.block([[K+delta*np.eye(n), np.ones((n, 1))],
                         [np.ones((1, n)), np.zeros((1, 1))]])
    value = solve(bordered, np.r_[rhs, 0.], assume_a='sym')
    return K@value[:-1], oldb+value[-1]


class CertifiedMMIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=2)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def setUp(self):
        self.K, self.y = fixture()
        self.penalty = .13
        self.alpha = np.arange(len(self.K), dtype=float)/128.
        self.offset = .17

    def fit(self, **kwargs):
        options = dict(alpha_init=self.alpha, offset_init=self.offset,
                       max_iter=3, stopping='fixed', psd_known=True)
        options.update(kwargs)
        return solver.solve_kernel(self.K, self.y, self.penalty, **options)

    def test_healthy_mm_never_requests_exact_certificate(self):
        with patch.object(factors, 'exact_kernel_factor', side_effect=AssertionError('Unneeded certificate')):
            for implementation in ('reference', 'optimized'):
                result = self.fit(implementation=implementation)
                self.assertNotIn('mm_function_recovery', result['diagnostics'])
            self.fit(acceleration='restart')
            self.fit(backend='lbfgs')

    def test_same_rhs_previous_offset_and_one_observation_per_general_q_update(self):
        original = KernelMMRecovery.step
        for rank in (1, 5):
            K, y = fixture(rank)
            for implementation in ('reference', 'optimized'):
                for q in (.5, 1., 2.):
                    with self.subTest(rank=rank, implementation=implementation, q=q):
                        states, actions = [], []
                        alpha = np.arange(len(K), dtype=float)/128.
                        def watched(instance, rhs, oldb, trigger=None):
                            actions.append((rhs.copy(), oldb))
                            return original(instance, rhs, oldb, trigger)
                        target = (SpectralLinearSystem if implementation == 'reference'
                                  else KernelLinearSystem)
                        method = 'refine_candidate' if implementation == 'reference' else 'solve_constrained'
                        with patch.object(target, method, side_effect=FloatingPointError('Injected ordinary rejection')) as failed, \
                             patch.object(KernelMMRecovery, 'step', watched):
                            result = solver.solve_kernel(K, y, self.penalty, q=q,
                                implementation=implementation, alpha_init=alpha, offset_init=self.offset,
                                max_iter=3, stopping='fixed', callback=lambda s: states.append(dict(s)))
                        self.assertEqual(failed.call_count, 1)
                        self.assertEqual([s['iteration'] for s in states], [0, 1, 2, 3])
                        self.assertEqual(len(actions), 3)
                        step = len(K)*q/(q+1)**2
                        for previous, current, (rhs, oldb) in zip(states, states[1:], actions):
                            expected_rhs = previous['training_scores']-step*y*V_grad(
                                y*previous['decision_values'], q=q)/len(K)
                            assert_allclose(rhs, expected_rhs, atol=2e-16, rtol=2e-16)
                            self.assertEqual(oldb, previous['offset'])
                            g, b = dense_step(K, rhs, 2*self.penalty*step, oldb)
                            assert_allclose(current['decision_values'], g+b, atol=2e-12, rtol=2e-12)
                        self.assertEqual(result['n_iter'], 3)
                        self.assertEqual(len(result['objective_history']), 4)
                        info = result['diagnostics']['mm_function_recovery']
                        self.assertEqual(info['accepted_actions'], 3)
                        self.assertEqual(info['native_small_spectral_actions']+info['rounded_exact_actions'], 3)
                        self.assertFalse(info['reduced_coefficient_gauge_checked'])
                        self.assertTrue(info['true_mm_function_checked'])
                        self.assertFalse(info['kernel_approximation_used'])
                        self.assertEqual(info['positive_directions_discarded'], 0)

    def test_bad_ordinary_proposal_is_rolled_back_before_identical_recovery(self):
        native = KernelMMRecovery.step
        calls, seen = [], []
        def corrupt(instance, rhs):
            # A new bad proposal has already replaced the local alpha/scores;
            # descent rejection must restore the previous offset and RHS.
            instance.last_product = np.full(len(rhs), 1e5)
            return np.full(len(rhs), 1e5), 73.
        def watched(instance, rhs, oldb, trigger=None):
            calls.append((rhs.copy(), oldb, trigger))
            return native(instance, rhs, oldb, trigger)
        with patch.object(KernelLinearSystem, 'solve_constrained', corrupt), \
             patch.object(KernelMMRecovery, 'step', watched):
            result = self.fit(max_iter=1, callback=lambda s: seen.append(dict(s)))
        expected = self.K@self.alpha-len(self.K)/4*self.y*V_grad(
            self.y*(self.K@self.alpha+self.offset))/len(self.K)
        assert_allclose(calls[0][0], expected, atol=1e-16)
        self.assertEqual(calls[0][1], self.offset)
        self.assertIn('increased the original objective', calls[0][2])
        self.assertEqual([s['iteration'] for s in seen], [0, 1])
        self.assertLess(result['objective_history'][1], result['objective_history'][0])

    def test_observer_floating_errors_are_not_recovered_or_replayed(self):
        for accelerated in (None, 'restart'):
            seen = []
            def observer(state):
                seen.append(state['iteration'])
                if state['iteration'] == 1:
                    raise FloatingPointError('User observer error')
            with patch.object(KernelMMRecovery, 'step', side_effect=AssertionError('Observer was intercepted')):
                with self.assertRaisesRegex(FloatingPointError, 'User observer error'):
                    self.fit(callback=observer, acceleration=accelerated)
            self.assertEqual(seen, [0, 1])

    def test_failed_recovery_never_records_an_uncompleted_update(self):
        seen = []
        with patch.object(KernelLinearSystem, 'solve_constrained', side_effect=FloatingPointError('Original failure')), \
             patch.object(KernelMMRecovery, 'step', side_effect=FloatingPointError('Recovery failure')) as recovery:
            with self.assertRaisesRegex(FloatingPointError, 'Recovery failure'):
                self.fit(callback=lambda state: seen.append(state['iteration']))
        self.assertEqual(recovery.call_count, 1)
        self.assertEqual(seen, [0])

    def test_callback_stops_after_one_recovered_update(self):
        for use_exception in (False, True):
            seen = []
            def observer(state):
                seen.append(state['iteration'])
                if state['iteration'] == 1:
                    if use_exception:
                        raise StopIteration
                    return True
            with patch.object(KernelLinearSystem, 'solve_constrained', side_effect=FloatingPointError('Injected')):
                result = self.fit(callback=observer)
            self.assertEqual(seen, [0, 1])
            self.assertEqual(result['termination_reason'], 'callback_stop')
            self.assertEqual(result['n_iter'], 1)
            self.assertEqual(result['diagnostics']['mm_function_recovery']['accepted_actions'], 1)

    def test_fixed_objective_and_optimality_boundaries_do_not_add_updates(self):
        with patch.object(KernelLinearSystem, 'solve_constrained', side_effect=FloatingPointError('Injected')):
            fixed = self.fit(max_iter=3)
            objective = self.fit(stopping='objective', obj_tol=2.)
            optimal = self.fit(stopping='optimality', tol=100.)
            zero = self.fit(max_iter=0)
        self.assertEqual((fixed['n_iter'], len(fixed['objective_history'])), (3, 4))
        self.assertEqual((objective['n_iter'], objective['termination_reason']), (1, 'objective_tolerance'))
        self.assertEqual((optimal['n_iter'], optimal['termination_reason']), (0, 'optimality_tolerance'))
        self.assertNotIn('mm_function_recovery', optimal['diagnostics'])
        self.assertEqual(zero['n_iter'], 0)
        self.assertNotIn('mm_function_recovery', zero['diagnostics'])

    def test_validation_restores_pre_recovery_representation_and_values(self):
        native = KernelLinearSystem.solve_constrained
        count = 0
        seen = []
        def fail_second(instance, rhs):
            nonlocal count
            count += 1
            if count == 2:
                raise FloatingPointError('Injected later failure')
            return native(instance, rhs)
        with patch.object(KernelLinearSystem, 'solve_constrained', fail_second):
            result = self.fit(stopping='validation', patience=2, check_interval=1,
                validation=(np.zeros((4, len(self.K))), np.array([-1., 1., -1., 1.])),
                callback=lambda s: seen.append(dict(s)))
        self.assertEqual((result['n_iter'], result['returned_iteration']), (3, 1))
        self.assertEqual(result['termination_reason'], 'validation_patience')
        self.assertEqual([s['iteration'] for s in seen], [0, 1, 2, 3])
        assert_array_equal(result['alpha'], seen[1]['alpha'])
        self.assertEqual(result['offset'], seen[1]['offset'])
        self.assertAlmostEqual(result['final_objective'], seen[1]['objective'], places=14)
        self.assertEqual(result['diagnostics']['mm_function_recovery']['accepted_actions'], 2)
        self.assertFalse(result['diagnostics']['returned_state_uses_certified_columns'])
        self.assertNotIn('coefficient_representation', result['diagnostics'])

    def test_accelerated_hook_receives_total_rhs_zero_old_offset(self):
        original = KernelMMRecovery.step
        actions, seen = [], []
        def watched(instance, rhs, oldb, trigger=None):
            actions.append((rhs.copy(), oldb))
            return original(instance, rhs, oldb, trigger)
        with patch.object(KernelLinearSystem, 'solve_constrained', side_effect=FloatingPointError('Injected')), \
             patch.object(KernelMMRecovery, 'step', watched):
            result = self.fit(acceleration='restart', callback=lambda s: seen.append(dict(s)))
        self.assertEqual([s['iteration'] for s in seen], [0, 1, 2, 3])
        self.assertEqual(result['diagnostics']['acceleration_accepted_updates'], 3)
        self.assertEqual(result['diagnostics']['acceleration_proposals'], len(actions))
        self.assertTrue(all(oldb == 0. for _, oldb in actions))
        total = seen[0]['decision_values']
        expected = total-len(self.K)/4*self.y*V_grad(self.y*total)/len(self.K)
        assert_allclose(actions[0][0], expected, atol=1e-16)
        # The first two accepted proposals have no momentum. Compare both to
        # a separate total-decision bordered action (absolute intercept).
        for previous, current, (rhs, _) in zip(seen[:2], seen[1:3], actions[:2]):
            g, b = dense_step(self.K, rhs, self.penalty*len(self.K)/2, 0.)
            assert_allclose(current['decision_values'], g+b, atol=2e-12, rtol=2e-12)

    def test_failed_momentum_certificate_then_native_restart_has_honest_metadata(self):
        native = KernelLinearSystem.solve_constrained
        calls, seen = [], []
        def fail_third(instance, rhs):
            calls.append(rhs.copy())
            if len(calls) == 3:
                raise FloatingPointError('Injected extrapolated solve failure')
            return native(instance, rhs)
        inapplicable = factors.FactorResult('injected_budget_exhausted', None, {})
        with patch.object(KernelLinearSystem, 'solve_constrained', fail_third), \
             patch.object(factors, 'exact_kernel_factor', return_value=inapplicable) as certificate:
            result = self.fit(acceleration='restart', callback=lambda s: seen.append(dict(s)))
        self.assertEqual(certificate.call_count, 1)
        self.assertEqual(len(calls), 4)
        self.assertEqual([s['iteration'] for s in seen], [0, 1, 2, 3])
        info = result['diagnostics']
        self.assertEqual(info['acceleration_accepted_updates'], 3)
        self.assertEqual(info['acceleration_numerical_restarts'], 1)
        self.assertEqual(info['acceleration_proposals'], 4)
        self.assertEqual(info['mm_function_recovery']['accepted_actions'], 0)
        self.assertFalse(info['returned_state_uses_certified_columns'])
        self.assertNotIn('coefficient_representation', info)

    def test_active_recovery_rejected_proposal_counts_actions_separately(self):
        native = RestartedMM._proposal
        calls, seen = [], []
        def reject_third(instance, total):
            result = native(instance, total)
            calls.append(result)
            if len(calls) == 3:
                # The action passed its own equations, but momentum may still
                # fail the outer descent rule. Only the retry is an MM update.
                return result[:3]+(1e4,)
            return result
        with patch.object(KernelLinearSystem, 'solve_constrained', side_effect=FloatingPointError('Injected')), \
             patch.object(RestartedMM, '_proposal', reject_third):
            result = self.fit(acceleration='restart', callback=lambda s: seen.append(s['iteration']))
        self.assertEqual(seen, [0, 1, 2, 3])
        info = result['diagnostics']
        self.assertEqual(info['mm_function_recovery']['accepted_actions'], 4)
        self.assertEqual(info['acceleration_proposals'], 4)
        self.assertEqual(info['acceleration_accepted_updates'], 3)
        self.assertEqual(info['acceleration_restarts'], 1)

    def test_public_previously_rejected_exact_rank_five_kernel(self):
        n = 30
        F = np.random.RandomState(91300+n).randint(-3, 4, (n, 5)).astype(float)/8.
        K = F@F.T
        self.assertEqual(hashlib.sha256(K.tobytes()).hexdigest(),
                         'eb13eba26fcbf855af04261aa2be6eb24481e063b6c35a1272c9972821676c14')
        y = np.where(np.arange(n) % 2, 1., -1.)
        original_bytes = K.tobytes()
        for implementation, initialization in [('reference', 'zero'), ('reference', 'auto'), ('optimized', 'zero')]:
            with self.subTest(implementation=implementation, initialization=initialization):
                states = []
                model = KernGDWD(kernel='precomputed', lambd=2e-14/n, q=1,
                    implementation=implementation, initialization=initialization,
                    random_state=314159, stopping='fixed', max_iter=5,
                    callback=lambda s: states.append(dict(s))).fit(K, y)
                self.assertEqual(model.n_iter_, 5)
                self.assertEqual([s['iteration'] for s in states], list(range(6)))
                self.assertTrue(np.isfinite(model.decision_function(K)).all())
                self.assertLessEqual(np.max(np.diff(model.objective_history_)), 1e-9)
                # Function equality and exact-factor identity have separate
                # Decimal/Fraction oracle coverage in the helper tests.
                assert_allclose(model.decision_function(K), states[-1]['decision_values'], atol=1e-9, rtol=1e-9)
                self.assertEqual(K.tobytes(), original_bytes)


if __name__ == '__main__':
    unittest.main(verbosity=2)
