"""Measured-accuracy recovery; low rcond alone is not a fit rejection."""
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import dwd._kernel_solver as implementation
import dwd._kernel_linear_system as linear_system
from dwd._kernel_solver import solve_kernel
from dwd._exact_kernel_factor import FactorResult


def bordered_mm(K, y, penalty, q, initial, offset, count):
    """Independent pivoted bordered-system oracle with scalar DWD loss."""
    n = len(y)
    t = n * q / (q + 1.) ** 2
    A = np.block([[K + 2 * penalty * t * np.eye(n), np.ones((n, 1))],
                  [np.ones((1, n)), np.zeros((1, 1))]])
    alpha, b = initial.copy(), float(offset)

    def objective():
        margins = y * (K @ alpha + b)
        loss = [1. - u if u <= q / (q + 1.)
                else q ** q / ((q + 1.) ** (q + 1.) * u ** q) for u in margins]
        return float(np.mean(loss) + penalty * alpha @ K @ alpha)

    history, decisions = [objective()], [K @ alpha + b]
    for _ in range(count):
        margins = y * (K @ alpha + b)
        slopes = np.array([-1. if u <= q / (q + 1.)
                           else -(q / ((q + 1.) * u)) ** (q + 1.) for u in margins])
        answer = np.linalg.solve(A, np.r_[K @ alpha - t * y * slopes / n, 0.])
        alpha, b = answer[:-1], b + answer[-1]
        history.append(objective())
        decisions.append(K @ alpha + b)
    return dict(alpha=alpha, offset=b, history=np.array(history), decisions=np.array(decisions))


class KernelAutoFallbackTests(unittest.TestCase):
    def setUp(self):
        self.K = np.diag([1., 1e-9, 1e-8, .5])
        self.y = np.array([-1., 1., -1., 1.])
        self.penalty = 1e-12

    def solve(self, backend='auto', **kwargs):
        options = dict(psd_known=True, max_iter=5, stopping='fixed')
        options.update(kwargs)
        return solve_kernel(self.K, self.y, self.penalty, backend=backend, **options)

    def test_low_rcond_auto_and_explicit_match_dense_and_spectral_oracles(self):
        initial = np.array([.1, -.15, .25, -.05])
        originals = (self.K.copy(), initial.copy(), self.y.copy())
        for q in (.5, 1., 2.):
            dense = bordered_mm(self.K, self.y, self.penalty, q, initial, .2, 5)
            spectral = self.solve('spectral', q=q, alpha_init=initial, offset_init=.2)
            for backend in ('auto', 'cholesky'):
                with self.subTest(q=q, backend=backend):
                    seen = []
                    result = self.solve(backend, q=q, alpha_init=initial, offset_init=.2,
                                        callback=lambda state: seen.append(dict(state)))
                    assert_allclose(result['objective_history'], dense['history'], rtol=2e-13, atol=3e-14)
                    assert_allclose(result['objective_history'], spectral['objective_history'], rtol=2e-13, atol=3e-14)
                    assert_allclose([s['decision_values'] for s in seen], dense['decisions'], rtol=2e-13, atol=3e-14)
                    assert_allclose(self.K @ result['alpha'] + result['offset'],
                                    self.K @ spectral['alpha'] + spectral['offset'], rtol=2e-13, atol=3e-14)
                    self.assertEqual(result['backend'], 'cholesky')
                    self.assertEqual([s['iteration'] for s in seen], list(range(6)))
                    assert_array_equal(seen[0]['alpha'], initial)
                    self.assertEqual(seen[0]['offset'], .2)
                    self.assertEqual(result['returned_iteration'], 5)
                    details = result['diagnostics']['linear_system_diagnostics']
                    self.assertLess(details['factorization_attempts'][0]['rcond'], np.sqrt(np.finfo(float).eps))
                    self.assertEqual(details['positive_eigenvalues_discarded'], 0)
                    self.assertEqual(details['added_objective_regularization'], 0.)
                    self.assertEqual(details['linear_solves'], 5)
                    self.assertFalse(details['residual_checks_are_interval_certificates'])
                    self.assertGreaterEqual(seen[0]['elapsed_seconds'], result['diagnostics']['setup_seconds'])
        for actual, original in zip((self.K, initial, self.y), originals):
            assert_array_equal(actual, original)

    def test_known_psd_accurate_path_avoids_eigendecomposition(self):
        with patch.object(implementation, 'validated_eigh', side_effect=AssertionError('Unnecessary eigenvalues')), \
                patch.object(implementation, '_eigensystem', side_effect=AssertionError('Unnecessary eigenvectors')):
            auto, explicit = self.solve(), self.solve('cholesky')
        assert_array_equal(auto['alpha'], explicit['alpha'])
        assert_array_equal(auto['objective_history'], explicit['objective_history'])
        self.assertFalse(auto['diagnostics']['auto_fallback'])

    def test_supplied_eigen_route_preserves_sign_and_order(self):
        original = self.solve('spectral', K_eig=(np.eye(4), self.K.diagonal()))
        order, signs = np.array([3, 1, 0, 2]), np.array([-1., 1., -1., 1.])
        with patch.object(linear_system, 'cho_factor', side_effect=AssertionError('Unexpected Cholesky')):
            changed = self.solve(K_eig=(np.eye(4)[:, order] * signs, self.K.diagonal()[order]))
        self.assertEqual(changed['backend'], 'spectral')
        assert_allclose(self.K @ changed['alpha'] + changed['offset'],
                        self.K @ original['alpha'] + original['offset'], rtol=2e-13, atol=3e-14)
        assert_allclose(changed['objective_history'], original['objective_history'], rtol=2e-13, atol=3e-14)

    def test_unknown_invalid_psd_and_unsafe_initial_state_reject_before_callbacks(self):
        bad = self.K.copy()
        bad[0, 1] = .1
        cases = [
            (bad, self.penalty, dict(psd_known=False), ValueError, 'symmetric'),
            (np.ones((4, 4)), .1, dict(psd_known=True, alpha_init=np.array([1e7, -1e7, 1e7, -1e7])),
             FloatingPointError, 'cancellation-prone'),
            (np.diag([-.1, 1., 1., 1.]), 1., dict(psd_known=False), ValueError, 'positive semidefinite'),
        ]
        for K, penalty, options, error, message in cases:
            seen = []
            with self.subTest(message=message), \
                    patch.object(linear_system, 'cho_factor', side_effect=AssertionError('Input validation bypassed')):
                with self.assertRaisesRegex(error, message):
                    solve_kernel(K, self.y, penalty, callback=lambda s: seen.append(s['iteration']), **options)
            self.assertEqual(seen, [])

    def test_singular_tiny_penalty_accepted_when_equations_are_accurate(self):
        for K in (np.zeros((4, 4)), np.ones((4, 4))):
            with self.subTest(kernel_trace=K.trace()):
                result = solve_kernel(K, self.y, 1e-12, psd_known=True, max_iter=5, stopping='fixed')
                assert_allclose(K @ result['alpha'] + result['offset'], np.zeros(4), atol=2e-12)
                assert_allclose(result['objective_history'], np.ones(6), atol=2e-12)
                self.assertEqual(result['diagnostics']['positive_eigenvalues_discarded'], 0)
                self.assertTrue(np.isfinite(result['alpha']).all())

    def test_failed_original_preparation_releases_array_before_centered_recovery(self):
        refs, calls = [], []
        original_factor = linear_system.cho_factor

        def fail_first(array, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                refs.append(weakref.ref(array))
                raise linear_system.LinAlgError('Injected original factorization failure')
            self.assertTrue(all(reference() is None for reference in refs))
            return original_factor(array, **kwargs)

        K, seen = np.diag([1., .2, .8, .5]), []
        with patch.object(linear_system, 'cho_factor', new=fail_first):
            result = solve_kernel(K, self.y, .1, psd_known=True, max_iter=4, stopping='fixed',
                                  callback=lambda s: seen.append(s['iteration']))
        dense = bordered_mm(K, self.y, .1, 1., np.zeros(4), 0., 4)
        assert_allclose(result['objective_history'], dense['history'], rtol=2e-13, atol=3e-14)
        assert_allclose(K @ result['alpha'] + result['offset'], dense['decisions'][-1], rtol=2e-13, atol=3e-14)
        self.assertEqual(seen, list(range(5)))
        self.assertEqual(calls, [1, 1])
        self.assertTrue(result['diagnostics']['auto_fallback'])
        self.assertEqual(result['diagnostics']['linear_system_diagnostics']['factor_representation'], 'centered')

    def test_failed_original_solves_release_factor_before_centered_recovery(self):
        refs, preparations = [], []
        original_factor, original_solve = linear_system.cho_factor, linear_system.cho_solve

        def tracked_factor(array, **kwargs):
            preparations.append(1)
            if len(preparations) > 1:
                self.assertTrue(all(reference() is None for reference in refs))
            result = original_factor(array, **kwargs)
            if len(preparations) == 1:
                refs.append(weakref.ref(result[0]))
            return result

        def inaccurate_original(factor, rhs, **kwargs):
            answer = original_solve(factor, rhs, **kwargs)
            if refs and factor[0] is refs[0]() and not np.array_equal(rhs, np.ones(4)):
                answer += np.array([.02, -.01, .03, -.04])
            return answer

        K, seen = np.diag([1., .2, .8, .5]), []
        with patch.object(linear_system, 'cho_factor', new=tracked_factor), \
                patch.object(linear_system, 'cho_solve', new=inaccurate_original):
            result = solve_kernel(K, self.y, .1, psd_known=True, max_iter=4, stopping='fixed',
                                  callback=lambda s: seen.append(s['iteration']))
        dense = bordered_mm(K, self.y, .1, 1., np.zeros(4), 0., 4)
        assert_allclose(result['objective_history'], dense['history'], rtol=2e-13, atol=3e-14)
        self.assertEqual(seen, list(range(5)))
        self.assertEqual(len(preparations), 2)
        self.assertGreaterEqual(result['diagnostics']['linear_system_diagnostics']['refinement_steps'], 3)

    def test_persistent_bad_solves_fail_when_certified_recovery_is_unavailable(self):
        seen = []
        unavailable = FactorResult('injected_certificate_unavailable', None, {})
        with patch.object(linear_system.KernelLinearSystem, '_candidate', return_value=(np.zeros(4), 0.)) as bad, \
                patch('dwd._exact_kernel_factor.exact_kernel_factor', return_value=unavailable):
            with self.assertRaisesRegex(FloatingPointError, 'Unable to solve.*accurately'):
                solve_kernel(np.eye(4), self.y, .1, psd_known=True,
                             callback=lambda s: seen.append(s['iteration']))
        self.assertEqual(seen, [0])
        # Two representations: initial action, one discarded native trial,
        # and the unchanged three committed-correction budget for each.
        self.assertEqual(bad.call_count, 10)

    def test_invalid_condition_diagnostics_cannot_escape_as_success(self):
        for rcond, status in ((1e-12, -1), (np.nan, 0), (-1e-12, 0)):
            seen = []
            with self.subTest(rcond=rcond, status=status), \
                    patch.object(linear_system, 'dpocon', return_value=(rcond, status)) as checks:
                with self.assertRaisesRegex(FloatingPointError, 'Invalid Cholesky'):
                    self.solve(callback=lambda s: seen.append(s['iteration']))
            self.assertEqual(seen, [])
            self.assertLessEqual(checks.call_count, 2)

    def test_initial_stop_and_max_iter_zero_preserve_original_state(self):
        initial = np.array([.1, -.15, .25, -.05])
        for max_iter, requested_stop in ((0, False), (5, True)):
            seen = []

            def callback(state):
                seen.append(dict(state))
                return requested_stop

            with self.subTest(max_iter=max_iter), \
                    patch.object(linear_system.KernelLinearSystem, 'solve_constrained',
                                 side_effect=AssertionError('Unexpected update')):
                result = self.solve(alpha_init=initial, offset_init=.2, max_iter=max_iter, callback=callback)
            self.assertEqual([s['iteration'] for s in seen], [0])
            assert_array_equal(result['alpha'], initial)
            assert_array_equal(seen[0]['training_scores'], self.K @ initial)
            self.assertEqual(result['offset'], .2)
            self.assertEqual(result['returned_iteration'], 0)

    def test_callback_stop_and_validation_restore_do_not_repeat_states(self):
        seen = []

        def stop_after_one(state):
            seen.append(state['iteration'])
            return state['iteration'] == 1

        stopped = self.solve(callback=stop_after_one)
        self.assertEqual(seen, [0, 1])
        self.assertEqual(stopped['termination_reason'], 'callback_stop')
        seen = []
        options = dict(stopping='validation', validation=(self.K, self.y), patience=1, min_delta=1., check_interval=1)
        restored = self.solve(callback=lambda s: seen.append(s['iteration']), **options)
        first = self.solve(max_iter=1)
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual([s['iteration'] for s in restored['validation_history']], [1, 2])
        self.assertEqual(restored['n_iter'], 2)
        self.assertEqual(restored['returned_iteration'], 1)
        self.assertEqual(restored['termination_reason'], 'validation_patience')
        assert_array_equal(restored['alpha'], first['alpha'])
        self.assertEqual(restored['offset'], first['offset'])
        self.assertEqual(restored['final_objective'], first['final_objective'])

    def test_callback_errors_do_not_trigger_numerical_recovery(self):
        seen, preparations = [], []
        original_prepare = linear_system.KernelLinearSystem._prepare

        def prepare(system, mode):
            preparations.append(mode)
            return original_prepare(system, mode)

        def fail(state):
            seen.append(state['iteration'])
            if state['iteration'] == 1:
                raise FloatingPointError('Synthetic callback failure')

        with patch.object(linear_system.KernelLinearSystem, '_prepare', new=prepare):
            with self.assertRaisesRegex(FloatingPointError, 'Synthetic callback failure'):
                solve_kernel(np.eye(4), self.y, .1, psd_known=True, callback=fail)
        self.assertEqual(seen, [0, 1])
        self.assertEqual(preparations, ['original'])

    def test_unknown_kernel_eigenvalue_failure_precedes_callbacks(self):
        for error in (linear_system.LinAlgError('Synthetic value failure'),
                      FloatingPointError('Synthetic nonfinite spectrum')):
            seen = []
            with self.subTest(error=type(error).__name__), \
                    patch.object(implementation, 'validated_eigh', side_effect=error):
                with self.assertRaises(type(error)):
                    solve_kernel(np.eye(4), self.y, .1, psd_known=False,
                                 callback=lambda s: seen.append(s['iteration']))
            self.assertEqual(seen, [])

    def test_rotated_low_rcond_case_preserves_objective_and_scores(self):
        rng = np.random.default_rng(67011)
        n = 24
        U, _ = np.linalg.qr(rng.normal(size=(n, n)))
        K = (U * np.r_[3e-8, np.linspace(.1, 1., n - 1)]) @ U.T
        y, penalty = np.r_[-np.ones(15), np.ones(9)], 1e-10 / (n / 2.)
        initial = np.linspace(-.01, .01, n)
        for q in (.5, 1., 2.):
            with self.subTest(q=q):
                options = dict(q=q, max_iter=7, stopping='fixed', alpha_init=initial, offset_init=.2)
                result = solve_kernel(K, y, penalty, backend='auto', psd_known=True, **options)
                explicit = solve_kernel(K, y, penalty, backend='cholesky', psd_known=True, **options)
                spectral = solve_kernel(K, y, penalty, backend='spectral', **options)
                dense = bordered_mm(K, y, penalty, q, initial, .2, 7)
                assert_array_equal(result['objective_history'], explicit['objective_history'])
                assert_allclose(result['objective_history'], spectral['objective_history'], atol=2e-9, rtol=2e-9)
                assert_allclose(result['objective_history'], dense['history'], atol=2e-9, rtol=2e-9)
                assert_allclose(K @ result['alpha'] + result['offset'], dense['decisions'][-1], atol=2e-8, rtol=2e-8)
                self.assertEqual(result['backend'], 'cholesky')
                self.assertEqual(result['returned_iteration'], 7)
                self.assertLess(result['diagnostics']['linear_system_diagnostics']['factorization_attempts'][0]['rcond'],
                                np.sqrt(np.finfo(float).eps))


if __name__ == '__main__':
    unittest.main(verbosity=2)
