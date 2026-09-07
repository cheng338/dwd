"""Independent small-system mathematics for the new full-kernel backends.

The references below use an augmented dense solve, feature-space derivatives,
an analytic free-intercept optimum, and optional convex primal/dual programs.
They do not use another implementation of the Schur update as their oracle.
"""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import cho_factor

from dwd._kernel_solver import optimality_diagnostics, solve_kernel
from dwd.gen_kern_dwd import KernGDWD


def scalar_loss(u, q):
    if u <= q / (q + 1):
        return 1. - u
    return q ** q / ((q + 1) ** (q + 1) * u ** q)


def scalar_slope(u, q):
    if u <= q / (q + 1):
        return -1.
    return -(q / ((q + 1) * u)) ** (q + 1)


def independent_objective(K, y, alpha, b, penalty, q):
    return float(np.mean([scalar_loss(u, q) for u in y * (K @ alpha + b)])
                 + penalty * alpha @ K @ alpha)


def dense_augmented_step(K, y, alpha, b, penalty, q, singular):
    n = len(y)
    t = n * q / (q + 1) ** 2
    shift = 2 * penalty * t
    z = y * np.array([scalar_slope(u, q) for u in y * (K @ alpha + b)]) / n
    ones = np.ones(n)
    P = np.block([[np.array([[n]]), (ones @ K)[None, :]],
                  [(K @ ones)[:, None], K @ K + shift * K]])
    rhs = t * np.r_[z.sum(), K @ (z + 2 * penalty * alpha)]
    step = np.linalg.pinv(P, rcond=1e-12) @ rhs if singular else np.linalg.solve(P, rhs)
    return step, P, rhs


class ExactKernelBackendMathTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(73012)
        self.X = rng.normal(size=(10, 3))
        self.K_singular = self.X @ self.X.T
        self.K = self.K_singular + .6 * np.eye(len(self.X))
        self.y = np.r_[np.ones(6), -np.ones(4)]
        self.alpha = rng.normal(size=len(self.y)) / 10
        self.offset = .23

    def test_one_mm_step_matches_augmented_equations_general_q_and_singular(self):
        for q in (.2, 1., 3.5):
            for singular, K in ((False, self.K), (True, self.K_singular)):
                for backend in ('spectral', 'cholesky'):
                    with self.subTest(q=q, singular=singular, backend=backend):
                        expected, P, rhs = dense_augmented_step(
                            K, self.y, self.alpha, self.offset, .13, q, singular)
                        fit = solve_kernel(K, self.y, .13, q=q, backend=backend,
                            alpha_init=self.alpha, offset_init=self.offset,
                            max_iter=1, stopping='fixed')
                        actual = np.r_[self.offset - fit['offset'], self.alpha - fit['alpha']]
                        assert_allclose(P @ actual, rhs, rtol=2e-10, atol=3e-11)
                        assert_allclose(actual[0] + K @ actual[1:],
                                        expected[0] + K @ expected[1:], rtol=2e-10, atol=3e-11)
                        if not singular:
                            assert_allclose(actual, expected, rtol=3e-10, atol=3e-11)
                        self.assertLessEqual(np.max(np.diff(fit['objective_history'])), 2e-12)

    def test_spectral_and_cholesky_multi_step_scores_objectives_and_intercept(self):
        for q in (.2, 1., 3.5):
            with self.subTest(q=q):
                kwargs = dict(q=q, alpha_init=self.alpha, offset_init=self.offset,
                              max_iter=35, stopping='fixed')
                spectral = solve_kernel(self.K_singular, self.y, .13, backend='spectral', **kwargs)
                cholesky = solve_kernel(self.K_singular, self.y, .13, backend='cholesky', **kwargs)
                assert_allclose(self.K_singular @ spectral['alpha'] + spectral['offset'],
                                self.K_singular @ cholesky['alpha'] + cholesky['offset'],
                                rtol=5e-11, atol=5e-11)
                assert_allclose(spectral['objective_history'], cholesky['objective_history'],
                                rtol=5e-11, atol=5e-12)

    def test_free_intercept_zero_kernel_has_analytic_imbalanced_optimum(self):
        y = np.r_[np.ones(6), -np.ones(2)]
        K = np.zeros((len(y), len(y)))
        for q in (.2, 1., 3.5):
            optimum = q / (q + 1) * 3. ** (1 / (q + 1))
            for backend in ('spectral', 'cholesky', 'lbfgs'):
                with self.subTest(q=q, backend=backend):
                    fit = solve_kernel(K, y, .7, q=q, backend=backend,
                                       stopping='optimality', tol=2e-9, max_iter=3000)
                    self.assertAlmostEqual(fit['offset'], optimum, delta=2e-7)
                    self.assertEqual(fit['rkhs_norm_squared'], 0.)
                    self.assertEqual(fit['C'], 0.)
                    self.assertLess(abs(fit['intercept_gradient']), 3e-9)
                    self.assertLess(fit['dual_gap'], 1e-8)

    def test_default_stopping_matches_first_absolute_objective_change(self):
        for backend in ('spectral', 'cholesky'):
            with self.subTest(backend=backend):
                full = solve_kernel(self.K, self.y, .13, backend=backend,
                                    max_iter=100, stopping='fixed')
                changes = np.abs(np.diff(full['objective_history']))
                hits = np.flatnonzero(changes < 1e-5)
                expected = int(hits[0] + 1) if len(hits) else 100
                default = solve_kernel(self.K, self.y, .13, backend=backend)
                self.assertEqual(default['n_iter'], expected)
                assert_allclose(default['objective_history'], full['objective_history'][:expected + 1],
                                rtol=0, atol=1e-14)
                self.assertEqual(default['termination_reason'],
                                 'objective_tolerance' if len(hits) else 'max_iter')

    def test_objective_rule_is_strict_and_zero_tolerance_exhausts_budget(self):
        K = np.zeros((4, 4))
        y = np.array([-1., 1., -1., 1.])
        for backend in ('spectral', 'cholesky'):
            default = solve_kernel(K, y, .2, backend=backend)
            self.assertEqual(default['n_iter'], 1)
            self.assertEqual(default['termination_reason'], 'objective_tolerance')
            no_tolerance = solve_kernel(K, y, .2, backend=backend, obj_tol=0)
            self.assertEqual(no_tolerance['n_iter'], 100)
            self.assertEqual(no_tolerance['termination_reason'], 'max_iter')

    def test_zero_iteration_preserves_initial_function_and_calls_callback_once(self):
        for backend in ('spectral', 'cholesky', 'lbfgs'):
            with self.subTest(backend=backend):
                seen = []
                fit = solve_kernel(self.K, self.y, .13, backend=backend, max_iter=0,
                    alpha_init=self.alpha, offset_init=self.offset,
                    callback=lambda state: seen.append(state['iteration']))
                self.assertEqual(seen, [0])
                self.assertEqual(fit['n_iter'], 0)
                self.assertEqual(fit['returned_iteration'], 0)
                self.assertEqual(len(fit['objective_history']), 1)
                assert_allclose(fit['alpha'], self.alpha, rtol=0, atol=2e-15)
                self.assertEqual(fit['offset'], self.offset)
                self.assertAlmostEqual(fit['final_objective'],
                    independent_objective(self.K, self.y, self.alpha, self.offset, .13, 1.), places=13)

    def test_all_positive_eigenmodes_are_retained_including_tiny_modes(self):
        values = np.array([1e-14, 1e-10, .01, .1, .7, 1.])
        K = np.diag(values)
        alpha = np.array([1e6, 1e3, .4, -.2, .1, .3])
        y = np.array([1., -1., 1., -1., 1., -1.])
        for backend in ('spectral', 'lbfgs'):
            fit = solve_kernel(K, y, .2, backend=backend, max_iter=0,
                               alpha_init=alpha, K_eig=(np.eye(6), values))
            assert_array_equal(fit['alpha'], alpha)
            self.assertEqual(fit['diagnostics']['positive_eigenvalues_retained'], 6)
            self.assertEqual(fit['diagnostics']['positive_eigenvalues_discarded'], 0)
            self.assertAlmostEqual(fit['rkhs_norm_squared'], alpha @ K @ alpha, places=14)

    def test_precomputed_eigenvector_sign_permutation_invariance(self):
        values, vectors = np.linalg.eigh(self.K)
        rng = np.random.RandomState(45)
        order = rng.permutation(len(values))
        signs = rng.choice([-1, 1], len(values))
        a = solve_kernel(self.K, self.y, .13, q=3.5, backend='spectral',
                         K_eig=(vectors, values), max_iter=12, stopping='fixed')
        b = solve_kernel(self.K, self.y, .13, q=3.5, backend='spectral',
                         K_eig=((vectors * signs)[:, order], values[order]),
                         max_iter=12, stopping='fixed')
        assert_allclose(a['alpha'], b['alpha'], rtol=0, atol=2e-14)
        assert_allclose(a['objective_history'], b['objective_history'], rtol=0, atol=2e-14)

    def test_shifted_cholesky_success_does_not_accept_unknown_indefinite_kernel(self):
        K = np.diag([-.1, 1., 1., 1.])
        y = np.array([-1., 1., -1., 1.])
        # q=1,n=4,lambda=1 implies shift=2; the shifted matrix is SPD.
        cho_factor(K + 2 * np.eye(4))
        for backend in ('auto', 'spectral', 'cholesky', 'lbfgs'):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                solve_kernel(K, y, 1., backend=backend)
        with self.assertRaises(ValueError):
            KernGDWD(kernel='precomputed', backend='cholesky').fit(K, y)

    def test_known_psd_auto_path_can_avoid_eigendecomposition(self):
        with patch('dwd._eigen.eigh', side_effect=AssertionError('Unexpected eigendecomposition')):
            fit = solve_kernel(self.K, self.y, .13, backend='auto', psd_known=True,
                               max_iter=2, stopping='fixed')
        self.assertEqual(fit['backend'], 'cholesky')
        self.assertIn('internally constructed PSD', fit['diagnostics']['psd_validation'])

    def test_tiny_penalty_singular_mm_remains_guarded_and_lbfgs_is_checked(self):
        for backend in ('spectral', 'cholesky'):
            with self.subTest(backend=backend), self.assertRaises(FloatingPointError):
                solve_kernel(self.K_singular, self.y, 1e-12, backend=backend)
        with self.assertRaises(FloatingPointError):
            solve_kernel(self.K_singular, self.y, 1e-12, backend='cholesky', psd_known=True)
        # The feature-coordinate optimizer can avoid the enormous nullspace
        # coefficients of MM on this example. Acceptance uses original-K
        # score/objective reconstruction, rather than blanket rank rejection.
        result = solve_kernel(self.K_singular, self.y, 1e-12, backend='lbfgs')
        self.assertTrue(np.isfinite(result['alpha']).all())
        self.assertLessEqual(result['diagnostics']['score_reconstruction_max_error'],
                             5e-7*max(1., np.max(np.abs(self.K_singular@result['alpha']))))
        self.assertAlmostEqual(result['final_objective'],
            independent_objective(self.K_singular, self.y, result['alpha'], result['offset'], 1e-12, 1.), places=10)

    def test_unsafe_external_nullspace_initialization_is_rejected(self):
        rng = np.random.RandomState(32)
        features = rng.normal(size=(12, 3))
        K = features @ features.T
        null_vector = np.linalg.svd(features.T, full_matrices=True)[2][-1]
        alpha = 1e7 * null_vector
        y = np.resize([-1., 1.], 12)
        # The feature-space norm is effectively zero, but evaluating alpha^T K
        # alpha from the rounded dense Gram matrix can spuriously give ~0.01.
        # A shifted-system condition check and small score reconstruction error
        # do not protect this externally supplied, cancellation-heavy state.
        self.assertLess(np.linalg.norm(features.T @ alpha) ** 2, 1e-12)
        for backend in ('spectral', 'cholesky', 'lbfgs'):
            for budget in (0, 1):
                with self.subTest(backend=backend, budget=budget):
                    with self.assertRaises((ValueError, FloatingPointError)):
                        solve_kernel(K, y, .1, backend=backend,
                                     alpha_init=alpha, max_iter=budget)

    def test_large_well_aligned_initial_coefficient_is_not_rejected_by_size_alone(self):
        K = np.diag([1e-12, 1., 2., 3.])
        alpha = np.array([1e8, 0., 0., 0.])
        y = np.array([-1., 1., -1., 1.])
        # |alpha|^T |K| |alpha| equals the norm here: there is no cancellation.
        # A coarse ||K||*||alpha||^2 bound would incorrectly reject this case.
        for backend in ('spectral', 'cholesky', 'lbfgs'):
            with self.subTest(backend=backend):
                fit = solve_kernel(K, y, .2, backend=backend,
                                   alpha_init=alpha, max_iter=0)
                self.assertAlmostEqual(fit['rkhs_norm_squared'], 1e4, places=9)
                self.assertAlmostEqual(fit['final_objective'],
                    independent_objective(K, y, alpha, 0., .2, 1.), places=9)

    def test_diagnostics_agree_with_independent_feature_space_finite_difference(self):
        q, penalty = 1.7, .13
        w = self.X.T @ self.alpha
        theta = np.r_[self.offset, w]

        def feature_objective(t):
            return np.mean([scalar_loss(u, q) for u in self.y * (self.X @ t[1:] + t[0])]) + penalty * (t[1:] @ t[1:])

        gradient = np.zeros_like(theta)
        for i in range(len(theta)):
            delta = np.zeros_like(theta)
            delta[i] = 2e-6
            gradient[i] = (feature_objective(theta + delta) - feature_objective(theta - delta)) / (4e-6)
        diag = optimality_diagnostics(self.K_singular, self.y, self.alpha, self.offset, penalty, q)
        self.assertAlmostEqual(diag['final_objective'], feature_objective(theta), places=14)
        self.assertAlmostEqual(diag['intercept_gradient'], gradient[0], delta=1e-9)
        self.assertAlmostEqual(diag['rkhs_gradient_norm'],
                               max(abs(gradient[0]), np.linalg.norm(gradient[1:])), delta=2e-9)
        expected_C = (q + 1) ** (q + 1) / q ** q * np.linalg.norm(w) ** (q + 1)
        self.assertAlmostEqual(diag['C'], expected_C, delta=2e-13)
        self.assertGreaterEqual(diag['dual_gap'], -1e-12)
        self.assertLess(diag['dual_equality_residual'], 1e-13)

    def test_validation_patience_restores_earliest_best_and_counts_terminal_callback(self):
        K = np.zeros((8, 8))
        y = np.r_[np.ones(6), -np.ones(2)]
        validation = (np.zeros((4, 8)), np.ones(4))
        for backend in ('spectral', 'cholesky', 'lbfgs'):
            with self.subTest(backend=backend):
                seen = []
                fit = solve_kernel(K, y, .2, backend=backend, stopping='validation',
                    validation=validation, patience=2, max_iter=20,
                    callback=lambda state: seen.append(state['iteration']))
                self.assertEqual(fit['termination_reason'], 'validation_patience')
                self.assertEqual(fit['n_iter'], 3)
                self.assertEqual(fit['returned_iteration'], 1)
                self.assertEqual(seen, [0, 1, 2, 3])
                first = solve_kernel(K, y, .2, backend=backend, stopping='fixed', max_iter=1)
                assert_allclose(fit['alpha'], first['alpha'], rtol=0, atol=1e-14)
                self.assertAlmostEqual(fit['offset'], first['offset'], places=14)
                self.assertAlmostEqual(fit['final_objective'], fit['objective_history'][1], places=14)

    def test_validation_check_interval_counts_checks_not_updates(self):
        K = np.zeros((8, 8))
        y = np.r_[np.ones(6), -np.ones(2)]
        fit = solve_kernel(K, y, .2, stopping='validation', backend='spectral',
            validation=(np.zeros((4, 8)), np.ones(4)), patience=2,
            check_interval=2, max_iter=20)
        # The first update is always monitored; subsequent regular checks occur
        # on multiples of check_interval. Patience counts observations, not steps.
        self.assertEqual([h['iteration'] for h in fit['validation_history']], [1, 2, 4])
        self.assertEqual(fit['n_iter'], 4)
        self.assertEqual(fit['returned_iteration'], 1)

    def test_callback_snapshot_cannot_mutate_state_and_can_stop(self):
        seen = []

        def callback(state):
            seen.append(state['iteration'])
            with self.assertRaises(TypeError):
                state['offset'] = 7.
            with self.assertRaises(ValueError):
                state['alpha'][0] = 7.
            # Even deliberately making the copy writable cannot alter the solve.
            state['alpha'].flags.writeable = True
            state['alpha'][:] = 123.
            return state['iteration'] == 4

        fit = solve_kernel(self.K, self.y, .13, callback=callback, stopping='fixed', max_iter=20)
        reference = solve_kernel(self.K, self.y, .13, stopping='fixed', max_iter=4)
        self.assertEqual(seen, [0, 1, 2, 3, 4])
        self.assertEqual(fit['termination_reason'], 'callback_stop')
        assert_allclose(fit['alpha'], reference['alpha'], rtol=0, atol=0)
        self.assertEqual(fit['offset'], reference['offset'])

    def test_lbfgs_native_stationarity_reports_actual_updates_under_fixed_budget(self):
        fit = solve_kernel(np.zeros((4, 4)), np.array([-1., 1., -1., 1.]),
                           .1, backend='lbfgs', stopping='fixed', max_iter=100)
        self.assertEqual(fit['n_iter'], 0)
        self.assertEqual(fit['termination_reason'], 'optimizer_stationary')
        self.assertTrue(fit['converged'])
        self.assertEqual(len(fit['objective_history']), 1)

    def test_lbfgs_initial_callback_cancellation_does_not_launch_optimizer(self):
        for budget in (0, 20):
            with self.subTest(budget=budget):
                seen = []

                def stop_at_initial(state):
                    seen.append(state['iteration'])
                    raise StopIteration()

                with patch('dwd._kernel_solver.minimize', side_effect=AssertionError('Optimizer should not start')):
                    fit = solve_kernel(self.K, self.y, .13, backend='lbfgs',
                        max_iter=budget, callback=stop_at_initial,
                        alpha_init=self.alpha, offset_init=self.offset)
                self.assertEqual(seen, [0])
                self.assertEqual(fit['n_iter'], 0)
                self.assertEqual(fit['termination_reason'], 'callback_stop')
                assert_allclose(fit['alpha'], self.alpha, rtol=0, atol=2e-15)
                self.assertEqual(fit['offset'], self.offset)

    def test_lbfgs_accepted_callback_cancellation_keeps_the_accepted_state(self):
        seen = []

        def stop_at_three(state):
            seen.append(state['iteration'])
            return state['iteration'] == 3

        fit = solve_kernel(self.K, self.y, .13, backend='lbfgs', stopping='fixed',
                           max_iter=20, callback=stop_at_three)
        reference = solve_kernel(self.K, self.y, .13, backend='lbfgs',
                                 stopping='fixed', max_iter=3)
        self.assertEqual(seen, [0, 1, 2, 3])
        self.assertEqual(fit['n_iter'], 3)
        self.assertEqual(fit['returned_iteration'], 3)
        self.assertEqual(fit['termination_reason'], 'callback_stop')
        assert_allclose(fit['alpha'], reference['alpha'], rtol=0, atol=0)
        self.assertEqual(fit['offset'], reference['offset'])
        self.assertAlmostEqual(fit['final_objective'],
            independent_objective(self.K, self.y, fit['alpha'], fit['offset'], .13, 1.), places=13)


def _has_cvxpy():
    try:
        return importlib.util.find_spec('cvxpy') is not None
    except ImportError:
        return False


@unittest.skipUnless(_has_cvxpy(), 'Optional CVXPY oracle is not installed')
class ConvexKernelOracleTests(unittest.TestCase):
    def test_generalized_backends_match_independent_primal_and_dual_programs(self):
        import cvxpy as cp
        if 'CLARABEL' not in cp.installed_solvers():
            self.skipTest('Independent oracle requires optional CLARABEL')
        rng = np.random.RandomState(773)
        Phi = rng.normal(size=(8, 3))
        K = Phi @ Phi.T
        y = np.r_[np.ones(5), -np.ones(3)]
        penalty, n = .3, len(y)
        options = dict(solver='CLARABEL', tol_gap_abs=1e-10,
                       tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=250)
        for q in (.5, 1., 2.):
            with self.subTest(q=q):
                w, b, d, xi = cp.Variable(3), cp.Variable(), cp.Variable(n), cp.Variable(n, nonneg=True)
                coefficient = q ** q / (q + 1) ** (q + 1)
                primal = cp.Problem(cp.Minimize(cp.sum(coefficient * cp.power(d, -q) + xi) / n
                                    + penalty * cp.sum_squares(w)),
                                    [d == cp.multiply(y, Phi @ w + b) + xi])
                optimum = primal.solve(**options)
                self.assertIn(primal.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
                rho = cp.Variable(n)
                signed = cp.multiply(y, rho)
                dual = cp.Problem(cp.Maximize(cp.sum(cp.power(rho, q / (q + 1))) / n
                                  - cp.sum_squares(Phi.T @ signed) / (4 * penalty * n * n)),
                                  [rho >= 0, rho <= 1, y @ rho == 0])
                dual_optimum = dual.solve(**options)
                self.assertIn(dual.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
                self.assertAlmostEqual(optimum, dual_optimum, delta=8e-8)
                for backend in ('spectral', 'cholesky', 'lbfgs'):
                    with self.subTest(backend=backend):
                        fit = solve_kernel(K, y, penalty, q=q, backend=backend,
                            stopping='optimality', tol=2e-8, max_iter=6000)
                        self.assertAlmostEqual(fit['final_objective'], optimum, delta=2e-7)
                        self.assertLess(fit['dual_gap'], 3e-7)
                        self.assertLessEqual(fit['dual_objective'], dual_optimum + 1e-7)
                        self.assertLess(fit['dual_equality_residual'], 1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
