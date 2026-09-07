"""Release contracts for corrected linear DWD and public loss helpers."""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize
from sklearn.base import clone

from dwd.gen_dwd import (GenDWD, GenDWDCV, V, V_, V_grad, V_grad_,
                         solve_gen_dwd, c_from_lambd)


class LinearReleaseTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.RandomState(71).normal(size=(24, 3)) + [1., -.5, 2.]
        self.y = np.tile([-1, 1], 12)

    def test_corrected_defaults_are_deterministic_and_keep_stop_defaults(self):
        model = GenDWD()
        self.assertEqual((model.solver_mode, model.initialization), ('schur', 'auto'))
        self.assertEqual((model.stopping, model.obj_tol, model.max_iter),
                         ('objective', 1e-5, 100))
        first = clone(model).fit(self.X, self.y)
        second = clone(model).fit(self.X, self.y)
        assert_array_equal(first.coef_, second.coef_)
        assert_array_equal(first.objective_history_, second.objective_history_)
        self.assertEqual(first.initialization_, 'zero')
        self.assertEqual(first.objective_history_[0], 1.)

    def test_objective_stop_does_not_claim_stationarity(self):
        fitted = GenDWD(obj_tol=10., tol=1e-12).fit(self.X, self.y)
        self.assertEqual(fitted.n_iter_, 1)
        self.assertEqual(fitted.termination_reason_, 'objective_tolerance')
        self.assertTrue(fitted.objective_tolerance_met_)
        self.assertTrue(fitted.stationarity_checked_)
        self.assertFalse(fitted.converged_)
        self.assertGreater(fitted.stationarity_residual_, fitted.tol)

    def test_fixed_and_initial_optimality_stops(self):
        fixed = GenDWD(stopping='fixed', max_iter=8, obj_tol=10.).fit(self.X, self.y)
        self.assertEqual(fixed.n_iter_, 8)
        self.assertEqual(fixed.termination_reason_, 'max_iter')
        initial = GenDWD(stopping='optimality').fit(np.zeros_like(self.X), self.y)
        self.assertEqual(initial.n_iter_, 0)
        self.assertEqual(initial.termination_reason_, 'optimality')
        self.assertTrue(initial.converged_)
        self.assertEqual(initial.C_, 0.)

    def test_explicit_initialization_and_legacy_random_choice(self):
        beta = np.array([.2, -.4, .8])
        explicit = GenDWD(max_iter=0).fit(self.X, self.y, beta_init=beta, offset_init=.3)
        assert_array_equal(explicit.coef_[0], beta)
        self.assertEqual(explicit.intercept_[0], .3)
        self.assertEqual(explicit.initialization_, 'explicit')
        old = GenDWD(solver_mode='legacy', random_state=9, max_iter=0).fit(self.X, self.y)
        assert_array_equal(old.coef_[0], np.random.RandomState(9).normal(size=3))
        self.assertEqual(old.initialization_, 'random')
        low = solve_gen_dwd(self.X, self.y, lambd=1., max_iter=0)
        assert_array_equal(low[0], np.zeros(3))

    def test_full_objective_agrees_with_independent_bfgs(self):
        penalty = .1
        n = len(self.y)

        # q=1 loss/derivative stated independently of package loss functions.
        def objective_gradient(theta):
            b, beta = theta[0], theta[1:]
            margin = self.y * (self.X @ beta + b)
            loss = 1. - margin
            deriv = np.full(n, -1.)
            tail = margin > .5
            loss[tail] = 1. / (4. * margin[tail])
            deriv[tail] = -1. / (4. * margin[tail] ** 2)
            z = self.y * deriv / n
            return (loss.mean() + penalty * beta @ beta,
                    np.r_[z.sum(), self.X.T @ z + 2 * penalty * beta])

        oracle = minimize(objective_gradient, np.zeros(4), jac=True, method='BFGS',
                          options={'gtol': 1e-9, 'maxiter': 3000})
        fitted = GenDWD(lambd=penalty, stopping='optimality', tol=1e-7,
                        max_iter=20000).fit(self.X, self.y)
        self.assertTrue(fitted.converged_)
        self.assertEqual(fitted.termination_reason_, 'optimality')
        self.assertLess(abs(fitted.final_objective_ - oracle.fun), 1e-9)
        theta = np.r_[fitted.intercept_[0], fitted.coef_[0]]
        _, gradient = objective_gradient(theta)
        assert_allclose(fitted.gradient_inf_norm_, np.max(np.abs(gradient)), atol=1e-14)
        assert_allclose(fitted.stationarity_residual_,
                        max(abs(gradient[0]), np.linalg.norm(gradient[1:])), atol=1e-14)

    def test_cv_forwards_initialization_and_stopping(self):
        fitted = GenDWDCV(lambd_vals=[.1], q_vals=[.5, 2.], cv=2,
                          initialization='zero', stopping='fixed', max_iter=4,
                          tol=2e-6, implicit_P=False).fit(self.X, self.y)
        best = fitted.best_estimator_
        self.assertEqual(best.solver_mode, 'schur')
        self.assertEqual((best.initialization, best.stopping, best.tol),
                         ('zero', 'fixed', 2e-6))
        self.assertEqual(best.n_iter_, 4)

    def test_loss_q_domain_checks_cover_scalar_and_array_helpers(self):
        for function in (V, V_, V_grad, V_grad_):
            for q in (0, -1, np.nan, np.inf, '1', True, np.array([1., 2.])):
                with self.subTest(function=function.__name__, q=q):
                    with self.assertRaises(ValueError):
                        function(.7, q=q)

    def test_large_q_loss_avoids_power_overflow_and_keeps_domains(self):
        for q in (.5, 1., 2.3, 1000.):
            threshold = q / (q + 1.)
            margins = np.array([-2., 0., threshold, 1.01, 2., np.inf])
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                loss, gradient = V(margins, q), V_grad(margins, q)
            self.assertTrue(np.isfinite(loss).all())
            self.assertTrue(np.isfinite(gradient).all())
            assert_allclose(loss, [V_(u, q) for u in margins], rtol=1e-14)
            assert_allclose(gradient, [V_grad_(u, q) for u in margins], rtol=1e-14)
            self.assertAlmostEqual(float(V(threshold, q)), 1. / (q + 1.), places=14)
        self.assertTrue(np.isnan(V(np.nan)))
        self.assertTrue(np.isnan(V_grad(np.nan)))
        assert_allclose(V([0, 2]), [1., .125])
        assert_allclose(V_grad([0, 2]), [-1., -.0625])

    def test_unsupported_weights_and_invalid_new_parameters_raise(self):
        for options in ({'initialization': 'warm'}, {'stopping': 'validation'},
                        {'tol': -1.}, {'tol': np.nan}, {'max_iter': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                GenDWD(**options).fit(self.X, self.y)
        with self.assertRaises(NotImplementedError):
            GenDWD().fit(self.X, self.y, sample_weight=np.ones(len(self.y)))

    def test_large_q_c_conversion_does_not_discard_a_finite_fit(self):
        q = 1000.
        expected_unit_norm = (q + 1.) * (1. + 1. / q) ** q
        assert_allclose(c_from_lambd(.1, q, np.ones(1)), expected_unit_norm,
                        rtol=2e-13)
        self.assertEqual(c_from_lambd(.1, q, np.zeros(1)), 0.)
        self.assertTrue(np.isinf(c_from_lambd(.1, q, np.array([3.]))))
        fitted = GenDWD(q=q, lambd=1e-6, stopping='fixed', max_iter=400).fit(
            np.array([[-.1], [.1]]), np.array([-1, 1]))
        self.assertTrue(np.isfinite(fitted.coef_).all())
        self.assertTrue(np.isfinite(fitted.intercept_).all())
        self.assertTrue(np.isfinite(fitted.final_objective_))
        self.assertTrue(np.isfinite(fitted.stationarity_residual_))
        self.assertTrue(np.isinf(fitted.C_))
        self.assertFalse(fitted.C_conversion_finite_)
        self.assertEqual(fitted.n_iter_, 400)


if __name__ == '__main__':
    unittest.main(verbosity=2)
