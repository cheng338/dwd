"""Independent generalized-DWD checks on a rank-deficient RBF problem."""
from decimal import Decimal, localcontext
import importlib.util
import unittest

import numpy as np

from dwd.gen_kern_dwd import KernGDWD


def independent_rbf(X, gamma):
    with localcontext() as ctx:
        ctx.prec = 80
        g = Decimal.from_float(gamma)
        return np.array([[float((-g * sum((Decimal.from_float(float(a)) -
                             Decimal.from_float(float(b))) ** 2 for a, b in zip(x, z))).exp())
                          for z in X] for x in X])


def independent_objective(K, y, alpha, intercept, penalty, q):
    margins = y * (K @ alpha + intercept)
    threshold = q / (q + 1)
    loss = [1 - u if u <= threshold else q ** q / (q + 1) ** (q + 1) / u ** q
            for u in margins]
    return float(np.mean(loss) + penalty * (alpha @ K @ alpha))


class RBFObjectiveTests(unittest.TestCase):
    def setUp(self):
        # Duplicated rows with conflicting labels exercise a singular Gram
        # matrix and a free, nonzero intercept under class imbalance.
        self.X = np.array([[-2.], [-1.], [-1.], [-.3], [.3], [1.], [1.], [2.], [2.]])
        self.y = np.array([-1, -1, 1, -1, -1, 1, 1, -1, -1])
        self.K = independent_rbf(self.X, .7)

    def model(self, **kw):
        return KernGDWD(kernel='rbf', kernel_kws={'gamma': .7}, lambd=.04, **kw)

    def test_fixed_budget_reports_the_returned_objective_without_claiming_convergence(self):
        model = self.model(stopping='fixed', max_iter=3).fit(self.X, self.y)
        self.assertEqual(model.n_iter_, 3)
        self.assertFalse(model.converged_)
        value = independent_objective(self.K, self.y, model.dual_coef_[0],
                                      model.intercept_[0], .04, 1.)
        self.assertAlmostEqual(model.final_objective_, value, delta=1e-12)

    @unittest.skipUnless(importlib.util.find_spec('cvxpy'), 'Optional CVXPY oracle is not installed')
    def test_generalized_backends_match_independent_rbf_primal_and_dual(self):
        import cvxpy as cp
        if 'CLARABEL' not in cp.installed_solvers():
            self.skipTest('Independent oracle requires optional CLARABEL')
        n, K, y, penalty = len(self.y), self.K, self.y, .04
        self.assertLess(np.linalg.matrix_rank(K), n)
        options = dict(solver='CLARABEL', tol_gap_abs=1e-10,
                       tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=250)
        for q in (.5, 1., 2.):
            alpha, b, r = cp.Variable(n), cp.Variable(), cp.Variable(n)
            slack = cp.Variable(n, nonneg=True)
            coefficient = q ** q / (q + 1) ** (q + 1)
            primal = cp.Problem(cp.Minimize(cp.sum(coefficient * cp.power(r, -q) + slack) / n
                                + penalty * cp.quad_form(alpha, cp.psd_wrap(K))),
                                [r == cp.multiply(y, K @ alpha + b) + slack])
            optimum = primal.solve(**options)
            rho = cp.Variable(n)
            signed = cp.multiply(y, rho)
            dual = cp.Problem(cp.Maximize(cp.sum(cp.power(rho, q / (q + 1))) / n
                              - cp.quad_form(signed, cp.psd_wrap(K)) / (4 * penalty * n * n)),
                              [rho >= 0, rho <= 1, y @ rho == 0])
            dual_optimum = dual.solve(**options)
            self.assertIn(primal.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
            self.assertIn(dual.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
            self.assertAlmostEqual(optimum, dual_optimum, delta=8e-8)
            for backend in ('cholesky', 'spectral', 'lbfgs'):
                with self.subTest(q=q, backend=backend):
                    model = self.model(q=q, backend=backend, stopping='optimality',
                                       tol=2e-8, max_iter=6000).fit(self.X, self.y)
                    value = independent_objective(K, y, model.dual_coef_[0],
                                                  model.intercept_[0], penalty, q)
                    self.assertTrue(model.converged_)
                    self.assertAlmostEqual(value, optimum, delta=4e-7)
                    self.assertAlmostEqual(value, model.final_objective_, delta=1e-12)


if __name__ == '__main__':
    unittest.main()
