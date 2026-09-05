"""Small independent regressions for the dependency's remaining public modules."""
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV

from dwd.kern_md import KernMD, kern_md
from dwd.kernel_utils import KernelScaler
from dwd.gen_kern_dwd import KernGDWD
from dwd.cv import run_cv

try:
    import cvxpy as cp
except ImportError:
    cp = None
else:
    from dwd.socp_dwd import DWD, auto_dwd_C, solve_dwd_socp
    from dwd.svm import SVM, solve_svm

try:
    import matplotlib
except ImportError:
    HAS_MATPLOTLIB = False
else:
    matplotlib.use('Agg')
    from dwd.viz import clf2D_slope_intercept
    from dwd.sim_fun import tuning_curve
    HAS_MATPLOTLIB = True


class RemainingAuditTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2., .2], [-1., -.1], [1., .3], [2., -.2]])
        self.y = np.array([-1, -1, 1, 1])

    def test_mean_difference_matches_feature_space_classifier(self):
        fitted = KernMD().fit(self.X, self.y)
        pos, neg = self.X[self.y > 0].mean(0), self.X[self.y < 0].mean(0)
        expected = self.X @ (pos - neg) - .5 * (pos @ pos - neg @ neg)
        np.testing.assert_allclose(fitted.decision_function(self.X), expected)
        np.testing.assert_array_equal(KernMD().fit([[0.], [2.]], [-1, 1]).predict([[0.], [2.]]), [-1, 1])
        self.assertTrue(is_classifier(clone(fitted)))
        with self.assertRaises(NotImplementedError):
            KernMD(naive_bayes=True).fit(self.X, self.y)

    def test_kernel_query_orientation_and_callable(self):
        query = self.X[:3] + .12
        fitted = KernMD(kernel='rbf', kernel_kws={'gamma': .3}).fit(self.X, self.y)
        precomputed = KernMD(kernel='precomputed').fit(rbf_kernel(self.X, gamma=.3), self.y)
        callback = KernMD(kernel=rbf_kernel, kernel_kws={'gamma': .3}).fit(self.X, self.y)
        np.testing.assert_allclose(precomputed.decision_function(rbf_kernel(query, self.X, gamma=.3)), fitted.decision_function(query))
        np.testing.assert_allclose(callback.decision_function(query), fitted.decision_function(query))
        with self.assertRaises(ValueError):
            precomputed.decision_function(np.zeros((3, 3)))

    def test_scaler_preserves_legacy_formula_and_rejects_invalid_diagonal(self):
        K = self.X @ self.X.T + np.eye(4)
        actual = KernelScaler().fit_transform(K)
        scales = np.sqrt(len(K) / np.diag(K))
        np.testing.assert_allclose(actual, scales[:, None] * K * scales)
        with self.assertRaises(ValueError):
            KernelScaler().fit(np.zeros((2, 2)))

    def test_precomputed_cross_validation_slices_both_axes(self):
        X = np.random.RandomState(23).normal(size=(12, 3))
        y = np.tile([-1, 1], 6)
        K = rbf_kernel(X, gamma=.3)
        options = dict(random_state=91, max_iter=5, obj_tol=0)
        named = KernGDWD(kernel='rbf', kernel_kws={'gamma': .3}, **options)
        precomputed = KernGDWD(kernel='precomputed', **options)
        expected = run_cv(named, X, y, {'lambd': [.1, .2]}, cv=3)
        actual = run_cv(precomputed, K, y, {'lambd': [.1, .2]}, cv=3)
        np.testing.assert_allclose(actual[3]['mean_test_score'], expected[3]['mean_test_score'])
        np.testing.assert_allclose(actual[2].decision_function(K), expected[2].decision_function(X), atol=1e-10)
        fitted = GridSearchCV(precomputed, {'lambd': [.1]}, cv=2, error_score='raise').fit(K, y)
        self.assertEqual(fitted.best_estimator_._Xfit.shape, K.shape)

    @unittest.skipIf(cp is None, 'cvxpy is an optional dependency; install dwd[test].')
    def test_socp_matches_independent_inverse_margin_formulation(self):
        fitted = DWD(C=2., solver_kws={'solver': 'CLARABEL'}).fit(self.X, self.y)
        w, b, eta = cp.Variable(2), cp.Variable(), cp.Variable(4, nonneg=True)
        margin = cp.multiply(self.y, self.X @ w + b) + eta
        problem = cp.Problem(cp.Minimize(cp.sum(cp.inv_pos(margin)) + 2 * cp.sum(eta)), [cp.norm(w) <= 1])
        problem.solve(solver='CLARABEL')
        self.assertAlmostEqual(fitted.problem_.value, problem.value, places=5)
        np.testing.assert_allclose(fitted.coef_[0], w.value, atol=2e-4)
        direction, intercept = fitted.direction
        np.testing.assert_array_equal(direction, fitted.coef_[0])
        self.assertEqual(intercept, -fitted.intercept_.item())
        auto = DWD(C='auto').fit(self.X, self.y)
        self.assertEqual(auto.C, 'auto')
        self.assertEqual(auto.C_, auto_dwd_C(self.X, self.y))
        self.assertEqual(clone(auto).C, 'auto')

    @unittest.skipIf(cp is None, 'cvxpy is an optional dependency; install dwd[test].')
    def test_l1_svm_binary_labels_and_parameter_checks(self):
        signed = SVM(C=.1).fit(self.X, self.y)
        zero_one = SVM(C=.1).fit(self.X, self.y > 0)
        np.testing.assert_allclose(signed.decision_function(self.X), zero_one.decision_function(self.X), atol=1e-8)
        np.testing.assert_array_equal(zero_one.predict(self.X), self.y > 0)
        self.assertTrue(is_classifier(signed))
        for solver in (solve_svm, solve_dwd_socp):
            for C in (0, -1, np.nan, np.inf):
                with self.assertRaises(ValueError):
                    solver(self.X, self.y, C=C)

    @unittest.skipIf(cp is None, 'cvxpy is an optional dependency; install dwd[test].')
    def test_solver_failure_is_explicit(self):
        def failed(problem, **kwargs):
            problem._status = cp.INFEASIBLE
        with patch.object(cp.Problem, 'solve', failed):
            for solver in (solve_svm, solve_dwd_socp):
                with self.assertRaisesRegex(RuntimeError, 'infeasible'):
                    solver(self.X, self.y, C=1)

    @unittest.skipUnless(HAS_MATPLOTLIB, 'matplotlib is optional; install dwd[test].')
    def test_visualization_and_tuning_adapters(self):
        slope, intercept = clf2D_slope_intercept(coef=np.array([1., 2.]), intercept=np.array([3.]))
        self.assertEqual((slope, intercept), (-.5, -1.5))
        with self.assertRaises(ValueError):
            clf2D_slope_intercept(coef=[1., 0.], intercept=1.)
        seen = []
        def solve(X, y, C, sample_weight=None, solver_kws=None):
            self.assertIsNone(sample_weight)
            seen.append(solver_kws)
            return np.array([1., 0.]), 0., None, None, None
        options = {'solver': 'CLARABEL'}
        train, test = tuning_curve(self.X, self.y, self.X, self.y, [1., 2.], solve, options)
        self.assertEqual(seen, [options, options])
        np.testing.assert_array_equal(train, [0., 0.])
        np.testing.assert_array_equal(test, train)


if __name__ == '__main__':
    unittest.main(verbosity=2)
