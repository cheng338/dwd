"""Numerical checks for optimizations and the explicitly selected solver repair."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from dwd.gen_dwd import V, V_grad, V_, V_grad_
from dwd.gen_kern_dwd import (KernGDWD, get_K_eig, solve_gen_kern_dwd,
                               get_step_implicit_P)
from dwd.cv import run_cv, DoL2LoD


def legacy_module(filename):
    source = Path(__file__).resolve().parent / '_legacy' / filename
    spec = importlib.util.spec_from_file_location('legacy_' + source.stem, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DWDPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(25)
        self.X = self.rng.normal(size=(34, 9))
        distance = ((self.X[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=2)
        self.K = np.exp(-0.15 * distance) + 0.03 * np.eye(len(self.X))
        self.y = np.where(self.X[:, 0] > 0, 1, -1)
        self.alpha = self.rng.normal(size=len(self.y)) / len(self.y)
        self.eig = get_K_eig(self.K)

    def test_array_loss_and_gradient_equal_scalar_formulas(self):
        for q in (0.5, 1.0, 2.3):
            u = np.array([-2., 0., q / (q + 1), 0.7, 1., 3.])
            assert_allclose(V(u, q=q), [V_(v, q=q) for v in u], rtol=1e-14)
            assert_allclose(V_grad(u, q=q), [V_grad_(v, q=q) for v in u], rtol=1e-14)
            self.assertAlmostEqual(float(V(2., q=q)), V_(2., q=q))
            self.assertAlmostEqual(float(V_grad(2., q=q)), V_grad_(2., q=q))
        assert_allclose(V(np.array([0, 2])), [1., .125])

    def test_legacy_full_objective_trajectory_matches_compat2(self):
        old = legacy_module('gen_kern_dwd.py')
        old_loss = legacy_module('gen_dwd.py')
        old.V, old.V_grad = old_loss.V, old_loss.V_grad
        for q, penalty in ((1., .1), (.5, .02), (2.3, 1.)):
            args = dict(K=self.K, y=self.y, q=q, lambd=penalty,
                        alpha_init=self.alpha, offset_init=.04,
                        obj_tol=0., max_iter=45, K_eig=self.eig)
            before = old.solve_gen_kern_dwd(**args)
            after = solve_gen_kern_dwd(**args, solver_mode='legacy')
            for a, b in zip(before, after):
                assert_allclose(a, b, rtol=1e-11, atol=1e-12)
            assert_array_equal(self.K @ before[0] + before[1] > 0,
                               self.K @ after[0] + after[1] > 0)

    def test_schur_step_agrees_with_independent_augmented_solve(self):
        n, q, penalty = len(self.y), 1.3, .07
        M = (q + 1) ** 2 / q
        shift = 2 * n * penalty / M
        ones = np.ones(n)
        U, lam = self.eig
        h_inv = np.linalg.solve(self.K + shift * np.eye(n), np.eye(n))
        v = h_inv @ ones
        g = 1. / (n - ones @ self.K @ v)
        got_a, got_b = get_step_implicit_P(
            self.K, self.y, q, penalty, self.alpha, .04, U, lam,
            lam * (lam + shift), v, g, solver_mode='schur')
        z = self.y * np.array([V_grad_(u, q=q) for u in
                               self.y * (self.K @ self.alpha + .04)]) / n
        gamma = np.r_[z.sum(), self.K @ z + 2 * penalty * self.K @ self.alpha]
        P = np.block([[np.array([[n]]), (ones @ self.K)[None, :]],
                      [(self.K @ ones)[:, None], self.K @ self.K + shift * self.K]])
        expected = (n / M) * np.linalg.solve(P, gamma)
        assert_allclose(np.r_[got_b, got_a], expected, rtol=1e-11, atol=1e-12)

    def test_precomputed_fit_reproducibility_diagnostics_and_cache(self):
        model = KernGDWD(kernel='rbf', kernel_kws={'gamma': .15},
                         random_state=19, max_iter=120, obj_tol=1e-8,
                         solver_mode='schur')
        # The explicit K in this test intentionally adds a diagonal ridge.
        # Scores use K directly so training-kernel semantics stay identical.
        a = clone(model).fit(self.X, self.y, K=self.K, K_eig=self.eig)
        b = clone(model).fit(self.X, self.y, K=self.K, K_eig=self.eig)
        assert_array_equal(a.dual_coef_, b.dual_coef_)
        self.assertEqual(a.n_iter_, len(a.obj_vals_) - 1)
        self.assertLessEqual(a.n_iter_, 120)
        self.assertTrue(np.isfinite(a.gradient_inf_norm_))
        self.assertLessEqual(np.max(np.diff(a.objective_history_)), 1e-12)
        model.cv_init(self.X)
        self.assertTrue(model._cv_cache_matches(self.X.copy()))
        model.set_params(kernel_kws={'gamma': .4})
        self.assertFalse(model._cv_cache_matches(self.X))
        model.set_params(kernel_kws={'gamma': .15})
        changed = self.X.copy()
        changed[0, 0] += .1
        self.assertFalse(model._cv_cache_matches(changed))

    def test_kernel_cache_avoids_eigendecomposition_and_rejects_bad_shapes(self):
        model = KernGDWD(kernel='rbf', kernel_kws={'gamma': .15}, random_state=2)
        with patch('dwd.gen_kern_dwd.get_K_eig', wraps=get_K_eig) as eig:
            model.cv_init(self.X)
            model.fit(self.X, self.y)
            model.set_params(lambd=.2).fit(self.X, self.y)
            self.assertEqual(eig.call_count, 0)
            model.set_params(kernel_kws={'gamma': .6}).fit(self.X, self.y)
            # Auto now uses Cholesky after invalidating the old gamma cache.
            # A fresh fit must agree; requiring a new eigendecomposition here
            # would force the implementation to discard the faster backend.
            self.assertEqual(eig.call_count, 0)
            self.assertEqual(model.backend_, 'cholesky')
            fresh = clone(model).fit(self.X, self.y)
            assert_allclose(model.decision_function(self.X),
                            fresh.decision_function(self.X), rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError):
            model.fit(self.X, self.y, K=np.eye(2))

    def test_cartesian_grid_folds_and_best_refit(self):
        self.assertEqual(len(DoL2LoD({'q': [.5, 1.], 'lambd': [.1, .2, .3]})), 6)
        clf = KernGDWD(kernel='rbf', random_state=3, max_iter=8)
        params = {'lambd': [.02, .1], 'kernel_kws': [{'gamma': .1}, {'gamma': .4}]}
        cv = StratifiedKFold(3)
        best, score, fitted, aggregate, folds = run_cv(clf, self.X, self.y, params, cv=cv)
        for key, value in best.items():
            self.assertEqual(fitted.get_params()[key], value)
        self.assertEqual(len(fitted._Xfit), len(self.X))
        for fold_index, (train, test) in enumerate(cv.split(self.X, self.y)):
            for index, setting in enumerate(aggregate['params']):
                reference = clone(clf).set_params(**setting).fit(self.X[train], self.y[train])
                expected = accuracy_score(self.y[test], reference.predict(self.X[test]))
                self.assertEqual(folds[fold_index]['test_score'][index], expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
