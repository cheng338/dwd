"""Independent numerical and API regression checks for linear generalized DWD."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.base import clone, is_classifier

from dwd.gen_dwd import (GenDWD, GenDWDCV, V, V_grad, V_, V_grad_, solve_gen_dwd,
                         get_P0_eig, get_step_implicit)


class LinearDWDAuditTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(15)
        self.X = self.rng.normal(size=(35, 4)) + np.array([2., -1., .5, 3.])
        self.y = np.where(self.X[:, 0] > 2., 1, -1)
        self.beta = self.rng.normal(size=4)
        self.eig = get_P0_eig(self.X)

    def test_legacy_trajectory_preserved(self):
        path = Path(__file__).resolve().parent / '_legacy' / 'gen_dwd.py'
        spec = importlib.util.spec_from_file_location('compat_gen_dwd', path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for implicit in (True, False):
            args = dict(X=self.X, y=self.y, lambd=.03, q=2.,
                        beta_init=self.beta, offset_init=.1, obj_tol=0,
                        max_iter=40, implicit_P=implicit, P0_eig=self.eig)
            before, after = old.solve_gen_dwd(**args), solve_gen_dwd(**args)
            for a, b in zip(before, after):
                assert_allclose(a, b, rtol=2e-11, atol=1e-12)

    def test_corrected_inverse_step_matches_direct_block_solve(self):
        n, p = self.X.shape
        q, penalty = 2., .03
        M = (q + 1) ** 2 / q
        shift = 2 * n * penalty / M
        P = np.block([[np.array([[n]]), self.X.sum(axis=0)[None, :]],
                      [self.X.sum(axis=0)[:, None], self.X.T @ self.X + shift * np.eye(p)]])
        z = self.y * np.array([V_grad_(u, q=q) for u in
                               self.y * (self.X @ self.beta + .1)]) / n
        gamma = np.r_[z.sum(), self.X.T @ z + 2 * penalty * self.beta]
        expected = (n / M) * np.linalg.solve(P, gamma)
        U, D = self.eig
        pi = D + shift
        v = (U / pi) @ U[0]
        g = shift / (1 - shift * v[0])
        got_beta, got_offset = get_step_implicit(self.X, self.y, self.beta,
                                                 .1, penalty, q, U, v, g, pi)
        assert_allclose(np.r_[got_offset, got_beta], expected, rtol=1e-11, atol=1e-12)

    def test_corrected_implicit_trajectory_matches_explicit(self):
        args = dict(X=self.X, y=self.y, lambd=.03, q=2.,
                    beta_init=self.beta, offset_init=.1, obj_tol=0,
                    max_iter=80, solver_mode='schur')
        implicit = solve_gen_dwd(**args, implicit_P=True)
        explicit = solve_gen_dwd(**args, implicit_P=False)
        for a, b in zip(implicit, explicit):
            assert_allclose(a, b, rtol=1e-10, atol=1e-11)
        self.assertLessEqual(np.max(np.diff(implicit[2])), 1e-12)

    def test_estimator_explicit_selection_rng_tags_and_diagnostics(self):
        model = GenDWD(lambd=.03, q=2., implicit_P=False, random_state=12,
                       max_iter=8, obj_tol=0, solver_mode='schur')
        a = clone(model).fit(self.X, self.y)
        b = clone(model).fit(self.X, self.y)
        expected = solve_gen_dwd(self.X, self.y, lambd=.03, q=2.,
                                 implicit_P=False, random_state=12, max_iter=8,
                                 obj_tol=0, solver_mode='schur')
        assert_allclose(a.coef_.ravel(), expected[0], rtol=1e-13, atol=1e-13)
        assert_array_equal(a.coef_, b.coef_)
        self.assertTrue(is_classifier(a))
        self.assertTrue(is_classifier(GenDWDCV()))
        self.assertEqual(a.n_iter_, 8)
        self.assertEqual(a.termination_reason_, 'max_iter')
        self.assertEqual(len(a.objective_history_), 9)
        self.assertTrue(np.isfinite(a.gradient_inf_norm_))
        a.set_params(max_iter=0).fit(self.X, self.y)
        self.assertEqual(a.n_iter_, 0)
        self.assertFalse(a.converged_)

    def test_cache_valid_only_for_same_data_and_sparse_parity(self):
        model = GenDWD(random_state=2, max_iter=12, solver_mode='schur')
        with patch('dwd.gen_dwd.get_P0_eig', wraps=get_P0_eig) as eig:
            model.cv_init(self.X)
            model.fit(self.X, self.y)
            self.assertEqual(eig.call_count, 1)
            changed = self.X.copy()
            changed[0, 0] += .5
            model.fit(changed, self.y)
            self.assertEqual(eig.call_count, 2)
        sparse = clone(model).fit(csr_matrix(self.X), self.y)
        dense = clone(model).fit(self.X, self.y)
        assert_allclose(sparse.coef_, dense.coef_, rtol=1e-10, atol=1e-11)
        assert_allclose(sparse.objective_history_, dense.objective_history_, rtol=1e-10)
        sparse.cv_init(csr_matrix(self.X))
        self.assertTrue(sparse._cv_cache_matches(csr_matrix(self.X)))
        self.assertFalse(sparse._cv_cache_matches(self.X))

    def test_validation_rejects_unsupported_or_undefined_inputs(self):
        for setting in ({'q': 0}, {'q': np.nan}, {'lambd': np.inf},
                        {'obj_tol': -1}, {'max_iter': -1}, {'solver_mode': 'typo'}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                GenDWD(**setting).fit(self.X, self.y)
        with self.assertRaises(NotImplementedError):
            GenDWDCV().fit(self.X, self.y, sample_weight=np.ones(len(self.y)))
        with self.assertRaises(ValueError):
            GenDWD().fit(self.X, np.arange(len(self.y)))
        with self.assertRaises(ValueError):
            GenDWD().fit(self.X, self.y, P0_eig=(np.eye(2), np.ones(2)))
        with self.assertRaises(ValueError):
            GenDWD().fit(self.X, self.y, beta_init=np.zeros(2))

    def test_loss_float32_promotes_before_arithmetic_and_nan_propagates(self):
        u = np.array([-.2, .8, 1.2, 2.5], dtype=np.float32)
        assert_allclose(V(u, q=1.7), [V_(float(v), q=1.7) for v in u], rtol=1e-14)
        assert_allclose(V_grad(u, q=1.7), [V_grad_(float(v), q=1.7) for v in u], rtol=1e-14)
        self.assertTrue(np.isnan(V(np.nan)))
        self.assertTrue(np.isnan(V_grad(np.nan)))

    def test_linear_cv_uses_cartesian_grid_and_refits_selected_solver(self):
        model = GenDWDCV(lambd_vals=[.02, .1], q_vals=[1., 2.],
                         cv=3, max_iter=5, random_state=9,
                         solver_mode='schur', implicit_P=False).fit(self.X, self.y)
        self.assertEqual(len(model.agg_cv_results_['params']), 4)
        self.assertEqual(model.best_estimator_.max_iter, 5)
        self.assertFalse(model.best_estimator_.implicit_P)
        self.assertEqual(model.best_estimator_.solver_mode, 'schur')
        self.assertEqual(model.predict(self.X).shape, self.y.shape)


if __name__ == '__main__':
    unittest.main(verbosity=2)
