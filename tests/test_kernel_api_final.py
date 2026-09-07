"""Regression checks for the final kernel estimator and CV API."""
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from dwd.cv import run_cv
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV, solve_gen_kern_dwd, c_from_lambd
from dwd.utils import parameters_equal


def ard_kernel(A, B, lengthscale):
    return rbf_kernel(A / lengthscale, B / lengthscale, gamma=.2)


class FinalKernelAPITests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(17)
        self.X = rng.normal(size=(24, 4))
        self.query = rng.normal(size=(7, 4))
        self.y = np.tile([-7, 9], 12)

    def model(self, **kwargs):
        options = dict(kernel='rbf', kernel_kws={'gamma': .3}, lambd=.04,
                       max_iter=8, stopping='fixed')
        options.update(kwargs)
        return KernGDWD(**options)

    def test_defaults_retain_objective_rule_and_use_corrected_zero_initialization(self):
        defaults = KernGDWD()
        self.assertEqual(defaults.solver_mode, 'schur')
        self.assertEqual(defaults.stopping, 'objective')
        self.assertEqual(defaults.obj_tol, 1e-5)
        self.assertEqual(defaults.max_iter, 100)
        a = self.model(stopping='objective').fit(self.X, self.y)
        b = self.model(stopping='objective', initialization='zero', random_state=91).fit(self.X, self.y)
        assert_array_equal(a.dual_coef_, b.dual_coef_)
        assert_array_equal(a.intercept_, b.intercept_)
        self.assertEqual(a.objective_history_[0], 1.)
        self.assertEqual(a.n_iter_, len(a.objective_history_) - 1)
        zero_alpha, zero_offset, history, _ = solve_gen_kern_dwd(
            rbf_kernel(self.X, gamma=.3), self.y, .04, max_iter=0)
        assert_array_equal(zero_alpha, np.zeros(24))
        self.assertEqual(zero_offset, 0.)
        self.assertEqual(history, [1.])

    def test_one_corrected_update_matches_independent_free_intercept_ridge_system(self):
        fitted = self.model(max_iter=1, backend='cholesky').fit(self.X, self.y)
        K = rbf_kernel(self.X, gamma=.3)
        n = len(self.y)
        ones = np.ones((n, 1))
        system = np.block([[K + (n * fitted.lambd / 2) * np.eye(n), ones],
                           [ones.T, np.zeros((1, 1))]])
        solution = np.linalg.solve(system, np.r_[np.where(self.y == 9, 1., -1.), 0.]) / 4
        assert_allclose(fitted.dual_coef_[0], solution[:-1], atol=2e-13, rtol=2e-13)
        assert_allclose(fitted.intercept_[0], solution[-1], atol=2e-13, rtol=2e-13)

    def test_explicit_legacy_keeps_random_initialization_and_trajectory(self):
        fitted = self.model(solver_mode='legacy', random_state=83).fit(self.X, self.y)
        K = rbf_kernel(self.X, gamma=.3)
        alpha, offset, history, _ = solve_gen_kern_dwd(
            K, self.y, fitted.lambd, q=1, max_iter=8, obj_tol=0.,
            random_state=83, solver_mode='legacy')
        assert_array_equal(fitted.dual_coef_[0], alpha)
        self.assertEqual(fitted.intercept_[0], offset)
        assert_array_equal(fitted.objective_history_, history)
        self.assertFalse(fitted.stationarity_checked_)

    def test_clone_and_grid_search_keep_notation_and_new_controls(self):
        base = self.model(initialization='zero', backend='spectral', prediction_batch_size=3)
        fitted = base.fit(self.X, self.y)
        copied = clone(fitted)
        self.assertTrue(is_classifier(copied))
        self.assertEqual(copied.get_params(), fitted.get_params())
        self.assertIsNot(copied.kernel_kws, fitted.kernel_kws)
        self.assertFalse(hasattr(copied, 'dual_coef_'))
        search = GridSearchCV(copied,
            {'lambd': [.02, .1], 'kernel_kws': [{'gamma': .2}, {'gamma': .5}]},
            cv=StratifiedKFold(3), error_score='raise').fit(self.X, self.y)
        self.assertEqual(len(search.cv_results_['params']), 4)
        for key, value in search.best_params_.items():
            self.assertEqual(search.best_estimator_.get_params()[key], value)
        self.assertEqual(search.predict(self.query).shape, (7,))

    def test_array_kernel_kwargs_cached_fit_and_cv(self):
        scale = np.array([1., 2., .5, 3.])
        model = self.model(kernel=ard_kernel, kernel_kws={'lengthscale': scale}, backend='spectral')
        fresh = clone(model).fit(self.X, self.y)
        model.cv_init(self.X).fit(self.X, self.y)
        assert_allclose(model.decision_function(self.query), fresh.decision_function(self.query), atol=1e-12)
        self.assertTrue(model._cv_cache_matches(self.X))
        scale[0] *= 1.5
        self.assertFalse(model._cv_cache_matches(self.X))
        model.fit(self.X, self.y)
        assert_allclose(model.decision_function(self.query), clone(model).fit(self.X, self.y).decision_function(self.query), atol=1e-12)
        best, _, fitted, aggregate, _ = run_cv(model, self.X, self.y,
            {'lambd': [.02, .1], 'kernel_kws': [{'lengthscale': scale.copy()}]}, cv=3)
        self.assertEqual(len(aggregate['params']), 2)
        self.assertEqual(fitted.lambd, best['lambd'])

    def test_nested_parameter_equality_and_precise_invalidation(self):
        a = {'nested': [{'x': np.array([1., 2.])}], 'pair': (1, 'a')}
        b = {'nested': [{'x': np.array([1., 2.])}], 'pair': (1, 'a')}
        self.assertTrue(parameters_equal(a, b))
        b['nested'][0]['x'] = b['nested'][0]['x'].astype(np.float32)
        self.assertFalse(parameters_equal(a, b))
        model = self.model().cv_init(self.X)
        self.assertTrue(model._cv_cache_matches(self.X.copy()))
        self.assertFalse(model._cv_cache_matches(self.X[::-1]))
        self.assertFalse(model._cv_cache_matches(csr_matrix(self.X)))
        self.assertFalse(model._cv_cache_matches(self.X.astype(np.float32)))
        model.set_params(backend='cholesky')
        self.assertFalse(model._cv_cache_matches(self.X))

    def test_explicit_path_reuses_eigenvalues_and_cholesky_does_not_require_them(self):
        model = self.model(backend='spectral').cv_init(self.X)
        with patch('dwd._eigen.eigh', side_effect=AssertionError('unexpected eigen computation')):
            model.fit(self.X, self.y)
            model.set_params(lambd=.1).fit(self.X, self.y)
        cholesky = self.model(backend='cholesky').cv_init(self.X)
        self.assertIsNone(cholesky._K_eig)
        with patch('dwd._eigen.eigh', side_effect=AssertionError('unexpected eigen computation')):
            cholesky.fit(self.X, self.y)

    def test_prediction_batches_bound_callable_query_size(self):
        seen = []
        def kernel(A, B):
            seen.append(len(B))
            return rbf_kernel(A, B, gamma=.3)
        fitted = self.model(kernel=kernel, kernel_kws=None).fit(self.X, self.y)
        expected = fitted.decision_function(self.query)
        seen.clear()
        fitted.set_params(prediction_batch_size=3)
        assert_allclose(fitted.decision_function(self.query), expected, atol=1e-14)
        self.assertEqual(seen, [3, 3, 1])
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                fitted.set_params(prediction_batch_size=bad).decision_function(self.query)

    def test_precomputed_orientation_batches_and_standard_cv(self):
        K = rbf_kernel(self.X, gamma=.3)
        query_K = rbf_kernel(self.query, self.X, gamma=.3)
        named = self.model(backend='spectral').fit(self.X, self.y)
        pre = self.model(kernel='precomputed', kernel_kws=None,
                         backend='spectral', prediction_batch_size=2).fit(K, self.y)
        assert_allclose(pre.decision_function(query_K), named.decision_function(self.query), atol=1e-12)
        search = GridSearchCV(clone(pre), {'lambd': [.02, .1]}, cv=3,
                              error_score='raise').fit(K, self.y)
        self.assertEqual(search.predict(query_K).shape, (7,))
        with self.assertRaises(ValueError):
            pre.decision_function(query_K.T)

    def test_validation_requires_explicit_data_and_accepts_one_training_class(self):
        model = self.model(stopping='validation', patience=2, check_interval=1)
        with self.assertRaisesRegex(ValueError, 'explicit validation_data'):
            model.fit(self.X, self.y)
        fitted = model.fit(self.X, self.y, validation_data=(self.query, np.full(7, -7)))
        self.assertGreater(len(fitted.validation_history_), 0)
        self.assertLessEqual(fitted.returned_iteration_, fitted.n_iter_)
        with self.assertRaisesRegex(ValueError, 'training classes'):
            clone(model).fit(self.X, self.y, validation_data=(self.query, np.full(7, 123)))
        with self.assertRaisesRegex(ValueError, 'used only'):
            self.model().fit(self.X, self.y, validation_data=(self.query, np.full(7, -7)))

    def test_cv_wrapper_forwards_controls_and_does_not_create_monitoring_split(self):
        model = KernGDWDCV(lambd_vals=[.02, .1], q_vals=[1], kernel='rbf',
                          kernel_kws_vals=[{'gamma': .3}], cv=3, max_iter=2,
                          stopping='fixed', initialization='zero', backend='cholesky',
                          prediction_batch_size=2).fit(self.X, self.y)
        self.assertEqual(model.best_estimator_.n_iter_, 2)
        self.assertEqual(model.best_estimator_.backend_, 'cholesky')
        assert_array_equal(model.predict(self.query), model.best_estimator_.predict(self.query))
        with self.assertRaisesRegex(ValueError, 'monitoring splits'):
            KernGDWDCV(stopping='validation').fit(self.X, self.y)

    def test_weights_unsupported_legacy_options_and_failed_refit(self):
        for model in (self.model(), KernGDWDCV()):
            with self.assertRaisesRegex(NotImplementedError, 'Sample weights'):
                model.fit(self.X, self.y, sample_weight=np.ones(24))
        with self.assertRaisesRegex(ValueError, 'Legacy mode'):
            self.model(solver_mode='legacy', backend='cholesky').fit(self.X, self.y)
        fitted = self.model().fit(self.X, self.y)
        with self.assertRaises((FloatingPointError, ValueError)):
            fitted.fit(self.X, self.y, alpha_init=np.full(24, np.inf))
        with self.assertRaises(NotFittedError):
            fitted.predict(self.query)

    def test_large_q_conversion_avoids_intermediate_power_overflow(self):
        value = c_from_lambd(np.eye(2), .1, 1000., np.array([1., 0.]))
        expected = np.exp(np.log1p(1000.) + 1000. * np.log1p(.001))
        self.assertEqual(value, expected)
        self.assertTrue(np.isinf(c_from_lambd(np.eye(2), .1, 1000., np.array([10., 0.]))))
        self.assertEqual(c_from_lambd(np.eye(2), .1, 1000., np.zeros(2)), 0.)
        fitted = self.model().fit(self.X, self.y)
        self.assertEqual(fitted.C_conversion_finite_, np.isfinite(fitted.C_))


if __name__ == '__main__':
    unittest.main()
