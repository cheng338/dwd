"""Public opt-in RBF policy, legacy state, and CV cache regression tests."""

import pickle
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import StratifiedKFold

from dwd._direct_rbf import accurate_rbf
from dwd.cv import run_cv
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV
from dwd.kernel_utils import KernelClfMixin


class RBFComputationTests(unittest.TestCase):
    def setUp(self):
        values = np.linspace(-2., 2., 16)
        self.X = np.column_stack((values, np.sin(values), np.cos(2 * values)))
        self.y = np.where(values > 0, 1, -1)

    def estimator(self, **overrides):
        options = dict(kernel='rbf', kernel_kws={'gamma': .3}, lambd=.1,
                       max_iter=2, stopping='fixed', initialization='zero', random_state=7)
        options.update(overrides)
        return KernGDWD(**options)

    def test_constructor_clone_and_objects_without_new_parameter(self):
        for model in (self.estimator(), KernGDWDCV()):
            self.assertEqual(model.get_params()['rbf_computation'], 'standard')
            model.set_params(rbf_computation='direct')
            self.assertEqual(clone(model).get_params()['rbf_computation'], 'direct')
            del model.__dict__['rbf_computation']
            restored = pickle.loads(pickle.dumps(model))
            self.assertEqual(restored.get_params()['rbf_computation'], 'standard')
            self.assertEqual(clone(restored).rbf_computation, 'standard')

    def test_standard_default_preserves_existing_kernel_path(self):
        for features in (self.X, csr_matrix(self.X)):
            with patch('dwd._direct_rbf.accurate_rbf', side_effect=AssertionError('Unexpected direct RBF')):
                implicit = self.estimator().fit(features, self.y)
                explicit = self.estimator(rbf_computation='standard').fit(features, self.y)
                np.testing.assert_array_equal(implicit.dual_coef_, explicit.dual_coef_)
                np.testing.assert_array_equal(implicit.intercept_, explicit.intercept_)
                np.testing.assert_array_equal(implicit.decision_function(features), explicit.decision_function(features))
                implicit.cv_init(features)
            self.assertEqual(implicit.kernel_computation_, 'sklearn')
            np.testing.assert_array_equal(implicit._cv_K,
                                          pairwise_kernels(features, features, metric='rbf', gamma=.3))

    def test_direct_fitted_policy_is_used_for_queries_and_validation(self):
        for features in (self.X, csr_matrix(self.X)):
            model = self.estimator(rbf_computation='direct').fit(features, self.y)
            self.assertEqual(model.kernel_computation_, 'direct_v1')
            query = self.X[[3, 0, 3, 9]]
            expected = accurate_rbf(self.X, query, gamma=.3)
            np.testing.assert_array_equal(model._compute_kernel(query), expected)
            np.testing.assert_array_equal(model._compute_kernel(csr_matrix(query)), expected)
            prepared, signed = model._prepare_validation((query, self.y[[3, 0, 3, 9]]))
            np.testing.assert_array_equal(prepared, expected.T)
            np.testing.assert_array_equal(signed, self.y[[3, 0, 3, 9]])
            model.set_params(rbf_computation='standard')
            # A constructor change takes effect on refit, not on existing coefficients.
            np.testing.assert_array_equal(model._compute_kernel(query), expected)
            restored = pickle.loads(pickle.dumps(model))
            self.assertEqual(restored.kernel_computation_, 'direct_v1')
            np.testing.assert_array_equal(restored._compute_kernel(query), expected)

    def test_old_norm_sum_model_and_cache_survive_serialization(self):
        original = pairwise_kernels

        def asymmetric_self(X, Y, metric, **kwargs):
            result = original(X, Y, metric=metric, **kwargs)
            if X is Y:
                result[0, 1] += 1e-9
            return result

        # Trigger the historical fallback; the source of asymmetry is incidental.
        with patch('dwd.kernel_utils.pairwise_kernels', side_effect=asymmetric_self):
            model = self.estimator().cv_init(self.X).fit(self.X, self.y)
        self.assertEqual(model.kernel_computation_, 'norm_sum')
        expected = model.decision_function(self.X.copy())
        del model.__dict__['rbf_computation']
        del model.__dict__['_cv_rbf_computation']
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.rbf_computation, 'standard')
        self.assertTrue(restored._cv_cache_matches(self.X))
        np.testing.assert_array_equal(restored.decision_function(self.X.copy()), expected)
        with patch('dwd._direct_rbf.accurate_rbf', side_effect=AssertionError('Unexpected direct RBF')):
            restored.fit(self.X, self.y)
        self.assertEqual(restored.kernel_computation_, 'norm_sum')
        np.testing.assert_array_equal(restored.decision_function(self.X.copy()), expected)

    def test_cv_cache_separates_policies_in_both_directions(self):
        model = self.estimator().cv_init(self.X)
        self.assertTrue(model._cv_cache_matches(self.X.copy()))
        model.set_params(rbf_computation='direct')
        self.assertFalse(model._cv_cache_matches(self.X))
        model.cv_init(self.X)
        self.assertTrue(model._cv_cache_matches(self.X))
        self.assertEqual(model._cv_kernel_computation, 'direct_v1')
        model.set_params(rbf_computation='standard')
        self.assertFalse(model._cv_cache_matches(self.X))
        del model.__dict__['_cv_rbf_computation']
        self.assertFalse(model._cv_cache_matches(self.X))

    def test_public_cv_forwards_explicit_option(self):
        model = KernGDWDCV(kernel='rbf', lambd_vals=[.1], q_vals=[1.],
                          kernel_kws_vals=[{'gamma': .3}], cv=2, max_iter=2,
                          stopping='fixed', initialization='zero', random_state=7,
                          rbf_computation='direct').fit(self.X, self.y)
        self.assertEqual(model.best_estimator_.rbf_computation, 'direct')
        self.assertEqual(model.best_estimator_.kernel_computation_, 'direct_v1')
        np.testing.assert_array_equal(model.predict(self.X), model.best_estimator_.predict(self.X))

    def test_mixed_policy_cv_matches_separate_fixed_fold_searches(self):
        folds = list(StratifiedKFold(2, shuffle=True, random_state=19).split(self.X, self.y))
        mixed = run_cv(self.estimator(), self.X, self.y,
                       {'rbf_computation': ['standard', 'direct'], 'lambd': [.1, .2]},
                       cv=folds, refit_best=False)[3]
        for policy in ('standard', 'direct'):
            separate = run_cv(self.estimator(rbf_computation=policy), self.X, self.y,
                              {'lambd': [.1, .2]}, cv=folds, refit_best=False)[3]
            for parameters, score in zip(separate['params'], separate['mean_test_score']):
                key = dict(parameters, rbf_computation=policy)
                index = mixed['params'].index(key)
                self.assertEqual(mixed['mean_test_score'][index], score)

    def test_rejects_invalid_policies_and_unsupported_configurations(self):
        for value in (None, True, 0, [], 'unknown'):
            for method in ('fit', 'cv_init'):
                with self.subTest(policy=value, method=method), self.assertRaises(ValueError):
                    model = self.estimator(rbf_computation=value)
                    model.fit(self.X, self.y) if method == 'fit' else model.cv_init(self.X)
        unsupported = [
            {'kernel': 'linear'}, {'kernel': 'precomputed'},
            {'kernel': lambda X, Y: X @ Y.T}, {'solver_mode': 'legacy'},
            {'kernel_kws': {'degree': 2}}, {'kernel_kws': {'gamma': -1}},
            {'kernel_kws': {'gamma': np.inf}}, {'kernel_kws': {'gamma': np.nan}},
        ]
        for options in unsupported:
            for method in ('fit', 'cv_init'):
                with self.subTest(options=options, method=method), self.assertRaises(ValueError):
                    model = self.estimator(rbf_computation='direct', **options)
                    model.fit(self.X, self.y) if method == 'fit' else model.cv_init(self.X)
        with self.assertRaises(ValueError):
            self.estimator(rbf_computation='direct').fit(self.X, self.y, K=np.eye(len(self.X)))

    def test_explicit_direct_rejects_overridden_construction(self):
        class CustomKernel(KernGDWD):
            def _compute_kernel(self, X):
                return KernelClfMixin._compute_kernel(self, X)

        class CustomTraining(KernGDWD):
            def _compute_training_kernel(self, X):
                return super()._compute_training_kernel(X)

        class CustomPSD(KernGDWD):
            def _known_psd_kernel(self):
                return True

        for constructor in (CustomKernel, CustomTraining, CustomPSD):
            for method in ('fit', 'cv_init'):
                with self.subTest(constructor=constructor, method=method), self.assertRaises(ValueError):
                    model = constructor(kernel='rbf', rbf_computation='direct')
                    model.fit(self.X, self.y) if method == 'fit' else model.cv_init(self.X)
        standard = CustomKernel(kernel='rbf').cv_init(self.X)
        self.assertEqual(standard.kernel_computation_, 'sklearn')

    def test_boolean_gamma_retains_estimator_compatibility(self):
        # The estimator's existing named-RBF contract accepts Python bool gamma.
        for gamma in (False, True):
            model = self.estimator(rbf_computation='direct', kernel_kws={'gamma': gamma}).cv_init(self.X)
            np.testing.assert_array_equal(model._cv_K, accurate_rbf(self.X, gamma=float(gamma)))


if __name__ == '__main__':
    unittest.main()
