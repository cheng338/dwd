"""Fitted predictors own their basis and use the kernel that produced it."""
import pickle
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import rbf_kernel

from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV


def weighted_kernel(train, query, *, weights):
    return (np.asarray(train) * weights) @ np.asarray(query).T


class _KernelFixture(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(17)
        self.X = rng.normal(size=(12, 3))
        self.y = np.tile([-1, 1], 6)
        self.query = rng.normal(size=(7, 3))

    def estimator(self, **kwargs):
        options = dict(kernel='rbf', kernel_kws={'gamma': .3}, lambd=.1,
                       stopping='fixed', max_iter=12)
        options.update(kwargs)
        return KernGDWD(**options)


class FittedKernelOwnershipTests(_KernelFixture):
    def test_dense_and_sparse_caller_mutation_cannot_change_fitted_scores(self):
        for layout in ('C', 'F', 'csr'):
            for cached in (False, True):
                with self.subTest(layout=layout, cached=cached):
                    X = (csr_matrix(self.X) if layout == 'csr'
                         else np.array(self.X, order=layout))
                    model = self.estimator()
                    if cached:
                        model.cv_init(X)
                        self.assertTrue(model._cv_cache_matches(X))
                    model.fit(X, self.y)
                    before = model.decision_function(self.query)
                    coefficients = model.dual_coef_.copy()
                    if layout == 'csr':
                        X.data[:] = -17
                    else:
                        self.assertEqual(model._Xfit.flags.f_contiguous,
                                         X.flags.f_contiguous)
                        X[:] = -17
                    assert_array_equal(model.decision_function(self.query), before)
                    assert_array_equal(model.dual_coef_, coefficients)
                    assert_allclose(model._Xfit.toarray() if layout == 'csr'
                                    else model._Xfit, self.X)
                    assert_array_equal(pickle.loads(pickle.dumps(model)).predict(self.query),
                                       model.predict(self.query))

    def test_fitted_kernel_parameters_are_independent_and_refit_uses_new_ones(self):
        kws = {'gamma': .3}
        model = self.estimator(kernel_kws=kws).fit(self.X, self.y)
        before = model.decision_function(self.query)
        kws['gamma'] = 9.
        model.set_params(kernel='linear')
        assert_array_equal(model.decision_function(self.query), before)
        self.assertEqual(model.kernel_, 'rbf')
        self.assertEqual(model.kernel_kws_, {'gamma': .3})
        model.set_params(kernel='rbf').fit(self.X, self.y)
        reference = clone(model).fit(self.X, self.y)
        assert_array_equal(model.decision_function(self.query),
                           reference.decision_function(self.query))
        self.assertGreater(np.max(np.abs(model.decision_function(self.query) - before)), 1e-4)

    def test_nested_callable_keywords_are_copied(self):
        weights = np.array([.5, 1., 2.])
        model = self.estimator(kernel=weighted_kernel,
                               kernel_kws={'weights': weights}).fit(self.X, self.y)
        before = model.decision_function(self.query)
        weights[:] = 99
        assert_array_equal(model.decision_function(self.query), before)
        assert_array_equal(model.kernel_kws_['weights'], [.5, 1., 2.])

    def test_cv_preparation_keeps_cache_valid_only_for_original_inputs(self):
        X = self.X.copy()
        kws = {'gamma': .3}
        model = self.estimator(kernel_kws=kws).cv_init(X)
        X[:] = 8
        kws['gamma'] = .6
        assert_array_equal(model._Xfit, self.X)
        self.assertFalse(model._cv_cache_matches(X))
        model.fit(X, self.y)
        fresh = clone(model).fit(X, self.y)
        assert_array_equal(model.decision_function(self.query), fresh.decision_function(self.query))

    def test_failed_refit_removes_fitted_kernel_specification(self):
        model = self.estimator().fit(self.X, self.y)
        with self.assertRaises(ValueError):
            model.fit(self.X, np.ones(len(self.y)))
        for attr in ('kernel_', 'kernel_kws_', '_Xfit', 'dual_coef_'):
            self.assertFalse(hasattr(model, attr), attr)
        with self.assertRaises(NotFittedError):
            model.decision_function(self.query)

    def test_precomputed_fit_does_not_duplicate_gram_and_keeps_row_shape(self):
        K = rbf_kernel(self.X, gamma=.3)
        query = rbf_kernel(self.query, self.X, gamma=.3)
        model = self.estimator(kernel='precomputed', kernel_kws=None).fit(K, self.y)
        before = model.decision_function(query)
        self.assertTrue(np.shares_memory(K, model._Xfit))
        K[:] = -999
        K.shape = (K.size, 1)
        self.assertEqual(model._Xfit.shape, (12, 12))
        assert_array_equal(model.decision_function(query), before)

    def test_direct_rbf_and_cached_fit_match_ordinary_fresh_fit(self):
        for policy in ('standard', 'direct'):
            with self.subTest(policy=policy):
                model = self.estimator(rbf_computation=policy).cv_init(self.X)
                model.fit(self.X, self.y)
                fresh = self.estimator(rbf_computation=policy).fit(self.X, self.y)
                assert_array_equal(model.dual_coef_, fresh.dual_coef_)
                assert_array_equal(model.decision_function(self.query), fresh.decision_function(self.query))

    def test_precomputed_cv_cache_owns_reused_values(self):
        K = rbf_kernel(self.X, gamma=.3)
        original = K.copy()
        query = rbf_kernel(self.query, self.X, gamma=.3)
        model = self.estimator(kernel='precomputed', kernel_kws=None).cv_init(K)
        self.assertFalse(np.shares_memory(K, model._cv_K))
        K[:] = 0
        self.assertTrue(model._cv_cache_matches(original))
        model.fit(original, self.y)
        fresh = self.estimator(kernel='precomputed', kernel_kws=None).fit(original, self.y)
        assert_array_equal(model.dual_coef_, fresh.dual_coef_)
        assert_array_equal(model.decision_function(query), fresh.decision_function(query))


class WrapperBatchControlTests(_KernelFixture):
    def test_cv_current_batch_limit_and_per_call_override_do_not_mutate_child(self):
        model = KernGDWDCV(kernel='rbf', kernel_kws_vals=[{'gamma': .3}],
                          lambd_vals=[.1], q_vals=[1.], cv=2, stopping='fixed',
                          max_iter=12, prediction_batch_size=6).fit(self.X, self.y)
        child = model.best_estimator_
        before = child.decision_function(self.query)
        original = child._compute_kernel
        for setting, override, expected in ((2, {}, [2, 2, 2, 1]),
                                            (2, {'prediction_batch_size': None}, [7]),
                                            (None, {}, [7]),
                                            (None, {'prediction_batch_size': 3}, [3, 3, 1])):
            model.set_params(prediction_batch_size=setting)
            with patch.object(child, '_compute_kernel', wraps=original) as compute:
                scores = model.decision_function(self.query, **override)
                self.assertEqual([len(call.args[0]) for call in compute.call_args_list], expected)
            assert_allclose(scores, before, rtol=1e-13, atol=1e-13)
            self.assertEqual(child.prediction_batch_size, 6)
        for invalid in (0, -1, True, np.bool_(False), 1.5):
            model.set_params(prediction_batch_size=invalid)
            with patch.object(child, '_compute_kernel', side_effect=AssertionError('must validate first')):
                with self.assertRaisesRegex(ValueError, 'prediction_batch_size'):
                    model.decision_function(self.query)
            self.assertEqual(child.prediction_batch_size, 6)

    def test_old_pickle_without_fitted_kernel_specification_still_scores(self):
        model = self.estimator().fit(self.X, self.y)
        before = model.decision_function(self.query)
        del model.kernel_
        del model.kernel_kws_
        assert_array_equal(pickle.loads(pickle.dumps(model)).decision_function(self.query), before)


if __name__ == '__main__':
    unittest.main()
