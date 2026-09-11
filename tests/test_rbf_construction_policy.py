"""Data-free RBF construction, provenance, cache and query-policy controls.

Injected self-kernel asymmetry tests dispatch without relying on BLAS-specific
rounding. The saved real-data failures are covered by the separate audit replay.
"""
import pickle
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import pairwise_kernels
from dwd._rbf import norm_sum_rbf
from dwd.gen_kern_dwd import KernGDWD


class RBFConstructionTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.],
                           [-1., 0.], [0., -1.], [-1., -1.], [2., 1.]]) / 4.
        self.y = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

    def estimator(self, **options):
        params = dict(kernel='rbf', kernel_kws={'gamma': .5}, lambd=.13,
                      initialization='zero', max_iter=3, stopping='fixed')
        return KernGDWD(**dict(params, **options))

    @staticmethod
    def perturbed_self_kernel(X, Y, metric, **kwargs):
        K = pairwise_kernels(X, Y, metric=metric, **kwargs)
        if X is Y:
            K[0, 1] += 1e-9
        return K

    def test_dense_and_csr_norm_sum_match_exact_dyadic_squared_distances(self):
        squared = np.sum((self.X[:, None, :] - self.X[None, :, :]) ** 2, axis=2)
        expected = np.exp(-.5 * squared)
        for features in (self.X, csr_matrix(self.X)):
            K = norm_sum_rbf(features, features, gamma=.5, self_kernel=True)
            assert_array_equal(K, K.T)
            assert_array_equal(np.diag(K), np.ones(len(self.X)))
            assert_allclose(K, expected, rtol=2e-15, atol=2e-15)
            query = features[[5, 1, 5, 0]]
            actual = norm_sum_rbf(features, query, gamma=.5)
            assert_allclose(actual, expected[:, [5, 1, 5, 0]], rtol=2e-15, atol=2e-15)

    def test_healthy_path_retains_sklearn_kernel(self):
        model = self.estimator()
        model._Xfit = self.X
        actual = model._compute_training_kernel(self.X)
        expected = pairwise_kernels(self.X, self.X, metric='rbf', gamma=.5)
        self.assertEqual(model.kernel_computation_, 'sklearn')
        self.assertEqual(model.kernel_symmetry_correction_, 0.)
        assert_array_equal(actual, expected)

    def test_reconstructed_policy_survives_cache_pickle_and_batched_queries(self):
        for implementation in ('optimized', 'reference'):
            with self.subTest(implementation=implementation), patch(
                    'dwd.kernel_utils.pairwise_kernels', side_effect=self.perturbed_self_kernel):
                fresh = self.estimator(implementation=implementation).fit(self.X, self.y)
                cached = self.estimator(implementation=implementation).cv_init(self.X).fit(self.X, self.y)
                expected = fresh.decision_function(self.X.copy())
                for model in (fresh, cached, pickle.loads(pickle.dumps(cached))):
                    self.assertEqual(model.kernel_computation_, 'norm_sum')
                    self.assertGreater(model.kernel_symmetry_correction_, 1e-10)
                    model.prediction_batch_size = 3
                    assert_allclose(model.decision_function(self.X.copy()), expected,
                                    atol=5e-7 * max(1., float(np.max(np.abs(expected)))), rtol=0.)
                    assert_array_equal(model.predict(self.X.copy()), fresh.predict(self.X.copy()))
                del cached._cv_kernel_computation
                self.assertFalse(cached._cv_cache_matches(self.X))
                with self.assertRaises(ValueError):
                    fresh.fit(self.X, np.ones(len(self.y)))
                self.assertFalse(hasattr(fresh, 'kernel_computation_'))
                with self.assertRaises(NotFittedError):
                    fresh.predict(self.X)

    def test_external_and_custom_asymmetric_kernels_remain_rejected(self):
        bad = pairwise_kernels(self.X, self.X, metric='rbf', gamma=.5)
        bad[0, 1] += 1e-9
        original = bad.copy()
        custom = lambda X, Y, **kwargs: bad.copy()
        for model, features, kwargs in (
                (self.estimator(), self.X, {'K': bad}),
                (self.estimator(kernel='precomputed'), bad, {}),
                (self.estimator(kernel=custom), self.X, {})):
            with self.subTest(kernel=model.kernel), self.assertRaisesRegex(ValueError, 'K must be symmetric'):
                model.fit(features, self.y, **kwargs)
        assert_array_equal(bad, original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
