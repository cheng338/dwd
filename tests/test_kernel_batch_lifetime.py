"""Completed prediction batches release owned kernels without changing results."""
import gc
import unittest
import weakref

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator

from dwd.kernel_utils import KernelClfMixin


class _ObservedKernelClassifier(KernelClfMixin, BaseEstimator):
    """Small fitted state with allocation observations, not retained kernel views."""

    def __init__(self, batch_size=2, retain=False, score_view=False):
        self.batch_size = batch_size
        self.retain = retain
        self.score_view = score_view
        self.prediction_batch_size = batch_size
        self.kernel = 'linear'
        self._Xfit = np.array([[1., 2.], [-1., 3.], [2., -2.]])
        self.dual_coef_ = np.array([[1., -2., .5]])
        self.intercept_ = np.array([.25])
        self.classes_ = np.array([-1, 1])
        self.references = []
        self.previous_live = []
        self.shapes = []
        self.owns_storage = []
        self.retained = []
        self.kernel_calls = 0
        self.score_calls = 0
        self.failure_stage = None
        self.failure_type = RuntimeError

    def _compute_kernel(self, X):
        # Permit delayed collection on Python implementations without refcounts.
        # An earlier batch still held by decision_function remains reachable.
        gc.collect()
        self.previous_live.append(bool(self.references and self.references[-1]() is not None))
        self.kernel_calls += 1
        self._maybe_fail('kernel', self.kernel_calls)
        K = self._Xfit @ X.T
        self.references.append(weakref.ref(K))
        self.shapes.append(K.shape)
        self.owns_storage.append(K.flags.owndata)
        if self.retain:
            self.retained.append((K, K.copy()))
        return K

    def _kernel_decision_product(self, K):
        self.score_calls += 1
        self._maybe_fail('score', self.score_calls)
        if self.score_view:
            return K[0].reshape(-1, 1)
        return super()._kernel_decision_product(K)

    def _maybe_fail(self, stage, calls):
        if self.failure_stage == stage and calls == 2:
            self.failure_stage = None
            raise self.failure_type('prediction failure')


class KernelBatchLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.query = np.array([[1., 0.], [0., 2.], [-1., 1.], [2., -1.], [3., 1.]])

    def expected_scores(self, model):
        return ((model._Xfit @ self.query.T).T @ model.dual_coef_.T + model.intercept_).ravel()

    def test_owned_kernel_is_released_before_next_batch_construction(self):
        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                model = _ObservedKernelClassifier(batch_size=batch_size)
                result = model.decision_function(self.query)
                count = (len(self.query) + batch_size - 1) // batch_size
                self.assertEqual(model.previous_live, [False] * count)
                self.assertTrue(all(model.owns_storage))
                self.assertEqual(model.shapes, [(3, min(batch_size, len(self.query) - start))
                                               for start in range(0, len(self.query), batch_size)])
                gc.collect()
                self.assertTrue(all(reference() is None for reference in model.references))
                assert_array_equal(result, self.expected_scores(model))

    def test_score_view_is_copied_before_kernel_release(self):
        for retain in (False, True):
            with self.subTest(externally_retained=retain):
                model = _ObservedKernelClassifier(score_view=True, retain=retain)
                result = model.decision_function(self.query)
                expected = model._Xfit[0] @ self.query.T
                assert_array_equal(result, expected)
                if retain:
                    for K, original in model.retained:
                        assert_array_equal(K, original)
                        self.assertFalse(np.shares_memory(result, K))
                        K.fill(-99.)
                    assert_array_equal(result, expected)
                else:
                    gc.collect()
                    self.assertTrue(all(reference() is None for reference in model.references))

    def test_external_kernel_owner_keeps_unchanged_arrays(self):
        model = _ObservedKernelClassifier(retain=True)
        result = model.decision_function(self.query)
        self.assertEqual(model.previous_live, [False, True, True])
        self.assertEqual(len(model.retained), 3)
        for reference, (K, original) in zip(model.references, model.retained):
            self.assertIs(reference(), K)
            assert_array_equal(K, original)
            self.assertFalse(np.shares_memory(result, K))
        assert_array_equal(result, self.expected_scores(model))

    def test_construction_or_scoring_failure_allows_unchanged_retry(self):
        for stage in ('kernel', 'score'):
            for error_type in (RuntimeError, MemoryError, KeyboardInterrupt):
                with self.subTest(stage=stage, error=error_type.__name__):
                    model = _ObservedKernelClassifier()
                    originals = [value.copy() for value in (self.query, model._Xfit,
                                                            model.dual_coef_, model.intercept_)]
                    model.failure_stage = stage
                    model.failure_type = error_type
                    with self.assertRaisesRegex(error_type, 'prediction failure'):
                        model.decision_function(self.query)
                    self.assertEqual(model.kernel_calls, 2)
                    self.assertEqual(model.score_calls, 1 if stage == 'kernel' else 2)
                    assert_array_equal(model.decision_function(self.query), self.expected_scores(model))
                    for current, original in zip((self.query, model._Xfit, model.dual_coef_,
                                                  model.intercept_), originals):
                        assert_array_equal(current, original)


if __name__ == '__main__':
    unittest.main()
