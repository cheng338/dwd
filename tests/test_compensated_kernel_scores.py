"""Original-matrix query accuracy; no classifier objective or kernel changes."""
from decimal import Decimal, localcontext
import io
import pickle
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.sparse import csr_matrix, csr_array
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from dwd._kernel_scores import compensated_kernel_matvec, adaptive_kernel_matvec
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV


def decimal_product(K, alpha):
    with localcontext() as context:
        context.prec = 100
        D = Decimal.from_float
        return np.array([float(sum((D(float(k))*D(float(a)) for k, a in zip(row, alpha)), Decimal(0)))
                         for row in K])


def sparse_linear_kernel(A, B):
    return csr_matrix(A @ B.T)


class CompensatedKernelScoreTests(unittest.TestCase):
    def setUp(self):
        self.alpha = np.array([1e16, 1., -1e16])
        self.query = np.array([[1., 1., 1.], [2., -3., 2.], [0., 4., 0.],
                               [1., 1e-10, 1.], [-1., -2., -1.], [.5, 3., .5],
                               [0., 0., 0.]])

    def model(self, kernel='precomputed', precision='compensated'):
        # A saved finite coefficient state isolates prediction from fitting.
        model = KernGDWD(kernel=kernel)
        model._Xfit = np.eye(3)
        model.classes_ = np.array([-7, 9])
        model.n_features_in_ = 3
        model.dual_coef_ = self.alpha[None, :].copy()
        model.intercept_ = np.array([.125])
        model.prediction_precision_ = precision
        return model

    def test_rectangular_dense_and_csr_match_decimal_products(self):
        expected = decimal_product(self.query, self.alpha)
        before = self.query.copy(), self.alpha.copy()
        for query in (self.query, self.query.astype(np.float32),
                      csr_matrix(self.query), csr_array(self.query)):
            assert_array_equal(compensated_kernel_matvec(query, self.alpha),
                               decimal_product(query.toarray() if hasattr(query, 'toarray') else query, self.alpha))
        assert_array_equal(compensated_kernel_matvec(self.query, self.alpha), expected)
        assert_array_equal(self.query, before[0])
        assert_array_equal(self.alpha, before[1])

    def test_high_low_products_resolve_error_missed_by_rounded_product_sum(self):
        eps = 2.**-27
        query = np.array([[1.+eps, -1.]])
        alpha = np.array([1.-eps, 1.])
        self.assertEqual(float(sum(query[0]*alpha)), 0.)
        self.assertEqual(compensated_kernel_matvec(query, alpha)[0], -2.**-54)
        assert_array_equal(compensated_kernel_matvec(query, alpha), decimal_product(query, alpha))

    def test_csr_duplicate_entries_are_not_prerounded(self):
        # The mathematical row is 1e16 + 1 - 1e16, kept as CSR contributions.
        query = csr_matrix((np.array([1e16, 1., -1e16]), np.zeros(3, dtype=int), np.array([0, 3])), shape=(1, 1))
        before = query.data.copy()
        assert_array_equal(compensated_kernel_matvec(query, np.array([1.])), [1.])
        assert_array_equal(query.data, before)

    def test_nonfinite_invalid_and_overflow_inputs_fail(self):
        for query, alpha in (([1., 2.], [1., 2.]), ([[1., 2.]], [[1., 2.]]),
                             ([[1., 2.]], [1.]), ([[1j]], [1.]), ([[1.]], [1j]),
                             ([['1']], [1.]), ([[1.]], []), (csr_matrix([[1.]]).tocsc(), [1.])):
            with self.assertRaises(ValueError):
                compensated_kernel_matvec(query, alpha)
        for query, alpha in (([[np.nan]], [1.]), ([[1.]], [np.inf]),
                             (csr_matrix([[np.inf]]), [1.]), ([[1e308]], [2.]),
                             ([[1e308, 1e308]], [1., 1.])):
            with self.assertRaises(FloatingPointError):
                compensated_kernel_matvec(query, alpha)

    def test_empty_query_rows_are_allowed(self):
        self.assertEqual(compensated_kernel_matvec(np.empty((0, 3)), self.alpha).shape, (0,))

    def test_prediction_dense_sparse_callable_and_batches(self):
        expected = decimal_product(self.query, self.alpha) + .125
        for kernel in ('precomputed', 'linear', sparse_linear_kernel):
            model = self.model(kernel=kernel)
            for batch in (None, 1, 2, 7, 200):
                model.prediction_batch_size = batch
                for query in (self.query, csr_matrix(self.query)) if kernel == 'precomputed' else (self.query,):
                    assert_array_equal(model.decision_function(query), expected)
                    assert_array_equal(model.predict(query), np.where(expected > 0, 9, -7))

    def test_ordinary_models_never_call_compensated_helper(self):
        model = self.model(precision='ordinary')
        with patch('dwd._kernel_scores.compensated_kernel_matvec', side_effect=AssertionError('unexpected slow path')):
            for batch in (None, 1, 2):
                model.prediction_batch_size = batch
                expected = np.concatenate([(row[None, :] @ self.alpha[:, None]).ravel() for row in self.query]) + .125
                # Native BLAS may sum whole/batched products differently; preserve its fast result.
                actual = model.decision_function(self.query)
                self.assertTrue(np.isfinite(actual).all())
            del model.prediction_precision_
            model.decision_function(self.query)

    def test_pickle_joblib_and_cv_delegation_keep_prediction_mode(self):
        model = self.model()
        expected = model.decision_function(self.query)
        for restored in (pickle.loads(pickle.dumps(model)),):
            self.assertEqual(restored.prediction_precision_, 'compensated')
            assert_array_equal(restored.decision_function(self.query), expected)
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        self.assertEqual(restored.prediction_precision_, 'compensated')
        assert_array_equal(restored.decision_function(self.query), expected)
        cv = KernGDWDCV(kernel='precomputed')
        cv.best_estimator_ = restored
        assert_array_equal(cv.decision_function(self.query), expected)
        fresh = clone(model)
        self.assertFalse(hasattr(fresh, 'prediction_precision_'))
        with self.assertRaises(NotFittedError):
            fresh.decision_function(self.query)

    def test_result_storage_defaults_and_failed_refit_clear_mode(self):
        result = dict(offset=.125, alpha=self.alpha, objective_history=[1.], n_iter=0,
                      returned_iteration=0, final_objective=1., converged=False,
                      termination_reason='max_iter', C=1., gradient_inf_norm=1.,
                      rkhs_gradient_norm=1., dual_gap=1., backend='cholesky', diagnostics={})
        model = self.model()
        model._set_fit_result(dict(result, prediction_precision='compensated'))
        self.assertEqual(model.prediction_precision_, 'compensated')
        model._set_fit_result(result)
        self.assertEqual(model.prediction_precision_, 'ordinary')
        with self.assertRaises(ValueError):
            model._set_fit_result(dict(result, prediction_precision='unknown'))
        model = self.model()
        model.set_params(stopping='invalid')
        with self.assertRaises(ValueError):
            model.fit(np.eye(3), [-1, 1, -1])
        self.assertFalse(hasattr(model, 'prediction_precision_'))
        self.assertFalse(hasattr(model, 'dual_coef_'))
        with self.assertRaises(NotFittedError):
            model.decision_function(self.query)

    def test_adaptive_dense_sparse_layout_and_unseen_query_cancellation(self):
        expected = decimal_product(self.query, self.alpha)
        for query in (self.query, np.asfortranarray(self.query), csr_matrix(self.query)):
            assert_array_equal(adaptive_kernel_matvec(query, self.alpha), expected)
        # Training-like diagonal rows are safe, yet unseen near-equal columns
        # can cancel with these same stored coefficients.
        with patch('dwd._kernel_scores._expanded_row', side_effect=AssertionError('safe rows expanded')):
            assert_array_equal(adaptive_kernel_matvec(np.eye(3), self.alpha), self.alpha)
        model = self.model(precision='adaptive')
        for batch in (None, 1, 2, 200):
            model.prediction_batch_size = batch
            assert_array_equal(model.decision_function(self.query), expected+.125)
            assert_array_equal(model.decision_function(csr_matrix(self.query)), expected+.125)
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.prediction_precision_, 'adaptive')
        assert_array_equal(restored.decision_function(self.query), expected+.125)

    def test_adaptive_expands_only_uncertain_rows(self):
        import dwd._kernel_scores as scores
        original = scores._expanded_row
        rows = []
        def observed(K, alpha, sparse, i):
            rows.append(i)
            return original(K, alpha, sparse, i)
        query = np.array([[1., 0., 0.], [1., 1., 1.], [0., 2., 0.]])
        with patch.object(scores, '_expanded_row', side_effect=observed):
            assert_array_equal(adaptive_kernel_matvec(query, self.alpha), [1e16, 1., 2.])
        self.assertEqual(rows, [1])

    def test_adaptive_overflowing_bound_and_subnormal_terms(self):
        # Each product is representable; only the nonnegative absolute sum
        # overflows. Accurate cancellation must still return a finite zero.
        for query in (np.array([[1e308, 1e308]]), csr_matrix([[1e308, 1e308]])):
            assert_array_equal(adaptive_kernel_matvec(query, np.array([1., -1.])), [0.])
        eta = np.nextafter(0., 1.)
        query = np.array([[eta, -eta], [2*eta, eta]])
        alpha = np.array([.5, 1.])
        actual = adaptive_kernel_matvec(query, alpha)
        # Error in ordinary subnormal sums is many orders below the absolute
        # policy threshold; the estimate explicitly accounts for underflow.
        assert_allclose(actual, decimal_product(query, alpha), rtol=0., atol=2*eta)

    def test_adaptive_low_risk_never_expands_products_and_checks_finiteness(self):
        query, alpha = np.array([[.5, .25], [1., -2.]]), np.array([.125, .5])
        with patch('dwd._kernel_scores._products', side_effect=AssertionError('unexpected expansion')):
            assert_array_equal(adaptive_kernel_matvec(query, alpha), query@alpha)
        with self.assertRaises(FloatingPointError):
            adaptive_kernel_matvec([[np.nan, 1.]], alpha)
        with self.assertRaises(FloatingPointError):
            adaptive_kernel_matvec(csr_matrix([[np.inf, 1.]]), alpha)
        with self.assertRaises(ValueError):
            adaptive_kernel_matvec([[1., 1.]], np.array([1.]))


if __name__ == '__main__':
    unittest.main()
