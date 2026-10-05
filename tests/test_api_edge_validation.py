"""Small public-API regressions for validation, lifecycle and fold layout."""
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from dwd.cv import run_cv
from dwd.gen_dwd import GenDWD
from dwd.gen_kern_dwd import KernGDWD
from dwd.kernel_utils import KernelScaler


class PublicAPIEdgeTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-1.], [1.]])
        self.y = np.array([-1, 1])

    def test_scaler_failed_first_fit_and_refit_leave_no_usable_state(self):
        for previous in (False, True):
            for invalid in (np.zeros((2, 2)), -np.eye(2), np.ones((2, 3)),
                            np.full((2, 2), np.nan)):
                with self.subTest(previous=previous, invalid=invalid.tolist()):
                    scaler = KernelScaler()
                    if previous:
                        scaler.fit(np.eye(2))
                    with self.assertRaises(ValueError):
                        scaler.fit(invalid)
                    with self.assertRaises(NotFittedError):
                        check_is_fitted(scaler)
                    with self.assertRaises(NotFittedError):
                        scaler.transform(np.eye(2))
                    self.assertFalse(hasattr(scaler, 'n_features_in_'))

    def test_scaler_owns_its_diagonal_and_preserves_ordinary_arithmetic(self):
        K = np.array([[4., 1.], [1., 9.]])
        original = K.copy()
        scaler = KernelScaler().fit(K)
        K[:] = 0.
        np.testing.assert_array_equal(scaler.K_diag_, [4., 9.])
        scales = 1. / np.sqrt(np.diag(original) / len(original))
        expected = original.copy()
        expected *= scales[None, :]
        expected *= scales[:, None]
        actual = scaler.transform(original)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(original, [[4., 1.], [1., 9.]])
        inplace = original.copy()
        self.assertIs(scaler.transform(inplace, copy=False), inplace)
        np.testing.assert_array_equal(inplace, expected)

    def test_scaler_positive_subnormal_diagonal_does_not_underflow(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                tiny = np.nextafter(dtype(0.), dtype(1.))
                K = np.diag(np.array([tiny, tiny], dtype=dtype))
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    actual = KernelScaler().fit_transform(K)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, 2 * np.eye(2), rtol=5 * np.finfo(dtype).eps)

    def test_scaler_unrepresentable_output_raises_instead_of_returning_inf(self):
        scaler = KernelScaler().fit(np.eye(2))
        with self.assertRaisesRegex(FloatingPointError, 'nonfinite transformed'):
            scaler.transform(np.full((2, 2), np.finfo(float).max))

    def test_complex_initialization_rejects_without_discarding_imaginary_parts(self):
        for imaginary in (0., 3.):
            for route in ('linear', 'legacy', 'schur'):
                with self.subTest(imaginary=imaginary, route=route):
                    if route == 'linear':
                        model, data = GenDWD(max_iter=0), self.X
                        initial = {'beta_init': np.array([1. + imaginary * 1j])}
                    else:
                        model = KernGDWD(kernel='precomputed', solver_mode=route, max_iter=0)
                        data = np.eye(2)
                        initial = {'alpha_init': np.array([1. + imaginary * 1j, -1.])}
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        with self.assertRaisesRegex(ValueError, 'must be real'):
                            model.fit(data, self.y, **initial)
                    self.assertEqual(caught, [])
                    with self.assertRaises(NotFittedError):
                        model.predict(data)

    def test_nonfinite_linear_scores_raise_with_dense_and_sparse_queries(self):
        X = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
        model = GenDWD(max_iter=0).fit(X, np.tile([-1, 1], 2),
                                     beta_init=np.array([2., -2.]), offset_init=1.)
        ordinary = np.array([[2., 1.], [-2., 1.]])
        expected = ordinary @ model.coef_[0] + model.intercept_[0]
        np.testing.assert_array_equal(model.decision_function(ordinary), expected)
        for query in (np.array([[1.e308, 1.e308]]), csr_matrix([[1.e308, 1.e308]])):
            with self.subTest(sparse=isinstance(query, csr_matrix)):
                for method in (model.decision_function, model.predict):
                    with self.assertRaisesRegex(FloatingPointError, 'Nonfinite linear'):
                        method(query)

    def test_kernel_path_preparation_invalidates_old_predictor_and_allows_refit(self):
        model = KernGDWD(kernel='linear', max_iter=0).fit(
            self.X, self.y, alpha_init=np.array([-.5, .5]))
        old_scores = model.decision_function([[1.]])
        model.cv_init(-self.X)
        with self.assertRaises(NotFittedError):
            model.predict([[1.]])
        self.assertTrue(model._cv_cache_matches(-self.X))
        # Reuse the prepared kernel; this must not rebuild it during fit.
        with patch.object(model, '_compute_training_kernel', side_effect=AssertionError('cache missed')):
            model.fit(-self.X, self.y, alpha_init=np.array([-.5, .5]))
        np.testing.assert_allclose(model.decision_function([[1.]]), -old_scores)

    def test_failed_kernel_path_preparation_does_not_expose_hybrid_state(self):
        model = KernGDWD(kernel='linear', max_iter=0).fit(
            self.X, self.y, alpha_init=np.array([-.5, .5]))
        error = RuntimeError('controlled kernel preparation failure')
        with patch.object(model, '_compute_training_kernel', side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                model.cv_init(-self.X)
        self.assertIs(caught.exception, error)
        with self.assertRaises(NotFittedError):
            model.predict([[1.]])

    def test_cv_slices_for_candidate_kernel_instead_of_constructor_kernel(self):
        K, y = np.eye(6) + .1, np.tile([-1, 1], 3)
        for target, original in (('precomputed', 'linear'), ('linear', 'precomputed')):
            with self.subTest(target=target):
                expected = run_cv(KernGDWD(kernel=target, max_iter=1, stopping='fixed'),
                                  K, y, {'lambd': [1.]}, cv=3)
                actual = run_cv(KernGDWD(kernel=original, max_iter=1, stopping='fixed'),
                                K, y, {'kernel': [target], 'lambd': [1.]}, cv=3)
                np.testing.assert_array_equal(actual[3]['mean_test_score'], expected[3]['mean_test_score'])
                for fold_actual, fold_expected in zip(actual[4], expected[4]):
                    np.testing.assert_array_equal(fold_actual['test_score'], fold_expected['test_score'])
                    np.testing.assert_array_equal(fold_actual['train_score'], fold_expected['train_score'])
                np.testing.assert_array_equal(actual[2].decision_function(K), expected[2].decision_function(K))
                self.assertEqual(actual[2].kernel, target)

    def test_nonsquare_precomputed_candidate_rejects_before_kernel_preparation(self):
        X, y = np.arange(12.).reshape(6, 2), np.tile([-1, 1], 3)
        with patch.object(KernGDWD, 'cv_init', side_effect=AssertionError('must not prepare')):
            with self.assertRaisesRegex(ValueError, 'requires a square input matrix'):
                run_cv(KernGDWD(max_iter=0), X, y, {'kernel': ['precomputed']}, cv=3)


if __name__ == '__main__':
    unittest.main()
