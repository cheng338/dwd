"""Distance calculations use represented sparse coordinates, not storage slots."""
import importlib.util
import pickle
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import coo_array, coo_matrix, csc_array, csc_matrix, csr_array, csr_matrix
from sklearn.exceptions import NotFittedError

from dwd._direct_rbf import accurate_rbf
from dwd._sparse_features import (canonical_sparse_features, csr_is_canonical,
                                  checked_prediction_basis)
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV
from dwd.cv import run_cv


def duplicated(X, cls=csr_matrix):
    """Split exactly representable values into repeated, reversed CSR slots."""
    original = cls(X)
    data = np.repeat(original.data / 2, 2).astype(original.dtype)
    indices = np.repeat(original.indices, 2)
    indptr = original.indptr * 2
    for start, stop in zip(indptr[:-1], indptr[1:]):
        data[start:stop] = data[start:stop][::-1]
        indices[start:stop] = indices[start:stop][::-1]
    return cls((data, indices, indptr), shape=original.shape)


def storage(X):
    return (X.data.copy(), X.indices.copy(), X.indptr.copy(),
            getattr(X, '_has_sorted_indices', None),
            getattr(X, '_has_canonical_format', None))


def assert_storage(test, X, saved):
    for actual, expected in zip((X.data, X.indices, X.indptr), saved[:3]):
        assert_array_equal(actual, expected)
    test.assertEqual(getattr(X, '_has_sorted_indices', None), saved[3])
    test.assertEqual(getattr(X, '_has_canonical_format', None), saved[4])


class SparseFeatureDistances(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0, 0], [1, 0, -1], [0, 2, 1], [-2, 1, 0],
                           [2, 0, 2], [-1, -2, 0], [1, 1, -2], [0, -1, 2]], dtype=float) / 4
        self.y = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
        self.Q = self.X[[3, 0, 5, 7, 1]].copy()

    def model(self, **options):
        params = dict(kernel='rbf', kernel_kws={'gamma': .5}, lambd=.13,
                      initialization='zero', max_iter=4, stopping='fixed')
        return KernGDWD(**dict(params, **options))

    def test_canonical_inputs_are_reused_without_altering_metadata(self):
        for cls in (csr_matrix, csr_array):
            X = cls(self.X)
            saved = storage(X)
            self.assertIs(canonical_sparse_features(X), X)
            assert_storage(self, X, saved)

    def test_duplicate_sorted_unsorted_empty_rows_and_explicit_zeros(self):
        for cls in (csr_matrix, csr_array):
            for sorted_input in (False, True):
                X = duplicated(self.X, cls)
                if sorted_input:
                    X.sort_indices()
                saved = storage(X)
                actual = canonical_sparse_features(X)
                self.assertTrue(csr_is_canonical(actual))
                assert_array_equal(actual.toarray(), self.X)
                self.assertFalse(np.shares_memory(X.data, actual.data))
                assert_storage(self, X, saved)
            zero = cls(([1., -1., 0.], [1, 1, 2], [0, 3, 3]), shape=(2, 3))
            result = canonical_sparse_features(zero)
            self.assertEqual(result.nnz, 2)
            assert_array_equal(result.data, [0., 0.])

    def test_stale_true_metadata_is_not_trusted(self):
        for cls in (csr_matrix, csr_array):
            X = duplicated(self.X, cls)
            X.has_canonical_format = True
            X.has_sorted_indices = True
            saved = storage(X)
            self.assertFalse(csr_is_canonical(X))
            actual = canonical_sparse_features(X)
            assert_array_equal(actual.toarray(), self.X)
            self.assertTrue(csr_is_canonical(actual))
            assert_storage(self, X, saved)

    def test_sparse_normalization_and_fitting_do_not_densify_features(self):
        X, Q = duplicated(self.X), duplicated(self.Q)
        original = csr_matrix.toarray
        def forbid_feature_dense(value, *args, **kwargs):
            if value.shape[1] == self.X.shape[1]:
                raise AssertionError('Sparse features were densified.')
            return original(value, *args, **kwargs)
        # Dense n_train by n_query Gram output is part of the kernel method;
        # only densifying the input feature storage is forbidden here.
        with patch.object(csr_matrix, 'toarray', forbid_feature_dense):
            model = self.model().fit(X, self.y)
            model.decision_function(Q)

    def test_readonly_buffers_and_integer_duplicates(self):
        X = csr_array((np.array([100, 100], dtype=np.int8), [0, 0], [0, 2]), shape=(1, 1))
        for array in (X.data, X.indices, X.indptr):
            array.flags.writeable = False
        saved = storage(X)
        result = canonical_sparse_features(X)
        self.assertEqual(result.dtype, np.dtype(np.float64))
        assert_array_equal(result.toarray(), [[200.]])
        assert_storage(self, X, saved)

    def test_private_owned_copy_can_be_normalized_in_place(self):
        X = duplicated(self.X)
        self.assertIs(canonical_sparse_features(X, copy=False), X)
        assert_array_equal(X.toarray(), self.X)

    def test_finite_duplicates_that_overflow_raise_without_mutating_caller(self):
        X = csr_matrix(([1e308, 1e308], [0, 0], [0, 2]), shape=(1, 1))
        saved = storage(X)
        with self.assertRaisesRegex(ValueError, 'nonfinite.*duplicate'):
            canonical_sparse_features(X)
        assert_storage(self, X, saved)

    def test_standard_fit_cache_query_batch_pickle_and_ownership(self):
        reference = self.model().fit(csr_matrix(self.X), self.y)
        expected = reference.decision_function(csr_matrix(self.Q))
        for cls in (csr_matrix, csr_array):
            for cached in (False, True):
                with self.subTest(cls=cls.__name__, cached=cached):
                    X, Q = duplicated(self.X, cls), duplicated(self.Q, cls)
                    saved_X, saved_Q = storage(X), storage(Q)
                    model = self.model()
                    if cached:
                        model.cv_init(X)
                        self.assertTrue(csr_is_canonical(model._cv_X))
                        self.assertTrue(model._cv_cache_matches(X))
                    model.fit(X, self.y)
                    self.assertTrue(csr_is_canonical(model._Xfit))
                    assert_array_equal(model.dual_coef_, reference.dual_coef_)
                    assert_array_equal(model.objective_history_, reference.objective_history_)
                    for batch in (None, 1, 3, 20):
                        assert_allclose(model.decision_function(Q, prediction_batch_size=batch),
                                        expected, atol=1e-13, rtol=1e-13)
                    assert_allclose(pickle.loads(pickle.dumps(model)).decision_function(Q), expected,
                                    atol=1e-13, rtol=1e-13)
                    assert_storage(self, X, saved_X)
                    assert_storage(self, Q, saved_Q)
                    X.data[:] = -99
                    assert_allclose(model.decision_function(Q), expected, atol=1e-13, rtol=1e-13)

    def test_all_sparse_formats_and_float_dtypes(self):
        reference = self.model().fit(self.X, self.y)
        expected = reference.decision_function(self.Q)
        for cls in (csr_matrix, csr_array, csc_matrix, csc_array, coo_matrix, coo_array):
            for dtype in (np.float32, np.float64):
                with self.subTest(cls=cls.__name__, dtype=dtype):
                    X = cls(duplicated(self.X.astype(dtype)))
                    Q = cls(duplicated(self.Q.astype(dtype)))
                    model = self.model().fit(X, self.y)
                    assert_allclose(model.decision_function(Q), expected, atol=1e-13, rtol=1e-13)

    def test_integer_coo_conversion_promotes_before_summing(self):
        rows = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        X = coo_matrix((np.full(8, 100, dtype=np.int8), (rows, rows % 2)), shape=(4, 2))
        dense = np.array([[200., 0], [0, 200.], [200., 0], [0, 200.]])
        y = [-1, 1, -1, 1]
        model = self.model(kernel_kws={'gamma': 1e-5}).fit(X, y)
        reference = self.model(kernel_kws={'gamma': 1e-5}).fit(dense, y)
        assert_array_equal(model._Xfit.toarray(), dense)
        assert_allclose(model.decision_function(X), reference.decision_function(dense), atol=1e-13, rtol=1e-13)
        cached = self.model(kernel_kws={'gamma': 1e-5}).cv_init(X)
        assert_array_equal(cached._cv_X.toarray(), dense)
        assert_allclose(model._prepare_validation((X, y))[0],
                        reference._prepare_validation((dense, y))[0], atol=1e-15, rtol=1e-15)
        options = dict(kernel='rbf', kernel_kws_vals=[{'gamma': 1e-5}], lambd_vals=[.13],
                       q_vals=[1.], cv=2, max_iter=3, stopping='fixed')
        wrapper = KernGDWDCV(**options).fit(X, y)
        expected_wrapper = KernGDWDCV(**options).fit(dense, y)
        assert_allclose(wrapper.decision_function(X), expected_wrapper.decision_function(dense),
                        atol=1e-13, rtol=1e-13)
        cv = run_cv(self.model(kernel_kws={'gamma': 1e-5}), X, y,
                    params={'lambd': [.13]}, cv=2)
        expected_cv = run_cv(self.model(kernel_kws={'gamma': 1e-5}), dense, y,
                             params={'lambd': [.13]}, cv=2)
        assert_allclose(cv[2].decision_function(X), expected_cv[2].decision_function(dense),
                        atol=1e-13, rtol=1e-13)

    def test_float32_coo_promotes_before_large_duplicate_sum(self):
        X = coo_matrix((np.array([2e38, 2e38, 1.], dtype=np.float32),
                        ([0, 0, 1], [0, 0, 0])), shape=(2, 1))
        model = self.model(kernel_kws={'gamma': 0.}, rbf_computation='direct', max_iter=0).fit(X, [-1, 1])
        self.assertTrue(np.isfinite(model._Xfit.data).all())
        self.assertEqual(model._Xfit[0, 0], float(np.float32(2e38)) * 2)

    def test_direct_fitted_policy_retains_duplicate_training_coordinates(self):
        model = self.model(rbf_computation='direct').fit(duplicated(self.X), self.y)
        self.assertEqual(model.kernel_computation_, 'direct_v1')
        expected = model.decision_function(csr_matrix(self.Q))
        model._Xfit.has_canonical_format = True
        before = storage(model._Xfit)
        for batch in (None, 2):
            assert_allclose(model.decision_function(duplicated(self.Q), prediction_batch_size=batch),
                            expected, atol=1e-13, rtol=1e-13)
        assert_storage(self, model._Xfit, before)

    def test_direct_low_level_stale_canonical_flags(self):
        X, Q = duplicated(self.X), duplicated(self.Q)
        X.has_canonical_format = Q.has_canonical_format = True
        saved_X, saved_Q = storage(X), storage(Q)
        assert_array_equal(accurate_rbf(X, Q, gamma=.5), accurate_rbf(self.X, self.Q, gamma=.5))
        assert_storage(self, X, saved_X)
        assert_storage(self, Q, saved_Q)

    def test_old_duplicate_fitted_basis_requires_refit_for_all_norm_policies(self):
        for cls in (csr_matrix, csr_array):
            for policy in ('sklearn', 'norm_sum', None):
                model = self.model().fit(csr_matrix(self.X), self.y)
                model._Xfit = duplicated(self.X, cls)
                model._Xfit.has_canonical_format = True
                model._Xfit.has_sorted_indices = True
                model.kernel_computation_ = policy
                if policy is None:
                    del model.kernel_computation_
                    del model.kernel_
                    del model.kernel_kws_
                    model.solver_mode = 'legacy'
                before = storage(model._Xfit)
                model = pickle.loads(pickle.dumps(model))
                for batch in (None, 1, 3):
                    with self.assertRaisesRegex(ValueError, 'Refit.*original training'):
                        model.decision_function(duplicated(self.Q), prediction_batch_size=batch)
                with self.assertRaisesRegex(ValueError, 'Refit'):
                    model._prepare_validation((self.Q, self.y[[3, 0, 5, 7, 1]]))
                assert_storage(self, model._Xfit, before)

    def test_reconstructed_wrong_gram_model_is_not_repaired_by_query_normalization(self):
        X = 4 * np.eye(10)
        y = np.tile([-1, 1], 5)
        raw = csr_matrix((np.tile([1., 3.], 10), np.repeat(np.arange(10), 2),
                          np.arange(0, 21, 2)), shape=(10, 10))
        # The old stored-entry norm was 1**2 + 3**2 = 10 instead of 4**2.
        # Use its actual positive-definite represented Gram as supplied K to
        # construct a deterministic unit control. A separately saved regression
        # also loads pickles genuinely fitted by the pre-fix released wheel.
        wrong = np.full((10, 10), np.exp(-2.))
        np.fill_diagonal(wrong, 1.)
        model = self.model(kernel_kws={'gamma': .1}).fit(X, y, K=wrong)
        model._Xfit = raw
        before = model.dual_coef_.copy()
        with self.assertRaisesRegex(ValueError, 'incorrect historical kernel.*Refit'):
            model.predict(X)
        assert_array_equal(model.dual_coef_, before)
        model.fit(raw, y)
        expected = self.model(kernel_kws={'gamma': .1}).fit(X, y)
        self.assertTrue(csr_is_canonical(model._Xfit))
        self.assertFalse(np.array_equal(model.dual_coef_, before))
        assert_array_equal(model.dual_coef_, expected.dual_coef_)
        assert_array_equal(model.predict(raw), expected.predict(X))

    def test_unsorted_unique_old_bases_are_sorted_privately_and_remain_usable(self):
        for cls in (csr_matrix, csr_array):
            X = cls(self.X)
            for a, b in zip(X.indptr[:-1], X.indptr[1:]):
                X.indices[a:b] = X.indices[a:b][::-1]
                X.data[a:b] = X.data[a:b][::-1]
            X.has_canonical_format = X.has_sorted_indices = True
            saved = storage(X)
            self.assertFalse(csr_is_canonical(X))
            actual = checked_prediction_basis(X)
            self.assertTrue(csr_is_canonical(actual))
            assert_array_equal(actual.toarray(), self.X)
            for policy in ('sklearn', 'norm_sum', None):
                model = self.model().fit(csr_matrix(self.X), self.y)
                model.kernel_computation_ = policy
                if policy is None:
                    del model.kernel_computation_
                expected = model.decision_function(csr_matrix(self.Q))
                model._Xfit = X
                for batch in (None, 2):
                    assert_allclose(model.decision_function(duplicated(self.Q), prediction_batch_size=batch),
                                    expected, atol=1e-13, rtol=1e-13)
            assert_storage(self, X, saved)

    def test_old_duplicate_basis_refusal_does_not_sum_or_mutate_overflowing_entries(self):
        bad = csr_array(([1e308, 1e308], [0, 0], [0, 2]), shape=(1, 3))
        bad.has_canonical_format = bad.has_sorted_indices = True
        for values in (bad.data, bad.indices, bad.indptr):
            values.flags.writeable = False
        saved = storage(bad)
        with np.errstate(all='raise'), self.assertRaisesRegex(ValueError, 'Refit'):
            checked_prediction_basis(bad)
        assert_storage(self, bad, saved)

    def test_old_noncanonical_norm_cache_is_rejected_and_recomputed(self):
        for policy in ('sklearn', 'norm_sum'):
            model = self.model().cv_init(csr_matrix(self.X))
            model._cv_X = duplicated(self.X)
            model._cv_X.has_canonical_format = True
            model._cv_kernel_computation = policy
            model._cv_K = np.full((8, 8), np.nan)
            before = storage(model._cv_X)
            self.assertFalse(model._cv_cache_matches(csr_matrix(self.X)))
            model.fit(duplicated(self.X), self.y)
            assert_allclose(model.decision_function(self.Q), self.model().fit(self.X, self.y).decision_function(self.Q),
                            atol=1e-13, rtol=1e-13)
            assert_storage(self, model._cv_X, before)

    def test_correct_old_direct_cache_remains_eligible(self):
        model = self.model(rbf_computation='direct').cv_init(csr_matrix(self.X))
        model._cv_X = duplicated(self.X)
        self.assertTrue(model._cv_cache_matches(csr_matrix(self.X)))
        model.fit(duplicated(self.X), self.y)
        assert_allclose(model.decision_function(self.Q), self.model(rbf_computation='direct').fit(self.X, self.y).decision_function(self.Q),
                        atol=1e-13, rtol=1e-13)

    def test_validation_matrix_and_monitoring_history_match(self):
        labels = self.y[[3, 0, 5, 7, 1]]
        options = dict(stopping='validation', patience=10)
        reference = self.model(**options).fit(self.X, self.y, validation_data=(self.Q, labels))
        model = self.model(**options).fit(duplicated(self.X), self.y,
                                         validation_data=(duplicated(self.Q), labels))
        assert_allclose(model._prepare_validation((duplicated(self.Q), labels))[0],
                        reference._prepare_validation((self.Q, labels))[0], atol=1e-15, rtol=1e-15)
        self.assertEqual(model.validation_history_, reference.validation_history_)

    def test_validation_and_prediction_duplicate_overflow_raise(self):
        bad = csr_matrix(([1e308, 1e308, 0.], [0, 0, 0], [0, 2, 3]), shape=(2, 3))
        before = storage(bad)
        model = self.model().fit(self.X, self.y)
        for batch in (None, 1):
            with self.assertRaisesRegex(ValueError, 'nonfinite.*duplicate'):
                model.decision_function(bad, prediction_batch_size=batch)
        with self.assertRaisesRegex(ValueError, 'nonfinite.*duplicate'):
            self.model(stopping='validation').fit(self.X, self.y, validation_data=(bad, [-1, 1]))
        assert_storage(self, bad, before)

    def test_cv_wrapper_uses_duplicate_coordinates_consistently(self):
        options = dict(kernel='rbf', kernel_kws_vals=[{'gamma': .5}], lambd_vals=[.13],
                       q_vals=[1.], cv=2, max_iter=3, stopping='fixed')
        reference = KernGDWDCV(**options).fit(csr_matrix(self.X), self.y)
        model = KernGDWDCV(**options).fit(duplicated(self.X), self.y)
        assert_allclose(model.decision_function(duplicated(self.Q)), reference.decision_function(csr_matrix(self.Q)),
                        atol=1e-13, rtol=1e-13)

    def test_failed_refit_and_preparation_remove_fitted_state(self):
        bad = csr_matrix(([1e308, 1e308, 0.], [0, 0, 0], [0, 2, 3]), shape=(2, 3))
        before = storage(bad)
        for operation in ('fit', 'cv_init'):
            model = self.model().fit(self.X, self.y)
            with self.assertRaisesRegex(ValueError, 'nonfinite.*duplicate'):
                if operation == 'fit':
                    model.fit(bad, [-1, 1])
                else:
                    model.cv_init(bad)
            with self.assertRaises(NotFittedError):
                model.decision_function(self.Q)
            assert_storage(self, bad, before)

    def test_callable_and_precomputed_keep_their_storage_semantics(self):
        observed = []
        def kernel(train, query):
            observed.append((train.nnz, query.nnz))
            return train.toarray() @ query.toarray().T + 1.
        X, Q = duplicated(self.X), duplicated(self.Q)
        model = self.model(kernel=kernel, kernel_kws=None).fit(X, self.y)
        model.decision_function(Q)
        self.assertEqual(observed, [(X.nnz, X.nnz), (X.nnz, Q.nnz)])
        K = self.X @ self.X.T + np.eye(8)
        pre = self.model(kernel='precomputed', kernel_kws=None).fit(K, self.y)
        query = duplicated(K[:3])
        saved = storage(query)
        assert_allclose(pre.decision_function(query), pre.decision_function(K[:3]), atol=1e-13, rtol=1e-13)
        assert_storage(self, query, saved)


@unittest.skipUnless(importlib.util.find_spec('cvxpy'), 'optional cvxpy is unavailable')
class SparseAutomaticPenalty(unittest.TestCase):
    setUp = SparseFeatureDistances.setUp
    def test_auto_C_duplicate_dense_and_format_controls(self):
        from dwd.socp_dwd import auto_dwd_C
        expected = auto_dwd_C(self.X, self.y)
        for cls in (csr_matrix, csr_array, csc_matrix, csc_array, coo_matrix, coo_array):
            X = cls(duplicated(self.X))
            assert_allclose(auto_dwd_C(X, self.y), expected, rtol=1e-15)
        integer = coo_matrix((np.full(8, 100, dtype=np.int8),
                              (np.arange(8)//2, np.arange(8)//2 % 2)), shape=(4, 2))
        self.assertEqual(auto_dwd_C(integer, [-1, 1, -1, 1]), 100. / 80000.)

    def test_auto_C_invalid_labels_and_distances(self):
        from dwd.socp_dwd import auto_dwd_C
        for y in (np.ones(8), np.arange(8), self.y.astype(complex) + 1j):
            with self.assertRaises(ValueError):
                auto_dwd_C(duplicated(self.X), y)
        with self.assertRaisesRegex(ValueError, 'positive median'):
            auto_dwd_C(csr_matrix(np.ones((8, 3))), self.y)
        for const in (0., -1., np.inf):
            with self.assertRaises(ValueError):
                auto_dwd_C(duplicated(self.X), self.y, const=const)

    def test_auto_C_preserves_float32_range_rejection_and_refit_cleanup(self):
        from dwd.socp_dwd import DWD, auto_dwd_C
        bad = coo_matrix((np.array([2e38, 2e38, 1.], dtype=np.float32),
                          ([0, 0, 1], [0, 0, 0])), shape=(2, 1))
        before = (bad.data.copy(), bad.row.copy(), bad.col.copy())
        with np.errstate(over='ignore', invalid='ignore'):
            with self.assertRaises(ValueError):
                auto_dwd_C(bad, [-1, 1])
            model = DWD(C='auto').fit(csr_matrix(self.X), self.y)
            with self.assertRaises(ValueError):
                model.fit(bad, [-1, 1])
        self.assertFalse(hasattr(model, 'coef_'))
        self.assertFalse(hasattr(model, 'C_'))
        for actual, expected in zip((bad.data, bad.row, bad.col), before):
            assert_array_equal(actual, expected)

    def test_small_auto_conic_fit_matches_and_keeps_caller(self):
        from dwd.socp_dwd import DWD
        X = duplicated(self.X)
        saved = storage(X)
        reference = DWD(C='auto').fit(csr_matrix(self.X), self.y)
        model = DWD(C='auto').fit(X, self.y)
        self.assertEqual(model.C_, reference.C_)
        assert_allclose(model.coef_, reference.coef_, rtol=1e-7, atol=1e-7)
        assert_allclose(model.intercept_, reference.intercept_, rtol=1e-7, atol=1e-7)
        assert_storage(self, X, saved)


if __name__ == '__main__':
    unittest.main()
