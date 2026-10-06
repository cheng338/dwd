"""Optional affine evaluation must preserve fitted policy and default results."""

from fractions import Fraction
import pickle
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from dwd.cv import run_cv
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV


def linear_callable(X, Y):
    return X @ Y.T


def sparse_linear_callable(X, Y):
    # The existing fit API requires a dense training Gram matrix. Query kernels
    # may be sparse, including the training-by-query CSR conversion path.
    return X @ Y.T if X is Y else csr_matrix(X @ Y.T)


def exact_scores(K, alpha, b):
    K = K.toarray() if hasattr(K, 'toarray') else K
    return np.array([float(sum((Fraction(float(k)) * Fraction(float(a))
                               for k, a in zip(row, alpha)), Fraction(float(b))))
                     for row in K])


class AffineComputationTests(unittest.TestCase):
    def setUp(self):
        self.limits = threadpool_limits(1)
        self.addCleanup(self.limits.restore_original_limits)
        self.X = np.array([[0.], [1.]])
        self.y = np.array([-3, 7])
        self.query = np.array([[np.nextafter(.5, np.inf)]])

    def model(self, **overrides):
        options = dict(kernel='linear', lambd=.1, q=1., max_iter=5,
                       stopping='fixed', initialization='zero', random_state=7)
        options.update(overrides)
        return KernGDWD(**options)

    def test_real_fitted_cancellation_dense_sparse_batch_and_pickle(self):
        ordinary = self.model().fit(self.X, self.y)
        joint = self.model(affine_computation='joint').fit(self.X, self.y)
        for name in ('dual_coef_', 'intercept_', 'objective_history_'):
            np.testing.assert_array_equal(getattr(ordinary, name), getattr(joint, name))
        expected = exact_scores(joint._compute_kernel(self.query).T,
                                joint.dual_coef_[0], joint.intercept_[0])
        self.assertGreater(expected[0], 0.)
        self.assertEqual(ordinary.decision_function(self.query)[0], 0.)
        self.assertEqual(ordinary.predict(self.query)[0], -3)
        for model in (joint, pickle.loads(pickle.dumps(joint))):
            self.assertEqual(model.affine_computation_, 'joint')
            for batch in (None, 1, 2):
                model.prediction_batch_size = batch
                for query in (np.repeat(self.query, 3, axis=0),
                              csr_matrix(np.repeat(self.query, 3, axis=0))):
                    np.testing.assert_array_equal(model.decision_function(query),
                                                  np.repeat(expected, 3))
                    np.testing.assert_array_equal(model.predict(query), [7, 7, 7])

    def test_fitted_policy_changes_only_on_refit(self):
        for fitted, changed in (('joint', 'standard'), ('standard', 'joint')):
            model = self.model(affine_computation=fitted).fit(self.X, self.y)
            expected = model.decision_function(self.query)
            model.set_params(affine_computation=changed)
            np.testing.assert_array_equal(model.decision_function(self.query), expected)
            self.assertEqual(model.affine_computation_, fitted)
            restored = pickle.loads(pickle.dumps(model))
            np.testing.assert_array_equal(restored.decision_function(self.query), expected)
            restored.fit(self.X, self.y)
            self.assertEqual(restored.affine_computation_, changed)
            self.assertEqual(restored.predict(self.query)[0], 7 if changed == 'joint' else -3)

    def test_default_does_not_call_joint_helper_even_for_validation(self):
        with patch('dwd._affine_scores.affine_scores', side_effect=AssertionError('Unexpected joint scoring')):
            default = self.model().fit(self.X, self.y)
            standard = self.model(affine_computation='standard').fit(self.X, self.y)
            np.testing.assert_array_equal(default.decision_function(self.query),
                                          standard.decision_function(self.query))
            for name in ('dual_coef_', 'intercept_', 'objective_history_'):
                np.testing.assert_array_equal(getattr(default, name), getattr(standard, name))
            for policy in (None, 'standard'):
                options = {} if policy is None else {'affine_computation': policy}
                self.model(stopping='validation', **options).fit(
                    self.X, self.y, validation_data=(self.query, np.array([7])))

    def test_clone_and_old_serialized_model_and_cache(self):
        model = self.model().cv_init(self.X).fit(self.X, self.y)
        expected = model.decision_function(self.query)
        for name in ('affine_computation', 'affine_computation_', '_cv_affine_computation'):
            del model.__dict__[name]
        old = pickle.loads(pickle.dumps(model))
        self.assertEqual(old.get_params()['affine_computation'], 'standard')
        self.assertEqual(clone(old).affine_computation, 'standard')
        self.assertTrue(old._cv_cache_matches(self.X))
        np.testing.assert_array_equal(old.decision_function(self.query), expected)
        old.set_params(affine_computation='joint')
        # A constructor setting on a historical fitted object cannot reinterpret it.
        np.testing.assert_array_equal(old.decision_function(self.query), expected)
        self.assertFalse(old._cv_cache_matches(self.X))
        for estimator in (self.model(affine_computation='joint'),
                          KernGDWDCV(affine_computation='joint')):
            self.assertEqual(clone(estimator).affine_computation, 'joint')
            del estimator.__dict__['affine_computation']
            restored = pickle.loads(pickle.dumps(estimator))
            self.assertEqual(clone(restored).affine_computation, 'standard')

    def test_cache_identity_and_failed_fit_cleanup(self):
        model = self.model().cv_init(self.X)
        self.assertTrue(model._cv_cache_matches(self.X))
        model.set_params(affine_computation='joint')
        self.assertFalse(model._cv_cache_matches(self.X))
        model.cv_init(self.X).fit(self.X, self.y)
        self.assertTrue(model._cv_cache_matches(self.X))
        model.set_params(affine_computation='standard')
        self.assertFalse(model._cv_cache_matches(self.X))
        model.set_params(affine_computation='invalid')
        with self.assertRaises(ValueError):
            model.fit(self.X, self.y)
        self.assertFalse(hasattr(model, 'affine_computation_'))
        with self.assertRaises(NotFittedError):
            model.predict(self.query)
        model.set_params(affine_computation='joint').fit(self.X, self.y)
        model.cv_init(self.X)
        self.assertFalse(hasattr(model, 'affine_computation_'))

    def test_callables_and_precomputed_share_joint_evaluation(self):
        for kernel in ('linear', linear_callable, sparse_linear_callable, 'precomputed'):
            with self.subTest(kernel=kernel):
                train = self.X @ self.X.T if kernel == 'precomputed' else self.X
                query = self.query @ self.X.T if kernel == 'precomputed' else self.query
                model = self.model(kernel=kernel, affine_computation='joint').fit(train, self.y)
                expected = exact_scores(model._compute_kernel(query).T,
                                        model.dual_coef_[0], model.intercept_[0])
                for layout in (query, csr_matrix(query)):
                    np.testing.assert_array_equal(model.decision_function(layout), expected)

    def test_joint_supports_both_fitted_precision_modes(self):
        model = self.model(affine_computation='joint').fit(self.X, self.y)
        expected = exact_scores(model._compute_kernel(self.query).T,
                                model.dual_coef_[0], model.intercept_[0])
        for precision in ('adaptive', 'compensated'):
            model.prediction_precision_ = precision
            np.testing.assert_array_equal(model.decision_function(self.query), expected)

    def test_validation_history_matches_represented_affine_signs(self):
        for options in ({}, {'backend': 'spectral'}, {'backend': 'lbfgs'},
                        {'implementation': 'reference'}, {'check_interval': 2},
                        {'patience': 2, 'min_delta': .25}):
            with self.subTest(options=options):
                snapshots = {}

                def callback(snapshot):
                    snapshots[snapshot['iteration']] = snapshot
                    np.testing.assert_array_equal(snapshot['decision_values'],
                                                  snapshot['training_scores'] + snapshot['offset'])

                model = self.model(affine_computation='joint', stopping='validation',
                                   callback=callback, **options).fit(
                                       self.X, self.y, validation_data=(self.query, np.array([7])))
                K = model._compute_kernel(self.query).T
                for row in model.validation_history_:
                    snapshot = snapshots[row['iteration']]
                    expected = exact_scores(K, snapshot['alpha'], snapshot['offset'])
                    self.assertEqual(row['score'], float(np.mean(expected > 0)))
                best = max(model.validation_history_, key=lambda row: row['score'])
                self.assertEqual(model.returned_iteration_, best['iteration'])
                self.assertEqual(float(np.mean(model.predict(self.query) == 7)), best['score'])

    def test_zero_iteration_validation_uses_same_fitted_policy(self):
        seed = self.model().fit(self.X, self.y)
        for policy, expected in (('standard', 0.), ('joint', 1.)):
            model = self.model(affine_computation=policy, stopping='validation', max_iter=0).fit(
                self.X, self.y, alpha_init=seed.dual_coef_[0], offset_init=float(seed.intercept_[0]),
                validation_data=(self.query, np.array([7])))
            self.assertEqual(model.validation_history_[0]['score'], expected)
            self.assertEqual(float(np.mean(model.predict(self.query) == 7)), expected)

    def test_public_cv_and_mixed_policy_grid(self):
        values = np.linspace(-2., 2., 16)
        X = np.column_stack((values, np.sin(values)))
        y = np.where(values > 0, 7, -3)
        cv = KernGDWDCV(lambd_vals=[.1], q_vals=[1.], cv=2, max_iter=2,
                        stopping='fixed', random_state=7, affine_computation='joint').fit(X, y)
        self.assertEqual(cv.affine_computation_, 'joint')
        self.assertEqual(cv.best_estimator_.affine_computation_, 'joint')
        expected = cv.decision_function(X)
        cv.set_params(affine_computation='standard')
        np.testing.assert_array_equal(cv.decision_function(X), expected)
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(cv)).decision_function(X), expected)
        folds = list(StratifiedKFold(2, shuffle=True, random_state=19).split(X, y))
        mixed = run_cv(self.model(max_iter=2), X, y,
                       {'affine_computation': ['standard', 'joint'], 'lambd': [.1, .2]},
                       cv=folds, refit_best=False)[3]
        for policy in ('standard', 'joint'):
            separate = run_cv(self.model(max_iter=2, affine_computation=policy), X, y,
                              {'lambd': [.1, .2]}, cv=folds, refit_best=False)[3]
            for params, score in zip(separate['params'], separate['mean_test_score']):
                index = mixed['params'].index(dict(params, affine_computation=policy))
                self.assertEqual(mixed['mean_test_score'][index], score)

    def test_joint_combines_with_direct_rbf(self):
        X = np.arange(12, dtype=float).reshape(6, 2) / 10.
        y = np.array([-1, -1, -1, 1, 1, 1])
        model = self.model(kernel='rbf', kernel_kws={'gamma': .3},
                           rbf_computation='direct', affine_computation='joint').fit(X, y)
        self.assertEqual(model.kernel_computation_, 'direct_v1')
        self.assertEqual(model.affine_computation_, 'joint')
        expected = exact_scores(model._compute_kernel(X).T,
                                model.dual_coef_[0], model.intercept_[0])
        np.testing.assert_array_equal(model.predict(X), model.classes_[(expected > 0).astype(int)])

    def test_invalid_options_and_legacy_fail_before_fitting(self):
        for value in (None, True, 0, [], np.array(['standard']), 'invalid'):
            for method in ('fit', 'cv_init'):
                with self.subTest(value=value, method=method), self.assertRaises(ValueError):
                    model = self.model(affine_computation=value)
                    model.fit(self.X, self.y) if method == 'fit' else model.cv_init(self.X)
        with self.assertRaises(ValueError):
            self.model(solver_mode='legacy', affine_computation='joint').fit(self.X, self.y)
        with self.assertRaises(ValueError):
            KernGDWDCV(solver_mode='legacy', affine_computation='joint').fit(self.X, self.y)


if __name__ == '__main__':
    unittest.main()
