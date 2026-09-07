"""Public-API contracts for the reference and optimized MM implementations.

All observations are manufactured here. CV tests verify API propagation and
pairwise slicing; they do not tune MNIST or access any external dataset.
"""
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import eigh
from sklearn.base import clone, is_classifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV
from dwd._reference_mm import reference_update


IMPLEMENTATIONS = ('reference', 'optimized')


class TwoImplementationPublicTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(612042)
        self.X = rng.normal(size=(24, 4))
        self.y = np.tile(np.array([-7, 9]), 12)
        self.query = rng.normal(size=(5, 4))
        self.alpha = rng.normal(size=len(self.y)) / 20.
        self.offset = .27

    def model(self, implementation, **kwargs):
        options = dict(implementation=implementation, kernel='rbf',
                       kernel_kws={'gamma': .3}, lambd=.07, q=1.,
                       max_iter=6, stopping='fixed', initialization='zero')
        options.update(kwargs)
        return KernGDWD(**options)

    def assert_model_function_equal(self, left, right, atol=2e-10):
        assert_allclose(left.decision_function(self.query), right.decision_function(self.query),
                        rtol=2e-9, atol=atol)
        assert_allclose(left.intercept_, right.intercept_, rtol=2e-9, atol=atol)
        assert_allclose(left.objective_history_, right.objective_history_, rtol=2e-9, atol=atol)

    def test_public_selection_clone_and_backend_diagnostics(self):
        self.assertEqual(KernGDWD().implementation, 'optimized')
        for implementation in IMPLEMENTATIONS:
            with self.subTest(implementation=implementation):
                model = self.model(implementation).fit(self.X, self.y)
                copied = clone(model)
                self.assertTrue(is_classifier(copied))
                self.assertEqual(copied.implementation, implementation)
                self.assertEqual(copied.get_params(), model.get_params())
                self.assertFalse(hasattr(copied, 'dual_coef_'))
                self.assertEqual(model.diagnostics_['implementation'], implementation)
                self.assertEqual(model.backend_, 'spectral' if implementation == 'reference' else 'cholesky')
        with self.assertRaisesRegex(ValueError, 'implementation'):
            self.model('unknown').fit(self.X, self.y)

    def test_reference_uses_validated_eigenbasis_and_coefficient_steps_without_cholesky(self):
        message = 'The coefficient/eigenbasis reference must not call Cholesky.'
        with patch('dwd._kernel_linear_system.cho_factor', side_effect=AssertionError(message)), \
                patch('dwd._kernel_linear_system.cho_solve', side_effect=AssertionError(message)), \
                patch('dwd._kernel_solver.cho_factor', side_effect=AssertionError(message)), \
                patch('dwd._kernel_solver.cho_solve', side_effect=AssertionError(message)), \
                patch('dwd._kernel_solver.reference_update', wraps=reference_update) as update, \
                patch('dwd._eigen.eigh', wraps=eigh) as eigen:
            model = self.model('reference', max_iter=3).fit(self.X, self.y,
                alpha_init=self.alpha, offset_init=self.offset)
        self.assertEqual(update.call_count, 3)
        self.assertGreaterEqual(eigen.call_count, 1)
        self.assertEqual(eigen.call_args_list[0].kwargs['driver'], 'evd')
        self.assertEqual(model.backend_, 'spectral')
        self.assertEqual(model.diagnostics_['implementation'], 'reference')
        for backend in ('cholesky', 'lbfgs'):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                self.model('reference', backend=backend).fit(self.X, self.y)

    def test_native_initialization_defaults_preserve_reference_random_and_optimized_zero(self):
        seed = 97
        expected = np.random.RandomState(seed).normal(size=len(self.y))
        expected /= np.linalg.norm(expected)
        for implementation in IMPLEMENTATIONS:
            with self.subTest(implementation=implementation):
                default = KernGDWD(implementation=implementation)
                self.assertEqual(default.initialization, 'auto')
                self.assertEqual(default.stopping, 'objective')
                self.assertEqual(default.obj_tol, 1e-5)
                self.assertEqual(default.max_iter, 100)
                actual = self.model(implementation, initialization='auto',
                                    random_state=seed, max_iter=0).fit(self.X, self.y)
                assert_allclose(actual.dual_coef_[0], expected if implementation == 'reference'
                                else np.zeros(len(self.y)), rtol=2e-13, atol=2e-13)
                self.assertEqual(actual.intercept_[0], 0.)
                self.assertEqual(actual.n_iter_, 0)

    def test_global_rng_reference_semantics_and_explicit_initialization_precedence(self):
        old_state = np.random.get_state()
        try:
            seed = 102
            expected_rng = np.random.RandomState(seed)
            expected = expected_rng.normal(size=len(self.y))
            expected /= np.linalg.norm(expected)
            expected_next = expected_rng.normal(size=4)
            np.random.seed(seed)
            fitted = self.model('reference', initialization='auto', random_state=None,
                                max_iter=0).fit(self.X, self.y)
            assert_allclose(fitted.dual_coef_[0], expected, atol=2e-13, rtol=2e-13)
            assert_array_equal(np.random.normal(size=4), expected_next)
            for implementation in IMPLEMENTATIONS:
                for initialization in ('auto', 'zero', 'random'):
                    with self.subTest(implementation=implementation, initialization=initialization):
                        np.random.seed(seed)
                        fitted = self.model(implementation, initialization=initialization,
                            random_state=None, max_iter=0).fit(self.X, self.y,
                            alpha_init=self.alpha, offset_init=self.offset)
                        assert_allclose(fitted.dual_coef_[0], self.alpha, atol=2e-13, rtol=2e-13)
                        self.assertEqual(fitted.intercept_[0], self.offset)
                        assert_array_equal(np.random.normal(size=4), np.random.RandomState(seed).normal(size=4))
        finally:
            np.random.set_state(old_state)

    def test_explicit_zero_and_random_options_work_for_both_implementations(self):
        expected = np.random.RandomState(33).normal(size=len(self.y))
        expected /= np.linalg.norm(expected)
        for implementation in IMPLEMENTATIONS:
            zero = self.model(implementation, initialization='zero', random_state=33,
                              max_iter=0).fit(self.X, self.y)
            random = self.model(implementation, initialization='random', random_state=33,
                                max_iter=0).fit(self.X, self.y)
            assert_allclose(zero.dual_coef_[0], np.zeros(len(self.y)), atol=1e-14)
            assert_allclose(random.dual_coef_[0], expected, rtol=2e-13, atol=2e-13)

    def test_external_gridsearch_selects_both_implementations_and_refits_public_classifier(self):
        base = Pipeline([('scale', StandardScaler()), ('model', self.model('optimized', max_iter=3))])
        search = GridSearchCV(base, {'model__implementation': list(IMPLEMENTATIONS),
                                    'model__lambd': [.04, .15]},
                              cv=StratifiedKFold(3), scoring='accuracy', error_score='raise').fit(self.X, self.y)
        self.assertEqual(len(search.cv_results_['params']), 4)
        self.assertTrue(np.isfinite(search.cv_results_['mean_test_score']).all())
        self.assertEqual({p['model__implementation'] for p in search.cv_results_['params']}, set(IMPLEMENTATIONS))
        best = search.best_estimator_.named_steps['model']
        self.assertEqual(best.implementation, search.best_params_['model__implementation'])
        self.assertEqual(best.diagnostics_['implementation'], best.implementation)
        self.assertTrue(set(search.predict(self.query)).issubset(set(self.y)))

    def test_external_pairwise_gridsearch_matches_explicit_fold_slicing_for_each_implementation(self):
        K = rbf_kernel(self.X, gamma=.3)
        query_K = rbf_kernel(self.query, self.X, gamma=.3)
        folds = list(StratifiedKFold(3).split(K, self.y))
        for implementation in IMPLEMENTATIONS:
            with self.subTest(implementation=implementation):
                base = self.model(implementation, kernel='precomputed', kernel_kws=None, max_iter=3)
                search = GridSearchCV(base, {'lambd': [.04, .15]}, cv=folds,
                                      scoring='accuracy', error_score='raise').fit(K, self.y)
                for candidate, lambd in enumerate((.04, .15)):
                    manual = []
                    for train, validation in folds:
                        model = clone(base).set_params(lambd=lambd).fit(K[np.ix_(train, train)], self.y[train])
                        manual.append(np.mean(model.predict(K[np.ix_(validation, train)]) == self.y[validation]))
                    observed = [search.cv_results_[f'split{i}_test_score'][candidate] for i in range(3)]
                    assert_array_equal(observed, manual)
                self.assertEqual(search.best_estimator_.implementation, implementation)
                self.assertEqual(search.predict(query_K).shape, (len(self.query),))

    def test_internal_cv_propagates_implementation_controls_and_refit(self):
        for implementation in IMPLEMENTATIONS:
            with self.subTest(implementation=implementation):
                fitted = KernGDWDCV(implementation=implementation, lambd_vals=[.04, .15],
                    q_vals=[1.], kernel='rbf', kernel_kws_vals=[{'gamma': .3}], cv=3,
                    initialization='zero', stopping='fixed', max_iter=2, random_state=77).fit(self.X, self.y)
                best = fitted.best_estimator_
                self.assertEqual(best.implementation, implementation)
                self.assertEqual(best.diagnostics_['implementation'], implementation)
                self.assertEqual(best.initialization, 'zero')
                self.assertEqual(best.n_iter_, 2)
                self.assertEqual(best.lambd, fitted.best_params_['lambd'])
                self.assertEqual(best.backend_, 'spectral' if implementation == 'reference' else 'cholesky')
                assert_array_equal(fitted.predict(self.query), best.predict(self.query))
                with self.assertRaisesRegex(ValueError, 'monitoring splits'):
                    KernGDWDCV(implementation=implementation, stopping='validation').fit(self.X, self.y)

    def test_fixed_and_objective_stopping_match_with_same_explicit_initial_state(self):
        for stopping, options in (('fixed', {'max_iter': 10}),
                                  ('objective', {'max_iter': 100, 'obj_tol': .02})):
            fits = []
            for implementation in IMPLEMENTATIONS:
                fitted = self.model(implementation, stopping=stopping, **options).fit(
                    self.X, self.y, alpha_init=self.alpha, offset_init=self.offset)
                fits.append(fitted)
                self.assertEqual(fitted.n_iter_, len(fitted.objective_history_) - 1)
                self.assertEqual(fitted.returned_iteration_, fitted.n_iter_)
                self.assertEqual(fitted.termination_reason_, 'max_iter' if stopping == 'fixed' else 'objective_tolerance')
                if stopping == 'fixed':
                    self.assertEqual(fitted.n_iter_, 10)
                else:
                    changes = np.abs(np.diff(fitted.objective_history_))
                    self.assertLess(changes[-1], .02)
                    self.assertTrue(np.all(changes[:-1] >= .02))
            self.assertEqual(fits[0].n_iter_, fits[1].n_iter_)
            self.assert_model_function_equal(*fits)

    def test_callback_counts_immutable_snapshots_and_stopping_match(self):
        fits, histories = [], []
        for implementation in IMPLEMENTATIONS:
            seen = []
            def callback(state):
                seen.append(state['iteration'])
                if state['iteration'] == 0:
                    assert_allclose(state['alpha'], self.alpha, atol=2e-13)
                    self.assertEqual(state['offset'], self.offset)
                    with self.assertRaises(ValueError):
                        state['alpha'][0] = 123.
                return state['iteration'] == 3
            fitted = self.model(implementation, callback=callback, max_iter=10).fit(
                self.X, self.y, alpha_init=self.alpha, offset_init=self.offset)
            self.assertEqual(seen, [0, 1, 2, 3])
            self.assertEqual(fitted.n_iter_, 3)
            self.assertEqual(fitted.termination_reason_, 'callback_stop')
            fits.append(fitted)
            histories.append(seen)
        self.assertEqual(histories[0], histories[1])
        self.assert_model_function_equal(*fits)

    def test_validation_patience_returns_same_earliest_best_and_preserves_stopped_history(self):
        X = np.zeros((8, 2))
        y = np.r_[np.full(6, 9), np.full(2, -7)]
        validation = (np.zeros((4, 2)), np.full(4, 9))
        fits = []
        for implementation in IMPLEMENTATIONS:
            seen = []
            fitted = self.model(implementation, kernel='linear', kernel_kws=None, lambd=.2,
                stopping='validation', max_iter=20, patience=2, check_interval=1,
                callback=lambda state: seen.append(state['iteration'])).fit(
                    X, y, alpha_init=np.zeros(8), offset_init=.2, validation_data=validation)
            self.assertEqual(seen, [0, 1, 2, 3])
            self.assertEqual(fitted.termination_reason_, 'validation_patience')
            self.assertEqual(fitted.n_iter_, 3)
            self.assertEqual(fitted.returned_iteration_, 1)
            self.assertEqual([v['iteration'] for v in fitted.validation_history_], [1, 2, 3])
            self.assertEqual([v['score'] for v in fitted.validation_history_], [1., 1., 1.])
            self.assertEqual(len(fitted.objective_history_), 4)
            self.assertAlmostEqual(fitted.final_objective_, fitted.objective_history_[1], delta=1e-12)
            fits.append(fitted)
        assert_allclose(fits[0].intercept_, fits[1].intercept_, atol=2e-12)
        assert_allclose(fits[0].objective_history_, fits[1].objective_history_, atol=2e-12)

    def test_one_step_keeps_package_lambda_original_kernel_and_free_intercept(self):
        K = rbf_kernel(self.X, gamma=.3)
        n, lambd = len(self.y), .07
        ones = np.ones((n, 1))
        augmented = np.block([[K + (n*lambd/2) * np.eye(n), ones],
                              [ones.T, np.zeros((1, 1))]])
        signed = np.where(self.y == 9, 1., -1.)
        expected = np.linalg.solve(augmented, np.r_[signed/4., 0.])
        for implementation in IMPLEMENTATIONS:
            fitted = self.model(implementation, max_iter=1).fit(
                self.X, self.y, alpha_init=np.zeros(n), offset_init=0.)
            self.assertEqual(fitted.lambd, lambd)
            self.assertEqual(fitted.q, 1.)
            self.assertEqual(fitted.kernel_kws, {'gamma': .3})
            assert_allclose(fitted.dual_coef_[0], expected[:-1], rtol=3e-10, atol=2e-12)
            assert_allclose(fitted.intercept_[0], expected[-1], rtol=3e-10, atol=2e-12)
            assert_allclose(fitted.decision_function(self.X), K @ expected[:-1] + expected[-1],
                            rtol=3e-10, atol=2e-12)
            norm = expected[:-1] @ K @ expected[:-1]
            self.assertAlmostEqual(fitted.C_, 4. * norm, delta=2e-11)


if __name__ == '__main__':
    unittest.main(verbosity=2)
