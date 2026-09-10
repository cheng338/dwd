"""Failed fitting must never expose old predictions under new data or labels."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from dwd.gen_dwd import GenDWD, GenDWDCV
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV
from dwd.kern_md import KernMD

try:
    import cvxpy
except ImportError:
    cvxpy = None
else:
    from dwd.socp_dwd import DWD
    from dwd.svm import SVM


class _FailureContract:
    def setUp(self):
        rng = np.random.RandomState(2026)
        self.X = rng.normal(size=(12, 3))
        self.X[:, 0] += np.repeat([-1.5, 1.5], 6)
        self.y = np.repeat([-1, 1], 6)
        self.new_X = np.column_stack((self.X[:, 1:], self.X[:, 0] ** 2, self.X[:, 0]))
        self.new_y = np.where(self.y < 0, 'new-negative', 'new-positive')

    def assert_unfitted(self, model):
        learned = [key for key in vars(model)
                   if key.endswith('_') and not key.startswith('_')]
        self.assertEqual(learned, [], 'Partial or stale learned attributes survived')
        self.assertFalse(hasattr(model, '_Xfit'))
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        for method in (model.predict, model.decision_function):
            with self.assertRaises(NotFittedError):
                method(self.X)

    def test_invalid_parameter_after_new_labels_clears_previous_fit(self):
        for model, invalid, target in self.cases():
            with self.subTest(model=type(model).__name__):
                model.fit(self.X, self.y)
                model.set_params(**invalid)
                with self.assertRaises((ValueError, NotImplementedError)):
                    model.fit(self.new_X, self.new_y)
                self.assert_unfitted(model)
                # Failure does not silently reset the user's constructor choices.
                self.assertEqual({key: model.get_params()[key] for key in invalid}, invalid)

    def test_invalid_input_and_single_class_clear_previous_fit(self):
        for model, invalid, target in self.cases():
            for X, y in ((self.X[:-1], self.y),
                         (self.new_X, np.full(len(self.y), 'only-new-class'))):
                with self.subTest(model=type(model).__name__, rows=len(X)):
                    model.fit(self.X, self.y)
                    with self.assertRaises(ValueError):
                        model.fit(X, y)
                    self.assert_unfitted(model)

    def test_late_solver_failure_clears_previous_fit_and_preserves_exception(self):
        for model, invalid, target in self.cases():
            with self.subTest(model=type(model).__name__):
                model.fit(self.X, self.y)
                failure = RuntimeError('controlled solver failure after input validation')
                with patch(target, side_effect=failure) as solver:
                    with self.assertRaises(RuntimeError) as caught:
                        model.fit(self.new_X, self.new_y)
                self.assertIs(caught.exception, failure)
                self.assertTrue(solver.called)
                self.assert_unfitted(model)

    def test_failed_initial_fit_leaves_no_partial_state(self):
        for model, invalid, target in self.cases():
            with self.subTest(model=type(model).__name__):
                with patch(target, side_effect=FloatingPointError('controlled numerical failure')):
                    with self.assertRaises(FloatingPointError):
                        model.fit(self.X, self.y)
                self.assert_unfitted(model)

    def test_successful_refit_and_recovery_match_fresh_estimator(self):
        for model, invalid, target in self.cases():
            with self.subTest(model=type(model).__name__):
                model.fit(self.X, self.y)
                fresh = clone(model).fit(self.new_X, self.new_y)
                self.assertIs(model.fit(self.new_X, self.new_y), model)
                assert_array_equal(model.classes_, fresh.classes_)
                assert_array_equal(model.predict(self.new_X), fresh.predict(self.new_X))
                assert_allclose(model.decision_function(self.new_X),
                                fresh.decision_function(self.new_X), rtol=0, atol=1e-12)
                with patch(target, side_effect=RuntimeError('controlled failure')):
                    with self.assertRaises(RuntimeError):
                        model.fit(self.X, self.y)
                self.assert_unfitted(model)
                model.fit(self.new_X, self.new_y)
                assert_allclose(model.decision_function(self.new_X),
                                fresh.decision_function(self.new_X), rtol=0, atol=1e-12)

    def test_public_fit_signature_and_clone_are_preserved(self):
        for model, invalid, target in self.cases():
            with self.subTest(model=type(model).__name__):
                method = type(model).fit
                self.assertEqual(inspect.signature(method), inspect.signature(method.__wrapped__))
                self.assertEqual(list(inspect.signature(method).parameters)[:3], ['self', 'X', 'y'])
                self.assertEqual(clone(model).get_params(), model.get_params())


class FailedRefitMMTests(_FailureContract, unittest.TestCase):
    def cases(self):
        options = dict(max_iter=3, stopping='fixed', random_state=42)
        return [
            (GenDWD(lambd=.1, **options), {'lambd': -1.}, 'dwd.gen_dwd.solve_gen_dwd'),
            (GenDWDCV(lambd_vals=[.1], q_vals=[1.], cv=2, **options),
             {'lambd_vals': [-1.]}, 'dwd.gen_dwd.solve_gen_dwd'),
            (KernGDWD(lambd=.1, kernel='rbf', **options),
             {'lambd': -1.}, 'dwd._kernel_solver.solve_kernel'),
            (KernGDWDCV(lambd_vals=[.1], q_vals=[1.], kernel='rbf', cv=2, **options),
             {'lambd_vals': [-1.]}, 'dwd._kernel_solver.solve_kernel'),
            (KernMD(), {'naive_bayes': True}, 'dwd.kern_md.kern_md'),
        ]

    def test_unsupported_weights_clear_existing_fit(self):
        for model, invalid, target in self.cases()[:-1]:
            with self.subTest(model=type(model).__name__):
                model.fit(self.X, self.y)
                with self.assertRaises(NotImplementedError):
                    model.fit(self.X, self.y, sample_weight=np.ones(len(self.y)))
                self.assert_unfitted(model)

    def test_failure_after_new_coefficients_are_assigned_is_cleaned(self):
        linear = GenDWD(max_iter=3).fit(self.X, self.y)
        with patch('dwd.gen_dwd._linear_gradient', side_effect=RuntimeError('diagnostic failure')):
            with self.assertRaisesRegex(RuntimeError, 'diagnostic failure'):
                linear.fit(self.new_X, self.new_y)
        self.assert_unfitted(linear)

        kernel = KernGDWD(max_iter=3, kernel='rbf').fit(self.X, self.y)
        set_result = kernel._set_fit_result

        def store_then_fail(result):
            set_result(result)
            raise RuntimeError('failure after complete result storage')

        with patch.object(kernel, '_set_fit_result', side_effect=store_then_fail):
            with self.assertRaisesRegex(RuntimeError, 'complete result storage'):
                kernel.fit(self.new_X, self.new_y)
        self.assert_unfitted(kernel)

    def test_interrupted_solver_is_cleaned_and_interrupt_propagates(self):
        model = GenDWD(max_iter=3).fit(self.X, self.y)
        failure = KeyboardInterrupt('controlled interruption')
        with patch('dwd.gen_dwd.solve_gen_dwd', side_effect=failure):
            with self.assertRaises(KeyboardInterrupt) as caught:
                model.fit(self.new_X, self.new_y)
        self.assertIs(caught.exception, failure)
        self.assert_unfitted(model)

    def test_linear_precomputation_survives_success_and_failure_and_is_reused(self):
        model = GenDWD(lambd=.1, max_iter=3).cv_init(self.X)
        cache = {key: value for key, value in vars(model).items() if key.startswith('_P0_')}
        expected = clone(model).fit(self.X, self.y).decision_function(self.X)
        for fail_first in (False, True):
            if fail_first:
                with self.assertRaises(ValueError):
                    model.fit(self.X, np.ones(len(self.y)))
                self.assert_unfitted(model)
            with patch('dwd.gen_dwd.get_P0_eig', side_effect=AssertionError('cache rebuilt')):
                model.fit(self.X, self.y)
            for key, value in cache.items():
                self.assertIs(getattr(model, key), value)
            assert_allclose(model.decision_function(self.X), expected, rtol=0, atol=1e-12)

    def test_kernel_precomputation_survives_failure_and_is_reused(self):
        for implementation in ('optimized', 'reference'):
            with self.subTest(implementation=implementation):
                model = KernGDWD(lambd=.1, max_iter=3, kernel='rbf', random_state=42,
                                 implementation=implementation).cv_init(self.X)
                cache = {key: value for key, value in vars(model).items()
                         if key.startswith('_cv_') or key == '_K_eig'}
                expected = clone(model).fit(self.X, self.y).decision_function(self.X)
                model.fit(self.X, self.y)
                with self.assertRaises(ValueError):
                    model.fit(self.new_X, np.ones(len(self.y)))
                self.assert_unfitted(model)
                with patch.object(model, '_compute_kernel', side_effect=AssertionError('cache rebuilt')):
                    model.fit(self.X, self.y)
                for key, value in cache.items():
                    self.assertIs(getattr(model, key), value)
                assert_allclose(model.decision_function(self.X), expected, rtol=0, atol=1e-10)

    def test_late_cv_scorer_failure_clears_all_selection_results(self):
        for model, invalid, target in (self.cases()[1], self.cases()[3]):
            with self.subTest(model=type(model).__name__):
                model.fit(self.X, self.y)
                calls = []
                failure = RuntimeError('second fold scorer failure')

                def score(estimator, X, y):
                    calls.append(len(X))
                    if len(calls) == 3:
                        raise failure
                    return estimator.score(X, y)

                model.set_params(scoring=score)
                with self.assertRaises(RuntimeError) as caught:
                    model.fit(self.new_X, self.new_y)
                self.assertIs(caught.exception, failure)
                self.assertEqual(len(calls), 3)
                self.assert_unfitted(model)

    def test_final_full_data_cv_refit_failure_clears_selection_results(self):
        import dwd.gen_dwd as linear_module
        import dwd._kernel_solver as kernel_module

        pairs = [(self.cases()[1][0], 'dwd.gen_dwd.solve_gen_dwd', linear_module.solve_gen_dwd),
                 (self.cases()[3][0], 'dwd._kernel_solver.solve_kernel', kernel_module.solve_kernel)]
        for model, target, original in pairs:
            with self.subTest(model=type(model).__name__):
                model.fit(self.X, self.y)
                sizes = []
                failure = RuntimeError('full-data selected-model refit failed')

                def solve(*args, **kwargs):
                    X = args[0] if args else kwargs['X']
                    sizes.append(len(X))
                    if len(X) == len(self.new_X):
                        raise failure
                    return original(*args, **kwargs)

                with patch(target, side_effect=solve):
                    with self.assertRaises(RuntimeError) as caught:
                        model.fit(self.new_X, self.new_y)
                self.assertIs(caught.exception, failure)
                self.assertEqual(sizes, [6, 6, 12])
                self.assert_unfitted(model)


@unittest.skipIf(cvxpy is None, 'CVXPY is an optional dependency.')
class FailedRefitSOCPTests(_FailureContract, unittest.TestCase):
    def cases(self):
        return [(DWD(C=2., solver_kws={'solver': 'CLARABEL'}),
                 {'C': -1.}, 'dwd.socp_dwd.solve_dwd_socp'),
                (SVM(C=.1, solver_kws={'solver': 'CLARABEL'}),
                 {'C': -1.}, 'dwd.svm.solve_svm')]

    def test_conic_failure_status_clears_model_and_direction(self):
        def fail(problem, **kwargs):
            problem._status = cvxpy.INFEASIBLE

        for model, invalid, target in self.cases():
            with self.subTest(model=type(model).__name__):
                model.fit(self.X, self.y)
                with patch.object(cvxpy.Problem, 'solve', fail):
                    with self.assertRaisesRegex(RuntimeError, 'infeasible'):
                        model.fit(self.new_X, self.new_y)
                self.assert_unfitted(model)
                if isinstance(model, DWD):
                    with self.assertRaises(NotFittedError):
                        _ = model.direction

    def test_auto_C_is_not_left_as_a_fitted_value_after_failure(self):
        model = DWD(C='auto', solver_kws={'solver': 'CLARABEL'}).fit(self.X, self.y)
        with patch('dwd.socp_dwd.solve_dwd_socp', side_effect=RuntimeError('controlled failure')):
            with self.assertRaises(RuntimeError):
                model.fit(self.new_X, self.new_y)
        self.assertEqual(model.C, 'auto')
        self.assert_unfitted(model)

    def test_unsupported_weights_clear_existing_fit(self):
        for model, invalid, target in self.cases():
            with self.subTest(model=type(model).__name__):
                model.fit(self.X, self.y)
                with self.assertRaises(NotImplementedError):
                    model.fit(self.X, self.y, sample_weight=np.ones(len(self.y)))
                self.assert_unfitted(model)


if __name__ == '__main__':
    unittest.main(verbosity=2)
