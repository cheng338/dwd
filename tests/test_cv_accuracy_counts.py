"""Accuracy comparisons preserve fold weighting and actual integer ties."""
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from dwd.cv import _accuracy_correct, _mean_accuracy, run_cv


class CountClassifier(ClassifierMixin, BaseEstimator):
    predict_calls = 0
    fit_calls = 0

    def __init__(self, candidate=0, patterns=((0, 1, 5), (1, 2, 3))):
        self.candidate = candidate
        self.patterns = patterns

    def cv_init(self, X):
        return self

    def fit(self, X, y):
        type(self).fit_calls += 1
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        type(self).predict_calls += 1
        block, position = X[:, 0].astype(int), X[:, 1].astype(int)
        truth = np.where(position % 2, 1, -1)
        correct = np.array(self.patterns[self.candidate])[block]
        return np.where(position < correct, truth, -truth)


class AccuracyCountsTests(unittest.TestCase):
    def run_case(self, sizes, patterns, scoring='accuracy'):
        X = np.array([(block, row) for block, size in enumerate(sizes)
                      for row in range(size)], dtype=float)
        y = np.where(X[:, 1].astype(int) % 2, 1, -1)
        folds = [(np.flatnonzero(X[:, 0] != block), np.flatnonzero(X[:, 0] == block))
                 for block in range(len(sizes))]
        CountClassifier.fit_calls = CountClassifier.predict_calls = 0
        return run_cv(CountClassifier(patterns=patterns), X, y,
                      {'candidate': list(range(len(patterns)))}, cv=folds,
                      scoring=scoring, refit_best=False)

    def test_rounding_does_not_split_an_actual_accuracy_tie(self):
        patterns = ((0, 1, 5), (1, 2, 3))
        self.assertLess(np.mean(np.array(patterns[0]) / 10),
                        np.mean(np.array(patterns[1]) / 10))
        best, score, _, aggregate, folds = self.run_case((10, 10, 10), patterns)
        self.assertEqual(best, {'candidate': 0})
        self.assertEqual(score, .2)
        self.assertEqual(aggregate['mean_test_score_fraction'], [[1, 5], [1, 5]])
        self.assertEqual(aggregate['mean_test_score'], [.2, .2])
        self.assertEqual([fold['test_correct'] for fold in folds], [[0, 1], [1, 2], [5, 3]])
        self.assertEqual(CountClassifier.fit_calls, 6)
        self.assertEqual(CountClassifier.predict_calls, 12)

    def test_unequal_fold_sizes_keep_equal_fold_weighting(self):
        # Candidate 0 wins the mean of folds; candidate 1 wins pooled accuracy.
        best, score, _, aggregate, _ = self.run_case((4, 8), ((4, 0), (0, 6)))
        self.assertEqual(best, {'candidate': 0})
        self.assertEqual(score, .5)
        self.assertEqual(aggregate['mean_test_score_fraction'], [[1, 2], [3, 8]])

    def test_one_additional_correct_prediction_is_not_a_tie(self):
        best, _, _, aggregate, _ = self.run_case((10, 10, 10), ((0, 1, 5), (0, 1, 6)))
        self.assertEqual(best, {'candidate': 1})
        self.assertEqual(aggregate['mean_test_score_fraction'], [[1, 5], [7, 30]])

    def test_arbitrary_callable_scorer_retains_scalar_behavior(self):
        def custom(model, X, y):
            return .20000000000000004 if model.candidate else .19999999999999998
        best, _, _, aggregate, folds = self.run_case((10, 10, 10), ((0, 1, 5), (1, 2, 3)), custom)
        self.assertEqual(best, {'candidate': 1})
        self.assertNotIn('mean_test_score_fraction', aggregate)
        self.assertNotIn('test_correct', folds[0])
        self.assertEqual(CountClassifier.predict_calls, 0)

    def test_count_reconstruction_is_exact_or_rejected(self):
        for count in (3, 7, 1200, 2**30, 2**52):
            for correct in (0, 1, count // 2, count - 1, count):
                self.assertEqual(_accuracy_correct(correct / count, count), correct)
        for score, count in ((.1, 3), (.5, True), (.5, 0), (.5, 2**53)):
            with self.assertRaises(ValueError):
                _accuracy_correct(score, count)
        self.assertEqual(_mean_accuracy([1, 2], [3, 7]), Fraction(13, 42))
        with self.assertRaises(ValueError):
            _mean_accuracy([True], [1])


class CompactDiagnosticTests(unittest.TestCase):
    def test_missing_facts_stay_unknown_and_arrays_are_not_copied(self):
        examples = str(Path(__file__).resolve().parents[1] / 'examples')
        if examples not in sys.path:
            sys.path.insert(0, examples)
        from resumable_kernel_cv import _diagnostics
        model = SimpleNamespace(max_iter=3, criterion_reached_=True,
                                optimality_met_=False, objective_tolerance_met_=True,
                                final_objective_=.2, diagnostics_={
                                    'setup_seconds': 1., 'history': np.ones(1000),
                                    'dual_certificate': {
                                        'fallback_reason': 'tiny_asymmetry',
                                        'kernel_assumption': 'nonnegative represented RKHS penalty',
                                        'primal_objective_and_gap_are_estimates': True},
                                    'linear_system_diagnostics': {'linear_recoveries': 0,
                                                                  'coefficients': np.ones(1000)},
                                    'mm_function_recovery': {
                                        'attempted': True, 'accepted_actions': 1,
                                        'certificate_status': 'certified',
                                        'kernel_approximation_used': False,
                                        'positive_directions_discarded': 0,
                                        'true_mm_function_checked': True,
                                        'state': np.ones(1000)}})
        observed = _diagnostics(model)
        self.assertTrue(observed['criterion_reached'])
        self.assertFalse(observed['optimality_met'])
        self.assertIsNone(observed['converged'])
        self.assertIsNone(observed['budget_exhausted'])
        self.assertIsNone(observed['total_iterations'])
        self.assertEqual(observed['final_objective'], .2)
        self.assertEqual(observed['solver_summary']['linear_system_diagnostics'], {'linear_recoveries': 0})
        self.assertEqual(observed['solver_summary']['summary'], {'setup_seconds': 1.})
        self.assertTrue(observed['solver_summary']['dual_certificate']['primal_objective_and_gap_are_estimates'])
        self.assertEqual(observed['solver_summary']['mm_function_recovery'], {
            'attempted': True, 'accepted_actions': 1, 'certificate_status': 'certified',
            'kernel_approximation_used': False, 'positive_directions_discarded': 0,
            'true_mm_function_checked': True})


if __name__ == '__main__':
    unittest.main(verbosity=2)
