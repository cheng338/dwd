"""Scorer-boundary controls; manufactured estimators do not optimize models."""
import unittest
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from dwd.cv import run_cv


class MetadataClassifier(ClassifierMixin, BaseEstimator):
    fits = []

    def __init__(self, candidate=0):
        self.candidate = candidate

    def cv_init(self, X):
        return self

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.fitted_X_ = X.copy()
        type(self).fits.append((self.candidate, len(y)))
        return self

    def predict(self, X):
        return np.repeat(self.classes_[0], len(X))


class CVScoreValidationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(16., dtype=np.float64).reshape(8, 2)
        self.y = np.tile([-1, 1], 4)
        self.folds = [(np.arange(4), np.arange(4, 8)),
                      (np.arange(4, 8), np.arange(4))]
        MetadataClassifier.fits = []

    def run_cv(self, scorer, candidates=(0, 1)):
        return run_cv(MetadataClassifier(), self.X, self.y,
                      {'candidate': list(candidates)}, scoring=scorer,
                      cv=self.folds, refit_best=True)

    def check_invalid(self, value, context):
        MetadataClassifier.fits = []

        def scorer(model, X, y):
            is_train = np.array_equal(X, model.fitted_X_)
            selected_context = is_train if context == 'train' else not is_train
            return value if model.candidate == 1 and selected_context else .5

        with self.assertRaisesRegex(
                ValueError, "candidate \\{'candidate': 1\\}, fold 0, " + context + ' score'):
            self.run_cv(scorer)
        # Candidate 0 completed its fold; candidate 1 failed during scoring.
        # Neither later folds nor the full-data refit should run.
        self.assertEqual(MetadataClassifier.fits, [(0, 4), (1, 4)])

    def test_nan_and_both_infinities_reject_before_selection_or_refit(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            for context in ('train', 'test'):
                with self.subTest(value=value, context=context):
                    self.check_invalid(value, context)

    def test_nonscalar_nonreal_and_nonnumeric_scores_reject_with_context(self):
        for value in ([.5], np.array([.5]), np.array([[.5]]), .5+0j, '.5', None):
            for context in ('train', 'test'):
                with self.subTest(value=repr(value), context=context):
                    self.check_invalid(value, context)

    def test_finite_scores_keep_winner_first_tie_means_and_refit(self):
        for cast in (float, np.float32, np.float64, lambda value: np.array(value)):
            with self.subTest(cast=repr(cast)):
                MetadataClassifier.fits = []

                def scorer(model, X, y):
                    return cast((-.5, .75, .75)[model.candidate])

                best, score, fitted, aggregate, folds = self.run_cv(scorer, (0, 1, 2))
                self.assertEqual(best, {'candidate': 1})
                self.assertEqual(score, .75)
                self.assertEqual(fitted.candidate, 1)
                self.assertEqual(aggregate['mean_test_score'], [-.5, .75, .75])
                self.assertEqual(aggregate['mean_train_score'], [-.5, .75, .75])
                self.assertEqual(aggregate['std_test_score'], [0., 0., 0.])
                self.assertEqual([row['test_score'] for row in folds],
                                 [[-.5, .75, .75], [-.5, .75, .75]])
                self.assertEqual(MetadataClassifier.fits[-1], (1, 8))
                self.assertEqual(len(MetadataClassifier.fits), 7)

    def test_default_accuracy_scorer_is_unchanged(self):
        best, score, fitted, aggregate, _ = self.run_cv('accuracy')
        self.assertEqual(best, {'candidate': 0})
        self.assertEqual(score, .5)
        self.assertEqual(aggregate['mean_test_score'], [.5, .5])
        self.assertEqual(MetadataClassifier.fits[-1], (0, 8))
        self.assertEqual(fitted.candidate, 0)

    def test_nonfinite_mean_of_finite_fold_scores_cannot_refit(self):
        def scorer(model, X, y):
            return np.finfo(float).max

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with self.assertRaisesRegex(ValueError, 'mean test_score across folds'):
                self.run_cv(scorer, (0,))
        self.assertEqual(MetadataClassifier.fits, [(0, 4), (0, 4)])

    def test_scorer_exception_is_not_hidden_or_reclassified(self):
        failure = RuntimeError('manufactured scorer failure')

        def scorer(model, X, y):
            raise failure

        with self.assertRaises(RuntimeError) as caught:
            self.run_cv(scorer)
        self.assertIs(caught.exception, failure)
        self.assertEqual(MetadataClassifier.fits, [(0, 4)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
