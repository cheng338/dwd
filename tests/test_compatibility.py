"""Regression tests for the 1.0.5+compat2 maintenance changes."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from sklearn import config_context
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, cross_validate

import dwd
from dwd.gen_dwd import GenDWD, V_grad, V_grad_
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV
from dwd.kernel_utils import KernelClfMixin

try:
    import cvxpy  # noqa: F401
except ImportError:
    HAS_CVXPY = False
else:
    HAS_CVXPY = True


class CompatibilityTests(unittest.TestCase):
    def test_compatibility_version(self):
        self.assertEqual(dwd.__version__, "2.0.2")

    def test_vectorized_gradient_preserves_fractional_values(self):
        margins = np.array([0, 2], dtype=np.int64)
        actual = V_grad(margins, q=1)
        expected = np.array([V_grad_(margin, q=1) for margin in margins])

        self.assertTrue(np.issubdtype(actual.dtype, np.floating))
        assert_allclose(actual, expected)
        self.assertNotEqual(actual[1], np.trunc(actual[1]))

    def test_binary_prediction_maps_scores_to_original_labels(self):
        class DummyClassifier(KernelClfMixin):
            classes_ = np.array([-7, 9], dtype=np.int64)

            def decision_function(self, X):
                return np.array([-0.5, 0.25])

        predicted = DummyClassifier().predict(np.zeros((2, 1)))
        np.testing.assert_array_equal(predicted, np.array([-7, 9]))

    def test_sklearn_mixins_precede_base_estimator(self):
        for estimator in (KernGDWD, KernGDWDCV):
            with self.subTest(estimator=estimator.__name__):
                mro = estimator.mro()
                self.assertLess(
                    mro.index(KernelClfMixin), mro.index(BaseEstimator)
                )

    @unittest.skipUnless(HAS_CVXPY, "cvxpy is an optional dependency")
    def test_socp_mixin_precedes_base_estimator_when_available(self):
        from dwd.linear_model import LinearClassifierMixin
        from dwd.socp_dwd import DWD

        mro = DWD.mro()
        self.assertLess(mro.index(LinearClassifierMixin), mro.index(BaseEstimator))


def test_routed_cross_validation_without_params():
    # scikit-learn 1.6.0 rejected the default params argument with routing enabled.
    X = np.array([[-3.], [3.], [-2.5], [2.5], [-2.], [2.],
                  [-1.5], [1.5], [-1.], [1.], [-0.5], [0.5]])
    y = np.where(X[:, 0] < 0, "low", "high")
    folds = list(StratifiedKFold(3, shuffle=True, random_state=27).split(X, y))
    model = GenDWD(lambd=0.1, stopping="fixed", max_iter=8, random_state=0)

    predictions = np.empty_like(y)
    scores = []
    for train, test in folds:
        fitted = clone(model).fit(X[train], y[train])
        predictions[test] = fitted.predict(X[test])
        scores.append(np.mean(predictions[test] == y[test]))
    np.testing.assert_array_equal(predictions, y)

    with config_context(enable_metadata_routing=True):
        options = dict(cv=folds, scoring="accuracy", error_score="raise", n_jobs=1)
        np.testing.assert_array_equal(cross_val_score(model, X, y, **options), scores)
        results = cross_validate(model, X, y, **options)
        np.testing.assert_array_equal(results["test_score"], scores)
        np.testing.assert_array_equal(cross_val_predict(model, X, y, cv=folds, n_jobs=1),
                                      predictions)


if __name__ == "__main__":
    unittest.main()
