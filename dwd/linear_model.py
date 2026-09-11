from sklearn.base import ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot

import numpy as np

__all__ = ['LinearClassifierMixin']


class LinearClassifierMixin(ClassifierMixin):
    """
    Shared binary prediction methods for the package's linear classifiers.
    The mixin follows scikit-learn's estimator conventions and requires
    coef_, intercept_, and classes_.
    Positive scores select classes_[1]; zero scores select classes_[0].
    """

    def decision_function(self, X):
        check_is_fitted(self, ['coef_', 'intercept_', 'classes_'])

        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        if X.shape[1] != self.coef_.shape[1]:
            raise ValueError('Feature count differs from training data.')
        return safe_sparse_dot(X, self.coef_.T, dense_output=True).flatten() + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)

        indices = (scores > 0).astype(int)

        return self.classes_[indices]
