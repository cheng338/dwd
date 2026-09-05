import cvxpy as cp
import numpy as np
from numbers import Real

from sklearn.base import BaseEstimator  # , TransformerMixin, ClassifierMixin

from dwd.linear_model import LinearClassifierMixin
from dwd.utils import pm1
from sklearn.utils import check_X_y


def solve_svm(X, y, C, sample_weight=None, solver_kws=None):
    """
    Solves soft-margin SVM problem.

    min_{beta, intercept}
    (1/n) * sum_{i=1}^n max(0, 1 - y_i * (x^T beta + intercept)) + C * ||beta||_1

    This is the package's L1-regularized mean-hinge model, not SVC's L2
    objective. Its parameter C is a regularization multiplier.

    Parameters
    ----------
    X: (n_samples, n_features)

    y: (n_samples, )

    C: float
        Strictly positive tuning parameter.

    sample_weight: None, (n_samples, )
        Weights for samples.

    solver_kws: dict
        Keyword arguments to cp.solve

    Output
    ------
    beta, intercept, problem

    beta: (n_features, )
        SVM normal vector.

    intercept: float
        SVM intercept.

    problem: cp.Problem

    y_hat = np.sign(x.dot(beta) + intercept)
    """
    if sample_weight is not None:
        raise NotImplementedError
    if not isinstance(C, Real) or not np.isfinite(C) or C <= 0:
        raise ValueError('C must be finite and strictly positive.')
    X, y = check_X_y(X, y, accept_sparse='csr', dtype='numeric')
    if np.unique(y).size != 2:
        raise ValueError('SVM requires exactly two classes.')
    y = pm1(y)

    n_samples, n_features = X.shape
    y = y.reshape(-1, 1)

    beta = cp.Variable((n_features, 1))
    intercept = cp.Variable()
    C = cp.Parameter(value=C, nonneg=True)

    # TODO: should we make this + intercept
    loss = cp.sum(cp.pos(1 - cp.multiply(y, X @ beta + intercept)))
    reg = cp.norm(beta, 1)
    objective = loss / n_samples + C * reg

    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(**(solver_kws or {}))
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f'SVM conic solver failed with status {problem.status!r}.')
    if any(value is None or not np.isfinite(value).all()
           for value in (beta.value, intercept.value)):
        raise FloatingPointError('SVM conic solver returned non-finite coefficients.')

    return beta.value, intercept.value, problem


class SVM(LinearClassifierMixin, BaseEstimator):

    def __init__(self, C=1.0, solver_kws=None):
        self.C = C
        self.solver_kws = solver_kws

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse='csr', dtype='numeric')
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.coef_, self.intercept_, self.problem_ = \
            solve_svm(X, y, C=self.C, sample_weight=sample_weight,
                      solver_kws=self.solver_kws)

        self.coef_ = self.coef_.reshape(1, -1)
        self.intercept_ = np.atleast_1d(self.intercept_)

        return self
