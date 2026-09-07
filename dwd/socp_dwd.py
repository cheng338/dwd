import logging
from numbers import Real

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_is_fitted

from dwd.utils import pm1
from dwd.linear_model import LinearClassifierMixin

try:
    import cvxpy as cp
except ImportError:
    logging.warning(
        'cvxpy is not installed, but is required for the conic solver.'
    )
    raise

class DWD(LinearClassifierMixin, BaseEstimator):
    """Binary distance-weighted discrimination using optional CVXPY SOCP.

    C remains the conic slack-penalty parameter. For C='auto', the constructor
    value stays unchanged and the fitted value is C_. solver_status_ preserves
    the solver's distinction between optimal and optimal_inaccurate; acceptance
    of the latter is not a claim of an exact numerical certificate.
    """
    def __init__(self, C=1.0, solver_kws=None):
        """
        Parameters
        ----------

        C : Union[float, 'auto']
            Penalty term.
        """
        self.C = C
        self.solver_kws = solver_kws

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target vector relative to X

        sample_weight : None
            Sample weights are unsupported and rejected explicitly.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, accept_sparse='csr', dtype='numeric')
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError('DWD requires exactly two classes.')
        self.n_features_in_ = X.shape[1]

        self.C_ = auto_dwd_C(X, y) if self.C == 'auto' else self.C

        # fit DWD
        self.coef_, self.intercept_, self.eta_, self.d_, self.problem_ = \
            solve_dwd_socp(X, y, C=self.C_,
                           sample_weight=sample_weight,
                           solver_kws=self.solver_kws)

        self.coef_ = self.coef_.reshape(1, -1)
        self.intercept_ = np.atleast_1d(self.intercept_)
        self.solver_status_ = self.problem_.status

        return self

    @property
    def direction(self):
        """
        The separating hyperplane is of the form 'p.d = d.i', where '.' is the dot
        product. If 'p.d < d.i', then 'p' is classified label 0. If 'p.d > d.i' then it
        is classified label 1.

        Returns
        -------
        direction: np.ndarray
            The DWD separating direction; normal to the hyperplane.

        intercept: float
            The intercept of the separating hyperplane.

        the DWD direction and intercept. The separating hyperplane is of the
        form 'p.d = d.i', where '.' is the dot product. If 'p.d < d.i', then 'p' is label
        0. If 'p.d > d.i', then 'p' is label 1.
        """

        check_is_fitted(self, ['coef_', 'intercept_', 'classes_'])
        direction = self.coef_.reshape(-1)
        intercept = -self.intercept_.item()
        return direction, intercept


def solve_dwd_socp(X, y, C=1.0, sample_weight=None, solver_kws=None):
    """
    Solves distance weighted discrimination optimization problem.

    Solves problem (2.7) from https://arxiv.org/pdf/1508.05913.pdf

    Parameters
    ----------
    X: (n_samples, n_features)

    y: (n_samples, )

    C: float
        Strictly positive tuning parameter.

    sample_weight: None
        Sample weights are unsupported and rejected explicitly.

    solver_kws: dict
        Keyword arguments to cp.solve

    Returns
    ------
    beta: (n_features, )
        DWD normal vector.

    intercept: float
        DWD intercept.

    eta, d: float
        Optimization variables.

    problem: cp.Problem

    """

    if not isinstance(C, Real) or not np.isfinite(C) or C <= 0:
        raise ValueError("Penalty term must be positive; got (C={})".format(C))

    # TODO: add sample weights
    if sample_weight is not None:
        raise NotImplementedError

    X, y = check_X_y(X, y,
                     accept_sparse='csr',
                     dtype='numeric')
    if np.unique(y).size != 2:
        raise ValueError('DWD requires exactly two classes.')

    # convert y to +/- 1
    y = pm1(y)

    n_samples, n_features = X.shape

    # problem data
    X = cp.Constant(X)
    y = cp.Constant(y)
    C = cp.Constant(C)

    # optimization variables
    beta = cp.Variable(shape=n_features)
    intercept = cp.Variable()
    eta = cp.Variable(shape=n_samples, nonneg=True)

    rho = cp.Variable(shape=n_samples)
    sigma = cp.Variable(shape=n_samples)

    # Unweighted objective; non-None sample weights were rejected above.
    v = np.ones(n_samples)
    objective = v.T @ (rho + sigma + C * eta)

    # setup constraints
    # TODO: do we need explicit SOCP constraints?
    constraints = [rho - sigma == cp.multiply(y, X @ beta + intercept) + eta,
                   cp.SOC(cp.Parameter(value=1), beta)]  # ||beta||_2^2 <= 1

    # rho^2 - sigma^2 >= 1
    # A vectorized cone creates the same n independent constraints while
    # avoiding n Python constraint objects and a dense n-by-n label diagonal.
    constraints.append(cp.SOC(rho, cp.vstack([sigma, np.ones(n_samples)]), axis=0))

    # solve problem
    problem = cp.Problem(cp.Minimize(objective),
                         constraints=constraints)

    problem.solve(**(solver_kws or {}))
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f'DWD conic solver failed with status {problem.status!r}.')
    if any(value is None or not np.isfinite(value).all()
           for value in (beta.value, intercept.value, eta.value, rho.value, sigma.value)):
        raise FloatingPointError('DWD conic solver returned non-finite coefficients.')

    # d = rho - sigma
    # rho = (1/d + d)/2, sigma = (1/d - d)/2
    d = rho.value - sigma.value

    return beta.value, intercept.value, eta.value, d, problem


def auto_dwd_C(X, y, const=100):
    """
    Automatic choice of C from Distance-Weighted Discrimination by Marron et al, 2007. Note this only is for the SOCP formulation of DWD.

    C = 100 / d ** 2

    Where d is the median distance between points in either class.

    Parameters
    ----------
    X: array-like, (n_samples, n_features)
        The input data.

    y: array-like, (n_samples, )
        The vector of binary class labels.

    const: float
        The constanted used to determine C. Originally suggested to be 100.

    """
    X, y = check_X_y(X, y, accept_sparse='csr', dtype='numeric')
    labels = np.unique(y)
    if len(labels) != 2:
        raise ValueError('Automatic DWD C requires exactly two classes.')
    if not isinstance(const, Real) or not np.isfinite(const) or const <= 0:
        raise ValueError('const must be finite and strictly positive.')

    # pariwise distances between points in each class
    D = euclidean_distances(X[y == labels[0], :],
                            X[y == labels[1], :])

    d = np.median(D.ravel())
    if not np.isfinite(d) or d <= 0:
        raise ValueError('Automatic DWD C requires a positive median between-class distance.')

    return const / d ** 2


def dwd_obj(X, y, C, beta, offset, eta):
    """
    Objective function for DWD.
    """
    d = y * (X.dot(beta) + offset) + eta

    return sum(1.0 / d) + C * sum(eta)
