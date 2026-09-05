import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_array, check_random_state

from dwd.utils import pm1
from dwd.cv import run_cv
from dwd.linear_model import LinearClassifierMixin


class GenDWD(LinearClassifierMixin, BaseEstimator):
    """
    Generalized Distance Weighted Discrimination

    Solves the gDWD problem using the MM algorithm derived in Wang and Zou, 2017.

    Primary reference: Another look at distance-weighted discrimination by Boxiang Wang and Hui Zou, 2017

    Note the tuning parameter lambd is on a different scale the parameter C which is used in the SOCP formulation.

    Parameters
    ----------
    lambd: float
        Tuning parameter for DWD.

    q: float
        Tuning parameter for generalized DWD (the exponent on the margin terms). When q = 1, gDWD is equivalent to DWD.

    implicit_P: bool
        Whether to use the implicit P^{-1} gamma formulation (in the publication) or the explicit computation (in the arxiv version).

    max_iter: int, default=100
        Maximum number of MM updates.

    obj_tol: float, default=1e-5
        Absolute tolerance on successive objective values.

    random_state: int, RandomState or None, default=None
        Initialization seed. None retains NumPy's global random state.

    solver_mode: {'legacy', 'schur'}, default='legacy'
        'legacy' preserves the released implicit solver. 'schur' corrects
        its Sherman--Morrison coefficient, which can change fitted models.
        This choice has no effect when implicit_P=False.

    """
    def __init__(self, lambd=1.0, q=1, implicit_P=True, max_iter=100,
                 obj_tol=1e-5, random_state=None, solver_mode='legacy'):
        self.lambd = lambd
        self.q = q
        self.implicit_P = implicit_P
        self.max_iter = max_iter
        self.obj_tol = obj_tol
        self.random_state = random_state
        self.solver_mode = solver_mode

    def fit(self, X, y, sample_weight=None, *, P0_eig=None,
            beta_init=None, offset_init=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target vector relative to X

        sample_weight : array-like, shape = [n_samples], optional
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.

        P0_eig : tuple of arrays, optional
            Eigensystem returned by get_P0_eig(X), for this exact training
            data and feature order. This explicit cache is trusted after
            shape/finite validation; callers must not mix folds or datasets.

        beta_init, offset_init : optional
            Explicit initial coefficients and intercept.

        Returns
        -------
        self : object
        """
        # TODO: what to do about multi-class

        # formatting
        # TODO: figure out what we should actually check
        X, y = check_X_y(X, y, accept_sparse='csr',
                         dtype=np.float64 if self.solver_mode == 'schur' else 'numeric')

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError('GenDWD requires exactly two classes.')
        self.n_features_in_ = X.shape[1]

        if P0_eig is None:
            P0_eig = self._get_P0_eig(X)

        # fit DWD
        self.coef_, self.intercept_, self.obj_vals_, self.C_ = \
            solve_gen_dwd(X=X, y=y,
                          lambd=self.lambd, q=self.q,
                          sample_weight=sample_weight,
                          P0_eig=P0_eig,
                          beta_init=beta_init, offset_init=offset_init,
                          implicit_P=self.implicit_P,
                          obj_tol=self.obj_tol, max_iter=self.max_iter,
                          random_state=self.random_state,
                          solver_mode=self.solver_mode)

        self.coef_ = self.coef_.reshape(1, -1)
        self.intercept_ = np.asarray(self.intercept_).reshape(-1)
        self.objective_history_ = np.asarray(self.obj_vals_)
        self.n_iter_ = len(self.obj_vals_) - 1
        self.final_objective_ = float(self.obj_vals_[-1])
        self.converged_ = bool(self.n_iter_ > 0 and
                               np.isfinite(self.final_objective_) and
                               abs(self.obj_vals_[-1] - self.obj_vals_[-2]) < self.obj_tol)
        self.termination_reason_ = ('objective_tolerance' if self.converged_
                                    else 'max_iter')
        if not np.all(np.isfinite(self.obj_vals_)):
            self.termination_reason_ = 'nonfinite_objective'
        beta = self.coef_.ravel()
        signed_y = pm1(y)
        z = signed_y * V_grad(signed_y * (X.dot(beta) + self.intercept_[0]),
                              q=self.q) / len(y)
        gradient_beta = X.T.dot(z) + 2 * self.lambd * beta
        self.gradient_inf_norm_ = float(max(abs(z.sum()),
                                            np.max(np.abs(gradient_beta))))

        return self

    def cv_init(self, X):
        """
        Initializes the object before computing a cross-valiation.
        """
        X = check_array(X, accept_sparse='csr',
                        dtype=np.float64 if self.solver_mode == 'schur' else 'numeric')
        self._set_P0_eig(X)
        return self

    def _set_P0_eig(self, X):
        """
        Precomputes eigen decomposition of P0 matrix which makes
        cross-validation much faster.
        """
        self._P0_eig = get_P0_eig(X)
        self._P0_X = X.copy()
        self._P0_solver_mode = self.solver_mode

    def _cv_cache_matches(self, X):
        """The linear precomputation is valid only for the same training data."""
        if (not hasattr(self, '_P0_X') or X.shape != self._P0_X.shape or
                X.dtype != self._P0_X.dtype or
                self.solver_mode != self._P0_solver_mode):
            return False
        if hasattr(X, 'toarray') or hasattr(self._P0_X, 'toarray'):
            if not (hasattr(X, 'toarray') and hasattr(self._P0_X, 'toarray')):
                return False
            return (X != self._P0_X).nnz == 0
        return np.array_equal(X, self._P0_X)

    def _get_P0_eig(self, X=None):
        if hasattr(self, '_P0_eig') and (X is None or self._cv_cache_matches(X)):
            return self._P0_eig
        else:
            return None


class GenDWDCV(LinearClassifierMixin, BaseEstimator):
    """
    Fits Genralized DWD with cross-validation. gDWD cross-validation
    can be significnatly faster if certain quantities are precomputed.

    Parameters
    ----------
    lambd_vals: list of floats
        The lambda values to cross-validate over.

    q_vals: list of floats
        The q-values to cross validate over.

    cv:
        How to perform cross-valdiation. See documetnation in sklearn.model_selection.GridSearchCV.

    scoring:
        What metric to use to score cross-validation. See documetnation in sklearn.model_selection.GridSearchCV.

    """
    def __init__(self,
                 lambd_vals=np.logspace(-2, 2, 10),
                 q_vals=np.logspace(-2, 2, 5),
                 cv=5, scoring='accuracy', implicit_P=True, max_iter=100,
                 obj_tol=1e-5, random_state=None, solver_mode='legacy'):
        self.lambd_vals = lambd_vals
        self.q_vals = q_vals

        self.cv = cv
        self.scoring = scoring
        self.implicit_P = implicit_P
        self.max_iter = max_iter
        self.obj_tol = obj_tol
        self.random_state = random_state
        self.solver_mode = solver_mode

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target vector relative to X

        sample_weight : array-like, shape = [n_samples], optional
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.
        Returns
        -------
        self : object
        """
        if sample_weight is not None:
            raise NotImplementedError('Sample weights are not implemented for GenDWDCV.')

        # formatting
        # TODO: figure out what we should actually check
        X, y = check_X_y(X, y, accept_sparse='csr',
                         dtype='numeric')

        self.classes_ = np.unique(y)

        # run cross validation
        params = {'q': self.q_vals, 'lambd': self.lambd_vals}
        best_params, best_score, best_clf, agg_results, all_cv_results = \
            run_cv(clf=GenDWD(implicit_P=self.implicit_P, max_iter=self.max_iter,
                              obj_tol=self.obj_tol, random_state=self.random_state,
                              solver_mode=self.solver_mode),
                   X=X, y=y,
                   params=params,
                   scoring=self.scoring,
                   cv=self.cv,
                   refit_best=True)

        self.best_estimator_ = best_clf

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.agg_cv_results_ = agg_results
        self.all_cv_results_ = all_cv_results

        self.coef_ = self.best_estimator_.coef_
        self.intercept_ = self.best_estimator_.intercept_
        self.n_features_in_ = self.best_estimator_.n_features_in_

        return self


def solve_gen_dwd(X, y, lambd, q=1,
                  sample_weight=None,
                  beta_init=None, offset_init=None,
                  implicit_P=True,
                  obj_tol=1e-5, max_iter=100,
                  P0_eig=None, random_state=None, solver_mode='legacy'):

    """
    Solves linear gDWD using the MM algorithm derived in Wang and Zou, 2017.

    Parameters
    ----------
    X: array-like, (n_samples, n_features)
        Input X data.

    y: array-like, (n_samples, )
        The vector of binary class labels.

    lambd: float
        Tuning parameter for DWD.

    q: float
        Tuning parameter for generalized DWD (the exponent on the margin terms). When q = 1, gDWD is equivalent to DWD.

    beta_init, offset_init:
        Initial values to start the optimization algorithm from.

    sample_weight: None, array-like (n_samples,)
        Optional weight for samples.

    implicit_P: bool
        Whether to implicitly calculate P^{-1} gamma. If P0_eig is precomputed
        this is much faster.

    obj_tol: float
        Stopping condition for difference between successive objective
        functions.

    max_iter: int
        Maximum number of iterations to perform.

    P0_eig: None, tuple of (U, D)
        Precomputed eigenvectors and eigenvalues of P0. Optional.

    random_state: int, RandomState or None
        Seed for initialization when beta_init is not supplied.

    solver_mode: {'legacy', 'schur'}
        Select the released or corrected Sherman--Morrison coefficient
        when implicit_P=True. Explicit inversion does not use this mode.
    """

    # argument checking and formatting
    if not np.isfinite(lambd) or lambd < 0:
        raise ValueError("Penalty term must be positive; got (lambd=%r)"
                         % lambd)

    if not np.isfinite(q) or q <= 0:
        raise ValueError("Weight term must be positive; got (q=%r)" % q)
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError('max_iter must be a nonnegative integer.')
    if not np.isfinite(obj_tol) or obj_tol < 0:
        raise ValueError('obj_tol must be finite and nonnegative.')
    if solver_mode not in ('legacy', 'schur'):
        raise ValueError("solver_mode must be 'legacy' or 'schur'.")

    # TODO: add sample weights
    if sample_weight is not None:
        raise NotImplementedError('Sample weights are not implemented for GenDWD.')

    X, y = check_X_y(X, y,
                     accept_sparse='csr',
                     dtype=np.float64 if solver_mode == 'schur' else 'numeric')
    if len(np.unique(y)) != 2:
        raise ValueError('GenDWD requires exactly two classes.')

    # convert y to +/- 1
    y = pm1(y)

    n_samples, n_features = X.shape
    M = (q + 1) ** 2 / q

    # initialize variables
    if beta_init is None:
        beta = check_random_state(random_state).normal(size=n_features)
    else:
        beta = np.asarray(beta_init, dtype=float).copy()
        if beta.shape != (n_features,) or not np.all(np.isfinite(beta)):
            raise ValueError('beta_init must be a finite vector with one entry per feature.')

    if offset_init is None:
        offset = 0.0
    else:
        if np.ndim(offset_init) != 0 or not np.isfinite(offset_init):
            raise ValueError('offset_init must be a finite scalar.')
        offset = float(offset_init)

    # precompute data
    if implicit_P:
        # data needed to implicitly calculate P_inv @ gamma
        if P0_eig is not None:
            U, D = P0_eig
            U, D = np.asarray(U), np.asarray(D).ravel()
            if U.shape != (n_features + 1, n_features + 1) or len(D) != n_features + 1:
                raise ValueError('P0_eig dimensions must match the augmented feature count.')
            if not (np.all(np.isfinite(U)) and np.all(np.isfinite(D))):
                raise ValueError('P0_eig must contain finite values.')
        else:
            U, D = get_P0_eig(X)

        pi = D + 2 * n_samples * lambd / M
        if np.any(pi <= 0):
            raise ValueError('The shifted augmented Gram matrix must be positive definite.')

        u1 = U[0, :]
        UP = np.multiply(U, 1.0 / pi)
        v = UP.dot(u1)
        if solver_mode == 'legacy':
            g = 2 * n_samples * q * lambd / \
                ((q + 1) ** 2 + 2 * n_samples * lambd * sum(v))
        else:
            shift = 2 * n_samples * lambd / M
            # P = (P0 + shift*I) - shift*e0*e0.T. Sherman--Morrison
            # leaves the intercept unpenalized; the released coefficient
            # incorrectly used sum(v) and a plus sign in its denominator.
            denominator = 1.0 - shift * v[0]
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError('The Sherman--Morrison denominator must be positive.')
            g = shift / denominator

    else:
        # explicitly create P_inv matrix
        col_sums = np.asarray(X.sum(axis=0)).reshape(-1, 1)
        gram = X.T.dot(X)
        if hasattr(gram, 'toarray'):
            gram = gram.toarray()
        d = gram + (2 * n_samples * lambd / M) * np.eye(n_features)
        P = np.block([[np.array([[n_samples]]), col_sums.T],
                      [col_sums, d]])
        P_inv = np.linalg.inv(P)

    # store objective values
    obj_vals = []
    X_beta = X.dot(beta)
    prev_obj = dwd_obj(X, y, q, lambd, beta, offset, X_beta=X_beta)
    if not np.isfinite(prev_obj):
        raise FloatingPointError('GenDWD produced a nonfinite initial objective.')
    obj_vals.append(prev_obj)

    for i in range(max_iter):

        # get step
        if implicit_P:
            beta_step, offset_step = \
                get_step_implicit(X, y, beta, offset, lambd, q,
                                  U, v, g, pi, UP=UP, X_beta=X_beta)
        else:
            beta_step, offset_step = \
                get_step_explicit(X, y, beta, offset, lambd, q, P_inv,
                                  X_beta=X_beta)

        offset = offset - offset_step  # step[0]
        beta = beta - beta_step  # step[1:]
        X_beta = X.dot(beta)

        # stopping condition
        current_obj = dwd_obj(X, y, q, lambd, beta, offset, X_beta=X_beta)
        if not np.isfinite(current_obj):
            raise FloatingPointError('GenDWD produced a nonfinite objective.')
        obj_vals.append(current_obj)

        if np.abs(current_obj - prev_obj) < obj_tol:
            break
        else:
            prev_obj = current_obj

    c = c_from_lambd(lambd, q, beta)
    if not np.isfinite(c):
        raise FloatingPointError('GenDWD produced a nonfinite SOCP-parameter conversion.')

    return beta, offset, obj_vals, c


def get_P0_eig(X):
    """
    Equation (3.6)

    Parameters
    ----------
    X: (n_samples, n_features)

    Output
    ------
    U, D

    U: (n_features, n_features)
        Eigenvectors of P0.

    D: (n_features, )
        Eigenvalues of P0 in decending order.
    """
    n = np.array([[X.shape[0]]])
    colsum = np.asarray(X.sum(axis=0)).reshape(-1, 1)
    gram = X.T.dot(X)
    if hasattr(gram, 'toarray'):
        gram = gram.toarray()

    # create P0
    P0 = [[n, colsum.T],
          [colsum, gram]]  # (d+1 x d+1)
    P0 = np.block(P0)

    # compute eigen decomp of P0
    D, U = np.linalg.eigh(P0)
    D = D[::-1]  # sort evals in decending order
    U = U[:, ::-1]

    return U, D


def get_step_implicit(X, y, beta, offset, lambd, q, U, v, g, pi,
                      UP=None, X_beta=None):
    """
    Gets the MM step by implicitly calculating P^{-1} gamma
    """
    n_samples = X.shape[0]

    # compute update
    if X_beta is None:
        X_beta = X.dot(beta)
    z = y * V_grad(y * (X_beta + offset), q=q) / n_samples

    gamma = X.T.dot(z) + 2 * lambd * beta
    gamma = np.insert(gamma, 0, z.sum())

    # implictly compute P_inv @ gamma

    if UP is None:
        UP = np.multiply(U, 1.0 / pi)
    P_inv_gamma = UP.dot(U.T.dot(gamma)) + \
        g * v * v.T.dot(gamma)

    # compute step
    step = (n_samples * q / (q + 1) ** 2) * P_inv_gamma

    offset_step = step[0]
    beta_step = step[1:]

    return beta_step, offset_step


def get_step_explicit(X, y, beta, offset, lambd, q, P_inv, X_beta=None):
    """
    Gets the MM step by explicitly calculating P^{-1} gamma
    """
    n_samples = X.shape[0]

    # compute update
    if X_beta is None:
        X_beta = X.dot(beta)
    z = y * V_grad(y * (X_beta + offset), q=q) / n_samples

    gamma = X.T.dot(z) + 2 * lambd * beta
    gamma = np.insert(gamma, 0, z.sum())

    step = (n_samples * q / (q + 1) ** 2) * P_inv @ gamma

    offset_step = step[0]
    beta_step = step[1:]

    return beta_step, offset_step


def V_(u, q=1):
    """
    DWD loss function
    """
    if u <= q / (q + 1.0):
        return 1 - u
    else:
        return (1.0 / u ** q) * (q ** q / (q + 1) ** (q + 1))


def V(u, q=1):
    """Evaluate the unchanged piecewise loss using NumPy array operations.

    Masked evaluation avoids computing fractional powers of negative margins.
    The floating result also avoids ``np.vectorize`` inferring integer output.
    """
    u = np.asarray(u)
    u = u.astype(np.result_type(u.dtype, np.float64), copy=False)
    result = np.array(1.0 - u, dtype=np.result_type(u.dtype, np.float64))
    mask = ~(u <= q / (q + 1.0))
    result[mask] = (1.0 / u[mask] ** q) * (q ** q / (q + 1) ** (q + 1))
    return result


def V_grad_(u, q=1):
    """
    DWD loss function gradient
    """
    # TODO: check
    if u <= q / (q + 1.0):
        return -1
    else:
        return - (q / (u * (q + 1))) ** (q + 1)
        # return - (1.0 / u ** (q + 1)) * (q / (q + 1)) ** (q + 1)


def V_grad(u, q=1):
    """Evaluate the unchanged piecewise gradient without Python scalar loops."""
    u = np.asarray(u)
    u = u.astype(np.result_type(u.dtype, np.float64), copy=False)
    result = np.full(u.shape, -1.0, dtype=np.result_type(u.dtype, np.float64))
    mask = ~(u <= q / (q + 1.0))
    result[mask] = -(q / (u[mask] * (q + 1))) ** (q + 1)
    return result


def dwd_obj(X, y, q, lambd, beta, offset, X_beta=None):
    """
    DWD objective function.
    """
    if X_beta is None:
        X_beta = X.dot(beta)
    return np.mean(V(y * (X_beta + offset), q=q) + lambd * beta.dot(beta))


def c_from_lambd(lambd, q, beta):
    """
    Gets the tuning paramter, C, for the SOCP formulation of DWD
    from lambda.
    """
    return ((q + 1) ** (q + 1) / q ** q) * np.linalg.norm(beta) ** (q + 1)
