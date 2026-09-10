import numpy as np
from dwd._eigen import validated_eigh
from dwd._fit_state import fit_with_cleanup
from numbers import Real

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
        Seed used for random initialization. Zero initialization ignores it.

    solver_mode: {'legacy', 'schur'}, default='schur'
        'schur' uses the corrected Sherman--Morrison coefficient. 'legacy'
        retains known incorrect implicit update algebra for reproducing old
        experiments. Explicit inversion uses the correct system in either mode.

    initialization: {'auto', 'zero', 'random'}, default='auto'
        'auto' selects zero for corrected mode and random for legacy mode.
        Explicit beta_init and offset_init supplied to fit take precedence.

    stopping: {'objective', 'fixed', 'optimality'}, default='objective'
        Stop at the original absolute objective-change threshold, at max_iter,
        or when the full primal gradient meets tol, respectively. No validation
        data or tuning is performed by fit.

    tol: float, default=1e-6
        Absolute stationarity tolerance: max(abs(intercept gradient),
        Euclidean norm of coefficient gradient). Also checked on the returned
        model for every stopping policy; converged_ reports this check.

    Notes
    -----
    The objective is mean DWD loss plus lambd times the squared coefficient
    norm; the intercept is unregularized. An objective-change stop is not a
    stationarity certificate. Inspect termination_reason_ and the final
    stationarity diagnostics separately. The auxiliary norm-to-C conversion
    can overflow for large q even when the fit is finite; C_ is then inf and
    C_conversion_finite_ is false. This does not change the fitted classifier.

    """
    def __init__(self, lambd=1.0, q=1, implicit_P=True, max_iter=100,
                 obj_tol=1e-5, random_state=None, solver_mode='schur',
                 initialization='auto', stopping='objective', tol=1e-6):
        self.lambd = lambd
        self.q = q
        self.implicit_P = implicit_P
        self.max_iter = max_iter
        self.obj_tol = obj_tol
        self.random_state = random_state
        self.solver_mode = solver_mode
        self.initialization = initialization
        self.stopping = stopping
        self.tol = tol

    @fit_with_cleanup
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

        sample_weight : None
            Sample weights are unsupported; a non-None value raises
            NotImplementedError.

        P0_eig : tuple of arrays, optional
            Eigensystem returned by get_P0_eig(X), for this exact training
            data and feature order. Supplied pairs are checked for numerical
            consistency with that augmented Gram matrix; invalid pairs reject.

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
                          solver_mode=self.solver_mode,
                          initialization=self.initialization,
                          stopping=self.stopping, tol=self.tol)

        self.coef_ = self.coef_.reshape(1, -1)
        self.intercept_ = np.asarray(self.intercept_).reshape(-1)
        self.objective_history_ = np.asarray(self.obj_vals_)
        self.n_iter_ = len(self.obj_vals_) - 1
        self.returned_iteration_ = self.n_iter_
        self.final_objective_ = float(self.obj_vals_[-1])
        self.C_conversion_finite_ = bool(np.isfinite(self.C_))
        self.objective_tolerance_met_ = bool(self.n_iter_ > 0 and
                               np.isfinite(self.final_objective_) and
                               abs(self.obj_vals_[-1] - self.obj_vals_[-2]) < self.obj_tol)
        beta = self.coef_.ravel()
        signed_y = pm1(y)
        gradient_offset, gradient_beta, residual = _linear_gradient(
            X, signed_y, self.q, self.lambd, beta, self.intercept_[0])
        self.gradient_inf_norm_ = float(max(abs(gradient_offset),
                                            np.max(np.abs(gradient_beta))))
        self.gradient_norm_ = residual
        self.stationarity_residual_ = residual
        self.stationarity_checked_ = True
        self.converged_ = bool(np.isfinite(residual) and residual <= self.tol)
        self.optimality_met_ = self.converged_
        self.termination_reason_ = 'max_iter'
        if self.stopping == 'optimality' and self.converged_:
            self.termination_reason_ = 'optimality'
        elif self.stopping == 'objective' and self.objective_tolerance_met_:
            self.termination_reason_ = 'objective_tolerance'
        self.criterion_reached_ = self.termination_reason_ in ('optimality', 'objective_tolerance')
        self.initialization_ = ('zero' if self.solver_mode == 'schur' else 'random') \
            if self.initialization == 'auto' else self.initialization
        if beta_init is not None:
            self.initialization_ = 'explicit'

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
        # Compare the feature representation used by corrected fit/cv_init.
        # Otherwise float32 paths rebuild the identical float64 P0 per lambda.
        if self.solver_mode == 'schur' and X.dtype != np.dtype(np.float64):
            X = X.astype(np.float64, copy=False)
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
                 obj_tol=1e-5, random_state=None, solver_mode='schur',
                 initialization='auto', stopping='objective', tol=1e-6):
        self.lambd_vals = lambd_vals
        self.q_vals = q_vals

        self.cv = cv
        self.scoring = scoring
        self.implicit_P = implicit_P
        self.max_iter = max_iter
        self.obj_tol = obj_tol
        self.random_state = random_state
        self.solver_mode = solver_mode
        self.initialization = initialization
        self.stopping = stopping
        self.tol = tol

    @fit_with_cleanup
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
            Sample weights are unsupported; a non-None value raises
            NotImplementedError.
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
                              solver_mode=self.solver_mode,
                              initialization=self.initialization,
                              stopping=self.stopping, tol=self.tol),
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
                  P0_eig=None, random_state=None, solver_mode='schur',
                  initialization='auto', stopping='objective', tol=1e-6):

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

    sample_weight: None
        Sample weights are unsupported and rejected explicitly.

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

    initialization: {'auto', 'zero', 'random'}
        Used when beta_init is absent. 'auto' means zero in corrected mode
        and random in legacy mode. The default intercept is always zero.

    stopping: {'objective', 'fixed', 'optimality'}
        Select the stopping condition; max_iter remains a hard cap.

    tol: float
        Absolute tolerance for the full primal stationarity residual.
    """

    # argument checking and formatting
    if (not isinstance(lambd, Real) or isinstance(lambd, (bool, np.bool_))
            or not np.isfinite(lambd) or lambd < 0):
        raise ValueError("Penalty term must be finite and nonnegative; got (lambd=%r)"
                         % lambd)

    _validate_q(q)
    if (not isinstance(max_iter, (int, np.integer))
            or isinstance(max_iter, (bool, np.bool_)) or max_iter < 0):
        raise ValueError('max_iter must be a nonnegative integer.')
    if (not isinstance(obj_tol, Real) or isinstance(obj_tol, (bool, np.bool_))
            or not np.isfinite(obj_tol) or obj_tol < 0):
        raise ValueError('obj_tol must be finite and nonnegative.')
    if solver_mode not in ('legacy', 'schur'):
        raise ValueError("solver_mode must be 'legacy' or 'schur'.")
    if initialization not in ('auto', 'zero', 'random'):
        raise ValueError("initialization must be 'auto', 'zero', or 'random'.")
    if stopping not in ('objective', 'fixed', 'optimality'):
        raise ValueError("stopping must be 'objective', 'fixed', or 'optimality'.")
    if (not isinstance(tol, Real) or isinstance(tol, (bool, np.bool_))
            or not np.isfinite(tol) or tol < 0):
        raise ValueError('tol must be finite and nonnegative.')

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
    M = _majorization_constant(q)

    # initialize variables
    if beta_init is None:
        use_zero = initialization == 'zero' or (
            initialization == 'auto' and solver_mode == 'schur')
        beta = (np.zeros(n_features) if use_zero else
                check_random_state(random_state).normal(size=n_features))
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
            D, U = validated_eigh(_augmented_gram(X), supplied=P0_eig)
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
        if stopping == 'optimality':
            if _linear_gradient(X, y, q, lambd, beta, offset, X_beta)[2] <= tol:
                break

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

        if stopping == 'objective' and np.abs(current_obj - prev_obj) < obj_tol:
            break
        else:
            prev_obj = current_obj

    c = c_from_lambd(lambd, q, beta)
    if np.isnan(c) or c < 0:
        raise FloatingPointError('GenDWD produced an invalid SOCP-parameter conversion.')

    return beta, offset, obj_vals, c


def _augmented_gram(X):
    """Build the unchanged free-intercept Gram matrix in float64 arithmetic."""
    X = check_array(X, accept_sparse='csr', dtype=np.float64)
    n = np.array([[float(X.shape[0])]])
    colsum = np.asarray(X.sum(axis=0)).reshape(-1, 1)
    gram = X.T.dot(X)
    if hasattr(gram, 'toarray'):
        gram = gram.toarray()
    return np.block([[n, colsum.T], [colsum, gram]])


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
    D, U = validated_eigh(_augmented_gram(X))
    D = D[::-1]  # sort evals in decending order
    U = np.asfortranarray(U[:, ::-1])

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
    step = (n_samples / _majorization_constant(q)) * P_inv_gamma

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

    step = (n_samples / _majorization_constant(q)) * P_inv @ gamma

    offset_step = step[0]
    beta_step = step[1:]

    return beta_step, offset_step


def _validate_q(q):
    """Validate the shared public loss exponent domain without coercing it."""
    if (not isinstance(q, Real) or isinstance(q, (bool, np.bool_))
            or not np.isfinite(q) or q <= 0):
        raise ValueError('q must be a finite, strictly positive real scalar.')


def _majorization_constant(q):
    _validate_q(q)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        value = np.float64(q) + 2.0 + 1.0 / np.float64(q)
    if not np.isfinite(value):
        raise ValueError('q produces a nonfinite MM curvature bound.')
    return float(value)


def _linear_gradient(X, y, q, lambd, beta, offset, X_beta=None):
    if X_beta is None:
        X_beta = X.dot(beta)
    z = y * V_grad(y * (X_beta + offset), q=q) / len(y)
    grad_b = float(z.sum())
    grad_beta = np.asarray(X.T.dot(z)).ravel() + 2 * lambd * beta
    if not np.isfinite(grad_b) or not np.isfinite(grad_beta).all():
        raise FloatingPointError('GenDWD produced a nonfinite objective gradient.')
    residual = float(max(abs(grad_b), np.linalg.norm(grad_beta)))
    if not np.isfinite(residual):
        raise FloatingPointError('GenDWD produced a nonfinite stationarity residual.')
    return grad_b, grad_beta, residual


def V_(u, q=1):
    """
    DWD loss function
    """
    _validate_q(q)
    if u <= q / (q + 1.0):
        return 1 - u
    else:
        return ((q / (q + 1.0)) / u) ** q / (q + 1.0)


def V(u, q=1):
    """Evaluate the unchanged piecewise loss using NumPy array operations.

    Masked evaluation avoids computing fractional powers of negative margins.
    The floating result also avoids ``np.vectorize`` inferring integer output.
    q must be a finite positive real scalar. NaN margins propagate; finite
    training data are validated by the estimators. Extreme exponents can still
    exceed a solver's curvature or coefficient-conversion precision range.
    """
    _validate_q(q)
    u = np.asarray(u)
    u = u.astype(np.result_type(u.dtype, np.float64), copy=False)
    result = np.array(1.0 - u, dtype=np.result_type(u.dtype, np.float64))
    mask = ~(u <= q / (q + 1.0))
    result[mask] = ((q / (q + 1.0)) / u[mask]) ** q / (q + 1.0)
    return result


def V_grad_(u, q=1):
    """
    DWD loss function gradient
    """
    _validate_q(q)
    if u <= q / (q + 1.0):
        return -1
    else:
        return - ((q / (q + 1.0)) / u) ** (q + 1.0)
        # return - (1.0 / u ** (q + 1)) * (q / (q + 1)) ** (q + 1)


def V_grad(u, q=1):
    """Evaluate the unchanged piecewise gradient without Python scalar loops."""
    _validate_q(q)
    u = np.asarray(u)
    u = u.astype(np.result_type(u.dtype, np.float64), copy=False)
    result = np.full(u.shape, -1.0, dtype=np.result_type(u.dtype, np.float64))
    mask = ~(u <= q / (q + 1.0))
    result[mask] = -((q / (q + 1.0)) / u[mask]) ** (q + 1.0)
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
    Return the auxiliary norm-based C conversion with the original scaling.

    Log arithmetic avoids overflow in intermediate q powers. Zero norm gives
    zero; a conversion larger than floating-point range is reported as inf.
    Neither result is a usable finite positive SOCP penalty. lambd remains in
    the public signature for compatibility; the formula depends on fitted
    norm and q, and is not a new choice of the fitting regularization.
    """
    _validate_q(q)
    norm = np.linalg.norm(beta)
    if norm == 0:
        return 0.0
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        log_c = (np.log1p(q) + q * np.log1p(1.0 / q)
                 + (q + 1.0) * np.log(norm))
        return float(np.exp(log_c))
