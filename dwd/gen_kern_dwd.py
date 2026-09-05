import numpy as np
from copy import deepcopy

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_array, check_random_state

from dwd.utils import pm1
from dwd.gen_dwd import V, V_grad
from dwd.kernel_utils import KernelClfMixin
from dwd.cv import run_cv


class KernGDWD(KernelClfMixin, BaseEstimator):
    """
    Kernel Generalized Distance Weighted Discrimination

    Solves the kernel gDWD problem using the MM algorithm derived in Wang and Zou, 2017.

    Primary reference: Another look at distance-weighted discrimination by Boxiang Wang and Hui Zou, 2017

    Note the tuning parameter lambd is on a different scale the parameter C which is used in the SOCP formulation.

    Parameters
    ----------
    lambd: float
        Tuning parameter for DWD.

    q: float
        Tuning parameter for generalized DWD (the exponent on the margin terms). When q = 1, gDWD is equivalent to DWD.

    kernel: str, callable(X, Y, \*\*kwargs)
        The kernel to use.

    kernel_kws: dict
        Any key word arguments for the kernel.

    implicit_P: bool
        Use the implicit inverse-product solver. False remains unsupported for
        kernel DWD; use the independent small-system tests as a reference.

    solver_mode: {'legacy', 'schur'}
        'legacy' preserves the released update for reproducibility, including
        its known algebra errors. 'schur' is the corrected published MM update.
        Neither mode regularizes the intercept. Scientific comparisons must
        identify the mode explicitly; old accuracy is not guaranteed to persist.

    max_iter, obj_tol, random_state:
        Maximum MM steps, absolute successive-objective stopping tolerance,
        and initialization seed. Objective tolerance is not a stationarity proof.
    """

    def __init__(self, lambd=1.0, q=1.0, kernel='linear',
                 kernel_kws=None, implicit_P=True, max_iter=100,
                 obj_tol=1e-5, random_state=None, solver_mode='legacy'):
        self.lambd = lambd
        self.q = q

        self.kernel = kernel
        self.kernel_kws = kernel_kws

        self.implicit_P = implicit_P
        self.max_iter = max_iter
        self.obj_tol = obj_tol
        self.random_state = random_state
        self.solver_mode = solver_mode

    def fit(self, X, y, sample_weight=None, *, K=None, K_eig=None,
            alpha_init=None, offset_init=None):
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
        # Explicit precomputation belongs to this exact training fold and
        # kernel. Reusing a decomposition from a full dataset leaks information.
        X, y = check_X_y(X, y, accept_sparse='csr',
                         dtype=np.float64 if self.solver_mode == 'schur' else 'numeric')
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError('KernGDWD requires exactly two classes.')
        self.n_features_in_ = X.shape[1]
        self._Xfit = X  # Store K so we can compute predictions

        if K is None:
            if self._cv_cache_matches(X):
                K = self._cv_K
                if K_eig is None:
                    K_eig = self._K_eig
            else:
                K = self._compute_kernel(X)
        if np.shape(K) != (X.shape[0], X.shape[0]):
            raise ValueError('K must be the square training kernel for X.')
        # Normalize the public array-like argument for both solve and diagnostics.
        K = check_array(K, accept_sparse=False,
                        dtype=np.float64 if self.solver_mode == 'schur' else 'numeric')

        # fit DWD
        alpha, offset, obj_vals, c = \
            solve_gen_kern_dwd(K=K,
                               y=y,
                               lambd=self.lambd,
                               q=self.q,
                               alpha_init=alpha_init,
                               offset_init=offset_init,
                               sample_weight=sample_weight,
                               implicit_P=self.implicit_P,
                               obj_tol=self.obj_tol, max_iter=self.max_iter,
                               K_eig=K_eig, random_state=self.random_state,
                               solver_mode=self.solver_mode)

        self.intercept_ = np.asarray(offset).reshape(-1)
        self.dual_coef_ = alpha.reshape(1, -1)
        self.obj_vals_ = obj_vals
        self.objective_history_ = np.asarray(obj_vals)
        self.n_iter_ = len(obj_vals) - 1
        self.final_objective_ = float(obj_vals[-1])
        self.converged_ = bool(len(obj_vals) > 1 and
                               np.isfinite(obj_vals[-1]) and
                               abs(obj_vals[-1] - obj_vals[-2]) < self.obj_tol)
        self.termination_reason_ = ('objective_tolerance' if self.converged_
                                    else 'max_iter')
        if not np.all(np.isfinite(obj_vals)):
            self.termination_reason_ = 'nonfinite_objective'
        self.C_ = c
        signed_y = pm1(y)
        K_alpha = K.dot(alpha)
        z = signed_y * V_grad(signed_y * (K_alpha + offset), q=self.q) / len(y)
        gradient_alpha = K.dot(z) + 2 * self.lambd * K_alpha
        self.gradient_inf_norm_ = float(max(abs(z.sum()),
                                            np.max(np.abs(gradient_alpha))))

        return self

    def cv_init(self, X):
        """
        Initializes the object before computing a cross-valiation.
        """
        # Promote before constructing the Gram matrix, not after rounding it.
        X = check_array(X, accept_sparse='csr',
                        dtype=np.float64 if self.solver_mode == 'schur' else 'numeric')
        self._Xfit = X

        # Warning: we compute the kernel twice -- any way around this
        # without messing up the SKlearn API too badly?
        K = self._compute_kernel(X)
        self._set_K_eig(K)
        self._cv_K = K
        self._cv_X = X.copy()
        self._cv_kernel = self.kernel
        self._cv_kernel_kws = deepcopy(self.kernel_kws)
        self._cv_solver_mode = self.solver_mode
        return self

    def _cv_cache_matches(self, X):
        """Invalidate hidden precomputation whenever data or kernel changes."""
        if not hasattr(self, '_cv_X'):
            return False
        if (self.kernel != self._cv_kernel or
                self.kernel_kws != self._cv_kernel_kws or
                self.solver_mode != self._cv_solver_mode or
                X.dtype != self._cv_X.dtype or
                X.shape != self._cv_X.shape):
            return False
        if hasattr(X, 'toarray') or hasattr(self._cv_X, 'toarray'):
            if not (hasattr(X, 'toarray') and hasattr(self._cv_X, 'toarray')):
                return False
            return (X != self._cv_X).nnz == 0
        return np.array_equal(X, self._cv_X)

    def _set_K_eig(self, X):
        """
        Precomputes eigen decomposition of K matrix which makes
        cross-validation much faster.
        """
        self._K_eig = get_K_eig(X)

    def _get_K_eig(self):
        if hasattr(self, '_K_eig'):
            return self._K_eig
        else:
            return None


class KernGDWDCV(KernelClfMixin, BaseEstimator):
    """
    Fits kernel gDWD with cross-validation. gDWD cross-validation
    can be significnatly faster if certain quantities are precomputed.

    Parameters
    ----------
    lambd_vals: list of floats
        The lambda values to cross-validate over.

    q_vals: list of floats
        The q-values to cross validate over.

    kernel: str, callable
        The kernel to use.

    kern_kws_vals: list of dicts
        The kernel parameters to validate over.

    cv:
        How to perform cross-valdiation. See documetnation in sklearn.model_selection.GridSearchCV.

    scoring:
        What metric to use to score cross-validation. See documetnation in sklearn.model_selection.GridSearchCV.

    """
    def __init__(self,
                 lambd_vals=np.logspace(-2, 2, 10),
                 q_vals=np.logspace(-2, 2, 5),
                 kernel='linear',
                 kernel_kws_vals=None,
                 cv=5, scoring='accuracy', max_iter=100, obj_tol=1e-5,
                 random_state=None, solver_mode='legacy'):

        self.lambd_vals = lambd_vals
        self.q_vals = q_vals
        self.kernel = kernel
        self.kernel_kws_vals = kernel_kws_vals

        self.cv = cv
        self.scoring = scoring
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
            raise NotImplementedError('Sample weights are not implemented for KernGDWDCV.')
        X, y = check_X_y(X, y, accept_sparse='csr',
                         dtype='numeric')

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError('KernGDWDCV requires exactly two classes.')
        self.n_features_in_ = X.shape[1]

        # run cross validation
        params = {'q': self.q_vals, 'lambd': self.lambd_vals,
                  'kernel_kws': {} if self.kernel_kws_vals is None else self.kernel_kws_vals}

        best_params, best_score, best_clf, agg_results, all_cv_results = \
            run_cv(clf=KernGDWD(kernel=self.kernel, max_iter=self.max_iter,
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

        self._Xfit = self.best_estimator_._Xfit
        self.intercept_ = self.best_estimator_.intercept_
        self.dual_coef_ = self.best_estimator_.dual_coef_

        self.decision_function = self.best_estimator_.decision_function

        return self


def solve_gen_kern_dwd(K, y, lambd, q=1,
                       alpha_init=None, offset_init=None,
                       sample_weight=None,
                       implicit_P=True,
                       obj_tol=1e-5, max_iter=100,
                       K_eig=None, random_state=None, solver_mode='legacy'):

    """
    Solves the kernel gDWD problem using the MM algorithm derived in Wang and Zou, 2017.

    Parameters
    ----------
    K: array-like, (n_samples, n_samples)
        The kernel.

    y: array-like, (n_samples, )
        The vector of binary class labels.

    lambd: float
        Tuning parameter for DWD.

    q: float
        Tuning parameter for generalized DWD (the exponent on the margin terms). When q = 1, gDWD is equivalent to DWD.

    alpha_init, offset_init:
        Initial values to start the optimization algorithm from.

    sample_weight: None, array-like (n_samples,)
        Optional weight for samples.

    implicit_P: bool
        Whether to use the implicit P^{-1} gamma formulation (in the publication) or the explicit computation (in the arxiv version).

    obj_tol: float
        Stopping condition for difference between successive objective
        functions.

    max_iter: int
        Maximum number of iterations to perform.

    K_eig: None or (U, D)
        Optional. Trusted precomputed eigendecomposition of this exact K.
        Shapes and finiteness are checked; callers are responsible for matching
        K and an orthonormal basis. A full O(n^3) reconstruction check is not
        repeated for each regularization candidate.
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
    if solver_mode == 'schur' and lambd <= 0:
        raise ValueError("solver_mode='schur' requires lambd > 0.")

    # TODO: add sample weights
    if sample_weight is not None:
        raise NotImplementedError

    K, y = check_X_y(K, y, accept_sparse=False,
                     dtype=np.float64 if solver_mode == 'schur' else 'numeric')

    if K.shape[0] != K.shape[1]:
        raise ValueError('K must be square.')
    # Block the symmetry check to avoid another large N-by-N temporary.
    for first in range(0, len(K), 128):
        if not np.allclose(K[first:first+128], K[:, first:first+128].T,
                           rtol=1e-10, atol=1e-12):
            raise ValueError('K must be symmetric.')

    # convert y to +/- 1
    y = pm1(y)  # convert y to  y +/- 1

    n_samples = K.shape[0]
    M = (q + 1) ** 2 / q

    # precompute data
    if implicit_P:

        # precompute data needed to do implicit P^{-1} gamma
        if K_eig is not None:
            U, Lam = K_eig
            U, Lam = np.asarray(U), np.asarray(Lam).ravel()
            if U.shape != (n_samples, n_samples) or len(Lam) != n_samples:
                raise ValueError('K_eig dimensions must match K.')
            if not np.isfinite(U).all() or not np.isfinite(Lam).all():
                raise ValueError('K_eig must contain finite values.')
        else:
            U, Lam = get_K_eig(K)

        # see section 4.1 of Wang and Zou, 2017 for details
        pi = Lam ** 2 + (2 * n_samples * lambd / M) * Lam

        Ucs = U.sum(axis=0)
        if solver_mode == 'legacy':
            # Preserve the released solver's trajectory for comparability.
            ULP = np.multiply(U, Lam * (1.0 / pi))
            v = ULP.dot(U.T.dot(np.ones(n_samples)))
            carson = Ucs.dot(np.multiply(ULP, Lam).dot(Ucs))
            g = 1.0 / (n_samples - carson)
        else:
            # Section 4.1 of Wang and Zou: cancel Lam in Lam/pi, then
            # evaluate the Schur denominator without subtracting near equals.
            shift = 2 * n_samples * lambd / M
            negative_tolerance = 100 * np.finfo(float).eps * n_samples * max(1., np.max(np.abs(Lam)))
            if np.min(Lam) < -negative_tolerance:
                raise ValueError('K must be positive semidefinite, not an indefinite kernel. '
                                 'If precomputed in float32, recompute K from float64 features; '
                                 'casting an already rounded Gram matrix cannot restore precision.')
            # Do not silently truncate positive eigenvalues or add a ridge.
            # On nearly singular kernels the canceled inverse can generate
            # huge non-identifiable coefficients whose quadratic norm loses
            # all accuracy. This is a conservative implementation boundary,
            # not a guarantee of accuracy for every matrix below the bound.
            spectral_scale = max(0., float(np.max(Lam)))
            if (np.min(Lam) <= negative_tolerance and
                    (spectral_scale + shift) / shift > 1.0 / np.sqrt(np.finfo(float).eps)):
                raise FloatingPointError('Rank-deficient kernel with regularization too small for stable '
                                         'coefficient arithmetic. Increase lambd or use a separately '
                                         'validated range-space solver; no kernel was modified.')
            shifted_eigenvalues = Lam + shift
            if np.any(shifted_eigenvalues <= 0):
                raise ValueError('K + (2*n*lambd/M)*I must be positive definite.')
            inverse_shifted = 1.0 / shifted_eigenvalues
            ULP = np.multiply(U, inverse_shifted)
            v = ULP.dot(Ucs)
            g = 1.0 / (shift * np.dot(Ucs * Ucs, inverse_shifted))

    else:
        raise NotImplementedError('Kernel DWD supports only implicit_P=True.')

    # initialize variables
    if alpha_init is None:
        alpha = check_random_state(random_state).normal(size=n_samples)
        alpha /= np.linalg.norm(alpha)
    else:
        alpha = np.asarray(alpha_init, dtype=float).copy()
        if alpha.shape != (n_samples,) or not np.isfinite(alpha).all():
            raise ValueError('alpha_init must be a finite vector with one coefficient per sample.')

    if offset_init is None:
        offset = 0.0
    else:
        if np.ndim(offset_init) != 0 or not np.isfinite(offset_init):
            raise ValueError('offset_init must be a finite scalar.')
        offset = float(offset_init)

    # store objective values
    obj_vals = []
    K_alpha = K.dot(alpha)
    prev_obj = kern_dwd_obj(K, y, q, lambd, alpha, offset, K_alpha=K_alpha)
    if not np.isfinite(prev_obj):
        raise FloatingPointError('Kernel DWD produced a nonfinite initial objective.')
    obj_vals.append(prev_obj)

    for i in range(max_iter):

        # get step
        alpha_step, offset_step = \
            get_step_implicit_P(K=K, y=y, q=q, lambd=lambd,
                                alpha=alpha, offset=offset,
                                U=U, Lam=Lam, pi=pi, v=v, g=g,
                                ULP=ULP, K_alpha=K_alpha,
                                solver_mode=solver_mode)

        # update parameters
        alpha = alpha - alpha_step
        offset = offset - offset_step
        K_alpha = K.dot(alpha)

        # stopping condition
        current_obj = kern_dwd_obj(K, y, q, lambd, alpha, offset,
                                   K_alpha=K_alpha)
        if not np.isfinite(current_obj):
            raise FloatingPointError('Kernel DWD produced a nonfinite objective.')
        obj_vals.append(current_obj)

        if np.abs(current_obj - prev_obj) < obj_tol:
            break
        else:
            prev_obj = current_obj

    # tuning paramter for SOCP formulation
    c = c_from_lambd(K, lambd, q, alpha, K_alpha=K_alpha)
    if not np.isfinite(c):
        raise FloatingPointError('Kernel DWD produced a nonfinite SOCP-parameter conversion.')

    return alpha, offset, obj_vals, c


def get_K_eig(K):
    """
    Computes the eigendecomposition of K.

    Parameters
    ----------
    K: (n_samples, n_samples)
        Kernel matrix.

    Output
    ------
    U, D

    U: (n_features, n_features)
        Eigenvectors of K.

    D: (n_features, )
        Eigenvalues of K in decending order.
    """

    Lam, U = np.linalg.eigh(K)
    Lam = Lam[::-1]  # sort evals in decending order
    # A reversed-column view has negative strides. BLAS would copy that entire
    # O(n^2) array at every matrix-vector product; materialize it once here.
    U = np.asfortranarray(U[:, ::-1])

    return U, Lam


def get_step_implicit_P(K, y, q, lambd, alpha, offset, U, Lam, pi, v, g,
                        ULP=None, K_alpha=None, solver_mode='legacy'):
    """
    Computes the step size for one step of the MM algorithm.
    See Wang and Zou, 2017 for details.
    """
    n_samples = K.shape[0]

    if K_alpha is None:
        K_alpha = K.dot(alpha)
    z = y * V_grad(y * (K_alpha + offset), q=q) / n_samples

    alice = z + 2 * lambd * alpha

    if solver_mode == 'schur':
        # K v = 1 - shift*v. Cancel shift analytically in the intercept
        # numerator/denominator to avoid subtracting near-equals at tiny lambda.
        factor = n_samples * q / ((q + 1) ** 2)
        shift = 2 * lambd * factor
        h = (v.dot(z) - (1.0 / factor) * (1.0 - shift * v).dot(alpha)) / v.sum()
        if ULP is None:
            ULP = np.multiply(U, 1.0 / (Lam + shift))
        offset_step = factor * h
        alpha_step = factor * ULP.dot(U.T.dot(alice)) - v * offset_step
        return alpha_step, offset_step

    bob = np.insert(-v, 0, values=1)
    projection = z if solver_mode == 'legacy' else v
    bob = (sum(z) - projection.T.dot(K.dot(alice))) * bob

    if ULP is None:
        if solver_mode == 'legacy':
            ULP = np.multiply(U, Lam * (1.0 / pi))
        else:
            ULP = np.multiply(U, 1.0 / (Lam + 2 * n_samples * lambd * q /
                                        ((q + 1) ** 2)))
    cathy = ULP.dot(U.T.dot(alice))
    cathy = np.insert(cathy, 0, 0)

    P_inv_gamma = g * bob + cathy

    step = (n_samples * q / ((q + 1) ** 2)) * P_inv_gamma

    offset_step = step[0]
    alpha_step = step[1:]

    return alpha_step, offset_step


def get_step_explicit_P(K, y, q, lambd, alpha, offset, P_inv):
    """
    Computes the step size for one step of the MM algorithm.
    See Wang and Zou, 2017 for details.
    """
    n_samples = K.shape[0]

    z = y * V_grad(y * (K.dot(alpha) + offset), q=q) / n_samples

    gamma = K.dot(z) + 2 * lambd * K.dot(alpha)
    gamma = np.insert(gamma, 0, z.sum())

    step = (n_samples * q / (q + 1) ** 2) * P_inv @ gamma

    offset_step = step[0]
    alpha_step = step[1:]

    return alpha_step, offset_step


def kern_dwd_obj(K, y, q, lambd, alpha, offset, K_alpha=None):
    """
    Objective function for kernel DWD.
    """
    if K_alpha is None:
        K_alpha = K.dot(alpha)
    return np.mean(V(y * (K_alpha + offset), q=q) +
                   lambd * alpha.T.dot(K_alpha))


def c_from_lambd(K, lambd, q, alpha, K_alpha=None):
    """
    Gets the tuning paramter, C, for the SOCP formulation of DWD
    from lambda.
    """
    if K_alpha is None:
        K_alpha = K.dot(alpha)
    beta_norm = np.sqrt(alpha.T.dot(K_alpha))
    return ((q + 1) ** (q + 1) / q ** q) * beta_norm ** (q + 1)
