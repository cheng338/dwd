import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from dwd._fit_state import fit_with_cleanup
from dwd._sparse_features import (canonical_sparse_features, promote_sparse_features,
                                  checked_prediction_basis)

_PREDICTION_BATCH_DEFAULT = object()


class KernelClfMixin(ClassifierMixin):
    """
    Mixin for kernel classifiers.
    """
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = getattr(self, 'kernel', None) == 'precomputed'
        return tags

    # def __init__(self, kernel, kernel_kws={}):
    #     # TODO: kernel centering
    #     self.kernel = kernel
    #     self.kernel_kws = kernel_kws

    def fit(self, X, y, sample_weights=None):

        # the fit function needs to set the following
        self.classes_ = np.unique(y)
        self.__Xfit = X
        self.intercept_ = None
        self.dual_coef_ = None
        raise NotImplementedError

    def _compute_kernel(self, X):
        """

        Parameters
        ----------
        X: array-like, shape (n_samples_test, n_features)
            A matrix of new data

        Returns
        -------
        K: array-like, shape (n_samples_train, n_samples_test)
        """

        # Older serialized models and other users of this mixin may not have
        # a fitted kernel specification. KernGDWD records one during fitting.
        kernel = getattr(self, 'kernel_', self.kernel)
        kernel_kws = getattr(self, 'kernel_kws_', self.kernel_kws)
        if kernel == 'precomputed':
            # Public sklearn convention: queries by training observations.
            # Internally this mixin multiplies K.T by training coefficients.
            X = check_array(X, dtype='numeric', accept_sparse='csr')
            if X.shape[1] != self._Xfit.shape[0]:
                raise ValueError('A precomputed query kernel must have one column per training observation.')
            return X.T

        elif callable(kernel):
            # Matrix-level callable(X_train, X_query, **kernel_kws), matching
            # KernGDWD's documented callback contract.
            K = kernel(self._Xfit, X, **(kernel_kws or {}))
            K = check_array(K, dtype='numeric', accept_sparse='csr')
            if K.shape != (self._Xfit.shape[0], X.shape[0]):
                raise ValueError('A callable kernel must return training-by-query values.')
            return K

        elif isinstance(kernel, str):
            if (kernel == 'rbf' and
                    getattr(self, 'kernel_computation_', None) == 'direct_v1'):
                from ._direct_rbf import accurate_rbf
                gamma = (kernel_kws or {}).get('gamma')
                if isinstance(gamma, bool):
                    gamma = float(gamma)
                return accurate_rbf(self._Xfit, X,
                                    gamma=gamma)
            basis = self._Xfit
            if kernel == 'rbf':
                # Corrected fits own canonical features. Older norm/dot fits
                # with duplicate coordinates may encode an incorrect Gram;
                # normalizing their basis cannot repair fitted coefficients.
                if getattr(self, 'kernel_computation_', None) in (None, 'sklearn', 'norm_sum'):
                    basis = checked_prediction_basis(basis)
                else:
                    basis = canonical_sparse_features(basis)
                X = basis if X is self._Xfit else canonical_sparse_features(X)
            mode = getattr(self, 'kernel_computation_', None)
            if kernel == 'rbf' and mode in ('guarded_sklearn_rel_v1', 'guarded_norm_sum_rel_v1'):
                from ._prediction_norms import standard_kernel_sources_ok
                if standard_kernel_sources_ok():
                    from ._guarded_rbf import guarded_rbf
                    return guarded_rbf(basis, X, gamma=(kernel_kws or {}).get('gamma'),
                                       mode='norm_sum' if mode == 'guarded_norm_sum_rel_v1' else 'sklearn',
                                       self_kernel=X is basis)
            if (kernel == 'rbf' and
                    mode in ('norm_sum', 'guarded_norm_sum_rel_v1')):
                from ._rbf import norm_sum_rbf
                return norm_sum_rbf(
                    basis, X, gamma=(kernel_kws or {}).get('gamma'),
                    self_kernel=X is basis)
            return pairwise_kernels(X=basis,
                                    Y=X,
                                    metric=kernel,
                                    **(kernel_kws or {}))
        raise ValueError('kernel must be a named kernel, matrix callable, or precomputed.')

    def decision_function(self, X, *, prediction_batch_size=_PREDICTION_BATCH_DEFAULT):
        """Predict confidence scores for samples.
        Return the functional decision value K(query, training) @ alpha + b.
        It is not divided by the RKHS norm to obtain a geometric distance.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        prediction_batch_size : positive int or None, optional
            Override the query batch limit for this call without changing the
            estimator. Omission uses its current setting; None disables batching.
        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """

        check_is_fitted(self, ['dual_coef_', 'intercept_', 'classes_', '_Xfit'])
        original_X = X
        X = promote_sparse_features(X, force_float64=getattr(self, 'solver_mode', None) == 'schur')
        X = check_array(X, dtype=np.float64 if getattr(self, 'solver_mode', None) == 'schur'
                        else 'numeric', accept_sparse='csr')
        if getattr(self, 'kernel_', self.kernel) != 'precomputed' and X.shape[1] != self._Xfit.shape[1]:
            raise ValueError('Feature count differs from training data.')
        batch_size = (getattr(self, 'prediction_batch_size', None)
                      if prediction_batch_size is _PREDICTION_BATCH_DEFAULT
                      else prediction_batch_size)
        if batch_size is not None and (isinstance(batch_size, (bool, np.bool_)) or
                not isinstance(batch_size, (int, np.integer)) or batch_size <= 0):
            raise ValueError('prediction_batch_size must be a positive integer or None.')
        if batch_size is None or batch_size >= X.shape[0]:
            K = self._compute_kernel(X)
            scores = self._kernel_decision_product(K)
        else:
            # Bound the query kernel allocation; training coefficients stay fixed.
            scores = np.empty((X.shape[0], self.dual_coef_.shape[0]), dtype=float)
            direct_kernel = None
            if getattr(self, 'kernel_computation_', None) == 'direct_v1':
                # Imported only during prediction, after both classes exist.
                from .gen_kern_dwd import _direct_prediction_kernel
                direct_kernel = _direct_prediction_kernel(self)
            prepared_norm_kernel = None
            if direct_kernel is None:
                from ._prediction_norms import prepare_norm_kernel
                prepared_norm_kernel = prepare_norm_kernel(self, X, original_X)
            for start in range(0, X.shape[0], batch_size):
                stop = min(start + batch_size, X.shape[0])
                query = X[start:stop]
                if direct_kernel is not None:
                    K = direct_kernel(query)
                elif prepared_norm_kernel is not None:
                    K = prepared_norm_kernel(query)
                else:
                    K = self._compute_kernel(query)
                scores[start:stop] = self._kernel_decision_product(K)
                del K
        return scores.ravel() if scores.shape[1] == 1 else scores

    def _kernel_decision_product(self, K):
        """Evaluate the training-by-query kernel using the fitted precision mode."""
        precision = getattr(self, 'prediction_precision_', 'ordinary')
        if getattr(self, 'affine_computation_', 'standard') == 'joint':
            from scipy.sparse import issparse
            from ._affine_scores import affine_scores
            query = K.T
            if issparse(query) and query.format != 'csr':
                query = query.tocsr()
            intercepts = np.broadcast_to(np.asarray(self.intercept_, dtype=float),
                                         (1, len(self.dual_coef_))).reshape(-1)
            policy = 'compensated' if precision == 'compensated' else 'adaptive'
            return np.column_stack([
                affine_scores(query, alpha, intercept, policy=policy)
                for alpha, intercept in zip(self.dual_coef_, intercepts)])
        if precision not in ('compensated', 'adaptive'):
            return safe_sparse_dot(K.T, self.dual_coef_.T,
                                   dense_output=True) + self.intercept_
        from scipy.sparse import issparse
        from ._kernel_scores import compensated_kernel_matvec, adaptive_kernel_matvec
        matvec = compensated_kernel_matvec if precision == 'compensated' else adaptive_kernel_matvec
        query = K.T
        if issparse(query) and query.format != 'csr':
            # A callable may return training-by-query CSR. Conversion stays
            # within the current query batch and is outside the row evaluator.
            query = query.tocsr()
        values = np.column_stack([matvec(query, alpha)
                                  for alpha in self.dual_coef_])
        with np.errstate(over='ignore', invalid='ignore'):
            values += self.intercept_
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite accurate kernel decision values.')
        return values

    def predict(self, X):
        """Predict class labels for samples in X.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]


# Definition-time identities are intentionally independent of mutable class
# attributes, so class-level replacements cannot be mistaken for originals.
_ORIGINAL_COMPUTE_KERNEL = KernelClfMixin._compute_kernel
_ORIGINAL_KERNEL_DECISION_PRODUCT = KernelClfMixin._kernel_decision_product
_ORIGINAL_DECISION_FUNCTION = KernelClfMixin.decision_function


class KernelScaler(TransformerMixin, BaseEstimator):
    """Legacy diagonal scaling of a square training kernel.

    Scales K_ij by n / sqrt(K_ii K_jj), so the resulting diagonal equals n.
    This operation neither centers features nor performs featurewise variance
    standardization. The historical normalization factor n is preserved.
    """

    def __init__(self):
        # Needed for backported inspect.signature compatibility with PyPy
        pass

    @fit_with_cleanup
    def fit(self, K, y=None):
        """Record the positive diagonal of a square training kernel.
        Parameters
        ----------
        K : numpy array of shape [n_samples, n_samples]
            Kernel matrix.
        Returns
        -------
        self : returns an instance of self.
        """
        K = check_array(K, dtype=FLOAT_DTYPES)
        if K.shape[0] != K.shape[1]:
            raise ValueError('KernelScaler requires a square training kernel.')
        self.K_diag_ = np.diag(K).copy()
        if np.any(self.K_diag_ <= 0):
            raise ValueError('KernelScaler requires a strictly positive diagonal.')
        self.n_features_in_ = K.shape[1]
        return self

    def transform(self, K, copy=True):
        """Apply the fitted diagonal scaling to a square kernel matrix.
        Parameters
        ----------
        K : numpy array of shape (n_samples, n_samples)
            Kernel matrix with the same square shape and training-row order as in fit.
        copy : boolean, optional, default True
            Set to False to perform inplace computation.
        Returns
        -------
        K_new : numpy array of shape (n_samples, n_samples)
        """
        check_is_fitted(self, 'K_diag_')

        K = check_array(K, copy=copy, dtype=FLOAT_DTYPES)

        n = len(self.K_diag_)
        if K.shape != (n, n):
            raise ValueError('KernelScaler only supports kernels with the fitted square shape.')
        with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
            s = 1.0 / np.sqrt(self.K_diag_ / n)
            if not np.isfinite(s).all() or np.any(s == 0):
                # Preserve ordinary scaling arithmetic. Only retry when the
                # preliminary division loses a positive subnormal diagonal.
                s = np.sqrt(float(n)) / np.sqrt(self.K_diag_)
            if not np.isfinite(s).all() or np.any(s == 0):
                raise FloatingPointError('KernelScaler produced nonfinite or zero scaling factors.')
            K *= s[None, :]
            K *= s[:, None]
        if not np.isfinite(K).all():
            raise FloatingPointError('KernelScaler produced nonfinite transformed values.')
        return K

    @property
    def _pairwise(self):
        return True
