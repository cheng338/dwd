import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


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

        if self.kernel == 'precomputed':
            # Public sklearn convention: queries by training observations.
            # Internally this mixin multiplies K.T by training coefficients.
            X = check_array(X, dtype='numeric', accept_sparse='csr')
            if X.shape[1] != self._Xfit.shape[0]:
                raise ValueError('A precomputed query kernel must have one column per training observation.')
            return X.T

        elif callable(self.kernel):
            # Matrix-level callable(X_train, X_query, **kernel_kws), matching
            # KernGDWD's documented callback contract.
            K = self.kernel(self._Xfit, X, **(self.kernel_kws or {}))
            K = check_array(K, dtype='numeric', accept_sparse='csr')
            if K.shape != (self._Xfit.shape[0], X.shape[0]):
                raise ValueError('A callable kernel must return training-by-query values.')
            return K

        elif isinstance(self.kernel, str):
            if (self.kernel == 'rbf' and
                    getattr(self, 'kernel_computation_', None) == 'norm_sum'):
                from ._rbf import norm_sum_rbf
                return norm_sum_rbf(
                    self._Xfit, X, gamma=(self.kernel_kws or {}).get('gamma'),
                    self_kernel=X is self._Xfit)
            return pairwise_kernels(X=self._Xfit,
                                    Y=X,
                                    metric=self.kernel,
                                    **(self.kernel_kws or {}))
        raise ValueError('kernel must be a named kernel, matrix callable, or precomputed.')

    def decision_function(self, X):
        """Predict confidence scores for samples.
        Return the functional decision value K(query, training) @ alpha + b.
        It is not divided by the RKHS norm to obtain a geometric distance.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """

        check_is_fitted(self, ['dual_coef_', 'intercept_', 'classes_', '_Xfit'])
        X = check_array(X, dtype=np.float64 if getattr(self, 'solver_mode', None) == 'schur'
                        else 'numeric', accept_sparse='csr')
        if self.kernel != 'precomputed' and X.shape[1] != self._Xfit.shape[1]:
            raise ValueError('Feature count differs from training data.')
        batch_size = getattr(self, 'prediction_batch_size', None)
        if batch_size is not None and (isinstance(batch_size, (bool, np.bool_)) or
                not isinstance(batch_size, (int, np.integer)) or batch_size <= 0):
            raise ValueError('prediction_batch_size must be a positive integer or None.')
        if batch_size is None or batch_size >= X.shape[0]:
            K = self._compute_kernel(X)
            scores = self._kernel_decision_product(K)
        else:
            # Bound the query kernel allocation; training coefficients stay fixed.
            scores = np.empty((X.shape[0], self.dual_coef_.shape[0]), dtype=float)
            for start in range(0, X.shape[0], batch_size):
                stop = min(start + batch_size, X.shape[0])
                K = self._compute_kernel(X[start:stop])
                scores[start:stop] = self._kernel_decision_product(K)
        return scores.ravel() if scores.shape[1] == 1 else scores

    def _kernel_decision_product(self, K):
        """Evaluate the training-by-query kernel using the fitted precision mode."""
        precision = getattr(self, 'prediction_precision_', 'ordinary')
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


class KernelScaler(TransformerMixin, BaseEstimator):
    """Legacy diagonal scaling of a square training kernel.

    Scales K_ij by n / sqrt(K_ii K_jj), so the resulting diagonal equals n.
    This operation neither centers features nor performs featurewise variance
    standardization. The historical normalization factor n is preserved.
    """

    def __init__(self):
        # Needed for backported inspect.signature compatibility with PyPy
        pass

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
        s = 1.0 / np.sqrt(self.K_diag_ / n)

        K *= s[None, :]
        K *= s[:, None]
        return K

    @property
    def _pairwise(self):
        return True
