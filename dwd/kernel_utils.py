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
            return pairwise_kernels(X=self._Xfit,
                                    Y=X,
                                    metric=self.kernel,
                                    **(self.kernel_kws or {}))
        raise ValueError('kernel must be a named kernel, matrix callable, or precomputed.')

    def decision_function(self, X):
        """Predict confidence scores for samples.
        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.
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
        X = check_array(X, dtype='numeric', accept_sparse='csr')
        if self.kernel != 'precomputed' and X.shape[1] != self._Xfit.shape[1]:
            raise ValueError('Feature count differs from training data.')
        K = self._compute_kernel(X)

        scores = safe_sparse_dot(K.T, self.dual_coef_.T,
                                 dense_output=True) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

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
        """Fit KernelCenterer
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
        """Center kernel matrix.
        Parameters
        ----------
        K : numpy array of shape [n_samples1, n_samples2]
            Kernel matrix.
        copy : boolean, optional, default True
            Set to False to perform inplace computation.
        Returns
        -------
        K_new : numpy array of shape [n_samples1, n_samples2]
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
