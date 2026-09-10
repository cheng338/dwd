import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y

from dwd.kernel_utils import KernelClfMixin
from dwd._fit_state import fit_with_cleanup


class KernMD(KernelClfMixin, BaseEstimator):
    """Binary RKHS mean-difference classifier with a midpoint intercept.

    ``naive_bayes=True`` is unsupported until a consistent transformation for
    both training and query kernels is specified. ``kernel_kws`` applies to
    named kernels or matrix-level callable kernels.
    """
    def __init__(self, kernel='linear', kernel_kws=None, naive_bayes=False):
        self.kernel = kernel
        self.kernel_kws = kernel_kws

        self.naive_bayes = naive_bayes

    @fit_with_cleanup
    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype='numeric')
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._Xfit = X  # Store K so we can compute predictions

        K = self._compute_kernel(X)

        self.dual_coef_, self.intercept_ = \
            kern_md(K, y, naive_bayes=self.naive_bayes)

        self.intercept_ = np.atleast_1d(self.intercept_)
        self.dual_coef_ = self.dual_coef_.reshape(1, -1)

        return self


def kern_md(K, y, naive_bayes=False):
    """
    Returns the coefficients for the kernel mean difference
    classifier.

    Parameters
    ----------
    K: array-like (n_samples, n_samples)
        Kernel matrix

    y: array-like (n_samples, )
        Vector of binary labels.

    naive_bayes: bool
        Compute naive bayes direction.

    Output
    ------
    alpha, intercept

    """
    K = check_array(K, dtype='numeric')
    y = np.asarray(y)
    labels = np.unique(y)
    if K.shape[0] != K.shape[1] or y.ndim != 1 or len(y) != K.shape[0]:
        raise ValueError('K must be square with one row per label.')
    if len(labels) != 2:
        raise ValueError('Kernel mean difference requires exactly two classes.')

    # Training-only normalization would give inconsistent query scores.
    if naive_bayes:
        raise NotImplementedError(
            'naive_bayes=True previously transformed only the training kernel '
            'and produced inconsistent query scores. Use naive_bayes=False '
            'until a train/query-consistent formulation is specified.')

    pos_ind = y == labels[1]
    neg_ind = y == labels[0]
    n_pos = sum(pos_ind)
    n_neg = sum(neg_ind)

    y_tilde = (pos_ind / n_pos) - (neg_ind / n_neg)
    # The RKHS normal is mean(phi(X_pos)) - mean(phi(X_neg)). Multiplying
    # these coefficients by K again would erroneously square the kernel action.
    alpha = y_tilde

    intercept = (1.0 / n_pos ** 2) * pos_ind.T.dot(K.dot(pos_ind)) - \
        (1.0 / n_neg ** 2) * neg_ind.T.dot(K.dot(neg_ind))

    return alpha, -0.5 * intercept
