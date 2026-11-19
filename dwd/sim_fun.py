import numpy as np
import matplotlib.pyplot as plt


def tuning_curve(X_train, y_train, X_test, y_test,
                 tuning_params, algo, solver_kws=None):
    """
    Plot errors for signed (-1, +1) labels and a low-level conic solver.

    tuning_params contains the conic tuning-parameter values C. The solver
    must accept solver_kws and return coefficients and intercept
    as its first two outputs. This plotting helper does not fit an estimator
    or infer arbitrary class labels.
    """

    n_tune_values = len(tuning_params)

    train_error = np.zeros(n_tune_values)
    test_error = np.zeros(n_tune_values)

    for i, param in enumerate(tuning_params):
        # Passing this positionally was interpreted as sample_weight by the
        # package's solvers and raised NotImplementedError.
        solution = algo(X_train, y_train, param, solver_kws=solver_kws)
        beta, offset = solution[:2]

        tr_pred = lin_clf_predict(X_train, beta, offset)
        test_pred = lin_clf_predict(X_test, beta, offset)

        train_error[i] = np.mean(tr_pred != y_train)
        test_error[i] = np.mean(test_pred != y_test)

    plt.plot(tuning_params, train_error, label="Train error")
    plt.plot(tuning_params, test_error, label="Test error")
    plt.xscale('log')
    plt.legend(loc='upper left')
    plt.xlabel(r"$C$", fontsize=16)
    plt.ylim(0, 1)

    return train_error, test_error


def lin_clf_predict(x, beta, offset):
    """
    Return signed linear predictions for one point or a batch.

    Computes sign(x @ beta + offset) without changing the zero-margin rule.

    Parameters
    ----------
    x: ndarray, shape (n_features,) or (n_samples, n_features)
        One data point or a batch of data points.

    beta: array-like, (n_features, )
        The classifier normal vector.

    offset: float
        The classifier offset.

    Returns
    -------
    predictions: ndarray, shape (1,) or (n_samples,)
        Signed predictions, always one-dimensional. An exact zero margin
        produces 0; positive and negative margins produce +1 and -1.
    """
    return np.sign(x.dot(beta) + offset).reshape(-1)


def sample_meatballs(n_samples=1000, n_features=20, n_test=1000,
                     sep=2, p_pos=.5):
    """
    Samples binary classification problem where classes are two spherical
    Gaussians.

    Parameters
    ----------
    n_samples: int
        Sample size of training data.

    n_features: int
        Number of features.

    n_test: int
        Number of test points.

    sep: float
        Norm of mean difference vector.

    p_pos: float
        Proportion of points in positive class

    Output
    ------
    X_train, y_train, X_test, y_test, beta_true

    """

    mean_neg = np.zeros(n_features)
    mean_pos = sep * np.ones(n_features) / np.sqrt(n_features)

    beta_true = mean_pos - mean_neg

    def sample_xy(n):
        n_pos = int(n * p_pos)
        n_neg = n - n_pos

        X_pos = np.random.multivariate_normal(mean=mean_pos, cov=np.eye(n_features), size=n_pos)
        X_neg = np.random.multivariate_normal(mean=mean_neg, cov=np.eye(n_features), size=n_neg)

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(n_pos), -1 * np.ones(n_neg)])

        return X, y

    X_train, y_train = sample_xy(n_samples)
    X_test, y_test = sample_xy(n_test)

    return X_train, y_train, X_test, y_test, beta_true


def cv_tuning(gscv, param_name, kind='test', log=True, std=True, color='red'):
    """
    Plot mean CV scores and optional standard-deviation curves.

    Parameters
    ----------
    gscv: fitted sklearn.model_selection.GridSearchCV
        Supplies cv_results_ with params and mean_test_score or
        mean_train_score. With std=True, the matching std_test_score or
        std_train_score is also required. Training scores must have been
        retained, for example with return_train_score=True.

    param_name: str
        Parameter present in each cv_results_['params'] entry.

    kind: {'test', 'train'}, default='test'
        Select test or training scores.

    log: bool, default=True
        Use a logarithmic horizontal axis; parameter values must be positive.

    std: bool, default=True
        Draw mean plus and minus one standard deviation as dashed curves.

    color: matplotlib color, default='red'
        Color for the mean and both standard-deviation curves.

    Notes
    -----
    The vertical display limits remain [0, 1]. Scores or standard-deviation
    curves outside this interval are clipped by the display; use a separate
    plot when another score range is required. Scores are not converted to
    classification errors.
    """
    param_vals = [p[param_name] for p in gscv.cv_results_['params']]

    if kind not in ('train', 'test'):
        raise ValueError("kind must be 'train' or 'test'.")
    st = '_{}_score'.format(kind)
    plt.plot(param_vals, gscv.cv_results_['mean' + st],
             color=color, label='cv {}'.format(kind))

    if std:
        plt.plot(param_vals,
                 gscv.cv_results_['mean' + st] + gscv.cv_results_['std' + st],
                 '--', color=color)
        plt.plot(param_vals,
                 gscv.cv_results_['mean' + st] - gscv.cv_results_['std' + st],
                 '--', color=color)
    if log:
        plt.xscale('log')
    plt.ylim(0, 1)
    plt.legend()
    plt.xlabel(param_name)
