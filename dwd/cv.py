from sklearn.base import clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.utils import check_X_y
from sklearn.model_selection import check_cv, ParameterGrid
from time import time
from copy import deepcopy
import numpy as np

# TODO: make this use parallelism


def run_cv(clf, X, y, params, scoring='accuracy', cv=5, refit_best=True):
    """
    Runs cross-validation on a classifier.
    This function is designed for classifiers where we can quickly  compute
    the whole tuning path if we are allowed to pre-compute some quantities
    based on X before fitting the model.

    The clf object should be a sklearn compatible classifier
    with an additional clf.cv_init(X) function which does some precomputation
    based on the X data before precomputing the whole path.

    Parameters
    ----------
    clf:

    X, y:

    params: dict of lists

    scoring:

    cv:

    Output
    ------
    best_params, best_scores, best_clf, agg_results, all_cv_results
    """

    X, y = check_X_y(X, y,
                     accept_sparse='csr',
                     dtype='numeric')

    scorer = check_scoring(estimator=clf, scoring=scoring)

    # init cross-validation generator
    cv = check_cv(cv, y=y, classifier=is_classifier(clf))
    folds = list(cv.split(X, y))
    n_folds = len(folds)

    all_param_settings = DoL2LoD(params)
    n_settings = len(all_param_settings)

    # get tuning path for each fold
    all_cv_results = []
    for f_idx, (train, test) in enumerate(folds):
        if getattr(clf, 'kernel', None) == 'precomputed':
            # A fold trains on its own Gram matrix and predicts with query
            # rows against training columns, as sklearn's pairwise CV does.
            X_train = X[train, :][:, train]
            X_test = X[test, :][:, train]
        else:
            X_train = X[train, :]
            X_test = X[test, :]
        y_train = y[train]
        y_test = y[test]

        # clone(clf) is critical to send a copy! otherwise clf gets modified
        fold_results = get_path_scores(clf=clone(clf),
                                       X_train=X_train,
                                       y_train=y_train,
                                       X_test=X_test,
                                       y_test=y_test,
                                       scorer=scorer,
                                       params=params)

        all_cv_results.append(fold_results)

    # aggregate metrics of interst over folds
    agg_results = {'params': all_param_settings}
    metric_keys = ['test_score', 'train_score', 'runtime']
    for metric in metric_keys:
        agg_results['mean_' + metric] = []
        agg_results['std_' + metric] = []

    for s in range(n_settings):
        for metric in metric_keys:

            # values of the metric for a given parameter setting across
            # all folds
            vals = [all_cv_results[f][metric][s] for f in range(n_folds)]

            agg_results['mean_' + metric].append(np.mean(vals))
            agg_results['std_' + metric].append(np.std(vals))

    # get the best tuning parameter setting
    idx_best = np.argmax(agg_results['mean_test_score'])
    best_params = all_param_settings[idx_best]
    best_score = agg_results['mean_test_score'][idx_best]

    if refit_best:
        # refit classifier on full training data with best parameters
        best_clf = clone(clf)
        best_clf = best_clf.set_params(**best_params)
        best_clf = best_clf.fit(X, y)
    else:
        best_clf = None

    return best_params, best_score, best_clf, agg_results, all_cv_results


def get_path_scores(clf, X_train, y_train, X_test, y_test, scorer, params):

    # initalize classifier before cross validation
    cv_results = {'params': [],
                  'train_score': [],
                  'test_score': [],
                  'runtime': [],
                  'init_time': 0.0}

    # fit and score classifier for each parameter setting
    # TODO: parallelize here
    for param_setting in DoL2LoD(params):
        clf.set_params(**param_setting)

        # KernGDWD's cache depends on both this training fold and the kernel
        # parameters. Lambda and q changes can reuse the same eigensystem.
        needs_init = not cv_results['params']
        if hasattr(clf, '_cv_cache_matches'):
            needs_init = not clf._cv_cache_matches(X_train)
        if needs_init:
            start_time = time()
            clf.cv_init(X_train)
            cv_results['init_time'] += time() - start_time

        start_time = time()
        clf.fit(X_train, y_train)
        runtime = time() - start_time

        tr_score = scorer(clf, X_train, y_train)
        tst_score = scorer(clf, X_test, y_test)

        cv_results['params'].append(param_setting)
        cv_results['runtime'].append(runtime)
        cv_results['train_score'].append(tr_score)
        cv_results['test_score'].append(tst_score)

    return cv_results


def listify(x):
    """
    Returns a list
    """
    if isinstance(x, (str, dict)) or not hasattr(x, '__len__'):
        return [x]
    else:
        return x


def DoL2LoD(DL):
    """
    Convert a dict of value lists to its full Cartesian parameter grid.
    """
    dl = deepcopy(DL)
    for k in dl.keys():
        dl[k] = listify(dl[k])

    return list(ParameterGrid(dl))
