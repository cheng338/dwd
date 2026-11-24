"""Process searches with caller-supplied representations of named folds."""
import os
from collections import UserList
from pickle import PicklingError
from threading import Lock, get_ident
from typing import NamedTuple

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV

from dwd.model_selection import fit_shared_fold_search


pytestmark = pytest.mark.usefixtures("search_parent_native_thread")


class Fold(NamedTuple):
    train: np.ndarray
    test: np.ndarray


class WeightedFold(Fold):
    def weight(self):
        return len(self.test)


def _display_type():
    display_lock = Lock()

    class DisplayFold(NamedTuple):
        train: np.ndarray
        test: np.ndarray

        def __repr__(self):
            with display_lock:
                return f'DisplayFold(train={self.train!r}, test={self.test!r})'

    return DisplayFold


def _folds(first_type, second_type):
    rows = np.arange(8)
    small, large = rows[:3], rows[3:]
    return UserList([first_type(large, small), second_type(small, large)])


def _search(cv, scoring, jobs):
    return GridSearchCV(DummyRegressor(strategy='constant'), {'constant': [0, 1]},
                        cv=cv, scoring=scoring, n_jobs=jobs, error_score='raise')


def _fit(search, threshold, expected, backend='loky'):
    X = np.arange(8)[:, None]
    y = (X[:, 0] >= threshold).astype(int)
    with parallel_config(backend=backend, **({'inner_max_num_threads': 1} if backend == 'loky' else {})):
        if search.n_jobs == 1:
            search.fit(X, y)
        else:
            assert fit_shared_fold_search(search, X, y) is search
    for key, scores in zip(('split0_test_score', 'split1_test_score', 'mean_test_score'),
                           expected, strict=True):
        np.testing.assert_allclose(search.cv_results_[key], scores, rtol=1e-14, atol=0)
    assert search.best_index_ == search.best_estimator_.constant == 1
    assert search.best_params_ == {'constant': 1}
    np.testing.assert_array_equal(search.predict(X), np.ones(len(X)))


_MAE = ([-1 / 3, -2 / 3], [-1, 0], [-2 / 3, -1 / 3])


@pytest.mark.parametrize('builtin', [True, False], ids=['builtin', 'independent-callable'])
def test_positional_folds_for_independent_scorer(builtin):
    record_type = _display_type()
    original = _folds(record_type, record_type)
    projected = [(fold.train, fold.test) for fold in original]
    parent = os.getpid()
    for jobs, cv in ((1, original), (2, projected)):
        def score(estimator, X, y):
            assert (os.getpid() == parent) == (jobs == 1)
            return -np.mean(np.abs(estimator.predict(X) - y))

        search = _search(cv, 'neg_mean_absolute_error' if builtin else score, jobs)
        _fit(search, 2, _MAE)
        assert search.cv is cv


@pytest.mark.parametrize('use_reducer', [False, True], ids=['replace-one-record', 'user-reducer'])
def test_named_fields_and_retained_record_in_mixed_folds(use_reducer):
    display_type = _display_type()

    class ReducibleFold(display_type):
        def __reduce__(self):
            return Fold, (self.train, self.test)

    original = _folds(ReducibleFold if use_reducer else display_type, WeightedFold)
    displayed, weighted = original
    prepared = [displayed if use_reducer else Fold(*displayed), weighted]
    parent = os.getpid()
    for jobs, cv in ((1, original), (2, prepared)):
        def score(estimator, X, y):
            assert (os.getpid() == parent) == (jobs == 1)
            assert search.cv is cv
            first, second = search.cv
            if jobs == 2:
                assert type(first) is Fold
            assert second is weighted and type(second) is WeightedFold
            assert first.train is second.test and first.test is second.train
            checked = getattr(search, '_checked_cv_orig', None)
            if checked is not None:
                assert checked.cv[0] is first and checked.cv[1] is second
            weight = second.weight() if np.array_equal(second.test, X[:, 0]) else len(first.test)
            return -weight * np.mean(np.abs(estimator.predict(X) - y))

        search = _search(cv, score, jobs)
        _fit(search, 3, ([0, -3], [-5, 0], [-2.5, -1.5]))
        assert search.cv is cv and original[0] is displayed and original[1] is weighted
    if use_reducer:
        assert prepared[0] is displayed


def test_positional_folds_preserve_scorers_own_serialization_error():
    record_type = _display_type()
    original = _folds(record_type, record_type)
    projected = [(fold.train, fold.test) for fold in original]
    display = original[0].__repr__

    def score(estimator, X, y):
        assert display().startswith('DisplayFold(')
        return -np.mean(np.abs(estimator.predict(X) - y))

    _fit(_search(original, score, 1), 2, _MAE)
    with pytest.raises(PicklingError) as caught:
        _fit(_search(projected, score, 2), 2, _MAE)
    assert '_thread.lock' in str(caught.value.__cause__)


def test_thread_workers_can_use_original_folds_and_their_display_lock():
    record_type = _display_type()
    original = _folds(record_type, record_type)
    display = original[0].__repr__
    main_thread = get_ident()
    for jobs in (1, 2):
        def score(estimator, X, y):
            assert (get_ident() == main_thread) == (jobs == 1)
            assert search.cv is original and search.cv[0] is display.__self__
            assert display().startswith('DisplayFold(')
            return -np.mean(np.abs(estimator.predict(X) - y))

        search = _search(original, score, jobs)
        _fit(search, 2, _MAE, backend='threading')
