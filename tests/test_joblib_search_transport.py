"""Check the values delivered by process-parallel public search fits."""
from collections import UserList, deque, namedtuple
import copyreg
from io import BytesIO
import os
from pickle import PicklingError, dumps, loads
from threading import Barrier, Lock, get_ident
from weakref import WeakKeyDictionary

import numpy as np
import pytest
from joblib import effective_n_jobs, parallel_config
from joblib._parallel_backends import MultiprocessingBackend
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from threadpoolctl import threadpool_info, threadpool_limits

from _search_resources import single_worker_search_parent

from dwd.kern_md import KernMD
from dwd.model_selection import _parallel_search_input, fit_shared_fold_search


pytestmark = pytest.mark.usefixtures("search_parent_native_thread")


def _mapping(tmp_path, name, values, order):
    path = tmp_path / (name + '.npy')
    np.save(path, np.array(values.T, order=order))
    return np.load(path, mmap_mode='r').T


class _InheritedSearch(GridSearchCV):
    pass


class _KeywordSearch(GridSearchCV):
    def _run_search(self, evaluate_candidates):
        evaluate_candidates(candidate_params=ParameterGrid(self.param_grid), cv=KFold(2))

    def __getstate__(self):
        return 'search', super().__getstate__()

    def __setstate__(self, state):
        tag, data = state
        assert tag == 'search'
        super().__setstate__(data)


def test_process_search_preserves_subclass_splitter_and_pickle_hooks():
    rows = np.arange(8)
    X, y = rows[:, None], (rows >= 2).astype(int)

    def make_search(n_jobs):
        cv = (pair for pair in [(rows[::2], rows[1::2]), (rows[1::2], rows[::2])])
        return _KeywordSearch(DummyRegressor(strategy='constant'), {'constant': [0, 1]},
                              cv=cv, n_jobs=n_jobs, scoring='neg_mean_absolute_error')

    expected = make_search(1).fit(X, y)
    actual = make_search(2)
    original_cv = actual.cv
    with parallel_config(backend='loky', inner_max_num_threads=1):
        assert fit_shared_fold_search(actual, X, y) is actual
    for i in range(2):
        field = f'split{i}_test_score'
        np.testing.assert_array_equal(actual.cv_results_[field], expected.cv_results_[field])
    assert actual.best_params_ == expected.best_params_ == {'constant': 1}
    assert actual.cv is original_cv and list(original_cv) == []
    assert '_run_search' not in vars(actual) and '__getstate__' not in vars(actual)


@pytest.mark.parametrize('layout,search_type', [
    ('C', GridSearchCV), ('F', _InheritedSearch), ('non-native', GridSearchCV),
])
@single_worker_search_parent()
def test_process_kernmd_search_matches_serial_fold_scores(tmp_path, layout, search_type):
    rng = np.random.default_rng(8419)
    y = rng.permutation(np.repeat([-1, 1], 32))
    values = rng.normal(0, .1, (len(y), 5))
    values[:, 0] += 3 * y
    if layout == 'non-native':
        X = values.astype(values.dtype.newbyteorder('S'))
    else:
        X = _mapping(tmp_path, 'features', values, layout)
    query = values[::3] + .05
    options = dict(param_grid={'kernel': ['linear']}, cv=3,
                   scoring='accuracy', error_score='raise')
    expected = GridSearchCV(KernMD(), n_jobs=1, **options).fit(values, y)
    with parallel_config(backend='loky', inner_max_num_threads=1, max_nbytes=512):
        actual = fit_shared_fold_search(search_type(KernMD(), n_jobs=2, **options), X, y)
    for fold in range(3):
        field = f'split{fold}_test_score'
        np.testing.assert_array_equal(actual.cv_results_[field], expected.cv_results_[field])
        assert actual.cv_results_[field][0] == 1.
    np.testing.assert_array_equal(actual.predict(query), expected.predict(query))
    np.testing.assert_array_equal(X, values)
    if layout != 'non-native':
        assert not X.flags.writeable


@pytest.mark.parametrize("fit_search", [GridSearchCV.fit, fit_shared_fold_search])
def test_parallel_cv_and_single_worker_timed_parent_refit(fit_search):
    rows = np.arange(16)
    X, y = np.column_stack([rows, rows % 3]), rows % 2
    folds_started = Barrier(2, timeout=30)
    fold_threads = set()
    final = []

    def native_counts():
        return {pool["filepath"]: pool["num_threads"] for pool in threadpool_info()}

    class ObservedKernMD(KernMD):
        def fit(self, features, labels):
            if len(labels) == len(y):
                final.append((effective_n_jobs(None), native_counts()))
            else:
                fold_threads.add(get_ident())
                folds_started.wait()
            return super().fit(features, labels)

    search = GridSearchCV(ObservedKernMD(kernel="linear"), {}, n_jobs=2, cv=2,
                          error_score="raise")
    with parallel_config(backend="threading", n_jobs=3), threadpool_limits(limits=3):
        before = native_counts()
        with single_worker_search_parent():
            fit_search(search, X, y)
        assert effective_n_jobs(None) == 3
        assert native_counts() == before

    assert len(fold_threads) == 2
    assert len(final) == 1
    workers, pools = final[0]
    assert workers == 1 and pools and set(pools.values()) == {1}
    assert search.refit_time_ >= 0


class _CheckWorkerRows(BaseEstimator):
    def __init__(self, parent_pid):
        self.parent_pid = parent_pid

    def fit(self, X, y, row_data):
        assert os.getpid() != self.parent_pid
        np.testing.assert_array_equal(y, np.column_stack([X[:, 0], -X[:, 0]]))
        np.testing.assert_array_equal(row_data, y + 7)
        return self

    def score(self, X, y):
        return float(np.array_equal(y, np.column_stack([X[:, 0], -X[:, 0]])))


def test_context_processes_deliver_mapped_targets_and_fit_parameters(tmp_path):
    rows = np.arange(24)
    X = np.column_stack([rows, rows + 1])
    y = np.column_stack([rows, -rows])
    mapped_y = np.asarray(_mapping(tmp_path, 'targets', y, 'C'))
    row_data = _mapping(tmp_path, 'row-data', y + 7, 'F')
    search = GridSearchCV(_CheckWorkerRows(os.getpid()), {}, n_jobs=None,
                          cv=3, error_score='raise', refit=False)
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        fit_shared_fold_search(search, X, mapped_y, row_data=row_data)
    assert search.cv_results_['mean_test_score'][0] == 1.


def test_matched_mappings_plain_arrays_and_local_backends_keep_originals(tmp_path):
    values = np.arange(40).reshape(8, 5)
    transpose = _mapping(tmp_path, 'mapped', values, 'C')
    root = transpose.base
    with parallel_config(backend='loky', inner_max_num_threads=1):
        for original in (root, root[1:], root[:, :1], root[:1], root[:0],
                         root[:, ::2], values, np.asfortranarray(values)):
            assert _parallel_search_input(original, 2) is original
        assert _parallel_search_input(transpose, 1) is transpose
        for view in (transpose, np.asarray(transpose), np.lib.stride_tricks.as_strided(transpose)):
            result = _parallel_search_input(view, 2)
            np.testing.assert_array_equal(result, view)
            assert not np.shares_memory(result, view)
            assert result.flags.writeable == view.flags.writeable
    with parallel_config(backend='threading', n_jobs=2):
        assert _parallel_search_input(transpose, None) is transpose


def _private_mapping(tmp_path, name, values):
    path = tmp_path / (name + '.npy')
    np.save(path, np.zeros_like(values))
    mapping = np.load(path, mmap_mode='c')
    mapping[...] = values
    return mapping


_score_bonuses = WeakKeyDictionary()


class _FieldCheckedSearch(GridSearchCV):
    __slots__ = ('weights',)

    def __getstate__(self):
        fields = vars(self).copy()
        return fields, tuple(fields), self.weights, _score_bonuses[self]

    def __setstate__(self, state):
        fields, names, self.weights, bonus = state
        if tuple(fields) != names:
            raise ValueError('saved search fields differ from their recorded names')
        vars(self).update(fields)
        _score_bonuses[self] = bonus

    def score_rows(self, estimator, X, y):
        assert os.getpid() != self.parent_pid
        assert self.scoring.__self__ is self
        if self.fail_score:
            raise ValueError('requested scoring failure')
        np.testing.assert_array_equal(X[:, 0], y)
        fold = [tuple(test) for _, test in self.cv].index(tuple(y))
        weight = self.weights[fold] + _score_bonuses[self]
        return float(estimator.constant * weight + y @ np.arange(1, len(y) + 1))


@pytest.mark.parametrize('fit', [GridSearchCV.fit, fit_shared_fold_search],
                         ids=['stock-owned', 'helper-private'])
def test_process_search_preserves_subclass_field_names(tmp_path, fit):
    X = np.arange(8)[:, None]
    order = np.array([[2, 4, 1, 3], [7, 0, 6, 5]])
    if fit is fit_shared_fold_search:
        first, second = _private_mapping(tmp_path, 'order', order)
    else:
        first, second = (row.copy() for row in order)
    folds = [(second, first), (first, second)]
    search = _FieldCheckedSearch(DummyRegressor(strategy='constant'), {'constant': [-1, 1]},
                                 cv=folds, n_jobs=2, pre_dispatch=1, error_score='raise')
    search.parent_pid = os.getpid()
    search.weights = (-4, 11)
    _score_bonuses[search] = 5
    search.scoring = search.score_rows
    with parallel_config(backend='loky', inner_max_num_threads=1, max_nbytes=32):
        search.fail_score = True
        with pytest.raises(ValueError, match='requested scoring failure'):
            fit(search, X, X[:, 0])
        assert search.cv is folds and '__getstate__' not in vars(search)
        search.fail_score = False
        assert fit(search, X, X[:, 0]) is search

    assert search.cv is folds and '__getstate__' not in vars(search)
    np.testing.assert_array_equal(search.cv_results_['split0_test_score'], [24., 26.])
    np.testing.assert_array_equal(search.cv_results_['split1_test_score'], [29., 61.])
    assert search.best_params_ == {'constant': 1}
    restored = loads(dumps(search))
    assert restored.scoring.__self__ is restored
    assert restored.weights == (-4, 11) and _score_bonuses[restored] == 5
    np.testing.assert_array_equal(restored.predict(X), np.ones(len(X)))


class _CheckPrivateRows(BaseEstimator):
    def __init__(self, parent_pid, process_parallel=True):
        self.parent_pid = parent_pid
        self.process_parallel = process_parallel

    def fit(self, X, y, rows, row_data):
        assert (os.getpid() != self.parent_pid) == self.process_parallel
        np.testing.assert_array_equal(X, np.column_stack([rows + 100, -rows - 5]))
        np.testing.assert_array_equal(y, rows % 3)
        np.testing.assert_array_equal(row_data, 3 * rows + 7)
        self.inputs_ = (X, y, row_data)
        return self

    def score(self, X, y):
        return float(np.array_equal(y, (X[:, 0] - 100) % 3))


@pytest.mark.parametrize('jobs,process_parallel,step', [
    (2, True, -1), (2.0, True, 1), (2.5, True, 1), (1.9, False, 1),
])
def test_workers_receive_private_features_labels_and_direct_fit_params(
        tmp_path, jobs, process_parallel, step):
    rows = np.arange(24)
    X = np.asarray(_private_mapping(tmp_path, 'X', np.column_stack([rows + 100, -rows - 5])))
    X.flags.writeable = False
    y = _private_mapping(tmp_path, 'y', rows % 3)
    row_data = np.asarray(_private_mapping(tmp_path, 'metadata', (3 * rows + 7)[::step]))[::step]
    search = GridSearchCV(_CheckPrivateRows(os.getpid(), process_parallel), {}, n_jobs=None,
                          cv=3, error_score='raise', refit=not process_parallel)
    with parallel_config(backend='loky', n_jobs=jobs, inner_max_num_threads=1, max_nbytes=64):
        fit_shared_fold_search(search, X, y, rows=rows, row_data=row_data)
    assert search.cv_results_['mean_test_score'][0] == 1.
    if not process_parallel:
        refit_X, refit_y, refit_data = search.best_estimator_.inputs_
        assert refit_X is X and refit_y is y and refit_data is row_data
    for name in ('X', 'y', 'metadata'):
        assert not np.any(np.load(tmp_path / (name + '.npy')))


def test_private_views_snapshot_only_for_processes_and_keep_writeability(tmp_path):
    root = _private_mapping(tmp_path, 'private', np.arange(40).reshape(8, 5))
    readonly = np.asarray(root[:, ::2])
    readonly.flags.writeable = False
    for view in (root, root.T, root[::2, ::-1], root[:, 0],
                 root[:1, :1], root[:0], readonly):
        with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
            assert _parallel_search_input(view, 1) is view
            snapshot = _parallel_search_input(view, None)
            assert type(snapshot) is np.ndarray and snapshot.flags.owndata
            assert not np.shares_memory(snapshot, view)
            assert snapshot.flags.writeable == view.flags.writeable
            np.testing.assert_array_equal(snapshot, view)
        with parallel_config(backend='threading', n_jobs=2):
            assert _parallel_search_input(view, None) is view


class _CheckFoldRows(BaseEstimator):
    def __init__(self, expected, parent_pid, fail_score=False):
        self.expected = expected
        self.parent_pid = parent_pid
        self.fail_score = fail_score

    def fit(self, X, y):
        self.fit_pid_ = os.getpid()
        self.train_rows_ = tuple(X[:, 0])
        np.testing.assert_array_equal(y, X[:, 0])
        return self

    def score(self, X, y):
        assert self.fit_pid_ == os.getpid() != self.parent_pid
        np.testing.assert_array_equal(y, X[:, 0])
        observed = (self.train_rows_, tuple(X[:, 0]))
        assert observed in self.expected
        if self.fail_score:
            raise RuntimeError('requested scoring failure')
        return self.expected.index(observed) + 1


_Fold = namedtuple('_Fold', ['train', 'test'])


class _CountedFolds(UserList):
    def __init__(self, pairs):
        super().__init__(pairs)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self.data)


def test_worker_search_accepts_reusable_folds_with_closed_buffer():
    rows = np.arange(8)
    small, large = rows[:3], rows[3:]
    pairs = [(large, small), (small, large)]
    folds = _CountedFolds(pairs)
    folds.source = BytesIO()
    folds.source.close()
    options = dict(estimator=DummyRegressor(strategy='constant'), param_grid={'constant': [0, 1]},
                   scoring='neg_mean_absolute_error', error_score='raise')
    labels = (rows >= 2).astype(int)
    reference = GridSearchCV(cv=pairs, n_jobs=1, **options).fit(rows[:, None], labels)
    search = GridSearchCV(cv=folds, n_jobs=2, **options)
    with parallel_config(backend='loky', inner_max_num_threads=1):
        assert fit_shared_fold_search(search, rows[:, None], labels) is search
    for key in ('split0_test_score', 'split1_test_score', 'mean_test_score'):
        np.testing.assert_array_equal(search.cv_results_[key], reference.cv_results_[key])
    assert search.best_index_ == reference.best_index_ == 1
    assert search.best_estimator_.constant == reference.best_estimator_.constant == 1
    assert search.cv is folds and folds.iterations == 1 and folds.source.closed
    assert search.shared_fold_info['completed']


@pytest.mark.parametrize('case', ['picklable-cv', 'locked-cv', 'shared-child'])
def test_worker_search_propagates_unrelated_scorer_pickle_error(case):
    class FailsOnce:
        def __init__(self):
            self.attempts = 0

        def __getstate__(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError('independent scorer pickle error')
            return vars(self).copy()

        def __call__(self, estimator, X, y):
            return 1.

    rows = np.arange(8)
    folds = _CountedFolds([(rows[:4], rows[4:]), (rows[4:], rows[:4])])
    scorer = FailsOnce()
    scoring = scorer
    if case == 'shared-child':
        class ScoreWithFolds:
            def __init__(self, cv, child):
                self.cv = cv
                self.child = child

            def __call__(self, estimator, X, y):
                return self.child(estimator, X, y)

        folds.child = scorer
        scoring = ScoreWithFolds(folds, scorer)
    if case != 'picklable-cv':
        folds.lock = Lock()
    search = GridSearchCV(DummyRegressor(), {}, scoring=scoring, n_jobs=2,
                          cv=folds, pre_dispatch=1, error_score='raise')
    with parallel_config(backend='loky', inner_max_num_threads=1):
        with pytest.raises(PicklingError) as caught:
            fit_shared_fold_search(search, rows[:, None], rows)
    assert 'independent scorer pickle error' in str(caught.value.__cause__)
    assert scorer.attempts == folds.iterations == 1
    assert search.cv is folds and not search.shared_fold_info['completed']
    assert not hasattr(search, 'best_estimator_')


def test_worker_search_preserves_resource_released_during_serialization():
    class Refresh:
        def __init__(self, cv):
            self.cv = cv

        def __getstate__(self):
            self.cv.resource = 'released'
            return {}

    class Score:
        def __init__(self, cv):
            self.cv = cv

        def __call__(self, estimator, X, y):
            if estimator.constant == 0:
                return float(isinstance(self.cv, UserList) and self.cv.resource == 'released')
            return .25

    def run(fit):
        rows = np.arange(8)
        cv = UserList([(rows[:4], rows[4:]), (rows[4:], rows[:4])])
        # Pickling this attribute releases the resource before the pickler reaches it.
        cv.refresh = Refresh(cv)
        cv.resource = Lock()
        search = GridSearchCV(DummyRegressor(strategy='constant'), {'constant': [0, 1]}, cv=cv,
                              scoring=Score(cv), n_jobs=2, pre_dispatch=1, error_score='raise')
        with parallel_config(backend='loky', inner_max_num_threads=1):
            assert fit(search, rows[:, None], rows) is search
        assert search.cv is cv and cv.resource == 'released'
        return search

    reference = run(GridSearchCV.fit)
    actual = run(fit_shared_fold_search)
    for key in ('split0_test_score', 'split1_test_score', 'mean_test_score'):
        np.testing.assert_array_equal(reference.cv_results_[key], [1., .25])
        np.testing.assert_array_equal(actual.cv_results_[key], reference.cv_results_[key])
    assert actual.best_index_ == reference.best_index_ == 0
    assert actual.best_estimator_.constant == reference.best_estimator_.constant == 0


@pytest.mark.parametrize('kind', ['selector', 'named_callback'])
def test_multimetric_refit_exposes_public_parameters_during_fit(tmp_path, kind):
    if kind == 'named_callback' and not hasattr(GridSearchCV, 'set_callbacks'):
        pytest.skip('This sklearn version has no public fit callbacks')
    rows = np.arange(8)
    expected = (((3, 1, 2, 0), (7, 5, 6, 4)), ((7, 5, 6, 4), (3, 1, 2, 0)))
    mapped = _private_mapping(tmp_path, 'public-folds', np.array(expected[0]))
    a, b = mapped
    cv = deque([(a, b), (b, a)])
    truth_checks = []
    names = []

    class Selector(BaseEstimator):
        def __init__(self, index):
            self.index = index

        def __bool__(self):
            params = search.get_params()
            visible = params['cv'] is cv and params['refit'] is self
            truth_checks.append(visible and params.get('refit__index') == self.index)
            return visible

        def __call__(self, results):
            assert search.cv is cv and search.refit is self
            return search.get_params()['refit__index']

    def choose(results):
        assert search.cv is cv and search.refit is choose
        return 1

    class Audit:
        def setup(self, estimator, context):
            params = estimator.get_params(deep=False)
            names.append(params['refit'].__name__)
            assert estimator is search and params['refit'] is choose and params['cv'] is cv

        teardown = setup

        def on_fit_task_begin(self, estimator, context):
            pass

        def on_fit_task_end(self, estimator, context):
            return False

    refit = Selector(1) if kind == 'selector' else choose
    grid = {'expected': [expected, expected[::-1]]}
    search = GridSearchCV(_CheckFoldRows(expected, os.getpid()), grid, cv=cv, n_jobs=2,
                          scoring={'rows': lambda est, X, y: est.score(X, y)},
                          refit=refit, error_score='raise')
    if kind == 'named_callback':
        search.set_callbacks(Audit())
    with parallel_config(backend='loky', inner_max_num_threads=1):
        fit_shared_fold_search(search, rows[:, None], rows)
    assert search.cv is cv and search.refit is refit
    if kind == 'selector':
        assert truth_checks and all(truth_checks)
    else:
        assert names == ['choose', 'choose']
    assert search.best_index_ == 1 and search.best_estimator_.fit_pid_ == os.getpid()
    np.testing.assert_array_equal(search.cv_results_['split0_test_rows'], [1, 2])
    np.testing.assert_array_equal(search.cv_results_['split1_test_rows'], [2, 1])
    np.testing.assert_array_equal(mapped, expected[0])
    assert not np.any(np.load(tmp_path / 'public-folds.npy'))


@pytest.mark.parametrize('cv_kind,private', [
    ('deque', False), ('namedtuple', False), ('namedtuple', True),
])
def test_callable_refit_reads_original_cv_and_weights_process_scores(tmp_path, cv_kind, private):
    rows = np.arange(8)
    # Labels are constant within each fold. The unweighted scores tie, and the
    # larger test fold decides which constant is best under sample weighting.
    X, y = rows[:, None], (rows >= 3).astype(int)
    indices = rows
    if private:
        path = tmp_path / 'refit-folds.npy'
        np.save(path, rows[::-1])
        indices = np.load(path, mmap_mode='c')
        indices[:] = rows
    small, large = indices[:3], indices[3:]
    if cv_kind == 'deque':
        folds = deque([(large, small), (small, large)])
    else:
        folds = [_Fold(large, small), _Fold(small, large)]

    def run_search(n_jobs, fit):
        def weighted_refit(results):
            weights = []
            scores = []
            for i in range(len(search.cv)):
                pair = search.cv[i]
                weights.append(len(pair.test if cv_kind == 'namedtuple' else pair[1]))
                scores.append(results[f'split{i}_test_score'])
            assert search.cv is folds
            assert search.refit is weighted_refit
            return np.average(scores, axis=0, weights=weights).argmax()

        search = GridSearchCV(DummyRegressor(strategy='constant'), {'constant': [0, 1]},
                              cv=folds, n_jobs=n_jobs, refit=weighted_refit,
                              scoring='neg_mean_absolute_error', error_score='raise')
        return fit(search, X, y)

    expected = run_search(1, GridSearchCV.fit)
    with parallel_config(backend='loky', inner_max_num_threads=1):
        actual = run_search(2, fit_shared_fold_search)
    for i in range(2):
        field = f'split{i}_test_score'
        np.testing.assert_array_equal(actual.cv_results_[field], expected.cv_results_[field])
    assert actual.best_index_ == expected.best_index_ == 1
    assert actual.best_params_ == expected.best_params_ == {'constant': 1}


@pytest.mark.parametrize('cv_kind,fail_score', [
    ('list', False), ('list', True), ('namedtuple', False),
    ('object-array', False), ('deque', False),
])
def test_process_search_uses_private_fold_indices_and_restores_cv(tmp_path, cv_kind, fail_score):
    # Both partitions are valid. Each ordered train/test pair distinguishes
    # the caller's private edits from the rows recorded on disk.
    expected = (((3, 1, 2, 0), (7, 5, 6, 4)), ((7, 5, 6, 4), (3, 1, 2, 0)))
    path = tmp_path / 'folds.npy'
    np.save(path, [[0, 2, 4, 6], [1, 3, 5, 7]])
    file_before = path.read_bytes()
    mapped = np.load(path, mmap_mode='c')
    mapped[:] = expected[0]
    first, second = mapped[0], np.asarray(mapped[1])
    first.flags.writeable = False
    fold_one, fold_two = (first, second), [second, first]
    if cv_kind == 'namedtuple':
        fold_one, fold_two = _Fold(first, second), _Fold(second, first)
    if cv_kind == 'object-array':
        cv = np.empty(2, dtype=object)
        cv[0], cv[1] = fold_one, fold_two
    elif cv_kind == 'deque':
        cv = deque((fold_one, fold_two))
    else:
        cv = [fold_one, fold_two]
    estimator = _CheckFoldRows(expected, os.getpid(), fail_score)
    grid = {}
    search = GridSearchCV(estimator, grid, cv=cv, n_jobs=2, refit=False, error_score='raise')
    rows = np.arange(8)
    with parallel_config(backend='loky', inner_max_num_threads=1):
        if fail_score:
            with pytest.raises(RuntimeError, match='requested scoring failure'):
                fit_shared_fold_search(search, rows[:, None], rows)
        else:
            assert fit_shared_fold_search(search, rows[:, None], rows) is search
            assert [search.cv_results_[f'split{i}_test_score'][0] for i in range(2)] == [1, 2]
    assert search.shared_fold_info['completed'] == (not fail_score)
    assert search.cv is cv and cv[0] is fold_one and cv[1] is fold_two
    assert fold_one[0] is fold_two[1] is first and fold_one[1] is fold_two[0] is second
    assert search.estimator is estimator and search.param_grid is grid
    assert not first.flags.writeable and second.flags.writeable
    np.testing.assert_array_equal(mapped, expected[0])
    assert path.read_bytes() == file_before


def test_process_search_preserves_one_shot_cv_with_reused_index_buffers(tmp_path):
    expected = (((3, 1, 2, 0), (7, 5, 6, 4)), ((7, 5, 6, 4), (3, 1, 2, 0)))
    path = tmp_path / 'generator-folds.npy'
    np.save(path, [[0, 2, 4, 6], [1, 3, 5, 7]])
    file_before = path.read_bytes()
    mapped = np.load(path, mmap_mode='c')
    train, test = mapped

    def splits():
        for pair in expected:
            mapped[:] = pair
            yield train, test

    cv = splits()
    search = GridSearchCV(_CheckFoldRows(expected, os.getpid()), {},
                          cv=cv, n_jobs=2, refit=False, error_score='raise')
    rows = np.arange(8)
    with parallel_config(backend='loky', inner_max_num_threads=1):
        fit_shared_fold_search(search, rows[:, None], rows)
    # Stock check_cv collects the generator before dispatch. Both references
    # see the final values of the same buffers.
    assert [search.cv_results_[f'split{i}_test_score'][0] for i in range(2)] == [2, 2]
    assert search.cv is cv
    assert list(cv) == []
    np.testing.assert_array_equal(mapped, expected[-1])
    assert path.read_bytes() == file_before


def _restore_reducer_fields(search, state):
    fields, names, saved_cv = state
    vars(search).update(fields)
    search.fields_match = tuple(fields) == names
    search.saved_cv = saved_cv


def _reduce_search_fields(search, fields=None):
    if fields is None:
        fields = search.__getstate__()
    state = fields, tuple(vars(search)), search.cv
    return copyreg.__newobj__, (type(search),), state, None, None, _restore_reducer_fields


class _ReducerScoredSearch(GridSearchCV):
    def score_fields(self, estimator, X, y):
        assert os.getpid() != self.parent_pid and self.scoring.__self__ is self
        assert self.saved_cv is self.cv
        np.testing.assert_array_equal(self.cv[0][1], [2, 4, 1, 3])
        np.testing.assert_array_equal(X[:, 0], y)
        assert any(np.array_equal(test, y) for _, test in self.cv)
        return float(estimator.constant == int(self.fields_match))


class _MethodReducerSearch(_ReducerScoredSearch):
    def __reduce_ex__(self, protocol):
        return _reduce_search_fields(self, vars(self).copy())


@pytest.mark.parametrize('reducer', ['method', 'copyreg', 'queue', 'queue-replaced', 'loky'])
def test_custom_search_reducer_preserves_fields_and_fold_alias(tmp_path, monkeypatch, reducer):
    X = np.arange(8)[:, None]
    order = np.array([[2, 4, 1, 3], [7, 0, 6, 5]])
    cls = _MethodReducerSearch if reducer == 'method' else _ReducerScoredSearch
    if reducer == 'copyreg':
        monkeypatch.setitem(copyreg.dispatch_table, cls, _reduce_search_fields)
    elif reducer == 'loky':
        from joblib.externals.loky.backend import reduction

        monkeypatch.setitem(reduction._dispatch_table, cls, _reduce_search_fields)

    mapped = _private_mapping(tmp_path, 'reducer-folds', order)
    try:
        for fit, indices in ((GridSearchCV.fit, order), (fit_shared_fold_search, mapped)):
            first, second = indices
            folds = [(second, first), (first, second)]
            search = cls(DummyRegressor(strategy='constant'), {'constant': [0, 1]},
                         cv=folds, n_jobs=2, pre_dispatch=1, error_score='raise')
            search.parent_pid = os.getpid()
            search.scoring = search.score_fields
            if reducer.startswith('queue'):
                backend = MultiprocessingBackend()
                options = {}
                if reducer == 'queue':
                    backend = MultiprocessingBackend(forward_reducers={cls: _reduce_search_fields})
                else:
                    def install_queue_reducer():
                        queue = backend._pool._inqueue
                        queue._reducers = {**queue._reducers, cls: _reduce_search_fields}
                    monkeypatch.setattr(backend, 'start_call', install_queue_reducer)
            else:
                backend, options = 'loky', {'inner_max_num_threads': 1}
            with parallel_config(backend=backend, max_nbytes=32, **options):
                assert fit(search, X, X[:, 0]) is search
            for fold in (0, 1):
                np.testing.assert_array_equal(search.cv_results_[f'split{fold}_test_score'], [0., 1.])
            assert search.best_params_ == {'constant': 1} and search.cv is folds
            assert {'__getstate__', 'fields_match', 'saved_cv'}.isdisjoint(vars(search))
            restored = loads(dumps(search))
            assert restored.scoring.__self__ is restored
            np.testing.assert_array_equal(restored.predict(X), np.ones(len(X)))
        np.testing.assert_array_equal(np.load(tmp_path / 'reducer-folds.npy'), np.zeros_like(order))
    finally:
        mapped._mmap.close()
