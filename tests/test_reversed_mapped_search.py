"""Process-search regression for ordinary slicing of file-mapped inputs."""
import os
from collections import UserList, deque
from copy import deepcopy
from functools import partial
from inspect import signature
from pickle import PicklingError, dumps, loads
from threading import Lock, get_ident
from types import SimpleNamespace

import numpy as np
import pytest
from joblib import parallel_config
from joblib.parallel import BatchedCalls
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold, check_cv
from sklearn.model_selection import _search

import dwd.model_selection as selection


def _reversed_mapping(tmp_path, name, values, key, mode):
    path = tmp_path / (name + '.npy')
    np.save(path, values[key])
    return np.asarray(np.load(path, mmap_mode=mode))[key]


def _check_snapshot(result, original):
    # Check ownership in the parent before any process dispatch.
    assert type(result) is np.ndarray and result.flags.owndata
    assert result.flags.writeable == original.flags.writeable
    np.testing.assert_array_equal(result, original)


def _track_snapshots(monkeypatch):
    # The historical comparison sources copied inside _parallel_search_input.
    boundary = ('_snapshot_search_input' if hasattr(selection, '_snapshot_search_input')
                else '_parallel_search_input')
    original = getattr(selection, boundary)
    snapshots = []

    def track(value, *args):
        safe = original(value, *args)
        if safe is not value:
            _check_snapshot(safe, value)
            snapshots.append(value)
        return safe

    monkeypatch.setattr(selection, boundary, track)
    return snapshots


@pytest.mark.parametrize('mode', ['r', 'r+'])
def test_reversed_process_preparation(tmp_path, mode):
    data = np.arange(60).reshape(12, 5)
    path = tmp_path / 'data.npy'
    np.save(path, data)
    backing = np.load(path, mmap_mode=mode)
    a = np.asarray(backing)
    reversed_views = [backing[::-1], a[:, ::-1], a[::-2, ::-1], a[::-1, 0]]
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        for view in reversed_views:
            _check_snapshot(selection._parallel_search_input(view, None), view)
            assert selection._parallel_search_input(view, 1) is view
        for ordinary in (backing, a[::2], a[::2, :1][:, ::-1], data[::-1]):
            assert selection._parallel_search_input(ordinary, None) is ordinary
    with parallel_config(backend='threading', n_jobs=2):
        for view in reversed_views:
            assert selection._parallel_search_input(view, None) is view


class _RecordingFold(BaseEstimator):
    def __init__(self, parent_pid, expected_folds):
        self.parent_pid = parent_pid
        self.expected_folds = expected_folds

    @staticmethod
    def _rows(X, y):
        rows = X[:, 0].astype(int)
        np.testing.assert_array_equal(X, np.column_stack([rows, 100 + rows, -rows]))
        np.testing.assert_array_equal(y, rows % 3)
        return tuple(rows)

    def _check_worker(self):
        assert os.getpid() != self.parent_pid

    def fit(self, X, y, row_data):
        self._check_worker()
        self.train_ = self._rows(X, y)
        np.testing.assert_array_equal(row_data, 7 + 3 * X[:, 0])
        return self

    def score(self, X, y):
        return float(self.expected_folds.index((self.train_, self._rows(X, y))) + 1)


@pytest.mark.parametrize('mode', ['r', 'r+'])
def test_reversed_public_search(tmp_path, mode, monkeypatch):
    rows = np.arange(18)
    data = np.column_stack([rows, 100 + rows, -rows]).astype(float)
    reverse = slice(None, None, -1)
    X = _reversed_mapping(tmp_path, 'X', data, (reverse, reverse), mode)
    y = _reversed_mapping(tmp_path, 'y', rows % 3, reverse, mode)
    metadata = _reversed_mapping(tmp_path, 'metadata', 7 + 3 * rows, reverse, mode)
    even, odd = rows[::2], rows[1::2]
    a = _reversed_mapping(tmp_path, 'even', even, reverse, mode)
    b = _reversed_mapping(tmp_path, 'odd', odd, reverse, mode)
    folds = [(a, b), (b, a)]
    expected_folds = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))
    estimator = _RecordingFold(os.getpid(), expected_folds)
    search = GridSearchCV(estimator, {}, cv=folds, refit=False, error_score='raise')
    prepare = selection._parallel_search_input
    prepared = []

    def checked_prepare(value, n_jobs, *args):
        result = prepare(value, n_jobs, *args)
        _check_snapshot(result, value)
        prepared.append(id(value))
        return result

    monkeypatch.setattr(selection, '_parallel_search_input', checked_prepare)
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        # A baseline retaining even one view must stop in the parent.
        for view in (X, y, metadata, a, b):
            _check_snapshot(prepare(view, None), view)
        assert selection.fit_shared_fold_search(search, X, y, row_data=metadata) is search
    assert sorted(prepared) == sorted(map(id, (X, y, metadata, a, b)))
    assert search.cv is folds
    assert search.cv_results_['split0_test_score'][0] == 1.
    assert search.cv_results_['split1_test_score'][0] == 2.
    np.testing.assert_array_equal(X, data)
    np.testing.assert_array_equal(y, rows % 3)
    np.testing.assert_array_equal(metadata, 7 + 3 * rows)
    np.testing.assert_array_equal(a, even)
    np.testing.assert_array_equal(b, odd)


class _AnnotatedFold(list):
    pass


class _FoldMetadata:
    def __init__(self, train, other_fold):
        self.train = train
        self.other_fold = other_fold
        self.fit_thread = get_ident()

    def __deepcopy__(self, memo):
        if get_ident() != self.fit_thread:
            raise RuntimeError('fold metadata copied outside fit thread')
        return _FoldMetadata(deepcopy(self.train, memo), deepcopy(self.other_fold, memo))


def _check_transport_array(array):
    assert isinstance(array, np.ndarray)
    assert all(n <= 1 or s >= 0 for n, s in zip(array.shape, array.strides)), (
        'unsafe worker-visible cv stride')
    base = array
    while base is not None:
        assert not (isinstance(base, np.memmap) and base.mode == 'c'), 'worker would reopen COW rows'
        base = getattr(base, 'base', None)


def _check_cv_container(cv, kind, expected):
    if kind == 'stack':
        assert type(cv) is np.ndarray
        containers = [cv]
    else:
        assert type(cv) is list
        containers = cv
        if kind == 'mixed':
            assert all(type(pair) is tuple for pair in cv)
            containers = [index for pair in cv for index in pair]
        elif kind in ('annotated-pairs', 'process-local'):
            annotated = cv[-1]
            assert type(annotated) is _AnnotatedFold
            if kind == 'annotated-pairs':
                assert type(annotated.metadata) is _FoldMetadata
                assert annotated.metadata.train is annotated[0]
                assert annotated.metadata.other_fold is cv[0]
            containers = [*cv[:-1], *annotated]
    for array in containers:
        _check_transport_array(array)
    np.testing.assert_array_equal(cv, expected)


class _ReadSearchCV:
    # A scorer can carry the search on sklearn versions that do not include
    # caller=search directly in the delayed fit arguments.
    def __init__(self, search, kind):
        self.search = search
        self.kind = kind

    def __call__(self, estimator, X, y):
        assert os.getpid() != estimator.parent_pid
        _check_worker_state(vars(self.search), self.kind, estimator.expected_folds)
        return estimator.score(X, y)


def _check_worker_state(state, kind, expected):
    _check_cv_container(state['cv'], kind, expected)
    if '_checked_cv_orig' in state:
        for pair, values in zip(state['_checked_cv_orig'].cv, expected, strict=True):
            for index, truth in zip(pair, values, strict=True):
                _check_transport_array(index)
                np.testing.assert_array_equal(index, truth)


def _check_job_indices(kwargs, expected):
    i, _ = kwargs['split_progress']
    for key, values in zip(('train', 'test'), expected[i], strict=True):
        index = kwargs[key]
        _check_transport_array(index)
        np.testing.assert_array_equal(index, values)
    return i


def _check_received_state(kind, same_rows, fit_and_score, *args, **kwargs):
    caller = kwargs.get('caller')
    search = getattr(kwargs['scorer'], 'search', caller)
    if search is not None:
        assert os.getpid() != search.estimator.parent_pid
        assert caller is None or caller is search
        expected = search.estimator.expected_folds
        _check_worker_state(vars(search), kind, expected)
        i = _check_job_indices(kwargs, expected)
        if same_rows:
            train, test = search.cv[i]
            assert train is kwargs['train'] and test is kwargs['test']
    return fit_and_score(*args, **kwargs)


def _check_dispatch(monkeypatch, rows, cv, expected, kind):
    X = np.column_stack([rows, 100 + rows, -rows])
    estimator = _RecordingFold(os.getpid(), expected)
    search = GridSearchCV(estimator, {}, cv=cv, refit=False, error_score='raise')
    search.scoring = _ReadSearchCV(search, kind)
    inspected = []

    class GuardedParallel(_search.Parallel):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def __call__(self, tasks):
            # Materialize the actual sklearn jobs before even opening a pool.
            # Unsafe mapped views must never reach joblib's worker reducer.
            jobs = list(tasks)
            assert search.cv is cv and '__getstate__' not in vars(search)
            for _, _, kwargs in jobs:
                i = _check_job_indices(kwargs, expected)
                caller = kwargs['scorer'].search
                assert kwargs.get('caller', caller) is caller is search
                assert caller.cv is cv
                inspected.append(i)
            assert inspected == [0, 1]
            return super().__call__(jobs)

    monkeypatch.setattr(_search, '_fit_and_score',
                        partial(_check_received_state, kind, kind == 'mixed', _search._fit_and_score))
    monkeypatch.setattr(_search, 'Parallel', GuardedParallel)
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        assert selection.fit_shared_fold_search(search, X, rows % 3, row_data=7 + 3 * rows) is search
    assert inspected == [0, 1]
    assert search.cv is cv
    assert [search.cv_results_[f'split{i}_test_score'][0] for i in range(2)] == [1., 2.]


def _reversed_container_folds(tmp_path, kind):
    rows = np.arange(18)
    even, odd = tuple(rows[::2]), tuple(rows[1::2])
    expected = ((even, odd), (odd, even))
    reverse = slice(None, None, -1)
    if kind == 'stack':
        cv = _reversed_mapping(tmp_path, 'stack', np.array(expected), reverse, 'r')
    else:
        cv = [_reversed_mapping(tmp_path, str(i), np.array(pair), reverse, 'r+')
              for i, pair in enumerate(expected)]
    return rows, cv, expected


@pytest.mark.parametrize('kind', ['stack', 'pair-list'])
def test_reversed_cv_container_dispatch(tmp_path, monkeypatch, kind):
    rows, cv, expected = _reversed_container_folds(tmp_path, kind)
    _check_dispatch(monkeypatch, rows, cv, expected, kind)
    assert cv.strides[0] < 0 if kind == 'stack' else all(pair.strides[0] < 0 for pair in cv)
    np.testing.assert_array_equal(cv, expected)


def _check_process_scoring(monkeypatch, rows, cv, expected, kind, scoring, check_unused=None):
    estimator = _RecordingFold(os.getpid(), expected)
    search = GridSearchCV(estimator, {}, cv=cv, refit=False, error_score='raise', pre_dispatch=1)
    if scoring == 'capturing':
        search.scoring = _ReadSearchCV(search, kind)

    serialized_cv = []
    has_caller = 'caller' in signature(_search._fit_and_score).parameters
    carries_search = has_caller or scoring == 'capturing'

    class GuardedParallel(_search.Parallel):
        def __call__(self, tasks):
            jobs = list(tasks)
            assert search.cv is cv and '__getstate__' not in vars(search)
            assert [_check_job_indices(kw, expected) for _, _, kw in jobs] == list(range(len(expected)))
            for _, _, kw in jobs:
                assert ('caller' in kw) == has_caller
                if has_caller:
                    assert kw['caller'] is search
                assert getattr(kw['scorer'], 'search', None) is (
                    search if scoring == 'capturing' else None)
            if not carries_search and check_unused is not None:
                check_unused()

            # Observe only actual materialization requests, including repeated
            # requests in different batches. The unpickled worker checks the result.
            get_worker_cv = selection._ProcessSplits.worker_cv.fget
            fit_thread = get_ident()

            def observe(splits):
                assert os.getpid() == estimator.parent_pid
                if kind in ('stack', 'pair-list'):
                    assert get_ident() != fit_thread
                value = get_worker_cv(splits)
                serialized_cv.append(value)
                return value

            with monkeypatch.context() as patch:
                patch.setattr(selection._ProcessSplits, 'worker_cv', property(observe))
                return super().__call__(jobs)

    monkeypatch.setattr(_search, '_fit_and_score',
                        partial(_check_received_state, kind, False, _search._fit_and_score))
    monkeypatch.setattr(_search, 'Parallel', GuardedParallel)
    X = np.column_stack([rows, 100 + rows, -rows])
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        assert selection.fit_shared_fold_search(search, X, rows % 3, row_data=7 + 3 * rows) is search

    if carries_search:
        # pre_dispatch=1 sends folds in separate jobs.
        assert len(serialized_cv) >= 2
        assert all(value is serialized_cv[0] for value in serialized_cv)
    else:
        assert not serialized_cv
    assert search.cv is cv
    np.testing.assert_array_equal(cv, expected)
    for i in range(len(expected)):
        assert search.cv_results_[f'split{i}_test_score'][0] == float(i + 1)
    return carries_search


@pytest.mark.parametrize('kind', ['stack', 'pair-list'])
@pytest.mark.parametrize('scoring', ['ordinary', 'capturing'])
def test_worker_cv_snapshot_count(tmp_path, monkeypatch, scoring, kind):
    rows, cv, expected = _reversed_container_folds(tmp_path, kind)
    containers = [cv] if kind == 'stack' else list(cv)
    assert all(array.strides[0] < 0 for array in containers)
    assert all(index.strides[0] > 0 for pair in cv for index in pair)
    snapshots = _track_snapshots(monkeypatch)

    def check_unused():
        assert not snapshots, 'unused public cv container snapshot'

    if _check_process_scoring(monkeypatch, rows, cv, expected, kind, scoring, check_unused):
        assert sorted(map(id, snapshots)) == sorted(map(id, containers))
    else:
        check_unused()
    assert all(array.strides[0] < 0 for array in containers)


@pytest.mark.parametrize('scoring', ['ordinary', 'capturing'])
def test_worker_cv_metadata_thread(tmp_path, monkeypatch, scoring):
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    numeric = _reversed_mapping(tmp_path, 'pair', np.array([even, odd]), slice(None, None, -1), 'r')
    path = tmp_path / 'train.npy'
    np.save(path, even)
    train = np.load(path, mmap_mode='c')
    train[:] = odd
    annotated = _AnnotatedFold([train, even])
    metadata = annotated.metadata = _FoldMetadata(train, numeric)
    cv = [numeric, annotated]
    expected = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))

    _check_process_scoring(monkeypatch, rows, cv, expected, 'annotated-pairs', scoring)

    assert cv[0] is numeric and cv[1] is annotated
    assert annotated[0] is train and annotated[1] is even
    assert annotated.metadata is metadata and metadata.train is train and metadata.other_fold is numeric
    assert numeric.strides[0] < 0
    np.testing.assert_array_equal(np.load(path), even)
    np.testing.assert_array_equal(train, odd)


class _RejectFoldCopy:
    def __init__(self):
        self.copy_calls = 0

    def __deepcopy__(self, memo):
        self.copy_calls += 1
        raise ValueError('fold rejected its metadata copy')


@pytest.mark.parametrize('scoring', ['ordinary', 'capturing'])
def test_unserializable_fold_metadata(tmp_path, monkeypatch, scoring):
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    numeric = _reversed_mapping(tmp_path, 'pair', np.array([even, odd]), slice(None, None, -1), 'r')
    annotated = _AnnotatedFold([odd, even])
    cv = [numeric, annotated]
    expected = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))
    assert numeric.strides[0] < 0 and all(index.strides[0] > 0 for pair in cv for index in pair)
    carries_search = 'caller' in signature(_search._fit_and_score).parameters or scoring == 'capturing'
    rejected = _RejectFoldCopy()
    for metadata, error, message in [(Lock(), TypeError, "cannot pickle '_thread.lock' object"),
                                     (rejected, ValueError, 'fold rejected its metadata copy')]:
        annotated.metadata = metadata
        with monkeypatch.context() as patch:
            if carries_search:
                with pytest.raises((error, PicklingError)) as failure:
                    _check_process_scoring(patch, rows, cv, expected, 'process-local', scoring)
                if isinstance(failure.value, PicklingError):
                    # loky wraps a failure from parent-side pickling. Its cause
                    # must still identify the failing hook or lock.
                    assert f'{error.__name__}: {message}' in str(failure.value.__cause__)
                else:
                    assert str(failure.value) == message
            else:
                # No task carries the search. Its process-local metadata must
                # not prevent either real worker fit from returning its score.
                _check_process_scoring(patch, rows, cv, expected, 'process-local', scoring)
        assert cv[0] is numeric and cv[1] is annotated and annotated.metadata is metadata
        assert annotated[0] is odd and annotated[1] is even
    if carries_search:
        assert rejected.copy_calls == 1
    else:
        assert rejected.copy_calls <= 1


class _FoldRecord:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def __iter__(self):
        yield self.train
        yield self.test


@pytest.mark.parametrize('order', ['numeric-first', 'custom-first'])
def test_mixed_fold_record_dispatch(tmp_path, monkeypatch, order):
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    path = tmp_path / 'pair.npy'
    np.save(path, [odd, even])
    numeric = np.load(path, mmap_mode='r')[::-1]
    assert numeric.strides[0] < 0 < numeric.strides[1]
    train = _reversed_mapping(tmp_path, 'train', odd, slice(None, None, -1), 'r')
    record = _FoldRecord(train, even)
    cv = [numeric, record]
    expected = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))
    if order == 'custom-first':
        cv.reverse()
        expected = expected[::-1]

    snapshots = _track_snapshots(monkeypatch)
    _check_dispatch(monkeypatch, rows, cv, expected, 'mixed')
    assert all(source is not numeric for source in snapshots), 'discarded numeric fold snapshot'
    assert len(snapshots) == 1 and snapshots[0] is train
    first, second = (numeric, record) if order == 'numeric-first' else (record, numeric)
    assert cv[0] is first and cv[1] is second
    assert record.train is train and record.test is even
    np.testing.assert_array_equal(numeric, [even, odd])
    np.testing.assert_array_equal(train, odd)


def _check_owner_and_fit(fit_and_score, *args, **kwargs):
    # Check the actual search unpickled with this job, including sklearn
    # versions that send it only through a user-provided scorer.
    caller = kwargs.get('caller')
    search = getattr(kwargs['scorer'], 'search', caller)
    if search is not None:
        assert os.getpid() != search.estimator.parent_pid
        if caller is not None:
            assert caller is search
        cv = search.cv
        assert cv[1].owner is search
        assert cv[1].metadata.other_fold is cv[0]
        assert cv[1].metadata.train is cv[1][0]
        expected = search.estimator.expected_folds
        _check_worker_state(vars(search), 'annotated-pairs', expected)
        i = _check_job_indices(kwargs, expected)
        for index, name in zip(cv[i], ('train', 'test'), strict=True):
            np.testing.assert_array_equal(index, kwargs[name])
    return fit_and_score(*args, **kwargs)


@pytest.mark.parametrize('mapped', [False, True], ids=['owning', 'reversed-map'])
@pytest.mark.parametrize('scoring', ['ordinary', 'capturing'])
def test_search_reference_in_fold(tmp_path, monkeypatch, mapped, scoring):
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    expected = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))
    numeric = np.array([even, odd])
    if mapped:
        numeric = _reversed_mapping(tmp_path, 'pair', numeric, slice(None, None, -1), 'r')
    annotated = _AnnotatedFold([odd, even])
    metadata = annotated.metadata = _FoldMetadata(odd, numeric)
    cv = [numeric, annotated]
    estimator = _RecordingFold(os.getpid(), expected)
    search = GridSearchCV(estimator, {}, cv=cv, refit=False, error_score='raise', pre_dispatch=1)
    annotated.owner = search
    if scoring == 'capturing':
        search.scoring = _ReadSearchCV(search, 'annotated-pairs')
    has_caller = 'caller' in signature(_search._fit_and_score).parameters

    class GuardedParallel(_search.Parallel):
        def __call__(self, tasks):
            jobs = list(tasks)
            assert search.cv is cv
            assert [_check_job_indices(kw, expected) for _, _, kw in jobs] == [0, 1]
            for _, _, kw in jobs:
                assert ('caller' in kw) == has_caller
                if has_caller:
                    assert kw['caller'] is search
            assert search.cv is cv and cv[1].owner is search
            if mapped:
                assert '__getstate__' not in vars(search)
            return super().__call__(jobs)

    monkeypatch.setattr(_search, '_fit_and_score', partial(_check_owner_and_fit, _search._fit_and_score))
    monkeypatch.setattr(_search, 'Parallel', GuardedParallel)
    X = np.column_stack([rows, 100 + rows, -rows])
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        assert selection.fit_shared_fold_search(search, X, rows % 3, row_data=7 + 3 * rows) is search
    assert [search.cv_results_[f'split{i}_test_score'][0] for i in range(2)] == [1., 2.]
    assert search.cv is cv and cv[0] is numeric and cv[1] is annotated
    assert annotated.owner is search and annotated.metadata is metadata
    assert metadata.other_fold is numeric and metadata.train is odd
    assert annotated[0] is odd and annotated[1] is even
    assert (numeric.strides[0] < 0) == mapped
    np.testing.assert_array_equal(numeric, [even, odd])


class _HeapMapping(np.ndarray):
    """Owning heap storage standing in for an mmap in parent-only tests."""


def _simulated_object_mapping(values, reverse=False):
    assert values.dtype.hasobject
    owner = np.ndarray.__new__(_HeapMapping, values.shape, values.dtype)
    owner[...] = values[::-1] if reverse else values
    mapped = owner.view(np.memmap)
    mapped.mode = 'r+'
    # memmap.__getitem__ strips the subclass for views of an array without
    # a real _mmap. Preserve the normal raw-memmap type and its base chain.
    return np.ndarray.__getitem__(mapped, slice(None, None, -1)) if reverse else mapped


class _ParentOnlyParallel(_search.Parallel):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def __call__(self, tasks):
        # Materialize only the first job in the parent. A failing baseline
        # never opens a process pool or sends a simulated map to a reducer.
        for _ in tasks:
            pytest.fail('job reached the dispatch boundary')
        pytest.fail('expected a fold or a parent-side rejection')


@pytest.mark.parametrize('position', ['outer', 'outer-view', 'pair', 'iterator-pair', 'row',
                                      'X', 'y', 'fit-param', 'structured-fit-param'])
def test_file_backed_object_search_input(monkeypatch, position):
    # No file ever holds PyObject pointers. Real heap arrays provide the
    # contents, and only the module's test for backing storage is extended.
    monkeypatch.setattr(selection, 'mmap', (selection.mmap, _HeapMapping))
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    cv = [(even, odd), (odd, even)]
    X = np.column_stack([rows, 100 + rows, -rows])
    y = rows % 3
    fit_params = {'row_data': 7 + 3 * rows}
    if position.startswith('outer'):
        values = np.empty(2, dtype=object)
        values[:] = cv
        cv = _simulated_object_mapping(values, reverse=True)
        if position == 'outer-view':
            cv = np.asarray(cv)
    elif position in ('pair', 'iterator-pair'):
        values = np.empty(2, dtype=object)
        values[:] = [even, odd]
        cv[0] = _simulated_object_mapping(values)
        if position == 'iterator-pair':
            cv = iter(cv)
    elif position == 'row':
        cv[0] = (_simulated_object_mapping(even.astype(object)), odd)
    elif position == 'X':
        X = _simulated_object_mapping(X.astype(object))
    elif position == 'y':
        y = _simulated_object_mapping(y.astype(object))
    else:
        dtype = [('value', object)] if position == 'structured-fit-param' else object
        values = np.empty(rows.shape, dtype=dtype)
        values[:] = rows
        fit_params['row_data'] = _simulated_object_mapping(values)

    monkeypatch.setattr(_search, 'Parallel', _ParentOnlyParallel)
    search = GridSearchCV(_RecordingFold(os.getpid(), ()), {}, cv=cv, refit=False)
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        with pytest.raises(ValueError, match='file-backed object arrays') as failure:
            selection.fit_shared_fold_search(search, X, y, **fit_params)
    assert 'in-memory object array' in str(failure.value)
    assert search.cv is cv and not search.shared_fold_info['completed']


@pytest.mark.parametrize('position', ['pair', 'row', 'iterator-pair', 'changed-public-row',
                                      'override-row'])
def test_override_file_backed_object_folds(monkeypatch, position):
    monkeypatch.setattr(selection, 'mmap', (selection.mmap, _HeapMapping))
    monkeypatch.setattr(_search, 'Parallel', _ParentOnlyParallel)
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    cv = [(even, odd), (odd, even)]
    if position.endswith('pair'):
        values = np.empty(2, dtype=object)
        values[:] = [even, odd]
        pair = _simulated_object_mapping(values)
    else:
        pair = (even, _simulated_object_mapping(odd.astype(object)))
    if position in ('pair', 'row', 'iterator-pair'):
        cv[0] = pair
    if position == 'iterator-pair':
        # Only sklearn's checked copy still contains the first pair.
        cv = iter(cv)

    class OverrideSearch(GridSearchCV):
        def _run_search(self, evaluate_candidates):
            override = KFold(2)
            if position == 'changed-public-row':
                # sklearn already copied the original list into its wrapper.
                self.cv[0] = pair
            elif position == 'override-row':
                override = check_cv([pair, (odd, even)])
            evaluate_candidates(candidate_params=[{}], cv=override)

    search = OverrideSearch(_RecordingFold(os.getpid(), ()), {}, cv=cv, refit=False)
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        with pytest.raises(ValueError, match='file-backed object arrays') as failure:
            selection.fit_shared_fold_search(search, rows[:, None], rows % 3)
    assert 'in-memory object array' in str(failure.value)
    assert search.cv is cv and not search.shared_fold_info['completed']


class _UnusedOverrideFold(_FoldRecord):
    def __iter__(self):
        raise AssertionError('overridden fold was consumed')


class _SplitOverrideSearch(GridSearchCV):
    def _run_search(self, evaluate_candidates):
        evaluate_candidates(candidate_params=[{}], cv=KFold(2))


class _ReadOverrideCV:
    def __init__(self, search):
        self.search = search

    def __call__(self, estimator, X, y):
        ordinary, unused = self.search.cv
        assert isinstance(unused, _UnusedOverrideFold)
        assert ordinary[0] is unused.test and ordinary[1] is unused.train
        np.testing.assert_array_equal(ordinary, [np.arange(0, 18, 2), np.arange(1, 18, 2)])
        return estimator.score(X, y)


def test_override_preserves_splitter_and_unused_folds(tmp_path):
    rows = np.arange(18)
    path = tmp_path / 'even.npy'
    np.save(path, rows[::2])
    even, odd = np.load(path, mmap_mode='r'), rows[1::2]
    unused = _UnusedOverrideFold(odd, even)
    cv = [(even, odd), unused]
    first, last = tuple(rows[:9]), tuple(rows[9:])
    estimator = _RecordingFold(os.getpid(), ((last, first), (first, last)))
    search = _SplitOverrideSearch(estimator, {}, cv=cv, refit=False, error_score='raise')
    search.scoring = _ReadOverrideCV(search)
    X = np.column_stack([rows, 100 + rows, -rows])
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        assert selection.fit_shared_fold_search(search, X, rows % 3, row_data=7 + 3 * rows) is search
    assert [search.cv_results_[f'split{i}_test_score'][0] for i in range(2)] == [1., 2.]
    assert search.cv is cv and cv[1] is unused
    assert unused.test is even and unused.train is odd


class _ReadObjectFolds:
    def __init__(self, search):
        self.search = search

    def __call__(self, estimator, X, y):
        assert os.getpid() != estimator.parent_pid
        cv = self.search.cv
        assert type(cv) is np.ndarray and cv.dtype == object
        for pair, expected in zip(cv, estimator.expected_folds, strict=True):
            for index, values in zip(pair, expected, strict=True):
                np.testing.assert_array_equal(index, values)
        return estimator.score(X, y)


def test_in_memory_object_folds(monkeypatch):
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    expected = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))
    values = np.empty(2, dtype=object)
    values[:] = [(odd, even), (even, odd)]
    cv = values[::-1]
    estimator = _RecordingFold(os.getpid(), expected)
    search = GridSearchCV(estimator, {}, cv=cv, refit=False, error_score='raise')
    search.scoring = _ReadObjectFolds(search)
    X = np.column_stack([rows, 100 + rows, -rows])
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        assert selection.fit_shared_fold_search(search, X, rows % 3, row_data=7 + 3 * rows) is search
        # A plain ndarray over a buffer, with no np.memmap owner, uses
        # joblib's by-value object serialization. Check this parent decision
        # using heap storage for the simulated buffer.
        buffer = np.ndarray.__new__(_HeapMapping, values.shape, values.dtype)
        buffer[:] = values
        plain = buffer.view(np.ndarray)
        assert plain.base is buffer
        monkeypatch.setattr(selection, 'mmap', (selection.mmap, _HeapMapping))
        assert selection._parallel_search_input(plain, None) is plain
    assert search.cv is cv and cv.strides[0] < 0 and cv[0] is values[1]
    assert [search.cv_results_[f'split{i}_test_score'][0] for i in range(2)] == [1., 2.]


class _EvaluationContextSearch(GridSearchCV):
    def _run_search(self, evaluate_candidates):
        # sklearn has already selected its Parallel. Keep the alternate
        # splitter local to this evaluation, just like a subclass creating it
        # here, so a serialized search cannot transport it as another attribute.
        override = vars(self).pop('_evaluation_cv', None)
        with parallel_config(backend=self.evaluation_backend, n_jobs=2):
            evaluate_candidates([{}], cv=override)


def test_selected_executor_rejects_object_row(monkeypatch):
    monkeypatch.setattr(selection, 'mmap', (selection.mmap, _HeapMapping))
    monkeypatch.setattr(_search, 'Parallel', _ParentOnlyParallel)
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    pair = _FoldRecord(_simulated_object_mapping(even.astype(object)), odd)
    cv = [pair, (odd, even)]
    search = _EvaluationContextSearch(_RecordingFold(os.getpid(), ()), {}, cv=cv, refit=False)
    search.evaluation_backend = 'threading'
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        with pytest.raises(ValueError, match='file-backed object arrays'):
            selection.fit_shared_fold_search(search, rows[:, None], rows % 3)
    assert search.cv is cv and not search.shared_fold_info['completed']


@pytest.mark.parametrize('position', ['train', 'test'])
def test_checked_override_rejects_userlist_row(monkeypatch, position):
    monkeypatch.setattr(selection, 'mmap', (selection.mmap, _HeapMapping))
    monkeypatch.setattr(_search, 'Parallel', _ParentOnlyParallel)
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    cv = [(even, odd), (odd, even)]
    pair = UserList([even, odd])
    i = 0 if position == 'train' else 1
    pair[i] = _simulated_object_mapping(pair[i].astype(object))
    search = _EvaluationContextSearch(_RecordingFold(os.getpid(), ()), {}, cv=cv, refit=False)
    search.evaluation_backend = 'loky'
    search._evaluation_cv = check_cv([pair, (odd, even)])
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        with pytest.raises(ValueError, match='file-backed object arrays'):
            selection.fit_shared_fold_search(search, rows[:, None], rows % 3)
    assert search.cv is cv and not search.shared_fold_info['completed']


class _LocalRecordingFold(_RecordingFold):
    def __init__(self, parent_pid, expected_folds, fit_thread, threaded):
        super().__init__(parent_pid, expected_folds)
        self.fit_thread = fit_thread
        self.threaded = threaded

    def _check_worker(self):
        assert os.getpid() == self.parent_pid
        assert (get_ident() != self.fit_thread) == self.threaded


class _OncePair(UserList):
    def __iter__(self):
        assert not getattr(self, 'read', False), 'selected fold was unpacked twice'
        self.read = True
        return iter(self.data)


class _SlottedEvaluationSearch(_EvaluationContextSearch):
    __slots__ = ('owner',)

    def __getstate__(self):
        assert self.owner is self
        return vars(self).copy(), self.owner

    def __setstate__(self, state):
        attributes, self.owner = state
        vars(self).update(attributes)


class _ReadOriginalSplitter:
    def __init__(self, search):
        self.search = search

    def __call__(self, estimator, X, y):
        assert self.search.owner is self.search
        assert isinstance(self.search.cv, KFold)
        if hasattr(self.search, '_checked_cv_orig'):
            assert self.search._checked_cv_orig is self.search.cv
        return estimator.score(X, y)


def _check_executor_numeric_folds(tmp_path, monkeypatch, executor, public=None, entry_backend=None):
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    a = _reversed_mapping(tmp_path, 'even', even, slice(None, None, -1), 'r')
    path = tmp_path / 'private.npy'
    np.save(path, even)
    b = np.load(path, mmap_mode='c')
    b[:] = odd
    sources = [(a, b), (b, a)]
    expected = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))
    process = executor == 'process'
    if process:
        estimator = _RecordingFold(os.getpid(), expected)
    else:
        estimator = _LocalRecordingFold(os.getpid(), expected, get_ident(), executor == 'threading')
    if public == 'public-splitter':
        cv = KFold(2)
    elif public == 'public-records':
        cv = [(even, odd), _UnusedOverrideFold(odd, even)]
    else:
        cv = sources
    search_type = _SlottedEvaluationSearch if isinstance(cv, KFold) else _EvaluationContextSearch
    search = search_type(estimator, {}, cv=cv, refit=False, error_score='raise')
    if isinstance(cv, KFold):
        search.owner = search
    if public is None:
        if process:
            search.scoring = _ReadSearchCV(search, 'mixed')
        search.evaluation_backend = 'threading' if process else 'loky'
    else:
        search.evaluation_backend = 'loky'
        pairs = [_OncePair(pair) for pair in sources]
        search._evaluation_cv = check_cv(iter(pairs))
        search.scoring = (_ReadOriginalSplitter(search) if isinstance(cv, KFold)
                          else _ReadOverrideCV(search))

    class GuardedParallel(_search.Parallel):
        def __call__(self, tasks):
            jobs = list(tasks)
            assert self._effective_n_jobs() == (1 if executor == 'serial' else 2)
            if executor != 'serial':
                assert bool(getattr(self._backend, 'supports_sharedmem', False)) != process
            assert len(jobs) == len(sources)
            for (_, _, kw), pair in zip(jobs, sources, strict=True):
                for name, source in zip(('train', 'test'), pair, strict=True):
                    if process:
                        assert kw[name] is not source, 'process job retained a reversed or private fold'
                        _check_snapshot(kw[name], source)
                    else:
                        assert kw[name] is source, 'shared job replaced a fold index'
            if process and public is None:
                assert search.cv is cv and '__getstate__' not in vars(search)
            return super().__call__(jobs)

    backend = 'threading' if executor == 'threading' else 'loky'
    if entry_backend is None:
        entry_backend = backend
        monkeypatch.setattr(_search, 'Parallel', GuardedParallel)
    else:
        class SelectedParallel(GuardedParallel):
            def __init__(self, **options):
                # Bind the real executor differently from both ambient contexts.
                limits = {} if backend == 'threading' else {'inner_max_num_threads': 1}
                with parallel_config(backend=backend, n_jobs=2, **limits):
                    super().__init__(**options)

        monkeypatch.setattr(_search, 'Parallel', SelectedParallel)
    options = {} if entry_backend == 'threading' else {'inner_max_num_threads': 1}
    X = np.column_stack([rows, 100 + rows, -rows])
    with parallel_config(backend=entry_backend, n_jobs=1 if executor == 'serial' else 2, **options):
        assert selection.fit_shared_fold_search(search, X, rows % 3, row_data=7 + 3 * rows) is search
    assert search.cv is cv and search.shared_fold_info['completed']
    assert [search.cv_results_[f'split{i}_test_score'][0] for i in range(2)] == [1., 2.]
    if public is not None:
        assert all(pair.read for pair in pairs)
    assert a.strides[0] < 0 and b.mode == 'c'
    np.testing.assert_array_equal(np.load(path), even)
    np.testing.assert_array_equal(a, even)
    np.testing.assert_array_equal(b, odd)


@pytest.mark.parametrize('executor', ['process', 'threading', 'serial'])
def test_selected_executor_numeric_folds(tmp_path, monkeypatch, executor):
    _check_executor_numeric_folds(tmp_path, monkeypatch, executor)


@pytest.mark.parametrize('executor', ['process', 'threading'])
def test_executor_differs_from_fit_entry(tmp_path, monkeypatch, executor):
    entry = 'threading' if executor == 'process' else 'loky'
    _check_executor_numeric_folds(tmp_path, monkeypatch, executor, entry_backend=entry)


@pytest.mark.parametrize('public', ['public-splitter', 'public-records'])
def test_checked_override_numeric_folds(tmp_path, monkeypatch, public):
    _check_executor_numeric_folds(tmp_path, monkeypatch, 'process', public)


class _NativeStateSearch(_SlottedEvaluationSearch):
    __slots__ = ('cv_alias',)

    def __getstate__(self):
        if os.getpid() == self.estimator.parent_pid:
            assert id(self) == self.parent_id, 'pickle hook received a different search'
            assert id(self.cv) == self.public_id, 'pickle hook observed a replaced public cv'
        attributes, owner = super().__getstate__()
        return attributes, {'owner': owner, 'cv_alias': self.cv_alias}

    def __setstate__(self, state):
        attributes, slots = state
        super().__setstate__((attributes, slots['owner']))
        self.cv_alias = slots['cv_alias']


def _check_native_fold_state(state, owner, search, expected, iterator):
    assert state['alias'] is owner is search
    if iterator:
        assert state['cv'] == ()
    else:
        _check_cv_container(state['cv'], 'mixed', expected)
    if '_checked_cv_orig' in state:
        checked = state['_checked_cv_orig'].cv
        for pair, truth in zip(checked, expected, strict=True):
            for index, values in zip(pair, truth, strict=True):
                _check_transport_array(index)
                np.testing.assert_array_equal(index, values)


class _ReadNativeState:
    def __init__(self, search, expected, iterator):
        self.search = search
        self.expected = expected
        self.iterator = iterator

    def __call__(self, estimator, X, y):
        assert self.search.cv_alias is self.search.public_alias is self.search.cv
        _check_native_fold_state(vars(self.search), self.search.owner, self.search,
                                 self.expected, self.iterator)
        return estimator.score(X, y)


def _fit_guarded_folds(monkeypatch, search, rows, inspect_jobs):
    class GuardedParallel(_search.Parallel):
        def __call__(self, tasks):
            jobs = list(tasks)
            # All parent assertions precede joblib submission, including for
            # the comparison source whose numeric views can still be attached.
            inspect_jobs(jobs)
            indices = [_check_job_indices(kw, search.estimator.expected_folds) for _, _, kw in jobs]
            assert indices == [0, 1]
            return super().__call__(jobs)

    monkeypatch.setattr(_search, 'Parallel', GuardedParallel)
    X = np.column_stack([rows, 100 + rows, -rows])
    cv = search.cv
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        assert selection.fit_shared_fold_search(search, X, rows % 3, row_data=7 + 3 * rows) is search
    assert search.cv is cv and search.shared_fold_info['completed']
    assert [search.cv_results_[f'split{i}_test_score'][0] for i in range(2)] == [1., 2.]
    assert '_run_search' not in vars(search) and '__getstate__' not in vars(search)


@pytest.mark.parametrize('case', ['list-alternate', 'iterator-alternate',
                                  'list-default', 'iterator-default'])
def test_native_search_state_folds(tmp_path, monkeypatch, case):
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    expected = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))
    alternate = case.endswith('alternate')
    iterator = case.startswith('iterator')
    if alternate:
        public = list(KFold(2).split(rows))
    else:
        a = _reversed_mapping(tmp_path, 'even', even, slice(None, None, -1), 'r')
        public = [(a, odd), (odd, a)]
    original = tuple((tuple(train), tuple(test)) for train, test in public)
    # An exhausted generator still needs a serializable worker representation.
    cv = (pair for pair in public) if iterator else public
    estimator = _RecordingFold(os.getpid(), expected)
    search = _NativeStateSearch(estimator, {}, cv=cv, refit=False, error_score='raise')
    search.parent_id, search.public_id = id(search), id(cv)
    search.owner = search.alias = search
    search.cv_alias = search.public_alias = cv
    search.evaluation_backend = 'loky'
    if alternate:
        search._evaluation_cv = check_cv([(even, odd), (odd, even)])
    search.scoring = _ReadNativeState(search, original, iterator)

    def inspect_jobs(jobs):
        assert search.cv is cv
        state, slots = search.__getstate__()
        assert slots['cv_alias'] is state['public_alias'] is state['cv'] is cv
        assert state['alias'] is slots['owner'] is search
        assert '__getstate__' not in state

    _fit_guarded_folds(monkeypatch, search, rows, inspect_jobs)
    assert search.owner is search.alias is search
    assert search.cv_alias is search.public_alias is cv
    if iterator:
        assert next(cv, None) is None
    if not alternate:
        assert public[0][0] is public[1][1] is a and a.strides[0] < 0


class _TruthKFold(KFold):
    def __init__(self, selected):
        super().__init__(2)
        self.selected = selected
        self.truth_calls = self.split_calls = 0

    def __bool__(self):
        self.truth_calls += 1
        assert self.truth_calls == 1, 'the override was truth-tested twice'
        return self.selected

    def split(self, *args, **kwargs):
        assert self.selected, 'the false-valued splitter was used'
        self.split_calls += 1
        return super().split(*args, **kwargs)


@pytest.mark.parametrize('selected', [False, True], ids=['false', 'true'])
def test_override_truth_selects_folds(tmp_path, monkeypatch, selected):
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    if selected:
        public = [_UnusedOverrideFold(even, odd), _UnusedOverrideFold(odd, even)]
        expected = tuple((tuple(train), tuple(test)) for train, test in KFold(2).split(rows))
    else:
        a = _reversed_mapping(tmp_path, 'even', even, slice(None, None, -1), 'r')
        public = [(a, odd), (odd, a)]
        expected = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))
    estimator = _RecordingFold(os.getpid(), expected)
    search = _EvaluationContextSearch(estimator, {}, cv=public, refit=False, error_score='raise')
    search._evaluation_cv = override = _TruthKFold(selected)
    search.evaluation_backend = 'loky'

    if not selected:
        search.scoring = _ReadSearchCV(search, 'mixed')

    def inspect_jobs(jobs):
        if not selected:
            assert jobs[0][2]['train'] is not a, 'false override left the selected mapped fold attached'
            assert search.cv is public and '__getstate__' not in vars(search)

    _fit_guarded_folds(monkeypatch, search, rows, inspect_jobs)
    assert override.truth_calls == 1 and override.split_calls == int(selected)


class _OpaqueStateSearch(_EvaluationContextSearch):
    def __getstate__(self):
        return SimpleNamespace(attributes={name: value for name, value in vars(self).items()
                                           if not name.startswith('__')})

    def __setstate__(self, state):
        vars(self).update(state.attributes)


def test_opaque_search_state_when_folds_are_unchanged(monkeypatch):
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    cv = [(even, odd), (odd, even)]
    expected = tuple((tuple(train), tuple(test)) for train, test in cv)
    search = _OpaqueStateSearch(_RecordingFold(os.getpid(), expected), {}, cv=cv,
                                refit=False, error_score='raise')
    search.evaluation_backend = 'loky'
    search.scoring = _ReadSearchCV(search, 'mixed')

    def inspect_jobs(jobs):
        state = search.__getstate__()
        assert type(state) is SimpleNamespace and state.attributes['cv'] is cv
        _check_worker_state(state.attributes, 'mixed', expected)

    _fit_guarded_folds(monkeypatch, search, rows, inspect_jobs)
    assert cv[0][0] is cv[1][1] is even and cv[0][1] is cv[1][0] is odd


def test_opaque_search_state_with_consumed_iterator(monkeypatch):
    rows = np.arange(18)
    pairs = list(KFold(2).split(rows))
    expected = tuple((tuple(train), tuple(test)) for train, test in pairs)
    cv = (pair for pair in pairs)
    search = _OpaqueStateSearch(_RecordingFold(os.getpid(), expected), {}, cv=cv, refit=False)
    search.evaluation_backend = 'loky'
    search._evaluation_cv = check_cv(pairs)

    def inspect_state(worker_search):
        assert search.__getstate__().attributes['cv'] is cv
        with pytest.raises(TypeError, match='__getstate__.*fold references.*n_jobs=1'):
            worker_search()

    _check_pickle_at_dispatch(monkeypatch, search, rows, inspect_state)
    assert search.cv is cv and next(cv, None) is None


def _check_pickle_at_dispatch(monkeypatch, search, rows, inspect_state):
    # Inspect the same batch joblib would serialize without executing its jobs.
    class CheckedDispatch(Exception):
        pass

    class Capture(_search.Parallel):
        def _dispatch(self, batch):
            self.restored_batch = loads(dumps(batch))

        def __call__(self, tasks):
            jobs = list(tasks)
            expected = search.estimator.expected_folds
            assert [_check_job_indices(kw, expected) for _, _, kw in jobs] == [0, 1]

            def worker_search():
                # Older sklearn versions need not carry the search in every job.
                batch = BatchedCalls([*jobs, (id, (search,), {})], self._backend.get_nested_backend())
                self._dispatch(batch)
                return self.restored_batch.items[-1][1][0]

            inspect_state(worker_search)
            raise CheckedDispatch

    cv = search.cv
    hooks = {key: value for key, value in vars(search).items()
             if key in ('_run_search', '__getstate__')}
    monkeypatch.setattr(_search, 'Parallel', Capture)
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        with pytest.raises(CheckedDispatch):
            selection.fit_shared_fold_search(search, rows[:, None], rows % 3)
    assert search.cv is cv and not search.shared_fold_info['completed']
    for name in ('_run_search', '__getstate__'):
        if name in hooks:
            assert vars(search)[name] is hooks[name]
        else:
            assert name not in vars(search)


@pytest.mark.parametrize('container', [tuple, list, deque, np.array],
                         ids=['tuple', 'list', 'deque', 'array'])
def test_safe_native_pair_pickle(monkeypatch, container):
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    expected = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))
    cv = [container((even, odd)), container((odd, even))]
    search = _OpaqueStateSearch(_RecordingFold(os.getpid(), expected), {}, cv=cv, refit=False)
    search.evaluation_backend = 'loky'

    def inspect_state(worker_search):
        restored = worker_search()
        for i, pair in enumerate(restored.cv):
            assert type(pair) is type(cv[i])
            np.testing.assert_array_equal(pair, expected[i])
            if hasattr(restored, '_checked_cv_orig'):
                assert restored._checked_cv_orig.cv[i] is pair

    _check_pickle_at_dispatch(monkeypatch, search, rows, inspect_state)


class _PrunedStateSearch(_EvaluationContextSearch):
    def __getstate__(self):
        return self.native_state

    def __setstate__(self, state):
        self.restored = state


@pytest.mark.parametrize('case', ['dict-iterator', 'scalar-iterator', 'dict-mapped'])
def test_pruned_search_pickle_state(monkeypatch, tmp_path, case):
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    expected = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))
    train = even
    if case == 'dict-mapped':
        train = _reversed_mapping(tmp_path, 'train', even, slice(None, None, -1), 'r')
    pairs = [(train, odd), (odd, train)]
    cv = pairs if case == 'dict-mapped' else (pair for pair in pairs)
    search = _PrunedStateSearch(_RecordingFold(os.getpid(), expected), {}, cv=cv, refit=False)
    search.evaluation_backend = 'loky'
    search.native_state = {'description': 'compact search'}
    if case == 'scalar-iterator':
        search.native_state = b'compact search'
        search._evaluation_cv = check_cv(pairs)

    def inspect_state(worker_search):
        restored = worker_search()
        assert restored.restored == search.native_state
        assert set(vars(restored)) == {'restored'}

    _check_pickle_at_dispatch(monkeypatch, search, rows, inspect_state)
    if case == 'dict-mapped':
        assert cv[0][0] is cv[1][1] is train and train.strides[0] < 0
    else:
        assert next(cv, None) is None


class _OpaqueLeaf:
    def __deepcopy__(self, memo):
        raise AssertionError('rewriting fold references invoked an unrelated copy hook')


class _GraphStateSearch(_PrunedStateSearch):
    def __getstate__(self):
        self.pickle_calls += 1
        return super().__getstate__()


def test_nested_fold_pickle_aliases(monkeypatch):
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    expected = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))
    cv = (pair for pair in [(even, odd), (odd, even)])
    search = _GraphStateSearch(_RecordingFold(os.getpid(), expected), {}, cv=cv, refit=False)
    search.evaluation_backend = 'loky'
    search.pickle_calls = 0
    leaf = _OpaqueLeaf()
    branch = {'folds': cv, 'leaf': leaf}
    links = ([branch], branch)
    branch['cycle'] = links
    search.native_state = {'cv': cv, 'links': links, 'leaf': leaf}

    def check_graph(state, expected_cv):
        first, second = state['links']
        assert state['cv'] is expected_cv
        assert second['folds'] is state['cv'], 'nested fold reference was not replaced'
        assert first[0] is second and second['cycle'] is state['links']
        assert second['leaf'] is state['leaf']

    def inspect_state(worker_search):
        state = search.__getstate__()
        check_graph(state, cv)
        assert state['leaf'] is leaf and search.pickle_calls == 1
        check_graph(worker_search().restored, ())
        assert search.pickle_calls == 2

    _check_pickle_at_dispatch(monkeypatch, search, rows, inspect_state)
    assert search.native_state['links'] is branch['cycle'] is links
    assert search.native_state['cv'] is branch['folds'] is cv and next(cv, None) is None


@pytest.mark.parametrize('retain', [False, True], ids=['omit', 'retain'])
def test_saved_fold_copy_error_only_if_emitted(monkeypatch, tmp_path, retain):
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    expected = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))
    numeric = _reversed_mapping(tmp_path, 'pair', np.array([even, odd]), slice(None, None, -1), 'r')
    annotated = _AnnotatedFold([odd, even])
    annotated.metadata = rejected = _RejectFoldCopy()
    cv = [numeric, annotated]
    search = _GraphStateSearch(_RecordingFold(os.getpid(), expected), {}, cv=cv, refit=False)
    search.evaluation_backend = 'loky'
    search.pickle_calls = 0
    search.native_state = {'cv': cv} if retain else {'description': 'omitted folds'}

    def inspect_state(worker_search):
        # Preparing the public record failed on the fit thread, but the two
        # actual training jobs already contain the right row indices.
        assert rejected.copy_calls == 1
        if retain:
            with pytest.raises(ValueError, match='^fold rejected its metadata copy$'):
                worker_search()
        else:
            assert worker_search().restored == search.native_state
        assert search.pickle_calls == 1, 'native pickle hook did not run exactly once'
        assert rejected.copy_calls == 1

    _check_pickle_at_dispatch(monkeypatch, search, rows, inspect_state)
    assert cv[0] is numeric and cv[1] is annotated and annotated.metadata is rejected


class _OrderedStateSearch(_EvaluationContextSearch):
    def __getstate__(self):
        if self.layout == 'tuple':
            return tuple(vars(self).items())
        records = list(vars(self).items())
        if self.layout == 'nested':
            return {'records': records, 'nested': ([records], records)}
        return records

    def instance_state(self):
        return type(self).__getstate__(self)

    def __setstate__(self, state):
        if type(state) is dict:
            records = state['records']
            assert state['nested'][0][0] is state['nested'][1] is records
        else:
            records = state
        layout = dict(records)['layout']
        assert type(records) is (tuple if layout == 'tuple' else list)
        vars(self).update(records)


class _ForwardState:
    def __init__(self, hook):
        self.hook = hook

    def __call__(self):
        return self.hook()


@pytest.mark.parametrize('case', [
    'tuple-class-direct', 'tuple-class-forward', 'tuple-instance-direct', 'tuple-instance-forward',
    'list-class-direct', 'list-class-forward', 'list-instance-direct', 'list-instance-forward',
    'nested-instance-forward',
])
def test_ordered_attribute_pickle(monkeypatch, case):
    layout, hook, dispatch = case.split('-')
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    expected = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))
    cv = (pair for pair in [(even, odd), (odd, even)])

    def make_search(folds):
        obj = _OrderedStateSearch(_RecordingFold(os.getpid(), expected), {}, cv=folds, refit=False)
        obj.evaluation_backend = 'loky'
        obj.layout = layout
        if hook == 'instance':
            obj.__getstate__ = obj.instance_state
        obj.after_hook = 'last caller field'
        return obj

    search = make_search(cv)
    original = vars(search).get('__getstate__')
    control = make_search(())  # Native pickle with the reusable worker representation.
    if dispatch == 'forward':
        control.__getstate__ = _ForwardState(control.__getstate__)
    native = loads(dumps(control))

    def inspect_state(worker_search):
        selected = original
        if dispatch == 'forward':
            selected = search.__getstate__ = _ForwardState(search.__getstate__)
        try:
            keys = list(vars(search))
            state = search.__getstate__()
            if layout == 'nested':
                records = state['records']
                assert state['nested'][0][0] is state['nested'][1] is records
            else:
                records = state
            assert type(records) is (tuple if layout == 'tuple' else list)
            outgoing = dict(records)
            if selected is None:
                assert '__getstate__' not in outgoing
            else:
                assert outgoing['__getstate__'] is selected
            assert [name for name, _ in records] == keys
            assert outgoing['cv'] is cv and outgoing['after_hook'] == 'last caller field'
            restored = worker_search()
            assert list(vars(restored)) == keys and restored.cv == ()
            assert restored.after_hook == search.after_hook
            restored_hook = restored.__getstate__
            native_hook = native.__getstate__
            if dispatch == 'forward':
                assert type(restored_hook) is type(native_hook) is _ForwardState
                restored_hook = restored_hook.hook
                native_hook = native_hook.hook
            assert restored_hook.__self__ is restored and native_hook.__self__ is native
            expected_hook = _OrderedStateSearch.instance_state if original else _OrderedStateSearch.__getstate__
            assert restored_hook.__func__ is native_hook.__func__ is expected_hook
        finally:
            if dispatch == 'forward':
                if original is None:
                    del search.__getstate__
                else:
                    search.__getstate__ = original

    _check_pickle_at_dispatch(monkeypatch, search, rows, inspect_state)
    assert search.cv is cv and next(cv, None) is None


class _UnlabelledHookSearch(_EvaluationContextSearch):
    def __getstate__(self):
        return {'cv': self.cv, 'callbacks': [self.__getstate__]}


def test_saved_bound_pickle_hook(monkeypatch):
    rows = np.arange(18)
    even, odd = rows[::2], rows[1::2]
    expected = ((tuple(even), tuple(odd)), (tuple(odd), tuple(even)))
    cv = (pair for pair in [(even, odd), (odd, even)])
    search = _UnlabelledHookSearch(_RecordingFold(os.getpid(), expected), {}, cv=cv, refit=False)
    search.evaluation_backend = 'loky'

    def inspect_state(worker_search):
        state = search.__getstate__()
        assert state['cv'] is cv and state['callbacks'][0].__self__ is search
        restored = worker_search()
        assert restored.cv == () and restored.callbacks[0].__self__ is restored
        assert restored.callbacks[0].__func__ is _UnlabelledHookSearch.__getstate__

    _check_pickle_at_dispatch(monkeypatch, search, rows, inspect_state)
