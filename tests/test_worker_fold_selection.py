"""Actual fit rows and public records carried by a search into a worker."""
import os
from collections import UserList, namedtuple
from dataclasses import dataclass
from pickle import PicklingError
from types import SimpleNamespace

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import GridSearchCV, KFold, check_cv
from sklearn.model_selection import _search

import dwd.model_selection as selection
from test_reversed_mapped_search import _HeapMapping, _ParentOnlyParallel, _simulated_object_mapping


Fold = namedtuple('Fold', ['train', 'test'])


class _RecordRows(DummyClassifier):
    def fit(self, X, y):
        self.training_sum_ = X[:, 0].sum()
        return super().fit(X, y)


class _ReadRows:
    def __init__(self, search):
        self.search = search

    def __call__(self, estimator, X, y):
        search = self.search
        checked = getattr(search, '_checked_cv_orig', None)
        if hasattr(search.cv, 'split'):
            assert search.cv is search.alias
            assert checked is None or checked is search.cv
            assert search.cv.description == 'public wrapper'
            records = search.cv.cv
        else:
            records = search.cv
        if len(records) == 3:
            assert records[0] is records[2]
        if search.named:
            assert isinstance(records[0], Fold)
            train, test = records[0].train, records[0].test
        else:
            assert isinstance(records[0], np.ndarray)
            train, test = records[0]
        return {'train': estimator.training_sum_, 'test': X[:, 0].sum(),
                'public_train': train.sum(), 'public_test': test.sum(), 'pid': os.getpid()}


class _Reselect(GridSearchCV):
    def _run_search(self, evaluate_candidates):
        evaluate_candidates([{}], cv=self.cv)


class _ChangePublic(GridSearchCV):
    def _run_search(self, evaluate_candidates):
        self.cv[0] = vars(self).pop('replacement')
        self.cv.append(self.cv[0])
        evaluate_candidates([{}])


class _Opaque(GridSearchCV):
    def __getstate__(self):
        return SimpleNamespace(attributes={name: item for name, item in vars(self).items()
                                           if name != '__getstate__'})

    def __setstate__(self, state):
        vars(self).update(state.attributes)


def _private_rows(tmp_path, disk_rows, private_rows):
    path = tmp_path / 'indices.npy'
    np.save(path, disk_rows)
    mapped = np.load(path, mmap_mode='c')
    mapped[:] = private_rows
    return mapped


def _fit(search, jobs):
    rows = np.arange(6)
    search.scoring = _ReadRows(search)
    with parallel_config(backend='loky', n_jobs=jobs, inner_max_num_threads=1):
        assert selection.fit_shared_fold_search(search, rows[:, None], rows % 2) is search
    scores = {metric: [search.cv_results_[f'split{i}_test_{metric}'][0] for i in range(2)]
              for metric in ('train', 'test', 'public_train', 'public_test', 'pid')}
    assert all((pid == os.getpid()) == (jobs == 1) for pid in scores.pop('pid'))
    return scores


@pytest.mark.parametrize('search_type', [GridSearchCV, _Reselect], ids=['constructor', 'reselected'])
def test_selected_public_checked_wrapper(tmp_path, search_type):
    even, odd = np.array([0, 2, 4]), np.array([1, 3, 5])
    mapped = _private_rows(tmp_path, even, odd)
    cv = check_cv([Fold(even, mapped), Fold(mapped, even)])
    cv.description = 'public wrapper'
    for jobs in (1, 2):
        search = search_type(_RecordRows(), {}, cv=cv, n_jobs=jobs, refit=False, error_score='raise')
        search.alias, search.named = cv, True
        scores = _fit(search, jobs)
        assert scores['test'] == [9, 6] and scores['train'] == [6, 9], (
            'workers used different checked rows')
        assert scores['public_test'] == [9, 9] and scores['public_train'] == [6, 6]
        assert search.cv is cv and cv.cv[0].test is cv.cv[1].train is mapped
    np.testing.assert_array_equal(np.load(tmp_path / 'indices.npy'), even)


@pytest.mark.parametrize('record', ['train-row', 'pair-matrix'])
def test_changed_public_records(tmp_path, record):
    even, odd = np.array([0, 2, 4]), np.array([1, 3, 5])
    if record == 'train-row':
        replacement = Fold(_private_rows(tmp_path, even, odd), even)
    else:
        replacement = _private_rows(tmp_path, [even, odd], [odd, even])
    for jobs in (1, 2):
        original = [Fold(even, odd), Fold(odd, even)]
        search = _ChangePublic(_RecordRows(), {}, cv=original, n_jobs=jobs, refit=False,
                               error_score='raise')
        search.replacement = replacement
        search.named = record == 'train-row'
        scores = _fit(search, jobs)
        assert scores['test'] == [9, 6] and scores['train'] == [6, 9]
        assert scores['public_train'] == [9, 9] and scores['public_test'] == [6, 6], (
            'workers used different public rows')
        assert search.cv is original and original[0] is original[2] is replacement


@pytest.mark.parametrize('storage', ['owning', 'required'])
def test_opaque_namedtuple_records(tmp_path, storage):
    even, odd = np.array([0, 2, 4]), np.array([1, 3, 5])
    if storage == 'required':
        odd = _private_rows(tmp_path, even, odd)
    cv = [Fold(even, odd), Fold(odd, even)]
    search = _Opaque(_RecordRows(), {}, cv=cv, n_jobs=2, refit=False, error_score='raise')
    search.named = True
    try:
        scores = _fit(search, 2)
    except (PicklingError, TypeError) as error:
        assert 'Preparing process folds requires __getstate__' in (str(error) + str(error.__cause__))
        if storage == 'owning':
            pytest.fail('safe namedtuple records required pickle-state replacement')
    else:
        assert storage == 'owning', 'opaque state silently transported private mapped rows'
        assert scores == {'train': [6, 9], 'test': [9, 6],
                          'public_train': [6, 6], 'public_test': [9, 9]}
    assert search.cv is cv and cv[0].test is cv[1].train is odd


@pytest.mark.parametrize('search_type', [GridSearchCV, _Reselect], ids=['constructor', 'reselected'])
def test_selected_checked_object_guard(monkeypatch, search_type):
    # Heap storage models the mapping. The executor only advances the first
    # job in the parent, so a failing guard cannot serialize the stand-in.
    monkeypatch.setattr(selection, 'mmap', (selection.mmap, _HeapMapping))
    monkeypatch.setattr(_search, 'Parallel', _ParentOnlyParallel)
    rows = np.arange(6)
    pair = UserList([rows[::2], _simulated_object_mapping(rows[1::2].astype(object))])
    cv = check_cv([pair, Fold(rows[1::2], rows[::2])])
    search = search_type(_RecordRows(), {}, cv=cv, n_jobs=2, refit=False, error_score='raise')
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        with pytest.raises(ValueError, match='file-backed object arrays'):
            selection.fit_shared_fold_search(search, rows[:, None], rows % 2)
    assert search.cv is cv and not search.shared_fold_info['completed']


class _ReplaceWrapper(GridSearchCV):
    def _run_search(self, evaluate_candidates):
        self.original = self.cv
        self.cv = self.replacement
        evaluate_candidates([{}], cv=self.original)


class _ReplacePublicOnly(GridSearchCV):
    def _run_search(self, evaluate_candidates):
        self.cv = KFold(2)
        evaluate_candidates([{}])


def _held_out_rows(estimator, X, y):
    return X[:, 0].sum()


def test_public_splitter_change_keeps_collected_fit_rows(tmp_path):
    even, odd = np.array([0, 2, 4]), np.array([1, 3, 5])
    private = _private_rows(tmp_path, even, odd)
    rows = np.arange(6)
    for jobs in (1, 2):
        search = _ReplacePublicOnly(_RecordRows(), {}, cv=[(even, private), (private, even)],
                                    n_jobs=jobs, scoring=_held_out_rows,
                                    refit=False, error_score='raise')
        with parallel_config(backend='loky', inner_max_num_threads=1):
            selection.fit_shared_fold_search(search, rows[:, None], rows % 2)
        observed = [search.cv_results_[f'split{i}_test_score'][0] for i in range(2)]
        assert observed == [9, 6], 'public splitter change lost the collected fit rows'
        assert type(search.cv) is KFold
    np.testing.assert_array_equal(np.load(tmp_path / 'indices.npy'), even)


class _ReadWrappers:
    def __init__(self, search):
        self.search = search

    def __call__(self, estimator, X, y):
        original = self.search.original
        public = self.search.cv
        assert original is getattr(self.search, '_checked_cv_orig', original)
        assert public is self.search.replacement and original is not public
        for wrapper in (original, public):
            assert wrapper.first is wrapper.cv[0]
        assert original.first.test is public.first.train
        assert original.cv[1] is public.first
        return {'test': X[:, 0].sum(), 'original': original.first.test.sum(),
                'public': public.first.test.sum(), 'pid': os.getpid()}


def test_original_wrapper_after_public_replacement(tmp_path):
    even, odd = np.array([0, 2, 4]), np.array([1, 3, 5])
    private = _private_rows(tmp_path, even, odd)
    original = check_cv([Fold(even, private), Fold(private, even)])
    replacement = check_cv(original.cv[::-1])
    for wrapper in (original, replacement):
        wrapper.first = wrapper.cv[0]
    for jobs in (1, 2):
        search = _ReplaceWrapper(_RecordRows(), {}, cv=original, n_jobs=jobs, refit=False,
                                 error_score='raise')
        search.replacement = replacement
        search.scoring = _ReadWrappers(search)
        rows = np.arange(6)
        with parallel_config(backend='loky', n_jobs=jobs, inner_max_num_threads=1):
            selection.fit_shared_fold_search(search, rows[:, None], rows % 2)
        scores = search.cv_results_
        assert [scores[f'split{i}_test_test'][0] for i in range(2)] == [9, 6], (
            'reselecting original after replacing public cv lost fitting rows')
        for i in range(2):
            assert scores[f'split{i}_test_original'][0] == 9
            assert scores[f'split{i}_test_public'][0] == 6
            assert (scores[f'split{i}_test_pid'][0] == os.getpid()) == (jobs == 1)
        assert search.cv is replacement and search.original is original
    np.testing.assert_array_equal(np.load(tmp_path / 'indices.npy'), even)


class _SelectWeightedWrapper:
    def __init__(self, when):
        self.when = when

    def setup(self, estimator, context):
        if self.when == 'setup':
            estimator.cv = estimator.requested

    def on_fit_task_begin(self, estimator, context, **kwargs):
        if self.when == 'root-begin' and context is estimator._callback_fit_ctx:
            estimator.cv = estimator.requested

    def on_fit_task_end(self, estimator, context, **kwargs):
        pass

    def teardown(self, estimator, context):
        pass


class _WeightedSearch(GridSearchCV):
    def _run_search(self, evaluate_candidates):
        self.selected = self.cv
        self.cv = KFold(2)
        evaluate_candidates([{'constant': 0}, {'constant': 1}])


class _WeightedError:
    def __init__(self, search):
        self.search = search

    def __call__(self, estimator, X, y):
        selected = self.search.selected
        assert selected is self.search._checked_cv_orig is self.search.requested
        assert type(self.search.cv) is KFold
        record = selected.cv[0]
        named = isinstance(record, Fold)
        aliases = named and record is selected.first and record.test is selected.cv[1].train
        weight = getattr(selected, 'odd_weight', 1) if y[0] == 1 else 2
        return {'weighted': -weight * np.abs(estimator.predict(X) - y).mean(),
                'rows': X[:, 0] @ [100, 10, 1], 'named': int(named),
                'aliases': int(aliases), 'pid': os.getpid()}


@pytest.mark.skipif(not hasattr(GridSearchCV, 'set_callbacks'), reason='sklearn callback API')
@pytest.mark.parametrize('when', ['setup', 'root-begin'])
@pytest.mark.usefixtures("search_parent_native_thread")
def test_callback_selected_wrapper_refit(tmp_path, when):
    X = np.arange(6)[:, None]
    even = np.array([0, 2, 4])
    odd = _private_rows(tmp_path, even, [5, 3, 1])
    selected = check_cv([Fold(even, odd), Fold(odd, even)])
    selected.odd_weight, selected.first = 4, selected.cv[0]
    for jobs in (1, 2):
        search = _WeightedSearch(DummyRegressor(strategy='constant'), {'constant': [0, 1]},
                                 cv=2, n_jobs=jobs, refit='weighted', error_score='raise')
        search.requested = selected
        search.scoring = _WeightedError(search)
        search.set_callbacks(_SelectWeightedWrapper(when))
        with parallel_config(backend='loky', inner_max_num_threads=1):
            if jobs == 1:
                search.fit(X, X[:, 0] % 2)
            else:
                selection.fit_shared_fold_search(search, X, X[:, 0] % 2)
        scores = search.cv_results_
        assert search.best_index_ == 1, 'worker fold weight changed the refitted candidate'
        np.testing.assert_array_equal(search.predict(X), np.ones(len(X)))
        np.testing.assert_array_equal(scores['mean_test_weighted'], [-2, -1])
        for i, expected in enumerate((531, 24)):
            assert all(scores[f'split{i}_test_rows'] == expected)
            assert all(scores[f'split{i}_test_named'] == 1)
            assert all(scores[f'split{i}_test_aliases'] == 1)
            assert all((scores[f'split{i}_test_pid'] == os.getpid()) == (jobs == 1))
        assert search.selected is selected and selected.first.test is odd
    np.testing.assert_array_equal(np.load(tmp_path / 'indices.npy'), even)


@dataclass
class _WeightedFold:
    train: np.ndarray
    test: np.ndarray
    weight: int = 7
    consumed: bool = False

    def __iter__(self):
        assert not self.consumed, 'custom record was unpacked twice'
        self.consumed = True
        return iter((self.train, self.test))


class _ReadWrapperRecord:
    def __init__(self, search):
        self.search = search

    def __call__(self, estimator, X, y):
        wrapper = self.search.cv
        record = wrapper.cv[0]
        native = (type(record) is np.ndarray and record.dtype == object if self.search.array_record
                  else isinstance(record, _WeightedFold) and record.weight == 7)
        test = record.test if isinstance(record, _WeightedFold) else record[1]
        return {'native': int(native), 'alias': int(record is wrapper.first),
                'public': test.sum(), 'test': X[:, 0].sum(), 'pid': os.getpid()}


@pytest.mark.parametrize('record_kind', ['object-array', 'dataclass'])
@pytest.mark.parametrize('storage', ['owning', 'private'])
def test_public_wrapper_record_state(tmp_path, record_kind, storage):
    even, odd = np.array([0, 2, 4]), np.array([1, 3, 5])
    if storage == 'private':
        odd = _private_rows(tmp_path, even, odd)
    if record_kind == 'object-array':
        record = np.empty(2, dtype=object)
        record[:] = [even, odd]
    else:
        record = _WeightedFold(even, odd)
    wrapper = check_cv([record, Fold(odd, even)])
    wrapper.first = record
    search = GridSearchCV(_RecordRows(), {}, cv=wrapper, n_jobs=2, refit=False, error_score='raise')
    search.array_record = record_kind == 'object-array'
    search.scoring = _ReadWrapperRecord(search)
    rows = np.arange(6)
    with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
        selection.fit_shared_fold_search(search, rows[:, None], rows % 2)
    scores = search.cv_results_
    for i, held_out in enumerate((9, 6)):
        assert scores[f'split{i}_test_native'][0] == 1, 'worker lost native record type or metadata'
        assert scores[f'split{i}_test_alias'][0] == 1, 'worker lost native record alias'
        assert scores[f'split{i}_test_public'][0] == 9, 'worker lost private record indices'
        assert scores[f'split{i}_test_test'][0] == held_out, 'worker fit lost private record indices'
        assert scores[f'split{i}_test_pid'][0] != os.getpid()
    assert search.cv is wrapper and wrapper.first is record
