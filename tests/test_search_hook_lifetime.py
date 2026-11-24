"""Search hooks during sklearn callback setup, dispatch and cleanup."""
import ast
from collections import namedtuple, OrderedDict
import copyreg
from concurrent.futures import ProcessPoolExecutor
from functools import partial, wraps
from importlib.util import module_from_spec, spec_from_file_location
import inspect
import os
import pickle
import threading
from textwrap import dedent
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from joblib import parallel_config
from joblib.parallel import ParallelBackendBase
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneGroupOut

from dwd.model_selection import fit_shared_fold_search


pytestmark = pytest.mark.usefixtures("search_parent_native_thread")


class _PickleAtSetup:
    def setup(self, estimator, context):
        try:
            restored = pickle.loads(pickle.dumps(estimator))
        except pickle.PicklingError as exc:
            raise AssertionError('setup could not pickle the native search') from exc
        assert type(restored) is GridSearchCV
        assert '_run_search' not in vars(restored) and '__getstate__' not in vars(restored)
        assert restored._run_search.__func__ is GridSearchCV._run_search
        assert restored.cv == estimator.cv
        self.calls += 1
        if self.stop:
            raise RuntimeError('stopped in setup')

    def teardown(self, estimator, context):
        pass

    def on_fit_task_begin(self, estimator, context, **kwargs):
        pass

    def on_fit_task_end(self, estimator, context, **kwargs):
        pass


def _stock(search, X, y, **params):
    return search.fit(X, y, **params)


class _ConstructorTuple(namedtuple('_ConstructorTupleBase', 'token')):
    def __reduce_ex__(self, protocol):
        self.saves += 1
        return type(self), (['worker'],)

    def __iter__(self):
        raise AssertionError('pickling flattened the constructor tuple')


class _ConstructorDict(OrderedDict):
    def __reduce_ex__(self, protocol):
        self.saves += 1
        return type(self), ([('token', ['worker'])],)


def _named_constructor(*args):
    raise AssertionError('pickle must select its NEWOBJ_EX instruction')


_named_constructor.__name__ = '__newobj_ex__'


class _ConstructorSearch(GridSearchCV):
    def __new__(cls, *args, **kwargs):
        result = object.__new__(cls)
        if 'token' in kwargs:
            result.received = kwargs['token'], args[0]
        elif 'folds' in kwargs:
            result.received = args[0], kwargs['folds']
        return result

    def __getnewargs_ex__(self):
        if type(self.argument) is _ConstructorDict:
            return (self.cv,), self.argument
        return self.argument, {'folds': self.cv}

    def __reduce_ex__(self, protocol):
        reduced = super().__reduce_ex__(protocol)
        assert reduced[0] is copyreg.__newobj_ex__
        return (_named_constructor, *reduced[1:]) if self.use_named_callable else reduced

    def __getstate__(self):
        fields = vars(self).copy()
        del fields['cv']  # The constructor is the only source of this public field.
        return fields

    def score_rows(self, estimator, X, y):
        token, folds = self.received
        stored = self.argument['token'] if type(self.argument) is _ConstructorDict else self.argument[0]
        assert token is stored and token == ['worker']
        assert self.scoring.__self__ is self and os.getpid() != self.parent_pid
        rows = {tuple(test) for _, test in folds}
        assert rows == {(2, 4, 1, 3), (7, 0, 6, 5)}
        assert tuple(X[:, 0]) == tuple(y) and tuple(y) in rows
        return float(estimator.constant)


@pytest.mark.parametrize('argument_type', [_ConstructorTuple, _ConstructorDict])
@pytest.mark.parametrize('use_named_callable', [False, True])
def test_newobj_ex_constructor_subclass(tmp_path, argument_type, use_named_callable):
    X = np.arange(8)[:, None]
    wanted = np.array([[2, 4, 1, 3], [7, 0, 6, 5]])
    writes = []
    for fit, backing in ((_stock, wanted), (fit_shared_fold_search, np.zeros_like(wanted))):
        path = tmp_path / (fit.__name__ + '.npy')
        np.save(path, backing)
        mapped = np.load(path, mmap_mode='c')
        mapped[:] = wanted
        a, b = mapped
        folds = [(b, a), (a, b)]
        search = _ConstructorSearch(DummyRegressor(strategy='constant'), {'constant': [0, 1]},
                                    cv=folds, n_jobs=2, pre_dispatch=1, error_score='raise')
        search.argument = (argument_type(['parent']) if argument_type is _ConstructorTuple
                           else argument_type(token=['parent']))
        search.use_named_callable = use_named_callable
        search.parent_pid = os.getpid()
        search.scoring = search.score_rows
        search.argument.saves = 0
        with parallel_config(backend='loky', inner_max_num_threads=1):
            assert fit(search, X, X[:, 0]) is search
        writes.append(search.argument.saves)
        for i in (0, 1):
            np.testing.assert_array_equal(search.cv_results_[f'split{i}_test_score'], [0, 1])
        assert search.best_params_ == {'constant': 1} and search.cv is folds
        assert 'received' not in vars(search) and '__getstate__' not in vars(search)
        np.testing.assert_array_equal(search.predict(X), np.ones(len(X)))
        np.testing.assert_array_equal(mapped, wanted)
        np.testing.assert_array_equal(np.load(path), backing)
    assert writes[0] > 0 and writes[1] == writes[0]


class _MalformedConstructorSearch(GridSearchCV):
    def __reduce_ex__(self, protocol):
        args = [type(self), (), {}]
        args[self.invalid_index] = []
        return copyreg.__newobj_ex__, tuple(args)


class _PositionalFolds(namedtuple('_PositionalFoldsBase', 'first second')):
    def __reduce_ex__(self, protocol):
        return type(self), (self.second, self.first)


_NamedFolds = namedtuple('_NamedFolds', 'first second')


class _RebuiltFolds(_NamedFolds):
    def __reduce_ex__(self, protocol):
        return type(self), (tuple(list(self.first)), tuple(list(self.second)))


class _KeywordFolds(dict):
    def __init__(self, first, second, label=31):
        super().__init__(first=first, second=second, label=label)

    def __iter__(self):
        return iter((self['first'], self['second']))

    def __reduce_ex__(self, protocol):
        return type(self), (self['first'], self['second'], 73)


def _ordinary_constructor(cls, folds, keywords):
    result = object.__new__(cls)
    result.ordinary = folds
    return result


class _ExtendedOpcode:
    @property
    def __name__(self):
        if hasattr(self, 'read'):
            raise AssertionError('pickle looked up the opcode name twice')
        self.read = True
        return '__newobj_ex__'

    def __call__(self, *args):
        raise AssertionError('pickle did not select NEWOBJ_EX')


class _WholeConstructorSearch(GridSearchCV):
    def __new__(cls, *args, **kwargs):
        result = object.__new__(cls)
        if 'positional' in kwargs:
            result.received = args
        if 'label' in kwargs:
            result.label = kwargs['label']
            result.received = kwargs['first'], kwargs['second']
        if not kwargs and len(args) == 2:
            result.ordinary = args[0]
        return result

    def __getnewargs_ex__(self):
        if type(self.cv) is _KeywordFolds:
            return (), self.cv
        return self.cv, {'positional': True}

    def __reduce_ex__(self, protocol):
        reduced = super().__reduce_ex__(protocol)
        assert protocol == self.protocol
        if self.mode == 'ordinary':
            return partial(_ordinary_constructor), *reduced[1:]
        if self.mode == 'newobj':
            return copyreg.__newobj__, *reduced[1:]
        if self.mode == 'named-keyword':
            return _ExtendedOpcode(), *reduced[1:]
        return reduced

    def score_rows(self, model, X, y):
        if self.mode in ('ordinary', 'newobj'):
            assert self.ordinary is self.cv
        elif self.mode in ('rebuilt', 'repeated', 'distinct'):
            first, second = self.received
            if self.mode == 'rebuilt':
                # Protocol 3 flattens the original tuple. NEWOBJ_EX saves
                # the subtype, whose reducer deliberately creates two pairs.
                assert (first is second) == (self.protocol < 4)
            else:
                assert (first is second) == (self.mode == 'repeated')
                assert first is self.cv[0] and second is self.cv[1]
            for pair, state_pair in zip(self.received, self.cv):
                assert pair[0] is state_pair[0] and pair[1] is state_pair[1]
                assert tuple(pair[0]) == (0, 2, 4, 6) and tuple(pair[1]) == (1, 3, 5, 7)
        else:
            assert {id(pair) for pair in self.received} == {id(pair) for pair in self.cv}
            expected = [(0, 2, 4, 6), (1, 3, 5, 7)]
            if self.mode == 'positional' and self.protocol >= 4:
                expected.reverse()
            if self.mode in ('keyword', 'named-keyword'):
                assert self.label == (31 if self.protocol < 4 else 73)
            assert [tuple(train) for train, _ in self.received] == expected
        assert tuple(X[:, 0]) == tuple(y) and tuple(y) in {(0, 2, 4, 6), (1, 3, 5, 7)}
        return float(model.constant)


@pytest.mark.parametrize('protocol', [3, 4, 5])
@pytest.mark.parametrize('mode', ['positional', 'plain-tuple', 'keyword', 'named-keyword', 'ordinary', 'newobj',
                                  'rebuilt', 'repeated', 'distinct'])
def test_newobj_ex_constructor_cv_alias(tmp_path, monkeypatch, protocol, mode):
    from joblib.externals.loky.backend import reduction

    native_dump = reduction.dump
    monkeypatch.setattr(reduction, 'dump', lambda obj, file, reducers=None, **kw:
                        native_dump(obj, file, reducers=reducers, protocol=protocol))
    X = np.arange(8)[:, None]
    wanted = np.array([[0, 2, 4, 6], [1, 3, 5, 7]])
    for fit, backing in ((_stock, wanted), (fit_shared_fold_search, wanted[::-1])):
        path = tmp_path / (fit.__name__ + '.npy')
        np.save(path, backing)
        mapped = np.load(path, mmap_mode='c')
        mapped[:] = wanted
        a, b = mapped
        if mode == 'plain-tuple':
            cv = ((a, b), (b, a))
        elif mode in ('rebuilt', 'repeated', 'distinct'):
            pair = (a, b)
            cls = _RebuiltFolds if mode == 'rebuilt' else _NamedFolds
            cv = cls(pair, tuple(list(pair)) if mode == 'distinct' else pair)
        else:
            cv = (_KeywordFolds if mode in ('keyword', 'named-keyword') else _PositionalFolds)((a, b), (b, a))
        search = _WholeConstructorSearch(DummyRegressor(strategy='constant'), {'constant': [0, 1]},
                                         cv=cv, n_jobs=2, pre_dispatch=1, error_score='raise')
        search.protocol, search.mode = protocol, mode
        search.scoring = search.score_rows
        with parallel_config(backend='loky', inner_max_num_threads=1):
            assert fit(search, X, X[:, 0]) is search
        for i in (0, 1):
            np.testing.assert_array_equal(search.cv_results_[f'split{i}_test_score'], [0, 1])
        np.testing.assert_array_equal(search.predict(X), np.ones(len(X)))
        assert search.cv is cv and '__getstate__' not in vars(search)
        np.testing.assert_array_equal(mapped, wanted)
        np.testing.assert_array_equal(np.load(path), backing)


def test_newobj_ex_constructor_nested_aliases():
    from dwd.model_selection import _constructor_members

    row, prepared = object(), object()
    pair = (row, row)
    other = tuple(list(pair))
    cycle = [pair]
    cycle.append(cycle)
    opaque = partial(tuple, (row,))
    kwargs = {'first': cycle, 'again': cycle, 'other': other, 'map': {pair: pair}, 'opaque': opaque}
    result = _constructor_members(kwargs, [(row, prepared)])
    first = result['first']
    assert first is result['again'] and first[1] is first and first is not cycle
    assert first[0] is not result['other'] and first[0] == result['other'] == (prepared, prepared)
    assert next(iter(result['map'])) is pair and result['map'][pair] is first[0]
    assert result['opaque'] is opaque and opaque.args[0][0] is row
    assert cycle[0] is pair and cycle[1] is cycle and pair[0] is row


def test_newobj_ex_constructor_distinct_snapshots():
    from dwd.model_selection import _WorkerCVFactory

    record, a, b, c = object(), object(), object(), object()
    records = [record] * 3
    rows = [(a, b), (c, b), (a, b)]
    splits = SimpleNamespace(cv=_NamedFolds(record, record), checked=SimpleNamespace(cv=records),
                             rows=rows, worker_cv=rows, fold_arrays={})
    factory = _WorkerCVFactory(splits)
    view = factory.constructor_cv()
    assert view[0] is view[2] and view[1] is rows[1] and view[1] is not view[0]
    assert view is not rows and rows[0] is not rows[2]
    assert all(source is not record for source, _ in factory.records(view))


class _CompanionTuple(tuple):
    def __new__(cls, first, second, bag):
        return super().__new__(cls, (first, second, bag))

    def __iter__(self):
        return iter(tuple.__getitem__(self, slice(2)))

    def __getnewargs__(self):
        return tuple.__getitem__(self, slice(None))


class _CompanionDict(dict):
    def __iter__(self):
        return iter((self['first'], self['second']))


class _PositionalCompanion(namedtuple('_PositionalCompanionBase', 'pair bag stamp')):
    def __reduce_ex__(self, protocol):
        return type(self), (self.pair, self.bag, 73)


class _KeywordCompanion(OrderedDict):
    def __reduce_ex__(self, protocol):
        fields = dict(self, stamp=73)
        return type(self), (fields,)


class _CompanionSearch(GridSearchCV):
    def __new__(cls, /, *args, **kwargs):
        search = object.__new__(cls)
        if 'mode' in kwargs:
            search.constructor_label = kwargs['cls']
        if kwargs.get('mode') == 'positional':
            search.whole = args[0], args[2]
            search.companion = kwargs['pair'], kwargs['bag']
            search.stamp = kwargs.get('stamp')
        if kwargs.get('mode') == 'keyword':
            search.whole = kwargs['first'], kwargs['bag']
            search.companion = args[:2]
            search.stamp = args[2] if len(args) > 2 else None
        return search

    def make_companion(self):
        if self.mode == 'positional':
            fields = dict(mode=self.mode, pair=self.pair, bag=self.bag, cls='fold schema')
            return fields if self.form == 'plain' else _KeywordCompanion(fields, stamp=31)
        return ((self.pair, self.bag) if self.form == 'plain'
                else _PositionalCompanion(self.pair, self.bag, 31))

    def __getnewargs_ex__(self):
        companion = self.stored if self.form == 'retained' else self.make_companion()
        if self.mode == 'positional':
            return self.cv, companion
        return companion, self.cv

    def score_rows(self, model, X, y):
        pair, bag = self.whole
        assert self.constructor_label == 'fold schema'
        assert pair is self.companion[0] is self.pair is next(iter(self.cv))
        assert bag is self.companion[1] is self.bag
        assert bag[0]['pair'] is pair and bag[0]['row'] is pair[0] and bag[1] is bag
        if self.form != 'plain':
            assert self.stamp == self.native_stamp
        assert [tuple(row) for row in pair] == [(0, 2, 4, 6), (1, 3, 5, 7)]
        assert tuple(X[:, 0]) == tuple(y) and tuple(y) in {tuple(row) for row in pair}
        return float(model.constant)


@pytest.mark.parametrize('protocol', [3, 4, 5])
@pytest.mark.parametrize('mode', ['positional', 'keyword'])
@pytest.mark.parametrize('form', ['plain', 'fresh', 'retained'])
def test_newobj_ex_constructor_companion_alias(tmp_path, monkeypatch, mode, protocol, form):
    from joblib.externals.loky.backend import reduction

    native_dump = reduction.dump
    monkeypatch.setattr(reduction, 'dump', lambda obj, file, reducers=None, **kw:
                        native_dump(obj, file, reducers=reducers, protocol=protocol))
    X = np.arange(8)[:, None]
    wanted = np.array([[0, 2, 4, 6], [1, 3, 5, 7]])
    for fit, backing in ((_stock, wanted), (fit_shared_fold_search, wanted[::-1])):
        path = tmp_path / (fit.__name__ + '.npy')
        np.save(path, backing)
        mapped = np.load(path, mmap_mode='c')
        mapped[:] = wanted
        a, b = mapped
        pair = a, b
        bag = [{'pair': pair, 'row': a}]
        bag.append(bag)
        cv = (_CompanionTuple(pair, (b, a), bag) if mode == 'positional'
              else _CompanionDict(first=pair, second=(b, a), bag=bag, mode=mode, cls='fold schema'))
        search = _CompanionSearch(DummyRegressor(strategy='constant'), {'constant': [0, 1]},
                                  cv=cv, n_jobs=2, pre_dispatch=1, error_score='raise')
        search.pair, search.bag, search.mode, search.form = pair, bag, mode, form
        if form == 'retained':
            search.stored = search.make_companion()
        search.native_stamp = pickle.loads(pickle.dumps(search, protocol=protocol)).stamp
        search.scoring = search.score_rows
        with parallel_config(backend='loky', inner_max_num_threads=1):
            assert fit(search, X, X[:, 0]) is search
        for i in (0, 1):
            np.testing.assert_array_equal(search.cv_results_[f'split{i}_test_score'], [0, 1])
        np.testing.assert_array_equal(search.predict(X), np.ones(len(X)))
        assert search.cv is cv and search.bag is bag and bag[0]['pair'] is pair and bag[1] is bag
        np.testing.assert_array_equal(mapped, wanted)
        np.testing.assert_array_equal(np.load(path), backing)


@pytest.mark.parametrize('invalid_index', [1, 2], ids=['positional', 'keyword'])
def test_newobj_ex_constructor_invalid(tmp_path, invalid_index):
    X = np.arange(8)[:, None]
    path = tmp_path / 'indices.npy'
    np.save(path, np.arange(8))
    mapped = np.load(path, mmap_mode='c')
    a, b = mapped[:4], mapped[4:]
    folds = [(a, b), (b, a)]
    search = _MalformedConstructorSearch(DummyRegressor(), {}, cv=folds, n_jobs=2)
    search.invalid_index = invalid_index
    search.scoring = search.score
    with pytest.raises(pickle.PicklingError) as native:
        pickle.dumps(search)
    with parallel_config(backend='loky', inner_max_num_threads=1):
        with pytest.raises(pickle.PicklingError) as process:
            fit_shared_fold_search(search, X, X[:, 0])
    assert str(native.value) in f'{process.value} {process.value.__cause__}'
    assert search.cv is folds and '__getstate__' not in vars(search)


class _ChangingBinding:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        instance.lookups += 1
        return partial(instance.evaluate_one, instance.lookups)


class _ChangingRunSearch(GridSearchCV):
    _run_search = _ChangingBinding()

    def evaluate_one(self, lookup, evaluate_candidates):
        self.calls.append(lookup)
        return evaluate_candidates([{'constant': lookup % 2}])


@pytest.mark.parametrize('backend', ['threading', 'loky'])
def test_nondata_run_search_descriptor(backend):
    X = np.arange(12)[:, None]
    y = X[:, 0] % 2
    folds = list(KFold(2).split(X))
    options = {'inner_max_num_threads': 1} if backend == 'loky' else {}
    observations = []
    with parallel_config(backend=backend, **options):
        for fit in (_stock, fit_shared_fold_search):
            search = _ChangingRunSearch(DummyClassifier(strategy='constant'),
                                        {'constant': [0, 1]}, cv=folds, n_jobs=2,
                                        error_score='raise')
            search.lookups, search.calls = 0, []
            assert fit(search, X, y) is search
            assert '_run_search' not in vars(search)
            observations.append((search.lookups, search.calls, search.best_params_,
                                 search.predict(X).tolist()))
    assert observations[1] == observations[0], 'helper must preserve native method lookup and selection'


class _ChangingFit(GridSearchCV):
    @property
    def fit(self):
        self.fit_lookups += 1
        return super().fit if self.fit_lookups == 1 else self.fit_alternate

    def fit_alternate(self, X, y):
        self.estimator.constant = 1
        return super().fit(X, y)


def test_fit_descriptor_called_once():
    X = np.arange(12)[:, None]
    y = X[:, 0] % 2
    folds = list(KFold(2).split(X))
    with parallel_config(backend='loky', inner_max_num_threads=1):
        for fit in (_stock, fit_shared_fold_search):
            search = _ChangingFit(DummyClassifier(strategy='constant', constant=0),
                                  {}, cv=folds, n_jobs=2, error_score='raise')
            search.fit_lookups = 0
            assert fit(search, X, y) is search
            assert search.fit_lookups == 1, 'fit descriptor must be read once'
            np.testing.assert_array_equal(search.predict(X), np.zeros(len(X)))


@pytest.mark.skipif(not hasattr(GridSearchCV, 'set_callbacks'), reason='sklearn callback API')
@pytest.mark.parametrize('n_jobs', [None, 2])
def test_setup_callback_pickle(n_jobs):
    X = np.arange(12)[:, None]
    y = X[:, 0] % 2
    for fit in (_stock, fit_shared_fold_search):
        callback = _PickleAtSetup()
        callback.calls = 0
        search = GridSearchCV(DummyClassifier(), {}, cv=2, n_jobs=n_jobs)
        search.set_callbacks(callback)
        callback.stop = True
        with pytest.raises(RuntimeError, match='^stopped in setup$'):
            fit(search, X, y)
        assert '_run_search' not in vars(search) and '__getstate__' not in vars(search)
        pickle.loads(pickle.dumps(search))
        callback.stop = False
        assert fit(search, X, y) is search
        assert callback.calls == 2
        assert '_run_search' not in vars(search) and '__getstate__' not in vars(search)
        restored = pickle.loads(pickle.dumps(search))
        np.testing.assert_array_equal(restored.predict(X), search.predict(X))


class _RunSlot(GridSearchCV):
    __slots__ = ('_run_search',)

    def run_candidates(self, evaluate_candidates):
        return evaluate_candidates([{}])

    def __getstate__(self):
        return vars(self).copy()

    def __setstate__(self, state):
        vars(self).update(state)
        self._run_search = self.run_candidates


class _GetstateSlot(GridSearchCV):
    __slots__ = ('__getstate__',)

    def native_state(self):
        return dict(vars(self), cv=())

    def __setstate__(self, state):
        vars(self).update(state)
        self.__getstate__ = self.native_state


def _check_slot(cls, name, method, backend, outcome):
    X = np.arange(12)[:, None]
    y = X[:, 0] % 2
    pairs = list(KFold(2).split(X))
    controls = []
    for fit in (_stock, fit_shared_fold_search):
        search = cls(DummyClassifier(), {}, n_jobs=2, error_score='raise')
        original = getattr(search, method)
        setattr(search, name, original)
        # A named slot takes precedence over this independent dict entry.
        vars(search)[name] = 'caller dictionary value'
        limits = {'inner_max_num_threads': 1} if backend == 'loky' else {}
        with parallel_config(backend=backend, **limits):
            for attempt in (outcome, 'success'):
                search.cv = iter(pairs) if name == '__getstate__' else pairs
                search.estimator.strategy = 'invalid' if attempt == 'failure' else 'most_frequent'
                if attempt == 'failure':
                    with pytest.raises(ValueError, match='strategy'):
                        fit(search, X, y)
                else:
                    assert fit(search, X, y) is search
                assert getattr(search, name) is original, f'original {name} slot was overwritten'
                assert vars(search)[name] == 'caller dictionary value'
                restored = pickle.loads(pickle.dumps(search))
                assert getattr(restored, name).__self__ is restored
                assert getattr(restored, name).__func__ is original.__func__
                assert vars(restored)[name] == 'caller dictionary value'
                if attempt == 'success':
                    np.testing.assert_array_equal(restored.predict(X), search.predict(X))
        controls.append((search.cv_results_['mean_test_score'], search.predict(X)))
    for native, helper in zip(*controls, strict=True):
        np.testing.assert_array_equal(helper, native)


@pytest.mark.parametrize('backend,outcome', [
    ('threading', 'success'), ('threading', 'failure'), ('loky', 'success'), ('loky', 'failure'),
])
def test_run_search_slot(backend, outcome):
    _check_slot(_RunSlot, '_run_search', 'run_candidates', backend, outcome)


@pytest.mark.parametrize('outcome', ['success', 'failure'])
def test_getstate_slot(outcome):
    _check_slot(_GetstateSlot, '__getstate__', 'native_state', 'loky', outcome)


_callback_api = pytest.mark.skipif(not hasattr(GridSearchCV, 'set_callbacks'),
                                   reason='sklearn callback API')


class _Callback:
    def setup(self, estimator, context):
        pass

    def teardown(self, estimator, context):
        pass

    def on_fit_task_begin(self, estimator, context, **kwargs):
        pass

    def on_fit_task_end(self, estimator, context, **kwargs):
        pass


_with_or_without_callbacks = pytest.mark.parametrize('callbacks', [
    pytest.param(False, id='plain'), pytest.param(True, marks=_callback_api, id='callback'),
])


class _BoundCallableFit:
    def __init__(self, function):
        self.function = function
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.function, name)

    def __get__(self, estimator, owner):
        return self if estimator is None else MethodType(self, estimator)

    def __call__(self, estimator, X, y, **params):
        self.calls.append(estimator)
        estimator.estimator.constant = 1
        return self.function(estimator, X, y, **params)


@_with_or_without_callbacks
def test_bound_callable_fit_descriptor(monkeypatch, callbacks):
    from sklearn.model_selection._search import BaseSearchCV

    X = np.arange(12)[:, None]
    descriptor = _BoundCallableFit(BaseSearchCV.fit)
    monkeypatch.setattr(BaseSearchCV, 'fit', descriptor)
    for fit in (_stock, fit_shared_fold_search):
        search = GridSearchCV(DummyClassifier(strategy='constant', constant=0), {}, cv=2, n_jobs=1)
        if callbacks:
            search.set_callbacks(_Callback())
        bound = search.fit
        assert type(bound) is MethodType and bound.__func__ is descriptor
        descriptor.calls.clear()
        assert fit(search, X, X[:, 0] % 2) is search
        assert descriptor.calls == [search], 'bound callable fit must run once on the original search'
        np.testing.assert_array_equal(search.predict(X), np.ones(len(X)))


class _WritableRun(GridSearchCV):
    @property
    def _run_search(self):
        self.reads += 1
        selected = getattr(self, 'operation', self.select_candidates)

        @wraps(selected)
        def logged(*args, **kwargs):
            self.calls += 1
            return selected(*args, **kwargs)

        return logged

    @_run_search.setter
    def _run_search(self, operation):
        self.writes += 1
        self.operation = operation

    def select_candidates(self, evaluate_candidates):
        if self.problem is not None:
            raise self.problem
        return evaluate_candidates([{'constant': 0}, {'constant': 1}])

    def __sklearn_clone__(self):
        raise AssertionError('fit cloned the search')


class _CheckTrainingWeights(DummyRegressor):
    def fit(self, X, y, sample_weight=None):
        np.testing.assert_array_equal(sample_weight, 100 + X[:, 0])
        np.testing.assert_array_equal(y, X[:, 0])
        self.training_rows_ = tuple(y)
        return super().fit(X, y, sample_weight=sample_weight)


class _PropertyState(GridSearchCV):
    @property
    def __getstate__(self):
        self.state_reads += 1
        selected = vars(self).get('selected', self.native_state)

        def getstate():
            self.state_calls += 1
            return selected()

        return getstate

    def native_state(self):
        if self.reject_state:
            raise ValueError('native state sentinel')
        return dict(vars(self), state_owner=id(self))


class _WritablePropertyState(_PropertyState):
    @_PropertyState.__getstate__.setter
    def __getstate__(self, selected):
        self.state_writes += 1
        self.selected = selected


class _PropertyStateAndReducer(_PropertyState):
    @property
    def __reduce_ex__(self):
        return super().__reduce_ex__


class _StateBinding:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return _PropertyState.__getstate__.fget(instance)


class _FreshState(_PropertyState):
    __getstate__ = _StateBinding()


_saving_folds = threading.local()


class _CaptureFoldsOnSave:
    def __init__(self, record):
        self.record = record

    def __reduce__(self):
        self.record['folds'] = _saving_folds.value
        del _saving_folds.value
        return int, ()


class _LateSavedState(GridSearchCV):
    def save_fields(self):
        record = {}
        _saving_folds.value = self.cv
        # The first item adds the fold reference before pickle saves the second.
        return dict(vars(self), audit=[_CaptureFoldsOnSave(record), record])

    def score_rows(self, estimator, X, y):
        assert os.getpid() != self.parent_pid
        saved = self.audit[1]['folds']
        public_rows = tuple(self.cv[0][1])
        assert public_rows == (5, 1, 5, 3)
        same_rows = tuple(saved[0][1]) == public_rows
        return {'preference': float(estimator.constant == int(same_rows)),
                'same_cv': float(saved is self.cv)}


class _LateMethodState(_LateSavedState):
    __getstate__ = _LateSavedState.save_fields


class _LateDynamicState(_LateSavedState):
    def __getattribute__(self, name):
        if name == '__getstate__':
            name = 'save_fields'
        return super().__getattribute__(name)


@pytest.mark.parametrize('search_type', [_LateMethodState, _LateDynamicState])
def test_saving_method_preserves_late_fold_reference(tmp_path, search_type):
    y = np.arange(8)
    even, odd = np.array([0, 2, 4, 6]), np.array([5, 1, 5, 3])
    path = tmp_path / 'late-fold.npy'
    np.save(path, even)
    mapped = np.load(path, mmap_mode='c')
    try:
        mapped[:] = odd
        with parallel_config(backend='loky', inner_max_num_threads=1):
            for fit, test_rows in ((_stock, odd), (fit_shared_fold_search, mapped)):
                cv = [(even, test_rows), (test_rows, even)]
                search = search_type(DummyRegressor(strategy='constant'), {'constant': [0, 1]},
                                     cv=cv, n_jobs=2, error_score='raise', refit='preference')
                search.parent_pid = os.getpid()
                search.scoring = search.score_rows
                assert fit(search, y[:, None], y) is search
                assert search.best_params_ == {'constant': 1}
                np.testing.assert_array_equal(search.cv_results_['mean_test_preference'], [0, 1])
                np.testing.assert_array_equal(search.cv_results_['mean_test_same_cv'], [1, 1])
                np.testing.assert_array_equal(search.predict(y[:, None]), np.ones_like(y))
                assert search.cv is cv and 'audit' not in vars(search)
        np.testing.assert_array_equal(mapped, odd)
        np.testing.assert_array_equal(np.load(path), even)
    finally:
        mapped._mmap.close()


class _FoldHolder:
    def __init__(self, value):
        self.value = value


class _ReviseOnSave:
    def __init__(self, record):
        self.record = record

    def __reduce__(self):
        self.record['version'] = 2
        return int, ()


class _LateFoldState(_PropertyState):
    def native_state(self):
        state = super().native_state()
        record = {}
        _saving_folds.value = self.cv
        shared = [_FoldHolder(1)]
        state['audit'] = [_CaptureFoldsOnSave(record), record, shared, _FoldHolder(shared)]
        first, last = ({'rows': self.cv, 'version': 1} for _ in range(2))
        first_holder, last_holder = _FoldHolder(first), _FoldHolder(last)
        first.update(edit=_ReviseOnSave(first), back=first_holder)
        last.update(edit=_ReviseOnSave(last), back=last_holder)
        state['mixed'] = [first_holder, first['edit'], first, last, last['edit'], last_holder]
        return state


class _OnePassState(_PropertyState):
    def native_state(self):
        shared = [self.cv, ()]
        return dict(super().native_state(), active=shared, holder=_FoldHolder(shared))


class _WriteDuringRestore:
    def __init__(self, records, kind):
        self.records, self.kind = records, kind

    def __getstate__(self):
        return self.records, self.kind

    def __setstate__(self, state):
        self.records, self.kind = state
        assert len(self.records) == 0
        if self.kind == 'list':
            self.records.append('written by hook')
        else:
            self.records[0 if self.kind == 'pending-key' else 'other'] = 'written by hook'


class _InspectRestore(_PropertyState):
    def __setstate__(self, state):
        vars(self).update(state)
        records = self.records if type(self.records) is list else self.records.values()
        self.fold_already_visible = any(type(item) is list for item in records)


class _ScoreEarlyWrite:
    def __init__(self, records, search):
        # The worker reconstructs these records before it visits the search.
        self.records = records
        self.search = search
        self.parent = os.getpid()

    def __call__(self, model, X, y):
        assert self.parent != os.getpid()
        assert tuple(self.search.cv[0][1]) == (5, 1, 5, 3)
        return float(model.constant == int(not self.search.fold_already_visible))


@pytest.mark.parametrize('kind', ['other-key', 'pending-key', 'list'])
def test_property_search_distinguishes_restore_writes(tmp_path, kind):
    y = np.arange(8)
    X = y[:, None]
    even, odd = y[::2], np.array([5, 1, 5, 3])
    path = tmp_path / 'rows.npy'
    np.save(path, even)
    private = np.load(path, mmap_mode='c')
    try:
        private[:] = odd
        with parallel_config(backend='loky', inner_max_num_threads=1):
            for fit, rows in ((_stock, odd), (fit_shared_fold_search, private)):
                pairs = [(even, rows), (rows, even)]
                search = _InspectRestore(DummyRegressor(strategy='constant'), {'constant': [0, 1]},
                                         cv=pairs, n_jobs=2, error_score='raise')
                search.state_reads = search.state_calls = 0
                search.reject_state = False
                records = [] if kind == 'list' else {}
                items = [pairs, _WriteDuringRestore(records, kind), search]
                if kind == 'list':
                    records.extend(items)
                else:
                    records.update(enumerate(items))
                search.records = records
                search.scoring = _ScoreEarlyWrite(records, search)
                fit(search, X, y)
                np.testing.assert_array_equal(search.cv_results_['mean_test_score'], [0, 1])
                assert search.best_params_ == {'constant': 1}
                np.testing.assert_array_equal(search.predict(X), np.ones(len(y)))
    finally:
        private._mmap.close()


class _ObserveRestore:
    def __init__(self, rows):
        self.rows = rows

    def __getstate__(self):
        return self.rows

    def __setstate__(self, rows):
        self.visible = len(rows)
        self.first_is_int = bool(rows) and type(rows[0]) is int
        if type(rows) is dict:
            self.nan_keys = tuple(key for key in rows if type(key) is float and np.isnan(key))
        self.rows = rows


class _ZeroKey:
    def __reduce__(self):
        return int, (0,)


class _BufferedState(_PropertyState):
    def native_state(self):
        state = super().native_state()
        records = [] if self.layout == 'list' else {}
        items = list(range(1101))
        first, second = _ObserveRestore(records), _ObserveRestore(records)
        items[0], items[20], items[1020] = self.cv, first, second
        if self.layout == 'list':
            records.extend(items)
        else:
            for i, item in enumerate(items):
                key = _ZeroKey() if self.layout == 'coalescing-dict' and i == 1000 else i
                if self.layout == 'nan-dict' and i == 5:
                    key = float('nan')
                records[key] = item
        return dict(state, records=records, observers=(first, second))


class _ScoreRestore:
    def __init__(self, search, expected):
        self.search = search
        self.expected = expected
        self.parent = os.getpid()

    def __call__(self, estimator, X, y):
        assert self.parent != os.getpid()
        first, second = self.search.observers
        assert first.rows is second.rows
        if self.search.layout == 'coalescing-dict':
            assert len(first.rows) == 1100 and self.search.records[0] == 1000
        else:
            assert len(first.rows) == 1101 and self.search.records[0] is self.search.cv
        if self.search.layout == 'nan-dict':
            original_key, = (key for key in first.rows if type(key) is float and np.isnan(key))
            assert not first.nan_keys and second.nan_keys == (original_key,)
            assert first.rows[original_key] == 5
        assert tuple(self.search.cv[0][1]) == (5, 1, 5, 3)
        observed = (first.visible, second.visible, second.first_is_int)
        return float(estimator.constant == int(observed == self.expected))


@pytest.mark.parametrize('layout', ['list', 'dict', 'coalescing-dict', 'nan-dict'])
def test_property_search_partial_restore_visibility(tmp_path, layout):
    y = np.arange(8)
    X = y[:, None]
    even, odd = y[::2], np.array([5, 1, 5, 3])
    path = tmp_path / 'indices.npy'
    np.save(path, even)
    mapped = np.load(path, mmap_mode='c')
    try:
        mapped[:] = odd
        with parallel_config(backend='loky', inner_max_num_threads=1):
            for fit, rows in ((_stock, odd), (fit_shared_fold_search, mapped)):
                search = _BufferedState(DummyRegressor(strategy='constant'), {'constant': [0, 1]},
                                        cv=[(even, rows), (rows, even)], n_jobs=2, error_score='raise')
                search.state_reads = search.state_calls = 0
                search.reject_state = False
                search.layout = layout
                native = pickle.loads(pickle.dumps(search)).observers
                expected = (native[0].visible, native[1].visible, native[1].first_is_int)
                assert expected[0] < expected[1]
                search.scoring = _ScoreRestore(search, expected)
                fit(search, X, y)
                np.testing.assert_array_equal(search.cv_results_['mean_test_score'], [0, 1])
                assert search.best_params_ == {'constant': 1}
                np.testing.assert_array_equal(search.predict(X), np.ones(len(y)))
    finally:
        mapped._mmap.close()


class _ScoreOnePass:
    def __init__(self, search):
        self.search = search
        self.parent = os.getpid()

    def __call__(self, estimator, X, y):
        assert self.parent != os.getpid()
        assert self.search.active[0] is self.search.cv and self.search.active[1] == ()
        first, second = self.search.holder.value
        assert type(first) is type(iter([]))
        return float(estimator.constant == int(second == ()))


def test_property_search_one_pass_folds_and_empty_metadata():
    y = np.arange(8)
    X = y[:, None]
    even, odd = y[::2], y[1::2]
    with parallel_config(backend='loky', inner_max_num_threads=1):
        for fit in (_stock, fit_shared_fold_search):
            search = _OnePassState(DummyRegressor(strategy='constant'), {'constant': [0, 1]},
                                   cv=iter([(even, odd), (odd, even)]), n_jobs=2, error_score='raise')
            search.scoring = _ScoreOnePass(search)
            search.state_reads = search.state_calls = 0
            search.reject_state = False
            assert fit(search, X, y) is search
            np.testing.assert_array_equal(search.cv_results_['mean_test_score'], [0, 1])
            assert search.best_params_ == {'constant': 1}
            restored = pickle.loads(pickle.dumps(search))
            np.testing.assert_array_equal(restored.predict(X), np.ones(len(y)))


def _allocate_search(cls, fold_record):
    search = object.__new__(cls)
    search.constructed_record = fold_record
    search.list_items = []
    return search


def _restore_search_state(search, state):
    assert state.pop('items_exhausted')
    blob, alias, distinct = state.pop('blobs')
    assert blob is alias and blob is not distinct and blob == distinct
    record = state.pop('fold_record')
    first, repeated, listed_record, listed_cv = search.list_items
    assert first is repeated and first['ready'] and first['cv'] is listed_cv
    assert search.first_value is search.second_value and search.first_value['ready']
    assert search.first_value['cv'] is listed_cv
    assert search.nested[1]['saved'] and search.nested[1]['cv'] is listed_cv
    assert search.changing['end'] == state['expected_last_value']
    assert record is search.constructed_record is listed_record is search.dict_record
    assert record[0] is state['cv'] is listed_cv is search.dict_cv
    vars(search).update(state)


class _MarkOnSave:
    def __init__(self, record, fail):
        self.record = record
        self.fail = fail

    def __reduce__(self):
        if self.fail:
            raise ValueError('nested serialization sentinel')
        self.record['saved'] = True
        return int, ()


class _ReinsertOnSave:
    def __init__(self, record):
        self.record = record

    def __reduce__(self):
        self.record.pop('end')
        self.record['end'] = 42
        return int, ()


def _changing_record(cv):
    record = {'cv': cv}
    record['end'] = _ReinsertOnSave(record)
    return record


class _ReductionStreams(_PropertyState):
    def __reduce_ex__(self, protocol):
        state = self.__getstate__()
        record = state['fold_record'] = [self.cv]
        blob = bytes(65536)
        state['blobs'] = (blob, blob, bytes(bytearray(blob)))

        def items():
            # Pickle can fetch the second item before serializing the first.
            stream_record = {'cv': self.cv}
            yield stream_record
            stream_record['ready'] = True
            yield stream_record
            yield record
            yield self.cv
            state['items_exhausted'] = True

        def pairs():
            stream_record = {'cv': self.cv}
            yield 'first_value', stream_record
            stream_record['ready'] = True
            yield 'second_value', stream_record
            yield 'dict_record', record
            yield 'dict_cv', self.cv
            nested = {'cv': self.cv}
            yield 'nested', [_MarkOnSave(nested, self.fail_in_value), nested]
            yield 'changing', _changing_record(self.cv)

        return _allocate_search, (type(self), record), state, items(), pairs(), _restore_search_state

    def append(self, item):
        self.list_items.append(item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class _ScorePropertyFolds:
    def __init__(self, search):
        self.search = search
        self.owner = id(search)
        self.parent_pid = os.getpid()
        if isinstance(search, _LateFoldState):
            self.private_rows = isinstance(search.cv[0][1], np.memmap)

    def __call__(self, estimator, X, y, *, sample_weight=None):
        assert os.getpid() != self.parent_pid
        assert self.search.state_owner == self.owner, 'pickle hook ran on another search'
        even, odd = (0, 2, 4, 6), (5, 1, 5, 3)
        expected = [(even, odd), (odd, even)]
        for pairs in (self.search.cv, getattr(self.search, '_checked_cv_orig', None)):
            if pairs is not None:
                rows = getattr(pairs, 'cv', pairs)
                assert [(tuple(train), tuple(test)) for train, test in rows] == expected, 'stale worker folds'
        assert (estimator.training_rows_, tuple(y)) in expected
        np.testing.assert_array_equal(X[:, 0], y)
        if sample_weight is not None:
            np.testing.assert_array_equal(sample_weight, 100 + y)
        if isinstance(self.search, _LateFoldState):
            _, record, shared, holder = self.search.audit
            assert shared is holder.value
            hidden_first, _, first, last, _, hidden_last = self.search.mixed
            assert first['rows'] is last['rows'] is self.search.cv
            assert [item['version'] for item in (hidden_first.value, first, last, hidden_last.value)] == [1] * 4
            for hidden in (hidden_first, hidden_last):
                # Independent opaque objects keep the backend's native rows.
                assert tuple(hidden.value['rows'][0][1]) == (even if self.private_rows else odd)
            late_rows = [(tuple(train), tuple(test)) for train, test in record['folds']]
            if late_rows != expected:
                return float(estimator.constant == 0)
            assert record['folds'] is self.search.cv
        labels = y % 2
        return -np.mean(np.where(labels, 5, 2) * np.abs(labels - estimator.predict(X)))


class _ScatterOnlyArray(np.ndarray):
    def __reduce_ex__(self, protocol):
        raise TypeError('the backend must scatter the original feature array')


def _nested_array(values):
    return values.view(_ScatterOnlyArray)


def _send_nested_array(values):
    return _nested_array, (np.asarray(values),)


class _BackendResource:
    def __init__(self, is_worker=False):
        self.is_worker = is_worker

    def __reduce_ex__(self, protocol):
        raise TypeError('this resource requires its executor reducer')


def _send_resource(resource):
    return _BackendResource, (True,)


def _reopen_file(mapping):
    return np.load, (str(mapping.filename), 'r')


def _run_calls(tasks):
    return [function(*args, **kwargs) for function, args, kwargs in tasks]


class _ScatteringBackend(ParallelBackendBase):
    supports_retrieve_callback = True

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.transfers = 0

    def effective_n_jobs(self, n_jobs):
        return n_jobs

    def configure(self, n_jobs, parallel, **kwargs):
        self.pool = ProcessPoolExecutor(max_workers=n_jobs)
        return n_jobs

    def submit(self, batch, callback=None):
        def scatter(value):
            if value is self.source:
                self.transfers += 1
                return np.asarray(value)
            return value

        tasks = [(fn, tuple(map(scatter, args)), {key: scatter(value) for key, value in kwargs.items()})
                 for fn, args, kwargs in batch.items]
        future = self.pool.submit(_run_calls, tasks)
        future.add_done_callback(callback)
        return future

    def retrieve_result_callback(self, future):
        return future.result()

    def terminate(self):
        self.pool.shutdown()


class _ResourceScore(_ScorePropertyFolds):
    def __call__(self, estimator, X, y, *, sample_weight=None):
        assert self.search.resource.is_worker
        assert type(X) is np.ndarray
        assert type(self.original_features) is _ScatterOnlyArray
        return super().__call__(estimator, X, y, sample_weight=sample_weight)


def test_property_search_preserves_backend_transport(tmp_path, monkeypatch):
    from multiprocessing.reduction import ForkingPickler

    monkeypatch.setitem(ForkingPickler._extra_reducers, _BackendResource, _send_resource)
    monkeypatch.setitem(ForkingPickler._extra_reducers, np.memmap, _reopen_file)
    monkeypatch.setitem(ForkingPickler._extra_reducers, _ScatterOnlyArray, _send_nested_array)
    y = np.arange(8)
    X = y[:, None].view(_ScatterOnlyArray)
    even, odd = np.array([0, 2, 4, 6]), np.array([5, 1, 5, 3])
    path = tmp_path / 'private.npy'
    np.save(path, even)
    private = np.load(path, mmap_mode='c')
    try:
        private[:] = odd
        for fit, rows in ((_stock, private), (_stock, odd), (fit_shared_fold_search, private)):
            search = _PropertyStateAndReducer(_CheckTrainingWeights(strategy='constant'),
                                              {'constant': [0, 1]}, cv=[(even, rows), (rows, even)],
                                              n_jobs=2, error_score='raise')
            search.state_reads = search.state_calls = 0
            search.reject_state = False
            search.resource = _BackendResource()
            search.scoring = _ResourceScore(search)
            search.scoring.original_features = X
            backend = _ScatteringBackend(X)
            with parallel_config(backend=backend):
                if fit is _stock and rows is private:
                    with pytest.raises(AssertionError, match='stale worker folds'):
                        fit(search, X, y, sample_weight=100 + y)
                    continue
                fit(search, X, y, sample_weight=100 + y)
            assert backend.transfers == 4
            np.testing.assert_array_equal(search.cv_results_['split0_test_score'], [-5, 0])
            np.testing.assert_array_equal(search.cv_results_['split1_test_score'], [0, -2])
            assert search.best_params_ == {'constant': 1}
            assert not search.resource.is_worker
    finally:
        private._mmap.close()


@pytest.mark.parametrize(('cls', 'backend'), [
    pytest.param(_PropertyState, 'loky', id='read-only'),
    pytest.param(_WritablePropertyState, 'loky', id='writable'),
    pytest.param(_FreshState, 'loky', id='fresh-binding'),
    pytest.param(_PropertyStateAndReducer, 'loky', id='two-properties'),
    pytest.param(_PropertyStateAndReducer, 'multiprocessing', id='multiprocessing'),
    pytest.param(_ReductionStreams, 'loky', id='reduction-streams'),
    pytest.param(_LateFoldState, 'loky', id='late-fold-state'),
])
def test_getstate_property_process_folds(tmp_path, cls, backend):
    y = np.arange(8)
    X = y[:, None]
    even, odd = np.array([0, 2, 4, 6]), np.array([5, 1, 5, 3])
    path = tmp_path / 'private-rows.npy'
    np.save(path, even)
    private = np.load(path, mmap_mode='c')
    private[:] = odd
    options = {'inner_max_num_threads': 1} if backend == 'loky' else {}
    with parallel_config(backend=backend, **options):
        for fit, rows in ((_stock, odd), (fit_shared_fold_search, private)):
            pairs = [(even, rows), (rows, even)]
            search = cls(_CheckTrainingWeights(strategy='constant'), {'constant': [0, 1]},
                         cv=pairs, n_jobs=2, error_score='raise')
            search.scoring = _ScorePropertyFolds(search)
            search.state_reads = search.state_calls = search.state_writes = 0
            search.reject_state = False
            if cls is _ReductionStreams:
                search.expected_last_value = pickle.loads(pickle.dumps(_changing_record([(even, odd)])))['end']
                search.fail_in_value = False
            before = pickle.loads(pickle.dumps(search))
            assert before.state_owner == id(search) and before.scoring.search is before
            assert fit(search, X, y, sample_weight=100 + y) is search
            assert search.state_writes == 0 and 'selected' not in vars(search), 'pickle property setter ran'
            np.testing.assert_array_equal(search.cv_results_['split0_test_score'], [-5, 0])
            np.testing.assert_array_equal(search.cv_results_['split1_test_score'], [0, -2])
            assert search.best_params_ == {'constant': 1} and search.best_score_ == -1
            saved = pickle.loads(pickle.dumps(search))
            assert saved.state_owner == id(search) and saved.scoring.search is saved
            for fitted in (search, saved):
                np.testing.assert_array_equal(fitted.predict(X), np.ones(len(X)))
                assert fitted.best_estimator_.training_rows_ == tuple(y)

            search.reject_state = True
            with pytest.raises((pickle.PicklingError, ValueError)) as caught:
                fit(search, X, y, sample_weight=100 + y)
            assert 'native state sentinel' in f'{caught.value} {caught.value.__cause__}'
            if fit is _stock:
                native_error_type = type(caught.value)
            else:
                assert type(caught.value) is native_error_type
            search.reject_state = False
            assert search.state_writes == 0 and 'selected' not in vars(search)
            assert search.state_reads == search.state_calls, 'pickle property was looked up speculatively'
            assert search.cv is pairs and search.cv[0][1] is rows
            assert pickle.loads(pickle.dumps(search)).state_owner == id(search)
            if cls is _ReductionStreams:
                threads = set(threading.enumerate())
                search.fail_in_value = True
                with pytest.raises((pickle.PicklingError, ValueError)) as nested:
                    fit(search, X, y, sample_weight=100 + y)
                assert 'nested serialization sentinel' in f'{nested.value} {nested.value.__cause__}'
                search.fail_in_value = False
                assert fit(search, X, y, sample_weight=100 + y) is search
                assert search.best_params_ == {'constant': 1}
                assert not any(t.name == 'DWD pickle cursor' for t in set(threading.enumerate()) - threads)
    np.testing.assert_array_equal(private, odd)
    np.testing.assert_array_equal(np.load(path), even)


class _WeightedFoldScore(BaseEstimator):
    def __init__(self):
        self.parent_pid = os.getpid()

    def score(self, estimator, X, y, *, sample_weight=None):
        assert os.getpid() != self.parent_pid
        even, odd = (0, 2, 4, 6), (5, 1, 5, 3)
        assert (estimator.training_rows_, tuple(y)) in ((even, odd), (odd, even))
        np.testing.assert_array_equal(X[:, 0], y)
        labels = y % 2
        np.testing.assert_array_equal(sample_weight, np.where(labels, 5, 2))
        return -np.mean(sample_weight * np.abs(labels - estimator.predict(X)))

    __call__ = score


@_with_or_without_callbacks
def test_writable_run_search_descriptor(tmp_path, callbacks):
    X = np.arange(8)[:, None]
    y = X[:, 0]
    even, odd = np.array([0, 2, 4, 6]), np.array([5, 1, 5, 3])
    path = tmp_path / 'private-rows.npy'
    np.save(path, even)
    private = np.load(path, mmap_mode='c')
    private[:] = odd
    params = dict(training=100 + y, cost=np.where(y % 2, 5, 2))
    native_reads = []
    with (config_context(enable_metadata_routing=True),
          parallel_config(backend='loky', inner_max_num_threads=1)):
        # Native processes receive owning rows with the same intended contents.
        for fit, rows in ((_stock, odd), (fit_shared_fold_search, private)):
            pairs = [(even, rows), (rows, even)]
            model = _CheckTrainingWeights(strategy='constant').set_fit_request(sample_weight='training')
            scorer = _WeightedFoldScore().set_score_request(sample_weight='cost')
            search = _WritableRun(model, {}, cv=pairs, n_jobs=2, scoring=scorer, error_score='raise')
            if callbacks:
                search.set_callbacks(_Callback())
            search.reads = search.writes = search.calls = 0
            for attempt, problem in enumerate((ValueError('run sentinel'), None), start=1):
                search.problem = problem
                if problem is not None:
                    with pytest.raises(ValueError) as caught:
                        fit(search, X, y, **params)
                    assert caught.value is problem
                else:
                    assert fit(search, X, y, **params) is search
                assert search.calls == attempt, 'writable run method must execute once per fit'
                assert search.writes == 0 and 'operation' not in vars(search)
                if fit is _stock:
                    native_reads.append(search.reads)
                else:
                    assert search.reads == native_reads[attempt - 1]
                assert search.cv is pairs and search.estimator is model
                restored = pickle.loads(pickle.dumps(search))
                assert restored.calls == attempt and 'operation' not in vars(restored)
            np.testing.assert_array_equal(search.cv_results_['split0_test_score'], [-5, 0])
            np.testing.assert_array_equal(search.cv_results_['split1_test_score'], [0, -2])
            assert search.best_params_ == {'constant': 1} and search.best_score_ == -1
            for fitted in (search, restored):
                np.testing.assert_array_equal(fitted.predict(X), np.ones(len(X)))
            np.testing.assert_array_equal(rows, odd)
            np.testing.assert_array_equal(np.load(path), even)


@_callback_api
def test_callback_class_fit_wrapper_is_called_once(monkeypatch):
    X = np.arange(12)[:, None]
    y = X[:, 0] % 2
    search = GridSearchCV(DummyClassifier(), {}, cv=2, n_jobs=1).set_callbacks(_Callback())
    original_fit = GridSearchCV.fit
    calls = []

    @wraps(original_fit)
    def fitted_with_logging(estimator, features, targets, **params):
        assert estimator is search and features is X and targets is y
        calls.append(estimator)
        return original_fit(estimator, features, targets, **params)

    monkeypatch.setattr(GridSearchCV, 'fit', fitted_with_logging)
    assert fit_shared_fold_search(search, X, y) is search
    assert calls == [search], 'class-level fit wrapper must run once'
    assert search.best_score_ == 0.5


@_callback_api
def test_callback_fit_bound_to_another_search():
    X = np.arange(12)[:, None]
    y = X[:, 0] % 2
    inner = GridSearchCV(DummyClassifier(strategy='constant', constant=1), {}, cv=2, n_jobs=1)
    outer = GridSearchCV(DummyClassifier(strategy='constant', constant=0), {}, cv=2, n_jobs=1)
    outer.set_callbacks(_Callback())
    outer.fit = inner.fit
    assert fit_shared_fold_search(outer, X, y) is inner, 'the delegated fit must use its own receiver'
    assert not hasattr(outer, 'best_estimator_')
    np.testing.assert_array_equal(inner.predict(X), np.ones(len(X)))


@_callback_api
def test_callback_callable_fit_proxy_runs_once():
    calls = []

    class LoggingFit:
        def __init__(self, method):
            self.method = method

        def __getattribute__(self, name):
            # A transparent proxy also forwards __class__ to the bound method.
            return getattr(object.__getattribute__(self, 'method'), name)

        def __call__(self, *args, **kwargs):
            calls.append(True)
            return object.__getattribute__(self, 'method')(*args, **kwargs)

    X = np.arange(12)[:, None]
    search = GridSearchCV(DummyClassifier(), {}, cv=2, n_jobs=1).set_callbacks(_Callback())
    search.fit = LoggingFit(search.fit)
    assert fit_shared_fold_search(search, X, X[:, 0] % 2) is search
    assert calls == [True], 'the callable fit proxy must run once'
    assert search.best_score_ == 0.5


@_callback_api
def test_native_process_folds_after_class_fit_restored(tmp_path, monkeypatch):
    from sklearn.model_selection._search import BaseSearchCV

    original = BaseSearchCV.fit
    calls = []

    @wraps(original)
    def logged(estimator, X, y, **kwargs):
        calls.append(estimator)
        return original(estimator, X, y, **kwargs)

    X = np.arange(6)[:, None]
    y = X[:, 0] % 2
    with monkeypatch.context() as active:
        active.setattr(BaseSearchCV, 'fit', logged)
        # Load an independent copy while logging is active, without reloading
        # the module used by other tests or changing worker imports.
        spec = spec_from_file_location('dwd._fit_restore_test', inspect.getfile(fit_shared_fold_search))
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        serial = GridSearchCV(DummyClassifier(), {}, cv=2, n_jobs=1).set_callbacks(_Callback())
        assert module.fit_shared_fold_search(serial, X, y) is serial and calls == [serial]

    even = np.array([0, 2, 4])
    path = tmp_path / 'indices.npy'
    np.save(path, even)
    odd = np.load(path, mmap_mode='c')
    odd[:] = [5, 3, 1]
    process = GridSearchCV(DummyClassifier(), {}, cv=[(even, odd), (odd, even)], n_jobs=2,
                           scoring=_HeldOutSum(True), refit=False, error_score='raise')
    process.set_callbacks(_Callback())
    with parallel_config(backend='loky', inner_max_num_threads=1):
        assert module.fit_shared_fold_search(process, X, y) is process
    scores = [process.cv_results_[f'split{i}_test_score'][0] for i in range(2)]
    assert scores == [9, 6], 'restored stock fit lost private fold edits'
    assert calls == [serial]
    np.testing.assert_array_equal(np.load(path), even)


@_callback_api
def test_callback_fit_signature_binding():
    # The helper rebinds signature only in the stock fit's private globals.
    # Check that a new sklearn version has not added a different inspection.
    function = ast.parse(dedent(inspect.getsource(GridSearchCV.fit.__wrapped__)))
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'signature']
    expected = ast.parse("'callback_ctx' in signature(self._run_search).parameters", mode='eval').body
    assert len(calls) == 1 and any(ast.dump(node) == ast.dump(expected)
                                   for node in ast.walk(function) if isinstance(node, ast.Compare))


class _CheckpointFolds(KFold):
    def get_n_splits(self, X=None, y=None, groups=None):
        try:
            self.checkpoint = pickle.dumps(self.search)
        except (AttributeError, pickle.PicklingError) as error:
            raise AssertionError('split-count checkpoint exposed a temporary search hook') from error
        if self.problem is not None:
            raise self.problem
        return super().get_n_splits(X, y, groups)


class _OwnFolds(_Callback):
    def setup(self, estimator, context):
        estimator.cv.search = estimator

    def teardown(self, estimator, context):
        estimator.teardowns += 1


class _RememberState(_OwnFolds):
    def setup(self, estimator, context):
        super().setup(estimator, context)
        self.state_view = vars(estimator)


@_callback_api
@pytest.mark.parametrize('backend,n_jobs', [('loky', 1), ('threading', 2), ('loky', 2)])
def test_callback_checkpoint_with_stored_state_view(backend, n_jobs):
    X = np.arange(12)[:, None]
    for fit in (_stock, fit_shared_fold_search):
        callback, cv = _RememberState(), _CheckpointFolds(2)
        cv.problem = None
        search = GridSearchCV(DummyClassifier(), {}, cv=cv, n_jobs=n_jobs).set_callbacks(callback)
        search.teardowns = 0
        limits = {'inner_max_num_threads': 1} if backend == 'loky' else {}
        with parallel_config(backend=backend, **limits):
            assert fit(search, X, X[:, 0] % 2) is search
        assert search.teardowns == 1 and search.best_score_ == 0.5
        assert callback.state_view is vars(search)
        restored = pickle.loads(cv.checkpoint)
        attrs = restored._skl_callbacks[0].state_view
        assert attrs['cv'] is restored.cv and restored.cv.search is restored
        assert attrs['_skl_callbacks'][0] is restored._skl_callbacks[0]
        assert not {'_run_search', '_init_callback_context', '__reduce_ex__'}.intersection(attrs)


class _ProtectedReducer(GridSearchCV):
    def __setattr__(self, name, value):
        if name == '__reduce_ex__':
            raise AttributeError('reducer is immutable')
        super().__setattr__(name, value)


@_callback_api
def test_callback_preserves_native_reducer_contract():
    X = np.arange(12)[:, None]
    y = X[:, 0] % 2
    for fit in (_stock, fit_shared_fold_search):
        cv = _CheckpointFolds(2)
        cv.problem = None
        search = GridSearchCV(DummyClassifier(), {}, cv=cv, n_jobs=1).set_callbacks(_OwnFolds())
        search.teardowns = 0
        search.__reduce_ex__ = None
        with pytest.raises(TypeError, match="'NoneType' object is not callable"):
            fit(search, X, y)
        assert search.teardowns == 1 and search.__reduce_ex__ is None
        protected = _ProtectedReducer(DummyClassifier(), {}, cv=2, n_jobs=1).set_callbacks(_Callback())
        assert fit(protected, X, y) is protected and protected.best_score_ == 0.5


@_callback_api
@pytest.mark.parametrize('n_jobs', [1, 2])
def test_callback_splitter_checkpoint(n_jobs):
    X = np.arange(12)[:, None]
    y = X[:, 0] % 2
    for fit in (_stock, fit_shared_fold_search):
        cv = _CheckpointFolds(2)
        cv.problem = ValueError('split-count sentinel')
        search = GridSearchCV(DummyClassifier(), {}, cv=cv, n_jobs=n_jobs).set_callbacks(_OwnFolds())
        search.teardowns = 0
        with pytest.raises(ValueError) as caught:
            fit(search, X, y)
        assert caught.value is cv.problem and search.teardowns == 1
        restored = pickle.loads(cv.checkpoint)
        assert restored.cv.search is restored and restored.cv is restored._checked_cv_orig
        assert '_run_search' not in vars(restored) and '_init_callback_context' not in vars(restored)
        assert restored._skl_callbacks_to_teardown == restored._skl_callbacks
        cv.problem = restored.cv.problem = None
        with parallel_config(backend='loky', inner_max_num_threads=1):
            assert fit(search, X, y) is search and search.teardowns == 2
            assert fit(restored, X, y) is restored and restored.teardowns == 1
        np.testing.assert_array_equal(restored.predict(X), search.predict(X))
        assert restored.best_score_ == search.best_score_ == 0.5


class _CheckpointClone(DummyClassifier):
    def __sklearn_clone__(self):
        if hasattr(self, 'search'):
            try:
                self.checkpoint = pickle.dumps(self.search, protocol=4)
            except (AttributeError, pickle.PicklingError) as error:
                raise AssertionError('clone checkpoint exposed a temporary search hook') from error
        return super().__sklearn_clone__()


class _ReductionTuple(tuple):
    def __iter__(self):
        raise AssertionError('pickling should read native reduction tuple fields directly')


class _NativeReduction(GridSearchCV):
    def reduce_search(self, protocol):
        assert protocol == 4 and self.__reduce_ex__ == self.reduce_search
        assert '_run_search' not in vars(self), 'clone checkpoint exposed a temporary search hook'
        assert self._skl_callbacks_to_teardown == self._skl_callbacks
        self.reductions += 1
        return _ReductionTuple(super().__reduce_ex__(protocol))

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__ = state


@_callback_api
def test_callback_clone_checkpoint_with_native_reducer():
    X = np.arange(12)[:, None]
    y = X[:, 0] % 2
    for fit in (_stock, fit_shared_fold_search):
        search = _NativeReduction(_CheckpointClone(), {}, cv=list(KFold(2).split(X)), n_jobs=1)
        search.set_callbacks(_Callback())
        search.reductions = 0
        search.__reduce_ex__ = original = search.reduce_search
        search.estimator.search = search
        search.alias = [(vars(search),)]
        assert fit(search, X, y) is search and search.reductions == 1
        assert search.__reduce_ex__ is original and search.alias[0][0] is vars(search)
        restored = pickle.loads(search.estimator.checkpoint)
        assert restored.estimator.search is restored and restored.alias[0][0] is vars(restored)
        assert restored.__reduce_ex__ == restored.reduce_search and '_run_search' not in vars(restored)
        assert fit(restored, X, y) is restored and restored.reductions == 2
        np.testing.assert_array_equal(restored.predict(X), search.predict(X))


class _ReadOnlyState(GridSearchCV):
    @property
    def __getstate__(self):
        self.state_reads += 1
        return super().__getstate__


class _EmptySlotState(GridSearchCV):
    __slots__ = ('__getstate__',)


class _ReadOnlyReducer(GridSearchCV):
    @property
    def __reduce_ex__(self):
        self.state_reads += 1
        return super().__reduce_ex__


class _ReadOnlyRun(GridSearchCV):
    @property
    def _run_search(self):
        return super()._run_search


@_callback_api
@pytest.mark.parametrize('cls', [_ReadOnlyState, _EmptySlotState, _ReadOnlyReducer, _ReadOnlyRun],
                         ids=['property', 'empty-slot', 'read-only-reducer', 'read-only-run'])
def test_callback_without_serialization(cls):
    X = np.arange(12)[:, None]
    for fit in (_stock, fit_shared_fold_search):
        search = cls(DummyClassifier(), {}, cv=2, n_jobs=2).set_callbacks(_Callback())
        search.state_reads = 0
        with parallel_config(backend='threading'):
            assert fit(search, X, X[:, 0] % 2) is search
        assert search.state_reads == 0
        if cls is _EmptySlotState:
            assert not hasattr(search, '__getstate__')


class _PickleCycle(_Callback):
    def check(self, estimator, context):
        assert estimator._callback_fit_ctx is context
        try:
            restored = pickle.loads(pickle.dumps(estimator))
        except (pickle.PicklingError, AttributeError) as error:
            raise AssertionError('native callback pickle lost the attribute dictionary cycle') from error
        attrs = restored.state_alias
        assert attrs['state_alias'] is attrs and attrs['self'] is restored
        for name in ('_init_callback_context', '_run_search', '__getstate__'):
            assert name not in attrs and name not in vars(restored)

    def setup(self, estimator, context):
        self.check(estimator, context)

    def on_fit_task_begin(self, estimator, context, **kwargs):
        if context is estimator._callback_fit_ctx:
            self.check(estimator, context)
            self.root_begins += 1


@_callback_api
def test_setup_pickle_with_attribute_cycle():
    X = np.arange(12)[:, None]
    for fit in (_stock, fit_shared_fold_search):
        callback = _PickleCycle()
        callback.root_begins = 0
        search = GridSearchCV(DummyClassifier(), {}, cv=2).set_callbacks(callback)
        search.self, search.state_alias = search, vars(search)
        assert fit(search, X, X[:, 0] % 2) is search
        assert callback.root_begins == 1 and search.state_alias is vars(search)


class _PropagatedPickle(_PickleCycle):
    max_propagation_depth = 1

    def setup(self, estimator, context):
        if hasattr(estimator, '_parent_callback_ctx'):
            raise AssertionError('propagated callback was set up twice')


@_callback_api
def test_propagated_root_pickle():
    X = np.arange(12)[:, None]
    for fit in (_stock, fit_shared_fold_search):
        callback = _PropagatedPickle()
        callback.root_begins = 0
        parent = GridSearchCV(DummyClassifier(), {}).set_callbacks(callback)
        parent_context = parent._init_callback_context(max_subtasks=1)
        leaf = parent_context.subcontext(task_name='child')
        search = GridSearchCV(DummyClassifier(), {}, cv=2)
        search.self, search.state_alias = search, vars(search)
        with leaf.propagate_callback_context(search):
            assert fit(search, X, X[:, 0] % 2) is search
        assert callback.root_begins == 1 and search.state_alias is vars(search)
        assert not hasattr(search, '_parent_callback_ctx') and not hasattr(search, '_skl_callbacks')
        assert '_run_search' not in vars(search) and '_init_callback_context' not in vars(search)


class _ChangedMethod(GridSearchCV):
    def selected_method(self, evaluate_candidates):
        self.method_calls += 1
        return evaluate_candidates([{}])


class _SelectFolds(_Callback):
    def setup(self, estimator, context):
        held_out = np.load(self.path, mmap_mode='c')
        held_out[:] = [1, 3, 5]
        other = np.array([0, 2, 4])
        estimator.cv = [(other, held_out), (held_out, other)]
        estimator.n_jobs = self.jobs


class _SelectFoldsAndMethod(_SelectFolds):
    def on_fit_task_begin(self, estimator, context, **kwargs):
        if context is estimator._callback_fit_ctx:
            estimator._run_search = estimator.selected_method


class _SelectJobsOnClone(DummyClassifier):
    def __sklearn_clone__(self):
        if hasattr(self, 'search'):
            self.search.n_jobs = self.jobs
        return super().__sklearn_clone__()


class _BindCandidates:
    def __get__(self, search, owner=None):
        return self if search is None else partial(search.run_candidates)


class _FreshRun(GridSearchCV):
    _run_search = _BindCandidates()

    def run_candidates(self, evaluate_candidates):
        evaluate_candidates([{}])


class _SelectInvocation(inspect.Signature):
    @property
    def parameters(self):
        try:
            pickle.dumps(self.search)
        except (AttributeError, pickle.PicklingError) as error:
            raise AssertionError('signature inspection saw a temporary search hook') from error
        self.search._run_search = self.search.run_candidates
        return super().parameters


class _InspectThenReplace:
    def __init__(self, search):
        self.search = search

    @property
    def __signature__(self):
        signature = _SelectInvocation()
        signature.search = self.search
        return signature

    def __call__(self, evaluate_candidates):
        raise AssertionError('fit ignored the method selected during signature inspection')


class _HeldOutSum:
    def __init__(self, process):
        self.parent = os.getpid()
        self.process = process

    def __call__(self, estimator, X, y):
        assert (os.getpid() != self.parent) == self.process
        return X.sum()


def _check_callback_selected_folds(tmp_path, jobs_on_clone):
    X = np.arange(6)[:, None]
    path = tmp_path / 'indices.npy'
    np.save(path, [0, 2, 4])
    for fit, jobs in ((_stock, 1), (fit_shared_fold_search, 2)):
        if jobs_on_clone:
            callback, estimator, search_class = _SelectFolds(), _SelectJobsOnClone(), GridSearchCV
        else:
            callback, estimator, search_class = _SelectFoldsAndMethod(), DummyClassifier(), _ChangedMethod
        callback.path, callback.jobs = path, 1 if jobs_on_clone else jobs
        search = search_class(estimator, {}, cv=2, n_jobs=1, refit=False,
                              scoring=_HeldOutSum(jobs == 2), error_score='raise')
        if jobs_on_clone:
            estimator.search, estimator.jobs = search, jobs
        else:
            search.method_calls = 0
        search.set_callbacks(callback)
        with parallel_config(backend='loky', inner_max_num_threads=1):
            assert fit(search, X, X[:, 0] % 2) is search
        if not jobs_on_clone:
            assert search._run_search == search.selected_method and search.method_calls == 1
        scores = [search.cv_results_[f'split{i}_test_score'][0] for i in range(2)]
        assert scores == [9, 6], 'callback-selected folds lost parent-private indices'
        np.testing.assert_array_equal(np.load(path), [0, 2, 4])


@_callback_api
def test_callback_selects_folds_jobs_and_method(tmp_path):
    _check_callback_selected_folds(tmp_path, jobs_on_clone=False)


@_callback_api
def test_callback_folds_when_clone_selects_jobs(tmp_path):
    _check_callback_selected_folds(tmp_path, jobs_on_clone=True)


@_callback_api
@pytest.mark.parametrize('change_method', [False, True], ids=['fresh-binding', 'signature-switch'])
def test_callback_freshly_bound_run_method(tmp_path, change_method):
    X = np.arange(6)[:, None]
    callback = _SelectFolds()
    callback.jobs, callback.path = 2, tmp_path / 'indices.npy'
    np.save(callback.path, [0, 2, 4])
    search = _FreshRun(DummyClassifier(), {}, cv=2, n_jobs=1, refit=False,
                       scoring=_HeldOutSum(True), error_score='raise').set_callbacks(callback)
    assert search._run_search is not search._run_search
    if change_method:
        search._run_search = _InspectThenReplace(search)
    with parallel_config(backend='loky', inner_max_num_threads=1):
        assert fit_shared_fold_search(search, X, X[:, 0] % 2) is search
    scores = [search.cv_results_[f'split{i}_test_score'][0] for i in range(2)]
    assert scores == [9, 6], 'signature-selected method lost parent-private fold indices'
    if change_method:
        assert search._run_search == search.run_candidates
    else:
        assert '_run_search' not in vars(search)


class _SelectObjectFolds(_Callback):
    def setup(self, estimator, context):
        estimator.cv = self.outer
        estimator.n_jobs = 2


@_callback_api
def test_callback_checks_stored_outer_before_setup(monkeypatch):
    import dwd.model_selection as selection
    from test_reversed_mapped_search import _HeapMapping, _simulated_object_mapping

    X = np.arange(12)[:, None]
    pairs = np.empty(2, dtype=object)
    pairs[:] = list(KFold(2).split(X))
    cv = _simulated_object_mapping(pairs)
    monkeypatch.setattr(selection, 'mmap', (selection.mmap, _HeapMapping))
    callback = _Callback()
    callback.setup = lambda *args, **kwargs: pytest.fail('unchecked stored object reached callback setup')
    search = GridSearchCV(DummyClassifier(), {}, cv=cv, n_jobs=2).set_callbacks(callback)
    with parallel_config(backend='loky', inner_max_num_threads=1):
        with pytest.raises(ValueError, match='file-backed object arrays'):
            fit_shared_fold_search(search, X, X[:, 0] % 2)


@_callback_api
def test_callback_checked_before_sklearn_collects_outer_folds(monkeypatch):
    from sklearn.model_selection import _search
    import dwd.model_selection as selection
    from test_reversed_mapped_search import _HeapMapping, _simulated_object_mapping

    X = np.arange(12)[:, None]
    pairs = np.empty(2, dtype=object)
    pairs[:] = list(KFold(2).split(X))
    callback = _SelectObjectFolds()
    callback.outer = _simulated_object_mapping(pairs)
    monkeypatch.setattr(selection, 'mmap', (selection.mmap, _HeapMapping))
    original_check = _search.check_cv

    def check(cv, *args, **kwargs):
        assert cv is not callback.outer, 'callback-selected outer object reached sklearn check_cv'
        return original_check(cv, *args, **kwargs)

    monkeypatch.setattr(_search, 'check_cv', check)
    search = GridSearchCV(DummyClassifier(), {}, cv=2, n_jobs=1).set_callbacks(callback)
    with parallel_config(backend='loky', inner_max_num_threads=1):
        with pytest.raises(ValueError, match='file-backed object arrays'):
            fit_shared_fold_search(search, X, X[:, 0] % 2)
    assert '_run_search' not in vars(search) and '_init_callback_context' not in vars(search)


class _PickleAtTeardown(_Callback):
    def __init__(self, label):
        self.label = label

    def setup(self, estimator, context):
        self.native_teardowns = estimator._skl_callbacks_to_teardown

    def teardown(self, estimator, context):
        assert estimator._skl_callbacks_to_teardown is self.native_teardowns
        restored = pickle.loads(pickle.dumps(estimator))
        assert '_run_search' not in vars(restored) and '_init_callback_context' not in vars(restored)
        estimator.ended.append(self.label)


class _RejectClone(DummyClassifier):
    def __sklearn_clone__(self):
        raise self.problem


@_callback_api
@pytest.mark.parametrize('failure', ['split-count', 'clone'])
def test_teardown_after_early_failure(failure):
    X = np.arange(12)[:, None]
    y = X[:, 0] % 2
    errors = []
    for fit in (_stock, fit_shared_fold_search):
        callbacks = [_PickleAtTeardown('first'), _PickleAtTeardown('second')]
        if failure == 'split-count':
            estimator, cv = DummyClassifier(), LeaveOneGroupOut()
        else:
            estimator, cv = _RejectClone(), list(KFold(2).split(X))
            estimator.problem = ValueError('original clone error')
        search = GridSearchCV(estimator, {}, cv=cv).set_callbacks(*callbacks)
        search.ended = []
        try:
            fit(search, X, y)
        except Exception as exc:
            assert type(exc) is ValueError, 'teardown pickle masked the original fit error'
            if failure == 'clone':
                assert exc is estimator.problem
            errors.append(str(exc))
        else:
            pytest.fail('the invalid fit completed')
        assert search.ended == ['first', 'second']
        assert search._skl_callbacks == callbacks and not hasattr(search, '_skl_callbacks_to_teardown')
        assert '_run_search' not in vars(search) and '_init_callback_context' not in vars(search)
        search.estimator, search.cv = DummyClassifier(), 2
        assert fit(search, X, y) is search
        assert search.ended == ['first', 'second', 'first', 'second']
    assert errors[0] == errors[1]
