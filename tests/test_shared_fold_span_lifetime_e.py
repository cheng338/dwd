"""E-only immutable single-span ownership and lifetime regressions.

Run with the isolated E module and its general semantic selection. These
checks do not claim lower process peak memory: cached_bytes is logical payload.
No candidate loader or workspace path is embedded in the test module.
"""
import gc
import weakref

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.model_selection import GridSearchCV

from dwd import model_selection as selection
from dwd.kern_md import KernMD


def dataset():
    X = np.arange(16*4, dtype=np.float64).reshape(16, 4)/8.
    y = np.array([-1, 1]*8)
    folds = [(np.arange(8), np.arange(8, 16)),
             (np.arange(8, 16), np.arange(8))]
    return X, y, folds


def search_for(folds, scoring='accuracy'):
    return GridSearchCV(KernMD(kernel='linear'), {'kernel_kws': [None, {}]},
                        cv=folds, n_jobs=1, scoring=scoring,
                        error_score='raise', return_train_score=True)


@pytest.mark.parametrize('rows,shared', [([1, 2, 3], True), ([-4, -3, -2], False),
                                        ([0], True), ([0, 1, 2, 3, 4], True)])
def test_e_span_or_negative_fallback_is_readonly_and_not_caller(rows, shared):
    owner = np.arange(20, dtype=np.float64).reshape(5, 4)
    X = owner.view()
    X.flags.writeable = False
    expected = owner[np.asarray(rows)].copy()
    store = selection._FoldDataset(X)
    token = store[(np.asarray(rows, dtype=np.intp), Ellipsis)]
    backing = token._frozen._bytes
    assert type(store._root._bytes) is bytes
    if shared:
        assert type(backing) is memoryview and backing.readonly
        assert backing.ndim == 1 and backing.format == 'B'
        assert backing.obj is store._root._bytes
    else:
        # E deliberately declines the negative-index optimization, retaining
        # the original NumPy normalization and independent byte snapshot.
        assert type(backing) is bytes
    owner[:] = -99
    owner.shape = (owner.size,)
    assert_array_equal(np.asarray(token), expected)
    assert_array_equal(token._borrow(), expected)
    assert not np.shares_memory(np.asarray(token), owner)
    with pytest.raises(TypeError):
        backing[0] = 0
    for value in (np.asarray(token), token._borrow()):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flags.writeable = True


def test_single_span_views_have_independent_shape_metadata_after_registry_clear():
    X, _, _ = dataset()
    store = selection._FoldDataset(X)
    token = store[(np.arange(2, 6), Ellipsis)]
    first, second = np.asarray(token), token._borrow()
    expected = X[2:6].copy()
    first.shape = (first.size,)
    assert second.shape == expected.shape
    assert np.asarray(token).shape == expected.shape
    store._tokens.clear()
    with pytest.raises(ValueError, match='Unregistered'):
        token._borrow()
    # Already borrowed immutable ndarrays own a valid lifetime chain.
    assert_array_equal(first.reshape(expected.shape), expected)
    assert_array_equal(second, expected)
    assert not first.flags.writeable and not second.flags.writeable


def test_borrowed_span_remains_valid_after_store_and_token_are_collected():
    X, _, _ = dataset()
    store = selection._FoldDataset(X)
    token = store[(np.arange(3, 7), Ellipsis)]
    expected = X[3:7].copy()
    borrowed = token._borrow()
    store_ref, root_ref, token_ref = weakref.ref(store), weakref.ref(store._root), weakref.ref(token)
    store._tokens.clear()
    del token, store
    gc.collect()
    assert store_ref() is None and root_ref() is None and token_ref() is None
    X[:] = -123
    assert_array_equal(borrowed, expected)
    assert not borrowed.flags.writeable
    with pytest.raises(ValueError):
        borrowed.flags.writeable = True


@pytest.mark.parametrize('rows', [[], [1, 1], [3, 2, 1], [0, 2, 4], [4, 0, 1]])
def test_empty_duplicate_reordered_and_multiple_spans_keep_owned_bytes(rows):
    X = np.arange(20, dtype=np.float64).reshape(5, 4)
    store = selection._FoldDataset(X)
    indices = np.asarray(rows, dtype=np.intp)
    token = store[(indices, Ellipsis)]
    assert type(token._frozen._bytes) is bytes
    assert_array_equal(np.asarray(token), X[indices])
    assert token._frozen._bytes == X[indices].tobytes(order='C')
    assert not np.asarray(token).flags.writeable


@pytest.mark.parametrize('shape,rows,shared', [((0, 4), [], False), ((5, 0), [1, 2], True),
                                             ((5, 0), [2, 2], False), ((0, 0), [], False)])
def test_zero_size_span_keeps_shape_and_logical_byte_counts(shape, rows, shared):
    X = np.empty(shape, dtype=np.float64)
    store = selection._FoldDataset(X)
    token = store[(np.asarray(rows, dtype=np.intp), Ellipsis)]
    assert (type(token._frozen._bytes) is memoryview) is shared
    assert np.asarray(token).shape == (len(rows), shape[1])
    assert store.info()['cached_bytes'] == 0
    assert not np.asarray(token).flags.writeable


@pytest.mark.parametrize('kind', ['bytearray', 'memoryview'])
def test_mutable_or_nonbytes_private_backing_declines_span_alias(kind):
    values = np.arange(20, dtype=np.float64).reshape(5, 4)
    mutable = bytearray(values.tobytes())
    store = selection._FoldDataset(values)
    store._root._bytes = mutable if kind == 'bytearray' else memoryview(mutable)
    rows = np.array([1, 2, 3], dtype=np.intp)
    assert selection._FrozenMatrix._try_contiguous_rows(store._root, rows) is None
    token = store[(rows, Ellipsis)]
    assert type(token._frozen._bytes) is bytes
    mutable[:] = bytes(len(mutable))
    assert_array_equal(np.asarray(token), values[1:4])
    assert not np.asarray(token).flags.writeable


def test_cached_bytes_counts_logical_subset_payload_not_unique_root_allocation():
    X = np.arange(20, dtype=np.float64).reshape(5, 4)
    store = selection._FoldDataset(X)
    first = store[(np.array([0, 1, 2], dtype=np.intp), Ellipsis)]
    second = store[(np.array([2, 3, 4], dtype=np.intp), Ellipsis)]
    assert first._frozen._bytes.obj is second._frozen._bytes.obj is store._root._bytes
    assert store[(np.array([0, 1, 2], dtype=np.intp), Ellipsis)] is first
    info = store.info()
    assert info['cached_subsets'] == 2 and info['cache_hits'] == 1
    assert info['root_bytes'] == X.nbytes
    assert info['cached_bytes'] == (3+3)*X.shape[1]*X.dtype.itemsize
    assert info['cached_bytes'] > info['root_bytes']


@pytest.mark.parametrize('scoring', [None, 'accuracy'])
def test_returned_public_model_owns_copy_and_does_not_keep_store_or_root(monkeypatch, scoring):
    X, y, folds = dataset()
    original_store = selection._FoldDataset
    refs = []
    class ObservedStore(original_store):
        def __init__(self, data):
            super().__init__(data)
            refs.append((weakref.ref(self), weakref.ref(self._root)))
    monkeypatch.setattr(selection, '_FoldDataset', ObservedStore)
    search = search_for(folds, scoring)
    constructor = search.estimator
    selection.fit_shared_fold_search(search, X, y)
    gc.collect()
    assert len(refs) == 1 and all(ref() is None for ref in refs[0])
    assert search.estimator is constructor
    assert type(search.best_estimator_) is KernMD
    assert search.best_estimator_._Xfit.flags.owndata
    assert search.best_estimator_._Xfit.flags.writeable
    assert not np.shares_memory(search.best_estimator_._Xfit, X)
    if scoring is None:
        assert search.scorer_._estimator is constructor
    expected = search.decision_function(X).copy()
    query = X.copy()
    X[:] = -73
    assert_array_equal(search.decision_function(query), expected)


@pytest.mark.parametrize('failure_type', [RuntimeError, KeyboardInterrupt])
def test_failed_fold_traceback_keeps_valid_readonly_span_and_original_error(monkeypatch, failure_type):
    X, y, folds = dataset()
    expected = X[:8].copy()
    search = search_for(folds)
    constructor = search.estimator
    failure = failure_type('single-span solve failure')
    def fail_solver(*args, **kwargs):
        raise failure
    with monkeypatch.context() as context:
        context.setattr(selection, 'kern_md', fail_solver)
        with pytest.raises(failure_type) as caught:
            selection.fit_shared_fold_search(search, X, y)
    assert caught.value is failure
    frames = []
    tb = failure.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    fold_frames = [frame for frame in frames if 'basis' in frame.f_locals
                   and type(frame.f_locals.get('X')) is selection._FoldToken]
    assert len(fold_frames) == 1
    state = fold_frames[0].f_locals
    token, borrowed = state['X'], state['basis']
    assert type(token._frozen._bytes) is memoryview
    assert token._store._tokens == {}
    assert '_Xfit' not in vars(state['self'])
    assert search.estimator is constructor and search.shared_fold_info['completed'] is False
    X[:] = -34
    gc.collect()
    assert_array_equal(borrowed, expected)
    assert not borrowed.flags.writeable
    with pytest.raises(ValueError, match='Unregistered'):
        token._borrow()
    # No promise that user-retained tracebacks free the private root. They may
    # retain it; the contract is valid immutable data and an empty token registry.
    selection.fit_shared_fold_search(search, expected.repeat(2, axis=0), np.tile(y[:8], 2))
    assert search.shared_fold_info['completed'] is True
