"""Search-exit fold cleanup preserves diagnostics, values and public ownership.

Proposed replacement for the early-release-specific module. It uses the same
public semantic oracle and works with the A-off helper or the native-gather
candidate. No optional native capability is assumed. Run only one version of
this module, with the existing test_shared_fold_search helper on the test path.
"""
import gc
import pickle
import weakref

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.model_selection import GridSearchCV

from dwd import model_selection as selection
from dwd.kern_md import KernMD

import test_shared_fold_search as shared_contract


pytestmark = pytest.mark.usefixtures("numerical_search_parent")


def capture_stores(monkeypatch):
    """Observe exact instances without replacing their class or indexing."""
    stores = []
    original = selection._FoldDataset.__init__
    def initialized(store, features):
        original(store, features)
        stores.append(store)
    monkeypatch.setattr(selection._FoldDataset, '__init__', initialized)
    return stores


@pytest.mark.parametrize('kind', ['span', 'gather'])
def test_search_cleanup_invalidates_tokens_but_retained_views_survive(kind, monkeypatch):
    X, y, folds = shared_contract._data()
    if kind == 'span':
        all_rows = np.arange(len(y), dtype=np.intp)
        folds = [(np.concatenate((all_rows[:8*i], all_rows[8*(i+1):])),
                  all_rows[8*i:8*(i+1)]) for i in range(3)]
    search = shared_contract._search(folds=folds)
    original = selection._FoldDataset.__getitem__
    seen = {}
    def indexed(store, key):
        token = original(store, key)
        if not seen:
            seen.update(token=token, view=token._borrow(),
                        expected=X[key[0], :].copy(),
                        store_ref=weakref.ref(store), root_ref=weakref.ref(store._root),
                        token_ref=weakref.ref(token))
        return token
    monkeypatch.setattr(selection._FoldDataset, '__getitem__', indexed)
    assert selection.fit_shared_fold_search(search, X, y) is search
    token, view = seen.pop('token'), seen['view']
    store = seen['store_ref']()
    assert store is not None and store._tokens == {}
    assert type(token._frozen._bytes) is (memoryview if kind == 'span' else bytes)
    with pytest.raises(ValueError, match='Unregistered private fold token'):
        token._borrow()
    assert_array_equal(np.asarray(token), seen['expected'])
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view.flags.writeable = True
    del token, store
    gc.collect()
    assert seen['store_ref']() is None and seen['root_ref']() is None
    assert seen['token_ref']() is None
    X[:] = -99
    assert_array_equal(view, seen['expected'])
    assert not view.flags.writeable


def test_failure_before_first_fold_has_zero_diagnostics_and_clears_store(monkeypatch):
    X, y, folds = shared_contract._data()
    search = shared_contract._search(folds=folds)
    constructor = search.estimator
    stores = capture_stores(monkeypatch)
    failure = RuntimeError('before first fold')
    def fail(self, features, labels, **kwargs):
        assert type(features) is selection._FoldDataset
        raise failure
    monkeypatch.setattr(GridSearchCV, 'fit', fail)
    with pytest.raises(RuntimeError) as caught:
        selection.fit_shared_fold_search(search, X, y)
    assert caught.value is failure and search.estimator is constructor
    assert len(stores) == 1 and stores[0]._tokens == {}
    assert not hasattr(search, 'scorer_') and not hasattr(search, 'best_estimator_')
    info = search.shared_fold_info
    assert info['route'] == 'shared' and info['reason'] is None and not info['completed']
    for field in ('index_calls', 'cache_hits', 'cached_subsets', 'borrowed_fits', 'cached_bytes'):
        assert info[field] == 0
    assert info['root_bytes'] == X.nbytes


def test_reused_search_replaces_detached_public_diagnostics(monkeypatch):
    X, y, folds = shared_contract._data()
    search = shared_contract._search(folds=folds)
    stores = capture_stores(monkeypatch)
    selection.fit_shared_fold_search(search, X, y)
    first = search.shared_fold_info
    snapshot = dict(first)
    first['cached_bytes'] = -1
    selection.fit_shared_fold_search(search, X, y)
    assert search.shared_fold_info is not first and search.shared_fold_info == snapshot
    assert first['cached_bytes'] == -1
    assert len(stores) == 2 and stores[0] is not stores[1]
    assert all(store._tokens == {} for store in stores)
    assert type(search.estimator) is type(search.best_estimator_) is KernMD


def test_diagnostic_snapshot_error_preserves_exception_and_still_cleans(monkeypatch):
    X, y, folds = shared_contract._data()
    search = shared_contract._search(folds=folds)
    constructor = search.estimator
    stores = capture_stores(monkeypatch)
    failure = MemoryError('diagnostic snapshot')
    def fail(store):
        raise failure
    monkeypatch.setattr(selection._FoldDataset, 'info', fail)
    with pytest.raises(MemoryError) as caught:
        selection.fit_shared_fold_search(search, X, y)
    assert caught.value is failure and search.estimator is constructor
    assert len(stores) == 1 and stores[0]._tokens == {}
    assert search.shared_fold_info == dict(route='shared', reason=None, completed=False)
    assert not hasattr(search, 'scorer_')
    # The canonical final fit succeeded before diagnostics failed. Existing
    # sklearn partial-state semantics do not promise a transactional rollback.
    assert type(search.best_estimator_) is KernMD
    assert search.best_estimator_._Xfit.flags.writeable
    assert not np.shares_memory(search.best_estimator_._Xfit, X)


@pytest.mark.parametrize('return_train_score', [False, True])
@pytest.mark.parametrize('candidates', [2, 8])
def test_final_fit_then_cleanup_preserves_table_diagnostics_and_public_model(
        candidates, return_train_score, monkeypatch):
    X, y, folds = shared_contract._data()
    shared = shared_contract._search(folds=folds, candidates=candidates).set_params(
        return_train_score=return_train_score)
    stock = shared_contract._search(folds=folds, candidates=candidates).set_params(
        return_train_score=return_train_score)
    stock.fit(X, y)
    final = []
    original_fit = KernMD.fit
    def final_fit(estimator, features, targets, **kwargs):
        assert type(features) is selection._FoldDataset
        before = features.info()
        assert before['index_calls'] == 6*candidates
        assert before['borrowed_fits'] == 3*candidates
        assert before['cached_subsets'] == 6 and before['cache_hits'] == 6*candidates-6
        # Final refit receives the complete immutable root. Keep one valid
        # borrowed view to verify search-exit cleanup does not invalidate data.
        token = next(iter(features._tokens.values()))
        view = token._borrow()
        final.append((features, before, view, view.copy()))
        return original_fit(estimator, features, targets, **kwargs)
    monkeypatch.setattr(KernMD, 'fit', final_fit)
    assert selection.fit_shared_fold_search(shared, X, y) is shared
    assert len(final) == 1
    store, before, view, expected = final[0]
    assert store._tokens == {}
    assert shared.shared_fold_info == dict(route='shared', reason=None, completed=True, **before)
    assert_array_equal(view, expected)
    assert not view.flags.writeable
    shared_contract._table_equal(shared, stock)
    shared_contract._predictions_equal(shared, stock, X)
    fitted = shared.best_estimator_._Xfit
    assert type(shared.best_estimator_) is KernMD
    assert fitted.flags.writeable and fitted.flags.owndata
    assert not np.shares_memory(fitted, X) and not np.shares_memory(fitted, view)
    payload = pickle.dumps(shared)
    for private_name in (b'_FoldDataset', b'_FoldToken', b'_SharedFoldKernMD'):
        assert private_name not in payload
    restored = pickle.loads(payload)
    shared_contract._table_equal(restored, stock)
    shared_contract._predictions_equal(restored, stock, X)
    assert restored.shared_fold_info == shared.shared_fold_info


@pytest.mark.parametrize('return_train_score', [False, True])
def test_no_refit_cleans_at_search_exit_without_canonical_final_fit(return_train_score, monkeypatch):
    X, y, folds = shared_contract._data()
    shared = shared_contract._search(folds=folds, refit=False).set_params(
        return_train_score=return_train_score)
    stock = shared_contract._search(folds=folds, refit=False).set_params(
        return_train_score=return_train_score)
    stock.fit(X, y)
    stores = capture_stores(monkeypatch)
    def unexpected(*args, **kwargs):
        raise AssertionError('No-refit search entered canonical final fit')
    monkeypatch.setattr(KernMD, 'fit', unexpected)
    assert selection.fit_shared_fold_search(shared, X, y) is shared
    shared_contract._table_equal(shared, stock)
    assert shared.shared_fold_info['completed']
    assert shared.shared_fold_info['cached_subsets'] == 6
    assert shared.shared_fold_info['borrowed_fits'] == 6
    assert len(stores) == 1 and stores[0]._tokens == {}
    assert not hasattr(shared, 'best_estimator_')


@pytest.mark.parametrize('prior_fit', [False, True])
@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
def test_final_failure_preserves_active_statistics_exception_and_cleanup(
        prior_fit, error_type, monkeypatch):
    X, y, folds = shared_contract._data()
    shared = shared_contract._search(folds=folds)
    if prior_fit:
        selection.fit_shared_fold_search(shared, X, y)
    constructor = shared.estimator
    had_scorer, scorer = hasattr(shared, 'scorer_'), getattr(shared, 'scorer_', None)
    failure, seen = error_type('deliberate final failure'), []
    def fail(estimator, features, targets, **kwargs):
        assert type(features) is selection._FoldDataset
        token = next(iter(features._tokens.values()))
        view = token._borrow()
        seen.append((features, features.info(), token, view, view.copy()))
        raise failure
    monkeypatch.setattr(KernMD, 'fit', fail)
    with pytest.raises(error_type) as caught:
        selection.fit_shared_fold_search(shared, X, y)
    assert caught.value is failure and len(seen) == 1
    assert shared.estimator is constructor and hasattr(shared, 'scorer_') is had_scorer
    if had_scorer:
        assert shared.scorer_ is scorer
    store, before, token, view, expected = seen[0]
    assert shared.shared_fold_info == dict(route='shared', reason=None, completed=False, **before)
    assert store._tokens == {} and before['cached_subsets'] == 6
    with pytest.raises(ValueError, match='Unregistered private fold token'):
        token._borrow()
    assert_array_equal(view, expected)
    assert not view.flags.writeable
    assert type(shared.estimator) is KernMD
    if hasattr(shared, 'best_estimator_'):
        assert type(shared.best_estimator_) is KernMD


@pytest.mark.parametrize('kind', ['custom-refit', 'parallel', 'custom-scorer'])
def test_unsupported_configuration_never_creates_private_store(kind, monkeypatch):
    X, y, folds = shared_contract._data()
    search = shared_contract._search(folds=folds)
    if kind == 'custom-refit':
        search.refit = lambda results: 0
    elif kind == 'parallel':
        search.n_jobs = 2
    else:
        search.scoring = lambda estimator, features, labels: 1.
    seen = []
    def forbidden(*args, **kwargs):
        raise AssertionError('Private store used by stock fallback')
    def stock_fit(self, features, labels, **kwargs):
        seen.append((self, features, labels, kwargs))
        return self
    monkeypatch.setattr(selection._FoldDataset, '__init__', forbidden)
    monkeypatch.setattr(GridSearchCV, 'fit', stock_fit)
    assert selection.fit_shared_fold_search(search, X, y) is search
    assert len(seen) == 1 and seen[0][0] is search and seen[0][1] is X and seen[0][2] is y
    assert seen[0][3] == {} and search.shared_fold_info['route'] == 'stock'
