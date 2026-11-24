"""Check the storage behind public pandas selectors used for row indexing."""
import numpy as np
import pytest
from joblib import parallel_config
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV

from dwd.model_selection import fit_shared_fold_search
from test_pandas_fold_selector_edges import _results, _signature

pd = pytest.importorskip('pandas')


@pytest.mark.parametrize('field', ['index-values', 'series-values', 'series-labels'])
def test_object_selector_buffer_stops_before_worker_fit(tmp_path, field):
    # The empty mapping carries the dtype but contains no Python object pointers.
    path = tmp_path / 'empty-object.bin'
    path.write_bytes(bytes(8))
    mapped = np.memmap(path, dtype=object, mode='r', shape=(0,))
    array = np.asarray(mapped)
    if field == 'index-values':
        selector = pd.Index(array, dtype=object, copy=False)
        view = selector.to_numpy(copy=False)
    elif field == 'series-values':
        selector = pd.Series(array, dtype=object, copy=False)
        view = selector.to_numpy(copy=False)
    else:
        axis = pd.Index(array, dtype=object, copy=False)
        selector = pd.Series(np.empty(0, dtype=np.int64), index=axis)
        view = selector.index.to_numpy(copy=False)
    while isinstance(view, np.ndarray) and view is not mapped:
        view = view.base
    assert view is mapped
    search = GridSearchCV(DummyClassifier(), {}, cv=[(selector, [0])],
                          n_jobs=2, error_score='raise', refit=False)
    with parallel_config(backend='loky', inner_max_num_threads=1):
        with pytest.raises(ValueError, match='file-backed object arrays'):
            fit_shared_fold_search(search, np.arange(4)[:, None], np.array([0, 1, 0, 1]))
    assert not hasattr(search, 'cv_results_')


@pytest.mark.parametrize('kind', ['Series', 'Index'])
def test_nullable_integer_selector_keeps_private_row_values(tmp_path, kind):
    path = tmp_path / 'nullable-rows.npy'
    disk = np.arange(8, dtype=np.int64).reshape(2, 4)
    np.save(path, disk)
    data = np.load(path, mmap_mode='c')
    rows = [(2, 0, 3, 1), (6, 4, 7, 5)]
    data[:] = rows
    selectors = []
    for source in data:
        values = pd.arrays.IntegerArray(source, np.zeros(4, dtype=bool), copy=False)
        selector = getattr(pd, kind)(values, copy=False)
        assert np.shares_memory(selector.array._data, source)
        selectors.append(selector)
    first, second = selectors
    folds = [(first, second), (second, first)]
    expected = [_signature(*rows), _signature(*reversed(rows))]
    assert _results(folds, process=False) == expected
    assert _results(folds, process=True) == expected
    np.testing.assert_array_equal(np.load(path), disk)


@pytest.mark.parametrize('storage', ['nullable-mask', 'categorical-codes'])
def test_worker_sees_private_series_labels(tmp_path, storage):
    path = tmp_path / 'index-storage.npy'
    disk = np.zeros(4, dtype=bool if storage == 'nullable-mask' else np.int8)
    np.save(path, disk)
    buf = np.load(path, mmap_mode='c')
    buf[:] = [0, 1, 0, 1]
    if storage == 'nullable-mask':
        data = pd.arrays.IntegerArray(np.arange(4, dtype=np.int64), buf, copy=False)
        labels = pd.Index(data, copy=False)
        assert np.shares_memory(labels.array._mask, buf)
        expected_labels = ['0', 'missing', '2', 'missing']
    else:
        labels = pd.CategoricalIndex(pd.Categorical.from_codes(buf, categories=['day', 'night']))
        assert np.shares_memory(labels.codes, buf)
        expected_labels = ['day', 'night', 'day', 'night']
    first, second = [pd.Series(row, index=labels) for row in np.arange(8).reshape(2, 4)]
    folds = [(first, second), (second, first)]

    def make_scorer(search):
        def score(model, X, y):
            for selector in search.cv[0]:
                assert list(selector.index.astype('string').fillna('missing')) == expected_labels
            return model.score(X, y)
        return score

    expected = _results(folds, process=False, scorer_factory=make_scorer)
    assert _results(folds, process=True, scorer_factory=make_scorer) == expected
    np.testing.assert_array_equal(np.load(path), disk)
