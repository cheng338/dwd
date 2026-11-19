"""E-only route witnesses for a span check with the original NumPy fallback.

These are untimed behavior tests. No candidate loader, historical package copy,
or workspace path is required. The frozen general and D suites remain intact.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dwd import model_selection as selection


def original_outcome(values, rows):
    try:
        return values[rows, :], None
    except IndexError as error:
        return None, (type(error), str(error))


@pytest.mark.parametrize('rows', [[], [0, 2], [3, 1, 1, 0], [-4, -3, -2],
                                  [5, 6], [4, 5],
                                  [np.iinfo(np.intp).min, np.iinfo(np.intp).max]])
def test_endpoint_rejection_skips_full_sequence_proof_and_uses_original_numpy(monkeypatch, rows):
    values = np.arange(20, dtype=np.float64).reshape(5, 4)
    indices = np.asarray(rows, dtype=np.intp)
    expected, expected_error = original_outcome(values, indices)
    store = selection._FoldDataset(values)
    original_view = selection._FrozenMatrix.view
    root_reads = []
    def viewed(matrix):
        if matrix is store._root:
            root_reads.append(1)
        return original_view(matrix)
    def forbidden_proof(*args, **kwargs):
        raise AssertionError('Endpoint-rejected selector must not allocate/compare a full sequence')
    with monkeypatch.context() as patch:
        patch.setattr(selection._FrozenMatrix, 'view', viewed)
        patch.setattr(selection.np, 'arange', forbidden_proof)
        patch.setattr(selection.np, 'array_equal', forbidden_proof)
        if expected_error is None:
            token = store[(indices, Ellipsis)]
        else:
            with pytest.raises(expected_error[0]) as caught:
                store[(indices, Ellipsis)]
            assert str(caught.value) == expected_error[1]
    assert root_reads == [1]
    if expected_error is None:
        assert type(token._frozen._bytes) is bytes
        assert_array_equal(np.asarray(token), expected)
        assert token._frozen._bytes == expected.tobytes(order='C')
    else:
        assert store._tokens == {}
    assert store._index_calls == 1 and store._cache_hits == 0


@pytest.mark.parametrize('rows', [[0, 2, 1, 3], [0, 1, 1, 3], [1, 3, 2, 4]])
def test_matching_endpoints_do_not_hide_interior_order_or_duplicates(monkeypatch, rows):
    values = np.arange(20, dtype=np.float64).reshape(5, 4)
    indices = np.asarray(rows, dtype=np.intp)
    expected = values[indices, :]
    store = selection._FoldDataset(values)
    original_equal, original_view = np.array_equal, selection._FrozenMatrix.view
    proofs, root_reads = [], []
    def compared(left, right, *args, **kwargs):
        proofs.append((left.copy(), right.copy()))
        return original_equal(left, right, *args, **kwargs)
    def viewed(matrix):
        if matrix is store._root:
            root_reads.append(1)
        return original_view(matrix)
    with monkeypatch.context() as patch:
        patch.setattr(selection.np, 'array_equal', compared)
        patch.setattr(selection._FrozenMatrix, 'view', viewed)
        token = store[(indices, Ellipsis)]
    assert len(proofs) == 1 and root_reads == [1]
    assert_array_equal(proofs[0][0], indices)
    assert_array_equal(proofs[0][1], np.arange(rows[0], rows[-1]+1, dtype=np.intp))
    assert type(token._frozen._bytes) is bytes
    assert_array_equal(np.asarray(token), expected)
    assert token._frozen._bytes == expected.tobytes(order='C')


@pytest.mark.parametrize('rows', [[1, 2, 3], [4]])
def test_successful_proof_reuses_root_span_and_cache_hit_does_not_reprobe(monkeypatch, rows):
    values = np.arange(20, dtype=np.float64).reshape(5, 4)
    indices = np.asarray(rows, dtype=np.intp)
    expected = values[indices, :].copy()
    store = selection._FoldDataset(values)
    original_equal, original_view = np.array_equal, selection._FrozenMatrix.view
    proofs = []
    def compared(left, right, *args, **kwargs):
        proofs.append((left.copy(), right.copy()))
        return original_equal(left, right, *args, **kwargs)
    def viewed(matrix):
        if matrix is store._root:
            raise AssertionError('A proved span must not execute the NumPy gather')
        return original_view(matrix)
    with monkeypatch.context() as patch:
        patch.setattr(selection.np, 'array_equal', compared)
        patch.setattr(selection._FrozenMatrix, 'view', viewed)
        token = store[(indices, Ellipsis)]
        assert store[(indices.copy(), Ellipsis)] is token
    assert len(proofs) == 1
    assert type(token._frozen._bytes) is memoryview
    assert token._frozen._bytes.readonly and token._frozen._bytes.obj is store._root._bytes
    assert_array_equal(np.asarray(token), expected)
    assert store._cache_hits == 1


@pytest.mark.parametrize('error_type', [LookupError, MemoryError])
def test_original_fallback_exception_identity_leaves_no_registered_token(monkeypatch, error_type):
    values = np.arange(20, dtype=np.float64).reshape(5, 4)
    store = selection._FoldDataset(values)
    original_view = selection._FrozenMatrix.view
    failure = error_type('original NumPy fallback boundary')
    calls = []
    def viewed(matrix):
        if matrix is store._root:
            calls.append(1)
            raise failure
        return original_view(matrix)
    monkeypatch.setattr(selection._FrozenMatrix, 'view', viewed)
    with pytest.raises(error_type) as caught:
        store[(np.array([0, 2], dtype=np.intp), Ellipsis)]
    assert caught.value is failure and calls == [1]
    assert store._tokens == {} and store._index_calls == 1


@pytest.mark.parametrize('boundary', ['float32-root', 'int32-indices',
                                      'uint64-indices', 'matrix-indices'])
def test_private_unsupported_metadata_declines_only_the_optional_span(boundary):
    values = np.arange(20, dtype=np.float64).reshape(5, 4)
    indices = np.array([1, 2, 3], dtype=np.intp)
    if boundary == 'float32-root':
        values = values.astype(np.float32)
    elif boundary == 'int32-indices':
        assert np.dtype(np.intp).itemsize == 8  # Both approved host protocols are 64-bit.
        indices = indices.astype(np.int32)
    elif boundary == 'uint64-indices':
        indices = indices.astype(np.uint64)
    else:
        indices = indices.reshape(1, 3)
    root = selection._FrozenMatrix(values)
    assert selection._FrozenMatrix._try_contiguous_rows(root, indices) is None


@pytest.mark.parametrize('rows', [[0, 99, 2], [0, -1, 2]])
def test_matching_endpoints_keep_original_interior_index_semantics(monkeypatch, rows):
    values = np.arange(20, dtype=np.float64).reshape(5, 4)
    indices = np.asarray(rows, dtype=np.intp)
    expected, expected_error = original_outcome(values, indices)
    store = selection._FoldDataset(values)
    original_equal, original_view = np.array_equal, selection._FrozenMatrix.view
    proofs, root_reads = [], []
    def compared(left, right, *args, **kwargs):
        proofs.append((left.copy(), right.copy()))
        return original_equal(left, right, *args, **kwargs)
    def viewed(matrix):
        if matrix is store._root:
            root_reads.append(1)
        return original_view(matrix)
    with monkeypatch.context() as patch:
        patch.setattr(selection.np, 'array_equal', compared)
        patch.setattr(selection._FrozenMatrix, 'view', viewed)
        if expected_error is None:
            token = store[(indices, Ellipsis)]
        else:
            with pytest.raises(expected_error[0]) as caught:
                store[(indices, Ellipsis)]
            assert str(caught.value) == expected_error[1]
    assert len(proofs) == 1 and root_reads == [1]
    assert_array_equal(proofs[0][0], indices)
    assert_array_equal(proofs[0][1], [0, 1, 2])
    if expected_error is None:
        assert type(token._frozen._bytes) is bytes
        assert_array_equal(np.asarray(token), expected)
    else:
        assert store._tokens == {}
