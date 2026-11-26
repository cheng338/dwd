"""End-to-end string label checks for the standalone DWD checkpoint example."""
from collections import UserList
from contextlib import nullcontext
from decimal import Decimal
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import StratifiedKFold

from dwd.cv import run_cv
from dwd.gen_kern_dwd import KernGDWD

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from resumable_kernel_cv import CheckpointError, _owned_inputs, run_resumable_cv


X = np.array([[-2.], [2.], [-2.], [2.], [2.], [-2.]] * 2)
NUMERIC = np.array([0, 1] * 6)
PARAMS = {'lambd': [.1]}


def model():
    return KernGDWD(kernel='linear', random_state=23, stopping='fixed', max_iter=3)


def native_dtype(**options):
    if not hasattr(np.dtypes, 'StringDType'):
        pytest.skip('Requires NumPy variable-width strings')
    return np.dtypes.StringDType(**options)


def test_native_strings_resume_and_refit_with_full_class_names(tmp_path):
    first = 'long class name: \0α'
    second = first + '\0'
    dtype = native_dtype(coerce=False, na_object=first)
    directory = tmp_path / 'run'

    def search(labels, **kwargs):
        return run_resumable_cv(model(), X, labels, PARAMS, directory, cv=2, **kwargs)

    expected = run_cv(model(), X, NUMERIC, PARAMS, cv=2)
    first_run = search(np.array([first, second] * 6, dtype=dtype))
    repeated = search(np.array([first, second] * 6, dtype=dtype), resume=True)
    assert first_run['work_summary']['executed_evaluations'] == 2
    assert repeated['work_summary']['executed_evaluations'] == 0
    assert repeated['work_summary']['replayed_evaluations'] == 2
    assert first_run['best_clf'] is not repeated['best_clf']
    for result in (first_run, repeated):
        assert result['best_score'] == expected[1]
        assert result['best_params'] == expected[0]
        assert result['best_clf'].classes_.dtype == dtype
        assert result['best_clf'].classes_.tolist() == [first, second]
        pred = result['best_clf'].predict(X)
        assert pred.dtype == dtype
        assert pred.tolist() == [[first, second][i] for i in expected[2].predict(X)]
        for fold, oracle in zip(result['all_cv_results'], expected[4], strict=True):
            for key in ('train_score', 'test_score', 'train_correct', 'test_correct'):
                assert fold[key] == oracle[key]
    for changed in (np.array([first, second + '\0'] * 6, dtype=dtype),
                    np.array([second, first] * 6, dtype=dtype),
                    np.array([first, second] * 6, dtype=native_dtype())):
        with pytest.raises(CheckpointError, match='identity'):
            search(changed, resume=True)


@pytest.mark.parametrize('carrier', [list, UserList, lambda rows: np.asarray(rows, dtype=object)])
def test_string_sources_distinguish_trailing_nuls_in_checkpoint_data(carrier):
    labels = ['longname', 'q\0'] * 6
    _, y, codes, ident = _owned_inputs(X, carrier(labels))
    assert y.tolist() == labels
    assert codes.tolist() == [0, 1] * 6
    assert ident != _owned_inputs(X, carrier(['longname', 'q'] * 6))[3]


@pytest.mark.parametrize('carrier', [list, UserList, lambda rows: np.asarray(rows, dtype=object)])
def test_checkpoint_refit_uses_the_public_string_representation(carrier, tmp_path):
    class Upper(str):
        def __str__(self):
            return str.upper(self)

    y = carrier([Upper('cat'), Upper('dog')] * 6)
    oracle = run_cv(model(), X, y, PARAMS, cv=2)
    result = run_resumable_cv(model(), X, y, PARAMS, tmp_path / 'run', cv=2)
    expected = ['cat', 'dog'] if isinstance(y, np.ndarray) else ['CAT', 'DOG']
    assert oracle[2].classes_.tolist() == expected
    assert result['best_clf'].classes_.tolist() == expected
    assert result['best_clf'].predict(X).tolist() == oracle[2].predict(X).tolist()
    assert result['best_score'] == oracle[1]


def test_checkpoint_uses_string_values_and_requires_identifiable_comparisons():
    class Named(str):
        def __str__(self):
            return 'display'

    rows = [Named('q'), Named('r')] * 6
    assert _owned_inputs(X, np.array(rows, dtype=object))[1].tolist() == ['q', 'r'] * 6
    for labels in (rows, UserList(rows)):
        with pytest.raises(ValueError, match='exactly two classes'):
            model().fit(X, labels)
        with pytest.raises(ValueError, match='exactly two classes'):
            _owned_inputs(X, labels)

    class Reversed(str):
        def __lt__(self, other):
            return str.__gt__(self, other)

    ordered = [Reversed('q'), Reversed('r')] * 6
    for labels in (ordered, UserList(ordered)):
        assert model().fit(X, labels).classes_.tolist() == ['q', 'r']
        assert _owned_inputs(X, labels)[1].tolist() == ['q', 'r'] * 6
    with pytest.raises(CheckpointError, match='custom string comparisons'):
        _owned_inputs(X, np.array(ordered, dtype=object))


def test_checkpoint_distinguishes_class_boundaries_and_native_dtype_metadata():
    def identity(names, **options):
        y = np.array(names * 6, dtype=native_dtype(**options))
        return _owned_inputs(X, y)[3]

    names = ['x', 'yz']
    assert identity(names) != identity(['xy', 'z'])
    assert identity(names, na_object=object()) == identity(names, na_object=object())
    policies = [dict(coerce=False), {}, dict(na_object=None), dict(na_object='unused'),
                dict(na_object=7), dict(na_object=8), dict(na_object=float('nan')),
                dict(na_object=np.float32('nan')), dict(na_object=Decimal('NaN12')),
                dict(na_object=Decimal('NaN13')), dict(na_object=np.datetime64('NaT', 'ns')),
                dict(na_object=np.datetime64('NaT', 'ms'))]
    identities = [identity(names, **policy)['sha256'] for policy in policies]
    assert len(set(identities)) == len(policies)


def test_resume_distinguishes_supplementary_character_from_literal_surrogates(tmp_path):
    directory = tmp_path / 'search'
    def target(name):
        return np.array(['alpha', name + '\0'] * 6, dtype=object)
    run_resumable_cv(model(), X, target('\U0001f680'), PARAMS, directory, cv=2)
    with pytest.raises(CheckpointError, match='identity'):
        run_resumable_cv(model(), X, target('\ud83d\ude80'), PARAMS, directory, cv=2, resume=True)


def test_checkpoint_refuses_retained_opaque_dtype_metadata(tmp_path):
    dtype = np.dtype(object, metadata={'family': 'first'})
    y = np.array(['cat', 'dog\0'] * 6, dtype=dtype)
    with pytest.raises(CheckpointError, match='dtype metadata'):
        run_resumable_cv(model(), X, y, PARAMS, tmp_path/'run', cv=2)
    assert not (tmp_path/'run').exists()


def test_pandas_uint64_labels_bind_fits_and_checkpoint_identity(tmp_path):
    pd = pytest.importorskip('pandas')
    def rows(upper):
        return np.array([0, upper] * 6, dtype=np.uint64)
    upper = 2**53 + 1
    exact = rows(upper)
    sparse = pd.arrays.SparseArray(exact, fill_value=0)
    levels = pd.Index(pd.arrays.SparseArray(exact[:2], fill_value=0))
    categorical = pd.Categorical.from_codes([0, 1] * 6, categories=levels)
    assert int(categorical[1]) == upper
    for carrier in (sparse, pd.Index(sparse), pd.Series(sparse), pd.DataFrame({'y':sparse}),
                    pd.Series(pd.array(exact, dtype='UInt64')), categorical,
                    pd.Series(categorical), pd.CategoricalIndex(categorical), pd.DataFrame({'y':categorical})):
        with pytest.warns(DataConversionWarning) if isinstance(carrier, pd.DataFrame) else nullcontext():
            owned = _owned_inputs(X, carrier)[1]
        assert owned.tolist() == exact.tolist()
        assert owned.dtype == exact.dtype
    for boxed in (exact.astype(object), pd.Series(exact, dtype=object)):
        with pytest.raises(ValueError, match='Object labels must contain only strings'):
            _owned_inputs(X, boxed)
    directory = tmp_path/'run'
    result = run_resumable_cv(model(), X, pd.Series(sparse), PARAMS, directory, cv=2)
    control = model().fit(X, exact)
    assert result['best_clf'].classes_.tolist() == control.classes_.tolist() == [0, upper]
    assert result['best_clf'].predict(X).tolist() == control.predict(X).tolist()
    changed = pd.Series(pd.arrays.SparseArray(rows(upper - 1), fill_value=0))
    with pytest.raises(CheckpointError, match='identity'):
        run_resumable_cv(model(), X, changed, PARAMS, directory, cv=2, resume=True)


def test_ordinary_native_strings_custom_split_and_missing_rows(tmp_path):
    dtype = native_dtype(na_object=None)
    y = np.array(['cat', 'dog'] * 6, dtype=dtype)
    seen = []

    class Splitter(StratifiedKFold):
        def split(self, X, labels, groups=None):
            assert labels.dtype == dtype and labels.tolist() == y.tolist()
            seen.append(True)
            yield from super().split(X, NUMERIC, groups)

    result = run_resumable_cv(model(), X, y, PARAMS, tmp_path / 'run', cv=Splitter(2))
    assert seen == [True]
    assert result['best_clf'].classes_.tolist() == ['cat', 'dog']
    assert result['best_clf'].classes_.dtype == dtype
    assert result['best_score'] == run_cv(model(), X, NUMERIC, PARAMS, cv=2)[1]
    y[0] = None
    with pytest.raises(ValueError, match='missing'):
        run_resumable_cv(model(), X, y, PARAMS, tmp_path / 'bad', cv=2)
    assert not (tmp_path / 'bad').exists()
