"""Explicit CV skips unused eigen-preparation without changing solve behavior."""
from fractions import Fraction

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import ParameterGrid, StratifiedKFold

import dwd.gen_dwd as gdwd
import dwd._explicit_factor as factors
from dwd.cv import run_cv


def _data(layout):
    rng = np.random.RandomState(3737)
    X = rng.normal(size=(24, 4)) + np.array([1.0, -0.5, 0.25, 2.0])
    X[rng.uniform(size=X.shape) < 0.35] = 0.0
    y = np.tile([-1, 1], 12)
    return (sparse.csr_matrix(X) if layout == 'csr' else X), y


def _changed(X):
    changed = X.copy()
    if sparse.issparse(changed):
        changed.data[0] += 0.125
    else:
        changed[0, 0] += 0.125
    return changed


def _buffers(X):
    return (X.data, X.indices, X.indptr) if sparse.issparse(X) else (X,)


def _snapshot(X):
    return [(a.copy(), a.flags.writeable) for a in _buffers(X)]


def _unchanged(X, before):
    for actual, (expected, writable) in zip(_buffers(X), before):
        assert_array_equal(actual, expected)
        assert actual.flags.writeable == writable


def _model_state(model, X):
    return {
        'classes': model.classes_.copy(), 'coef': model.coef_.copy(),
        'intercept': model.intercept_.copy(),
        'objective_history': model.objective_history_.copy(),
        'C': model.C_, 'n_iter': model.n_iter_,
        'final_objective': model.final_objective_,
        'stationarity_residual': model.stationarity_residual_,
        'converged': model.converged_, 'termination_reason': model.termination_reason_,
        'prediction': model.predict(X), 'decision': model.decision_function(X),
    }


def _assert_state_matches(actual, expected, *, implicit):
    assert actual.keys() == expected.keys()
    exact = {'classes', 'n_iter', 'converged', 'termination_reason', 'prediction'}
    for key in actual:
        if not implicit or key in exact:
            assert_array_equal(actual[key], expected[key], err_msg=key)
        else:
            # Independent recomputed/cached eigensystems can differ in rounding.
            assert_allclose(actual[key], expected[key], rtol=2e-11, atol=1e-12,
                            err_msg=key)


def _instrument(monkeypatch):
    counts = {'eigendecompositions': 0, 'cv_init_calls': 0, 'eig_shapes': []}
    original_eig = gdwd.get_P0_eig
    original_init = gdwd.GenDWD.cv_init

    def observed_eig(X):
        counts['eigendecompositions'] += 1
        counts['eig_shapes'].append(list(X.shape))
        return original_eig(X)

    def observed_init(self, X):
        counts['cv_init_calls'] += 1
        return original_init(self, X)

    monkeypatch.setattr(gdwd, 'get_P0_eig', observed_eig)
    monkeypatch.setattr(gdwd.GenDWD, 'cv_init', observed_init)
    return counts


def _estimator(implicit=False, **kw):
    options = dict(lambd=0.1, q=1.0, implicit_P=implicit, max_iter=4,
                   stopping='fixed', initialization='zero', random_state=19)
    options.update(kw)
    return gdwd.GenDWD(**options)


def _splitter():
    return StratifiedKFold(3, shuffle=True, random_state=37)


def _assert_cv_scores_match_direct_fits(X, y, params, aggregate, folds,
                                      best_params, best_score, *, implicit=False):
    """Check prepared CV against direct fits, without preparation/cache reuse."""
    settings = list(ParameterGrid(params))
    assert aggregate['params'] == settings
    splits = list(_splitter().split(X, y))
    assert len(folds) == len(splits)
    fractions = {'train': [[] for _ in settings], 'test': [[] for _ in settings]}
    for fold, (train, test) in zip(folds, splits):
        assert fold['params'] == settings
        for index, setting in enumerate(settings):
            control = _estimator(implicit, **setting).fit(X[train], y[train])
            for prefix, indices in (('train', train), ('test', test)):
                correct = int(np.count_nonzero(control.predict(X[indices]) == y[indices]))
                fraction = Fraction(correct, len(indices))
                assert fold[prefix + '_correct'][index] == correct
                assert fold[prefix + '_count'][index] == len(indices)
                assert fold[prefix + '_score'][index] == float(fraction)
                fractions[prefix][index].append(fraction)
    means = {}
    for prefix in ('train', 'test'):
        means[prefix] = [sum(values, Fraction()) / len(values)
                         for values in fractions[prefix]]
        assert aggregate['mean_' + prefix + '_score'] == list(map(float, means[prefix]))
        assert aggregate['mean_' + prefix + '_score_fraction'] == [
            [value.numerator, value.denominator] for value in means[prefix]]
        assert_array_equal(aggregate['std_' + prefix + '_score'],
                           [np.std(list(map(float, values)))
                            for values in fractions[prefix]])
    best_index = max(range(len(settings)), key=means['test'].__getitem__)
    assert best_params == settings[best_index]
    assert best_score == float(means['test'][best_index])


@pytest.mark.parametrize('layout', ['dense', 'csr'])
@pytest.mark.parametrize('initialization', ['zero', 'random'])
def test_explicit_cv_init_keeps_direct_fit_state(layout, initialization, monkeypatch):
    X, y = _data(layout)
    before = _snapshot(X)
    model = _estimator(initialization=initialization)
    with monkeypatch.context() as mp:
        counts = _instrument(mp)
        assert model.cv_init(X) is model
        assert counts['eigendecompositions'] == 0
        assert not hasattr(model, '_P0_eig')
        assert not model._cv_cache_matches(X)
        model.fit(X, y)
        assert counts == {'eigendecompositions': 0, 'cv_init_calls': 1, 'eig_shapes': []}
    plain = _estimator(initialization=initialization).fit(X, y)
    _assert_state_matches(_model_state(model, X), _model_state(plain, X), implicit=False)
    _unchanged(X, before)


@pytest.mark.parametrize('layout', ['dense', 'csr'])
def test_implicit_cache_reuse_and_changed_data_remain_checked(layout, monkeypatch):
    X, y = _data(layout)
    changed = _changed(X)
    model = _estimator(True)
    with monkeypatch.context() as mp:
        counts = _instrument(mp)
        model.cv_init(X)
        cache = tuple(a.copy() for a in model._P0_eig)
        model.fit(X, y)
        assert counts['eigendecompositions'] == 1
        for actual, expected in zip(model._P0_eig, cache):
            assert_array_equal(actual, expected)
        model.fit(changed, y)
        assert counts['eigendecompositions'] == 2
        assert model._cv_cache_matches(X)
        assert not model._cv_cache_matches(changed)
        model.cv_init(changed).fit(changed, y)
        assert counts == {'eigendecompositions': 3, 'cv_init_calls': 2,
                          'eig_shapes': [[24, 4]] * 3}
        assert model._cv_cache_matches(changed)
    control = _estimator(True).fit(changed, y)
    _assert_state_matches(_model_state(model, changed), _model_state(control, changed),
                          implicit=True)


@pytest.mark.parametrize('layout', ['dense', 'csr'])
@pytest.mark.parametrize('implicit', [False, True])
def test_public_cv_preserves_preparation_and_scientific_outputs(layout, implicit, monkeypatch):
    X, y = _data(layout)
    before = _snapshot(X)
    params = {'lambd': [0.05, 0.2], 'q': [1.0, 2.0]}
    model = gdwd.GenDWDCV(lambd_vals=params['lambd'], q_vals=params['q'],
                         cv=_splitter(), implicit_P=implicit, max_iter=4,
                         stopping='fixed', initialization='zero', random_state=19)
    with monkeypatch.context() as mp:
        counts = _instrument(mp)
        model.fit(X, y)
    assert counts == ({'eigendecompositions': 4, 'cv_init_calls': 3,
                       'eig_shapes': [[16, 4]] * 3 + [[24, 4]]} if implicit else
                      {'eigendecompositions': 0, 'cv_init_calls': 12, 'eig_shapes': []})
    direct = _estimator(implicit, **model.best_params_).fit(X, y)
    _assert_state_matches(_model_state(model.best_estimator_, X),
                          _model_state(direct, X), implicit=implicit)
    _assert_cv_scores_match_direct_fits(X, y, params, model.agg_cv_results_,
                                      model.all_cv_results_, model.best_params_,
                                      model.best_score_, implicit=implicit)
    _unchanged(X, before)


@pytest.mark.parametrize('layout', ['dense', 'csr'])
@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_mixed_solver_grid_retains_safe_cache_transitions(layout, order, monkeypatch):
    X, y = _data(layout)
    params = {'implicit_P': list(order), 'lambd': [0.05, 0.2], 'q': [1.0]}
    with monkeypatch.context() as mp:
        counts = _instrument(mp)
        result = run_cv(_estimator(), X, y, params, cv=_splitter())
    best_params, best_score, best, aggregate, folds = result
    final_implicit = bool(best_params['implicit_P'])
    assert counts == {'eigendecompositions': 3 + int(final_implicit),
                      'cv_init_calls': 9 if order[0] is False else 3,
                      'eig_shapes': [[16, 4]] * 3 + ([[24, 4]] if final_implicit else [])}
    direct = _estimator(**best_params).fit(X, y)
    _assert_state_matches(_model_state(best, X), _model_state(direct, X),
                          implicit=final_implicit)
    _assert_cv_scores_match_direct_fits(X, y, params, aggregate, folds, best_params, best_score)


@pytest.mark.parametrize('layout', ['dense', 'csr'])
def test_false_true_false_true_with_changed_data(layout, monkeypatch):
    X, y = _data(layout)
    changed = _changed(X)
    model = _estimator(False)
    states, phases = [], []
    with monkeypatch.context() as mp:
        counts = _instrument(mp)
        for implicit, data, prepare in ((False, X, True), (True, X, False),
                                        (False, changed, True), (True, changed, False)):
            model.set_params(implicit_P=implicit)
            if prepare:
                model.cv_init(data)
            model.fit(data, y)
            phases.append(counts['eigendecompositions'])
            states.append(_model_state(model, data))
            assert not hasattr(model, '_P0_eig')
        assert counts == {'eigendecompositions': 2, 'cv_init_calls': 2,
                          'eig_shapes': [[24, 4]] * 2}
    assert phases == [0, 1, 1, 2]
    for state, data, implicit in zip(states, (X, X, changed, changed),
                                     (False, True, False, True)):
        control = _estimator(implicit).fit(data, y)
        _assert_state_matches(state, _model_state(control, data), implicit=implicit)


@pytest.mark.parametrize('layout', ['dense', 'csr'])
def test_preexisting_implicit_cache_remains_data_bound_after_explicit_preparation(
        layout, monkeypatch):
    X, y = _data(layout)
    changed = _changed(X)
    model = _estimator(True)
    with monkeypatch.context() as mp:
        counts = _instrument(mp)
        model.cv_init(X).fit(X, y)
        saved_cache = tuple(a.copy() for a in model._P0_eig)
        assert counts['eigendecompositions'] == 1
        model.set_params(implicit_P=False).cv_init(changed).fit(changed, y)
        assert counts['eigendecompositions'] == 1
        explicit_state = _model_state(model, changed)
        assert model._cv_cache_matches(X)
        assert not model._cv_cache_matches(changed)
        for actual, expected in zip(model._P0_eig, saved_cache):
            assert_array_equal(actual, expected)
        model.set_params(implicit_P=True).fit(changed, y)
        assert counts == {'eigendecompositions': 2, 'cv_init_calls': 2,
                          'eig_shapes': [[24, 4]] * 2}
        implicit_state = _model_state(model, changed)
    for state, implicit in ((explicit_state, False), (implicit_state, True)):
        control = _estimator(implicit).fit(changed, y)
        _assert_state_matches(state, _model_state(control, changed), implicit=implicit)


@pytest.mark.parametrize('bad', ['nan', 'inf', 'complex', 'one_dimensional', 'empty_features'])
def test_explicit_cv_init_keeps_input_validation_before_preparation(bad, monkeypatch):
    X, _ = _data('dense')
    if bad in ('nan', 'inf'):
        X[0, 0] = np.nan if bad == 'nan' else np.inf
    elif bad == 'complex':
        X = X.astype(complex) + 1j
    elif bad == 'one_dimensional':
        X = X[:, 0]
    else:
        X = X[:, :0]
    with monkeypatch.context() as mp:
        counts = _instrument(mp)
        with pytest.raises(ValueError):
            _estimator(False).cv_init(X)
        assert counts == {'eigendecompositions': 0, 'cv_init_calls': 1, 'eig_shapes': []}


@pytest.mark.parametrize('layout', ['dense', 'csr'])
@pytest.mark.parametrize('stage', ['factor', 'solve'])
def test_explicit_numerical_failure_is_not_hidden_after_cv_init(layout, stage, monkeypatch):
    X, y = _data(layout)
    model = _estimator(False).fit(X, y)
    with monkeypatch.context() as mp:
        counts = _instrument(mp)
        model.cv_init(X)
        failure = FloatingPointError('deliberate explicit ' + stage + ' refusal')

        def refuse(*args, **kwargs):
            raise failure

        if stage == 'factor':
            mp.setattr(factors, 'ExplicitFactor', refuse)
        else:
            mp.setattr(factors.ExplicitFactor, 'solve', refuse)
        with pytest.raises(FloatingPointError) as caught:
            model.fit(X, y)
        assert caught.value is failure
        assert counts == {'eigendecompositions': 0, 'cv_init_calls': 1, 'eig_shapes': []}
        assert not [name for name in vars(model)
                    if name.endswith('_') and not name.startswith('_')]
        with pytest.raises(NotFittedError):
            model.predict(X)


@pytest.mark.parametrize('layout', ['dense', 'csr'])
def test_explicit_zero_penalty_rank_refusal_survives_cv_preparation(layout, monkeypatch):
    X = np.ones((12, 3))
    X = sparse.csr_matrix(X) if layout == 'csr' else X
    y = np.tile([-1, 1], 6)
    model = _estimator(False, lambd=0.0)
    with monkeypatch.context() as mp:
        counts = _instrument(mp)
        model.cv_init(X)
        with pytest.raises(ValueError, match='rank deficient|singular'):
            model.fit(X, y)
        assert counts == {'eigendecompositions': 0, 'cv_init_calls': 1, 'eig_shapes': []}


@pytest.mark.parametrize('implicit', [False, True])
def test_replaced_eigen_preparation_is_still_required_only_for_implicit(implicit, monkeypatch):
    X, y = _data('dense')
    calls = []
    failure = RuntimeError('deliberate eigen-preparation failure')

    def refuse(features):
        calls.append(list(features.shape))
        raise failure

    model = _estimator(implicit)
    with monkeypatch.context() as mp:
        mp.setattr(gdwd, 'get_P0_eig', refuse)
        if implicit:
            with pytest.raises(RuntimeError) as caught:
                model.cv_init(X)
            assert caught.value is failure
            assert calls == [[24, 4]]
        else:
            assert model.cv_init(X) is model
            model.fit(X, y)
            assert np.isfinite(model.coef_).all()
            assert calls == []
