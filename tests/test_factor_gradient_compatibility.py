"""Separate inverse/factor rounding from exact gradient-reuse/API contracts.

The original inverse-reference tests are preserved unchanged in a local,
hash-pinned historical fixture. It supplies identical inputs and hook
instrumentation, not acceptance rules.
Only listed floating model fields permit rtol=2e-12/atol=2e-13; metadata,
predictions, mutation behavior and hook counts remain exact. Gradient reuse on/off
must be bitwise identical while both runs use the same factor algorithm.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import csr_matrix, issparse

import dwd.gen_dwd as gdwd

RTOL, ATOL = 2e-12, 2e-13
NUMERICAL_FIELDS = {
    'coef_', 'intercept_', 'obj_vals_', 'C_', 'objective_history_',
    'final_objective_', 'gradient_inf_norm_', 'gradient_norm_', 'stationarity_residual_',
}
FIXTURE_SHA = '66b3acc85ba86caa0aa22ffde7fb3565c80799a625ec49bb4e0d030c817cd7e5'


@pytest.fixture(scope='module')
def prior():
    path = (Path(__file__).resolve().parent / '_legacy' /
            'linear_gradient_inverse_reference.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FIXTURE_SHA
    spec = importlib.util.spec_from_file_location('_f6_pinned_gradient_fixtures', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.gdwd is gdwd
    assert Path(module.__file__).resolve().parent == Path(__file__).resolve().parent / '_legacy'
    return module


def fixture_case(prior):
    case = prior.LinearGradientReuseTests()
    case.setUp()
    return case


def input_state(prior, X):
    flags = {name: vars(X).get(name, '<absent>')
             for name in ('_has_sorted_indices', '_has_canonical_format')} if issparse(X) else {}
    return prior._encoded(X), flags


def numerical_difference(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    assert expected.shape == actual.shape
    assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False)
    finite = np.isfinite(expected) & np.isfinite(actual)
    delta = np.abs(actual[finite] - expected[finite])
    denominator = ATOL + RTOL * np.abs(expected[finite])
    return {'max_absolute': float(delta.max(initial=0.)),
            'max_tolerance_fraction': float((delta / denominator).max(initial=0.))}


def compare_models(prior, expected, actual, X, *, bitwise=False):
    a = {k: v for k, v in vars(expected).items() if k.endswith('_')}
    b = {k: v for k, v in vars(actual).items() if k.endswith('_')}
    assert set(a) == set(b)
    metrics = {}
    for name in a:
        if name in NUMERICAL_FIELDS:
            metrics[name] = numerical_difference(a[name], b[name])
        else:
            assert prior._encoded(a[name]) == prior._encoded(b[name]), name
    metrics['scores'] = numerical_difference(expected.decision_function(X), actual.decision_function(X))
    assert_array_equal(expected.predict(X), actual.predict(X))
    if bitwise:
        assert prior._model_state(expected) == prior._model_state(actual)
        assert prior._encoded(expected.decision_function(X)) == prior._encoded(actual.decision_function(X))
    return metrics


def independent_final_checks(model, X, y):
    rows = X.toarray() if issparse(X) else np.asarray(X)
    beta, b, q, penalty = model.coef_.ravel(), float(model.intercept_[0]), model.q, model.lambd
    threshold, losses, z = q / (q + 1.), [], []
    for row, label in zip(rows, y):
        margin = float(label) * (math.fsum(float(v) * float(w) for v, w in zip(row, beta)) + b)
        if margin <= threshold:
            losses.append(1. - margin)
            derivative = -1.
        else:
            losses.append((threshold / margin) ** q / (q + 1.))
            derivative = -(threshold / margin) ** (q + 1.)
        z.append(float(label) * derivative / len(y))
    objective = math.fsum(losses) / len(y) + penalty * math.fsum(float(v) ** 2 for v in beta)
    gradient = np.array([math.fsum(z)] + [
        math.fsum(float(row[j]) * value for row, value in zip(rows, z)) + 2 * penalty * beta[j]
        for j in range(len(beta))])
    assert_allclose(model.final_objective_, objective, rtol=RTOL, atol=ATOL)
    assert_allclose(model.gradient_inf_norm_, np.max(np.abs(gradient)), rtol=RTOL, atol=ATOL)
    assert_allclose(model.stationarity_residual_, max(abs(gradient[0]), np.linalg.norm(gradient[1:])),
                    rtol=RTOL, atol=ATOL)


def report(record_property, metrics):
    record_property('numerical_differences', json.dumps(metrics, sort_keys=True))


@pytest.mark.parametrize('sparse', (False, True))
@pytest.mark.parametrize('mode', ('schur',))
@pytest.mark.parametrize('q', (.5, 1., 2.3))
@pytest.mark.parametrize('implicit', (False, True))
def test_inverse_factor_trajectory_contracts(prior, record_property, sparse, mode, q, implicit):
    case = fixture_case(prior)
    X = csr_matrix(case.X) if sparse else case.X
    before = input_state(prior, X)
    options = dict(q=q)
    expected = prior._fit(X, case.y, implicit=implicit, reference=True, options=options)
    actual = prior._fit(X, case.y, implicit=implicit, options=options)
    report(record_property, compare_models(prior, expected, actual, X, bitwise=implicit))
    for model in (expected, actual):
        independent_final_checks(model, X, case.y)
    assert input_state(prior, X) == before


@pytest.mark.parametrize('implicit', (False, True))
@pytest.mark.parametrize('options,cached', (
    ({}, True), ({'stopping': 'objective', 'obj_tol': 0.}, False),
    ({'stopping': 'fixed', 'obj_tol': 0.}, False), ({'max_iter': 0}, False),
    ({'tol': 1e100}, False), ({'max_iter': 1}, False)))
def test_cache_and_stopping_contracts(prior, record_property, implicit, options, cached):
    case = fixture_case(prior)
    before = input_state(prior, case.X)
    expected = prior._fit(case.X, case.y, implicit=implicit, reference=True, options=options, cached=cached)
    with prior._observe() as observed:
        actual = prior._fit(case.X, case.y, implicit=implicit, options=options, cached=cached)
    if options.get('stopping') in ('objective', 'fixed'):
        assert actual.n_iter_ == 6
        assert observed['reuses'] == 0
        assert len(observed['gradients']) == 1
        assert observed['loss_calls'] == 7
    else:
        updates = 0 if options.get('max_iter') == 0 or options.get('tol') == 1e100 else options.get('max_iter', 6)
        gradients = 1 if options.get('max_iter') == 0 else updates + 1 + int(options.get('tol') == 1e100)
        assert actual.n_iter_ == updates
        assert observed['reuses'] == updates
        assert len(observed['gradients']) == gradients
        assert observed['loss_calls'] == gradients
    report(record_property, compare_models(prior, expected, actual, case.X, bitwise=implicit))
    independent_final_checks(actual, case.X, case.y)
    assert input_state(prior, case.X) == before


@pytest.mark.parametrize('implicit', (False, True))
@pytest.mark.parametrize('restore', (False, True))
@pytest.mark.parametrize('hook', ('V_grad', '_linear_gradient', 'step'))
def test_hook_calls_restoration_and_predictions(prior, record_property, implicit, restore, hook):
    case = fixture_case(prior)
    before = input_state(prior, case.X)
    name = ('get_step_implicit' if implicit else 'get_step_explicit') if hook == 'step' else hook
    expected, expected_calls, old_observed = case._hook_fit(implicit, name, reference=True, restore=restore)
    actual, actual_calls, observed = case._hook_fit(implicit, name, restore=restore)
    assert actual_calls == expected_calls
    if restore:
        assert actual_calls == 1
        assert observed['steps'][0] is False
        assert observed['reuses'] == actual.n_iter_ - 1
        assert old_observed['loss_calls'] - observed['loss_calls'] == observed['reuses']
    else:
        assert actual_calls == {'V_grad': 13, '_linear_gradient': 7, 'step': 6}[hook]
        assert observed['reuses'] == 0
        assert old_observed['loss_calls'] == observed['loss_calls']
    report(record_property, compare_models(prior, expected, actual, case.X, bitwise=implicit))
    assert input_state(prior, case.X) == before


def decode_numeric(value):
    if isinstance(value, tuple) and value and value[0] == 'array':
        return np.frombuffer(value[3], dtype=np.dtype(value[2])).reshape(value[1])
    if isinstance(value, tuple) and value and value[0] == 'scalar':
        return np.frombuffer(value[2], dtype=np.dtype(value[1]))[0]
    if isinstance(value, str):
        return float.fromhex(value)
    if isinstance(value, (tuple, list)):
        return np.asarray([decode_numeric(item) for item in value])
    return value


def compare_callback_results(prior, expected, actual, *, bitwise):
    assert expected[0][0] == actual[0][0]
    assert expected[1] == actual[1]  # Exact deliberate mutation, or no mutation.
    assert expected[2] == actual[2]  # All callback and step-handler counts.
    if actual[0][0] == 'raised':
        assert actual[0] == expected[0] == ('raised', 'step handler requested failure', [])
        return {}
    a, b = expected[0][1], actual[0][1]
    assert set(a) == set(b)
    metrics = {}
    for name in a:
        if name in NUMERICAL_FIELDS:
            metrics[name] = numerical_difference(decode_numeric(a[name]), decode_numeric(b[name]))
        else:
            assert a[name] == b[name], name
    X = decode_numeric(actual[1])
    scores_a = X @ decode_numeric(a['coef_']).T + decode_numeric(a['intercept_'])
    scores_b = X @ decode_numeric(b['coef_']).T + decode_numeric(b['intercept_'])
    metrics['scores'] = numerical_difference(scores_a, scores_b)
    classes = decode_numeric(a['classes_'])
    assert_array_equal(classes[(scores_a.ravel() > 0).astype(int)],
                       classes[(scores_b.ravel() > 0).astype(int)])
    if bitwise:
        assert expected[:3] == actual[:3]
    return metrics


@pytest.mark.parametrize('implicit', (False, True))
@pytest.mark.parametrize('policy', ('call', 'log'))
@pytest.mark.parametrize('action', ('raise', 'mutate', 'disable_policy'))
def test_error_callbacks_counts_mutation_cleanup(prior, record_property, implicit, policy, action):
    case = fixture_case(prior)
    expected = case._error_policy_fit(implicit, policy, action, True)
    actual = case._error_policy_fit(implicit, policy, action, False)
    assert expected[3]['loss_calls'] == actual[3]['loss_calls']
    assert actual[3]['reuses'] == 0
    if action == 'mutate':
        assert actual[1] == prior._encoded(np.array([[-4.], [1.]]))
        assert actual[2]['step'] == 1
    elif action == 'disable_policy':
        assert actual[2]['handler'] == 1
        assert actual[3]['loss_calls'] == 3
    report(record_property, compare_callback_results(prior, expected, actual, bitwise=implicit))


@pytest.mark.parametrize('implicit', (False, True))
def test_underflow_warning_and_callback_contracts(prior, record_property, implicit):
    case = fixture_case(prior)
    expected = case._error_policy_fit(implicit, 'warn', 'none', True)
    actual = case._error_policy_fit(implicit, 'warn', 'none', False)
    assert expected[4] == actual[4] + 1
    assert actual[3]['reuses'] == 1
    assert expected[3]['loss_calls'] == actual[3]['loss_calls'] + 1
    assert actual[1] == prior._encoded(np.array([[-1.], [1.]]))
    report(record_property, compare_callback_results(prior, expected, actual, bitwise=implicit))


@contextmanager
def factor_calls():
    from dwd._explicit_factor import ExplicitFactor
    counts = {'solve': 0, 'inverse': 0}
    codes = {ExplicitFactor.solve.__code__: 'solve', ExplicitFactor.inverse_for_hook.__code__: 'inverse'}
    previous = sys.getprofile()
    def observe(frame, event, value):
        if previous is not None:
            previous(frame, event, value)
        if event == 'call' and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1
    sys.setprofile(observe)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize('sparse', (False, True))
@pytest.mark.parametrize('mode', ('schur',))
@pytest.mark.parametrize('q', (.5, 1., 2.3))
@pytest.mark.parametrize('cached', (False, True))
def test_gradient_reuse_on_off_is_bitwise_identical(prior, sparse, mode, q, cached):
    case = fixture_case(prior)
    X = csr_matrix(case.X) if sparse else case.X
    before = input_state(prior, X)
    options = dict(q=q)
    with prior._observe() as enabled, factor_calls() as enabled_factor:
        actual = prior._fit(X, case.y, implicit=False, options=options, cached=cached)
    with patch.object(gdwd, '_REUSE_GRADIENT_DTYPE', np.dtype(np.float32)), prior._observe() as disabled, factor_calls() as disabled_factor:
        expected = prior._fit(X, case.y, implicit=False, options=options, cached=cached)
    assert enabled_factor == disabled_factor == {'solve': 6, 'inverse': 0}
    assert enabled['reuses'] == 6 and disabled['reuses'] == 0
    assert enabled['loss_calls'] == 7 and disabled['loss_calls'] == 13
    compare_models(prior, expected, actual, X, bitwise=True)
    assert input_state(prior, X) == before


@pytest.mark.parametrize('implicit', (False, True))
def test_cv_selection_predictions_and_nonruntime_metadata(prior, record_property, implicit):
    case = fixture_case(prior)
    before = input_state(prior, case.X)
    options = dict(lambd_vals=[.03, .3], q_vals=[.5, 2.], cv=2, implicit_P=implicit,
                   stopping='optimality', tol=0., max_iter=4, random_state=7)
    with prior._step_reference(implicit):
        expected = gdwd.GenDWDCV(**options).fit(case.X, case.y)
    actual = gdwd.GenDWDCV(**options).fit(case.X, case.y)
    assert actual.best_params_ == expected.best_params_
    assert actual.best_score_ == expected.best_score_
    assert_array_equal(actual.predict(case.X), expected.predict(case.X))
    for name in ('all_cv_results_', 'agg_cv_results_'):
        assert prior._encoded(prior._without_runtime(getattr(expected, name))) == prior._encoded(prior._without_runtime(getattr(actual, name)))
    report(record_property, compare_models(prior, expected.best_estimator_, actual.best_estimator_, case.X, bitwise=implicit))
    assert input_state(prior, case.X) == before
