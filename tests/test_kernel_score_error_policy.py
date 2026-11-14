"""Adaptive score constants do not signal; genuine underflow still does.

The public workflow checks use eight fixed updates, not a convergence claim.
Historical arithmetic parity remains covered by the frozen score fixtures.
"""
from fractions import Fraction
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_array, csr_matrix, issparse

import dwd._kernel_scores as scores
import dwd._kernel_solver as solver
from dwd.gen_kern_dwd import KernGDWD
from _kernel_reference import kernel_route


LAYOUTS = ('dense', 'csr_matrix', 'csr_array')
MODES = ('raise', 'warn', 'call', 'log', 'print', 'ignore', 'allraise')
DEFAULT_POLICY = dict(divide='warn', over='warn', invalid='warn', under='ignore')


class Sink:
    def __init__(self):
        self.calls, self.logs = [], []

    def __call__(self, message, flag):
        self.calls.append((str(message), int(flag)))

    def write(self, message):
        self.logs.append(str(message))


def capture(function, mode, capfd):
    """Retain signals and verify the function leaves caller policy untouched."""
    previous, previous_handler = np.geterr(), np.geterrcall()
    requested = (dict.fromkeys(DEFAULT_POLICY, 'raise') if mode == 'allraise'
                 else dict(DEFAULT_POLICY, under=mode))
    handler, result, error = Sink(), None, None
    capfd.readouterr()
    try:
        np.seterrcall(handler)
        np.seterr(**requested)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result = function()
            except Exception as exception:
                error = exception
        assert np.geterr() == requested and np.geterrcall() is handler
        printed = capfd.readouterr()
        signals = dict(error=error, warnings=list(caught), calls=handler.calls,
                       logs=handler.logs, stdout=printed.out, stderr=printed.err)
    finally:
        np.seterr(**previous)
        np.seterrcall(previous_handler)
    assert np.geterr() == previous and np.geterrcall() is previous_handler
    return result, signals


def assert_quiet(signals):
    assert signals['error'] is None, repr(signals['error'])
    assert not signals['warnings']
    assert not signals['calls'] and not signals['logs']
    assert not signals['stdout'] and not signals['stderr']


def as_layout(array, layout):
    if layout == 'dense':
        return array
    return {'csr_matrix': csr_matrix, 'csr_array': csr_array}[layout](array)


def snapshot(value):
    if issparse(value):
        return (value.shape, value.format, snapshot(value.data),
                snapshot(value.indices), snapshot(value.indptr))
    array = np.asarray(value)
    return array.dtype.str, array.shape, array.strides, array.flags.writeable, array.tobytes()


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('mode', MODES)
def test_constant_creation_respects_all_caller_modes(layout, mode, capfd):
    K, alpha = as_layout(np.eye(3), layout), np.array([1., 2., 3.])
    before = snapshot(K), snapshot(alpha)
    actual, signals = capture(lambda: scores.adaptive_kernel_matvec(K, alpha), mode, capfd)
    assert_quiet(signals)
    assert_array_equal(actual, alpha)
    assert (snapshot(K), snapshot(alpha)) == before


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('case', ('zero_coefficients', 'empty_rows'))
def test_zero_and_empty_queries_under_strict_policy(layout, case, capfd):
    K = np.array([[2., -3.], [4., 5.]]) if case == 'zero_coefficients' else np.empty((0, 2))
    K = as_layout(K, layout)
    alpha = np.array([0., -0.]) if case == 'zero_coefficients' else np.array([1., 2.])
    before = snapshot(K), snapshot(alpha)
    actual, signals = capture(lambda: scores.adaptive_kernel_matvec(K, alpha), 'allraise', capfd)
    assert_quiet(signals)
    assert_array_equal(actual, np.zeros(K.shape[0]))
    assert (snapshot(K), snapshot(alpha)) == before


@pytest.mark.parametrize('layout', LAYOUTS)
def test_cancellation_has_independent_rational_reference(layout, capfd):
    dense = np.array([[1., 1., 1.], [2., -3., 2.], [0., 4., 0.]])
    alpha = np.array([1e16, 1., -1e16])
    expected = np.array([float(sum((Fraction.from_float(float(k)) * Fraction.from_float(float(a))
                                   for k, a in zip(row, alpha)), Fraction())) for row in dense])
    assert_array_equal(expected, [1., -3., 4.])
    K = as_layout(dense, layout)
    alpha.flags.writeable = False
    if issparse(K):
        K.data.flags.writeable = K.indices.flags.writeable = K.indptr.flags.writeable = False
    else:
        K.flags.writeable = False
    before = snapshot(K), snapshot(alpha)
    actual, signals = capture(lambda: scores.adaptive_kernel_matvec(K, alpha), 'allraise', capfd)
    assert_quiet(signals)
    assert_array_equal(actual, expected)
    assert (snapshot(K), snapshot(alpha)) == before


@pytest.mark.parametrize('position', ('before_constant', 'after_constant'))
@pytest.mark.parametrize('mode', ('raise', 'call'))
def test_underflow_outside_constant_is_not_suppressed(position, mode, monkeypatch, capfd):
    name = '_dense_l1_upper' if position == 'before_constant' else '_dense_first_tier'
    original, observed = getattr(scores, name), []

    def actual_underflow(*args, **kwargs):
        observed.append(np.geterr())
        assert np.geterr()['under'] == mode
        np.multiply(np.float64(np.finfo(float).tiny), np.float64(.1))
        return original(*args, **kwargs)

    monkeypatch.setattr(scores, name, actual_underflow)
    actual, signals = capture(lambda: scores.adaptive_kernel_matvec(np.eye(2), np.ones(2)), mode, capfd)
    assert len(observed) == 1
    assert not signals['warnings'] and not signals['logs']
    assert not signals['stdout'] and not signals['stderr']
    if mode == 'raise':
        assert isinstance(signals['error'], FloatingPointError)
        assert 'multiply' in str(signals['error'])
        assert not signals['calls']
    else:
        assert signals['error'] is None
        assert signals['calls'] == [('underflow', 4)]
        assert_array_equal(actual, np.ones(2))


def test_gradual_underflow_refusal_is_never_bypassed(monkeypatch, capfd):
    calls = []

    def refuse():
        calls.append(np.geterr())
        raise FloatingPointError('injected gradual-underflow refusal')

    monkeypatch.setattr(scores, '_gradual_underflow', refuse)
    _, signals = capture(lambda: scores.adaptive_kernel_matvec(np.eye(2), np.ones(2)), 'raise', capfd)
    assert isinstance(signals['error'], FloatingPointError)
    assert str(signals['error']) == 'injected gradual-underflow refusal'
    assert len(calls) == 1 and calls[0]['under'] == 'raise'


@pytest.mark.parametrize('location', ('alpha', 'dense_kernel', 'csr_matrix_kernel', 'csr_array_kernel'))
@pytest.mark.parametrize('mode', ('raise', 'warn', 'call'))
def test_real_wider_conversion_underflow_is_preserved(location, mode, capfd):
    info = np.finfo(np.longdouble)
    if info.minexp - info.nmant > -1075:
        pytest.skip('Requires longdouble representation of half the smallest binary64 subnormal')
    with np.errstate(under='ignore'):
        value = np.ldexp(np.longdouble(1), -1075)
    assert value > 0
    if location == 'alpha':
        K, alpha = np.ones((1, 1)), np.array([value], dtype=np.longdouble)
    else:
        K, alpha = np.array([[value]], dtype=np.longdouble), np.ones(1)
        if location != 'dense_kernel':
            K = as_layout(K, location.removesuffix('_kernel'))
    before = snapshot(K), snapshot(alpha)
    actual, signals = capture(lambda: scores.adaptive_kernel_matvec(K, alpha), mode, capfd)
    assert not signals['logs'] and not signals['stdout'] and not signals['stderr']
    if mode == 'raise':
        assert isinstance(signals['error'], FloatingPointError)
        assert 'cast' in str(signals['error'])
        assert not signals['warnings'] and not signals['calls']
    else:
        assert signals['error'] is None
        assert_array_equal(actual, [0.])
        if mode == 'warn':
            assert len(signals['warnings']) == 1
            assert signals['warnings'][0].category is RuntimeWarning
            assert 'cast' in str(signals['warnings'][0].message)
            assert not signals['calls']
        else:
            assert signals['calls'] == [('underflow', 4)]
            assert not signals['warnings']
    assert (snapshot(K), snapshot(alpha)) == before


def model(kind):
    return kernel_route(kernel='precomputed', implementation='reference' if kind == 'reference' else 'optimized', affine_computation='standard',
                    lambd=.1, q=1., max_iter=8, stopping='fixed', initialization='zero',
                    acceleration=None, random_state=13,
                    backend='spectral' if kind == 'reference' else 'cholesky')


@pytest.mark.parametrize('kind', ('optimized_standard', 'reference'))
def test_public_fit_and_prediction_policy(kind, monkeypatch, capfd):
    K, y = np.eye(6), np.array([-1, 1, -1, 1, -1, 1])
    before = snapshot(K), snapshot(y)
    reference_model, strict_model = model(kind), model(kind)
    calls = dict(reference_fit=0, reference_prediction=0, strict_fit=0, strict_prediction=0)
    stage = 'reference_fit'
    production = scores.adaptive_kernel_matvec

    def observed(matrix, coefficients):
        calls[stage] += 1
        return production(matrix, coefficients)

    monkeypatch.setattr(solver, 'adaptive_kernel_matvec', observed)
    monkeypatch.setattr(scores, 'adaptive_kernel_matvec', observed)
    _, signals = capture(lambda: reference_model.fit(K, y), 'ignore', capfd)
    assert signals['error'] is None
    stage = 'reference_prediction'
    expected, signals = capture(lambda: reference_model.decision_function(K), 'ignore', capfd)
    assert signals['error'] is None
    stage = 'strict_fit'
    _, signals = capture(lambda: strict_model.fit(K, y), 'raise', capfd)
    assert signals['error'] is None
    stage = 'strict_prediction'
    actual, signals = capture(lambda: strict_model.decision_function(K), 'raise', capfd)
    assert signals['error'] is None
    assert strict_model.n_iter_ == reference_model.n_iter_ == 8
    for attribute in ('dual_coef_', 'intercept_', 'objective_history_'):
        assert_allclose(getattr(strict_model, attribute), getattr(reference_model, attribute),
                        rtol=1e-12, atol=1e-13)
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
    assert_array_equal(actual > 0, expected > 0)
    # Both current MM implementations share adaptive final reconstruction/readout.
    assert all(count > 0 for count in calls.values())
    assert strict_model.prediction_precision_ == reference_model.prediction_precision_ == 'adaptive'
    assert (snapshot(K), snapshot(y)) == before
