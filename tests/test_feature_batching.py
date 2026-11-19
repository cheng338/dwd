"""Zero-fit checks of bounded batching against the frozen scalar arithmetic.

The fixture is byte-pinned, not a second transcription of the new loops. Exact
Fraction actions and a separate 110-digit directed Decimal dual calculation
also check mathematical meaning. No optimizer or public fit is invoked here.
"""
from decimal import Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
import gc
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import sys
import weakref

import numpy as np
import pytest

from dwd import _linear_feature_diagnostics as batch
from dwd import _quadratic_bounds as bounds


FIXTURE = Path(__file__).resolve().parent/'fixtures'/'scalar_feature_diagnostics.py'
FIXTURE_SHA = 'fc1b138e92347fc22da231d9fa9780f504e05aa1cadffd617f7f067800e2777d'
assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA
SPEC = importlib.util.spec_from_file_location('dwd._ir13_scalar_feature_oracle', FIXTURE)
scalar = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scalar
SPEC.loader.exec_module(scalar)


def frac(value):
    return Fraction.from_float(float(value))


def exact_dot(left, right):
    return sum((frac(a)*frac(b) for a, b in zip(left, right)), Fraction())


def same(actual, expected):
    if isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, e in zip(actual, expected):
            same(a, e)
    elif isinstance(expected, (float, np.floating)):
        assert struct.pack('!d', actual) == struct.pack('!d', expected)
    else:
        assert actual == expected


@pytest.fixture(autouse=True)
def unchanged_policy():
    policy, callback = np.geterr(), np.geterrcall()
    yield
    assert np.geterr() == policy and np.geterrcall() is callback


def layout(values, kind):
    if kind == 'c':
        return np.array(values, order='C')
    if kind == 'f':
        return np.array(values, order='F')
    storage = np.zeros((values.shape[0]*2, values.shape[1]*2))
    storage[::2, ::2] = values
    return storage[::2, ::2]


def data(n, p, kind='c'):
    rng = np.random.RandomState(1100+n+p)
    F = layout(rng.randint(-15, 16, size=(n, p))/32., kind)
    alpha = rng.randint(-7, 8, size=n)/64.
    z = rng.randint(-7, 8, size=n)/128.
    signed = rng.randint(-4, 5, size=n)/8.
    return F, alpha, z, signed


def scalar_upper(F, signed):
    result = []
    for column in F.T:
        value, error = scalar._compensated_dot(column, signed)
        result.append(math.nextafter(abs(value)+error, math.inf))
    return np.array(result)


BOUNDARIES = [(n, p, kind) for n in (32, 33, 128)
              for width in (min(64, (1 << 20)//(512*n)),)
              for p in (width-1, width, width+1, 2*width+3)
              for kind in ('c', 'f', 'strided')]
BOUNDARIES += [(n, p, 'strided') for n in (2048, 2049) for p in (1, 2, 5)]


@pytest.mark.parametrize('n,p,kind', BOUNDARIES)
def test_tile_boundaries_and_layouts_are_bitwise_scalar(n, p, kind):
    F, alpha, z, signed = data(n, p, kind)
    if p % 2:
        F.setflags(write=False)
    pins = [(v.tobytes(), v.flags.writeable) for v in (F, alpha, z, signed)]
    same(batch._feature_coefficients(F, alpha), scalar._feature_coefficients(F, alpha))
    same(batch._feature_gradient(F, z, alpha, .75), scalar._feature_gradient(F, z, alpha, .75))
    same(batch._dual_coordinate_upper(F, signed), scalar_upper(F, signed))
    assert [(v.tobytes(), v.flags.writeable) for v in (F, alpha, z, signed)] == pins


@pytest.mark.parametrize('n,p', [(1, 1), (1, 4096), (32, 1000), (33, 1000),
                               (128, 1000), (2048, 1000), (2049, 1000), (10000, 3)])
def test_tile_width_obeys_finite_buffer_budget(n, p):
    width = batch._tile_columns(n, p)
    assert width == max(1, min(64, p, (1 << 20)//(512*n)))
    assert width == 1 or 512*n*width <= 1 << 20


def expected_report_inputs(F, y, alpha, offset, lambd, q):
    beta = [exact_dot(column, alpha) for column in F.T]
    score = [sum((frac(x)*b for x, b in zip(row, beta)), Fraction()) for row in F]
    with localcontext() as context:
        context.prec = 110
        qq = Decimal.from_float(q)
        threshold = qq/(qq+1)
        losses = []
        for label, value in zip(y, score):
            margin = frac(label)*(value+frac(offset))
            t = Decimal(margin.numerator)/Decimal(margin.denominator)
            losses.append(1-t if t <= threshold else
                          (qq/(qq+1))**qq/(qq+1)*t**(-qq))
        penalty = frac(lambd)*sum((b*b for b in beta), Fraction())
        objective = sum(losses)/len(y)+Decimal(penalty.numerator)/Decimal(penalty.denominator)
    return np.array([float(v) for v in score]), float(objective)


@pytest.mark.parametrize('scenario', ['q02', 'q1', 'q23', 'cancellation', 'zero_rank', 'small_C'])
def test_complete_report_and_metadata_bitwise_scalar(scenario):
    n, p, q, offset, lambd = 6, 7, 1., .8, .75
    F, alpha, _, _ = data(n, p, 'strided')
    F = F/4
    y = np.array([-1., 1.]*3)
    if scenario.startswith('q'):
        q = {'q02': .2, 'q1': 1., 'q23': 2.3}[scenario]
    elif scenario == 'cancellation':
        F = np.array([[1.]*3, [1.]*3, [0.]*3, [0.]*3])
        y, alpha = np.array([1., 1., -1., -1.]), np.array([2.**53, -2.**53, 0., 0.])
        offset, lambd = 0., .5
    elif scenario == 'zero_rank':
        F = np.zeros((n, p))
        offset = 0.
    else:
        F, y = np.ldexp(np.eye(2), -400), np.array([-1., 1.])
        alpha, q, offset, lambd = np.ldexp(y, -300), .2, 0., 1.
    scores, objective = expected_report_inputs(F, y, alpha, offset, lambd, q)
    kwargs = dict(checked_scores=scores, checked_objective=objective,
                  optimizer_scores=scores, optimizer_objective=objective)
    original = [v.copy() for v in (F, y, alpha, scores)]
    actual = batch.linear_feature_diagnostics(F, y, alpha, offset, lambd, q, **kwargs)
    expected = scalar.linear_feature_diagnostics(F, y, alpha, offset, lambd, q, **kwargs)
    same(actual, expected)
    if scenario == 'cancellation':
        assert actual[0]['rkhs_gradient_norm'] == math.sqrt(.75)
        assert actual[0]['gradient_inf_norm'] == 1.5
    if scenario == 'small_C':
        assert actual[0]['rkhs_norm_squared'] == 0. and actual[0]['C'] > 0.
    for a, e in zip((F, y, alpha, scores), original):
        same(a, e)


def outcome(function):
    try:
        return ('returned', function())
    except (FloatingPointError, OverflowError) as error:
        return ('refused', type(error).__name__, str(error))


def invoke(module, family, F, vector):
    if family == 'beta':
        return module._feature_coefficients(F, vector)
    if family == 'gradient':
        return module._feature_gradient(F, vector, np.zeros(len(vector)), 1.)
    return scalar_upper(F, vector) if module is scalar else module._dual_coordinate_upper(F, vector)


def ordered_failure(family, reverse=False):
    if family == 'dual':
        F = np.array([[2.**1022, 2.**1023], [2.**1022, 0.]])
        vector = np.full(2, 2.)
    else:
        F = np.array([[2.**1023, float.fromhex('0x1p-1074')], [2.**1023, 0.]])
        vector = np.ones(2)
    return (F[:, ::-1] if reverse else F), vector


@pytest.mark.parametrize('family', ['beta', 'gradient', 'dual'])
@pytest.mark.parametrize('reverse', [False, True])
def test_original_first_column_error_precedes_later_tile_error(family, reverse):
    F, vector = ordered_failure(family, reverse)
    expected = outcome(lambda: invoke(scalar, family, F, vector))
    assert expected[0] == 'refused'
    same(outcome(lambda: invoke(batch, family, F, vector)), expected)


@pytest.mark.parametrize('case', ['beta_erasure', 'gradient_erasure', 'beta_overflow',
                                 'gradient_overflow', 'triple_regularizer', 'cancelling_gradient'])
def test_range_adversaries_retain_original_result_or_refusal(case):
    if case == 'triple_regularizer':
        F, alpha = np.ldexp(np.eye(2), 500), np.ldexp(np.array([-1., 1.]), -500)
        args = (F, np.zeros(2), alpha, 2.**-699)
        actual, expected = batch._feature_gradient(*args), scalar._feature_gradient(*args)
        same(actual, expected)
        same(actual, np.ldexp(np.array([-1., 1.]), -699))
    elif case == 'cancelling_gradient':
        F = np.array([[1.]*3, [1.]*3, [0.]*3, [0.]*3])
        alpha = np.array([2.**53, -2.**53, 0., 0.])
        z = np.array([-.25, -.25, .25, .25])
        expected = np.full(3, -.5)
        same(batch._feature_gradient(F, z, alpha, 1.), expected)
        same(scalar._feature_gradient(F, z, alpha, 1.), expected)
    else:
        family = case.split('_')[0]
        F, vector = ordered_failure(family, reverse='erasure' in case)
        if 'erasure' in case:
            # The first product is genuinely erased, not merely subnormal;
            # a later large feature could otherwise amplify the lost beta.
            F = np.array([[2.**-500, 1.], [2.**500, 0.]])
            vector = np.array([2.**-700, 0.])
        expected = outcome(lambda: invoke(scalar, family, F, vector))
        assert expected[0] == 'refused'
        same(outcome(lambda: invoke(batch, family, F, vector)), expected)


TILE_NAMES = {'beta': '_beta_tile', 'gradient': '_gradient_tile', 'dual': '_dual_dot_tile'}


@pytest.mark.parametrize('family', ['beta', 'gradient', 'dual'])
@pytest.mark.parametrize('failure', [False, True])
def test_allocation_failure_replays_scalar_and_disables_remaining_tiles(monkeypatch, family, failure):
    F, vector, _, _ = data(2, 131)
    if failure:
        first, vector = ordered_failure(family)
        F[:, :2] = first
    pins = F.tobytes(), vector.tobytes()
    expected = outcome(lambda: invoke(scalar, family, F, vector))
    calls, references = [], []
    def unavailable(columns, *args):
        calls.append(columns.shape)
        references.append(weakref.ref(columns))
        raise MemoryError('deliberate bounded-tile allocation failure')
    monkeypatch.setattr(batch, TILE_NAMES[family], unavailable)
    same(outcome(lambda: invoke(batch, family, F, vector)), expected)
    assert calls == [(64, 2)]
    gc.collect()
    assert all(reference() is None for reference in references)
    assert (F.tobytes(), vector.tobytes()) == pins


@pytest.mark.parametrize('family', ['beta', 'gradient', 'dual'])
def test_unexpected_tile_exception_is_not_suppressed(monkeypatch, family):
    F, vector, _, _ = data(4, 7)
    sentinel = RuntimeError('programming error sentinel')
    def broken(*args):
        raise sentinel
    monkeypatch.setattr(batch, TILE_NAMES[family], broken)
    with pytest.raises(RuntimeError) as caught:
        invoke(batch, family, F, vector)
    assert caught.value is sentinel


@pytest.mark.parametrize('family', ['beta', 'gradient', 'dual'])
def test_numerical_tile_refusal_replays_then_keeps_later_batching(monkeypatch, family):
    F, vector, _, _ = data(2, 131)
    original = getattr(batch, TILE_NAMES[family])
    calls = []
    def first_refuses(columns, *args):
        calls.append(columns.shape)
        if len(calls) == 1:
            raise FloatingPointError('deliberate tile-only numerical refusal')
        return original(columns, *args)
    monkeypatch.setattr(batch, TILE_NAMES[family], first_refuses)
    same(invoke(batch, family, F, vector), invoke(scalar, family, F, vector))
    assert calls == [(64, 2), (64, 2), (3, 2)]


@pytest.mark.parametrize('mode', ['ignore', 'raise', 'call', 'log'])
def test_caller_error_policy_and_handler_are_preserved(mode):
    F, alpha, z, signed = data(8, 9)
    observed = []
    class Log:
        def write(self, message):
            observed.append(message)
    callback = Log() if mode == 'log' else lambda *args: observed.append(args)
    previous = np.geterrcall()
    try:
        np.seterrcall(callback)
        with np.errstate(all=mode):
            same(batch._feature_coefficients(F, alpha), scalar._feature_coefficients(F, alpha))
            same(batch._feature_gradient(F, z, alpha, .75), scalar._feature_gradient(F, z, alpha, .75))
            same(batch._dual_coordinate_upper(F, signed), scalar_upper(F, signed))
            assert np.geterr() == dict.fromkeys(('divide', 'over', 'under', 'invalid'), mode)
            assert np.geterrcall() is callback
        assert observed == []
    finally:
        np.seterrcall(previous)


@pytest.mark.parametrize('kind', ['ordinary', 'cancellation', 'subnormal_products', 'zero', 'strided'])
def test_dual_coordinate_value_error_and_upper_match_and_bound_exact_action(kind):
    F, _, _, signed = data(8, 7, 'strided' if kind == 'strided' else 'c')
    if kind == 'cancellation':
        F[::2] = 2.**50
        F[1::2] = -2.**50
        signed = np.ones(8)
    elif kind == 'subnormal_products':
        F.fill(float.fromhex('0x1p-1074'))
        signed = np.full(8, .5)
    elif kind == 'zero':
        F.fill(0.)
    results = batch._dual_dot_tile(F.T, signed, batch._dual_dot_constants(len(signed)))
    for column, result in zip(F.T, results):
        value, error = scalar._compensated_dot(column, signed)
        upper = math.nextafter(abs(value)+error, math.inf)
        same(result, (value, error, upper))
        exact = exact_dot(column, signed)
        assert abs(exact-frac(value)) <= frac(error)
        assert abs(exact) <= frac(upper)
    same(batch._dual_coordinate_upper(F, signed), scalar_upper(F, signed))


def test_dual_reduction_keeps_original_imported_fsum_binding(monkeypatch):
    F, _, _, signed = data(8, 7)
    expected = scalar_upper(F, signed)
    constants = batch._dual_dot_constants(len(signed))
    expected_triples = []
    for column in F.T:
        value, error = scalar._compensated_dot(column, signed)
        expected_triples.append((value, error, math.nextafter(abs(value)+error, math.inf)))
    def unexpected(*args):
        raise RuntimeError('live math.fsum is not the original quadratic-bound binding')
    with monkeypatch.context() as patch:
        patch.setattr(math, 'fsum', unexpected)
        actual = batch._dual_coordinate_upper(F, signed)
        original = scalar_upper(F, signed)
        triples = batch._dual_dot_tile(F.T, signed, constants)
    same(actual, expected)
    same(original, expected)
    same(triples, expected_triples)


@pytest.mark.parametrize('attribute', ['fsum', '_upward_nonnegative'])
def test_dual_reduction_observes_same_quadratic_bound_binding(monkeypatch, attribute):
    F, _, _, signed = data(8, 7)
    constants = batch._dual_dot_constants(len(signed))
    sentinel = RuntimeError('quadratic-bound accumulator replacement')
    def replaced(*args):
        raise sentinel
    with monkeypatch.context() as patch:
        patch.setattr(bounds, attribute, replaced)
        for operation in (lambda: scalar_upper(F, signed),
                          lambda: batch._dual_coordinate_upper(F, signed),
                          lambda: batch._dual_dot_tile(F.T, signed, constants)):
            with pytest.raises(RuntimeError) as caught:
                operation()
            assert caught.value is sentinel


def independent_dual_lower(F, y, rho, lambd, q):
    signed = [frac(label)*frac(slope) for label, slope in zip(y, rho)]
    assert all(0 <= frac(value) <= 1 for value in rho) and sum(signed) == 0
    quadratic = sum((sum((frac(x)*s for x, s in zip(column, signed)), Fraction())**2
                     for column in F.T), Fraction())
    exponent = frac(q)/(frac(q)+1)
    with localcontext() as context:
        context.prec, context.rounding = 110, ROUND_CEILING
        upper_exponent = Decimal(exponent.numerator)/Decimal(exponent.denominator)
        penalty = quadratic/(4*frac(lambd)*len(y)**2)
        upper_penalty = Decimal(penalty.numerator)/Decimal(penalty.denominator)
    with localcontext() as context:
        context.prec, context.rounding = 110, ROUND_FLOOR
        total = Decimal(0)
        for value in rho:
            r = Decimal.from_float(float(value))
            total += r if r in (0, 1) else (upper_exponent*r.ln().next_minus()).exp().next_minus()
        return Fraction(total/len(y)-upper_penalty)


@pytest.mark.parametrize('q', [.2, 1., 2.3])
def test_complete_dual_endpoint_preserves_witness_and_independent_bound(monkeypatch, q):
    F, _, _, _ = data(8, 7, 'strided')
    F = F/16
    y = np.array([-1., 1.]*4)
    raw = np.array([.1, .3, .5, .9, .7, .2, .8, .4])
    witness = []
    original = batch.feasible_dual_slopes
    def observe(*args):
        rho = original(*args)
        witness.append(rho.copy())
        return rho
    monkeypatch.setattr(batch, 'feasible_dual_slopes', observe)
    actual = batch._dual_endpoint(F, y, raw, .75, q)
    same(actual, scalar._dual_endpoint(F, y, raw, .75, q))
    assert len(witness) == 1
    lower = independent_dual_lower(F, y, witness[0], .75, q)
    assert lower > 0 and actual[0] > 0
    assert frac(actual[0]) <= lower
    assert actual[1]['fallback_reason'] is None


@pytest.mark.parametrize('reason', ['weak_bound', 'arithmetic_refusal'])
def test_zero_dual_fallback_is_unchanged(reason):
    # Opposite features and labels give a large nonzero signed contraction.
    magnitude = 2.**10 if reason == 'weak_bound' else 2.**1023
    F = np.array([[magnitude]*3, [-magnitude]*3])
    y, slopes = np.array([1., -1.]), np.ones(2)
    actual = batch._dual_endpoint(F, y, slopes, .75, 1.)
    same(actual, scalar._dual_endpoint(F, y, slopes, .75, 1.))
    assert actual[0] == 0.
    expected = ('zero_slopes_give_a_stronger_certified_bound' if reason == 'weak_bound'
                else 'nonzero_feature_dual_bound_unavailable')
    assert actual[1]['fallback_reason'] == expected
