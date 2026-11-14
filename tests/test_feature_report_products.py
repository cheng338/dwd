"""Raw and joint feature scores retain independent sums and original dispatch.

The oracles are separate installed score operations and exact dyadic arithmetic.
Broad full-report, layout, batching and public-estimator coverage stays in the
existing feature arithmetic/public and feature-batching modules.
"""
import math
import sys
from types import ModuleType
from fractions import Fraction

import numpy as np
import pytest

from dwd import _affine_scores as affine
from dwd import _compensated_residual as arithmetic
from dwd import _compiled_residual as native
from dwd import _kernel_scores as raw

from _feature_report_test_utils import (feature, support, same, exact_dot,
    fraction as frac, separate, pins as _pins, numpy_policy, portable_scores)

pytestmark = pytest.mark.usefixtures('numpy_policy')
ETA = float.fromhex('0x1p-1074')


def outcome(operation):
    try:
        return ('returned', operation())
    except (FloatingPointError, OverflowError, ValueError, MemoryError) as error:
        return ('refused', type(error).__name__, str(error))

def with_layout(values, layout):
    if layout == 'C':
        return np.array(values, dtype=np.float64, order='C')
    if layout == 'F':
        return np.array(values, dtype=np.float64, order='F')
    if layout == 'readonly':
        result = np.array(values, dtype=np.float64)
        result.flags.writeable = False
        return result
    storage = np.zeros((2*values.shape[0], 2*values.shape[1]))
    storage[::2, ::2] = values
    return storage[::2, ::2]


def score_case(name):
    if name == 'ordinary':
        rng = np.random.RandomState(47055)
        return rng.randint(-15, 16, size=(5, 9))/16., rng.randint(-7, 8, size=9)/32., .125
    if name == 'round_then_add':
        return np.array([[1., 2.**-53], [1., -2.**-53]]), np.ones(2), -1.
    if name == 'large_cancellation':
        return np.array([[2.**500, 1., -2.**500], [2.**53, -1., -2.**53]]), np.ones(3), .5
    if name == 'offset_cancellation':
        return np.array([[2.**53, 1., -2.**53]]), np.ones(3), -1.
    if name == 'zero_offset_underflow':
        return np.array([[ETA, ETA], [ETA, -ETA]]), np.full(2, .5), 0.
    if name == 'original_affine_underflow':
        mantissa = .5+2.**-53
        x, a = math.ldexp(mantissa, -484), math.ldexp(mantissa, -485)
        offset = -2.*float(frac(x)*frac(a))
        return np.array([[x, x]]), np.array([a, a]), offset
    if name == 'maximum_cancellation':
        return np.ones((2, 2)), np.array([np.finfo(float).max, -np.finfo(float).max]), 1.
    if name == 'zero_features':
        return np.zeros((3, 4)), np.full(4, np.finfo(float).max), -0.
    raise AssertionError(name)


SCORE_CASES = ['ordinary', 'round_then_add', 'large_cancellation',
               'offset_cancellation', 'zero_offset_underflow',
               'original_affine_underflow', 'maximum_cancellation', 'zero_features']


@pytest.mark.parametrize('case', SCORE_CASES)
def test_raw_joint_pair_matches_separate_and_exact_oracle(portable_scores, case):
    matrix, beta, offset = score_case(case)
    matrix = with_layout(matrix, 'strided')
    before = matrix.tobytes(), beta.tobytes(), matrix.flags.writeable
    expected = separate(matrix, beta, offset)
    actual = feature._feature_scores_and_margins(matrix, beta, offset)
    same(actual, expected)
    for index, row in enumerate(matrix):
        dot = exact_dot(row, beta)
        joint = dot+frac(offset)
        # All panel inputs are deliberately chosen so the joint exact value is
        # either exactly accumulated or routed through established exact recovery.
        assert float(actual[1][index]).hex() == float(joint).hex()
        # Compensated raw products need not recover erased subnormal terms.
        # This explicit dyadic bound allows only their documented product error
        # and conservative final-sum rounding, not a generic allclose tolerance.
        bound = 2*len(beta)*frac(ETA)+4*frac(np.finfo(float).eps)*abs(frac(actual[0][index]))+frac(ETA)
        assert abs(dot-frac(actual[0][index])) <= bound
    if case == 'round_then_add':
        assert actual[0][0]+offset == 0. and actual[1][0] == 2.**-53
    if case == 'zero_offset_underflow':
        assert actual[0][0] == 0. and actual[1][0] == ETA
    if case == 'original_affine_underflow':
        assert actual[1][0] == ETA
    assert (matrix.tobytes(), beta.tobytes(), matrix.flags.writeable) == before


@pytest.mark.parametrize('n,p', [(17, 65), (3, 2049)])
def test_row_and_width_boundaries_keep_distinct_results(portable_scores, n, p):
    rng = np.random.RandomState(191+n+p)
    matrix = with_layout(rng.randint(-3, 4, size=(n, p))/16., 'strided')
    beta = rng.randint(-3, 4, size=p)/32.
    actual = feature._feature_scores_and_margins(matrix, beta, -.125)
    same(actual, separate(matrix, beta, -.125))
    same(actual[0], np.array([float(exact_dot(row, beta)) for row in matrix]))
    same(actual[1], np.array([float(exact_dot(row, beta)-Fraction(1, 8)) for row in matrix]))


@pytest.mark.parametrize('case', ['individual_product', 'raw_sum', 'raw_sum_affine_finite',
                                 'later_raw_row', 'nonfinite_feature', 'nonfinite_beta'])
def test_existing_raw_refusal_is_not_broadened_by_affine_recovery(portable_scores, case):
    maximum = np.finfo(float).max
    matrix, beta, offset = np.ones((1, 2)), np.ones(2), 0.
    if case == 'individual_product':
        matrix, beta, offset = np.array([[2., 2.]]), np.array([maximum, -maximum]), 1.
        assert exact_dot(matrix[0], beta)+frac(offset) == 1
    elif case == 'raw_sum':
        matrix, beta = np.array([[maximum, maximum, -maximum, -maximum]]), np.ones(4)
    elif case == 'raw_sum_affine_finite':
        matrix, offset = np.array([[maximum, maximum]]), -maximum
        assert float(exact_dot(matrix[0], beta)+frac(offset)) == maximum
    elif case == 'later_raw_row':
        matrix, beta = np.array([[1., -1.], [maximum, maximum]]), np.ones(2)
    elif case == 'nonfinite_feature':
        matrix[0, 0] = np.nan
    else:
        beta[0] = np.inf
    expected = outcome(lambda: separate(matrix, beta, offset))
    assert expected[0] == 'refused'
    same(outcome(lambda: feature._feature_scores_and_margins(matrix, beta, offset)), expected)


HOOKS = [(raw, '_products'), (raw, '_fsum'), (raw, '_expanded_dense_rows'),
         (affine, '_finish_row'), (affine, '_exact_row')]


@pytest.mark.parametrize('module,attribute', HOOKS, ids=[m.__name__.split('.')[-1]+'.'+a for m, a in HOOKS])
def test_replaced_existing_arithmetic_hooks_keep_dispatch(portable_scores, monkeypatch, module, attribute):
    original = getattr(module, attribute)
    observed = []
    def observing(*args, **kwargs):
        observed.append((attribute, tuple(np.shape(a) for a in args)))
        return original(*args, **kwargs)
    monkeypatch.setattr(module, attribute, observing)
    total_observations = 0
    # A singleton exercises scalar hooks; multiple rows exercise batch hooks.
    for count in (1, 3):
        matrix, beta = np.ones((count, 2)), np.array([1., -1.])
        expected = separate(matrix, beta, 0.)
        original_calls = list(observed)
        total_observations += len(original_calls)
        observed.clear()
        same(feature._feature_scores_and_margins(matrix, beta, 0.), expected)
        assert observed == original_calls
        observed.clear()
    assert total_observations > 0


def test_all_raw_rows_precede_any_affine_recovery(portable_scores, monkeypatch):
    events = []
    def forbidden(*args, **kwargs):
        events.append('affine_recovery')
        raise RuntimeError('affine must not precede the later raw failure')
    monkeypatch.setattr(affine, '_exact_row', forbidden)
    maximum = np.finfo(float).max
    matrix = np.array([[1., -1.], [maximum, maximum]])
    beta = np.ones(2)
    expected = outcome(lambda: separate(matrix, beta, 0.))
    assert expected[0] == 'refused' and events == []
    same(outcome(lambda: feature._feature_scores_and_margins(matrix, beta, 0.)), expected)
    assert events == []


@pytest.mark.parametrize('attribute', ['compensated_kernel_matvec', 'affine_scores'])
def test_high_level_replacement_is_not_bypassed(monkeypatch, attribute):
    sentinel = RuntimeError('original high-level dispatch sentinel')
    def replaced(*args, **kwargs):
        raise sentinel
    monkeypatch.setattr(feature, attribute, replaced)
    with pytest.raises(RuntimeError) as caught:
        feature._feature_scores_and_margins(np.eye(2), np.ones(2), 0.)
    assert caught.value is sentinel


@pytest.mark.parametrize('module,attribute', [(raw, 'compensated_kernel_matvec'), (affine, '_finish_row')])
def test_in_place_code_replacement_keeps_original_dispatch(portable_scores, monkeypatch, module, attribute):
    function = getattr(module, attribute)
    def changed(*args, **kwargs):
        raise RuntimeError('in-place code replacement sentinel')
    monkeypatch.setattr(function, '__code__', changed.__code__)
    matrix, beta, offset = np.eye(2), np.ones(2), 0.
    assert not support.can_share_products(
        matrix, beta, feature.compensated_kernel_matvec,
        feature.affine_scores)
    for operation in (lambda: separate(matrix, beta, offset),
                      lambda: feature._feature_scores_and_margins(matrix, beta, offset)):
        with pytest.raises(RuntimeError, match='in-place code replacement sentinel'):
            operation()


def test_changed_defaults_decline_without_changing_existing_result(portable_scores, monkeypatch):
    monkeypatch.setattr(affine._finish_score, '__defaults__', ())
    matrix, beta, offset = score_case('ordinary')
    assert not support.can_share_products(
        matrix, beta, feature.compensated_kernel_matvec,
        feature.affine_scores)
    same(feature._feature_scores_and_margins(matrix, beta, offset),
         separate(matrix, beta, offset))


@pytest.mark.parametrize('kind', ['custom_equality', 'array'])
def test_changed_keyword_default_is_compared_by_identity(portable_scores, monkeypatch, kind):
    class NoEquality:
        def __eq__(self, other):
            raise AssertionError('a compatibility guard must not invoke custom equality')
    replacement = NoEquality() if kind == 'custom_equality' else np.ones(2)
    monkeypatch.setattr(arithmetic._products, '__kwdefaults__', {'_split_b': replacement})
    matrix, beta, offset = score_case('ordinary')
    assert not support.can_share_products(
        matrix, beta, feature.compensated_kernel_matvec,
        feature.affine_scores)
    # Both existing dense helpers supply their prepared split explicitly.
    same(feature._feature_scores_and_margins(matrix, beta, offset),
         separate(matrix, beta, offset))


@pytest.mark.parametrize('mode', ['ignore', 'raise', 'call', 'log'])
def test_caller_error_policy_and_handler_are_unchanged(portable_scores, mode):
    events = []
    class Logger:
        def write(self, message):
            events.append(message)
    callback = Logger() if mode == 'log' else lambda *args: events.append(args)
    previous = np.geterrcall()
    matrix, beta, offset = score_case('zero_offset_underflow')
    try:
        np.seterrcall(callback)
        with np.errstate(all=mode):
            expected = separate(matrix, beta, offset)
            expected_events = list(events)
            events.clear()
            same(feature._feature_scores_and_margins(matrix, beta, offset), expected)
            assert events == expected_events
            assert np.geterrcall() is callback
            assert np.geterr() == dict.fromkeys(('divide', 'over', 'under', 'invalid'), mode)
    finally:
        np.seterrcall(previous)


@pytest.mark.parametrize('error_class', [MemoryError, FloatingPointError, OverflowError, ValueError])
def test_expected_fused_failure_replays_original_whole_operations(portable_scores, monkeypatch, error_class):
    matrix, beta, offset = score_case('round_then_add')
    expected = separate(matrix, beta, offset)
    calls = []
    def fail(*args):
        calls.append(True)
        raise error_class('controlled fused attempt failure')
    monkeypatch.setattr(support, '_fused_pair', fail)
    same(feature._feature_scores_and_margins(matrix, beta, offset), expected)
    assert calls == [True]


@pytest.mark.parametrize('error_class', [RuntimeError, KeyboardInterrupt])
def test_unexpected_fused_failure_keeps_original_exception(portable_scores, monkeypatch, error_class):
    sentinel = error_class('unexpected fused implementation failure')
    def fail(*args):
        raise sentinel
    monkeypatch.setattr(support, '_fused_pair', fail)
    with pytest.raises(error_class) as caught:
        feature._feature_scores_and_margins(np.eye(2), np.ones(2), 0.)
    assert caught.value is sentinel


def _stack_contains(code):
    """Observe an existing call stack without wrapping a guarded helper."""
    frame = sys._getframe(1)
    try:
        while frame is not None:
            if frame.f_code is code:
                return True
            frame = frame.f_back
        return False
    finally:
        del frame


class InputCallbackError(LookupError):
    """Distinct sentinel; not a numerical range error that permits replay."""


_EXPECTED_PAIR_EVENTS = {
    'finfo': [('finfo', 'float'), ('finfo', 'float')],
    # Affine validates its intercept before its second whole input validation.
    'iscomplexobj': [('iscomplexobj', (3,)), ('iscomplexobj', (2, 3)),
                     ('iscomplexobj', ()), ('iscomplexobj', (3,)),
                     ('iscomplexobj', (2, 3))],
    'issparse': [('issparse', (2, 3)), ('issparse', (2, 3))],
    'zeros': [('zeros', 2, '|b1'), ('zeros', 2, '|b1')],
}
_LATE_CALLBACK = {'finfo': 2, 'iscomplexobj': 4, 'issparse': 2, 'zeros': 2}


def _pair_outcome(operation):
    try:
        return ('returned', operation())
    except Exception as error:
        return ('raised', type(error).__name__, str(error))


@pytest.mark.parametrize('hook', ['finfo', 'iscomplexobj', 'issparse', 'zeros'])
@pytest.mark.parametrize('failure', ['none', 'first', 'late'])
def test_raw_joint_pair_preserves_original_input_callbacks_and_exceptions(
        monkeypatch, portable_scores, hook, failure):
    """Pair-helper compatibility, not an estimator fit or future guard test."""
    features = np.array([[1., .25, -.5], [.5, -.25, 1.]])
    beta, offset = np.array([.5, .25, .125]), .25
    before = _pins((features, beta))
    owner = raw if hook == 'issparse' else np
    original = getattr(owner, hook)
    threshold = None if failure == 'none' else 1 if failure == 'first' else _LATE_CALLBACK[hook]
    message = 'original %s callback %s' % (hook, threshold)

    def run(module, combined):
        events = []
        with monkeypatch.context() as scoped:
            def observing(*args, **kwargs):
                in_inputs = _stack_contains(raw._inputs.__code__)
                in_intercept = _stack_contains(affine._intercept.__code__)
                relevant = ((hook in ('finfo', 'issparse') and in_inputs)
                            or (hook == 'iscomplexobj' and (in_inputs or in_intercept))
                            or (hook == 'zeros' and _stack_contains(affine.affine_scores.__code__)))
                if relevant:
                    if hook == 'finfo':
                        event = (hook, 'float' if args[0] is float else repr(args[0]))
                    elif hook == 'zeros':
                        event = (hook, args[0], np.dtype(kwargs['dtype']).str)
                    else:
                        event = (hook, tuple(np.shape(args[0])))
                    events.append(event)
                    if len(events) == threshold:
                        raise InputCallbackError(message)
                return original(*args, **kwargs)

            scoped.setattr(owner, hook, observing)
            if combined:
                operation = lambda: module._feature_scores_and_margins(features, beta, offset)
            else:
                def operation():
                    raw = module.compensated_kernel_matvec(features, beta)
                    joint = module.affine_scores(features, beta, offset, policy='compensated')
                    return raw, joint
            result = _pair_outcome(operation)
        assert getattr(owner, hook) is original
        return result, events

    expected, expected_events = run(feature, False)
    actual, events = run(feature, True)
    assert expected_events == (_EXPECTED_PAIR_EVENTS[hook] if threshold is None
                               else _EXPECTED_PAIR_EVENTS[hook][:threshold])
    if threshold is None:
        assert expected[0] == 'returned'
        same(expected[1], (np.array([.5, .3125]), np.array([.75, .5625])))
    else:
        assert expected == ('raised', 'InputCallbackError', message)
    same(actual, expected)
    assert events == expected_events
    assert _pins((features, beta)) == before

@pytest.mark.parametrize('layout', ['C', 'F', 'readonly'])
def test_ordinary_score_layout_and_input_ownership(portable_scores, layout):
    matrix, beta, offset = score_case('ordinary')
    matrix = with_layout(matrix, layout)
    before = _pins((matrix, beta))
    same(feature._feature_scores_and_margins(matrix, beta, offset), separate(matrix, beta, offset))
    assert _pins((matrix, beta)) == before


def test_installed_native_entry_keeps_original_scores(monkeypatch):
    # Actual availability is observed; no fake accelerator is made to compute.
    matrix, beta, offset = score_case('ordinary')
    expected = separate(matrix, beta, offset)
    entry = (vars(native._ACCEL).get('score_rows')
             if type(native._ACCEL) is ModuleType else None)
    if entry is not None:
        def forbidden(*args, **kwargs):
            raise AssertionError('native entry presence must bypass the fusion guard')
        monkeypatch.setattr(support, 'feature_scores_and_margins', forbidden)
    same(feature._feature_scores_and_margins(matrix, beta, offset), expected)


@pytest.mark.parametrize('failure', [None, 'raw', 'affine'])
def test_native_presence_preserves_whole_operation_order(monkeypatch, failure):
    class Entry:
        def __call__(self, *args, **kwargs):
            raise AssertionError('presence must not execute the entry')
        def __bool__(self):
            raise AssertionError('presence must not test entry truthiness')
    accelerator = ModuleType('native_test')
    accelerator.score_rows = Entry()
    residual = ModuleType('residual_test')
    residual._ACCEL = accelerator
    compiled = ModuleType('compiled_test')
    compiled._compiled_residual = residual
    def no_missing_attribute(name):
        raise AssertionError('presence must not invoke module __getattr__: ' + name)
    accelerator.__getattr__ = residual.__getattr__ = compiled.__getattr__ = no_missing_attribute
    monkeypatch.setattr(support, '_compiled', compiled)
    events = []
    sentinel = LookupError('original dispatch sentinel')
    raw_value, joint_value = object(), object()
    def raw_call(*args):
        events.append('raw')
        if failure == 'raw':
            raise sentinel
        return raw_value
    def affine_call(*args, **kwargs):
        events.append('affine')
        assert kwargs == {'policy': 'compensated'}
        if failure == 'affine':
            raise sentinel
        return joint_value
    def forbidden(*args, **kwargs):
        raise AssertionError('unused general guard was evaluated')
    monkeypatch.setattr(feature, 'compensated_kernel_matvec', raw_call)
    monkeypatch.setattr(feature, 'affine_scores', affine_call)
    monkeypatch.setattr(support, 'feature_scores_and_margins', forbidden)
    if failure is None:
        result = feature._feature_scores_and_margins(object(), object(), object())
        assert result[0] is raw_value and result[1] is joint_value
    else:
        with pytest.raises(LookupError) as caught:
            feature._feature_scores_and_margins(object(), object(), object())
        assert caught.value is sentinel
    assert events == (['raw'] if failure == 'raw' else ['raw', 'affine'])


class _CustomModule(ModuleType):
    def __getattribute__(self, name):
        if name in ('__dict__', '_compiled', '_compiled_residual', '_ACCEL', 'score_rows'):
            raise AssertionError('early metadata read on a custom module: ' + name)
        return super().__getattribute__(name)


class _CustomAccelerator:
    @property
    def score_rows(self):
        raise AssertionError('early property read on an arbitrary accelerator')


def _native_chain(accelerator):
    residual = ModuleType('residual_test')
    residual._ACCEL = accelerator
    compiled = ModuleType('compiled_test')
    compiled._compiled_residual = residual
    return compiled, residual


@pytest.mark.parametrize('case', [
    'absent_accelerator', 'empty_module', 'none_entry', 'nonmodule_accelerator',
    'custom_accelerator', 'custom_compiled', 'custom_residual', 'custom_support',
])
def test_portable_or_custom_metadata_delegates_without_probing(monkeypatch, case):
    accelerator = ModuleType('native_test')
    accelerator.score_rows = object()
    if case == 'absent_accelerator':
        accelerator = None
    elif case == 'empty_module':
        accelerator = ModuleType('empty_native')
        def no_missing_attribute(name):
            raise AssertionError('presence invoked a missing attribute: ' + name)
        accelerator.__getattr__ = no_missing_attribute
    elif case == 'none_entry':
        accelerator.score_rows = None
    elif case == 'nonmodule_accelerator':
        accelerator = _CustomAccelerator()
    elif case == 'custom_accelerator':
        accelerator = _CustomModule('custom_native')
    compiled, residual = _native_chain(accelerator)
    if case == 'custom_compiled':
        compiled = _CustomModule('custom_compiled')
    elif case == 'custom_residual':
        compiled._compiled_residual = _CustomModule('custom_residual')
    monkeypatch.setattr(support, '_compiled', compiled)
    features, beta, offset, result = object(), object(), object(), object()
    calls = []

    def delegate(actual_features, actual_beta, actual_offset, *, raw, affine):
        assert actual_features is features and actual_beta is beta and actual_offset is offset
        assert raw is feature.compensated_kernel_matvec and affine is feature.affine_scores
        calls.append('general')
        return result

    if case == 'custom_support':
        custom = _CustomModule('custom_support')
        custom.feature_scores_and_margins = delegate
        monkeypatch.setattr(feature, '_report_optimization', custom)
    else:
        monkeypatch.setattr(support, 'feature_scores_and_margins', delegate)
    assert feature._feature_scores_and_margins(features, beta, offset) is result
    assert calls == ['general']


@pytest.mark.parametrize('custom', [False, True])
def test_general_dispatch_exception_is_preserved(monkeypatch, custom):
    compiled, _ = _native_chain(_CustomModule('custom_native') if custom else None)
    monkeypatch.setattr(support, '_compiled', compiled)
    calls = []
    sentinel = RuntimeError('original general dispatch refusal')

    def refuse(*args, **kwargs):
        calls.append('general')
        raise sentinel

    monkeypatch.setattr(support, 'feature_scores_and_margins', refuse)
    with pytest.raises(RuntimeError) as caught:
        feature._feature_scores_and_margins(object(), object(), object())
    assert caught.value is sentinel and calls == ['general']
