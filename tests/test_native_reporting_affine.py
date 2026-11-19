"""Finite native affine-reporting contracts; no timing or altered acceptance.

Genuine-entry cases compare the candidate with its retained Python evaluator.
Synthetic refusals are labelled and never counted as native arithmetic evidence.
Largest fixture is nine rows by seven columns; the solver control uses four rows.
"""
from contextlib import contextmanager
import ctypes
from fractions import Fraction
import math
import platform
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from dwd import _affine_scores as affine
from dwd import _compiled_affine_scores as compiled
from dwd import _compiled_residual as loader
from dwd import _kernel_solver as solver


def bits(value):
    value = np.asarray(value)
    return value.dtype.str, value.shape, value.tobytes(order='C')


def same(actual, expected):
    assert bits(actual) == bits(expected)


def portable(K, alpha, b, **kwargs):
    # The existing public default is the original Python route. Only the new
    # reporting flag opts in; no mathematical function or safeguard is replaced.
    return affine.affine_scores(K, alpha, b, policy='compensated',
                                return_info=True, **kwargs)


def reporting(K, alpha, b, **kwargs):
    return affine.affine_scores(K, alpha, b, policy='compensated',
                                return_info=True, _reporting=True, **kwargs)


def rational_rows(K, alpha, b):
    return np.array([float(sum((Fraction.from_float(float(x)) *
                               Fraction.from_float(float(a))
                               for x, a in zip(row, alpha)),
                              Fraction.from_float(float(b)))) for row in K])


@pytest.fixture
def native_entry():
    entry = getattr(loader._ACCEL, 'score_affine_rows', None)
    if entry is None:
        pytest.skip('Optional native affine reporting entry is unavailable')
    if not compiled._supported(reporting=True):
        pytest.skip('Runtime is outside the native affine reporting domain')
    return entry


def arithmetic_case(name):
    if name == 'ordinary':
        K = (np.arange(45, dtype=float).reshape(9, 5) - 17.) / 8.
        return K, np.array([1., -2., 3., -4., 1.]) / 8., .125, 0
    if name == 'joint_intercept':
        return np.tile([2.**54, 1.], (9, 1)), np.ones(2), -2.**54, 0
    if name == 'exact_zero':
        return np.tile([1., -1., 0., -0.], (9, 1)), np.ones(4), -0., 0
    if name == 'subnormal':
        eta = np.array([1], dtype=np.uint64).view(np.float64)[0]
        return np.tile([0., -0., eta], (9, 1)), np.ones(3), 0., 9
    if name == 'lost_underflow':
        v = .5 + 2.**-53
        x, a = math.ldexp(v, -484), math.ldexp(v, -485)
        K, alpha = np.tile([x, x], (9, 1)), np.array([a, a])
        high, _ = affine._products(K[0], alpha)
        return K, alpha, -2. * float(high[0]), 9
    if name == 'overflow_cancellation':
        maximum = np.finfo(float).max
        return np.tile([2., 2.], (9, 1)), np.array([maximum, -maximum]), 1., 9
    raise AssertionError(name)


@pytest.mark.parametrize('name', ['ordinary', 'joint_intercept', 'exact_zero',
                                 'subnormal', 'lost_underflow', 'overflow_cancellation'])
@pytest.mark.parametrize('layout', ['C', 'F'])
def test_actual_native_matches_retained_python_and_exact_rows(name, layout, native_entry):
    K, alpha, b, exact_rows = arithmetic_case(name)
    K = np.array(K, order=layout)
    before = bits(K), bits(alpha)
    expected, info = portable(K, alpha, b)
    statuses = []

    def observed(*args):
        status = native_entry(*args)
        statuses.append(status)
        return status

    with patch.object(loader._ACCEL, 'score_affine_rows', observed):
        actual, actual_info = reporting(K, alpha, b)
    assert statuses, 'Actual native entry was not reached'
    if name == 'overflow_cancellation':
        assert any(status != 0 for status in statuses)
    else:
        assert all(status == 0 for status in statuses)
    same(actual, expected)
    same(actual, rational_rows(K, alpha, b))
    assert actual_info == info and info['exact_rows'] == exact_rows
    assert info['joint_rows'] == 9
    assert before == (bits(K), bits(alpha))
    if name == 'joint_intercept':
        assert actual[0] == 1. and float(K[0] @ alpha) + b == 0.


def test_readonly_inputs_and_scalar_tail_remain_unchanged(native_entry):
    K, alpha, b, _ = arithmetic_case('ordinary')
    K.flags.writeable = alpha.flags.writeable = False
    before = bits(K), bits(alpha)
    expected = portable(K, alpha, b)
    with patch.object(loader._ACCEL, 'score_affine_rows', wraps=native_entry) as observed:
        actual = reporting(K, alpha, b)
    assert observed.call_count > 0
    same(actual[0], expected[0])
    assert actual[1] == expected[1]
    assert before == (bits(K), bits(alpha))
    assert not K.flags.writeable and not alpha.flags.writeable


@pytest.mark.parametrize('kind', ['float32', 'integer', 'strided', 'csr_duplicates'])
def test_ineligible_storage_keeps_original_fallback(kind):
    K, alpha, b, _ = arithmetic_case('ordinary')
    if kind == 'float32':
        K = K.astype(np.float32)
    elif kind == 'integer':
        K, alpha, b = np.arange(27).reshape(9, 3), np.ones(3), .25
    elif kind == 'strided':
        K, alpha = K[:, ::-1], alpha[::-1].copy()
    else:
        K = csr_matrix((np.tile([2.**54, 1., -2.**54, -1.], 3),
                        np.zeros(12, dtype=np.int32), np.arange(0, 13, 4)), shape=(3, 1))
        alpha, b = np.ones(1), 0.
    before = ((bits(K.data), bits(K.indices), bits(K.indptr))
              if kind == 'csr_duplicates' else bits(K)), bits(alpha)
    expected = portable(K, alpha, b)
    entry = Mock(side_effect=AssertionError('Ineligible storage entered native affine'))
    with patch.object(loader, '_ACCEL', SimpleNamespace(score_affine_rows=entry)):
        actual = reporting(K, alpha, b)
    entry.assert_not_called()
    same(actual[0], expected[0])
    assert actual[1] == expected[1]
    after = ((bits(K.data), bits(K.indices), bits(K.indptr))
             if kind == 'csr_duplicates' else bits(K)), bits(alpha)
    assert before == after


def test_public_default_and_adaptive_calls_do_not_opt_in():
    K, alpha, b, _ = arithmetic_case('joint_intercept')
    entry = Mock(side_effect=AssertionError('Default prediction entered reporting-only native'))
    expected = [affine.affine_scores(K, alpha, b, policy=p, return_info=True)
                for p in ('adaptive', 'compensated')]
    with patch.object(loader, '_ACCEL', SimpleNamespace(score_affine_rows=entry)):
        for policy, reference in zip(('adaptive', 'compensated'), expected):
            got = affine.affine_scores(K, alpha, b, policy=policy, return_info=True)
            same(got[0], reference[0])
            assert got[1] == reference[1]
    entry.assert_not_called()


def test_missing_entry_and_unsupported_runtime_keep_python():
    K, alpha, b, _ = arithmetic_case('ordinary')
    expected = portable(K, alpha, b)
    for extension in (None, SimpleNamespace(score_rows=lambda *args: 0)):
        with patch.object(loader, '_ACCEL', extension):
            actual = reporting(K, alpha, b)
        same(actual[0], expected[0])
        assert actual[1] == expected[1]
    entry = Mock(side_effect=AssertionError('Unsupported runtime entered native affine'))
    with patch.object(compiled, '_supported', return_value=False), \
            patch.object(loader, '_ACCEL', SimpleNamespace(score_affine_rows=entry)):
        actual = reporting(K, alpha, b)
    entry.assert_not_called()
    same(actual[0], expected[0])
    assert actual[1] == expected[1]


@pytest.mark.parametrize('name', ['_products', '_fsum', '_split_operand', '_finish_row',
                                 '_finish_score', '_exact_row', '_zero_has_exact_products'])
def test_custom_arithmetic_and_finish_hooks_remain_observable(name):
    K, alpha, b, _ = arithmetic_case('exact_zero')
    original = getattr(affine, name)
    entry = Mock(side_effect=AssertionError('Custom hook was bypassed'))
    with patch.object(affine, name, wraps=original) as hook:
        expected = portable(K, alpha, b)
        hook.reset_mock()
        with patch.object(loader, '_ACCEL', SimpleNamespace(score_affine_rows=entry)):
            actual = reporting(K, alpha, b)
        assert hook.call_count > 0
    entry.assert_not_called()
    same(actual[0], expected[0])
    assert actual[1] == expected[1]


def test_numpy_handlers_and_replaced_math_fsum_keep_python():
    K, alpha, b, _ = arithmetic_case('ordinary')
    entry = Mock(side_effect=AssertionError('Observable arithmetic policy bypassed'))
    old_handler = np.geterrcall()
    try:
        for mode in ('call', 'log'):
            with np.errstate(all=mode):
                expected = portable(K, alpha, b)
                with patch.object(loader, '_ACCEL', SimpleNamespace(score_affine_rows=entry)):
                    actual = reporting(K, alpha, b)
                same(actual[0], expected[0])
                assert actual[1] == expected[1]
                assert np.geterr() == dict.fromkeys(('divide', 'over', 'under', 'invalid'), mode)
        original = math.fsum
        with patch.object(math, 'fsum', wraps=original), \
                patch.object(loader, '_ACCEL', SimpleNamespace(score_affine_rows=entry)):
            actual = reporting(K, alpha, b)
            expected = portable(K, alpha, b)
        same(actual[0], expected[0])
        assert actual[1] == expected[1]
    finally:
        assert np.geterrcall() is old_handler
    entry.assert_not_called()


@pytest.mark.parametrize('failure', ['status', 'memory', 'floating', 'overflow'])
def test_native_partial_output_is_discarded_before_python_retry(failure):
    K, alpha, b, _ = arithmetic_case('ordinary')
    expected = portable(K, alpha, b)
    count = []

    def refuse(*args):
        count.append(1)
        args[-2][:] = 123456789.
        if failure == 'status':
            return 6
        raise {'memory': MemoryError, 'floating': FloatingPointError,
               'overflow': OverflowError}[failure]('Injected native refusal')

    with patch.object(compiled, '_supported', return_value=True), \
            patch.object(loader, '_ACCEL', SimpleNamespace(score_affine_rows=refuse)):
        actual = reporting(K, alpha, b)
    assert count
    same(actual[0], expected[0])
    assert actual[1] == expected[1]


def test_unexpected_native_error_and_postcheck_error_preserve_identity(native_entry):
    K, alpha, b, _ = arithmetic_case('ordinary')
    failure = RuntimeError('Unexpected native entry failure')
    with patch.object(loader._ACCEL, 'score_affine_rows', side_effect=failure):
        with pytest.raises(RuntimeError) as caught:
            reporting(K, alpha, b)
    assert caught.value is failure
    # A changed finish helper must be observed in the original evaluator; it is
    # not an arithmetic-refusal status from the C entry.
    postcheck = FloatingPointError('Postcheck sentinel')
    with patch.object(affine, '_finish_score', side_effect=postcheck):
        with pytest.raises(FloatingPointError) as caught:
            reporting(K, alpha, b)
    assert caught.value is postcheck


def direct_args():
    K = np.array([[1., 2., 3.], [-1., .5, 2.]])
    alpha = np.array([.25, -.5, .75])
    return [K, alpha, *affine._split_operand(alpha), np.full(2, -987.), .125]


def test_bridge_valid_buffers_and_safe_invalid_refusals(native_entry):
    args = direct_args()
    inputs = [bits(v) for v in args[:-2]]
    assert native_entry(*args) == 0
    same(args[-2], rational_rows(args[0], args[1], args[-1]))
    assert inputs == [bits(v) for v in args[:-2]]
    invalid = []
    for index, value in ((0, args[0][:, ::-1]), (0, np.asfortranarray(args[0])),
                         (0, args[0].astype(np.float32)),
                         (0, np.ones((9, 3))), (0, np.empty((0, 3))),
                         (1, np.ones(2)), (3, args[3].astype(np.int64)),
                         (3, args[3].astype('>i4')), (3, args[3].reshape(1, -1)),
                         (6, args[1][:2]), (6, np.zeros(3))):
        changed = list(args)
        changed[index] = value
        invalid.append(changed)
    readonly = np.zeros(2)
    readonly.flags.writeable = False
    invalid.append([*args[:6], readonly, args[-1]])
    backing = bytearray(2 * 3 * 8 + 1)
    unaligned = np.ndarray((2, 3), dtype=float, buffer=backing, offset=1)
    unaligned[:] = args[0]
    invalid.append([unaligned, *args[1:]])
    for changed in invalid:
        before = [bits(v) for v in changed[:-1]]
        refs = [sys.getrefcount(v) for v in changed[:-1]]
        with pytest.raises((ValueError, TypeError, BufferError)):
            native_entry(*changed)
        assert before == [bits(v) for v in changed[:-1]]
        assert refs == [sys.getrefcount(v) for v in changed[:-1]]
    assert native_entry(*args) == 0


def test_bridge_nonfinite_and_final_overflow_never_become_success(native_entry):
    for b in (float('nan'), float('inf'), -float('inf')):
        args = direct_args()
        args[-1] = b
        before = [bits(v) for v in args[:-2]]
        assert native_entry(*args) != 0
        assert before == [bits(v) for v in args[:-2]]
    K = np.full((3, 2), np.finfo(float).max)
    alpha = np.ones(2)
    before = bits(K), bits(alpha)
    with pytest.raises(FloatingPointError) as expected:
        portable(K, alpha, 0.)
    with patch.object(loader._ACCEL, 'score_affine_rows', wraps=native_entry) as observed:
        with pytest.raises(FloatingPointError) as actual:
            reporting(K, alpha, 0.)
    assert observed.call_count > 0 and str(actual.value) == str(expected.value)
    assert before == (bits(K), bits(alpha))


def test_helper_rejects_alias_readonly_and_bad_indices():
    K, alpha, b, _ = arithmetic_case('ordinary')
    split = affine._split_operand(alpha)
    entry = Mock(side_effect=AssertionError('Unsafe helper storage entered native'))
    output = np.full(len(K), -321.)
    exact = np.zeros(len(K), dtype=bool)
    readonly = output.copy()
    readonly.flags.writeable = False
    cases = [(K[:, 0], exact, range(len(K))), (readonly, exact, range(len(K))),
             (output, exact, range(8, -1, -1)), (output, exact, [-1, 0]),
             (output, exact, np.array([-1, 0], dtype=np.intp)),
             (output, output.view(np.bool_)[:len(K)], range(len(K)))]
    with patch.object(compiled, '_supported', return_value=True), \
            patch.object(loader, '_ACCEL', SimpleNamespace(score_affine_rows=entry)):
        for result, flags, indices in cases:
            before = bits(K), bits(alpha), bits(result), bits(flags)
            assert compiled.try_expanded_dense_rows(
                K, alpha, b, indices, split, result, flags,
                batch_enabled=True, reporting=True) is None
            assert before == (bits(K), bits(alpha), bits(result), bits(flags))
    entry.assert_not_called()


@contextmanager
def downward_rounding():
    # Same audited APIs/bit constants as tests/test_vector_bounds.py.
    if sys.platform == 'win32':
        control = ctypes.CDLL('ucrtbase')._controlfp_s
        control.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        control.restype = ctypes.c_int
        state = ctypes.c_uint()
        assert control(ctypes.byref(state), 0, 0) == 0
        saved = state.value
        enter = lambda: control(ctypes.byref(state), 0x100, 0x300)
        leave = lambda: control(ctypes.byref(state), saved, 0x300)
        def unchanged():
            assert control(ctypes.byref(state), 0, 0) == 0
            return state.value & 0x300 == 0x100
    elif sys.platform == 'linux' and platform.machine().lower() in ('x86_64', 'amd64'):
        lib = ctypes.CDLL(None)
        get, set_mode = lib.fegetround, lib.fesetround
        get.argtypes, get.restype = [], ctypes.c_int
        set_mode.argtypes, set_mode.restype = [ctypes.c_int], ctypes.c_int
        saved = get()
        assert saved >= 0
        enter, leave = lambda: set_mode(0x400), lambda: set_mode(saved)
        unchanged = lambda: get() == 0x400
    else:
        pytest.skip('Requires audited Windows or Linux x86-64 rounding controls')
    try:
        assert enter() == 0
        yield unchanged
    finally:
        assert leave() == 0


def test_real_rounding_mode_refuses_native_and_preserves_caller(native_entry):
    args = direct_args()
    K, alpha, b, _ = arithmetic_case('exact_zero')
    assert affine._round_to_nearest_now()
    with downward_rounding() as unchanged:
        assert native_entry(*args) != 0
        expected = portable(K, alpha, b)
        actual = reporting(K, alpha, b)
        same(actual[0], expected[0])
        assert actual[1] == expected[1] and unchanged()
    assert affine._round_to_nearest_now()


@pytest.mark.parametrize('case', ['ordinary', 'cancellation'])
def test_complete_report_is_unchanged(case, native_entry):
    if case == 'ordinary':
        K = np.eye(4) + np.ones((4, 4)) / 8.
        y, alpha, b, penalty, kwargs = np.array([1., -1., 1., -1.]), np.array([.25, -.125, .125, -.25]), .3, .1, {}
    else:
        K = np.full((4, 4), 2.**52)
        y, alpha, b, penalty, kwargs = np.array([1., -1., 1., -1.]), np.array([1., 2.**-52, -1., 0.]), -.25, 2.**-40, {'scores': np.zeros(4)}
    before = bits(K), bits(y), bits(alpha)
    with patch.object(loader._ACCEL, 'score_affine_rows', None):
        expected = solver.optimality_diagnostics(K, y, alpha, b, penalty, 1., **kwargs)
    with patch.object(loader._ACCEL, 'score_affine_rows', wraps=native_entry) as observed:
        actual = solver.optimality_diagnostics(K, y, alpha, b, penalty, 1., **kwargs)
    assert observed.call_count > 0 and actual == expected
    assert before == (bits(K), bits(y), bits(alpha))
    if case == 'cancellation':
        assert actual['intercept_gradient'] == pytest.approx(5. / 18., rel=1e-15, abs=0.)


def test_full_solver_returned_state_and_diagnostics_are_unchanged(native_entry):
    K = np.eye(4) + np.ones((4, 4)) / 8.
    y = np.array([1., -1., 1., -1.])
    settings = dict(lambd=.1, q=1., backend='cholesky', max_iter=3,
                    stopping='fixed', acceleration=None)
    with patch.object(loader._ACCEL, 'score_affine_rows', None):
        expected = solver.solve_kernel(K, y, **settings)
    with patch.object(loader._ACCEL, 'score_affine_rows', wraps=native_entry) as observed:
        actual = solver.solve_kernel(K, y, **settings)
    assert observed.call_count > 0
    def stable(value):
        if isinstance(value, np.ndarray):
            return bits(value)
        if isinstance(value, dict):
            return {key: stable(item) for key, item in value.items()
                    if key != 'seconds' and not key.endswith('_seconds')}
        if isinstance(value, (list, tuple)):
            return [stable(item) for item in value]
        if isinstance(value, float):
            return value.hex()
        return value
    assert stable(actual) == stable(expected)
    assert actual['n_iter'] == actual['returned_iteration'] == 3

